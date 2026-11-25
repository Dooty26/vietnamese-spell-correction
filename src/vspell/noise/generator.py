"""Sinh cặp (src lỗi, tgt đúng) từ câu sạch theo configs/noise.yaml.

- Chỉ làm nhiễu token là âm tiết tiếng Việt hợp lệ (bỏ qua số, dấu câu, từ nước ngoài, viết tắt toàn hoa).
- Mỗi âm tiết hợp lệ bị làm nhiễu với xác suất syllable_error_rate; câu được chọn làm nhiễu có ít nhất 1 lỗi.
- clean_sentence_ratio câu được giữ nguyên (src == tgt) để mô hình học cách KHÔNG sửa.
- Thay thế theo vị trí ký tự nên giữ nguyên khoảng trắng/dấu câu của câu gốc; số token không đổi.
- errors[].pos là chỉ số token trong split_syllables(src) — khớp với vspell.eval.metrics.per_error_type.
Phụ trách: TV1.
"""
from __future__ import annotations

import random

from vspell.normalize import _TOKEN_RE, is_vietnamese_syllable, normalize, split_syllables
from vspell.noise.rules import RULES


class NoiseGenerator:
    def __init__(self, config: dict, seed: int = 42):
        """config: nội dung configs/noise.yaml."""
        self.config = config
        self.seed = seed
        self.rng = random.Random(seed)
        weights = config["error_type_weights"]
        self.types = list(weights)
        self.weights = [weights[t] for t in self.types]
        self.rate = config["syllable_error_rate"]
        self.clean_ratio = config.get("clean_sentence_ratio", 0.0)

    def _corrupt_syllable(self, syl: str) -> tuple[str, str] | None:
        """Bốc thứ tự loại lỗi theo trọng số (không lặp); loại đầu không áp dụng được thì thử loại kế tiếp."""
        types, weights = list(self.types), list(self.weights)
        order = []
        while types:
            k = self.rng.choices(range(len(types)), weights=weights, k=1)[0]
            order.append(types.pop(k))
            weights.pop(k)
        for typ in order:
            noisy = RULES[typ](syl, self.rng)
            if noisy and noisy != syl and normalize(noisy) != syl:
                return noisy, typ
        return None

    def corrupt(self, sentence: str, force: bool = False) -> tuple[str, list[dict]]:
        """Trả về (câu lỗi, [{"pos", "orig", "noisy", "type"}]); giữ nguyên số token.

        force=True: luôn làm nhiễu (bỏ qua clean_sentence_ratio).
        """
        tgt = normalize(sentence)
        if not force and self.rng.random() < self.clean_ratio:
            return tgt, []
        matches = list(_TOKEN_RE.finditer(tgt))
        cands = [i for i, m in enumerate(matches)
                 if m.group().isalpha() and not (m.group().isupper() and len(m.group()) > 1)
                 and is_vietnamese_syllable(m.group())]
        if not cands:
            return tgt, []
        chosen = [i for i in cands if self.rng.random() < self.rate]
        if not chosen:
            chosen = [self.rng.choice(cands)]

        errors, pieces, last = [], [], 0
        for i in chosen:
            m = matches[i]
            res = self._corrupt_syllable(m.group())
            if res is None:
                continue
            noisy, typ = res
            pieces += [tgt[last:m.start()], noisy]
            last = m.end()
            errors.append({"pos": i, "orig": m.group(), "noisy": noisy, "type": typ})
        pieces.append(tgt[last:])
        src = normalize("".join(pieces))
        # An toàn: chỉ giữ lỗi còn tồn tại sau chuẩn hóa và số token không đổi
        src_toks = split_syllables(src)
        if len(src_toks) != len(matches):
            return tgt, []
        errors = [{**e, "noisy": src_toks[e["pos"]]} for e in errors if src_toks[e["pos"]] != e["orig"]]
        return src, errors

    def make_pairs(self, sentences: list[str], prefix: str = "syn") -> list[dict]:
        """-> [{"id", "src", "tgt", "dataset": "synthetic", "errors": [...]}]."""
        out = []
        for k, sent in enumerate(sentences):
            src, errors = self.corrupt(sent)
            out.append({"id": f"{prefix}-{k}", "src": src, "tgt": normalize(sent),
                        "dataset": "synthetic", "errors": errors})
        return out
