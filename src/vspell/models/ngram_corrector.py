"""Mô hình A (baseline thống kê): noisy channel = mô hình kênh (chi phí sửa) + LM n-gram, giải mã beam search.

Với câu x, chọn y = argmax  Σ channel(x_i -> y_i) + λ · log10 P_LM(y)
- Từ điển âm tiết (lấy từ LM-corpus) quyết định từ "có thật".
- Ứng viên cho mỗi âm tiết (CandidateGenerator), mỗi nguồn một chi phí kênh (log10, âm):
    telex     : giải mã chuỗi gõ Telex chưa chuyển ("đươcj" -> "được")
    skeleton  : cùng khung không dấu ("duoc" -> "được", "đuợc")
    confusion : tập dễ nhầm vùng miền (s/x, tr/ch, r/d/gi, l/n, -n/-ng, -t/-c, hỏi/ngã)
    teencode  : từ điển teencode/viết tắt đảo ngược ("ko" -> "không", "đc" -> "được")
    edit1/2   : khoảng cách Damerau-Levenshtein 1/2 trong từ điển (SymSpell)
- Âm tiết ngoài từ điển: giữ nguyên bị phạt oov_penalty.
  Âm tiết trong từ điển: mọi ứng viên bị cộng thêm real_word_penalty (chỉ sửa khi LM "rất tin").
Phụ trách: TV2.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache

from vspell.models.ngram_lm import BOS, EOS, NgramLM, lm_token, lm_tokens
from vspell.noise.rules import _ACRONYMS, _FINAL_SWAPS, _INITIAL_SWAPS, _TEEN_WORDS, _TONE_SWAPS, _match_case
from vspell.normalize import (TONE_ACUTE, TONE_DOT, TONE_GRAVE, TONE_HOOK, TONE_TILDE, _TOKEN_RE, add_tone,
                              is_vietnamese_syllable, normalize, parse_syllable, remove_diacritics, split_tone,
                              to_nfc)

DEFAULT_COSTS = {"telex": -0.5, "skeleton": -1.0, "confusion": -1.0, "teencode": -1.0,
                 "edit1": -1.5, "edit2": -3.0}
STRONG_SOURCES = {"telex", "skeleton", "teencode"}


def is_iy_variant(a: str, b: str) -> bool:
    """Hai âm tiết chỉ khác nhau ở biến thể chính tả i/y (lí/lý, kì/kỳ, quí/quý, Mĩ/Mỹ): không coi là lỗi.

    Chỉ áp dụng khi vần chỉ gồm một nguyên âm i/y, không âm cuối, có phụ âm đầu ("ý"/"í" là hai từ khác nhau).
    """
    if a == b:
        return False
    (ba, ta), (bb, tb) = split_tone(a.lower()), split_tone(b.lower())
    pa, pb = parse_syllable(ba), parse_syllable(bb)
    if pa is None or pb is None or ta != tb:
        return False
    return (pa[0] == pb[0] != "" and pa[2] == pb[2] == "" and {pa[1], pb[1]} <= {"i", "y"})


def build_lexicon(sentences: list[str], min_count: int = 3) -> dict[str, int]:
    """Từ điển âm tiết hợp lệ (đã normalize, chữ thường) kèm tần suất."""
    freq: Counter = Counter()
    for s in sentences:
        for t in split_words(s):
            freq[t.lower()] += 1
    return {w: c for w, c in freq.items() if c >= min_count and is_vietnamese_syllable(w)}


def split_words(sentence: str) -> list[str]:
    return [m.group() for m in _TOKEN_RE.finditer(normalize(sentence)) if m.group().isalpha()]


# ---------------------------------------------------------------------- Telex
_TELEX_TONES = {"s": TONE_ACUTE, "f": TONE_GRAVE, "r": TONE_HOOK, "x": TONE_TILDE, "j": TONE_DOT}
_TELEX_PAIRS = [("dd", "đ"), ("aa", "â"), ("aw", "ă"), ("ee", "ê"), ("oo", "ô"), ("ow", "ơ"), ("uw", "ư")]


def telex_decode(token: str) -> set[str]:
    """Các cách hiểu chuỗi Telex chưa chuyển đổi: "dduwowcj" -> {"được"}, "gias" -> {"giá"}, "đươcj" -> {"được"}."""
    tok = token.lower()
    base, tone = split_tone(tok)
    outs = set()
    tone_options = [(base, tone)]
    if base and base[-1] in _TELEX_TONES and tone is None:
        tone_options.append((base[:-1], _TELEX_TONES[base[-1]]))
    for b, t in tone_options:
        s = b
        for src, dst in _TELEX_PAIRS:
            s = s.replace(src, dst)
        s = s.replace("uơ", "ươ")  # "uow" -> "ươ"
        if s != tok or t != tone:
            outs.add(to_nfc(add_tone(s, t)))
    return {o for o in outs if o != tok}


# ---------------------------------------------------------------------- SymSpell
def _deletes(word: str, max_d: int) -> set[str]:
    out, frontier = {word}, {word}
    for _ in range(max_d):
        nxt = {w[:i] + w[i + 1:] for w in frontier for i in range(len(w))}
        out |= nxt
        frontier = nxt
    return out


def osa_distance(a: str, b: str) -> int:
    """Khoảng cách Damerau-Levenshtein (optimal string alignment)."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


class CandidateGenerator:
    def __init__(self, lexicon: dict[str, int], max_edit_distance: int = 2, max_candidates: int = 20,
                 costs: dict[str, float] | None = None):
        self.lexicon = lexicon
        self.max_d = max_edit_distance
        self.max_candidates = max_candidates
        self.costs = {**DEFAULT_COSTS, **(costs or {})}
        self.skeleton: defaultdict = defaultdict(list)
        self.deletes: defaultdict = defaultdict(set)
        for w in lexicon:
            self.skeleton[remove_diacritics(w)].append(w)
            for d in _deletes(w, self.max_d):
                self.deletes[d].add(w)
        self.teen_inverse: defaultdict = defaultdict(set)
        for word, forms in _TEEN_WORDS.items():
            for f in forms:
                self.teen_inverse[f].add(word)
        for word, form in _ACRONYMS.items():
            self.teen_inverse[form].add(word)
        self.candidates = lru_cache(maxsize=500_000)(self._candidates)

    def _confusions(self, tok: str) -> set[str]:
        base, tone = split_tone(tok)
        parts = parse_syllable(base)
        if parts is None:
            return set()
        ini, nuc, fin = parts
        outs = set()
        for i2 in [ini, *_INITIAL_SWAPS.get(ini, [])]:
            for f2 in [fin, *_FINAL_SWAPS.get(fin, [])]:
                for t2 in [tone, *_TONE_SWAPS.get(tone, [])]:
                    outs.add(add_tone(i2 + nuc + f2, t2))
        outs.discard(tok)
        return outs

    def _candidates(self, tok: str) -> tuple[tuple[str, str, float], ...]:
        """tok (chữ thường) -> ((ứng viên, nguồn, chi phí), ...) sắp theo chi phí rồi tần suất; không gồm tok."""
        found: dict[str, tuple[str, float]] = {}

        def add(cand: str, source: str) -> None:
            if cand == tok or cand not in self.lexicon:
                return
            cost = self.costs[source]
            if cand not in found or cost > found[cand][1]:
                found[cand] = (source, cost)

        for c in telex_decode(tok):
            add(c, "telex")
        for c in self.skeleton.get(remove_diacritics(tok), []):
            add(c, "skeleton")
        for c in self._confusions(tok):
            add(c, "confusion")
        for c in self.teen_inverse.get(tok, ()):
            add(c, "teencode")
        pool = set()
        for d in _deletes(tok, self.max_d):
            pool |= self.deletes.get(d, set())
        for c in pool:
            dist = osa_distance(tok, c)
            if dist == 1:
                add(c, "edit1")
            elif dist == 2 and self.max_d >= 2:
                add(c, "edit2")
        ranked = sorted(found.items(), key=lambda kv: (-kv[1][1], -self.lexicon[kv[0]]))
        return tuple((c, src, cost) for c, (src, cost) in ranked[:self.max_candidates])


class NgramCorrector:
    def __init__(self, lm: NgramLM, lexicon: dict[str, int], config: dict):
        """config: nội dung configs/ngram.yaml."""
        self.lm = lm
        self.lexicon = lexicon
        self.config = config
        cc, dc = config.get("candidates", {}), config.get("decoding", {})
        self.gen = CandidateGenerator(lexicon, cc.get("max_edit_distance", 2), cc.get("max_candidates", 20),
                                      cc.get("costs"))
        self.beam_size = dc.get("beam_size", 5)
        self.lm_weight = dc.get("lm_weight", 1.0)
        self.oov_penalty = dc.get("oov_penalty", -3.0)
        self.real_word_penalty = dc.get("real_word_penalty", -2.0)
        self.real_word_sources = set(dc.get("real_word_sources", ["telex", "skeleton", "confusion", "teencode", "edit1"]))
        # S4: giảm sửa quá tay (mặc định tắt -> tái lập đúng mô hình A cũ)
        self.ignore_iy = dc.get("ignore_iy", False)                      # không đổi i <-> y
        self.keep_non_vietnamese = dc.get("keep_non_vietnamese", False)  # âm tiết sai cấu trúc TV: chỉ sửa bằng nguồn mạnh
        self.capital_penalty = dc.get("capital_penalty", 0.0)            # phạt thêm khi sửa từ viết hoa giữa câu
        self.edit_margin = dc.get("edit_margin", 0.0)                    # ngưỡng tin cậy: phạt thêm mỗi lần sửa

    # ------------------------------------------------------------------ lựa chọn cho từng token
    def _options(self, tok: str, is_first_word: bool) -> list[tuple[str, float]]:
        """-> [(token LM chọn, điểm kênh)], phần tử đầu là giữ nguyên."""
        low = tok.lower()
        keep = [(lm_token(tok), 0.0)]
        if not tok.isalpha() or (tok.isupper() and len(tok) > 1):
            return keep  # số, dấu câu, viết tắt toàn hoa: không sửa
        cands = self.gen.candidates(low)
        in_lex = low in self.lexicon
        capital_mid = tok[0].isupper() and not is_first_word
        if not in_lex and capital_mid:
            # Có thể là tên riêng/từ nước ngoài: chỉ sửa khi có bằng chứng mạnh
            cands = tuple(c for c in cands if c[1] in STRONG_SOURCES)
        if (self.keep_non_vietnamese and not in_lex and remove_diacritics(low) == low
                and not is_vietnamese_syllable(low)):
            # "manga", "online": không dấu và không phải âm tiết TV -> có thể là từ nước ngoài;
            # chỉ sửa khi là Telex/không dấu/teencode. ("vànv" có dấu -> vẫn là lỗi gõ, sửa bình thường)
            cands = tuple(c for c in cands if c[1] in STRONG_SOURCES)
        if self.ignore_iy:
            cands = tuple(c for c in cands if not is_iy_variant(low, c[0]))
        extra = self.edit_margin + (self.capital_penalty if capital_mid else 0.0)
        if in_lex:
            opts = [(c, cost + self.real_word_penalty + extra) for c, src, cost in cands
                    if src in self.real_word_sources]
            return keep + opts
        if not cands:
            return keep
        return [(low, self.oov_penalty)] + [(c, cost + extra) for c, _, cost in cands]

    def correct_one(self, sentence: str) -> str:
        text = normalize(sentence)
        matches = list(_TOKEN_RE.finditer(text))
        first_word = next((i for i, m in enumerate(matches) if m.group().isalpha()), -1)
        n = self.lm.order
        # beam: ngữ cảnh LM (n-1 token cuối) -> (điểm, lựa chọn)
        beams: dict[tuple, tuple[float, list[str]]] = {(BOS,) * (n - 1): (0.0, [])}
        for i, m in enumerate(matches):
            opts = self._options(m.group(), i == first_word)
            new_beams: dict[tuple, tuple[float, list[str]]] = {}
            for ctx, (score, choice) in beams.items():
                for cand, channel in opts:
                    s = score + channel + self.lm_weight * self.lm.logprob(cand, ctx)
                    nctx = (*ctx, cand)[-(n - 1):] if n > 1 else ()
                    if nctx not in new_beams or s > new_beams[nctx][0]:
                        new_beams[nctx] = (s, [*choice, cand])
            beams = dict(sorted(new_beams.items(), key=lambda kv: -kv[1][0])[:self.beam_size])
        best = max(beams.items(), key=lambda kv: kv[1][0] + self.lm_weight * self.lm.logprob(EOS, kv[0]))
        choice = best[1][1]

        out, last = [], 0
        for m, cand in zip(matches, choice):
            tok = m.group()
            if tok.isalpha() and cand != tok.lower() and cand != lm_token(tok):
                out += [text[last:m.start()], _match_case(cand, tok)]
                last = m.end()
        out.append(text[last:])
        return "".join(out)

    def detect(self, sentence: str) -> list[bool]:
        """Đánh dấu token mô hình cho là sai (token bị thay đổi sau khi sửa)."""
        src = [m.group() for m in _TOKEN_RE.finditer(normalize(sentence))]
        pred = [m.group() for m in _TOKEN_RE.finditer(self.correct_one(sentence))]
        return [a != b for a, b in zip(src, pred)]

    def correct(self, sentences: list[str]) -> list[str]:
        return [self.correct_one(s) for s in sentences]


def train(corpus_sentences: list[str], config: dict) -> tuple[NgramLM, dict[str, int]]:
    """Huấn luyện LM + từ điển từ các câu sạch."""
    lm = NgramLM(order=config["order"], smoothing=config["smoothing"],
                 discount=config.get("discount", 0.75), add_k=config.get("add_k", 1.0),
                 min_count=config.get("min_count", 1))
    lm.fit(lm_tokens(s) for s in corpus_sentences)
    lexicon = build_lexicon(corpus_sentences, config.get("lexicon_min_count", 3))
    return lm, lexicon


