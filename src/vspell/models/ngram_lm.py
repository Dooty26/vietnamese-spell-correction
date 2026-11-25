"""Mô hình ngôn ngữ n-gram mức âm tiết, tự cài đặt (không cần KenLM):
- smoothing="laplace":    P(w|h) = (c(h,w) + k) / (c(h) + k·V)        (chỉ dùng bậc cao nhất)
- smoothing="kneser_ney": Kneser-Ney nội suy với chiết khấu D cố định:
      P(w|h) = max(c(h,w) − D, 0)/c(h) + D·N1+(h•)/c(h) · P_KN(w|h')
  bậc thấp dùng "continuation count" N1+(•h w) thay cho số đếm thô; bậc 1 nội suy với phân phối đều.

Từ được đưa về chữ thường; số -> <num>; từ hiếm (< min_count) -> <unk>. Mọi xác suất trả về ở dạng log10.
Phụ trách: TV2.
"""
from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from vspell.normalize import normalize, split_syllables

BOS, EOS, UNK, NUM = "<s>", "</s>", "<unk>", "<num>"


def lm_token(tok: str) -> str:
    """Token văn bản -> token của LM."""
    return NUM if tok[:1].isdigit() else tok.lower()


def lm_tokens(sentence: str) -> list[str]:
    return [lm_token(t) for t in split_syllables(normalize(sentence))]


class NgramLM:
    def __init__(self, order: int = 3, smoothing: str = "kneser_ney", discount: float = 0.75,
                 add_k: float = 1.0, min_count: int = 1):
        if smoothing not in {"kneser_ney", "laplace"}:
            raise ValueError(smoothing)
        self.order = order
        self.smoothing = smoothing
        self.discount = discount
        self.add_k = add_k
        self.min_count = min_count
        self.vocab: dict[str, int] = {}
        self._cache_clear()

    # ------------------------------------------------------------------ huấn luyện
    def fit(self, sentences: Iterable[list[str]]) -> "NgramLM":
        """sentences: các câu đã tách token LM (xem lm_tokens)."""
        sentences = [s for s in sentences if s]
        freq = Counter(t for s in sentences for t in s)
        words = sorted(w for w, c in freq.items() if c >= self.min_count)
        self.vocab = {w: i for i, w in enumerate([BOS, EOS, UNK, *words])}
        self.V = len(self.vocab) - 1  # không tính <s> (không bao giờ được dự đoán)
        n = self.order
        bos, unk = self.vocab[BOS], self.vocab[UNK]

        raw: Counter = Counter()  # n-gram bậc cao nhất (đếm thô)
        for s in sentences:
            ids = [bos] * (n - 1) + [self.vocab.get(t, unk) for t in s] + [self.vocab[EOS]]
            for i in range(n - 1, len(ids)):
                raw[tuple(ids[i - n + 1:i + 1])] += 1

        # num[k][gram]: số đếm dùng ở bậc k (bậc n: thô; bậc < n: continuation count)
        self.num: dict[int, dict] = {n: dict(raw)}
        for k in range(n - 1, 0, -1):
            cont: Counter = Counter()
            for gram in self.num[k + 1]:
                cont[gram[1:]] += 1
            self.num[k] = dict(cont)
        self.den: dict[int, dict] = {}
        self.types: dict[int, dict] = {}
        for k, table in self.num.items():
            den: defaultdict = defaultdict(int)
            typ: defaultdict = defaultdict(int)
            for gram, c in table.items():
                den[gram[:-1]] += c
                typ[gram[:-1]] += 1
            self.den[k], self.types[k] = dict(den), dict(typ)
        self._cache_clear()
        return self

    # ------------------------------------------------------------------ truy vấn
    def _cache_clear(self) -> None:
        self._prob_ids = lru_cache(maxsize=2_000_000)(self._prob_ids_uncached)

    def _ids(self, words) -> tuple[int, ...]:
        unk = self.vocab[UNK]
        return tuple(self.vocab.get(w, unk) for w in words)

    def _prob_ids_uncached(self, w: int, h: tuple[int, ...]) -> float:
        if self.smoothing == "laplace":
            num, den = self.num[self.order], self.den[self.order]
            return (num.get(h + (w,), 0) + self.add_k) / (den.get(h, 0) + self.add_k * self.V)
        return self._kn(w, h)

    def _kn(self, w: int, h: tuple[int, ...]) -> float:
        k = len(h) + 1
        D = self.discount
        if k == 1:
            d = self.den[1].get((), 0)
            lower = 1.0 / self.V
            if d == 0:
                return lower
            return max(self.num[1].get((w,), 0) - D, 0) / d + D * self.types[1].get((), 0) / d * lower
        d = self.den[k].get(h, 0)
        lower = self._kn(w, h[1:])
        if d == 0:
            return lower
        return max(self.num[k].get(h + (w,), 0) - D, 0) / d + D * self.types[k][h] / d * lower

    def logprob(self, word: str, context: tuple[str, ...]) -> float:
        """log10 P(word | context); context là các token LM đứng trước (có thể chứa <s>)."""
        n = self.order
        ctx = (BOS,) * max(0, n - 1 - len(context)) + tuple(context[-(n - 1):]) if n > 1 else ()
        (w,) = self._ids([word])
        return math.log10(self._prob_ids(w, self._ids(ctx)))

    def sentence_logprob(self, tokens: list[str]) -> float:
        ctx: tuple[str, ...] = ()
        total = 0.0
        for t in [*tokens, EOS]:
            total += self.logprob(t, ctx)
            ctx = (*ctx, t)[-(self.order - 1):] if self.order > 1 else ()
        return total

    def perplexity(self, sentences: Iterable[list[str]]) -> float:
        """Perplexity trên tập held-out (tính cả </s>)."""
        total, count = 0.0, 0
        for s in sentences:
            total += self.sentence_logprob(s)
            count += len(s) + 1
        return 10 ** (-total / max(count, 1))

    # ------------------------------------------------------------------ lưu/đọc
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {k: v for k, v in self.__dict__.items() if k != "_prob_ids"}
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "NgramLM":
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            obj.__dict__.update(pickle.load(f))
        obj._cache_clear()
        return obj
