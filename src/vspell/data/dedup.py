"""Chống rò rỉ: loại khỏi Binhvq các câu trùng/gần trùng với VSEC và Viwiki (cả src lẫn tgt),
và loại câu trùng lặp ngay trong Binhvq.

Phụ trách: TV1.
"""
from __future__ import annotations

import hashlib
import re

from vspell.normalize import normalize, split_syllables

_NON_WORD = re.compile(r"[^\w]+")


def _words(sentence: str) -> list[str]:
    return [t.lower() for t in split_syllables(normalize(sentence)) if t.isalnum()]


def sentence_key(sentence: str) -> str:
    """Khóa so khớp trùng tuyệt đối (hash sau normalize + chữ thường + bỏ dấu câu)."""
    return hashlib.sha1(" ".join(_words(sentence)).encode("utf-8")).hexdigest()


def _ngrams(words: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def build_blocklist(test_records: list[dict], ngram: int) -> tuple[set[str], set[tuple[str, ...]]]:
    """Tạo tập khóa câu và tập n-gram âm tiết từ src và tgt của các tập test."""
    keys: set[str] = set()
    grams: set[tuple[str, ...]] = set()
    for r in test_records:
        for text in (r["src"], r["tgt"]):
            keys.add(sentence_key(text))
            grams.update(_ngrams(_words(text), ngram))
    return keys, grams


def is_leaked(sentence: str, keys: set[str], ngrams: set[tuple[str, ...]], ngram: int, threshold: float) -> bool:
    """True nếu câu trùng tuyệt đối hoặc tỉ lệ n-gram trùng với tập test >= threshold."""
    if sentence_key(sentence) in keys:
        return True
    grams = _ngrams(_words(sentence), ngram)
    if not grams:
        return False
    return sum(g in ngrams for g in grams) / len(grams) >= threshold
