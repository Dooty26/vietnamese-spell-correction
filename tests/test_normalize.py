import unicodedata

import pytest

from vspell.normalize import is_vietnamese_syllable, normalize, split_syllables


@pytest.mark.parametrize("raw, expected", [
    ("hoà", "hòa"), ("thuý", "thúy"), ("khoẻ", "khỏe"), ("tóan", "toán"),
    ("qũy", "quỹ"), ("Hoà Bình", "Hòa Bình"), ("THUÝ", "THÚY"),
    # đã đúng chuẩn thì giữ nguyên
    ("người", "người"), ("ngoài", "ngoài"), ("khuỷu", "khuỷu"), ("của", "của"),
    ("gì", "gì"), ("già", "già"), ("quá", "quá"), ("giường", "giường"), ("hoàng", "hoàng"),
    # không phải tiếng Việt thì không đụng tới
    ("Begonia", "Begonia"), ("A.Rich", "A.Rich"),
])
def test_tone_placement(raw, expected):
    assert normalize(raw) == expected


def test_nfd_input_and_whitespace():
    nfd = unicodedata.normalize("NFD", "hoà   bình ")
    assert normalize(nfd) == "hòa bình"


def test_split_punctuation_and_numbers():
    assert split_syllables("Chiến lược điều chỉnh giá:") == ["Chiến", "lược", "điều", "chỉnh", "giá", ":"]
    assert split_syllables("40.000 đồng/lượng") == ["40.000", "đồng", "/", "lượng"]


@pytest.mark.parametrize("tok, ok", [
    ("toán", True), ("các", True), ("nghiêng", True), ("ghe", True),
    ("càc", False), ("kan", False), ("ge", False), ("ngiêng", False), ("bátt", False),
])
def test_is_vietnamese_syllable(tok, ok):
    assert is_vietnamese_syllable(tok) is ok
