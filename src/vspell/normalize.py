"""Chuẩn hóa văn bản tiếng Việt. PHẢI áp dụng cho mọi tập dữ liệu và mọi đầu ra mô hình trước khi chấm.

Quy ước vị trí dấu thanh: kiểu "cũ" phổ biến trên báo chí -> "hòa", "thúy", "khỏe", "quý", "toán", "người".
Phụ trách: TV2.
"""
from __future__ import annotations

import re
import unicodedata

# Dấu thanh dạng combining (NFD): huyền, sắc, ngã, hỏi, nặng
_TONE_MARKS = {"̀", "́", "̃", "̉", "̣"}
_VOWELS = "aăâeêioôơuưy"
_SPECIAL_VOWELS = "ăâêôơư"  # nguyên âm có dấu phụ: ưu tiên mang dấu thanh

_SYLLABLE_RE = re.compile(
    r"^(ngh|ng|gh|nh|ch|kh|ph|th|tr|gi|qu|[bcdđghklmnpqrstvx])?"
    r"([aăâeêioôơuưy]+)"
    r"(ch|ng|nh|[cmnpt])?$"
)
_WORD_RE = re.compile(r"[^\W\d_]+")  # chuỗi chữ cái liên tiếp (không số, không gạch dưới)
_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*|[^\W_]+|_|[^\w\s]")


def to_nfc(text: str) -> str:
    """Đưa về Unicode NFC và thay các ký tự trông giống 'Đ' về đúng mã."""
    text = unicodedata.normalize("NFC", text)
    return text.replace("Ð", "Đ")  # Ð (Eth) -> Đ


def _split_tone(syllable: str) -> tuple[str, str | None]:
    """Tách âm tiết thành (phần không dấu thanh, NFC; dấu thanh combining hoặc None)."""
    tone = None
    chars = []
    for ch in unicodedata.normalize("NFD", syllable):
        if ch in _TONE_MARKS:
            tone = ch
        else:
            chars.append(ch)
    return unicodedata.normalize("NFC", "".join(chars)), tone


def _tone_index(base_lower: str) -> int | None:
    """Vị trí (trong âm tiết) của nguyên âm mang dấu thanh; None nếu không phải âm tiết tiếng Việt."""
    m = _SYLLABLE_RE.match(base_lower)
    if not m:
        return None
    start, end = m.span(2)
    nucleus = m.group(2)
    has_final = m.group(3) is not None
    # 1) Có nguyên âm mang dấu phụ -> đặt lên đó (với "ươ" chọn "ơ": lấy nguyên âm phụ cuối cùng)
    special = [i for i, c in enumerate(nucleus) if c in _SPECIAL_VOWELS]
    if special:
        return start + special[-1]
    if len(nucleus) == 1:
        return start
    # 2) Có phụ âm cuối -> nguyên âm cuối của cụm ("toán", "hoàng", "quyết" đã xử lý ở bước 1)
    if has_final:
        return end - 1
    # 3) Âm tiết mở: 3 nguyên âm -> giữa ("ngoài", "khuỷu"); 2 nguyên âm -> đầu ("hòa", "thúy", "của")
    if len(nucleus) >= 3:
        return start + 1
    return start


def _place_tone(word: str) -> str:
    base, tone = _split_tone(word)
    if tone is None:
        return word
    idx = _tone_index(base.lower())
    if idx is None:
        return word
    target = unicodedata.normalize("NFD", base[idx])
    new_char = unicodedata.normalize("NFC", target + tone)
    return base[:idx] + new_char + base[idx + 1:]


# Các hằng số dấu thanh dùng chung (combining, NFD)
TONE_GRAVE, TONE_ACUTE, TONE_TILDE, TONE_HOOK, TONE_DOT = "̀", "́", "̃", "̉", "̣"


def split_tone(syllable: str) -> tuple[str, str | None]:
    """Tách âm tiết -> (phần không dấu thanh, dấu thanh combining hoặc None). Vd "được" -> ("đươc", nặng)."""
    return _split_tone(to_nfc(syllable))


def add_tone(base: str, tone: str | None) -> str:
    """Gắn dấu thanh vào âm tiết không dấu theo quy ước đặt dấu của module. Không phải âm tiết TV -> đặt lên nguyên âm cuối."""
    if not tone:
        return base
    idx = _tone_index(base.lower())
    if idx is None:
        vowels = [i for i, c in enumerate(base.lower()) if c in _VOWELS]
        if not vowels:
            return base
        idx = vowels[-1]
    return base[:idx] + unicodedata.normalize("NFC", unicodedata.normalize("NFD", base[idx]) + tone) + base[idx + 1:]


def parse_syllable(base_lower: str) -> tuple[str, str, str] | None:
    """Âm tiết không dấu thanh, chữ thường -> (phụ âm đầu, vần chính, phụ âm cuối); None nếu không hợp lệ."""
    m = _SYLLABLE_RE.match(base_lower)
    if not m:
        return None
    return m.group(1) or "", m.group(2), m.group(3) or ""


def remove_diacritics(text: str) -> str:
    """Bỏ toàn bộ dấu (thanh và dấu phụ): "Đường" -> "Duong"."""
    text = to_nfc(text).replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


def normalize_tone_placement(text: str) -> str:
    """Quy vị trí dấu thanh về một kiểu thống nhất (vd: "hoà" -> "hòa", "thuý" -> "thúy").

    Chỉ sửa những từ có cấu trúc âm tiết tiếng Việt hợp lệ; từ khác giữ nguyên.
    """
    return _WORD_RE.sub(lambda m: _place_tone(m.group(0)), text)


def normalize(text: str) -> str:
    """NFC + đặt dấu thống nhất + chuẩn hóa khoảng trắng. Không hạ chữ thường, không bỏ dấu câu."""
    text = to_nfc(text)
    text = normalize_tone_placement(text)
    return " ".join(text.split())


def split_syllables(text: str) -> list[str]:
    """Tách câu thành âm tiết; dấu câu là token riêng; số giữ nguyên (vd: "giá: 40.000" -> ["giá", ":", "40.000"])."""
    return _TOKEN_RE.findall(text)


def is_vietnamese_syllable(token: str) -> bool:
    """Kiểm tra token có cấu trúc âm tiết tiếng Việt hợp lệ (phụ âm đầu + vần + thanh).

    Kiểm tra cấu trúc, không kiểm tra từ điển: "tóan" vẫn hợp lệ về cấu trúc.
    """
    base, tone = _split_tone(to_nfc(token))
    base = base.lower()
    m = _SYLLABLE_RE.match(base)
    if not m:
        return False
    initial, nucleus, final = m.group(1), m.group(2), m.group(3)
    if len(nucleus) > 3:
        return False
    # Âm cuối tắc (c, ch, p, t) chỉ đi với thanh sắc hoặc nặng
    if final in {"c", "ch", "p", "t"} and tone not in {None, "́", "̣"}:
        return False
    if final in {"c", "ch", "p", "t"} and tone is None:
        return False
    # Chính tả k/gh/ngh đứng trước i, e, ê, y; c/g/ng không đứng trước i, e, ê
    front = nucleus[0] in "ieêy"
    if initial in {"k", "gh", "ngh"} and not front:
        return False
    if initial in {"c", "ng"} and front:
        return False
    if initial == "g" and nucleus[0] in "eê":
        return False
    return True
