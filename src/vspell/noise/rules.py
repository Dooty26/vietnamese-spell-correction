"""Các luật sinh lỗi trên MỘT âm tiết (đã chuẩn hóa NFC). Mỗi hàm trả về âm tiết lỗi (khác đầu vào)
hoặc None nếu không áp dụng được. Kết quả luôn là một token chữ cái (không chứa dấu cách, dấu câu).

Tỉ lệ loại lỗi cấu hình trong configs/noise.yaml. Phụ trách: TV1.
"""
from __future__ import annotations

import random
import unicodedata

from vspell.normalize import (TONE_ACUTE, TONE_DOT, TONE_GRAVE, TONE_HOOK, TONE_TILDE, add_tone,
                              is_vietnamese_syllable, parse_syllable, remove_diacritics, split_tone)

_LETTERS = "aăâbcdđeêghiklmnoôơpqrstuưvxy"


def _match_case(new: str, old: str) -> str:
    """Giữ kiểu chữ hoa của âm tiết gốc (toàn hoa / hoa chữ đầu / thường)."""
    if old.isupper() and len(old) > 1:
        return new.upper()
    if old[:1].isupper():
        return new[:1].upper() + new[1:]
    return new


def _decompose(syl: str):
    """-> (initial, nucleus, final, tone) cho âm tiết chữ thường; None nếu không phải âm tiết TV."""
    base, tone = split_tone(syl.lower())
    parts = parse_syllable(base)
    if parts is None:
        return None
    return (*parts, tone)


def _finish(new_lower: str, syl: str) -> str | None:
    if not new_lower or new_lower == syl.lower():
        return None
    return _match_case(new_lower, syl)


# ---------------------------------------------------------------- vùng miền
_INITIAL_SWAPS = {
    "s": ["x"], "x": ["s"], "tr": ["ch"], "ch": ["tr"],
    "r": ["d", "gi"], "d": ["r", "gi", "v"], "gi": ["d", "r"], "v": ["d"],
    "l": ["n"], "n": ["l"],
}
_FINAL_SWAPS = {"n": ["ng"], "ng": ["n"], "t": ["c"], "c": ["t"], "nh": ["n"], "ch": ["t"]}
_TONE_SWAPS = {TONE_HOOK: [TONE_TILDE], TONE_TILDE: [TONE_HOOK], TONE_DOT: [TONE_HOOK]}


def region_error(syl: str, rng: random.Random) -> str | None:
    """Lỗi phát âm vùng miền: s/x, tr/ch, r/d/gi, l/n, -n/-ng, -t/-c, hỏi/ngã."""
    d = _decompose(syl)
    if d is None:
        return None
    ini, nuc, fin, tone = d
    options = []
    for new in _INITIAL_SWAPS.get(ini, []):
        if new == "gi" and nuc.startswith("i"):
            continue
        options.append((new, nuc, fin, tone))
    for new in _FINAL_SWAPS.get(fin, []):
        options.append((ini, nuc, new, tone))
    if fin not in {"c", "ch", "p", "t"}:  # âm cuối tắc chỉ đi với sắc/nặng
        for new in _TONE_SWAPS.get(tone, []):
            options.append((ini, nuc, fin, new))
    rng.shuffle(options)
    for i, n, f, t in options:  # lỗi vùng miền tạo ra âm tiết có thật (lỗi real-word)
        new = add_tone(i + n + f, t)
        if is_vietnamese_syllable(new):
            return _finish(new, syl)
    return None


# ---------------------------------------------------------------- Telex
_TELEX_CHAR = {"â": "aa", "ă": "aw", "ê": "ee", "ô": "oo", "ơ": "ow", "ư": "uw", "đ": "dd"}
_TELEX_TONE = {TONE_ACUTE: "s", TONE_GRAVE: "f", TONE_HOOK: "r", TONE_TILDE: "x", TONE_DOT: "j"}


def telex_error(syl: str, rng: random.Random) -> str | None:
    """Lỗi gõ Telex chưa được bộ gõ chuyển đổi:
    - phím dấu thanh bị để lại cuối âm tiết: "được" -> "đươcj";
    - một dấu phụ không được chuyển: "được" -> "dduwowcj" (toàn bộ) hoặc "đuwơc" (một ký tự);
    """
    low = syl.lower()
    base, tone = split_tone(low)
    options = []
    if tone:
        options.append(base + _TELEX_TONE[tone])
    specials = [i for i, c in enumerate(base) if c in _TELEX_CHAR]
    if specials:
        i = rng.choice(specials)
        options.append(add_tone(base[:i] + _TELEX_CHAR[base[i]] + base[i + 1:], tone))
        full = "".join(_TELEX_CHAR.get(c, c) for c in base) + (_TELEX_TONE[tone] if tone else "")
        options.append(full)
    if not options:
        return None
    return _finish(rng.choice(options), syl)


# ---------------------------------------------------------------- trượt phím
_QWERTY = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
_NEIGHBORS: dict[str, str] = {}
for r, row in enumerate(_QWERTY):
    for c, ch in enumerate(row):
        near = [row[c - 1] if c > 0 else "", row[c + 1] if c + 1 < len(row) else ""]
        for rr in (r - 1, r + 1):
            if 0 <= rr < len(_QWERTY):
                near += [x for x in _QWERTY[rr][max(c - 1, 0):c + 2]]
        _NEIGHBORS[ch] = "".join(x for x in near if x)


def fat_finger_error(syl: str, rng: random.Random) -> str | None:
    """Gõ nhầm phím lân cận trên QWERTY (chữ bị gõ nhầm mất dấu phụ của ký tự đó)."""
    low = syl.lower()
    positions = [i for i, c in enumerate(low) if remove_diacritics(c) in _NEIGHBORS]
    if not positions:
        return None
    i = rng.choice(positions)
    new_char = rng.choice(_NEIGHBORS[remove_diacritics(low[i])])
    if rng.random() < 0.3:  # thỉnh thoảng chèn thêm phím lân cận thay vì thay thế
        return _finish(low[:i + 1] + new_char + low[i + 1:], syl)
    return _finish(low[:i] + new_char + low[i + 1:], syl)


# ---------------------------------------------------------------- teencode
_TEEN_WORDS = {
    "không": ["ko", "k", "hông", "khum", "khong"], "gì": ["j", "gi"], "vậy": ["z", "zậy", "v"],
    "rồi": ["rùi", "r", "ròi"], "biết": ["bít", "bik"], "yêu": ["iu"], "quá": ["wá", "qá"],
    "được": ["đc", "dc", "đượk"], "anh": ["a"], "em": ["e"], "với": ["vs", "zới"], "thôi": ["thui"],
    "luôn": ["lun"], "chồng": ["ck"], "vợ": ["vk"], "bạn": ["bn"], "thế": ["thía"], "này": ["nì", "nài"],
    "nhiều": ["nhìu"], "mình": ["mk", "mềnh"], "người": ["ng"], "chưa": ["chx"], "của": ["củ"],
}
_TEEN_INITIAL = {"ph": "f", "gi": "j", "qu": "w", "d": "z", "v": "z", "c": "k", "k": "c"}


def teencode_error(syl: str, rng: random.Random) -> str | None:
    """Teencode: từ quen thuộc ("không" -> "ko", "gì" -> "j") hoặc thay phụ âm đầu (ph->f, gi->j, qu->w, d/v->z)."""
    low = syl.lower()
    if low in _TEEN_WORDS:
        return _finish(rng.choice(_TEEN_WORDS[low]), syl)
    d = _decompose(low)
    if d is None:
        return None
    ini, nuc, fin, tone = d
    if ini in _TEEN_INITIAL:
        return _finish(add_tone(_TEEN_INITIAL[ini] + nuc + fin, tone), syl)
    return None


# ---------------------------------------------------------------- viết tắt
_ACRONYMS = {
    "được": "đc", "người": "ng", "không": "k", "với": "vs", "vậy": "v", "rồi": "r", "trước": "trc",
    "những": "nhg", "chúng": "cg", "thành": "tp", "phố": "ph", "việt": "vn", "nam": "n", "các": "cac",
    "bởi": "bởi", "cũng": "cg", "nhưng": "nhg", "trong": "trg", "đồng": "đ", "tháng": "th",
}


def acronym_error(syl: str, rng: random.Random) -> str | None:
    """Viết tắt phổ biến khi gõ nhanh: "được" -> "đc", "người" -> "ng", "trước" -> "trc"."""
    return _finish(_ACRONYMS.get(syl.lower(), ""), syl)


# ---------------------------------------------------------------- edit distance
def edit_distance_error(syl: str, rng: random.Random) -> str | None:
    """Chèn/xóa/thay/hoán vị một ký tự ngẫu nhiên, hoặc bỏ hết dấu ("không" -> "khong")."""
    low = syl.lower()
    op = rng.choice(["insert", "delete", "substitute", "transpose", "unaccent"])
    i = rng.randrange(len(low))
    if op == "insert":
        new = low[:i] + rng.choice(_LETTERS) + low[i:]
    elif op == "delete" and len(low) > 1:
        new = low[:i] + low[i + 1:]
    elif op == "substitute":
        new = low[:i] + rng.choice(_LETTERS) + low[i + 1:]
    elif op == "transpose" and len(low) > 1:
        i = min(i, len(low) - 2)
        new = low[:i] + low[i + 1] + low[i] + low[i + 2:]
    else:
        new = remove_diacritics(low)
    return _finish(unicodedata.normalize("NFC", new), syl)


RULES = {
    "region": region_error,
    "telex": telex_error,
    "fat_finger": fat_finger_error,
    "teencode": teencode_error,
    "acronym": acronym_error,
    "edit_distance": edit_distance_error,
}
