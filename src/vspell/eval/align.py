"""Căn chỉnh âm tiết giữa src và một chuỗi khác (tgt hoặc dự đoán) bằng difflib.

Mọi thay đổi được quy về vị trí âm tiết trong src: {chỉ số âm tiết src: chuỗi thay thế}.
- Thay 1-1: "đều" -> "điều"                     => {i: "điều"}
- Tách/ghép: "nh iên" -> "nhiên"                 => {i: "nhiên", i+1: ""}
- Chèn thêm âm tiết sau vị trí i                 => {i: "src[i] <âm tiết chèn>"}
Nhờ đó Detection/Correction được so trên cùng một hệ chỉ số cho cả gold và dự đoán.
Phụ trách: TV2.
"""
from __future__ import annotations

from difflib import SequenceMatcher


def align(src_tokens: list[str], other_tokens: list[str]) -> list[tuple[int | None, int | None]]:
    """Ghép cặp chỉ số âm tiết src <-> other; None khi chèn/xóa."""
    pairs: list[tuple[int | None, int | None]] = []
    sm = SequenceMatcher(None, src_tokens, other_tokens, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            pairs.extend(zip(range(i1, i2), range(j1, j2)))
        elif tag == "delete":
            pairs.extend((i, None) for i in range(i1, i2))
        elif tag == "insert":
            pairs.extend((None, j) for j in range(j1, j2))
        else:  # replace khác độ dài
            n = min(i2 - i1, j2 - j1)
            pairs.extend(zip(range(i1, i1 + n), range(j1, j1 + n)))
            pairs.extend((i, None) for i in range(i1 + n, i2))
            pairs.extend((None, j) for j in range(j1 + n, j2))
    return pairs


def _changes(src_tokens: list[str], other_tokens: list[str]) -> dict[int, str]:
    changes: dict[int, str] = {}
    sm = SequenceMatcher(None, src_tokens, other_tokens, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        new = other_tokens[j1:j2]
        if tag == "insert":
            if i1 > 0:
                anchor = i1 - 1
                prev = changes.get(anchor, src_tokens[anchor])
                changes[anchor] = " ".join(t for t in [prev, *new] if t)
            elif src_tokens:
                changes[0] = " ".join([*new, changes.get(0, src_tokens[0])])
            continue
        n_src = i2 - i1
        if tag == "replace" and n_src == len(new):
            for k in range(n_src):
                changes[i1 + k] = new[k]
            continue
        # delete hoặc replace khác độ dài: âm tiết src cuối cùng của khối gánh phần dư
        for k in range(n_src):
            if k < n_src - 1:
                changes[i1 + k] = new[k] if k < len(new) else ""
            else:
                changes[i1 + k] = " ".join(new[k:])
    return changes


def error_positions(src_tokens: list[str], tgt_tokens: list[str]) -> dict[int, str]:
    """Vị trí âm tiết sai trong src -> chuỗi đúng tương ứng trong tgt."""
    return _changes(src_tokens, tgt_tokens)


def edit_positions(src_tokens: list[str], pred_tokens: list[str]) -> dict[int, str]:
    """Vị trí âm tiết mô hình đã thay đổi -> chuỗi mô hình đề xuất."""
    return _changes(src_tokens, pred_tokens)
