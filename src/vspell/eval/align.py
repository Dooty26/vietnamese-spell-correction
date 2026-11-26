"""Căn chỉnh âm tiết giữa src và một chuỗi khác (tgt hoặc dự đoán) theo khoảng cách chỉnh sửa (Levenshtein).

Dùng số phép sửa ít nhất (thay/chèn/xóa một âm tiết, chi phí 1), không dùng difflib: SequenceMatcher ưu tiên
cụm trùng dài nhất nên có thể đếm thừa lỗi (vd "tôi học tôi học" -> "tôi hỏi tôi học" thành 3 thay đổi thay vì 1).

Mọi thay đổi được quy về vị trí âm tiết trong src: {chỉ số âm tiết src: chuỗi thay thế}.
- Thay 1-1: "đều" -> "điều"                     => {i: "điều"}
- Tách/ghép: "nh iên" -> "nhiên"                 => {i: "nhiên", i+1: ""}
- Chèn thêm âm tiết sau vị trí i                 => {i: "src[i] <âm tiết chèn>"}
Nhờ đó Detection/Correction được so trên cùng một hệ chỉ số cho cả gold và dự đoán.
Phụ trách: TV2.
"""
from __future__ import annotations

Opcode = tuple[str, int, int, int, int]


def _band_dp(a: list[str], b: list[str], k: int) -> dict[tuple[int, int], int] | None:
    """Levenshtein trong dải |i - j| <= k (Ukkonen). Trả None nếu khoảng cách > k (cần dải rộng hơn)."""
    n, m = len(a), len(b)
    INF = n + m + 1
    D: dict[tuple[int, int], int] = {}
    for i in range(n + 1):
        for j in range(max(0, i - k), min(m, i + k) + 1):
            if i == 0 or j == 0:
                D[i, j] = i + j
                continue
            D[i, j] = min(D[i - 1, j - 1] + (a[i - 1] != b[j - 1]),
                          D.get((i - 1, j), INF) + 1, D.get((i, j - 1), INF) + 1)
    return D if D.get((n, m), INF) <= k else None


def opcodes(a: list[str], b: list[str]) -> list[Opcode]:
    """Như `SequenceMatcher.get_opcodes()` nhưng theo căn chỉnh có số phép sửa ít nhất.

    Các phép sửa liền nhau gộp thành một khối: "replace" (có thể khác độ dài), "delete" hoặc "insert".
    """
    n, m = len(a), len(b)
    pre = 0
    while pre < min(n, m) and a[pre] == b[pre]:
        pre += 1
    suf = 0
    while suf < min(n, m) - pre and a[n - 1 - suf] == b[m - 1 - suf]:
        suf += 1
    ca, cb = a[pre:n - suf], b[pre:m - suf]
    k = max(abs(len(ca) - len(cb)), 1)
    while (D := _band_dp(ca, cb, k)) is None:
        k *= 2
    # truy vết từ cuối: ưu tiên khớp, rồi thay, xóa, chèn -> danh sách (khớp?, i, j) theo âm tiết
    INF = len(ca) + len(cb) + 1
    steps: list[tuple[bool, int, int]] = []  # (là khớp, bước i, bước j)
    i, j = len(ca), len(cb)
    while i > 0 or j > 0:
        cur = D[i, j]
        if i > 0 and j > 0 and ca[i - 1] == cb[j - 1] and D[i - 1, j - 1] == cur:
            steps.append((True, 1, 1)); i -= 1; j -= 1
        elif i > 0 and j > 0 and D[i - 1, j - 1] + 1 == cur:
            steps.append((False, 1, 1)); i -= 1; j -= 1
        elif i > 0 and D.get((i - 1, j), INF) + 1 == cur:
            steps.append((False, 1, 0)); i -= 1
        else:
            steps.append((False, 0, 1)); j -= 1
    steps.reverse()
    ops: list[Opcode] = [("equal", 0, pre, 0, pre)] if pre else []
    i = j = pre
    for match, di, dj in steps:
        tag = "equal" if match else "edit"
        if ops and (ops[-1][0] == "equal") == match:
            t, i1, _, j1, _ = ops[-1]
            ops[-1] = (t, i1, i + di, j1, j + dj)
        else:
            ops.append((tag, i, i + di, j, j + dj))
        i += di
        j += dj
    if suf:
        if ops and ops[-1][0] == "equal":
            t, i1, _, j1, _ = ops[-1]
            ops[-1] = (t, i1, n, j1, m)
        else:
            ops.append(("equal", i, n, j, m))
    out: list[Opcode] = []
    for t, i1, i2, j1, j2 in ops:
        if t == "edit":
            t = "replace" if i2 > i1 and j2 > j1 else ("delete" if i2 > i1 else "insert")
        out.append((t, i1, i2, j1, j2))
    return out


def align(src_tokens: list[str], other_tokens: list[str]) -> list[tuple[int | None, int | None]]:
    """Ghép cặp chỉ số âm tiết src <-> other; None khi chèn/xóa."""
    pairs: list[tuple[int | None, int | None]] = []
    for tag, i1, i2, j1, j2 in opcodes(src_tokens, other_tokens):
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
    # Vị trí được neo vào âm tiết src: nguồn rỗng không có chỗ để ghi phần sinh thêm, nên nếu bỏ qua thì
    # nội dung thừa không bị tính vào precision. Từ chối thay vì chấm sai.
    if not src_tokens and other_tokens:
        raise ValueError(f"Câu nguồn rỗng nhưng câu đích/dự đoán có {len(other_tokens)} âm tiết "
                         f"({' '.join(other_tokens[:5])!r}…): không căn chỉnh được theo vị trí âm tiết nguồn")
    changes: dict[int, str] = {}
    for tag, i1, i2, j1, j2 in opcodes(src_tokens, other_tokens):
        if tag == "equal":
            continue
        new = other_tokens[j1:j2]
        if tag == "insert":
            if i1 > 0:
                anchor = i1 - 1
                prev = changes.get(anchor, src_tokens[anchor])
                changes[anchor] = " ".join(t for t in [prev, *new] if t)
            else:
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
