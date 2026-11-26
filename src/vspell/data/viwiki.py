"""Viwiki-spelling (lỗi thật, Wikipedia, CC BY 4.0) -> định dạng chung. Chỉ dùng làm test.

Định dạng gốc (spelling_test.json, mỗi dòng một bài viết):
    {"_id", "text", "mistakes": [{"text", "start_offset" (chuỗi số), "suggest": [...]}], "current_id", "parent_id", "page_id"}
- Lỗi được đánh dấu theo vị trí ký tự trong toàn bài; suggest có thể chứa nhiều gợi ý -> lấy gợi ý đầu.
- Có cả lỗi dính từ ("đượcmột" -> "được một").
Bài viết được tách thành câu; giữ cả câu không lỗi (dùng để đo tỉ lệ sửa quá tay).
Phụ trách: TV2.
"""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

from vspell.normalize import normalize, split_syllables

VIWIKI_URL = "https://raw.githubusercontent.com/heraclex12/Viwiki-spelling/main/spelling_test.json"

# Ứng viên kết thúc câu: . ! ? … (có thể kèm ngoặc/nháy đóng) rồi khoảng trắng. Chỉ cắt khi ký tự chữ/số
# tiếp theo (sau ngoặc/nháy mở) là chữ hoa hoặc số -- kiểm tra bằng str.isupper() để đúng với mọi chữ có dấu
# (regex cũ [^\W_a-zđ] coi "ấ", "ế"... là chữ hoa) -- và từ đứng trước dấu chấm không phải chữ viết tắt.
_SENT_END = re.compile(r"[.!?…]+[\"'”’)\]]*\s+")
# Quy tắc cũ (trước 26/11): coi mọi chữ không phải a-z/đ là chữ hoa. Chỉ giữ để tách Binhvq đúng như lúc
# tạo dữ liệu train đã dùng cho các checkpoint (xem binhvq.split_sentences); không dùng cho dữ liệu mới.
_SENT_END_LEGACY = re.compile(r"[.!?…]+[\"'”’)\]]*\s+(?=[\"'“‘(\[]?[^\W_a-zđ])")
_OPENERS = "\"'“‘(["
# Viết tắt thường đứng trước tên riêng/chữ hoa: không coi dấu chấm sau chúng là hết câu
ABBREVIATIONS = {"tp", "mr", "mrs", "ms", "dr", "st", "gs", "pgs", "ts", "ths", "bs", "th", "q", "p", "tt", "no",
                 "vs", "jr", "sr", "co", "inc", "ltd", "mt", "ft"}
_PREV_WORD = re.compile(r"([^\W\d_]+)\.$")


def _is_sentence_end(line: str, m: re.Match) -> bool:
    j = m.end()
    while j < len(line) and line[j] in _OPENERS:
        j += 1
    if j >= len(line) or not (line[j].isupper() or line[j].isdigit()):
        return False
    if m.group(0)[0] == "." and m.group(0)[1:2] != ".":  # một dấu chấm đơn (không phải "..." hay "?")
        w = _PREV_WORD.search(line[:m.start() + 1])
        if w and (w.group(1).lower() in ABBREVIATIONS or (len(w.group(1)) == 1 and w.group(1).isupper())):
            return False  # "TP. Hồ Chí Minh", "John F. Kennedy"
    return True


def download_viwiki(dest: str | Path, url: str = VIWIKI_URL) -> Path:
    """Tải spelling_test.json nếu chưa có."""
    dest = Path(dest)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def sentence_spans(text: str, legacy: bool = False) -> list[tuple[int, int]]:
    """Các khoảng [start, end) của câu trong bài: tách theo dòng rồi theo dấu kết thúc câu.

    legacy=True: quy tắc cũ (nhận nhầm chữ thường có dấu như "ấ" là đầu câu), chỉ dành cho Binhvq.
    """
    spans = []
    pos = 0
    for line in text.split("\n"):
        start = pos
        for m in (_SENT_END_LEGACY if legacy else _SENT_END).finditer(line):
            if not legacy and not _is_sentence_end(line, m):
                continue
            spans.append((start, pos + m.end()))
            start = pos + m.end()
        spans.append((start, pos + len(line)))
        pos += len(line) + 1
    return [(s, e) for s, e in spans if text[s:e].strip()]


def parse_viwiki_doc(doc: dict, min_syllables: int = 3, stats: dict | None = None) -> tuple[list[dict], int]:
    """Một bài viết -> (danh sách câu {"id", "src", "tgt", "dataset", "doc_id", "n_errors"}, số lỗi bị bỏ).

    Lỗi bị bỏ: vị trí không khớp chữ / không có gợi ý, chồng lấn, hoặc vắt qua ranh giới câu. Câu chạm vào một
    lỗi bị bỏ cũng bị loại (không xuất câu đó với src == tgt như câu sạch, vì nó vẫn chứa lỗi chưa gán nhãn).
    `stats` (nếu có) được cộng dồn: dropped_<lý do>, dropped_sentences, short_sentences.
    """
    st = stats if stats is not None else {}

    def count(key: str, n: int = 1) -> None:
        st[key] = st.get(key, 0) + n

    text = doc["text"]
    mistakes, bad = [], []  # bad: khoảng ký tự của các lỗi không dùng được
    for m in doc["mistakes"]:
        off = int(m["start_offset"])
        span = (off, off + len(m["text"]))
        if text[off:off + len(m["text"])] != m["text"] or not m["suggest"]:
            count("dropped_mismatch")
            bad.append(span)
            continue
        mistakes.append((*span, m["suggest"][0]))
    mistakes.sort()

    records = []
    dropped = len(bad)
    mi = 0
    spans = sentence_spans(text)
    for k, (s, e) in enumerate(spans):
        inside = []
        while mi < len(mistakes) and mistakes[mi][0] < e:
            ms, me, sug = mistakes[mi]
            if ms >= s and me <= e and (not inside or ms >= inside[-1][1]):
                inside.append((ms, me, sug))
            else:
                count("dropped_overlap" if ms >= s and me <= e else "dropped_cross_sentence")
                bad.append((ms, me))
                dropped += 1
            mi += 1
        if any(bs < e and be > s for bs, be in bad):  # câu còn lỗi không gán nhãn được -> loại
            count("dropped_sentences")
            dropped += len(inside)
            continue
        src = text[s:e]
        tgt = src
        for ms, me, sug in reversed(inside):  # thay từ cuối lên để không lệch vị trí
            tgt = tgt[:ms - s] + sug + tgt[me - s:]
        src_n, tgt_n = normalize(src), normalize(tgt)
        if len(split_syllables(src_n)) < min_syllables:
            count("short_sentences")
            dropped += len(inside)
            continue
        records.append({
            "id": f"viwiki-{doc['_id']}-{k}",
            "src": src_n,
            "tgt": tgt_n,
            "dataset": "viwiki",
            "doc_id": str(doc["_id"]),
            "n_errors": len(inside),
        })
    return records, dropped


def load_viwiki(path: str | Path, min_syllables: int = 3, only_with_errors: bool = False,
                stats: dict | None = None) -> list[dict]:
    """Đọc spelling_test.json -> [{"id", "src", "tgt", "dataset": "viwiki", ...}] theo câu.

    `stats`: dict nhận thống kê số lỗi/câu bị bỏ (xem `parse_viwiki_doc`)."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs, _ = parse_viwiki_doc(json.loads(line), min_syllables, stats)
                records.extend(recs)
    if only_with_errors:
        records = [r for r in records if r["n_errors"] > 0]
    return records
