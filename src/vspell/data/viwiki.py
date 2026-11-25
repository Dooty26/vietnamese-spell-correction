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

# Kết thúc câu: . ! ? … (có thể kèm ngoặc/nháy đóng) rồi khoảng trắng, sau đó là chữ hoa/số/ngoặc/nháy mở
_SENT_END = re.compile(r"[.!?…]+[\"'”’)\]]*\s+(?=[\"'“‘(\[]?[^\W_a-zđ])")


def download_viwiki(dest: str | Path, url: str = VIWIKI_URL) -> Path:
    """Tải spelling_test.json nếu chưa có."""
    dest = Path(dest)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Các khoảng [start, end) của câu trong bài: tách theo dòng rồi theo dấu kết thúc câu."""
    spans = []
    pos = 0
    for line in text.split("\n"):
        start = pos
        for m in _SENT_END.finditer(line):
            spans.append((start, pos + m.end()))
            start = pos + m.end()
        spans.append((start, pos + len(line)))
        pos += len(line) + 1
    return [(s, e) for s, e in spans if text[s:e].strip()]


def parse_viwiki_doc(doc: dict, min_syllables: int = 3) -> tuple[list[dict], int]:
    """Một bài viết -> (danh sách câu {"id", "src", "tgt", "dataset", "doc_id", "n_errors"}, số lỗi bị bỏ).

    Lỗi bị bỏ: vị trí không khớp chữ, chồng lấn, hoặc vắt qua ranh giới câu.
    """
    text = doc["text"]
    mistakes = []
    dropped = 0
    for m in doc["mistakes"]:
        off = int(m["start_offset"])
        if text[off:off + len(m["text"])] != m["text"] or not m["suggest"]:
            dropped += 1
            continue
        mistakes.append((off, off + len(m["text"]), m["suggest"][0]))
    mistakes.sort()

    records = []
    mi = 0
    for k, (s, e) in enumerate(sentence_spans(text)):
        inside = []
        while mi < len(mistakes) and mistakes[mi][0] < e:
            ms, me, sug = mistakes[mi]
            if ms >= s and me <= e and (not inside or ms >= inside[-1][1]):
                inside.append((ms, me, sug))
            else:
                dropped += 1
            mi += 1
        src = text[s:e]
        tgt = src
        for ms, me, sug in reversed(inside):  # thay từ cuối lên để không lệch vị trí
            tgt = tgt[:ms - s] + sug + tgt[me - s:]
        src_n, tgt_n = normalize(src), normalize(tgt)
        if len(split_syllables(src_n)) < min_syllables:
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


def load_viwiki(path: str | Path, min_syllables: int = 3, only_with_errors: bool = False) -> list[dict]:
    """Đọc spelling_test.json -> [{"id", "src", "tgt", "dataset": "viwiki", ...}] theo câu."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs, _ = parse_viwiki_doc(json.loads(line), min_syllables)
                records.extend(recs)
    if only_with_errors:
        records = [r for r in records if r["n_errors"] > 0]
    return records
