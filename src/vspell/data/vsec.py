"""VSEC (lỗi thật, miền tin tức) -> định dạng chung + chia dev/test.

Định dạng gốc (VSEC.jsonl, GitHub VSEC2021/VSEC), mỗi dòng một câu:
    {"text": "...", "annotations": [{"id", "current_syllable", "is_correct", "alternative_syllables": [...]}]}
- text = các current_syllable nối bằng dấu cách (dấu câu có thể dính âm tiết: "giá:").
- Âm tiết bị tách được gán nhãn theo kiểu: "nh" -> "" và "iên," -> "nhiên,".
- alternative_syllables có thể có 2 gợi ý -> lấy gợi ý đầu làm tgt, giữ toàn bộ trong "alternatives".

Lỗi gán nhãn đã biết: có chỗ mọi mảnh của âm tiết bị tách đều được gán "" (vd "lư ờng" -> "", "")
làm mất chữ trong tgt. Hàm `_repair_lost_merges` ghép lại các mảnh này khi ghép thành âm tiết hợp lệ.
Phụ trách: TV2.
"""
from __future__ import annotations

import json
import random
import re
import urllib.request
from pathlib import Path

from vspell.normalize import is_vietnamese_syllable, normalize

VSEC_URL = "https://raw.githubusercontent.com/VSEC2021/VSEC/main/Dataset/VSEC.jsonl"


def download_vsec(dest: str | Path, url: str = VSEC_URL) -> Path:
    """Tải VSEC.jsonl nếu chưa có."""
    dest = Path(dest)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def _repair_lost_merges(tokens: list[str], fixes: list[str | None]) -> int:
    """Sửa tại chỗ các chuỗi >=2 mảnh liên tiếp đều bị gán "" mà không có mảnh nào nhận phần ghép.

    fixes[i] = None nếu âm tiết đúng, ngược lại là chuỗi sửa. Trả về số chỗ đã sửa.
    """
    repaired = 0
    i, n = 0, len(tokens)
    while i < n:
        if fixes[i] != "":
            i += 1
            continue
        j = i
        while j < n and fixes[j] == "":
            j += 1
        merged_into_next = j < n and fixes[j] not in (None, "")
        if j - i >= 2 and not merged_into_next:
            joined = "".join(tokens[i:j])
            if is_vietnamese_syllable(joined):
                fixes[i] = joined
                repaired += 1
        i = j
    return repaired


def parse_vsec_record(raw: dict, idx: int) -> dict:
    """Một dòng VSEC -> {"id", "src", "tgt", "dataset", "alternatives", "n_repaired"}."""
    anns = raw["annotations"]
    tokens = [a["current_syllable"] for a in anns]
    fixes: list[str | None] = [
        None if a["is_correct"] else (a["alternative_syllables"][0].strip() if a["alternative_syllables"] else None)
        for a in anns
    ]
    n_repaired = _repair_lost_merges(tokens, fixes)
    tgt_tokens = [tok if fix is None else fix for tok, fix in zip(tokens, fixes)]
    alternatives = {
        str(i): a["alternative_syllables"]
        for i, a in enumerate(anns)
        if not a["is_correct"] and len(a["alternative_syllables"]) > 1
    }
    return {
        "id": f"vsec-{idx}",
        "src": normalize(" ".join(tokens)),
        "tgt": normalize(" ".join(t for t in tgt_tokens if t)),
        "dataset": "vsec",
        "alternatives": alternatives,
        "n_repaired": n_repaired,
    }


def load_vsec(source: str | Path) -> list[dict]:
    """Đọc VSEC.jsonl (định dạng GitHub gốc) -> [{"id", "src", "tgt", "dataset": "vsec", ...}]."""
    records = []
    with open(source, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                records.append(parse_vsec_record(json.loads(line), idx))
    return records


_NON_WORD = re.compile(r"[^\w\s]")


def target_key(text: str) -> str:
    """Khóa gom câu đích trùng nhau: chuẩn hóa, chữ thường, bỏ dấu câu và khoảng trắng thừa."""
    return " ".join(_NON_WORD.sub(" ", normalize(text).lower()).split())


def split_dev_test(records: list[dict], dev_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Chia cố định VSEC-dev / VSEC-test (xáo trộn theo seed, giữ thứ tự id trong từng tập).

    Các câu có cùng câu đích (`target_key`: sau `normalize`, không phân biệt hoa thường và dấu câu) luôn nằm cùng một tập: VSEC có
    những câu đúng giống nhau nhưng mang lỗi khác nhau, nếu tách ra hai tập thì đánh giá sau khi chỉnh tham số
    trên dev không còn độc lập. Cách làm: xáo trộn và cắt như cũ, rồi nhóm nào đã có câu rơi vào dev thì
    chuyển cả nhóm sang dev. Nhờ vậy test chỉ bớt đi đúng các câu trùng với dev, mọi câu khác giữ nguyên tập.
    """
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    n_dev = round(len(records) * dev_ratio)
    key = [target_key(r["tgt"]) for r in records]
    dev_keys = {key[i] for i in order[:n_dev]}
    dev_idx = sorted(i for i in order if key[i] in dev_keys)
    test_idx = sorted(i for i in order if key[i] not in dev_keys)
    return [records[i] for i in dev_idx], [records[i] for i in test_idx]
