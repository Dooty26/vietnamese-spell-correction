"""Kiểm tra các file dự đoán cùng tập câu (id, src, tgt) với tập gốc, dùng chung cho evaluate/analyze."""
from __future__ import annotations

from pathlib import Path

from vspell.normalize import normalize
from vspell.utils import read_jsonl


def sentence_set(recs: list[dict], name: str) -> dict:
    """{id: (src, tgt) đã chuẩn hóa}; báo lỗi khi thiếu hoặc trùng id."""
    out = {}
    for r in recs:
        i = r.get("id")
        if i is None or i in out:
            raise ValueError(f"{name}: thiếu id hoặc id trùng ({i})")
        out[i] = (normalize(r["src"]), normalize(r["tgt"]))
    return out


def check_same_sentences(files: dict[str, dict], reference: tuple[str, dict] | None = None) -> None:
    """files: {tên file: sentence_set}. So từng file với `reference` (tập gốc) hoặc với file đầu tiên."""
    ref_name, ref = reference or next(iter(files.items()))
    problems = []
    for name, cur in files.items():
        if cur == ref:
            continue
        missing, extra = ref.keys() - cur.keys(), cur.keys() - ref.keys()
        diff = sum(1 for i in cur.keys() & ref.keys() if cur[i] != ref[i])
        problems.append(f"{name}: {len(cur)} câu so với {len(ref)} của {ref_name} "
                        f"(thiếu {len(missing)}, thừa {len(extra)}, khác src/tgt {diff})")
    if problems:
        raise ValueError("Các file dự đoán không cùng tập câu:" + "".join("\n  " + x for x in problems))


def check_against_gold(files: dict[str, list[dict]], gold: Path, log=None) -> None:
    """files: {tên file: bản ghi}. Đối chiếu với `gold` nếu có, nếu không thì đối chiếu các file với nhau."""
    sets = {name: sentence_set(recs, name) for name, recs in files.items()}
    ref = (gold.name, sentence_set(list(read_jsonl(gold)), str(gold))) if gold.exists() else None
    if ref is None and log is not None:
        log.warning("Không có %s: chỉ đối chiếu %s với nhau", gold, ", ".join(files))
    check_same_sentences(sets, ref)
