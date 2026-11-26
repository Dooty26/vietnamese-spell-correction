"""Đóng gói data/processed/*.jsonl để upload lên Kaggle dataset: nén gzip, cắt thành phần < 9 MB.

Tập train chỉ giữ id/src/tgt; tập val/test giữ nguyên (cần `errors` cho bảng theo loại lỗi).
Notebook 04_bartpho tự ghép các phần `{tên}.partNN.jsonl.gz` về `data/processed/{tên}.jsonl`.

Chạy: python scripts/pack_for_kaggle.py [--out data/kaggle_upload]
"""
import argparse
import gzip
import json
import os
from pathlib import Path

TRAIN = ["syn_train", "syn_train_d05", "syn_train_d12", "syn_train_d25", "syn_train_v4"]
EVAL = ["syn_val", "syn_test", "vsec_test", "viwiki_test"]


def write_parts(lines: list[str], name: str, out: Path, limit: int) -> list[int]:
    # Xóa các phần cũ cùng tên trước: lần trước có thể cắt thành nhiều phần hơn, phần dư sẽ bị notebook ghép vào.
    for old in out.glob(f"{name}.part[0-9][0-9].jsonl.gz"):
        old.unlink()
    sample = gzip.compress("".join(lines[:20000]).encode(), 9)
    per = max(1, int(limit * 0.9 / (len(sample) / min(len(lines), 20000))))
    sizes = []
    for i in range(0, len(lines), per):
        p = out / f"{name}.part{i // per:02d}.jsonl.gz"
        with gzip.open(p, "wt", encoding="utf-8", compresslevel=9) as f:
            f.writelines(lines[i:i + per])
        sizes.append(p.stat().st_size)
        assert sizes[-1] < limit, f"{p} vượt {limit} byte"
    return sizes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed")
    ap.add_argument("--out", default="data/kaggle_upload")
    ap.add_argument("--limit", type=int, default=9_000_000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in TRAIN + EVAL:
        path = Path(args.src) / f"{name}.jsonl"
        if not path.exists():
            print(f"bỏ qua {path} (không có)")
            continue
        with open(path, encoding="utf-8") as f:
            if name in TRAIN:
                lines = [json.dumps({k: r.get(k) for k in ("id", "src", "tgt")}, ensure_ascii=False) + "\n"
                         for r in map(json.loads, f)]
            else:
                lines = f.readlines()
        sizes = write_parts(lines, name, out, args.limit)
        print(f"{name}: {len(lines)} dòng -> {len(sizes)} phần {[round(s / 1e6, 1) for s in sizes]} MB")


if __name__ == "__main__":
    main()
