"""Đóng gói data/processed/*.jsonl để upload lên Kaggle dataset: nén gzip, cắt thành phần < 9 MB.

Tập train chỉ giữ id/src/tgt; tập val/test giữ nguyên (cần `errors` cho bảng theo loại lỗi).
Mỗi tập kèm manifest `{tên}.parts.json` (số phần, số dòng và SHA-256 của mỗi phần, SHA-256 của nội dung ghép): scripts/unpack_parts.py (notebook
04_bartpho) dùng nó để phát hiện phần bị thiếu (kể cả phần cuối) trước khi ghép về data/processed/{tên}.jsonl.
Phải upload cả file manifest cùng các phần.

Chạy: python scripts/pack_for_kaggle.py [--out data/kaggle_upload]
"""
import argparse
import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path

TRAIN = ["syn_train", "syn_train_d05", "syn_train_d12", "syn_train_d25", "syn_train_v4"]
EVAL = ["syn_val", "syn_test", "vsec_test", "viwiki_test"]


def _gz(lines: list[str]) -> bytes:
    return gzip.compress("".join(lines).encode("utf-8"), compresslevel=9, mtime=0)


def split_parts(lines: list[str], limit: int) -> list[tuple[bytes, int]]:
    """Cắt `lines` thành các khối gzip, mỗi khối < limit byte (giữ thứ tự dòng).

    Số dòng mỗi phần được ước lượng từ 20.000 dòng đầu; phần nào nén xong vẫn vượt limit (vì khó nén hơn
    đoạn đầu) thì được chia đôi tiếp cho tới khi vừa. Một dòng đơn lẻ đã vượt limit -> ValueError.
    """
    if not lines:
        return [(_gz([]), 0)]
    sample = _gz(lines[:20000])
    per = max(1, int(limit * 0.9 / (len(sample) / min(len(lines), 20000))))
    parts: list[tuple[bytes, int]] = []

    def add(chunk: list[str]) -> None:
        data = _gz(chunk)
        if len(data) < limit:
            parts.append((data, len(chunk)))
        elif len(chunk) == 1:
            raise ValueError(f"một dòng nén còn {len(data)} byte, vượt giới hạn {limit}")
        else:
            mid = len(chunk) // 2
            add(chunk[:mid])
            add(chunk[mid:])

    for i in range(0, len(lines), per):
        add(lines[i:i + per])
    return parts


def write_parts(lines: list[str], name: str, out: Path, limit: int) -> list[int]:
    """Ghi các phần `{name}.partNN.jsonl.gz` và manifest `{name}.parts.json`. Mọi phần được nén và kiểm tra trước; chỉ khi thành công
    mới xóa các phần cũ cùng tên (lần trước có thể nhiều phần hơn, phần dư sẽ bị notebook ghép vào)."""
    parts = split_parts(lines, limit)
    if len(parts) > 100:
        raise ValueError(f"{name}: {len(parts)} phần, vượt quá 2 chữ số partNN")
    tmp = out / f".{name}.packing"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    try:
        for k, (data, _) in enumerate(parts):
            (tmp / f"{name}.part{k:02d}.jsonl.gz").write_bytes(data)
        manifest = {"name": name, "n_parts": len(parts), "n_lines": len(lines), "lines": [n for _, n in parts],
                    # SHA-256 của từng file .gz và của toàn bộ nội dung đã ghép: phát hiện phần bị thay bằng dữ liệu
                    # của lần chạy khác dù cùng số dòng
                    "sha256": [hashlib.sha256(d).hexdigest() for d, _ in parts],
                    "content_sha256": hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()}
        (tmp / f"{name}.parts.json").write_text(json.dumps(manifest), encoding="utf-8")
        for old in out.glob(f"{name}.part[0-9][0-9].jsonl.gz"):
            old.unlink()
        for f in sorted(tmp.iterdir(), key=lambda f: f.name.endswith(".parts.json")):  # manifest sau cùng
            os.replace(f, out / f.name)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return [len(d) for d, _ in parts]


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
