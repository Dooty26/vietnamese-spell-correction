"""Ghép các phần `{tên}.partNN.jsonl(.gz)` (tạo bằng scripts/pack_for_kaggle.py) về `{dest}/{tên}.jsonl`.

Dùng trong notebooks/04_bartpho.ipynb. Mỗi tập được kiểm tra trước khi ghép:
- chỉ số phần liên tiếp từ 00, không trùng (vd có part00 và part02 mà thiếu part01 -> lỗi);
- khớp manifest `{tên}.parts.json` (số phần, số dòng mỗi phần), nên phát hiện cả khi thiếu phần cuối.
Ghép vào file tạm `{tên}.jsonl.tmp`, đọc đủ mới đổi tên thành file đích: lần ghép bị lỗi không để lại
file đích dở dang, và mỗi lần chạy đều ghép lại (không bỏ qua file đã có).
File `{tên}.jsonl` nguyên bản (không chia phần) được liên kết/chép thẳng sang.

Chạy: python scripts/unpack_parts.py --src /kaggle/input --dest data/processed
      --allow-no-manifest: chấp nhận bộ upload cũ chưa có manifest (chỉ kiểm tra chỉ số liên tiếp).
"""
import argparse
import gzip
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

PART_RE = re.compile(r"(.+?)\.part(\d+)\.jsonl(\.gz)?$")
MANIFEST_SUFFIX = ".parts.json"


def find_parts(src: Path):
    """-> (phần theo tên tập, manifest theo tên tập, file .jsonl nguyên bản theo tên file)."""
    parts, manifests, plain = defaultdict(list), {}, {}
    for f in sorted(p for p in src.rglob("*") if p.is_file()):
        m = PART_RE.match(f.name)
        if m:
            parts[m.group(1)].append((int(m.group(2)), f))
        elif f.name.endswith(MANIFEST_SUFFIX):
            name = f.name[: -len(MANIFEST_SUFFIX)]
            if name in manifests:
                raise ValueError(f"{name}: có 2 manifest ({manifests[name]}, {f})")
            manifests[name] = f
        elif f.name.endswith(".jsonl"):
            if f.name in plain:
                raise ValueError(f"{f.name}: có 2 bản ({plain[f.name]}, {f})")
            plain[f.name] = f
    return parts, manifests, plain


def check_parts(name: str, fs: list[tuple[int, Path]], manifest: dict | None) -> list[Path]:
    """Kiểm tra chỉ số phần không trùng, liên tiếp từ 0 và khớp số phần trong manifest."""
    idx = sorted(i for i, _ in fs)
    dup = sorted({i for i in idx if idx.count(i) > 1})
    if dup:
        raise ValueError(f"{name}: phần bị trùng {dup} ({[str(f) for i, f in fs if i in dup]})")
    if idx != list(range(len(idx))):
        missing = sorted(set(range(max(idx) + 1)) - set(idx))
        raise ValueError(f"{name}: thiếu phần {missing} (có {idx})")
    if manifest is not None and manifest["n_parts"] != len(idx):
        raise ValueError(f"{name}: manifest ghi {manifest['n_parts']} phần nhưng có {len(idx)} phần {idx}")
    return [f for _, f in sorted(fs)]


def join_parts(name: str, files: list[Path], dst: Path, manifest: dict | None) -> int:
    """Ghép `files` vào dst qua file tạm; kiểm tra số dòng từng phần theo manifest. Trả về số dòng."""
    tmp = dst.with_name(dst.name + ".tmp")
    total = 0
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as out:
            for k, f in enumerate(files):
                opener = gzip.open if f.suffix == ".gz" else open
                n = 0
                with opener(f, "rt", encoding="utf-8", newline="") as fin:
                    for line in fin:
                        out.write(line)
                        n += 1
                if manifest is not None and manifest["lines"][k] != n:
                    raise ValueError(f"{name}: {f.name} có {n} dòng, manifest ghi {manifest['lines'][k]}")
                total += n
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)
    return total


def unpack(src: Path, dest: Path, allow_no_manifest: bool = False) -> dict[str, int]:
    dest.mkdir(parents=True, exist_ok=True)
    parts, manifests, plain = find_parts(src)
    for fname, f in plain.items():
        if fname[: -len(".jsonl")] in parts:
            raise ValueError(f"{fname}: vừa có bản nguyên vừa có các phần chia nhỏ")
        dst = dest / fname
        if dst.exists() and dst.resolve() == f.resolve():
            continue
        dst.unlink(missing_ok=True)
        try:
            os.symlink(f, dst)
        except OSError:  # Windows không có quyền tạo symlink
            shutil.copyfile(f, dst)
    counts = {}
    for name in sorted(set(parts) | set(manifests)):
        manifest = json.loads(manifests[name].read_text(encoding="utf-8")) if name in manifests else None
        if name not in parts:
            raise ValueError(f"{name}: có manifest nhưng không có phần nào")
        if manifest is None and not allow_no_manifest:
            raise ValueError(f"{name}: không có {name}{MANIFEST_SUFFIX} (đóng gói lại bằng pack_for_kaggle.py "
                             "hoặc chạy với --allow-no-manifest)")
        files = check_parts(name, parts[name], manifest)
        counts[name] = join_parts(name, files, dest / f"{name}.jsonl", manifest)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dest", default="data/processed")
    ap.add_argument("--allow-no-manifest", action="store_true")
    args = ap.parse_args()
    for name, n in unpack(Path(args.src), Path(args.dest), args.allow_no_manifest).items():
        print(f"{name}: {n} dòng")


if __name__ == "__main__":
    main()
