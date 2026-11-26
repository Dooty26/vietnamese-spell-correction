import gzip
import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pk = _load("pack_for_kaggle")
up = _load("unpack_parts")

LINES = [f'{{"id": {i}, "src": "câu số {i} khá dài để nén", "tgt": "x"}}\n' for i in range(3000)]


def _pack(tmp_path):
    src = tmp_path / "upload"
    src.mkdir()
    pk.write_parts(LINES, "syn_train", src, limit=4000)
    parts = sorted(src.glob("syn_train.part*.jsonl.gz"))
    assert len(parts) >= 3 and (src / "syn_train.parts.json").exists()
    return src, parts


def test_roundtrip(tmp_path):
    src, _ = _pack(tmp_path)
    dest = tmp_path / "processed"
    assert up.unpack(src, dest) == {"syn_train": len(LINES)}
    assert (dest / "syn_train.jsonl").read_text(encoding="utf-8") == "".join(LINES)


def test_corrupt_part_leaves_no_file_and_rerun_rejoins(tmp_path):
    src, parts = _pack(tmp_path)
    dest = tmp_path / "processed"
    good = parts[1].read_bytes()
    parts[1].write_bytes(good[: len(good) // 2])  # phần thứ hai hỏng (tải dở)
    with pytest.raises(Exception):
        up.unpack(src, dest)
    assert not (dest / "syn_train.jsonl").exists() and not list(dest.glob("*.tmp"))
    parts[1].write_bytes(good)  # sửa phần hỏng rồi chạy lại: ghép đủ
    up.unpack(src, dest)
    assert (dest / "syn_train.jsonl").read_text(encoding="utf-8") == "".join(LINES)


def test_rerun_replaces_existing_incomplete_file(tmp_path):
    src, _ = _pack(tmp_path)
    dest = tmp_path / "processed"
    dest.mkdir()
    (dest / "syn_train.jsonl").write_text(LINES[0], encoding="utf-8")  # file thiếu từ lần chạy cũ
    up.unpack(src, dest)
    assert (dest / "syn_train.jsonl").read_text(encoding="utf-8") == "".join(LINES)


def test_missing_middle_part(tmp_path):
    src, parts = _pack(tmp_path)
    (src / "syn_train.parts.json").unlink()
    parts[1].unlink()
    with pytest.raises(ValueError, match=r"thiếu phần \[1\]"):
        up.unpack(src, tmp_path / "processed", allow_no_manifest=True)


def test_missing_last_part_caught_by_manifest(tmp_path):
    src, parts = _pack(tmp_path)
    parts[-1].unlink()
    with pytest.raises(ValueError, match="manifest ghi"):
        up.unpack(src, tmp_path / "processed")
    assert not (tmp_path / "processed" / "syn_train.jsonl").exists()


def test_duplicate_part_and_missing_manifest(tmp_path):
    src, parts = _pack(tmp_path)
    sub = src / "copy"
    sub.mkdir()
    (sub / parts[0].name).write_bytes(parts[0].read_bytes())
    with pytest.raises(ValueError, match="trùng"):
        up.unpack(src, tmp_path / "processed")
    (sub / parts[0].name).unlink()
    (src / "syn_train.parts.json").unlink()
    with pytest.raises(ValueError, match="parts.json"):
        up.unpack(src, tmp_path / "processed")


def test_line_count_mismatch(tmp_path):
    src, parts = _pack(tmp_path)
    with gzip.open(parts[0], "rt", encoding="utf-8") as f:
        lines = f.readlines()
    parts[0].write_bytes(gzip.compress("".join(lines[:-1]).encode()))  # phần đọc được nhưng thiếu dòng
    with pytest.raises(ValueError, match="dòng, manifest ghi"):
        up.unpack(src, tmp_path / "processed")
