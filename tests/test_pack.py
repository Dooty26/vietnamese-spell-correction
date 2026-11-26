import gzip
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "pack_for_kaggle", Path(__file__).resolve().parents[1] / "scripts" / "pack_for_kaggle.py")
pk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pk)


def _read_all(out: Path, name: str) -> list[str]:
    """Ghép các phần như notebook 04_bartpho: mọi file {name}.partNN.jsonl.gz theo thứ tự."""
    lines = []
    for f in sorted(out.glob(f"{name}.part*.jsonl.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as fin:
            lines += fin.readlines()
    return lines


def test_repack_removes_stale_parts(tmp_path):
    big = [f'{{"id": {i}, "src": "câu số {i} khá dài để nén", "tgt": "x"}}\n' for i in range(3000)]
    pk.write_parts(big, "syn_train", tmp_path, limit=4000)
    assert len(list(tmp_path.glob("syn_train.part*"))) > 1
    small = big[:2]
    pk.write_parts(small, "syn_train", tmp_path, limit=4000)
    assert _read_all(tmp_path, "syn_train") == small
    other = tmp_path / "syn_train_v4.part00.jsonl.gz"  # tập khác tên không bị xóa
    pk.write_parts(small, "syn_train_v4", tmp_path, limit=4000)
    pk.write_parts(small, "syn_train", tmp_path, limit=4000)
    assert other.exists()


def _hard_tail(n_easy=2000, n_hard=400):
    """Đầu dễ nén, đuôi ngẫu nhiên khó nén: ước lượng từ 20.000 dòng đầu sẽ quá lạc quan cho phần sau."""
    import random
    rnd = random.Random(0)
    easy = [f'{{"id": {i}, "src": "aaaa aaaa aaaa", "tgt": "aaaa"}}\n' for i in range(n_easy)]
    hard = ["".join(rnd.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(200)) + "\n"
            for _ in range(n_hard)]
    return easy + hard


def test_parts_split_further_when_later_lines_compress_worse(tmp_path):
    lines = _hard_tail()
    sizes = pk.write_parts(lines, "syn_train", tmp_path, limit=6000)
    assert all(s < 6000 for s in sizes) and len(sizes) > 2
    assert _read_all(tmp_path, "syn_train") == lines


def test_old_parts_kept_when_packing_fails(tmp_path):
    import pytest
    old = [f'{{"id": {i}}}\n' for i in range(5)]
    pk.write_parts(old, "syn_train", tmp_path, limit=4000)
    with pytest.raises(ValueError, match="vượt giới hạn"):
        pk.write_parts(["x" * 50 + "\n", "".join(chr(0x4e00 + (i * 7919) % 20000) for i in range(3000)) + "\n"],
                       "syn_train", tmp_path, limit=1000)
    assert _read_all(tmp_path, "syn_train") == old
    assert not list(tmp_path.glob(".*packing"))
