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
