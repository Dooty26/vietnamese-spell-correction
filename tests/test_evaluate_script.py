import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("evaluate_script", ROOT / "scripts" / "evaluate.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

RECS = [{"id": 1, "src": "đều chỉnh", "tgt": "điều chỉnh", "pred": "điều chỉnh"},
        {"id": 2, "src": "trời đẹp", "tgt": "trời đẹp", "pred": "trời đẹp"}]


def test_check_same_sentences():
    full = ev.sentence_set(RECS, "a")
    ev.check_same_sentences({"a": full, "b": ev.sentence_set(RECS[::-1], "b")})  # khác thứ tự: vẫn được
    with pytest.raises(ValueError, match="thiếu 1"):
        ev.check_same_sentences({"a": full, "b": ev.sentence_set(RECS[:1], "b")})
    changed = [dict(RECS[0], tgt="đều chỉnh"), RECS[1]]
    with pytest.raises(ValueError, match="khác src/tgt 1"):
        ev.check_same_sentences({"b": ev.sentence_set(changed, "b")}, ("gold", full))
    with pytest.raises(ValueError, match="id trùng"):
        ev.sentence_set([RECS[0], RECS[0]], "c")


def _write(path, recs):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")


def test_script_rejects_subset(tmp_path):
    pred, data = tmp_path / "pred", tmp_path / "data"
    pred.mkdir(), data.mkdir()
    _write(data / "vsec_test.jsonl", RECS)
    _write(pred / "a__vsec_test.jsonl", RECS)
    _write(pred / "b__vsec_test.jsonl", RECS[:1])  # như chạy với --limit 1
    cmd = [sys.executable, "-X", "utf8", str(ROOT / "scripts/evaluate.py"), "--pred-dir", str(pred),
           "--data-dir", str(data), "--out", str(tmp_path / "out.csv")]
    env = {**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, encoding="utf-8")
    assert r.returncode != 0 and "không cùng tập câu" in r.stderr and not (tmp_path / "out.csv").exists()
    r = subprocess.run(cmd + ["--allow-mismatch"], capture_output=True, text=True, env=env, encoding="utf-8")
    assert r.returncode == 0 and (tmp_path / "out.csv").exists()
