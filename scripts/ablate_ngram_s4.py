"""S4: ablation từng luật giảm sửa quá tay (chỉ để BÁO CÁO; tham số đã chọn trên VSEC-dev bằng tune_ngram_s4.py).

Mỗi biến thể = mô hình A ban đầu + bật đúng một luật (giá trị đã chọn). Hai dòng "A" và "A+S4" có sẵn ở main_results.csv.
Đầu ra: outputs/tables/ngram_s4_ablation.csv

Chạy: python scripts/ablate_ngram_s4.py --workers 4
"""
import argparse
import csv
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

from vspell.utils import get_logger, read_jsonl

CHOSEN = {"ignore_iy": True, "keep_non_vietnamese": True, "capital_penalty": -2.0, "edit_margin": -0.5}
OFF = {"ignore_iy": False, "keep_non_vietnamese": False, "capital_penalty": 0.0, "edit_margin": 0.0}

_corr = None


def _init(checkpoint: str) -> None:
    global _corr
    from vspell.models.ngram_corrector import NgramCorrector
    from vspell.models.ngram_lm import NgramLM
    d = Path(checkpoint)
    lm = NgramLM.load(d / "lm.pkl")
    lexicon = json.loads((d / "lexicon.json").read_text(encoding="utf-8"))
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    _corr = NgramCorrector(lm, lexicon, cfg)


def _run(job: tuple[str, str]) -> dict:
    from vspell.eval.metrics import evaluate
    rule, data = job
    for k, v in {**OFF, rule: CHOSEN[rule]}.items():
        setattr(_corr, k, v)
    recs = list(read_jsonl(data))
    srcs = [r["src"] for r in recs]
    m = evaluate(srcs, _corr.correct(srcs), [r["tgt"] for r in recs])
    return {"variant": f"A + {rule}", "dataset": Path(data).stem,
            **{k: round(m[k], 4) for k in ("cor_p", "cor_r", "cor_f1", "sent_acc", "over_correction")},
            "n_edits": m["n_edits"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/checkpoints/ngram/o3_kneser_ney")
    ap.add_argument("--data", nargs="+", default=["data/processed/viwiki_test.jsonl", "data/processed/vsec_test.jsonl"])
    ap.add_argument("--out", default="outputs/tables/ngram_s4_ablation.csv")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    log = get_logger()
    jobs = list(itertools.product(CHOSEN, args.data))
    rows = []
    with Pool(args.workers, initializer=_init, initargs=(args.checkpoint,)) as pool:
        for row in pool.imap_unordered(_run, jobs):
            rows.append(row)
            log.info("[%d/%d] %s", len(rows), len(jobs), row)
    rows.sort(key=lambda r: (r["dataset"], r["variant"]))
    out = Path(args.out)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    log.info("Đã ghi %s", out)


if __name__ == "__main__":
    main()
