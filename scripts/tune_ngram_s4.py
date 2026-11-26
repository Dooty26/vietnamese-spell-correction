"""S4: dò tham số các luật giảm sửa quá tay của mô hình A trên VSEC-dev (song song nhiều tiến trình).

Luật (configs/ngram.yaml -> decoding): ignore_iy, keep_non_vietnamese, capital_penalty, edit_margin.
Chỉ dùng VSEC-dev; Viwiki-test KHÔNG được dùng để chọn tham số.
Tiêu chí chọn: Cor F0.5 (ưu tiên precision vì tập thật như Viwiki có rất ít lỗi), ghi kèm F1.

Đầu ra:
    outputs/tables/ngram_s4_tuning.csv       mọi cấu hình trong lưới
    {output}/lm.pkl, lexicon.json, config.json   checkpoint A + luật S4 (lm.pkl, lexicon chép từ --checkpoint)

Chạy:
    python scripts/tune_ngram_s4.py --workers 6
"""
import argparse
import csv
import itertools
import json
import shutil
import time
from multiprocessing import Pool
from pathlib import Path

from vspell.utils import get_logger, read_jsonl

GRID = {
    "ignore_iy": [False, True],
    "keep_non_vietnamese": [False, True],
    "capital_penalty": [0.0, -1.0, -2.0, -3.0],
    "edit_margin": [0.0, -0.25, -0.5, -1.0, -1.5],
}

_corr = None
_dev = None


def _init(checkpoint: str, dev_path: str) -> None:
    global _corr, _dev
    from vspell.models.ngram_corrector import NgramCorrector
    from vspell.models.ngram_lm import NgramLM
    d = Path(checkpoint)
    lm = NgramLM.load(d / "lm.pkl")
    lexicon = json.loads((d / "lexicon.json").read_text(encoding="utf-8"))
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    _corr = NgramCorrector(lm, lexicon, cfg)  # bộ đệm ứng viên dùng chung cho mọi cấu hình trong tiến trình
    _dev = list(read_jsonl(dev_path))


def f_beta(p: float, r: float, beta: float) -> float:
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r) if p + r else 0.0


def _run(params: dict) -> dict:
    from vspell.eval.metrics import evaluate
    for k, v in params.items():
        setattr(_corr, k, v)
    t0 = time.time()
    srcs = [r["src"] for r in _dev]
    m = evaluate(srcs, _corr.correct(srcs), [r["tgt"] for r in _dev])
    # Giữ nguyên độ chính xác: chỉ làm tròn khi ghi bảng (select_best xếp hạng trên điểm chưa làm tròn).
    return {**params, **{k: m[k] for k in ("det_p", "det_r", "cor_p", "cor_r", "cor_f1", "over_correction")},
            "cor_f05": f_beta(m["cor_p"], m["cor_r"], 0.5), "n_edits": m["n_edits"],
            "seconds": round(time.time() - t0, 1)}


def select_best(rows: list[dict], metric: str) -> list[dict]:
    """Sắp các cấu hình: điểm `metric` chưa làm tròn giảm dần, hòa thì theo thứ tự cố định của lưới (GRID).

    Kết quả về từ imap_unordered theo thứ tự hoàn thành; khóa phá hòa theo bộ tham số giúp mọi lần chạy
    chọn cùng một cấu hình.
    """
    order = {tuple(vals): i for i, vals in enumerate(itertools.product(*GRID.values()))}
    return sorted(rows, key=lambda r: (-r[metric], order[tuple(r[k] for k in GRID)]))


def _rounded(row: dict) -> dict:
    return {k: round(v, 4) if isinstance(v, float) and k not in GRID and k != "seconds" else v
            for k, v in row.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/checkpoints/ngram/o3_kneser_ney")
    ap.add_argument("--dev", default="data/processed/vsec_dev.jsonl")
    ap.add_argument("--output", default="outputs/checkpoints/ngram/o3_kneser_ney_s4")
    ap.add_argument("--table", default="outputs/tables/ngram_s4_tuning.csv")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--metric", default="cor_f05", choices=["cor_f05", "cor_f1"])
    args = ap.parse_args()
    log = get_logger()

    grid = [dict(zip(GRID, vals)) for vals in itertools.product(*GRID.values())]
    log.info("%d cấu hình, %d tiến trình", len(grid), args.workers)
    rows = []
    with Pool(args.workers, initializer=_init, initargs=(args.checkpoint, args.dev)) as pool:
        for row in pool.imap_unordered(_run, grid):
            rows.append(row)
            log.info("[%d/%d] %s", len(rows), len(grid), row)

    rows = select_best(rows, args.metric)
    table = Path(args.table)
    table.parent.mkdir(parents=True, exist_ok=True)
    with open(table, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(_rounded(r) for r in rows)
    rows = [_rounded(r) for r in rows]
    base = next(r for r in rows if not r["ignore_iy"] and not r["keep_non_vietnamese"]
                and r["capital_penalty"] == 0 and r["edit_margin"] == 0)
    best = rows[0]
    log.info("Mô hình A cũ (dev): %s", base)
    log.info("Chọn theo %s: %s", args.metric, best)

    src, out = Path(args.checkpoint), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    # Luôn chép lại LM + từ điển từ checkpoint nguồn: tham số vừa chọn chỉ đúng với LM đã dùng để chọn
    # (giữ file cũ khi đã có sẽ trộn LM của lần train trước với cấu hình của lần này).
    if src.resolve() != out.resolve():
        for name in ("lm.pkl", "lexicon.json"):
            shutil.copy2(src / name, out / name)
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    cfg["decoding"].update({k: best[k] for k in GRID})
    cfg["s4_selection"] = {"metric": args.metric, "dev": args.dev, **{k: best[k] for k in best if k not in GRID}}
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Đã lưu %s", out)


if __name__ == "__main__":
    main()
