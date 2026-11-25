"""Bước 3A: huấn luyện LM n-gram + từ điển âm tiết; đo perplexity; (tùy chọn) chỉnh tham số trên VSEC-dev.

Đầu ra: {output_dir}/{tên}/lm.pkl, lexicon.json, config.json (config cuối cùng, đã gồm tham số chỉnh)
        outputs/tables/ngram_perplexity.csv (cộng dồn qua các lần chạy -> bảng ablation)
        outputs/tables/ngram_tuning_{tên}.csv (nếu --tune)

Chạy:
    python scripts/train_ngram.py                                   # bậc 3, Kneser-Ney
    python scripts/train_ngram.py --order 2 --smoothing laplace     # ablation
    python scripts/train_ngram.py --tune                            # chỉnh oov/real-word penalty trên VSEC-dev
"""
import argparse
import csv
import itertools
import json
import time
from pathlib import Path

from vspell.eval.metrics import evaluate
from vspell.models.ngram_corrector import NgramCorrector, train
from vspell.models.ngram_lm import lm_tokens
from vspell.utils import get_logger, load_yaml, read_jsonl


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ngram.yaml")
    ap.add_argument("--corpus", default="data/processed/lm_corpus.txt")
    ap.add_argument("--heldout", default="data/processed/clean_syn_val.txt")
    ap.add_argument("--dev", default="data/processed/vsec_dev.jsonl")
    ap.add_argument("--order", type=int, default=None)
    ap.add_argument("--smoothing", choices=["kneser_ney", "laplace"], default=None)
    ap.add_argument("--name", default=None, help="tên thư mục mô hình (mặc định: o{order}_{smoothing})")
    ap.add_argument("--tune", action="store_true")
    args = ap.parse_args()
    log = get_logger()

    cfg = load_yaml(args.config)
    if args.order:
        cfg["order"] = args.order
    if args.smoothing:
        cfg["smoothing"] = args.smoothing
    name = args.name or f"o{cfg['order']}_{cfg['smoothing']}"
    out_dir = Path(cfg["output_dir"]) / name

    corpus = Path(args.corpus).read_text(encoding="utf-8").splitlines()
    t0 = time.time()
    lm, lexicon = train(corpus, cfg)
    train_s = time.time() - t0
    heldout = [lm_tokens(s) for s in Path(args.heldout).read_text(encoding="utf-8").splitlines() if s.strip()]
    ppl = lm.perplexity(heldout)
    log.info("%s: %d câu, |V|=%d, |lexicon|=%d, perplexity=%.2f (%.1fs)",
             name, len(corpus), lm.V, len(lexicon), ppl, train_s)
    append_csv(Path("outputs/tables/ngram_perplexity.csv"), {
        "name": name, "order": cfg["order"], "smoothing": cfg["smoothing"], "train_sentences": len(corpus),
        "vocab": lm.V, "heldout_sentences": len(heldout), "perplexity": round(ppl, 2), "train_seconds": round(train_s, 1),
    })

    if args.tune:
        dev = list(read_jsonl(args.dev))[:cfg["tuning"]["n_sentences"]]
        grid = itertools.product(cfg["tuning"]["oov_penalty"], cfg["tuning"]["real_word_penalty"])
        best = None
        for oov, rw in grid:
            cfg["decoding"]["oov_penalty"], cfg["decoding"]["real_word_penalty"] = oov, rw
            corr = NgramCorrector(lm, lexicon, cfg)
            m = evaluate([r["src"] for r in dev], corr.correct([r["src"] for r in dev]), [r["tgt"] for r in dev])
            row = {"oov_penalty": oov, "real_word_penalty": rw,
                   **{k: round(m[k], 4) for k in ("det_p", "det_r", "det_f1", "cor_p", "cor_r", "cor_f1")}}
            append_csv(Path(f"outputs/tables/ngram_tuning_{name}.csv"), row)
            log.info("tune %s", row)
            if best is None or m["cor_f1"] > best[0]:
                best = (m["cor_f1"], oov, rw)
        cfg["decoding"]["oov_penalty"], cfg["decoding"]["real_word_penalty"] = best[1], best[2]
        log.info("Chọn oov_penalty=%s real_word_penalty=%s (cor_f1 dev=%.4f)", best[1], best[2], best[0])

    out_dir.mkdir(parents=True, exist_ok=True)
    lm.save(out_dir / "lm.pkl")
    (out_dir / "lexicon.json").write_text(json.dumps(lexicon, ensure_ascii=False), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Đã lưu mô hình vào %s", out_dir)


if __name__ == "__main__":
    main()
