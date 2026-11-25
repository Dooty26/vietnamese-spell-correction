"""Bước 2: sinh cặp lỗi->đúng từ câu sạch (mỗi dòng file input là một câu).

Chạy:
    python scripts/gen_noise.py --input data/processed/clean_syn_train.txt --output data/processed/syn_train.jsonl
    python scripts/gen_noise.py --input data/processed/clean_syn_val.txt   --output data/processed/syn_val.jsonl   --seed 43
    python scripts/gen_noise.py --input data/processed/clean_syn_test.txt  --output data/processed/syn_test.jsonl  --seed 44
    # Ablation mật độ nhiễu (train nhỏ 50k):
    python scripts/gen_noise.py --input data/processed/clean_syn_train.txt --error-rate 0.05 --limit 50000 \
        --output data/processed/syn_train_r005.jsonl
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from vspell.noise.generator import NoiseGenerator
from vspell.utils import get_logger, load_yaml, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default="configs/noise.yaml")
    ap.add_argument("--error-rate", type=float, default=None, help="ghi đè syllable_error_rate")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None, help="ghi đè seed (nên khác nhau cho train/val/test)")
    args = ap.parse_args()
    log = get_logger()

    cfg = load_yaml(args.config)
    if args.error_rate is not None:
        cfg["syllable_error_rate"] = args.error_rate
    seed = args.seed if args.seed is not None else cfg["seed"]

    sentences = [s for s in Path(args.input).read_text(encoding="utf-8").splitlines() if s.strip()]
    if args.limit:
        sentences = sentences[:args.limit]
    pairs = NoiseGenerator(cfg, seed=seed).make_pairs(sentences, prefix=Path(args.output).stem)
    write_jsonl(pairs, args.output)

    types = Counter(e["type"] for p in pairs for e in p["errors"])
    n_err = sum(types.values())
    n_tok = sum(len(p["tgt"].split()) for p in pairs)
    stats = {
        "n_pairs": len(pairs),
        "n_clean": sum(not p["errors"] for p in pairs),
        "n_errors": n_err,
        "errors_per_word": round(n_err / max(n_tok, 1), 4),
        "error_types": {t: round(n / max(n_err, 1), 4) for t, n in types.most_common()},
        "syllable_error_rate": cfg["syllable_error_rate"],
        "seed": seed,
    }
    Path(args.output).with_suffix(".stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                                            encoding="utf-8")
    log.info("Đã ghi %d cặp vào %s: %s", len(pairs), args.output, stats)


if __name__ == "__main__":
    main()
