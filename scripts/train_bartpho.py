"""Bước 3C: fine-tune BARTpho-syllable (chạy trên Kaggle/Colab Pro).

Chạy:
    python scripts/train_bartpho.py --config configs/bartpho.yaml \
        --train data/processed/syn_train.jsonl --val data/processed/syn_val.jsonl
"""
import argparse

from vspell.utils import load_yaml, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bartpho.yaml")
    ap.add_argument("--train", default="data/processed/syn_train.jsonl")
    ap.add_argument("--val", default="data/processed/syn_val.jsonl")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    set_seed(cfg["seed"])
    # TODO(TV3): vspell.models.bartpho.train(cfg, args.train, args.val)
    raise NotImplementedError


if __name__ == "__main__":
    main()
