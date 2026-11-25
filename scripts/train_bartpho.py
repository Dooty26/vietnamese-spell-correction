"""Bước 3C: fine-tune BARTpho-syllable (chạy trên Kaggle/Colab, GPU).

Chạy đầy đủ (150k cặp, 3 epoch):
    python -X utf8 scripts/train_bartpho.py --config configs/bartpho.yaml \
        --train data/processed/syn_train.jsonl --val data/processed/syn_val.jsonl

Ablation mật độ nhiễu (50k cặp, 1 epoch, mỗi mật độ một thư mục):
    python -X utf8 scripts/train_bartpho.py --train data/processed/syn_train_d05.jsonl \
        --limit 50000 --epochs 1 --output-dir outputs/checkpoints/bartpho_d05

Chạy thử trên CPU (200 cặp, vài bước):
    python -X utf8 scripts/train_bartpho.py --limit 200 --val-limit 50 --max-steps 4 \
        --batch-size 4 --eval-steps 2 --output-dir outputs/checkpoints/bartpho_smoke

Kết quả: {output_dir}/best/ (model + tokenizer + vspell_config.json), dùng cho
    python -X utf8 scripts/predict.py --model bartpho --checkpoint {output_dir}/best --data ...
"""
import argparse

from vspell.utils import get_logger, load_yaml, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bartpho.yaml")
    ap.add_argument("--train", default="data/processed/syn_train.jsonl",
                    help="file JSONL train (ablation: syn_train_d05/d12/d25.jsonl)")
    ap.add_argument("--val", default="data/processed/syn_val.jsonl")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--limit", type=int, default=None, help="chỉ dùng N cặp train đầu (ablation 50000; chạy thử 200)")
    ap.add_argument("--val-limit", type=int, default=None, help="chỉ dùng N cặp Syn-val đầu")
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="dừng sau N bước (chạy thử)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--eval-steps", type=int, default=None, help="eval + lưu checkpoint mỗi N bước")
    ap.add_argument("--model-name", default=None, help="ghi đè model_name (vd thư mục mô hình nhỏ để test)")
    ap.add_argument("--resume", action="store_true", help="tiếp tục từ checkpoint-* mới nhất trong output_dir")
    args = ap.parse_args()
    log = get_logger()

    cfg = load_yaml(args.config)
    overrides = {
        "output_dir": args.output_dir, "train_limit": args.limit, "val_limit": args.val_limit,
        "num_train_epochs": args.epochs, "max_steps": args.max_steps,
        "per_device_train_batch_size": args.batch_size, "per_device_eval_batch_size": args.batch_size,
        "eval_steps": args.eval_steps, "save_steps": args.eval_steps, "model_name": args.model_name,
    }
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    if args.resume:
        cfg["resume_from_checkpoint"] = "auto"
    if args.max_steps is not None and args.eval_steps is None:
        cfg["eval_steps"] = cfg["save_steps"] = max(1, args.max_steps // 2)
    set_seed(cfg["seed"])

    from vspell.models.bartpho import train
    best = train(cfg, args.train, args.val)
    log.info("Xong. Checkpoint tốt nhất: %s", best)


if __name__ == "__main__":
    main()
