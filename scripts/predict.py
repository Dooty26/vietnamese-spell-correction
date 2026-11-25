"""Bước 4: chạy một mô hình trên một tập test, ghi dự đoán.

Đầu ra mặc định: outputs/predictions/{model}__{tên file data}.jsonl
Mỗi dòng: bản ghi gốc ({"id", "src", "tgt", ...}) + "pred".

Chạy:
    python scripts/predict.py --model identity --data data/processed/vsec_test.jsonl
    python scripts/predict.py --model ngram   --data data/processed/vsec_test.jsonl   # mặc định o3_kneser_ney
    python scripts/predict.py --model ngram --checkpoint outputs/checkpoints/ngram/o2_laplace --tag ngram_o2_laplace --data ...
    python scripts/predict.py --model bartpho --checkpoint outputs/checkpoints/bartpho/best --data ...
"""
import argparse
from pathlib import Path

from vspell.utils import get_logger, read_jsonl, write_jsonl


def load_corrector(name: str, checkpoint: str | None):
    """Trả về hàm list[str] -> list[str]."""
    if name == "identity":  # mốc kiểm tra: không sửa gì
        return lambda sentences: list(sentences)
    if name == "bartpho":
        from vspell.models.bartpho import BartphoCorrector
        return BartphoCorrector(checkpoint).correct
    if name == "ngram":
        import json
        from pathlib import Path

        from vspell.models.ngram_corrector import NgramCorrector
        from vspell.models.ngram_lm import NgramLM

        d = Path(checkpoint or "outputs/checkpoints/ngram/o3_kneser_ney")
        lm = NgramLM.load(d / "lm.pkl")
        lexicon = json.loads((d / "lexicon.json").read_text(encoding="utf-8"))
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        corrector = NgramCorrector(lm, lexicon, cfg)
        from tqdm import tqdm
        return lambda sentences: [corrector.correct_one(s) for s in tqdm(sentences, desc="ngram")]
    # TODO (tùy chọn): bilstm
    raise NotImplementedError(f"Chưa cài mô hình '{name}'")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["identity", "ngram", "bartpho", "bilstm"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--tag", default=None, help="tên mô hình trong bảng kết quả (vd ngram_o2_laplace)")
    ap.add_argument("--limit", type=int, default=None, help="chỉ chạy N câu đầu (thử nhanh)")
    args = ap.parse_args()
    log = get_logger()

    records = list(read_jsonl(args.data))[:args.limit]
    correct = load_corrector(args.model, args.checkpoint)
    preds = correct([r["src"] for r in records])
    for r, p in zip(records, preds):
        r["pred"] = p

    tag = args.tag or args.model
    out = args.output or f"outputs/predictions/{tag}__{Path(args.data).stem}.jsonl"
    n = write_jsonl(records, out)
    log.info("Đã ghi %d dự đoán vào %s", n, out)


if __name__ == "__main__":
    main()
