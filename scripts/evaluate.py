"""Bước 5: chấm các file dự đoán và gộp thành bảng so sánh.

Đọc mọi file outputs/predictions/{model}__{dataset}.jsonl (có trường src, tgt, pred).
Đầu ra:
    outputs/tables/main_results.csv     mỗi dòng = (model, dataset) + các chỉ số
    outputs/tables/per_error_type.csv   recall theo loại lỗi (chỉ cho tập có trường "errors")

Trước khi chấm, kiểm tra mọi file của cùng một tập có cùng tập câu (id, src, tgt) với
data/processed/{dataset}.jsonl (nếu có) hoặc với nhau: file chạy với --limit, thiếu/trùng id hay khác
nội dung sẽ làm script dừng, vì điểm trên tập con không so được với điểm trên tập đầy đủ.
--allow-mismatch: bỏ qua kiểm tra (chỉ để thử nhanh).

Chạy: python scripts/evaluate.py --pred-dir outputs/predictions --out outputs/tables/main_results.csv
"""
import argparse
import csv
from pathlib import Path

from vspell.eval.metrics import evaluate, per_error_type
from vspell.normalize import normalize
from vspell.utils import get_logger, read_jsonl

METRIC_COLS = ["det_p", "det_r", "det_f1", "cor_p", "cor_r", "cor_f1",
               "sent_acc", "over_correction", "n_sent", "n_errors", "n_edits"]


def _fmt(v):
    return round(v, 4) if isinstance(v, float) else v


def sentence_set(recs: list[dict], name: str) -> dict:
    """{id: (src, tgt) đã chuẩn hóa}; báo lỗi khi thiếu hoặc trùng id."""
    out = {}
    for r in recs:
        i = r.get("id")
        if i is None or i in out:
            raise ValueError(f"{name}: thiếu id hoặc id trùng ({i})")
        out[i] = (normalize(r["src"]), normalize(r["tgt"]))
    return out


def check_same_sentences(files: dict[str, dict], reference: tuple[str, dict] | None = None) -> None:
    """files: {tên file: sentence_set}. So từng file với `reference` (tập gốc) hoặc với file đầu tiên."""
    ref_name, ref = reference or next(iter(files.items()))
    problems = []
    for name, cur in files.items():
        if cur == ref:
            continue
        missing, extra = ref.keys() - cur.keys(), cur.keys() - ref.keys()
        diff = sum(1 for i in cur.keys() & ref.keys() if cur[i] != ref[i])
        problems.append(f"{name}: {len(cur)} câu so với {len(ref)} của {ref_name} "
                        f"(thiếu {len(missing)}, thừa {len(extra)}, khác src/tgt {diff})")
    if problems:
        raise ValueError("Các file dự đoán không cùng tập câu:" + "".join("\n  " + x for x in problems))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="outputs/predictions")
    ap.add_argument("--out", default="outputs/tables/main_results.csv")
    ap.add_argument("--data-dir", default="data/processed", help="tập gốc để đối chiếu id/src/tgt")
    ap.add_argument("--allow-mismatch", action="store_true", help="không kiểm tra tập câu (chỉ để thử nhanh)")
    args = ap.parse_args()
    log = get_logger()

    loaded = []
    for path in sorted(Path(args.pred_dir).glob("*__*.jsonl")):
        model, dataset = path.stem.split("__", 1)
        loaded.append((model, dataset, path, list(read_jsonl(path))))
    if not args.allow_mismatch:
        for ds in sorted({d for _, d, _, _ in loaded}):
            files = {p.name: sentence_set(r, p.name) for _, d, p, r in loaded if d == ds}
            gold = Path(args.data_dir) / f"{ds}.jsonl"
            ref = (gold.name, sentence_set(list(read_jsonl(gold)), str(gold))) if gold.exists() else None
            if ref is None:
                log.warning("Không có %s: chỉ đối chiếu các file %s với nhau", gold, ds)
            check_same_sentences(files, ref)

    rows, type_rows = [], []
    for model, dataset, path, recs in loaded:
        preds = [r["pred"] for r in recs]
        m = evaluate([r["src"] for r in recs], preds, [r["tgt"] for r in recs])
        rows.append({"model": model, "dataset": dataset, **{k: _fmt(m[k]) for k in METRIC_COLS}})
        log.info("%-10s %-15s cor_f1=%.4f det_f1=%.4f", model, dataset, m["cor_f1"], m["det_f1"])
        if any("errors" in r for r in recs):
            for typ, st in per_error_type(recs, preds).items():
                type_rows.append({"model": model, "dataset": dataset, "error_type": typ,
                                  **{k: _fmt(v) for k, v in st.items()}})

    if not rows:
        log.warning("Không tìm thấy file dự đoán nào dạng {model}__{dataset}.jsonl trong %s", args.pred_dir)
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "dataset", *METRIC_COLS])
        w.writeheader()
        w.writerows(rows)
    log.info("Đã ghi %s", out)
    if type_rows:
        type_out = out.with_name("per_error_type.csv")
        with open(type_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(type_rows[0].keys()))
            w.writeheader()
            w.writerows(type_rows)
        log.info("Đã ghi %s", type_out)


if __name__ == "__main__":
    main()
