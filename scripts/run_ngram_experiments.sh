#!/usr/bin/env bash
# Chạy toàn bộ thực nghiệm mô hình A: dự đoán của mô hình chính + ablation bậc n-gram/smoothing.
# Giả định đã chạy: python scripts/train_ngram.py --tune  (tạo outputs/checkpoints/ngram/o3_kneser_ney)
# Các mô hình ablation dùng chung tham số giải mã đã chỉnh (configs/ngram.yaml -> decoding).
set -euo pipefail
export PYTHONPATH=src
PY="python -X utf8"

# 1) Mô hình chính trên 3 tập test
for ds in syn_test vsec_test viwiki_test; do
  $PY scripts/predict.py --model ngram --checkpoint outputs/checkpoints/ngram/o3_kneser_ney \
      --tag ngram --data data/processed/$ds.jsonl
done

# 2) Ablation: huấn luyện các biến thể và chấm trên syn_test + vsec_test
for variant in "2 laplace" "2 kneser_ney" "3 laplace" "4 kneser_ney"; do
  set -- $variant
  name="o$1_$2"
  $PY scripts/train_ngram.py --order "$1" --smoothing "$2"
  for ds in syn_test vsec_test; do
    $PY scripts/predict.py --model ngram --checkpoint "outputs/checkpoints/ngram/$name" \
        --tag "ngram_$name" --data data/processed/$ds.jsonl
  done
done

$PY scripts/evaluate.py
