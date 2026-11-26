# Phát hiện và sửa lỗi chính tả tiếng Việt: thống kê và học sâu

Bài tập lớn môn Xử lý ngôn ngữ tự nhiên. Kế hoạch chi tiết: [report/ke_hoach.md](report/ke_hoach.md).

## Mục tiêu
Nhập một câu tiếng Việt có thể chứa lỗi chính tả → chỉ ra âm tiết sai và đề xuất câu đã sửa. So sánh:
- **A – N-gram LM** (KenLM Kneser-Ney + tập dễ nhầm + beam search): baseline thống kê
- **C – BARTpho-syllable** fine-tune seq2seq: mô hình học sâu
- *(B – BiLSTM + char-CNN: tùy chọn)*

## Dữ liệu
| Dataset | Nguồn | Vai trò |
|---|---|---|
| Binhvq news | [vietgpt/binhvq_news_vi](https://huggingface.co/datasets/vietgpt/binhvq_news_vi) | Corpus sạch: LM n-gram + sinh dữ liệu lỗi tổng hợp |
| VSEC | [VSEC2021/VSEC](https://github.com/VSEC2021/VSEC) | Lỗi thật (tin tức): dev 20% / test 80% |
| Viwiki-spelling | [heraclex12/Viwiki-spelling](https://github.com/heraclex12/Viwiki-spelling) (CC BY 4.0) | Lỗi thật (Wikipedia): chỉ test |

Dữ liệu không được đưa lên repo; các script bên dưới tự tải và tái tạo.

## Cấu trúc thư mục
```
configs/      tham số dữ liệu, sinh nhiễu, n-gram, BARTpho
src/vspell/   mã nguồn chính (normalize, data, noise, models, eval)
scripts/      CLI chạy từng bước
notebooks/    notebook mỏng gọi src/ (EDA, huấn luyện, phân tích)
data/         raw/ interim/ processed/ (gitignore)
outputs/      predictions/ tables/ figures/ checkpoints/
demo/         ứng dụng Gradio
tests/        kiểm thử normalize, noise, metrics
report/       kế hoạch, dàn ý báo cáo, hình
```

## Cài đặt
Python 3.12.
```bash
git clone https://github.com/Dooty26/vietnamese-spell-correction.git
cd vietnamese-spell-correction
pip install -r requirements.txt
pip install -e .
```
Mô hình A dùng LM n-gram tự cài (không cần KenLM), chạy được trên Windows/Linux bằng CPU. Mô hình C cần GPU (Kaggle/Colab): xem `notebooks/04_bartpho.ipynb`.

## Chạy lại từ đầu
```bash
python scripts/prepare_data.py --config configs/data.yaml          # thêm --scale 0.01 để chạy thử nhanh
python scripts/gen_noise.py --input data/processed/clean_syn_train.txt --output data/processed/syn_train.jsonl
python scripts/gen_noise.py --input data/processed/clean_syn_val.txt   --output data/processed/syn_val.jsonl  --seed 43
python scripts/gen_noise.py --input data/processed/clean_syn_test.txt  --output data/processed/syn_test.jsonl --seed 44
python scripts/train_ngram.py --config configs/ngram.yaml
python scripts/train_bartpho.py --config configs/bartpho.yaml
python scripts/predict.py --model ngram   --data data/processed/vsec_test.jsonl
python scripts/predict.py --model bartpho --checkpoint outputs/checkpoints/bartpho/best --data data/processed/vsec_test.jsonl
python scripts/evaluate.py --pred-dir outputs/predictions --out outputs/tables/main_results.csv
python demo/app.py
```

## Ablation mật độ nhiễu
Ba tập train 50k cặp từ cùng 50k câu đầu của `clean_syn_train.txt`, cùng seed 42, chỉ khác `--error-rate` (5% / 12% / 25% âm tiết hợp lệ bị lỗi). Thống kê: `outputs/tables/noise_density_stats.csv`.
```bash
python scripts/gen_noise.py --input data/processed/clean_syn_train.txt --limit 50000 --error-rate 0.05 --seed 42 --output data/processed/syn_train_d05.jsonl
python scripts/gen_noise.py --input data/processed/clean_syn_train.txt --limit 50000 --error-rate 0.12 --seed 42 --output data/processed/syn_train_d12.jsonl
python scripts/gen_noise.py --input data/processed/clean_syn_train.txt --limit 50000 --error-rate 0.25 --seed 42 --output data/processed/syn_train_d25.jsonl
```
Lưu ý: `errors_per_word` trong `.stats.json` thấp hơn tham số (~0,83×) vì tham số chỉ áp lên âm tiết ứng viên (bỏ số, dấu câu, tên riêng viết hoa) và 15% câu được giữ sạch; trên âm tiết ứng viên của các câu bị nhiễu, tỉ lệ thực tế là 5,2% / 12,1% / 25,1%.

## Demo
Ứng dụng Gradio trong `demo/app.py` (cần `pip install gradio`). Chạy từ thư mục gốc dự án:
```bash
python -X utf8 demo/app.py                    # A + C: tự nạp BARTpho nếu có outputs/checkpoints/bartpho/best
python -X utf8 demo/app.py --no-bartpho       # chỉ mô hình A (n-gram), khởi động nhanh
```
Mở http://127.0.0.1:7860. Các tham số:
- `--ngram-dir`: mặc định `outputs/checkpoints/ngram/o3_kneser_ney`, thư mục chứa `lm.pkl`, `lexicon.json`, `config.json`;
- `--bartpho-checkpoint`: mặc định `outputs/checkpoints/bartpho/best`;
- `--port`, `--host`, `--share`.

**Checkpoint BARTpho** (khoảng 1,6 GB, không nằm trong git) nằm ở *Output* của notebook Kaggle, trong `repo/outputs/checkpoints/bartpho/best/`. Chỉ cần tải `model.safetensors` về thư mục trên. Các file tokenizer/config giống hệt `vinai/bartpho-syllable` và có thể chép từ bất kỳ checkpoint nào do `scripts/train_bartpho.py` tạo ra.

**Chạy được trên CPU:**
- nạp mô hình khoảng 15 giây;
- mỗi câu khoảng 0,3–1 giây với beam 4;
- có GPU CUDA thì tự chạy fp16.

Chức năng:
- nhập một câu hoặc cả đoạn văn (tự tách câu);
- chọn mô hình: n-gram, BARTpho, hoặc cả hai đặt cạnh nhau;
- tô các âm tiết bị sửa theo dạng `sai → đúng`, kèm câu đã sửa;
- với mô hình A, bảng ứng viên cho từng âm tiết bị sửa: nguồn ứng viên (telex, skeleton, confusion, teencode, edit1/2), điểm kênh, điểm LM cục bộ và tổng (log10);
- thời gian xử lý của từng mô hình;
- 10 câu ví dụ có sẵn:
  - 8 câu theo từng loại lỗi: Telex chưa chuyển, lỗi vùng miền (dấu hỏi/ngã, l/n), trượt phím, teencode, mất dấu, câu có tên riêng (không bị sửa), đoạn văn hai câu;
  - 2 câu so sánh A và C: sai dấu thanh `đòng góp` (A sửa, C bỏ qua) và sai nhiều ký tự `biện phpasp` (C sửa, A không có ứng viên).

## Đánh giá
Detection P/R/F1 và Correction P/R/F1 mức âm tiết, sentence accuracy, tỉ lệ sửa quá tay. Chạy trên Syn-test, VSEC-test, Viwiki-test. Mọi kết quả cố định seed 42.

## Thành viên
| Thành viên | Phụ trách |
|---|---|
| TV1 | Dữ liệu, sinh nhiễu, EDA |
| TV2 | Baseline n-gram, script đánh giá |
| TV3 | BARTpho, demo |
