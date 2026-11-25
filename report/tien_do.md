# Tiến độ & danh sách task theo phiên làm việc

*Cập nhật: 25/11/2026. Dự án bắt đầu 12/11; từ 12/11 đến nay chưa có thay đổi mới trong thư mục dự án.*

## 1. Đã xong

| Hạng mục | Kết quả | File chính |
|---|---|---|
| Chuẩn hóa | NFC, quy kiểu đặt dấu, tách âm tiết, kiểm tra âm tiết hợp lệ | `src/vspell/normalize.py` |
| Đánh giá | Căn chỉnh âm tiết; Det/Cor P/R/F1, sent acc, sửa quá tay, recall theo loại lỗi | `src/vspell/eval/`, `scripts/evaluate.py` |
| Dữ liệu thật | VSEC dev 1.868 / test 7.473 câu; Viwiki test 15.432 câu | `data/processed/vsec_*.jsonl`, `viwiki_test.jsonl` |
| Binhvq | 571k câu sạch, đã loại trùng/rò rỉ: LM 400k, train 150k, val 5k, test 5k | `data/processed/lm_corpus.txt`, `clean_syn_*.txt` |
| Sinh nhiễu | 6 loại lỗi, mật độ 5% âm tiết, 15% câu sạch | `src/vspell/noise/`, `syn_{train,val,test}.jsonl` |
| **Mô hình A** | Tự cài n-gram LM + noisy channel + beam search; đã chỉnh tham số trên VSEC-dev | `src/vspell/models/ngram_*.py` |
| Kết quả A | Cor F1: Syn 0,90 · VSEC 0,78 · Viwiki 0,30 | `outputs/tables/main_results.csv` |
| Ablation A | Bậc 2/3/4 × Laplace/KN; perplexity | `outputs/tables/ngram_perplexity.csv` |
| Test | 65 test pass | `tests/` |

**Phát hiện mới (S5a):** sửa quá tay trên Viwiki chủ yếu do lỗi thật quá ít (0,29%): tỉ lệ sửa quá tay trên token sạch 0,48% (VSEC 0,33%), nhưng bằng 1,64× số lỗi thật. 45% là từ thường, 35% ngoài từ điển (manga→mang), 21% tên riêng. Nghi chuẩn hóa chưa gộp biến thể i/y (`kỉ→kỷ` 15 lần), cần kiểm tra.

## 2. Chưa xong (so với yêu cầu bài tập lớn)

- [x] **Mô hình C (BARTpho)**: v3 xong 26/11 (Kaggle 2×T4, 5,6 giờ). VSEC cor_f1 0,536, over-correction 0,026; Syn-test cor_f1 0,959; Viwiki cor_f1 0,240.
- [x] Ablation C (mật độ nhiễu 5/12/25%, 50k cặp × 1 epoch, Colab): VSEC cor_f1 0,572 / 0,622 / 0,624. Cả 3 đều cao hơn mô hình đầy đủ (150k × 3 epoch, 0,536) → train lâu trên dữ liệu tổng hợp làm giảm khả năng chuyển sang lỗi thật (recall 0,38 so với 0,48).
- [ ] Bảng so sánh A vs C
- [~] Ví dụ lỗi và hình cho A đã có (chọn bản cuối A vs C: S5b)
- [x] Demo Gradio với A (còn gắn BARTpho: S6b)
- [ ] Báo cáo PDF 20–30 trang (mới có dàn ý)
- [ ] README tái lập đầy đủ; ghim phiên bản `requirements.txt` (bỏ dòng `kenlm`, vì không còn dùng)
- [~] Git: đã có repo cục bộ, **chưa push GitHub** (cần trước S2 vì notebook clone repo)
- [ ] Slide và demo 10 phút
- [ ] (Tùy chọn) Mô hình B (BiLSTM); giảm sửa quá tay của A trên Viwiki

> **Cảnh báo thời gian:** theo kế hoạch 14 ngày tính từ 12/11, hạn chót rơi vào khoảng 26/11. Nếu hạn thật gần như vậy, chỉ làm **S0 → S1 → S2 → S5 → S6 → S7**. Bỏ S3 (ablation C) vì ablation A đã đủ yêu cầu "≥1 ablation". Bỏ cả S4 và S8.

## 3. Danh sách task (mỗi task = một phiên làm việc)

Ký hiệu: 🤖 Claude làm được trên máy này · 🧑 cần người chạy (GPU Kaggle/Colab, nộp bài) · ⏱ ước lượng.

### Đợt 1: chạy song song được (không đụng file của nhau)

**S0. Git + môi trường** ✅ *(25/11: đã git init, commit đầu, ghim requirements; còn push GitHub, chờ tên repo)* 🤖 ⏱ 30 phút. *Làm trước tiên, để các phiên sau có worktree riêng.*
- Việc: `git init`; rà `.gitignore` (không commit `data/`, `outputs/checkpoints/`, `outputs/predictions/`); ghim `requirements.txt` theo `pip freeze` và bỏ dòng `kenlm`; commit đầu tiên; tạo repo GitHub và push (🧑 xác nhận tên repo).
- Xong khi: `git status` sạch; `pip install -e . && pytest` pass trên môi trường mới.

**S1. Cài mô hình C (BARTpho), phần code** ✅ *(chạy thử CPU OK; chọn checkpoint theo eval_loss)* 🤖 ⏱ 1 phiên
- File: `src/vspell/models/bartpho.py`, `scripts/train_bartpho.py`, `configs/bartpho.yaml`, `notebooks/04_bartpho.ipynb`, `tests/test_bartpho.py`.
- Việc:
  - `build_datasets` (tokenize src→input_ids, tgt→labels, max_len 128);
  - `train` bằng `Seq2SeqTrainer` (fp16, lưu checkpoint tốt nhất theo cor_f1 hoặc loss trên Syn-val, `--limit` để chạy thử);
  - `BartphoCorrector.correct` (sinh theo batch, beam 4, hậu xử lý về đúng số âm tiết hoặc chuẩn hóa);
  - notebook Kaggle: clone repo → cài → train → predict 3 tập test → nén prediction để tải về.
- Xong khi: chạy thử trên CPU với 200 cặp và vài bước train ra checkpoint, `predict.py --model bartpho --limit 20` chạy được, test pass.

**S3. Dữ liệu ablation mật độ nhiễu** ✅ *(syn_train_d05/d12/d25 trong data/processed; tỉ lệ thực 5,2/12,1/25,1%)* 🤖 ⏱ 30 phút *(tùy chọn, xem cảnh báo)*
- File: `data/processed/syn_train_d{05,12,25}.jsonl` (50k câu mỗi file), ghi chú cách sinh vào README.
- Việc: `gen_noise.py --input clean_syn_train.txt --limit 50000 --error-rate {0.05,0.12,0.25}` với cùng seed.
- Xong khi: có 3 file và `.stats.json` đi kèm, mật độ lỗi thực tế khớp tham số.

**S5a. Phân tích lỗi và hình, phần mô hình A** ✅ *(8 hình, error_examples.md, overcorrection_viwiki.csv)* 🤖 ⏱ 1 phiên
- File: `scripts/analyze_errors.py` (mới), `notebooks/01_eda.ipynb`, `notebooks/05_eval_analysis.ipynb`, `outputs/figures/`, `outputs/tables/`.
- Việc:
  - **Hình EDA:** phân bố độ dài câu, mật độ lỗi của 3 tập, phân bố loại lỗi.
  - **Hình kết quả:** biểu đồ ablation (bậc × smoothing; perplexity so với F1), recall theo loại lỗi.
  - **Phân loại tự động ca sai của A:** sửa sai, bỏ sót, sửa quá tay, tên riêng/từ nước ngoài; trích 10–15 ví dụ ứng viên kèm src/pred/tgt.
  - **Script dùng lại được:** khi có `bartpho__*.jsonl`, chạy lại là ra bảng A vs C.
- Xong khi: các file hình trong `outputs/figures/` và `outputs/tables/error_examples.md`.

**S6a. Demo Gradio, phần mô hình A** ✅ 🤖 ⏱ 1 phiên
- File: `demo/app.py`, `README.md` (mục demo).
- Việc: ô nhập câu → chọn mô hình → `gr.HighlightedText` tô âm tiết bị sửa (dùng `eval.align.edit_positions`) → bảng ứng viên. Nạp BARTpho nếu có `--bartpho-checkpoint`. Thêm vài câu ví dụ có sẵn.
- Xong khi: `python demo/app.py` mở được trên localhost và sửa đúng câu ví dụ.

**S7a. Báo cáo, các phần không phụ thuộc C** ✅ *(report/bao_cao.md; chưa xuất PDF: cần pandoc)* 🤖 ⏱ 1–2 phiên
- File: `report/bao_cao.md` (sau xuất PDF), `report/figures/`.
- Việc: viết mục 1 Giới thiệu, 2 Dữ liệu (lấy số liệu từ `stats_*.json`), 3 Phương pháp (A đầy đủ, C theo thiết kế), 4 Thực nghiệm. Kết quả A và ablation A đặt vào mục 5. Thêm phụ lục sử dụng AI.
- Ghi rõ các hạn chế:
  - Binhvq được chia theo đoạn văn, không theo bài báo (dữ liệu không có id bài);
  - khác biệt kiểu đặt dấu không được tính là lỗi;
  - không xử lý lỗi tách/ghép âm tiết.
- Xong khi: các mục 1–4 hoàn chỉnh, khoảng 12–15 trang.

### Đợt 2: phụ thuộc S1

**S2. Huấn luyện C trên GPU** ✅ *(v1 hết đĩa → `save_only_model`; v2 lỗi `<unk>` với kiểu dấu cũ (tag `bartpho_v1unk`, VSEC over-correction 0,52) → sửa `to_bartpho_style`; v3: Colab bị ngắt ở bước 3325, bản dự phòng Kaggle 2×T4 chạy xong 26/11 (20.1k giây). Ablation C chạy song song trên Colab)* 🧑 (+🤖 hỗ trợ đọc log) ⏱ khoảng 3–5 giờ GPU
- Chạy `notebooks/04_bartpho.ipynb` trên Kaggle/Colab với 150k cặp, 3 epoch.
- Tải về `outputs/predictions/bartpho__{syn_test,vsec_test,viwiki_test}.jsonl`. Checkpoint để trên Kaggle/Drive, không đưa vào git.
- Nếu làm S3: train thêm 3 mô hình 50k, 1 epoch, rồi predict trên VSEC-test, gắn tag `bartpho_d05`, `bartpho_d12`, `bartpho_d25`.
- Xong khi: chạy `scripts/evaluate.py` thấy các dòng `bartpho` trong `main_results.csv`.

### Đợt 3: phụ thuộc S2

**S5b. Phân tích A vs C** 🤖 ⏱ ½ phiên
- Chạy lại `analyze_errors.py` có thêm C.
- Chọn 5–10 ví dụ cuối cùng: A đúng C sai, C đúng A sai, cả hai sai. Kèm phân tích từng ví dụ.

**S6b. Gắn BARTpho vào demo** 🤖 hoặc 🧑 ⏱ ½ phiên. Nếu máy không có GPU, chạy demo trên Kaggle/Colab và dùng `share=True`.

**S7b. Hoàn thiện báo cáo** 🤖 + 🧑 ⏱ 1–2 phiên
- Viết mục 5 Kết quả (bảng chính, các ablation, ví dụ lỗi) và mục 6 Kết luận.
- Hoàn thiện README.
- Xuất PDF và kiểm tra số trang (20–30).

**S9. Chốt nộp bài** 🧑 ⏱ ½ ngày
- Clone repo sạch, chạy lại theo README.
- Làm slide 10 phút, tập demo.
- Hoàn thiện bảng phân công.

### Tùy chọn (chỉ khi dư thời gian)

- **S4. Giảm sửa quá tay của A trên Viwiki** 🤖: ưu tiên giữ nguyên tên riêng, từ nước ngoài và từ viết hoa; thêm ngưỡng tin cậy. **Chỉ chỉnh trên VSEC-dev, không chỉnh trên Viwiki-test.**
- **S8. Mô hình B (BiLSTM với char-CNN, gắn nhãn chuỗi)** 🤖 viết code, 🧑 train GPU.

## 4. Thứ tự và phân công gợi ý

```
S0 ──┬─ S1 ── S2 ──┬─ S5b ─┐
     ├─ S3 ───────┘│       │
     ├─ S5a ───────┴───────┤
     ├─ S6a ──── S6b ──────┼─ S7b ── S9
     └─ S7a ───────────────┘
```

| Thành viên | Task |
|---|---|
| TV1 (dữ liệu) | S3, S5a, S5b, mục 2 của báo cáo |
| TV2 (A và đánh giá) | S0, S7a (mục 1, 3, 4), S4 nếu làm |
| TV3 (C và demo) | S1, S2, S6a, S6b, S7b (mục 5, 6) |

**Khi mở một phiên mới**, dán: *"Làm task S… trong `report/tien_do.md`. Chỉ sửa các file được liệt kê cho task đó; xong thì đánh dấu [x] và ghi kết quả vào mục 1."*
