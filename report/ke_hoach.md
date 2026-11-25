# Kế hoạch BTL NLP: Phát hiện & sửa lỗi chính tả tiếng Việt (thống kê + học sâu)

## Context
Bài tập lớn môn NLP (yêu cầu trong `Slide_YC/Huong_dan_bai_tap_lon_Xu_ly_ngon_ngu_tu_nhien.docx`): pipeline dữ liệu → tiền xử lý → mô hình → đánh giá → phân tích, ≥2 phương án (baseline + cải tiến), ≥1 ablation, bảng so sánh, 5–10 ví dụ lỗi, báo cáo PDF 20–30 trang, README tái lập, demo 10 phút.
Ràng buộc nhóm: **3 người, 2 tuần, có AI hỗ trợ, GPU Colab Pro/Kaggle**. Phạm vi: **A (n-gram) + C (BARTpho)** trước; **B (BiLSTM)** chỉ làm nếu dư thời gian. Dữ liệu train **tự sinh từ Binhvq**. Code: **package `src/` + notebook mỏng**, đẩy lên GitHub.
Thư mục dự án `D:\Do\NLP\Project_NLP` hiện chỉ có `Slide_YC/` → dựng mới hoàn toàn.

## Dữ liệu (chốt)
| Dataset | Vai trò | Chia |
|---|---|---|
| Binhvq (`vietgpt/binhvq_news_vi`, streaming) | Corpus sạch: từ điển âm tiết, LM n-gram, nguồn sinh nhiễu | Lấy mẫu ~600k câu, **chia theo bài báo**: LM-corpus ~400k câu · Syn-train ~150k cặp · Syn-val 5k · Syn-test 5k |
| VSEC (`nguyenthanhasia/vsec-vietnamese-spell-correction` hoặc GitHub gốc) | Lỗi thật, miền tin tức | Seed 42: VSEC-dev 20% (chỉnh ngưỡng/beam) · **VSEC-test 80%** |
| Viwiki-spelling (GitHub heraclex12, CC BY 4.0) | Lỗi thật, miền Wikipedia | Chỉ test |

- Chống leakage: loại khỏi Binhvq câu trùng (hash sau chuẩn hóa) / gần trùng (n-gram overlap) với VSEC (`text`, `corrected_text`) và Viwiki.
- Sinh nhiễu: chỉ lỗi trong âm tiết (giữ 1-1 âm tiết); tỉ lệ loại lỗi theo arXiv 2405.02573 (vùng miền 31%, Telex 30%, trượt phím 30%, teencode/viết tắt/edit-distance 3% mỗi loại); mật độ mặc định 5% âm tiết (gần VSEC); 15% câu giữ sạch.
- Chuẩn hóa: NFC + quy về một kiểu đặt dấu (`hòa`), áp dụng thống nhất cho mọi tập và mọi đầu ra mô hình trước khi chấm.

## Mô hình
- **A – Baseline thống kê**: từ điển âm tiết (từ LM-corpus) → phát hiện (âm tiết ngoài từ điển hoặc log-prob thấp hơn ngưỡng) → sinh ứng viên (tập dễ nhầm + edit distance ≤2 trong từ điển) → chọn bằng KenLM trigram Kneser-Ney + beam search.
- **C – Fine-tune `vinai/bartpho-syllable`**: seq2seq câu lỗi→câu đúng, max_len 96, fp16, 2–3 epoch trên 150k cặp, chọn checkpoint theo Syn-val.
- **B (tùy chọn)** – BiLSTM gắn nhãn chuỗi với char-CNN, đầu ra = âm tiết đúng.

## Đánh giá (một script chung cho mọi mô hình)
Căn chỉnh âm tiết đầu vào/đầu ra/gold (difflib) → Detection P/R/F1, Correction P/R/F1 (mức âm tiết), sentence accuracy, tỉ lệ sửa quá tay trên câu sạch. Bảng chính: {A, C, (B)} × {Syn-test, VSEC-test, Viwiki-test}.
Ablation: (1) A: bậc n-gram 2/3/4 + Laplace vs KN; (2) C: mật độ nhiễu train 5% / 12% / 25% (train nhỏ 50k để kịp giờ).

## Cấu trúc thư mục (sẽ dựng sau khi duyệt)
```
Project_NLP/
├── README.md                 # mục tiêu, dữ liệu, cài đặt, các bước chạy lại
├── requirements.txt          # phiên bản cố định
├── configs/
│   ├── data.yaml             # kích thước mẫu, tỉ lệ chia, seed
│   ├── noise.yaml            # tỉ lệ loại lỗi, mật độ
│   ├── ngram.yaml
│   └── bartpho.yaml
├── src/vspell/
│   ├── __init__.py
│   ├── normalize.py          # NFC, kiểu đặt dấu, tách âm tiết/dấu câu
│   ├── data/
│   │   ├── binhvq.py         # streaming, làm sạch, tách câu, chia theo bài
│   │   ├── vsec.py           # đọc VSEC → (src, tgt, nhãn từng âm tiết)
│   │   ├── viwiki.py         # đọc Viwiki → cùng định dạng
│   │   └── dedup.py          # loại trùng/gần trùng với tập test
│   ├── noise/
│   │   ├── rules.py          # Telex, vùng miền, bàn phím, teencode, viết tắt
│   │   └── generator.py      # sinh cặp lỗi→đúng theo config
│   ├── models/
│   │   ├── ngram_corrector.py  # mô hình A
│   │   ├── bartpho.py          # train/infer mô hình C
│   │   └── bilstm.py           # mô hình B (tùy chọn)
│   ├── eval/
│   │   ├── align.py          # căn chỉnh âm tiết
│   │   └── metrics.py        # P/R/F1 phát hiện & sửa, over-correction
│   └── utils.py              # seed, logging, IO jsonl
├── scripts/                  # CLI chạy từng bước (prepare_data, gen_noise, train_ngram, train_bartpho, predict, evaluate)
├── notebooks/                # notebook mỏng gọi src/ (01_eda, 02_data, 03_ngram, 04_bartpho, 05_eval_analysis)
├── data/                     # raw/, interim/, processed/ (gitignore, có script tải)
├── outputs/                  # predictions/, tables/, figures/, checkpoints/ (gitignore checkpoint)
├── demo/app.py               # Gradio: nhập câu → tô lỗi + gợi ý
├── tests/                    # test nhỏ cho normalize, noise, metrics
└── report/                   # dàn ý báo cáo, hình
```

## Timeline 14 ngày & phân công
TV1 = Dữ liệu & sinh nhiễu · TV2 = Baseline A & đánh giá · TV3 = BARTpho C & demo

| Ngày | TV1 | TV2 | TV3 |
|---|---|---|---|
| 1–2 | Tải/làm sạch Binhvq, tách bài/câu | `normalize.py`, đọc VSEC/Viwiki về định dạng chung | Dựng repo, requirements, thử load BARTpho trên Kaggle |
| 3–4 | `noise/` + dedup + chia tập; EDA | `align.py`, `metrics.py` + test (chạy thử với "mô hình không sửa gì") | Script train C, chạy thử 10k cặp |
| 5–7 | Sinh dữ liệu ablation (5/12/25%) | KenLM + mô hình A, chỉnh trên VSEC-dev | Train C đầy đủ (150k), chọn checkpoint |
| 8–9 | Bảng EDA, hình cho báo cáo | Ablation A (bậc n-gram, smoothing) | Ablation C (mật độ nhiễu), dự đoán 3 tập test |
| 10–11 | Phân tích lỗi theo loại lỗi (5–10 ví dụ) | Bảng so sánh tổng hợp | Demo Gradio; (B nếu dư thời gian) |
| 12–13 | Viết báo cáo: Dữ liệu | Viết: Phương pháp, Thực nghiệm | Viết: Kết quả, Kết luận; README |
| 14 | Rà soát, chạy lại từ đầu theo README, slide 10 phút | ← | ← |

Mốc kiểm tra: **cuối ngày 4** có dữ liệu + script đánh giá chạy được; **cuối ngày 9** đủ bảng kết quả chính (đây là bản tối thiểu nộp được).

## Việc làm ngay sau khi duyệt
Dựng cấu trúc thư mục ở trên trong `D:\Do\NLP\Project_NLP`: các thư mục, `README.md` (khung), `requirements.txt`, `.gitignore`, các file `configs/*.yaml` với giá trị mặc định, và file `.py` khung (docstring + chữ ký hàm, chưa cài logic).

## Kiểm tra
- `python -c "import vspell"` chạy được sau `pip install -e .` (thêm `pyproject.toml`).
- Cây thư mục khớp thiết kế; `.gitignore` loại `data/` và `outputs/checkpoints/`.
- Khi cài logic sau này: `pytest tests/` cho normalize/noise/metrics; chạy "mô hình đồng nhất" (không sửa) phải cho Correction Recall = 0 và over-correction = 0.
