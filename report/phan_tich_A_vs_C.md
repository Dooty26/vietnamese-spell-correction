# Phân tích A (n-gram) và C (BARTpho)

*Bản nháp cho mục 5 của báo cáo (S5b). Số liệu được sinh bởi `PYTHONPATH=src python -X utf8 scripts/analyze_errors.py --models ngram bartpho --steps errors,results`.*

- **Bảng:** `outputs/tables/main_results.csv`, `model_comparison.csv`, `ab_recall_by_kind.csv`, `ablation_noise_density.csv`, `real_error_kinds.csv`.
- **Hình:** `outputs/figures/model_comparison_test_sets.png`, `ab_recall_word_kind.png`, `ablation_noise_density.png`, `recall_by_error_type.png`, `error_breakdown.png`.
- **Ví dụ:** `outputs/tables/error_examples.md`, các ví dụ 15–23.

## 1. Kết quả chung

| Mô hình | Syn-test cor F1 | VSEC-test cor F1 | Viwiki-test cor F1 | Sửa quá tay (Syn / VSEC / Viwiki) |
|---|---|---|---|---|
| A: n-gram 3-gram KN | 0,898 | **0,778** | **0,304** | 0,127 / 0,063 / 0,128 |
| C: BARTpho v2 (lỗi `<unk>`) | 0,919 | 0,490 | 0,159 | 0,159 / 0,523 / 0,139 |
| C: BARTpho v3 | **0,959** | 0,536 | 0,240 | **0,020 / 0,026 / 0,043** |

- **Trên dữ liệu tổng hợp, C thắng rõ.** Correction F1 là 0,959 so với 0,898, và C gần như không sửa quá tay (0,020).
- **Trên lỗi thật, A thắng.** Trên VSEC: 0,778 so với 0,536. Trên Viwiki: 0,304 so với 0,240.
- **C có hồ sơ "thận trọng".** Precision trên VSEC là 0,888, cao hơn A (0,819). Nhưng recall chỉ 0,384, so với 0,741 của A.
- **Đối chiếu từng vị trí lỗi thật trên VSEC** (`model_comparison.csv`):

  | Nhóm | Số vị trí |
  |---|---|
  | Cả hai sửa đúng | 2.881 |
  | Chỉ A đúng | 3.531 |
  | Chỉ C đúng | 442 |
  | Cả hai sai | 1.797 |

  Gần như mọi lỗi C sửa được thì A cũng sửa được. Ngược lại thì không.

**Bài học từ v2 sang v3.** Vocab của BARTpho chỉ có kiểu dấu mới (hoà, thoả, khoẻ, tuỳ, uỷ). Các âm tiết viết theo kiểu dấu cũ (hòa, thỏa…) bị tokenizer biến thành `<unk>`. Mô hình học cách xóa chúng, nên tỉ lệ sửa quá tay trên VSEC lên tới 0,52. Đổi sang kiểu dấu mới trước khi tokenize (`to_bartpho_style`) đã đưa tỉ lệ này về 0,026. Đây là lỗi tiền xử lý, không phải giới hạn của mô hình. Nó cho thấy chuẩn hóa Unicode và vị trí dấu phải khớp với vocab của mô hình tiền huấn luyện.

## 2. Vì sao C thua trên lỗi thật: lệch phân bố loại lỗi

Phân bố kiểu lỗi được suy từ cặp (âm tiết sai, âm tiết đúng), lấy từ `real_error_kinds.csv`:

| Kiểu lỗi | Syn-train | VSEC-test | Viwiki-test |
|---|---|---|---|
| Chỉ sai dấu thanh | 6,6% | **49,6%** | **34,1%** |
| Sai dấu phụ (ă â ê ô ơ ư đ) | 0,4% | 12,0% | 15,7% |
| Sai 1 chữ cái | **83,8%** | 25,6% | 34,7% |
| Sai ≥2 ký tự | 9,3% | 4,4% | 7,5% |
| Tách/ghép âm tiết | 0,0% | 8,4% | 7,8% |

Bộ sinh nhiễu (tỉ lệ loại lỗi theo arXiv 2405.02573) chủ yếu tạo lỗi thay hoặc thêm một chữ cái. Lỗi thật thì một nửa là sai dấu thanh. Recall sửa lỗi theo kiểu lỗi trên VSEC (`ab_recall_by_kind.csv`):

| Kiểu lỗi (VSEC) | n | A | C |
|---|---|---|---|
| Chỉ sai dấu thanh | 4.288 | **0,876** | 0,330 |
| Sai dấu phụ | 1.038 | **0,897** | 0,491 |
| Sai 1 chữ cái | 2.203 | **0,745** | 0,542 |
| Sai ≥2 ký tự | 363 | 0,234 | **0,444** |
| Tách/ghép âm tiết | 418 | 0,002 | **0,089** |

**Nhận xét:**

- **A không phụ thuộc vào phân bố nhiễu khi huấn luyện.** Tập ứng viên của A luôn chứa mọi biến thể dấu thanh và dấu phụ của âm tiết. Việc chọn ứng viên do mô hình ngôn ngữ quyết định, mà mô hình này được học trên văn bản sạch.
- **C chỉ học những phép sửa đã thấy nhiều khi huấn luyện.** Lỗi dấu thanh chỉ chiếm 6,6% dữ liệu train, nên C hiếm khi dám đổi dấu. Kết quả là recall thấp và precision cao.
- **C mạnh hơn ở lỗi khó.** Đó là lỗi sai từ 2 ký tự trở lên (ví dụ `phpasp` → `pháp`) và một phần lỗi tách/ghép âm tiết. A không làm được các lỗi này vì giới hạn khoảng cách sửa ≤ 2 và vì A giữ ánh xạ một âm tiết sang một âm tiết.

## 3. Lỗi tạo ra từ có thật

Ta gọi một lỗi là **"tạo ra từ có thật"** khi âm tiết sai vẫn nằm trong từ điển âm tiết, ví dụ giảm→giam, tác giả→tác gia. Những lỗi này chỉ phát hiện được nhờ ngữ cảnh. Ngược lại, lỗi "tạo ra từ không có thật" là khi âm tiết sai nằm ngoài từ điển (hình `ab_recall_word_kind.png`).

| | VSEC: không có thật | VSEC: có thật | Viwiki: không có thật | Viwiki: có thật |
|---|---|---|---|---|
| n | 1.677 | **6.638** | 357 | **1.159** |
| A | 0,751 | **0,776** | 0,389 | **0,528** |
| C | 0,678 | 0,329 | 0,331 | 0,249 |

- **Đa số lỗi thật thuộc loại tạo ra từ có thật:** 80% trên VSEC và 76% trên Viwiki.
- **Với lỗi tạo ra từ không có thật,** C chỉ thua A khoảng 7 điểm.
- **Với lỗi tạo ra từ có thật,** C chỉ sửa được 33%. Mô hình mặc định giữ nguyên một âm tiết hợp lệ, trừ khi ngữ cảnh rất rõ.

Hai nhận xét này trùng nhau: phần lớn lỗi tạo ra từ có thật chính là lỗi sai dấu thanh.

## 4. Ablation C: mật độ nhiễu và quy mô huấn luyện

Kết quả trên VSEC-test (`ablation_noise_density.csv`, hình `ablation_noise_density.png`):

| Cấu hình | P | R | F1 | Sửa quá tay |
|---|---|---|---|---|
| 5% nhiễu · 50k cặp · 1 epoch | 0,884 | 0,422 | 0,572 | 0,024 |
| 12% · 50k · 1 epoch | 0,882 | **0,480** | 0,622 | 0,028 |
| 25% · 50k · 1 epoch | 0,895 | 0,479 | **0,624** | 0,045 |
| 12% · 150k · 3 epoch (mô hình chính) | 0,888 | 0,384 | 0,536 | 0,026 |

- **Mật độ nhiễu:** tăng từ 5% lên 12% làm recall tăng 6 điểm, còn precision gần như không đổi. Lên 25% thì F1 không tăng thêm, trong khi tỉ lệ sửa quá tay gần gấp đôi. Mức 12% là điểm cân bằng hợp lý.
- **Quy mô huấn luyện:** mô hình chính được train nhiều hơn khoảng 9 lần (150k cặp × 3 epoch). Nó tốt hơn trên Syn-test nhưng **kém hơn** trên VSEC (0,536 so với 0,622). Train càng lâu trên dữ liệu tổng hợp, mô hình càng khớp với phân bố nhiễu tự sinh và càng ít dám sửa những lỗi không giống dữ liệu train. Vì vậy, chọn checkpoint theo Syn-val không phải là tiêu chí tốt khi mục tiêu là lỗi thật.
- **Hướng cải thiện (chưa làm):**
  - tăng tỉ lệ lỗi dấu thanh và dấu phụ khi sinh nhiễu cho khớp với VSEC;
  - chọn checkpoint hoặc dừng sớm theo VSEC-dev;
  - trộn một phần dữ liệu lỗi thật vào tập train.

## 5. Ví dụ tiêu biểu

Các ví dụ được đánh số theo `outputs/tables/error_examples.md`. In **đậm** là vị trí lỗi hoặc vị trí bị sửa.

**Ví dụ 15. A đúng, C sai: lỗi tạo ra từ có thật (Viwiki).**

- src: Không những Voltaire, tác **gia** Thụy Điển là Frans G.
- A: tác **giả**. C: giữ nguyên `tác gia`.
- Nhận xét: "gia" là âm tiết hợp lệ. A nhận ra nhờ trigram "tác giả Thụy" có xác suất cao hơn hẳn. C coi "tác gia" là cụm hợp lệ (nó cũng là một từ Hán Việt hiếm) nên không sửa.

**Ví dụ 16. A đúng, C sai: chỉ sai dấu thanh (VSEC).**

- src: công sức **đòng** góp của vợ, chồng
- A: **đóng** góp. C: giữ nguyên `đòng`.
- Nhận xét: đây là kiểu lỗi chiếm một nửa VSEC. A thử mọi dấu thanh của "đong" và chọn theo LM. C hiếm khi thấy kiểu sửa này khi train.

**Ví dụ 17. A đúng, C sai: sai dấu thanh, ngay cả trên Syn-test.**

- src: **Buỗi** chiều, các đại biểu sẽ thảo luận theo tổ.
- A: **Buổi**. C: giữ nguyên `Buỗi`.
- Nhận xét: "Buỗi" không phải âm tiết hợp lệ. Dù vậy C vẫn bỏ qua, vì lỗi sai dấu thanh chỉ chiếm 6,6% dữ liệu train.

**Ví dụ 18. C đúng, A sai: tên riêng bị lỗi gõ (Syn).**

- src: ông Đinh Văn **Fhu** **lpà** Đinh Văn Vũ
- A: chỉ sửa `lpà` → `là`, giữ nguyên `Fhu`. C: sửa cả hai, `Fhu` → **Thu** và `lpà` → **là**.
- Nhận xét: A tránh sửa từ viết hoa giữa câu (coi là tên riêng). C dùng ngữ cảnh "Đinh Văn ___" để đoán đúng tên.

**Ví dụ 19. C đúng, A sai: sai nhiều ký tự (VSEC).**

- src: thực hiện biện **phpasp** này
- A: giữ nguyên. C: **pháp**.
- Nhận xét: khoảng cách sửa từ "phpasp" đến "pháp" lớn hơn 2 (lỗi Telex gõ lặp), nên A không có ứng viên đúng. C sinh lại cả chuỗi nên không bị giới hạn này.

**Ví dụ 20. C đúng, A sai: lỗi cần hiểu cụm từ (Viwiki).**

- src: và **cơ** **ngợi** những chiến thắng
- A: sửa `ngợi` → `ngơi`, thành "cơ ngơi", đúng ngữ pháp nhưng sai nghĩa. C: **ca** ngợi.
- Nhận xét: gold chỉ sửa "cơ" → "ca". A chọn sửa sai âm tiết vì "cơ ngơi" là một cụm phổ biến trong LM.

**Ví dụ 21. Cả hai sai: hai lỗi liền kề (VSEC).**

- src: vay vốn tín **chất** để **phụ** vụ cho hoạt động nuôi trồng
- A: sửa được `chất` → **chấp** nhưng bỏ sót `phụ` → `phục`. C: không sửa gì.
- Nhận xét: cả hai lỗi đều tạo ra từ có thật. Với "phụ" → "phục", hình phạt khi sửa một âm tiết hợp lệ của A lớn hơn mức chênh điểm LM. C thì bỏ qua cả hai, đúng với hồ sơ "thận trọng" đã nêu ở mục 1.

**Ví dụ 22. Cả hai sai: viết tắt hoặc tên câu lạc bộ (Viwiki).**

- src: **Chel** 85 điểm. Gold: chelsea.
- Nhận xét: đây là lỗi "tách/ghép" và viết tắt, nằm ngoài phạm vi thiết kế của cả hai mô hình (giữ một âm tiết sang một âm tiết, không có tên riêng nước ngoài trong từ điển). Ví dụ này cũng cho thấy nhãn của Viwiki có độ nhiễu nhất định.

**Ví dụ 23. Cả hai sai: lỗi cần kiến thức thế giới (Syn).**

- src: hoa đào, hoa **hai**, bánh chưng… Gold: hoa **mai**.
- A: sửa thành "hồi" (hoa hồi, đúng ngữ pháp). C: giữ nguyên "hai".
- Nhận xét: cả hai đều sửa đúng `khoong` → `không` ở đầu câu. Nhưng muốn sửa "hai" → "mai" thì cần biết cặp "hoa đào, hoa mai" ngày Tết, điều mà trigram và mô hình đã fine-tune trên tin tức đều không nắm được.

## 6. Kết luận cho mục 5

1. **Trên dữ liệu cùng phân bố với lúc huấn luyện, C (BARTpho) tốt hơn A** cả về F1 lẫn độ an toàn (ít sửa quá tay).
2. **Trên lỗi thật, A tốt hơn.** Nguyên nhân chính là lệch phân bố loại lỗi giữa nhiễu tổng hợp và lỗi thật: lỗi sai dấu thanh và lỗi tạo ra từ có thật chiếm đa số trong lỗi thật nhưng rất ít trong dữ liệu train. Bản thân kiến trúc không phải nguyên nhân chính.
3. **Hai mô hình bổ sung cho nhau.** C mạnh ở lỗi sai nhiều ký tự và lỗi cần ngữ cảnh rộng, A mạnh ở lỗi dấu. Kết hợp hai mô hình (ví dụ dùng A tạo ứng viên rồi C xếp hạng, hoặc ensemble) là hướng phát triển tự nhiên.
4. **Kết quả ablation C:** tăng mật độ nhiễu lên 12% có lợi. Train lâu hơn trên dữ liệu tổng hợp lại có hại khi chuyển sang lỗi thật.
