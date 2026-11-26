# Phân tích A (n-gram) và C (BARTpho)

*Bản nháp cho mục 5 của báo cáo (S5b). Số liệu được sinh bởi `PYTHONPATH=src python -X utf8 scripts/analyze_errors.py --models ngram bartpho bartpho_v4 --steps errors,results`.*

- **Bảng:** `outputs/tables/main_results.csv`, `model_comparison.csv`, `ab_recall_by_kind.csv`, `ablation_noise_density.csv`, `real_error_kinds.csv`.
- **Hình:** `outputs/figures/model_comparison_test_sets.png`, `ab_recall_word_kind.png`, `ablation_noise_density.png`, `recall_by_error_type.png`, `error_breakdown.png`.
- **Ví dụ:** mục 5 (id câu kèm theo); danh sách lấy mẫu đầy đủ trong `outputs/tables/error_examples.md` (A vs C v3 và A vs C v4).

## 1. Kết quả chung

| Mô hình | Syn-test cor F1 | VSEC-test cor F1 | Viwiki-test cor F1 | Sửa quá tay (Syn / VSEC / Viwiki) |
|---|---|---|---|---|
| A: n-gram 3-gram KN | 0,898 | **0,778** | **0,304** | 0,127 / 0,063 / 0,128 |
| C: BARTpho v2 (lỗi `<unk>`) | 0,919 | 0,490 | –¹ | 0,159 / 0,523 / – |
| C: BARTpho v3 | **0,959** | 0,536 | 0,240 | **0,020 / 0,026 / 0,043** |
| C: BARTpho v4 (nhiễu dấu, 50k · 1 epoch) | 0,912 | 0,749 | **0,322** | 0,066 / 0,026 / 0,059 |

¹ *Viwiki-test được tạo lại ngày 26/11 sau khi sửa bộ tách câu (15.153 câu, trước là 15.432). Checkpoint v2 không còn nên không chấm lại được trên tập mới; trên tập cũ v2 đạt 0,159.*

- **Trên dữ liệu tổng hợp, C thắng rõ.** Correction F1 là 0,959 so với 0,898, và C gần như không sửa quá tay (0,020).
- **Trên lỗi thật, A thắng.** Trên VSEC: 0,778 so với 0,536. Trên Viwiki: 0,304 so với 0,240.
- **C có hồ sơ "thận trọng".** Precision trên VSEC là 0,888, cao hơn A (0,819). Nhưng recall chỉ 0,384, so với 0,741 của A.
- **Đối chiếu từng vị trí lỗi thật trên VSEC** (`model_comparison.csv`):

  | Nhóm | Số vị trí |
  |---|---|
  | Cả hai sửa đúng | 2.876 |
  | Chỉ A đúng | 3.529 |
  | Chỉ C đúng | 442 |
  | Cả hai sai | 1.796 |

  Gần như mọi lỗi C sửa được thì A cũng sửa được. Ngược lại thì không.

*Các nhận xét dưới đây so sánh A với C v3 (mô hình sinh nhiễu gốc). v4 là thí nghiệm sửa đúng nguyên nhân tìm ra ở mục 2, trình bày ở mục 4.2.*

**Bài học từ v2 sang v3.** Vocab của BARTpho chỉ có kiểu dấu mới (hoà, thoả, khoẻ, tuỳ, uỷ). Các âm tiết viết theo kiểu dấu cũ (hòa, thỏa…) bị tokenizer biến thành `<unk>`. Mô hình học cách xóa chúng, nên tỉ lệ sửa quá tay trên VSEC lên tới 0,52. Đổi sang kiểu dấu mới trước khi tokenize (`to_bartpho_style`) đã đưa tỉ lệ này về 0,026. Đây là lỗi tiền xử lý, không phải giới hạn của mô hình. Nó cho thấy chuẩn hóa Unicode và vị trí dấu phải khớp với vocab của mô hình tiền huấn luyện.

## 2. Vì sao C thua trên lỗi thật: lệch phân bố loại lỗi

Phân bố kiểu lỗi được suy từ cặp (âm tiết sai, âm tiết đúng), lấy từ `real_error_kinds.csv`:

| Kiểu lỗi | Syn-train | VSEC-test | Viwiki-test |
|---|---|---|---|
| Chỉ sai dấu thanh | 6,6% | **49,6%** | **34,1%** |
| Sai dấu phụ (ă â ê ô ơ ư đ) | 0,4% | 12,0% | 15,7% |
| Sai 1 chữ cái | **83,8%** | 25,6% | 34,7% |
| Sai ≥2 ký tự | 9,3% | 4,5% | 7,5% |
| Tách/ghép âm tiết | 0,0% | 8,3% | 7,8% |

Bộ sinh nhiễu (tỉ lệ loại lỗi theo arXiv 2405.02573) chủ yếu tạo lỗi thay hoặc thêm một chữ cái. Lỗi thật thì một nửa là sai dấu thanh. Recall sửa lỗi theo kiểu lỗi trên VSEC (`ab_recall_by_kind.csv`):

| Kiểu lỗi (VSEC) | n | A | C v3 | C v4 |
|---|---|---|---|---|
| Chỉ sai dấu thanh | 4.286 | **0,875** | 0,330 | 0,718 |
| Sai dấu phụ | 1.036 | **0,897** | 0,491 | 0,764 |
| Sai 1 chữ cái | 2.201 | **0,745** | 0,541 | 0,595 |
| Sai ≥2 ký tự | 372 | 0,229 | 0,433 | **0,551** |
| Tách/ghép âm tiết | 407 | 0,000 | 0,091 | **0,268** |

**Nhận xét:**

- **A không phụ thuộc vào phân bố nhiễu khi huấn luyện.** Tập ứng viên của A luôn chứa mọi biến thể dấu thanh và dấu phụ của âm tiết. Việc chọn ứng viên do mô hình ngôn ngữ quyết định, mà mô hình này được học trên văn bản sạch.
- **C chỉ học những phép sửa đã thấy nhiều khi huấn luyện.** Lỗi dấu thanh chỉ chiếm 6,6% dữ liệu train, nên C hiếm khi dám đổi dấu. Kết quả là recall thấp và precision cao.
- **C mạnh hơn ở lỗi khó.** Đó là lỗi sai từ 2 ký tự trở lên (ví dụ `phpasp` → `pháp`) và một phần lỗi tách/ghép âm tiết. A không làm được các lỗi này vì giới hạn khoảng cách sửa ≤ 2 và vì A giữ ánh xạ một âm tiết sang một âm tiết.

## 3. Lỗi tạo ra từ có thật

Ta gọi một lỗi là **"tạo ra từ có thật"** khi âm tiết sai vẫn nằm trong từ điển âm tiết, ví dụ giảm→giam, tác giả→tác gia. Những lỗi này chỉ phát hiện được nhờ ngữ cảnh. Ngược lại, lỗi "tạo ra từ không có thật" là khi âm tiết sai nằm ngoài từ điển (hình `ab_recall_word_kind.png`).

| | VSEC: không có thật | VSEC: có thật | Viwiki: không có thật | Viwiki: có thật |
|---|---|---|---|---|
| n | 1.674 | **6.633** | 358 | **1.159** |
| A | 0,751 | **0,776** | **0,388** | **0,528** |
| C v3 | 0,678 | 0,329 | 0,330 | 0,249 |
| C v4 | **0,819** | 0,621 | 0,377 | 0,368 |

- **Đa số lỗi thật thuộc loại tạo ra từ có thật:** 80% trên VSEC và 76% trên Viwiki.
- **Với lỗi tạo ra từ không có thật,** C chỉ thua A khoảng 7 điểm.
- **Với lỗi tạo ra từ có thật,** C chỉ sửa được 33%. Mô hình mặc định giữ nguyên một âm tiết hợp lệ, trừ khi ngữ cảnh rất rõ.

Hai nhận xét này trùng nhau: phần lớn lỗi tạo ra từ có thật chính là lỗi sai dấu thanh.

## 4. Ablation C

### 4.1. Mật độ nhiễu và quy mô huấn luyện

Kết quả trên VSEC-test (`ablation_noise_density.csv`, hình `ablation_noise_density.png`):

| Cấu hình | P | R | F1 | Sửa quá tay |
|---|---|---|---|---|
| 5% nhiễu · 50k cặp · 1 epoch | 0,884 | 0,422 | 0,571 | 0,024 |
| 12% · 50k · 1 epoch | 0,882 | **0,480** | 0,622 | 0,028 |
| 25% · 50k · 1 epoch | 0,895 | 0,479 | **0,624** | 0,045 |
| 5% · 150k · 3 epoch (mô hình chính, v3) | 0,888 | 0,384 | 0,536 | 0,026 |

*Mô hình chính train trên `syn_train.jsonl`, sinh bằng `configs/noise.yaml` với mật độ 5% (`syn_train.stats.json`: `syllable_error_rate` 0,05), không phải 12%. Vì vậy mốc so sánh đúng cho quy mô huấn luyện là dòng 5% · 50k, không phải dòng 12%.*

- **Mật độ nhiễu:** tăng từ 5% lên 12% làm recall tăng 6 điểm, còn precision gần như không đổi. Lên 25% thì F1 không tăng thêm, trong khi tỉ lệ sửa quá tay gần gấp đôi. Mức 12% là điểm cân bằng hợp lý.
- **Quy mô huấn luyện** (cùng mật độ 5%): mô hình chính được train nhiều hơn khoảng 9 lần (150k cặp × 3 epoch). Nó tốt hơn trên Syn-test nhưng **kém hơn** trên VSEC (F1 0,536 so với 0,571; recall 0,384 so với 0,422). Đây là kết quả quan sát được, **chưa phải kết luận về nguyên nhân**: hai cấu hình khác nhau đồng thời ở số cặp (50k → 150k), số epoch (1 → 3) và bước chọn checkpoint (theo eval_loss trên Syn-val), nên chưa tách được yếu tố nào gây giảm. *Giả thuyết* là train càng lâu trên dữ liệu tổng hợp, mô hình càng khớp với phân bố nhiễu tự sinh và càng ít dám sửa những lỗi không giống dữ liệu train. Muốn kiểm chứng cần giữ nguyên dữ liệu và so các checkpoint theo từng epoch (hoặc theo số bước) trên VSEC-dev; nhóm chưa làm thí nghiệm này. Hệ quả thực tế vẫn đứng: chọn checkpoint theo Syn-val không đảm bảo tốt nhất trên lỗi thật.

### 4.2. Loại lỗi sinh nhiễu (v4)

Mục 2 cho thấy nguyên nhân chính khiến C thua trên lỗi thật là lệch phân bố loại lỗi. v4 kiểm chứng trực tiếp giả thuyết này:

*Lưu ý về tính độc lập của tập kiểm tra. S4 và v4 là *thí nghiệm phát triển sau khi phân tích lỗi trên tập kiểm tra*: các luật S4 được thiết kế từ phân tích sửa quá tay của A trên Viwiki-test, còn tỉ lệ lỗi dấu thanh/dấu phụ của v4 được đặt theo phân bố lỗi của VSEC-test. Tham số số của S4 được chọn trên VSEC-dev và v4 không chỉnh gì thêm theo test, nhưng tập kiểm tra vẫn đã ảnh hưởng tới quyết định thiết kế. Vì vậy kết quả của A + S4 và C v4 trên VSEC-test/Viwiki-test có thể lạc quan hơn thực tế và chưa chứng minh được khả năng tổng quát hóa; muốn khẳng định cần đánh giá trên một tập lỗi thật chưa từng được xem (nhóm chưa có tập như vậy). Các kết quả của A, C v3 và các ablation mật độ nhiễu không chịu ảnh hưởng này.*


- **Hai luật nhiễu mới** trong `src/vspell/noise/rules.py`:
  - `tone_error`: đổi hoặc bỏ dấu thanh;
  - `diacritic_error`: đổi a/ă/â, o/ô/ơ, u/ư, d/đ.
  - Cả hai ưu tiên tạo ra âm tiết có thật.
- **Cấu hình** `configs/noise_v4.yaml`: dấu thanh 40%, dấu phụ 12%, các luật cũ chiếm phần còn lại.
- **Phân bố kiểu lỗi** trong dữ liệu train mới: chỉ sai dấu thanh 47%, sai dấu phụ 13%, sai 1 chữ cái 37%. Phân bố này gần với VSEC.
- **Giữ nguyên mọi thứ khác so với d12:** cùng 50k câu gốc, cùng seed, mật độ 12%, 1 epoch. Vì vậy chênh lệch chỉ đến từ loại lỗi.

| Cấu hình | Syn-test F1 | VSEC P | VSEC R | VSEC F1 | Viwiki F1 | Sửa quá tay VSEC |
|---|---|---|---|---|---|---|
| d12 (nhiễu gốc) | – | 0,882 | 0,480 | 0,622 | – | 0,028 |
| **v4 (nhiễu dấu)** | 0,912 | **0,910** | **0,637** | **0,749** | **0,322** | 0,026 |
| A: n-gram | 0,898 | 0,819 | **0,741** | **0,778** | 0,304 | 0,063 |

**Nhận xét:**

- **F1 trên VSEC tăng 12,7 điểm** (0,622 → 0,749), chỉ nhờ đổi loại lỗi khi sinh nhiễu.
  - Recall tăng mạnh, từ 0,480 lên 0,637. Precision vẫn tăng nhẹ và tỉ lệ sửa quá tay không đổi.
  - Recall lỗi chỉ sai dấu thanh tăng từ 0,47 (d12) và 0,33 (v3) lên 0,72.
  - Recall lỗi tạo ra từ có thật tăng từ 0,33 (v3) lên 0,62.
- **Trên Viwiki, v4 vượt A** (0,322 so với 0,304).
  - Đây là lần đầu C thắng A trên một tập lỗi thật.
  - Precision của v4 cao hơn A (0,288 so với 0,221), và v4 sửa quá tay ít hơn nhiều (0,059 so với 0,128).
- **Trên VSEC, A vẫn nhỉnh hơn** (0,778 so với 0,749). Khoảng cách đã thu hẹp từ 24 điểm (so với v3) xuống còn 3 điểm.
  - v4 vẫn kém A ở lỗi dấu thanh (0,72 so với 0,88).
  - Ngược lại, v4 tốt hơn A ở lỗi sai ≥2 ký tự và lỗi tách/ghép âm tiết.
- **Đánh đổi:** trên Syn-test (sinh bằng nhiễu gốc), F1 của v4 giảm từ 0,959 (v3) xuống 0,912 và tỉ lệ sửa quá tay tăng lên 0,066. Điều này cũng khẳng định lại rằng mô hình khớp theo phân bố nhiễu mà nó được train.
- **Kết luận:** giả thuyết "lệch phân bố loại lỗi" ở mục 2 được xác nhận. So với d12 (cùng dữ liệu, chỉ khác loại lỗi), v4 tăng 12,7 điểm F1 trên VSEC. v4 cũng hơn mô hình chính (150k × 3 epoch) 21 điểm, nhưng so sánh này đổi cùng lúc loại lỗi, mật độ nhiễu và quy mô train, nên chỉ cho thấy một bộ sinh nhiễu tốt có thể bù cho ít dữ liệu hơn, không đo riêng được tác động của quy mô.
- **Hướng cải thiện tiếp (chưa làm):**
  - train v4 với nhiều dữ liệu hơn, chọn checkpoint theo VSEC-dev;
  - trộn một phần dữ liệu lỗi thật vào tập train;
  - kết hợp A và C (trên VSEC, có 760 vị trí chỉ C v4 sửa đúng và 1.665 vị trí chỉ A sửa đúng).

## 5. Ví dụ tiêu biểu

Ví dụ 15–23 so sánh A với C v3, chọn từ `outputs/tables/error_examples.md`; mỗi ví dụ ghi kèm id câu để tra trong `outputs/predictions/`. File đó được lấy mẫu ngẫu nhiên mỗi lần chạy `analyze_errors.py`, nên số thứ tự trong file có thể khác số ở đây. In **đậm** là vị trí lỗi hoặc vị trí bị sửa.

**Ví dụ 15. A đúng, C sai: lỗi tạo ra từ có thật (Viwiki) [`viwiki-6810-542`].**

- src: Không những Voltaire, tác **gia** Thụy Điển là Frans G. Bengtsson và Giáo sư Ragnhild Hatton đã viết những cuốn tiểu sử…
- A: tác **giả**. C: giữ nguyên `tác gia`.
- Nhận xét: "gia" là âm tiết hợp lệ. A nhận ra nhờ trigram "tác giả Thụy" có xác suất cao hơn hẳn. C coi "tác gia" là cụm hợp lệ (nó cũng là một từ Hán Việt hiếm) nên không sửa.

**Ví dụ 16. A đúng, C sai: chỉ sai dấu thanh (VSEC) [`vsec-404`].**

- src: công sức **đòng** góp của vợ, chồng
- A: **đóng** góp. C: giữ nguyên `đòng`.
- Nhận xét: đây là kiểu lỗi chiếm một nửa VSEC. A thử mọi dấu thanh của "đong" và chọn theo LM. C v3 hiếm khi thấy kiểu sửa này khi train. **C v4 sửa đúng** thành `đóng`.

**Ví dụ 17. A đúng, C sai: sai dấu thanh, ngay cả trên Syn-test [`syn_test-4838`].**

- src: **Buỗi** chiều, các đại biểu sẽ thảo luận theo tổ.
- A: **Buổi**. C: giữ nguyên `Buỗi`.
- Nhận xét: "Buỗi" không phải âm tiết hợp lệ. Dù vậy C vẫn bỏ qua, vì lỗi sai dấu thanh chỉ chiếm 6,6% dữ liệu train.

**Ví dụ 18. C đúng, A sai: tên riêng bị lỗi gõ (Syn) [`syn_test-3260`].**

- src: ông Đinh Văn **Fhu** **lpà** Đinh Văn Vũ
- A: chỉ sửa `lpà` → `là`, giữ nguyên `Fhu`. C: sửa cả hai, `Fhu` → **Thu** và `lpà` → **là**.
- Nhận xét: A tránh sửa từ viết hoa giữa câu (coi là tên riêng). C dùng ngữ cảnh "Đinh Văn ___" để đoán đúng tên.

**Ví dụ 19. C đúng, A sai: sai nhiều ký tự (VSEC) [`vsec-8988`].**

- src: thực hiện biện **phpasp** này
- A: giữ nguyên. C: **pháp**.
- Nhận xét: khoảng cách sửa từ "phpasp" đến "pháp" lớn hơn 2 (lỗi Telex gõ lặp), nên A không có ứng viên đúng. C sinh lại cả chuỗi nên không bị giới hạn này.

**Ví dụ 20. C đúng, A sai: lỗi cần hiểu cụm từ (Viwiki) [`viwiki-1022-1`].**

- src: và **cơ** **ngợi** những chiến thắng
- A: sửa `ngợi` → `ngơi`, thành "cơ ngơi", đúng ngữ pháp nhưng sai nghĩa. C: **ca** ngợi.
- Nhận xét: gold chỉ sửa "cơ" → "ca". A chọn sửa sai âm tiết vì "cơ ngơi" là một cụm phổ biến trong LM.

**Ví dụ 21. Cả hai sai: lỗi tách âm tiết (VSEC) [`vsec-8083`].**

- src: Từ **b ài** toán, Giáo viên yêu cầu học sinh khái quát… Gold: Từ **bài** toán.
- A: sửa thành `bỏ tài` và còn sửa quá tay `khái` → `khối`. C: không sửa gì.
- Nhận xét: A giữ ánh xạ một âm tiết sang một âm tiết, nên không thể ghép "b ài" thành "bài". Nó chỉ chọn được hai âm tiết riêng lẻ hợp với LM. C có thể ghép nhưng hiếm khi thấy kiểu lỗi này khi train (tách/ghép chiếm 0% Syn-train), nên bỏ qua.

**Ví dụ 22. Cả hai sai: lỗi dấu thanh tạo ra từ có thật ở đầu câu (Viwiki) [`viwiki-6988-72`].**

- src: **Dùng** lại nitrôglyxêrin các triệu chứng có thể dừng. Gold: **dừng** lại.
- A và C đều giữ nguyên.
- Nhận xét: "Dùng lại" là cụm hợp lệ và phổ biến. Muốn sửa phải hiểu nghĩa cả câu (ngừng dùng thuốc thì triệu chứng dừng). Gold cũng đổi chữ hoa đầu câu thành chữ thường, cho thấy nhãn của Viwiki có độ nhiễu nhất định.

**Ví dụ 23. Cả hai sai: lỗi cần kiến thức thế giới (Syn) [`syn_test-1468`].**

- src: hoa đào, hoa **hai**, bánh chưng… Gold: hoa **mai**.
- A: sửa thành "hồi" (hoa hồi, đúng ngữ pháp). C: giữ nguyên "hai".
- Nhận xét: cả hai đều sửa đúng `khoong` → `không` ở đầu câu. Nhưng muốn sửa "hai" → "mai" thì cần biết cặp "hoa đào, hoa mai" ngày Tết, điều mà trigram và mô hình đã fine-tune trên tin tức đều không nắm được.

## 6. Kết luận cho mục 5

1. **Trên dữ liệu cùng phân bố với lúc huấn luyện, C (BARTpho) tốt hơn A** cả về F1 lẫn độ an toàn (ít sửa quá tay).
2. **Trên lỗi thật, C v3 thua A.** Nguyên nhân chính là lệch phân bố loại lỗi giữa nhiễu tổng hợp và lỗi thật: lỗi sai dấu thanh và lỗi tạo ra từ có thật chiếm đa số trong lỗi thật nhưng rất ít trong dữ liệu train. Bản thân kiến trúc không phải nguyên nhân chính.
   - v4 xác nhận điều này. Khi chỉ đổi loại lỗi sinh nhiễu, C tăng từ 0,622 lên 0,749 trên VSEC (A đạt 0,778) và vượt A trên Viwiki (0,322 so với 0,304).
3. **Hai mô hình bổ sung cho nhau.** C mạnh ở lỗi sai nhiều ký tự và lỗi cần ngữ cảnh rộng, A mạnh ở lỗi dấu. Kết hợp hai mô hình (ví dụ dùng A tạo ứng viên rồi C xếp hạng, hoặc ensemble) là hướng phát triển tự nhiên.
4. **Kết quả ablation C:**
   - Tăng mật độ nhiễu lên 12% có lợi.
   - Mô hình train nhiều hơn (150k × 3 epoch) kém hơn bản 50k × 1 epoch trên lỗi thật (cùng mật độ 5%: 0,571 so với 0,536 trên VSEC). Nguyên nhân (số epoch, số cặp hay cách chọn checkpoint) chưa được tách riêng; "train lâu gây khớp quá mức với nhiễu tổng hợp" mới là giả thuyết.
   - Loại lỗi sinh nhiễu là yếu tố ảnh hưởng lớn nhất (+12,7 điểm F1 trên VSEC).
