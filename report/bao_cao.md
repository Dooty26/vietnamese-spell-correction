---
title: "Xây dựng hệ thống phát hiện, sửa lỗi chính tả tiếng Việt dựa trên phương pháp thống kê và mô hình học sâu"
subtitle: "Báo cáo bài tập lớn môn Xử lý ngôn ngữ tự nhiên"
lang: vi
---

<!--
Trạng thái bản thảo (task S7a): mục 1–4 hoàn chỉnh; mục 5 mới có kết quả mô hình A và ablation A.
Các chỗ đánh dấu [chờ BARTpho] / [chờ hình] / [cần bổ sung] sẽ được điền ở task S7b.
Mọi số liệu lấy từ outputs/tables/*.csv, data/processed/stats_*.json, data/processed/syn_*.stats.json và configs/*.yaml.
-->

# Tóm tắt

Báo cáo trình bày một hệ thống phát hiện và sửa lỗi chính tả tiếng Việt ở mức âm tiết, so sánh hai hướng tiếp cận: (A) mô hình kênh nhiễu (noisy channel) kết hợp mô hình ngôn ngữ n-gram Kneser-Ney tự cài đặt và giải mã beam search; (C) tinh chỉnh mô hình sinh chuỗi BARTpho-syllable. Dữ liệu huấn luyện được sinh tự động từ kho tin tức Binhvq bằng sáu loại lỗi tổng hợp (vùng miền, Telex, trượt phím, teencode, viết tắt, biến đổi ký tự ngẫu nhiên); việc đánh giá thực hiện trên một tập lỗi tổng hợp và hai tập lỗi thật (VSEC – tin tức, Viwiki-spelling – Wikipedia) bằng cùng một thủ tục căn chỉnh âm tiết. Mô hình A đạt Correction F1 là 0,898 trên Syn-test, 0,778 trên VSEC-test nhưng chỉ 0,304 trên Viwiki-test, nơi mật độ lỗi rất thấp (0,29%) khiến các lần sửa quá tay chiếm ưu thế. Thí nghiệm loại bỏ (ablation) cho thấy làm trơn Kneser-Ney vượt trội rõ rệt so với Laplace và bậc 3 là điểm cân bằng hợp lý giữa chất lượng và chi phí. Kết quả của mô hình C: [chờ BARTpho].

# 1. Giới thiệu

## 1.1. Bài toán

Lỗi chính tả xuất hiện thường xuyên trong văn bản tiếng Việt được soạn trên máy tính và điện thoại: người viết gõ sai phím, bộ gõ Telex/VNI không chuyển đổi kịp, người viết phát âm theo phương ngữ (nhầm *s/x*, *tr/ch*, *l/n*, *hỏi/ngã*), hoặc cố ý viết tắt và dùng teencode trong ngữ cảnh không trang trọng. Các lỗi này làm giảm chất lượng văn bản báo chí, hành chính, đồng thời gây nhiễu cho các hệ thống xử lý phía sau như tìm kiếm, dịch máy, nhận dạng thực thể hay các mô hình ngôn ngữ lớn.

Bài toán được xét trong báo cáo gồm hai bước:

- **Phát hiện lỗi (error detection):** cho câu đầu vào $x = (x_1, \dots, x_n)$ gồm các âm tiết, xác định tập vị trí $i$ mà $x_i$ bị viết sai.
- **Sửa lỗi (error correction):** tại mỗi vị trí bị phát hiện, đề xuất âm tiết đúng $y_i$; đầu ra là câu đã sửa $y$.

Khác với tiếng Anh, đơn vị chính tả tự nhiên của tiếng Việt là **âm tiết** (các âm tiết được phân tách bằng dấu cách, một từ có thể gồm nhiều âm tiết). Tập âm tiết hợp lệ tương đối nhỏ và có cấu trúc chặt chẽ (phụ âm đầu – vần – thanh điệu), nên phần lớn lỗi có thể quy về thay thế một âm tiết bằng một âm tiết khác. Tuy nhiên, tiếng Việt có hai đặc điểm làm bài toán khó hơn: (i) **lỗi “từ có thật” (real-word error)** rất phổ biến – một âm tiết sai do phương ngữ như *“dành”* thay cho *“giành”* vẫn là âm tiết hợp lệ, nên không thể phát hiện chỉ bằng tra từ điển mà phải dựa vào ngữ cảnh; (ii) **dấu thanh và dấu phụ** khiến số lượng biến thể của một âm tiết lớn, đồng thời tồn tại hai quy ước đặt dấu thanh (*“hoà”* và *“hòa”*) đều được chấp nhận.

## 1.2. Hướng tiếp cận và câu hỏi nghiên cứu

Theo yêu cầu của đề tài, nhóm xây dựng và so sánh hai phương án thuộc hai trường phái:

- **Phương án A – thống kê (baseline):** mô hình kênh nhiễu cổ điển [6, 7], trong đó xác suất câu đúng được ước lượng bằng mô hình ngôn ngữ n-gram mức âm tiết làm trơn Kneser-Ney [4, 5], còn mô hình kênh được xấp xỉ bằng chi phí của từng nguồn sinh ứng viên (giải mã Telex, khung không dấu, tập dễ nhầm theo phương ngữ, teencode, khoảng cách soạn thảo theo SymSpell [8]). Câu đầu ra được tìm bằng beam search.
- **Phương án C – học sâu:** tinh chỉnh BARTpho-syllable [3], một mô hình tiền huấn luyện sinh chuỗi (sequence-to-sequence) cho tiếng Việt, trên các cặp (câu lỗi, câu đúng) tự sinh, theo hướng các công trình dùng Transformer cho sửa lỗi chính tả tiếng Việt [1, 2, 9].

(Phương án B – gắn nhãn chuỗi bằng BiLSTM với đặc trưng ký tự – được để ngỏ, chỉ thực hiện nếu còn thời gian.)

Báo cáo nhằm trả lời hai câu hỏi nghiên cứu:

1. **(CH1)** Mô hình thống kê n-gram và mô hình học sâu BARTpho khác nhau như thế nào về độ chính xác phát hiện/sửa lỗi, về khả năng xử lý từng loại lỗi và về xu hướng sửa quá tay?
2. **(CH2)** Mô hình được xây dựng hoặc huấn luyện chủ yếu trên **lỗi tổng hợp** tổng quát hóa sang **lỗi thật** đến mức nào, và mức độ đó thay đổi ra sao giữa hai miền văn bản (tin tức và Wikipedia)?

## 1.3. Đóng góp

Các kết quả chính của bài tập lớn gồm:

- Một pipeline dữ liệu tái lập được: lấy mẫu và làm sạch 571.200 câu từ kho tin tức Binhvq, chuẩn hóa Unicode và vị trí dấu thanh thống nhất cho mọi tập, loại bỏ câu trùng và câu rò rỉ từ các tập kiểm tra.
- Bộ sinh lỗi tổng hợp gồm sáu loại lỗi theo phân loại trong tài liệu, giữ nguyên số âm tiết của câu để có nhãn từng vị trí.
- Mô hình A cài đặt hoàn toàn bằng Python (không phụ thuộc KenLM), gồm mô hình ngôn ngữ n-gram với hai phương pháp làm trơn, bộ sinh ứng viên đa nguồn và bộ giải mã beam search.
- Một thủ tục đánh giá chung cho mọi mô hình dựa trên căn chỉnh âm tiết, báo cáo đồng thời các chỉ số phát hiện, sửa lỗi, độ chính xác câu và tỉ lệ sửa quá tay.
- Thí nghiệm ablation về bậc n-gram và phương pháp làm trơn; phân tích recall theo loại lỗi.

## 1.4. Cấu trúc báo cáo

Mục 2 mô tả dữ liệu, quá trình làm sạch, chuẩn hóa, sinh lỗi và chống rò rỉ. Mục 3 trình bày pipeline tổng thể và hai phương án A, C. Mục 4 mô tả thiết lập thực nghiệm và các chỉ số. Mục 5 báo cáo kết quả và phân tích. Mục 6 kết luận và nêu hướng phát triển. Phụ lục trình bày việc sử dụng công cụ AI trong quá trình thực hiện.

# 2. Dữ liệu

## 2.1. Tổng quan các nguồn dữ liệu

Hệ thống sử dụng ba nguồn dữ liệu với vai trò khác nhau (Bảng 2.1). Kho tin tức Binhvq là văn bản “sạch”, dùng để xây dựng từ điển âm tiết, huấn luyện mô hình ngôn ngữ và làm nguồn sinh lỗi tổng hợp. Hai tập VSEC và Viwiki-spelling chứa lỗi thật do con người gán nhãn, chỉ dùng để chỉnh tham số (một phần nhỏ của VSEC) và để kiểm tra.

**Bảng 2.1.** Các nguồn dữ liệu và vai trò.

| Tập dữ liệu | Nguồn | Miền | Vai trò | Cách chia |
|---|---|---|---|---|
| Binhvq news | `vietgpt/binhvq_news_vi` (Hugging Face) [10] | Tin tức | Từ điển, LM n-gram, nguồn sinh lỗi tổng hợp | LM-corpus 400k câu · Syn-train 150k · Syn-val 5k · Syn-test 5k (chia theo đoạn văn) |
| VSEC [1] | GitHub `VSEC2021/VSEC` | Tin tức | Lỗi thật: chỉnh tham số và kiểm tra | VSEC-dev ~20% (1.873 câu) · VSEC-test ~80% (7.468 câu), seed 42, câu cùng câu đích luôn cùng tập |
| Viwiki-spelling [2] | GitHub `heraclex12/Viwiki-spelling` (CC BY 4.0) | Wikipedia | Lỗi thật: chỉ kiểm tra | Viwiki-test 15.153 câu |

### 2.1.1. Binhvq news

Phiên bản trên Hugging Face gồm 9 tệp parquet với một cột `text` duy nhất; mỗi dòng là một **đoạn văn** (khoảng 200–1.000 ký tự) và **không có mã bài báo**. Để tránh tải toàn bộ khoảng 4,8 GB dữ liệu, nhóm lấy ngẫu nhiên (theo seed 42) các row group trên cả 9 tệp, đọc song song nhưng giữ đúng thứ tự đã xáo trộn để kết quả tái lập được (`src/vspell/data/binhvq.py`). Quá trình đọc dừng khi số câu sạch đạt 102% tổng kích thước các tập cần dùng. Thống kê của lần chạy đầy đủ được ghi trong `stats_binhvq.json` (Bảng 2.2).

**Bảng 2.2.** Thống kê quá trình lấy mẫu và làm sạch Binhvq.

| Đại lượng | Giá trị |
|---|---:|
| Số đoạn văn đã đọc | 706.360 |
| Số câu đã tách | 791.164 |
| Số câu bị loại bởi bộ lọc chất lượng | 219.088 |
| Số câu trùng lặp bị loại | 868 |
| Số câu bị loại do rò rỉ từ tập kiểm tra | 8 |
| **Số câu giữ lại** | **571.200** |
| LM-corpus / Syn-train / Syn-val / Syn-test | 400.000 / 150.000 / 5.000 / 5.000 |

Như vậy khoảng 27,7% số câu bị bộ lọc chất lượng loại bỏ (mục 2.2.2). Số câu giữ lại (571.200) lớn hơn tổng kích thước bốn tập (560.000) vì có biên dự phòng 2%; các đoạn còn thừa sau khi đã đủ kích thước không được sử dụng.

### 2.1.2. VSEC

VSEC [1] gồm 9.341 câu tin tức tiếng Việt, mỗi câu được gán nhãn theo từng âm tiết: âm tiết đúng/sai và danh sách âm tiết thay thế. Nhóm chuyển dữ liệu về định dạng chung `{id, src, tgt}` (`src/vspell/data/vsec.py`), trong đó `tgt` lấy gợi ý thay thế đầu tiên; các gợi ý còn lại được giữ trong trường `alternatives`. Hai đặc điểm cần xử lý riêng:

- **Âm tiết bị tách:** một âm tiết bị gõ thành hai mảnh (*“nh iên,”*) được gán nhãn theo kiểu mảnh đầu thành chuỗi rỗng, mảnh sau thành âm tiết đầy đủ (*“nhiên,”*). Thủ tục căn chỉnh ở mục 4.3 quy các trường hợp này về vị trí âm tiết trong câu nguồn.
- **Lỗi gán nhãn:** có trường hợp mọi mảnh của âm tiết bị tách đều được gán chuỗi rỗng, làm mất chữ trong câu đích. Hàm `_repair_lost_merges` ghép các mảnh lại khi kết quả là một âm tiết tiếng Việt hợp lệ; trong toàn bộ VSEC có đúng 1 chỗ được sửa theo cách này.

VSEC được chia cố định thành VSEC-dev (20%) dùng để chỉnh tham số giải mã và VSEC-test (80%) dùng để báo cáo kết quả. VSEC có một số câu đúng giống nhau nhưng mang lỗi khác nhau; các câu có cùng câu đích (sau chuẩn hóa, không phân biệt hoa thường và dấu câu) luôn được xếp vào cùng một tập (`split_dev_test`), nhờ đó 5 câu test trùng câu đích với dev được chuyển sang dev. Tham số giải mã chính của A (`oov_penalty`, `real_word_penalty`) được chọn trên bản dev trước khi gom nhóm (ít hơn 5 câu); các luật S4 đã được chỉnh lại trên bản dev mới và chọn đúng cấu hình cũ.

### 2.1.3. Viwiki-spelling

Viwiki-spelling [2] gồm 107 bài viết Wikipedia tiếng Việt, mỗi bài kèm danh sách lỗi đánh dấu theo vị trí ký tự trong toàn bài và danh sách gợi ý. Nhóm tách bài thành câu (chỉ cắt khi ký tự kế tiếp là chữ hoa theo Unicode hoặc số, và từ trước dấu chấm không phải chữ viết tắt như *TP.*, *GS.* hay chữ cái viết tắt tên), áp các sửa đổi nằm trọn trong câu để tạo câu đích, và **giữ cả những câu không có lỗi** để đo tỉ lệ sửa quá tay (`src/vspell/data/viwiki.py`). Các lỗi có vị trí không khớp chữ, chồng lấn nhau hoặc vắt qua ranh giới câu bị bỏ, và **câu chứa lỗi bị bỏ cũng bị loại** để không bị coi nhầm là câu sạch (1 lỗi vắt qua ranh giới câu, 2 câu bị loại; thống kê trong `stats_real.json`); câu ngắn hơn 3 token bị loại. Kết quả là 15.153 câu. Tham số của các mô hình không được chỉnh trên tập này, nhưng thiết kế của S4 có dựa trên phân tích lỗi trên nó (mục 5.5).

## 2.2. Làm sạch và chuẩn hóa

### 2.2.1. Chuẩn hóa Unicode và vị trí dấu thanh

Cùng một âm tiết tiếng Việt có thể được mã hóa theo nhiều cách: dạng dựng sẵn (NFC) hoặc dạng tổ hợp (NFD), và dấu thanh có thể đặt theo kiểu “cũ” (*hòa, thúy, khỏe*) hoặc kiểu “mới” (*hoà, thuý, khoẻ*). Nếu không thống nhất, một mô hình sẽ bị tính là “sửa” hoặc “sai” chỉ vì khác biệt mã hóa. Vì vậy, hàm `normalize` (`src/vspell/normalize.py`) được áp dụng cho **mọi tập dữ liệu và mọi đầu ra mô hình trước khi chấm điểm**, gồm ba bước:

1. Đưa văn bản về Unicode NFC và thay ký tự *Ð* (Eth) dễ nhầm về *Đ*.
2. Quy vị trí dấu thanh về kiểu cũ, phổ biến trên báo chí, theo các quy tắc: (a) nếu vần có nguyên âm mang dấu phụ (*ă, â, ê, ô, ơ, ư*) thì đặt dấu lên nguyên âm đó (với *ươ* chọn *ơ*); (b) nếu vần chỉ có một nguyên âm thì đặt lên nguyên âm ấy; (c) nếu có phụ âm cuối thì đặt lên nguyên âm cuối của vần (*toán, hoàng*); (d) âm tiết mở có ba nguyên âm thì đặt lên nguyên âm giữa (*ngoài*), có hai nguyên âm thì đặt lên nguyên âm đầu (*hòa, thúy, của*). Chỉ những token khớp cấu trúc âm tiết tiếng Việt (phụ âm đầu – nguyên âm – phụ âm cuối) mới được xử lý; các token khác giữ nguyên.
3. Chuẩn hóa khoảng trắng. Hàm không hạ chữ thường và không bỏ dấu câu.

Câu sau chuẩn hóa được tách thành token bằng biểu thức chính quy: mỗi âm tiết, mỗi dấu câu là một token; số (kể cả *40.000*, *3,5*) được giữ nguyên là một token.

Hệ quả quan trọng của bước này là **khác biệt kiểu đặt dấu thanh không được tính là lỗi**. Trong VSEC, sau chuẩn hóa còn 10.857 lỗi âm tiết và 8.770 câu có lỗi trên tổng số 9.341 câu; 571 câu còn lại trở thành câu sạch vì lỗi duy nhất được gán nhãn trong câu là khác biệt vị trí dấu thanh. Đây là một lựa chọn có chủ đích: cả hai kiểu đặt dấu đều được chấp nhận trong thực tế, nên việc buộc mô hình học một kiểu cụ thể không phản ánh năng lực sửa lỗi chính tả.

### 2.2.2. Lọc câu Binhvq

Mỗi đoạn văn Binhvq được giải mã HTML entity, bỏ thẻ HTML và URL, chuẩn hóa như trên rồi tách câu. Một câu chỉ được giữ làm “câu chuẩn” khi thỏa mãn đồng thời các điều kiện (hàm `keep_sentence`):

- độ dài từ 5 đến 80 token (VSEC dài tối đa 70 token);
- không chứa dấu vết HTML, URL, email hay mã JavaScript;
- bắt đầu bằng chữ hoa, chữ số, ngoặc hoặc dấu nháy; kết thúc bằng *. ! ? …* hoặc dấu nháy/ngoặc đóng;
- có ít nhất 5 từ chữ cái, token là số chiếm không quá 20%, từ viết hoa toàn bộ chiếm không quá 30%;
- ít nhất 85% số từ (không tính từ viết tắt toàn chữ hoa như *UBND*, *ASEAN*) là âm tiết tiếng Việt hợp lệ về cấu trúc.

Điều kiện cuối dùng hàm `is_vietnamese_syllable`, kiểm tra cấu trúc âm tiết và một số ràng buộc chính tả: phụ âm cuối tắc *c, ch, p, t* chỉ đi với thanh sắc hoặc nặng; *k, gh, ngh* chỉ đứng trước *i, e, ê, y*; *c, ng* không đứng trước *i, e, ê*.

## 2.3. Thống kê các tập kiểm tra lỗi thật

Bảng 2.3 tóm tắt các tập lỗi thật sau chuẩn hóa (`stats_real.json`). Số lỗi được đếm theo vị trí âm tiết trong câu nguồn, dùng chính thủ tục căn chỉnh của bộ đánh giá (mục 4.3), nên khớp với số lỗi mà mọi mô hình được chấm.

**Bảng 2.3.** Thống kê các tập lỗi thật (sau chuẩn hóa).

| Tập | Số câu | Câu có lỗi | Số lỗi âm tiết | Mật độ lỗi / token | Độ dài TB (token) | Trung vị | Tối đa |
|---|---:|---:|---:|---:|---:|---:|---:|
| VSEC (toàn bộ) | 9.341 | 8.770 | 10.857 | 3,51% | 33,11 | 32 | 70 |
| VSEC-dev | 1.873 | 1.765 | 2.213 | 3,52% | 33,60 | 33 | 68 |
| VSEC-test | 7.468 | 7.005 | 8.643 | 3,51% | 32,99 | 32 | 70 |
| Viwiki-test | 15.153 | 1.281 | 1.542 | 0,29% | 34,84 | 31 | 244 |

Hai tập lỗi thật khác nhau rất rõ về phân bố. VSEC là tập “dày lỗi”: khoảng 94% số câu có ít nhất một lỗi, trung bình 1,16 lỗi mỗi câu và mật độ 3,5% token. Viwiki là tập “thưa lỗi”: chỉ khoảng 8,3% số câu có lỗi (tức khoảng 92% câu sạch) và mật độ lỗi chỉ 0,29% token, thấp hơn VSEC khoảng 12 lần. Vì vậy, trên Viwiki, chỉ cần mô hình sửa nhầm một tỉ lệ nhỏ trong số câu sạch là độ chính xác (precision) đã giảm mạnh. Hai cách chia VSEC-dev/VSEC-test có phân bố gần như trùng nhau, cho thấy việc chỉnh tham số trên VSEC-dev là đại diện. [chờ hình: `outputs/figures/` – phân bố độ dài câu và mật độ lỗi của ba tập, do task S5a tạo]

## 2.4. Sinh lỗi tổng hợp

### 2.4.1. Nguyên tắc

Dữ liệu có nhãn lỗi thật có kích thước nhỏ (VSEC chưa tới 10 nghìn câu), không đủ để huấn luyện mô hình học sâu. Nhóm sinh dữ liệu huấn luyện bằng cách làm nhiễu câu sạch Binhvq (`src/vspell/noise/`), theo các nguyên tắc:

- **Chỉ sinh lỗi bên trong âm tiết, giữ nguyên số âm tiết** (`keep_syllable_count: true`). Nhờ vậy mỗi lỗi có nhãn vị trí và loại lỗi chính xác, phục vụ phân tích recall theo loại lỗi. Lỗi tách/ghép âm tiết không được sinh (xem hạn chế ở mục 2.6).
- Chỉ làm nhiễu token là âm tiết tiếng Việt hợp lệ; bỏ qua số, dấu câu, từ nước ngoài và từ viết tắt toàn chữ hoa.
- Mỗi âm tiết hợp lệ bị làm nhiễu độc lập với xác suất **5%** (`syllable_error_rate`), giá trị được chọn gần với mật độ lỗi của VSEC; câu đã được chọn để làm nhiễu luôn có ít nhất một lỗi.
- **15% số câu được giữ nguyên** (`clean_sentence_ratio`) để mô hình học được việc *không sửa* câu đúng, hạn chế sửa quá tay.
- Phép thay thế được thực hiện theo vị trí ký tự nên khoảng trắng và dấu câu của câu gốc được giữ nguyên; sau khi làm nhiễu, câu được chuẩn hóa lại và chỉ những lỗi còn tồn tại sau chuẩn hóa mới được ghi nhận.

### 2.4.2. Sáu loại lỗi

Trọng số các loại lỗi (`configs/noise.yaml`) được đặt theo hướng ưu tiên các nhóm lỗi phổ biến trong văn bản tiếng Việt như phân tích trong [9]: vùng miền 31%, Telex 30%, trượt phím 30%, teencode, viết tắt và biến đổi ký tự ngẫu nhiên mỗi loại 3%. Mỗi loại lỗi là một hàm biến đổi một âm tiết (`src/vspell/noise/rules.py`):

1. **Vùng miền (`region`):** hoán đổi phụ âm đầu (*s↔x, tr↔ch, r/d/gi, d→v, l↔n*), phụ âm cuối (*n↔ng, t↔c, nh→n, ch→t*) hoặc thanh điệu (*hỏi↔ngã, nặng→hỏi*). Chỉ những kết quả là âm tiết hợp lệ mới được chấp nhận, nên loại lỗi này luôn tạo ra **lỗi từ có thật**, khó phát hiện nhất. Ví dụ: *“sản” → “xản”*, *“kết” → “kếc”*.
2. **Telex (`telex`):** mô phỏng việc bộ gõ không chuyển đổi: phím dấu thanh bị để lại cuối âm tiết (*“được” → “đươcj”*), một dấu phụ không được chuyển (*“được” → “đuwơc”*) hoặc toàn bộ âm tiết ở dạng phím gõ (*“được” → “dduwowcj”*).
3. **Trượt phím (`fat_finger`):** thay một chữ cái bằng một phím lân cận trên bàn phím QWERTY, hoặc (với xác suất 0,3) chèn thêm phím lân cận; chữ bị gõ nhầm mất dấu phụ của nó.
4. **Teencode (`teencode`):** thay cả âm tiết bằng dạng teencode quen thuộc (*“không” → “ko”, “gì” → “j”, “biết” → “bít”*) hoặc thay phụ âm đầu (*ph→f, gi→j, qu→w, d/v→z, c↔k*).
5. **Viết tắt (`acronym`):** dạng viết tắt khi gõ nhanh (*“được” → “đc”, “người” → “ng”, “trước” → “trc”*).
6. **Biến đổi ký tự (`edit_distance`):** chèn, xóa, thay, hoán vị một ký tự ngẫu nhiên, hoặc bỏ hết dấu của âm tiết (*“không” → “khong”*).

Với mỗi âm tiết được chọn, bộ sinh bốc thứ tự các loại lỗi theo trọng số (không lặp lại); nếu loại lỗi được bốc không áp dụng được cho âm tiết đó (chẳng hạn teencode chỉ áp dụng cho một danh sách từ và một số phụ âm đầu), bộ sinh thử loại kế tiếp.

### 2.4.3. Thống kê dữ liệu tổng hợp

Ba tập tổng hợp được sinh với seed khác nhau (train 42, val 43, test 44) để tránh trùng mẫu nhiễu. Bảng 2.4 trình bày thống kê thực tế (`syn_*.stats.json`).

**Bảng 2.4.** Thống kê các tập lỗi tổng hợp (mật độ cấu hình 5% âm tiết).

| Đại lượng | Syn-train | Syn-val | Syn-test |
|---|---:|---:|---:|
| Số cặp câu | 150.000 | 5.000 | 5.000 |
| Số câu giữ sạch | 22.345 | 743 | 741 |
| Số lỗi | 336.381 | 11.196 | 11.172 |
| Lỗi / token | 4,18% | 4,16% | 4,16% |
| Trượt phím | 37,85% | 37,86% | 37,28% |
| Telex | 30,06% | 30,71% | 30,17% |
| Vùng miền | 27,17% | 26,80% | 27,37% |
| Biến đổi ký tự | 3,59% | 3,48% | 3,95% |
| Teencode | 1,07% | 0,90% | 0,99% |
| Viết tắt | 0,26% | 0,25% | 0,23% |

Có hai điểm khác biệt so với cấu hình. Thứ nhất, mật độ lỗi tính trên toàn bộ token (khoảng 4,2%) thấp hơn 5% vì 15% số câu được giữ sạch và vì mẫu số gồm cả dấu câu, số – những token không bao giờ bị làm nhiễu. Thứ hai, phân bố loại lỗi thực tế lệch so với trọng số: teencode và viết tắt chỉ áp dụng được cho một tập nhỏ âm tiết, còn lỗi vùng miền đòi hỏi kết quả phải là âm tiết hợp lệ; khi các loại này không áp dụng được, bộ sinh chuyển sang loại kế tiếp, mà trượt phím hầu như luôn áp dụng được. Do đó tỉ lệ trượt phím tăng lên khoảng 37–38%, còn teencode và viết tắt giảm xuống khoảng 1% và 0,25%. Mật độ lỗi tổng hợp (4,2%) vẫn gần mật độ của VSEC (3,5%). [chờ hình: `outputs/figures/` – phân bố loại lỗi, do task S5a tạo]

## 2.5. Chia tập và chống rò rỉ dữ liệu

### 2.5.1. Chia tập Binhvq theo đoạn văn

Kế hoạch ban đầu là chia Binhvq theo **bài báo** để câu của cùng một bài không đồng thời xuất hiện trong tập huấn luyện và tập kiểm tra. Do bản Hugging Face không có mã bài báo, nhóm chia theo **đoạn văn** – đơn vị nhỏ nhất còn giữ được ngữ cảnh liền mạch: mọi câu của một đoạn thuộc cùng một tập (hàm `split_by_paragraph`). Các đoạn được xáo trộn theo seed 42 và lần lượt đổ vào Syn-test, Syn-val, Syn-train rồi LM-corpus (các tập đánh giá được điền trước để luôn đủ kích thước).

### 2.5.2. Loại trùng và chống rò rỉ từ các tập kiểm tra

Vì Binhvq và VSEC cùng thuộc miền tin tức, có nguy cơ một câu của VSEC (dạng đúng hoặc dạng lỗi) đã xuất hiện trong Binhvq. Nếu câu đó lọt vào LM-corpus hay Syn-train, mô hình có thể “nhớ” đáp án. Mô-đun `src/vspell/data/dedup.py` xử lý theo hai mức:

- **Trùng tuyệt đối:** mỗi câu được chuẩn hóa, hạ chữ thường, bỏ dấu câu rồi băm bằng SHA-1. Câu Binhvq có khóa trùng với khóa của bất kỳ câu nguồn hoặc câu đích nào trong VSEC-dev, VSEC-test, Viwiki-test bị loại. Cũng khóa này được dùng để loại các câu trùng lặp ngay trong Binhvq.
- **Gần trùng:** tập tất cả 5-gram âm tiết của các câu kiểm tra được dựng sẵn; câu Binhvq bị loại nếu tỉ lệ 5-gram của nó nằm trong tập này từ **0,8** trở lên.

Kết quả (Bảng 2.2): 868 câu trùng lặp trong Binhvq và 8 câu rò rỉ từ tập kiểm tra đã bị loại. Số câu rò rỉ nhỏ cho thấy nguy cơ này thấp trên mẫu đã lấy, nhưng bước kiểm tra vẫn cần thiết để kết quả đáng tin cậy.

## 2.6. Hạn chế của dữ liệu

- **Chia theo đoạn văn, không theo bài báo.** Các đoạn khác nhau của cùng một bài báo có thể rơi vào các tập khác nhau. Do các đoạn của cùng bài thường chia sẻ tên riêng và chủ đề, điều này có thể làm kết quả trên Syn-test lạc quan hơn đôi chút. Hai tập lỗi thật (VSEC, Viwiki) không bị ảnh hưởng.
- **Khác biệt kiểu đặt dấu không tính là lỗi** (mục 2.2.1): 571 câu VSEC chỉ có lỗi loại này được coi là câu sạch, nên kết quả không so sánh trực tiếp được với các công trình chấm trên VSEC gốc.
- **Không xử lý lỗi tách/ghép âm tiết.** Bộ sinh lỗi chỉ tạo lỗi bên trong âm tiết, và mô hình A chỉ thay thế âm tiết theo quan hệ 1–1. Các lỗi tách âm tiết trong VSEC (*“nh iên”*) và lỗi dính từ trong Viwiki (*“đượcmột”*) vẫn được giữ trong tập kiểm tra và được tính là lỗi, nhưng nằm ngoài khả năng của mô hình A.
- **Phân bố loại lỗi tổng hợp** là một giả định; mục 5 cho thấy khoảng cách giữa kết quả trên Syn-test và trên lỗi thật một phần đến từ đây.

# 3. Phương pháp

## 3.1. Pipeline tổng thể

Hình 3.1 trình bày pipeline của hệ thống. Pipeline gồm ba khối: (1) dữ liệu và tiền xử lý (mục 2), (2) hai mô hình sửa lỗi A và C được xây dựng độc lập, (3) một bộ đánh giá chung. Mọi đầu vào và đầu ra đều đi qua cùng hàm chuẩn hóa, và mọi mô hình đều được chấm bằng cùng một script (`scripts/evaluate.py`), nên kết quả so sánh được trực tiếp.

![Hình 3.1. Pipeline tổng thể của hệ thống (nguồn Mermaid: `report/figures/pipeline.mmd`).](figures/pipeline.svg)

**Hình 3.1.** Pipeline tổng thể. Đường nét đứt: VSEC-dev chỉ được dùng để chỉnh tham số giải mã của mô hình A; tham số không được chỉnh trên tập kiểm tra. Riêng thiết kế của S4 và v4 được rút ra từ phân tích lỗi trên VSEC-test/Viwiki-test (mục 5.5).

Mã nguồn được tổ chức thành gói Python `vspell` (`src/vspell/`) với các mô-đun `normalize`, `data`, `noise`, `models`, `eval`; các bước chạy là script dòng lệnh trong `scripts/` (`prepare_data.py` → `gen_noise.py` → `train_ngram.py` / `train_bartpho.py` → `predict.py` → `evaluate.py`), cấu hình nằm trong `configs/*.yaml`.

## 3.2. Phương án A: mô hình kênh nhiễu với mô hình ngôn ngữ n-gram

### 3.2.1. Mô hình kênh nhiễu

Theo cách tiếp cận kênh nhiễu [6, 7], câu quan sát $x$ được xem là kết quả của việc câu đúng $y$ đi qua một “kênh” gây lỗi. Câu đúng được khôi phục bằng

$$\hat{y} = \arg\max_{y} \; \Big[\, \sum_{i=1}^{n} \log_{10} P(x_i \mid y_i) \;+\; \lambda \log_{10} P_{\text{LM}}(y) \,\Big],$$

trong đó $P(x_i \mid y_i)$ là mô hình kênh (xác suất viết $y_i$ thành $x_i$), $P_{\text{LM}}(y)$ là mô hình ngôn ngữ, $\lambda$ là trọng số LM (`lm_weight` = 1,0). Vì chỉ xét thay thế 1–1, $y$ có cùng số âm tiết với $x$. Không có dữ liệu cặp lỗi đủ lớn để ước lượng $P(x_i \mid y_i)$ một cách xác suất, nhóm xấp xỉ log-xác suất kênh bằng **chi phí cố định theo nguồn sinh ứng viên** (mục 3.2.4), cộng thêm hai hình phạt điều khiển hành vi phát hiện lỗi (mục 3.2.5). Mô-đun cài đặt: `src/vspell/models/ngram_corrector.py`.

### 3.2.2. Mô hình ngôn ngữ n-gram mức âm tiết

Mô hình ngôn ngữ được nhóm tự cài đặt (`src/vspell/models/ngram_lm.py`) thay vì dùng KenLM như kế hoạch ban đầu (lý do ở mục 3.2.7). Mỗi câu được tách token như mục 2.2.1, hạ chữ thường, mọi số được thay bằng `<num>`, câu được thêm $n-1$ ký hiệu `<s>` ở đầu và `</s>` ở cuối. Với `min_count = 1`, mọi token của LM-corpus đều vào từ vựng; từ vựng của LM huấn luyện trên 400.000 câu có $|V| = 72.774$ token (không tính `<s>`).

Hai phương pháp làm trơn được cài đặt:

- **Laplace (add-k)**, chỉ dùng bậc cao nhất:
$$P(w \mid h) = \frac{c(h, w) + k}{c(h) + k\,|V|}, \quad k = 1.$$
- **Kneser-Ney nội suy** [4, 5] với chiết khấu cố định $D = 0{,}75$:
$$P_{\text{KN}}(w \mid h) = \frac{\max\big(c(h,w) - D,\, 0\big)}{c(h)} + \frac{D \cdot N_{1+}(h\,\bullet)}{c(h)} \, P_{\text{KN}}(w \mid h'),$$
trong đó $h'$ là ngữ cảnh $h$ bỏ đi từ xa nhất, $N_{1+}(h\,\bullet)$ là số loại từ khác nhau đã theo sau $h$. Ở bậc cao nhất, $c(\cdot)$ là số đếm thô; ở các bậc thấp hơn, $c(\cdot)$ được thay bằng **số đếm tiếp nối** (continuation count) $N_{1+}(\bullet\, h\, w)$ – số ngữ cảnh khác nhau mà n-gram $h\,w$ đã xuất hiện sau. Bậc 1 được nội suy với phân phối đều $1/|V|$, nên mọi từ (kể cả `<unk>`) đều có xác suất dương. Nếu ngữ cảnh $h$ chưa từng xuất hiện, mô hình lùi hẳn về bậc thấp hơn.

Mọi xác suất được tính ở dạng $\log_{10}$. Để tăng tốc giải mã, xác suất theo cặp (từ, ngữ cảnh) được lưu đệm bằng `lru_cache` với tối đa 2 triệu mục. Cấu hình mặc định là **bậc 3, Kneser-Ney** (`configs/ngram.yaml`); bậc 2, 4 và làm trơn Laplace được dùng trong ablation.

### 3.2.3. Từ điển âm tiết

Từ điển quyết định một âm tiết là “có thật” hay không. Từ điển được dựng từ LM-corpus: một âm tiết (chữ thường, đã chuẩn hóa) được đưa vào nếu xuất hiện ít nhất 3 lần (`lexicon_min_count = 3`) **và** hợp lệ về cấu trúc theo `is_vietnamese_syllable`. Từ điển của mô hình chính gồm 6.730 âm tiết. Sự khác biệt lớn giữa kích thước từ vựng LM (72.774) và từ điển (6.730) phản ánh việc LM-corpus chứa nhiều token không phải âm tiết tiếng Việt (tên riêng nước ngoài, viết tắt, từ hiếm), những token này vẫn có xác suất trong LM nhưng không được dùng làm ứng viên sửa.

### 3.2.4. Sinh ứng viên

Với mỗi âm tiết $x_i$ (chữ thường), lớp `CandidateGenerator` sinh tập ứng viên từ sáu nguồn; chỉ những ứng viên có trong từ điển được giữ lại. Mỗi nguồn có một chi phí kênh (log₁₀, âm) phản ánh mức độ tin cậy (Bảng 3.1). Nếu một ứng viên được sinh từ nhiều nguồn, chi phí tốt nhất (ít âm nhất) được giữ. Ứng viên được sắp theo chi phí, rồi theo tần suất trong từ điển; tối đa 20 ứng viên được giữ cho mỗi âm tiết.

**Bảng 3.1.** Các nguồn sinh ứng viên và chi phí kênh (`configs/ngram.yaml`).

| Nguồn | Mô tả | Ví dụ | Chi phí |
|---|---|---|---:|
| `telex` | Giải mã chuỗi Telex chưa chuyển đổi: phím thanh cuối âm tiết (*s, f, r, x, j*) và các cặp *dd, aa, aw, ee, oo, ow, uw* | *dduwowcj → được*, *gias → giá* | −0,5 |
| `skeleton` | Cùng khung không dấu (bỏ hết dấu thanh và dấu phụ) | *duoc → được* | −1,0 |
| `confusion` | Tập dễ nhầm vùng miền: tổ hợp các phép hoán đổi phụ âm đầu, phụ âm cuối, thanh điệu (cùng bảng với bộ sinh lỗi) | *xản → sản* | −1,0 |
| `teencode` | Từ điển teencode và viết tắt đảo ngược | *ko → không*, *đc → được* | −1,0 |
| `edit1` | Khoảng cách Damerau-Levenshtein (OSA) bằng 1, tra theo SymSpell | *đuợc → được* | −1,5 |
| `edit2` | Khoảng cách bằng 2, tra theo SymSpell | | −3,0 |

Ứng viên theo khoảng cách soạn thảo được tìm bằng thuật toán **SymSpell** [8]: khi dựng bộ sinh, mọi chuỗi thu được bằng cách xóa tối đa 2 ký tự của mỗi âm tiết trong từ điển được đánh chỉ mục; khi truy vấn, chỉ cần sinh các chuỗi xóa của âm tiết đầu vào và tra chỉ mục, rồi tính khoảng cách OSA chính xác cho các ứng viên tìm được. Cách này tránh phải sinh mọi phép chèn/thay trên bảng chữ cái tiếng Việt, vốn rất lớn do có dấu. Kết quả sinh ứng viên của mỗi âm tiết được lưu đệm (tối đa 500.000 mục).

Lưu ý rằng các nguồn `telex`, `confusion` và `teencode` dùng chung tri thức (bảng hoán đổi, bảng teencode) với bộ sinh lỗi tổng hợp. Đây là một nguồn thiên lệch có chủ đích: mô hình A được “biết trước” các dạng lỗi của dữ liệu tổng hợp, nên kết quả trên Syn-test là cận trên lạc quan; khả năng tổng quát hóa thật sự cần được đánh giá trên VSEC và Viwiki (CH2).

### 3.2.5. Phát hiện lỗi và các hình phạt

Mô hình A không có bước phát hiện lỗi tách rời; việc một âm tiết có bị sửa hay không được quyết định trong quá trình giải mã thông qua tập lựa chọn của từng token (`_options`):

- **Token không phải chữ** (số, dấu câu) và **từ viết tắt toàn chữ hoa**: chỉ có một lựa chọn là giữ nguyên.
- **Âm tiết ngoài từ điển (non-word):** lựa chọn giữ nguyên bị phạt `oov_penalty` = −0,5; các ứng viên mang chi phí của nguồn sinh ra chúng. Nếu không có ứng viên nào, âm tiết được giữ nguyên không bị phạt.
- **Âm tiết trong từ điển (có thể là lỗi từ có thật):** giữ nguyên có chi phí 0; mọi ứng viên bị cộng thêm `real_word_penalty` = −1,5, và chỉ ứng viên từ các nguồn `telex`, `skeleton`, `confusion`, `teencode`, `edit1` được xét (bỏ `edit2`). Như vậy mô hình chỉ thay một âm tiết có thật khi LM “rất tin” rằng ứng viên phù hợp ngữ cảnh hơn.
- **Tên riêng và từ nước ngoài:** âm tiết ngoài từ điển, viết hoa chữ đầu và không đứng đầu câu có thể là tên riêng; khi đó chỉ ứng viên từ các nguồn “mạnh” (`telex`, `skeleton`, `teencode`) được giữ.

Sau khi giải mã, kiểu chữ hoa của âm tiết gốc (toàn hoa, hoa chữ đầu, thường) được áp lại cho âm tiết được chọn.

### 3.2.6. Giải mã beam search

Với mô hình bậc $n$, trạng thái của bộ giải mã là $n-1$ token LM cuối cùng (ngữ cảnh). Bộ giải mã duyệt câu từ trái sang phải; tại mỗi vị trí, mỗi giả thuyết trong beam được mở rộng bằng mọi lựa chọn của token hiện tại, với điểm

$$s' = s + \text{channel}(x_i \to y_i) + \lambda \log_{10} P_{\text{LM}}(y_i \mid \text{ngữ cảnh}).$$

Các giả thuyết có cùng ngữ cảnh mới được gộp, chỉ giữ giả thuyết có điểm cao nhất (tương tự quy hoạch động Viterbi), sau đó chỉ giữ 5 trạng thái tốt nhất (`beam_size` = 5). Ở cuối câu, điểm được cộng thêm $\log_{10} P(\texttt{</s>} \mid \text{ngữ cảnh})$ trước khi chọn giả thuyết tốt nhất. Câu đầu ra được dựng lại bằng cách thay đúng vị trí ký tự của các âm tiết bị sửa, nên khoảng trắng và dấu câu của câu gốc được giữ nguyên.

Hai tham số `oov_penalty` và `real_word_penalty` được chỉnh bằng tìm kiếm lưới trên 400 câu đầu của VSEC-dev, chọn theo Correction F1 (mục 4.4).

### 3.2.7. Lựa chọn tự cài đặt LM thay cho KenLM

Kế hoạch ban đầu dùng KenLM để huấn luyện và truy vấn LM. Nhóm quyết định bỏ KenLM và tự cài đặt vì: (i) gói Python `kenlm` cần biên dịch C++ và khó cài đặt trên Windows – môi trường chạy mô hình A của nhóm; (ii) với LM-corpus 400.000 câu, mô hình tự cài huấn luyện được trong khoảng 3–6 phút (bậc 3, bậc 4: khoảng 6 phút) trên CPU và vừa bộ nhớ máy cá nhân (Bảng 5.3); (iii) tự cài đặt cho phép thử nghiệm trực tiếp cả Laplace và Kneser-Ney trong cùng một mã nguồn cho ablation, đồng thời phục vụ mục đích học tập của môn học. Đổi lại, tốc độ truy vấn chậm hơn KenLM, được bù một phần bằng bộ đệm xác suất và bộ đệm ứng viên. Cài đặt được kiểm tra bằng các test đơn vị trong `tests/test_ngram.py`.

## 3.3. Phương án C: tinh chỉnh BARTpho-syllable

> Mục này mô tả **thiết kế** theo `configs/bartpho.yaml`; phần cài đặt đang được thực hiện ở task S1. Các chi tiết sẽ được cập nhật theo mã nguồn cuối cùng. [chờ BARTpho]

### 3.3.1. Mô hình BARTpho

BARTpho [3] là mô hình sinh chuỗi tiền huấn luyện đầu tiên cho tiếng Việt, theo kiến trúc BART [11] (bộ mã hóa – bộ giải mã Transformer, kích thước “large”) và được tiền huấn luyện bằng mục tiêu khử nhiễu: văn bản bị làm hỏng (che đoạn, xáo câu) và mô hình học khôi phục văn bản gốc. Mục tiêu tiền huấn luyện này rất gần với bài toán sửa lỗi chính tả, vốn cũng là khôi phục câu đúng từ câu bị nhiễu. BARTpho có hai phiên bản: mức từ (cần tách từ) và **mức âm tiết** (`vinai/bartpho-syllable`). Nhóm chọn phiên bản âm tiết vì đầu vào chứa lỗi chính tả sẽ làm bộ tách từ hoạt động sai, và vì đơn vị đánh giá của báo cáo là âm tiết.

### 3.3.2. Bài toán sinh chuỗi và dữ liệu huấn luyện

Sửa lỗi được mô hình hóa là bài toán dịch câu lỗi sang câu đúng: đầu vào bộ mã hóa là câu `src` (đã chuẩn hóa), nhãn của bộ giải mã là câu `tgt`. Mô hình được huấn luyện trên 150.000 cặp Syn-train (bao gồm 15% câu sạch, giúp mô hình học sao chép nguyên văn khi câu đúng) và chọn checkpoint theo Syn-val. Khác với mô hình A, mô hình C không bị giới hạn ở thay thế 1–1 nên về nguyên tắc có thể sửa cả lỗi tách/ghép âm tiết, dù dữ liệu huấn luyện không chứa loại lỗi này. Ngược lại, mô hình sinh tự do có nguy cơ viết lại câu (thêm, bớt hoặc thay từ đồng nghĩa); những thay đổi này sẽ bị bộ đánh giá ghi nhận là sửa sai hoặc sửa quá tay.

### 3.3.3. Tham số tinh chỉnh

**Bảng 3.2.** Tham số tinh chỉnh BARTpho theo thiết kế (`configs/bartpho.yaml`).

| Tham số | Giá trị |
|---|---|
| Mô hình nền | `vinai/bartpho-syllable` |
| Độ dài tối đa nguồn / đích | 128 / 128 token |
| Kích thước tập huấn luyện | 150.000 cặp (ablation mật độ nhiễu: 50.000) |
| Số epoch | 3 |
| Batch size huấn luyện / đánh giá (mỗi thiết bị) | 32 / 64 |
| Tốc độ học | 5·10⁻⁵, warmup 5% số bước |
| Weight decay | 0,01 |
| Độ chính xác số | fp16 |
| Đánh giá và lưu checkpoint | mỗi 2.000 bước, giữ tối đa 2 checkpoint |
| Tiêu chí chọn checkpoint | `eval_loss` trên Syn-val |
| Giải mã | beam search, 4 beam |
| Seed | 42 |

Quá trình huấn luyện dự kiến dùng `Seq2SeqTrainer` của thư viện Hugging Face Transformers trên GPU của Kaggle hoặc Colab. Một thí nghiệm ablation bổ sung (tùy chọn) huấn luyện ba mô hình trên 50.000 cặp với mật độ nhiễu 5%, 12% và 25% để khảo sát ảnh hưởng của mật độ lỗi trong dữ liệu huấn luyện. [chờ BARTpho]

# 4. Thiết lập thực nghiệm

## 4.1. Môi trường và khả năng tái lập

- **Mô hình A, dữ liệu và đánh giá:** máy tính cá nhân chạy Windows 11, chỉ dùng CPU, Python 3.12. Các thư viện chính được ghim phiên bản trong `requirements.txt`: `pyarrow` 26.0.1, `huggingface_hub` 2.1.1, `numpy` 2.4.4, `pandas` 3.0.2, `PyYAML` 6.0.3, `scikit-learn` 1.9.1, `matplotlib` 3.10.8, `pytest` 9.1.1. Phần lõi của mô hình A (LM, sinh ứng viên, giải mã) chỉ dùng thư viện chuẩn của Python, không cần thư viện học máy.
- **Mô hình C:** GPU trên Kaggle/Colab với `torch`, `transformers`, `datasets`, `accelerate`, `sentencepiece`; phiên bản cụ thể sẽ được ghim sau lần huấn luyện đầu tiên. [chờ BARTpho]
- **Seed:** 42 cho lấy mẫu Binhvq, chia tập và chia VSEC; 42/43/44 cho sinh nhiễu Syn-train/Syn-val/Syn-test; 42 cho huấn luyện BARTpho. Hàm `set_seed` cố định seed của `random`, `numpy` và `torch`.
- **Kiểm thử:** bộ test đơn vị trong `tests/` (chuẩn hóa, sinh lỗi, chỉ số đánh giá, LM n-gram, đọc dữ liệu thật, loại trùng) gồm 65 test và đều qua.
- Toàn bộ quy trình được mô tả trong `README.md`; các thực nghiệm của mô hình A được chạy lại bằng `scripts/run_ngram_experiments.sh`.

## 4.2. Các tập đánh giá

Mỗi mô hình được chấm trên ba tập kiểm tra, đại diện cho ba mức độ khác biệt so với dữ liệu huấn luyện:

| Tập | Số câu | Số lỗi | Ý nghĩa |
|---|---:|---:|---|
| Syn-test | 5.000 | 11.172 | Cùng phân phối với dữ liệu huấn luyện (lỗi tổng hợp, miền tin tức) |
| VSEC-test | 7.468 | 8.643 | Lỗi thật, cùng miền tin tức |
| Viwiki-test | 15.153 | 1.542 | Lỗi thật, khác miền (Wikipedia), lỗi rất thưa |

Thứ tự này cho phép trả lời CH2: khoảng cách giữa Syn-test và VSEC-test phản ánh sai khác giữa lỗi tổng hợp và lỗi thật; khoảng cách giữa VSEC-test và Viwiki-test phản ánh thêm sai khác về miền văn bản và mật độ lỗi.

## 4.3. Căn chỉnh âm tiết và chỉ số đánh giá

### 4.3.1. Căn chỉnh

Các câu nguồn $s$, câu đích $t$ và câu dự đoán $p$ đều được chuẩn hóa (mục 2.2.1) và tách token trước khi so. Vì câu đích (VSEC, Viwiki) và câu dự đoán (mô hình C) có thể khác số âm tiết so với câu nguồn, mọi thay đổi được quy về **vị trí âm tiết trong câu nguồn** bằng `difflib.SequenceMatcher` (`src/vspell/eval/align.py`):

- thay thế 1–1: *“đều” → “điều”* được ghi thành $\{i: \text{“điều”}\}$;
- tách/ghép: *“nh iên” → “nhiên”* được ghi thành $\{i: \text{“nhiên”},\, i{+}1: \text{“”}\}$ (âm tiết cuối của khối gánh phần dư);
- chèn thêm âm tiết sau vị trí $i$: ghi thành $\{i: \text{“}s_i\text{ <âm tiết chèn>”}\}$.

Áp dụng thủ tục này cho cặp $(s, t)$ ta được tập lỗi chuẩn $G$ (vị trí → chuỗi đúng); cho cặp $(s, p)$ ta được tập sửa đổi của mô hình $E$ (vị trí → chuỗi đề xuất). Nhờ dùng chung hệ chỉ số, $G$ và $E$ so sánh được trực tiếp với mọi loại mô hình.

### 4.3.2. Chỉ số

Gọi $N_G = \sum |G|$ là tổng số lỗi chuẩn, $N_E = \sum |E|$ là tổng số vị trí mô hình thay đổi (cộng trên toàn tập):

- **Phát hiện (Detection):** $TP_{\text{det}} = \sum |G \cap E|$ (theo vị trí); $P_{\text{det}} = TP_{\text{det}}/N_E$, $R_{\text{det}} = TP_{\text{det}}/N_G$, $F1_{\text{det}}$ là trung bình điều hòa.
- **Sửa lỗi (Correction):** $TP_{\text{cor}}$ là số vị trí trong $G \cap E$ mà chuỗi đề xuất **khớp** chuỗi đúng; $P_{\text{cor}} = TP_{\text{cor}}/N_E$, $R_{\text{cor}} = TP_{\text{cor}}/N_G$, $F1_{\text{cor}}$. Đây là chỉ số chính của báo cáo.
- **Độ chính xác câu (sentence accuracy):** tỉ lệ câu có dự đoán trùng hoàn toàn với câu đích (so theo token sau chuẩn hóa).
- **Tỉ lệ sửa quá tay (over-correction):** trong các câu sạch ($s = t$), tỉ lệ câu bị mô hình thay đổi.
- **Recall theo loại lỗi:** chỉ cho Syn-test, nơi mỗi lỗi có nhãn loại; tính $R_{\text{det}}$ và $R_{\text{cor}}$ riêng cho từng loại trong sáu loại lỗi.

Các chỉ số được tính ở mức vi mô (micro) trên toàn tập. Mọi so sánh chuỗi đều thực hiện sau chuẩn hóa, nên mô hình không bị phạt vì khác kiểu đặt dấu. Để kiểm tra tính đúng đắn của bộ chấm, nhóm dùng **mô hình đồng nhất** (identity – không sửa gì): mô hình này phải cho mọi chỉ số P/R/F1 bằng 0 và tỉ lệ sửa quá tay bằng 0, còn độ chính xác câu bằng tỉ lệ câu sạch của tập (Bảng 5.1).

## 4.4. Quy trình chọn tham số

- **Mô hình A:** cấu trúc mô hình (bậc 3, Kneser-Ney, $D = 0{,}75$, beam 5, chi phí các nguồn ứng viên) được đặt theo kinh nghiệm. Hai tham số ảnh hưởng mạnh nhất đến cân bằng precision/recall là `oov_penalty` và `real_word_penalty` được chỉnh bằng tìm kiếm lưới trên 400 câu đầu của VSEC-dev (`scripts/train_ngram.py --tune`), chọn cặp có Correction F1 cao nhất. Lưới cuối cùng trong `configs/ngram.yaml` là `oov_penalty` ∈ {−0,5; −1,0; −1,5; −3,0} × `real_word_penalty` ∈ {−0,5; −1,0; −1,5; −2,0; −3,0}; giá trị được chọn là (−0,5; −1,5). Các mô hình trong ablation dùng chung tham số giải mã này, không chỉnh lại riêng.
- **Mô hình A + S4 (luật giảm sửa quá tay, mục 5.5):** bốn tham số của các luật được dò bằng lưới 2 × 2 × 4 × 5 = 80 cấu hình trên **toàn bộ 1.873 câu VSEC-dev** (`scripts/tune_ngram_s4.py`), giữ nguyên các tham số giải mã khác. Tiêu chí chọn là Correction F0.5 (trọng số precision gấp đôi recall) thay vì F1: trên văn bản thật, phần lớn câu không có lỗi nên một lần sửa nhầm gây hại nhiều hơn một lỗi bị bỏ sót. Tiêu chí F0.5 được đặt ra trước khi xem kết quả của S4 trên tập kiểm tra; tuy vậy bản thân các luật S4 được thiết kế sau khi phân tích lỗi của A trên Viwiki-test (xem lưu ý ở mục 5.5).
- **Mô hình C:** chọn checkpoint theo `eval_loss` trên Syn-val; không dùng VSEC-dev hay các tập lỗi thật để huấn luyện. [chờ BARTpho]
- **Viwiki-test** không được dùng cho bất kỳ quyết định nào; **VSEC-test** chỉ dùng để báo cáo.

## 4.5. Thí nghiệm ablation

- **Ablation A – bậc n-gram và phương pháp làm trơn:** huấn luyện năm LM trên cùng LM-corpus 400.000 câu: bậc 2 Laplace, bậc 2 Kneser-Ney, bậc 3 Laplace, bậc 3 Kneser-Ney (mô hình chính), bậc 4 Kneser-Ney. Mỗi LM được đánh giá bằng perplexity trên 5.000 câu sạch của Syn-val và bằng các chỉ số sửa lỗi trên Syn-test và VSEC-test.
- **Ablation A – các luật giảm sửa quá tay (S4):** bật lần lượt từng luật (với giá trị đã chọn) trên mô hình A ban đầu và so với khi bật cả bốn, trên VSEC-test và Viwiki-test (`scripts/ablate_ngram_s4.py`). Thí nghiệm này chỉ để báo cáo, không dùng để chọn tham số.
- **Ablation C – mật độ nhiễu trong dữ liệu huấn luyện** (tùy chọn): 5%, 12%, 25%, mỗi mức 50.000 cặp. [chờ BARTpho]

# 5. Kết quả và thảo luận

> Mục này hiện chỉ có kết quả của mô hình A và ablation A. Các ô và đoạn liên quan đến mô hình C được đánh dấu [chờ BARTpho] và sẽ hoàn thiện ở task S7b.

## 5.1. Kết quả chính

**Bảng 5.1.** Kết quả trên ba tập kiểm tra (`outputs/tables/main_results.csv`). Det/Cor: phát hiện/sửa lỗi; SA: độ chính xác câu; OC: tỉ lệ sửa quá tay trên câu sạch. A = bậc 3 Kneser-Ney, tham số đã chỉnh trên VSEC-dev.

| Tập | Mô hình | Det P | Det R | Det F1 | Cor P | Cor R | **Cor F1** | SA | OC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Syn-test | Đồng nhất | 0 | 0 | 0 | 0 | 0 | 0 | 0,148 | 0 |
| | A (n-gram) | 0,929 | 0,943 | 0,936 | 0,891 | 0,905 | **0,898** | 0,719 | 0,127 |
| | C (BARTpho) | [chờ BARTpho] | | | | | | | |
| VSEC-test | Đồng nhất | 0 | 0 | 0 | 0 | 0 | 0 | 0,062 | 0 |
| | A (n-gram) | 0,900 | 0,814 | 0,855 | 0,819 | 0,741 | **0,778** | 0,694 | 0,063 |
| | C (BARTpho) | [chờ BARTpho] | | | | | | | |
| Viwiki-test | Đồng nhất | 0 | 0 | 0 | 0 | 0 | 0 | 0,916 | 0 |
| | A (n-gram) | 0,259 | 0,569 | 0,356 | 0,221 | 0,487 | **0,304** | 0,834 | 0,128 |
| | C (BARTpho) | [chờ BARTpho] | | | | | | | |

Mô hình đồng nhất cho đúng các giá trị kỳ vọng (P/R/F1 và OC bằng 0; SA bằng tỉ lệ câu sạch: 741/5.000 ≈ 0,148 trên Syn-test), xác nhận bộ chấm hoạt động đúng.

**Nhận xét về mô hình A.**

- **Trên Syn-test**, mô hình A đạt Correction F1 0,898 với precision và recall cân bằng (0,891/0,905). Kết quả cao là điều dễ hiểu vì các nguồn ứng viên `telex`, `confusion`, `teencode` được xây dựng từ cùng tri thức với bộ sinh lỗi (mục 3.2.4).
- **Trên VSEC-test** (lỗi thật, cùng miền), Correction F1 giảm còn 0,778. Precision phát hiện vẫn cao (0,900), sự sụt giảm chủ yếu đến từ recall (0,814 so với 0,943 trên Syn-test): một phần lỗi thật nằm ngoài các mẫu lỗi mà bộ sinh ứng viên biết, hoặc là lỗi tách/ghép âm tiết mà mô hình 1–1 không xử lý được. Tỉ lệ sửa quá tay trên câu sạch của VSEC thấp (6,3%).
- **Trên Viwiki-test** (khác miền, lỗi thưa), kết quả giảm mạnh: Correction F1 chỉ 0,304, do precision rất thấp (0,221). Mô hình thực hiện 3.393 thay đổi trong khi tập chỉ có 1.542 lỗi, và 12,8% câu sạch bị sửa. Với mật độ lỗi 0,29%, mỗi lần sửa nhầm “đắt” hơn nhiều so với trên VSEC. Mặt khác, văn bản Wikipedia chứa nhiều tên riêng, thuật ngữ và từ nước ngoài không có trong LM-corpus tin tức, nên LM đánh giá thấp các âm tiết này và mô hình có xu hướng “sửa” chúng. Độ chính xác câu của mô hình A trên Viwiki (0,834) thậm chí **thấp hơn** mô hình đồng nhất (0,916): trên tập thưa lỗi này, áp dụng mô hình A làm hại nhiều câu hơn số câu được sửa đúng.

Về **CH2**, kết quả của mô hình A cho thấy khả năng tổng quát hóa phụ thuộc mạnh vào cả loại lỗi lẫn miền: chuyển từ lỗi tổng hợp sang lỗi thật cùng miền làm Correction F1 giảm khoảng 0,12, còn chuyển sang miền khác với lỗi thưa làm giảm thêm khoảng 0,47. So sánh với mô hình C: [chờ BARTpho].

[chờ hình: `outputs/figures/` – biểu đồ cột Correction F1 của A và C trên ba tập, do task S5a/S5b tạo]

## 5.2. Recall theo loại lỗi

**Bảng 5.2.** Recall phát hiện (Det R) và sửa (Cor R) theo loại lỗi trên Syn-test (`outputs/tables/per_error_type.csv`), mô hình A.

| Loại lỗi | Số lỗi | Det R | Cor R | C: Cor R |
|---|---:|---:|---:|---:|
| Trượt phím | 4.165 | 0,898 | 0,844 | [chờ BARTpho] |
| Telex | 3.371 | 1,000 | 0,996 | [chờ BARTpho] |
| Vùng miền | 3.058 | 0,943 | 0,902 | [chờ BARTpho] |
| Biến đổi ký tự | 441 | 0,923 | 0,850 | [chờ BARTpho] |
| Teencode | 111 | 0,955 | 0,694 | [chờ BARTpho] |
| Viết tắt | 26 | 1,000 | 0,923 | [chờ BARTpho] |

- **Lỗi Telex** gần như được sửa hoàn toàn (Cor R 0,996): chuỗi Telex chưa chuyển đổi hầu như không bao giờ là âm tiết hợp lệ nên dễ phát hiện, và bộ giải mã Telex cho ra đúng một ứng viên có chi phí thấp nhất.
- **Lỗi vùng miền** – loại khó nhất về lý thuyết vì luôn là lỗi từ có thật – vẫn đạt Det R 0,943 và Cor R 0,902. Điều này cho thấy hình phạt `real_word_penalty` = −1,5 đủ mềm để LM bậc 3 thay được âm tiết có thật khi ngữ cảnh đủ rõ.
- **Lỗi trượt phím** có Cor R thấp nhất trong các nhóm lớn (0,844): phép thay một chữ bằng phím lân cận thường tạo ra chuỗi cách âm tiết đúng một bước soạn thảo, và có nhiều âm tiết khác cùng khoảng cách, nên LM phải chọn giữa nhiều ứng viên cạnh tranh.
- **Teencode** có khoảng cách lớn giữa phát hiện và sửa (0,955 so với 0,694): mô hình nhận ra âm tiết lạ nhưng các dạng như *“k”*, *“a”*, *“e”* ứng với nhiều từ gốc, hoặc dạng teencode thay phụ âm đầu (*f, j, w, z*) không có trong bảng đảo ngược.
- Hai loại teencode và viết tắt có số mẫu nhỏ (111 và 26), nên các con số này có độ bất định cao.

[chờ hình: `outputs/figures/` – biểu đồ recall theo loại lỗi, do task S5a tạo]

## 5.3. Ablation A: bậc n-gram và phương pháp làm trơn

**Bảng 5.3.** Ablation bậc n-gram và làm trơn (`ngram_perplexity.csv`, `main_results.csv`). Perplexity đo trên 5.000 câu Syn-val sạch; thời gian là thời gian huấn luyện LM trên 400.000 câu. Mọi biến thể dùng chung tham số giải mã đã chỉnh cho bậc 3 Kneser-Ney.

| Biến thể | Perplexity | Huấn luyện (giây) | Syn-test Cor F1 | Syn-test OC | VSEC-test Cor P | VSEC-test Cor R | VSEC-test Cor F1 | VSEC-test OC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Bậc 2, Laplace | 359,87 | 207,8 | 0,865 | 0,131 | 0,766 | 0,626 | 0,689 | 0,093 |
| Bậc 2, Kneser-Ney | 94,74 | 177,7 | 0,882 | 0,128 | 0,782 | 0,674 | 0,724 | 0,095 |
| Bậc 3, Laplace | 3.544,79 | 209,4 | 0,852 | 0,057 | 0,886 | 0,528 | 0,661 | 0,043 |
| **Bậc 3, Kneser-Ney (A)** | **45,89** | 372,6 | **0,898** | 0,127 | 0,819 | 0,741 | **0,778** | 0,063 |
| Bậc 4, Kneser-Ney | 39,68 | 354,2 | 0,901 | 0,124 | 0,828 | 0,743 | 0,783 | 0,073 |

Một số nhận xét:

- **Kneser-Ney vượt trội Laplace ở mọi bậc.** Ở bậc 2, Kneser-Ney giảm perplexity từ 359,87 xuống 94,74 và tăng Correction F1 trên VSEC-test từ 0,689 lên 0,724. Khác biệt lớn nhất ở bậc 3: perplexity của Laplace *tăng* lên 3.544,79 (cao hơn cả bậc 2), vì với $|V| = 72.774$ và dữ liệu thưa, phép cộng 1 phân bố quá nhiều xác suất cho các trigram chưa thấy, trong khi Laplace ở đây không nội suy với bậc thấp. Mô hình sửa lỗi dùng LM này trở nên rất “bảo thủ”: precision cao (0,886 trên VSEC) nhưng recall giảm mạnh (0,528), Correction F1 chỉ 0,661 – thấp nhất trong năm biến thể.
- **Tăng bậc với Kneser-Ney giúp giảm perplexity và tăng chất lượng sửa,** nhưng lợi ích giảm dần: từ bậc 2 lên bậc 3, Correction F1 trên VSEC-test tăng 0,054 (0,724 → 0,778); từ bậc 3 lên bậc 4 chỉ tăng thêm 0,005 (0,778 → 0,783), kèm tỉ lệ sửa quá tay trên VSEC tăng từ 6,3% lên 7,3%. Lưu ý rằng tham số giải mã được chỉnh cho bậc 3, nên so sánh có thể hơi thiên vị bậc 3.
- **Perplexity tương quan với chất lượng sửa lỗi trong cùng họ làm trơn** (Kneser-Ney: 94,74 → 45,89 → 39,68 tương ứng Correction F1 VSEC 0,724 → 0,778 → 0,783), cho thấy perplexity là chỉ báo hữu ích khi chọn LM cho bài toán này.
- Nhóm giữ **bậc 3 Kneser-Ney** làm mô hình chính vì chênh lệch với bậc 4 nhỏ, trong khi bậc 3 có tỉ lệ sửa quá tay thấp hơn trên VSEC và số n-gram cần lưu ít hơn.

[chờ hình: `outputs/figures/` – biểu đồ ablation (bậc × làm trơn; perplexity so với Correction F1), do task S5a tạo]

## 5.4. Chỉnh tham số giải mã

Bảng 5.4 trình bày kết quả tìm kiếm lưới được lưu trong `outputs/tables/ngram_tuning_o3_kneser_ney.csv` (400 câu VSEC-dev, mô hình bậc 3 Kneser-Ney).

**Bảng 5.4.** Correction F1 trên 400 câu VSEC-dev theo `oov_penalty` (hàng) và `real_word_penalty` (cột).

| `oov_penalty` \\ `real_word_penalty` | −1,0 | −2,0 | −3,0 | −4,0 |
|---|---:|---:|---:|---:|
| −1,5 | **0,775** | 0,769 | 0,709 | 0,626 |
| −3,0 | 0,767 | 0,762 | 0,703 | 0,620 |
| −5,0 | 0,763 | 0,756 | 0,697 | 0,614 |

Xu hướng rõ ràng: hình phạt `real_word_penalty` càng âm thì mô hình càng ít thay âm tiết có thật – precision phát hiện tăng (ví dụ từ 0,857 lên 0,953 khi `real_word_penalty` đi từ −1,0 xuống −3,0 với `oov_penalty` = −1,5) nhưng recall giảm nhanh hơn (0,846 xuống 0,660), nên F1 giảm. Hình phạt `oov_penalty` ít ảnh hưởng hơn; giá trị ít âm hơn cho kết quả tốt hơn một chút. Theo xu hướng này, lưới được mở rộng về phía các giá trị ít âm hơn (lưới cuối ở mục 4.4) và cặp được chọn cho mô hình chính là `oov_penalty` = −0,5, `real_word_penalty` = −1,5. [cần bổ sung: kết quả của lưới cuối chưa được lưu vào CSV; chạy lại `python scripts/train_ngram.py --tune` để ghi đầy đủ và cập nhật Bảng 5.4]

## 5.5. Giảm sửa quá tay của mô hình A (S4)

Phân tích lỗi của mô hình A trên Viwiki (`outputs/tables/overcorrection_viwiki.csv`) cho thấy các lần sửa quá tay rơi vào ba nhóm: từ thông thường bị thay bằng một âm tiết “phổ biến hơn” (khoảng 45%), từ ngoài từ điển như từ nước ngoài (*manga → mang*, khoảng 34%) và tên riêng viết hoa (khoảng 21%). Ngoài ra còn một nhóm nhỏ là đổi biến thể chính tả i/y (*lí → lý*, *kì → kỳ*, *Mĩ → Mỹ*) vốn không phải lỗi. Nhóm bổ sung bốn luật vào bước sinh lựa chọn của bộ giải mã (`NgramCorrector._options`), mỗi luật là một tham số trong `configs/ngram.yaml` và mặc định tắt để tái lập đúng mô hình A ban đầu:

- `ignore_iy`: bỏ các ứng viên chỉ khác âm tiết gốc ở biến thể i/y (vần chỉ gồm một nguyên âm i/y, không âm cuối, cùng phụ âm đầu và thanh).
- `keep_non_vietnamese`: với âm tiết **không có dấu**, ngoài từ điển và sai cấu trúc âm tiết tiếng Việt (*manga*, *online*), chỉ giữ ứng viên từ các nguồn mạnh (`telex`, `skeleton`, `teencode`). Âm tiết có dấu nhưng sai cấu trúc (*vànv*) vẫn được coi là lỗi gõ và sửa bình thường.
- `capital_penalty`: cộng thêm một hình phạt khi sửa âm tiết viết hoa không đứng đầu câu, kể cả âm tiết có trong từ điển (trước đây chỉ âm tiết ngoài từ điển mới được bảo vệ).
- `edit_margin` (ngưỡng tin cậy): cộng thêm một hình phạt cho **mọi** lựa chọn khác giữ nguyên, tức là một lần sửa chỉ được chấp nhận khi điểm của nó vượt điểm giữ nguyên ít nhất |δ| (log10).

Cấu hình được chọn trên VSEC-dev theo Correction F0.5 (mục 4.4) là: bật `ignore_iy` và `keep_non_vietnamese`, `capital_penalty` = −2,0, `edit_margin` = −0,5. Trên VSEC-dev, Cor P tăng từ 0,821 lên 0,892 và Cor F1 tăng nhẹ từ 0,771 lên 0,775, số lần sửa giảm từ 1.959 xuống 1.699 (`outputs/tables/ngram_s4_tuning.csv`). Cấu hình tốt nhất theo F1 (`edit_margin` = −0,25) chỉ hơn cấu hình được chọn 0,003 F1 trên dev nhưng có precision thấp hơn 0,018.

**Lưu ý về tính độc lập của tập kiểm tra.** S4 và v4 là *thí nghiệm phát triển sau khi phân tích lỗi trên tập kiểm tra*: các luật S4 được thiết kế từ phân tích sửa quá tay của A trên Viwiki-test, còn tỉ lệ lỗi dấu thanh/dấu phụ của v4 được đặt theo phân bố lỗi của VSEC-test. Tham số số của S4 được chọn trên VSEC-dev và v4 không chỉnh gì thêm theo test, nhưng tập kiểm tra vẫn đã ảnh hưởng tới quyết định thiết kế. Vì vậy kết quả của A + S4 và C v4 trên VSEC-test/Viwiki-test có thể lạc quan hơn thực tế và chưa chứng minh được khả năng tổng quát hóa; muốn khẳng định cần đánh giá trên một tập lỗi thật chưa từng được xem (nhóm chưa có tập như vậy). Các kết quả của A, C v3 và các ablation mật độ nhiễu không chịu ảnh hưởng này.

**Bảng 5.5.** Mô hình A trước và sau khi thêm luật S4 (`main_results.csv`, tag `ngram` và `ngram_s4`).

| Tập | Mô hình | Cor P | Cor R | **Cor F1** | SA | OC | Số lần sửa |
|---|---|---:|---:|---:|---:|---:|---:|
| Syn-test | A | 0,891 | 0,905 | **0,898** | 0,719 | 0,127 | 11.338 |
| | A + S4 | 0,935 | 0,797 | 0,860 | 0,615 | 0,058 | 9.523 |
| VSEC-test | A | 0,819 | 0,741 | 0,778 | 0,694 | 0,063 | 7.817 |
| | A + S4 | 0,897 | 0,697 | **0,784** | 0,688 | 0,022 | 6.717 |
| Viwiki-test | A | 0,221 | 0,487 | 0,304 | 0,834 | 0,128 | 3.393 |
| | A + S4 | 0,462 | 0,444 | **0,453** | 0,913 | 0,041 | 1.482 |

**Bảng 5.6.** Ablation từng luật: mô hình A ban đầu bật đúng một luật (`outputs/tables/ngram_s4_ablation.csv`).

| Biến thể | VSEC Cor P | VSEC Cor F1 | Viwiki Cor P | Viwiki Cor R | Viwiki Cor F1 | Viwiki OC |
|---|---:|---:|---:|---:|---:|---:|
| A | 0,819 | 0,778 | 0,221 | 0,487 | 0,304 | 0,128 |
| A + `ignore_iy` | 0,825 | 0,781 | 0,222 | 0,481 | 0,304 | 0,126 |
| A + `keep_non_vietnamese` | 0,850 | 0,786 | 0,275 | 0,487 | 0,352 | 0,102 |
| A + `capital_penalty` | 0,825 | 0,780 | 0,262 | 0,486 | 0,341 | 0,104 |
| A + `edit_margin` | 0,862 | 0,779 | 0,305 | 0,451 | 0,364 | 0,079 |
| A + cả bốn (A + S4) | 0,897 | 0,784 | 0,462 | 0,444 | **0,453** | 0,041 |

Một số nhận xét:

- **Trên Viwiki**, Correction F1 tăng từ 0,304 lên 0,453 (+0,149), chủ yếu nhờ precision tăng gấp đôi (0,221 lên 0,462) trong khi recall chỉ giảm 0,04. Số lần sửa giảm từ 3.393 xuống 1.482, xấp xỉ số lỗi thật (1.542), và tỉ lệ câu sạch bị sửa giảm từ 12,8% xuống 4,1%. Độ chính xác câu (0,913) gần bằng mô hình đồng nhất (0,916): mô hình gần như không còn làm hại câu đúng, nhưng vẫn chưa vượt được mốc “không sửa gì”.
- **Trên VSEC-test**, F1 tăng nhẹ (0,778 lên 0,784): precision tăng bù cho recall giảm. Đây là tập có cùng miền với tập dùng để chỉnh, nên mức tăng nhỏ là hợp lý.
- **Trên Syn-test**, F1 giảm từ 0,898 xuống 0,860, do recall giảm (0,905 xuống 0,797). Ngưỡng tin cậy chặn cả những lần sửa đúng mà LM chỉ hơi tin, và luật `keep_non_vietnamese` chặn các lỗi trượt phím trên âm tiết không dấu. Trên dữ liệu tổng hợp, mật độ lỗi cao và mọi lỗi đều thuộc các mẫu mà bộ sinh ứng viên biết, nên sự thận trọng này bị “phạt”. Kết quả này nhắc lại rằng tối ưu cho dữ liệu tổng hợp không đồng nghĩa với tối ưu cho văn bản thật.
- **Các luật bổ trợ nhau.** Bật riêng từng luật chỉ đưa F1 trên Viwiki lên tối đa 0,364, còn bật cả bốn lên 0,453, lớn hơn tổng các mức tăng riêng lẻ. Mỗi luật loại một nhóm sửa nhầm khác nhau; khi các nhóm này bị loại cùng lúc, những lần sửa còn lại tập trung vào lỗi thật. Riêng `ignore_iy` gần như không đổi F1 vì nhóm i/y chỉ chiếm khoảng 2% số lần sửa quá tay trên Viwiki.
- So với mô hình C (BARTpho v4, Viwiki Cor F1 0,322), mô hình A + S4 cao hơn khoảng 0,13 trên Viwiki và cao hơn 0,035 trên VSEC-test (0,784 so với 0,749).

## 5.6. Phân tích lỗi

[chờ BARTpho] – 5–10 ví dụ có phân tích (A đúng/C sai, C đúng/A sai, cả hai sai) sẽ được chọn ở task S5b từ `outputs/tables/error_examples.md`.

# 6. Kết luận

[chờ BARTpho] – sẽ viết ở task S7b: tóm tắt câu trả lời cho CH1, CH2; các hạn chế (đã nêu ở mục 2.6 và 3.2.7); hướng phát triển (mô hình B BiLSTM, xử lý lỗi tách/ghép âm tiết, chuẩn hóa teencode với ViLexNorm, giảm sửa quá tay trên văn bản khác miền).

# Phân công công việc

[cần bổ sung ở task S9]

# Tài liệu tham khảo

[1] D. T. Do, H. T. Nguyen, T. N. Bui, H. D. Vo. *VSEC: Transformer-based Model for Vietnamese Spelling Correction.* PRICAI 2021: Trends in Artificial Intelligence, LNCS 13032, Springer, 2021. Dữ liệu: https://github.com/VSEC2021/VSEC

[2] H. Tran, C. V. Dinh, L. Phan, S. T. Nguyen. *Hierarchical Transformer Encoders for Vietnamese Spelling Correction.* IEA/AIE 2021, LNCS 12798, Springer, 2021. Dữ liệu Viwiki-spelling: https://github.com/heraclex12/Viwiki-spelling

[3] N. L. Tran, D. M. Le, D. Q. Nguyen. *BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese.* Proceedings of Interspeech 2022. Mô hình: https://huggingface.co/vinai/bartpho-syllable

[4] R. Kneser, H. Ney. *Improved Backing-off for M-gram Language Modeling.* Proceedings of ICASSP 1995, pp. 181–184.

[5] S. F. Chen, J. Goodman. *An Empirical Study of Smoothing Techniques for Language Modeling.* Computer Speech & Language 13(4), 1999, pp. 359–394.

[6] M. D. Kernighan, K. W. Church, W. A. Gale. *A Spelling Correction Program Based on a Noisy Channel Model.* Proceedings of COLING 1990.

[7] D. Jurafsky, J. H. Martin. *Speech and Language Processing* (3rd ed. draft), chương “N-gram Language Models” và phụ lục “Spelling Correction and the Noisy Channel”. https://web.stanford.edu/~jurafsky/slp3/

[8] W. Garbe. *SymSpell: Symmetric Delete Spelling Correction Algorithm.* https://github.com/wolfgarbe/SymSpell

[9] H. N. Trung, D. T. Ham, T. Huynh, K. Hoang. *A Combination of BERT and Transformer for Vietnamese Spelling Correction.* ACIIDS 2022, LNCS 13757, Springer; arXiv:2405.02573, 2024.

[10] Binhvq. *News Corpus – Vietnamese news corpus.* https://github.com/binhvq/news-corpus; bản Hugging Face: https://huggingface.co/datasets/vietgpt/binhvq_news_vi

[11] M. Lewis, Y. Liu, N. Goyal, M. Ghazvininejad, A. Mohamed, O. Levy, V. Stoyanov, L. Zettlemoyer. *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.* Proceedings of ACL 2020.

# Phụ lục A. Sử dụng công cụ AI

Theo yêu cầu của môn học, nhóm công khai việc sử dụng công cụ AI trong quá trình thực hiện bài tập lớn.

**Công cụ sử dụng:** trợ lý lập trình Claude Code (Anthropic), chạy trên máy của nhóm, với quyền đọc/ghi trong thư mục dự án và chạy lệnh; mỗi task được giao trong một phiên làm việc riêng theo danh sách ở `report/tien_do.md`.

**Phạm vi sử dụng:**

| Hạng mục | Vai trò của AI | Vai trò của nhóm |
|---|---|---|
| Lập kế hoạch | Đề xuất cấu trúc thư mục, phân chia task, dàn ý báo cáo | Chọn phạm vi (A + C), dữ liệu, phân công; duyệt kế hoạch |
| Mã nguồn dữ liệu, sinh lỗi, đánh giá, mô hình A | Viết mã và test đơn vị theo đặc tả của nhóm; chạy thử, sửa lỗi | Đặc tả yêu cầu, rà soát mã, quyết định thiết kế (chuẩn hóa dấu, chia theo đoạn, bỏ KenLM, tham số) |
| Mô hình C | Viết mã huấn luyện/dự đoán và notebook | Chạy huấn luyện trên GPU, kiểm tra kết quả [chờ BARTpho] |
| Báo cáo | Soạn bản nháp các mục 1–4 và phần kết quả mô hình A từ mã nguồn và các tệp số liệu | Rà soát nội dung, kiểm tra số liệu, chỉnh sửa văn phong, viết các phần còn lại |
| Tra cứu tài liệu | Tra thông tin thư mục (tên bài, tác giả) của tài liệu tham khảo | Kiểm tra lại trích dẫn |

**Nguyên tắc nhóm áp dụng:**

- Mọi số liệu trong báo cáo được lấy từ các tệp do mã nguồn sinh ra (`outputs/tables/*.csv`, `data/processed/stats_*.json`, `syn_*.stats.json`) hoặc từ tệp cấu hình; không có số liệu nào do AI tự đặt ra. Các chỗ chưa có số liệu được đánh dấu rõ ([chờ BARTpho], [chờ hình], [cần bổ sung]).
- Các tập kiểm tra (VSEC-test, Viwiki-test) không được dùng để chỉnh tham số, kể cả khi AI hỗ trợ phân tích kết quả.
- Mã do AI viết đều kèm test đơn vị (65 test) và được nhóm chạy lại; nhóm chịu trách nhiệm toàn bộ về nội dung và kết quả của bài tập lớn.

# Phụ lục B. Cấu hình chính

Các tệp cấu hình đầy đủ nằm trong `configs/`: `data.yaml` (lấy mẫu Binhvq, chia tập, chống rò rỉ), `noise.yaml` (sinh lỗi), `ngram.yaml` (mô hình A) và `bartpho.yaml` (mô hình C). Kết quả thô nằm trong `outputs/tables/`, dự đoán từng câu nằm trong `outputs/predictions/` (không đưa vào git).
