# Ví dụ phân tích lỗi (tự sinh bởi `scripts/analyze_errors.py`, seed=7)

Câu được tách token như khi chấm (dấu câu tách riêng), cắt cửa sổ quanh vị trí lỗi.
In **đậm**: ở `src` là vị trí lỗi thật hoặc bị mô hình sửa; ở `pred`/`tgt` là chuỗi thay thế; `∅` là âm tiết bị xóa.

## Mô hình A: n-gram (3-gram KN)

**Ví dụ 1. Viwiki – sửa quá tay tên riêng/viết hoa [viwiki-5829-34]**

- src : Kết quả là Pisco bị **Gin** giết vì thất bại trong nhiệm vụ trước khi kịp nói cho …
- pred: Kết quả là Pisco bị **Gió** giết vì thất bại trong nhiệm vụ trước khi kịp nói cho …
- tgt : Kết quả là Pisco bị Gin giết vì thất bại trong nhiệm vụ trước khi kịp nói cho …

  - Phân loại: Sửa quá tay / Tên riêng/viết hoa: `Gin` → `Gió` (đúng: `Gin`)

**Ví dụ 2. Viwiki – sửa quá tay tên riêng/viết hoa [viwiki-8084-137]**

- src : … Bạch Ốc với Tòa nhà Quốc hội Hoa Kỳ , và Đường **K** có nhiều văn phòng của nhiều nhóm vận động hành lang .
- pred: … Bạch Ốc với Tòa nhà Quốc hội Hoa Kỳ , và Đường **Không** có nhiều văn phòng của nhiều nhóm vận động hành lang .
- tgt : … Bạch Ốc với Tòa nhà Quốc hội Hoa Kỳ , và Đường K có nhiều văn phòng của nhiều nhóm vận động hành lang .

  - Phân loại: Sửa quá tay / Tên riêng/viết hoa: `K` → `Không` (đúng: `K`)

**Ví dụ 3. Viwiki – sửa quá tay tên riêng/viết hoa [viwiki-8262-0]**

- src : … Hán : 周慧敏 ; bính âm : Zhōu Hùi **Mǐn** / Vivian **Chow** ) ( sinh ngày 10 tháng 11 năm 1967 ) là ca …
- pred: … Hán : 周慧敏 ; bính âm : Zhōu Hùi **Min** / Vivian **Chơ** ) ( sinh ngày 10 tháng 11 năm 1967 ) là ca …
- tgt : … Hán : 周慧敏 ; bính âm : Zhōu Hùi Mǐn / Vivian Chow ) ( sinh ngày 10 tháng 11 năm 1967 ) là ca …

  - Phân loại: Sửa quá tay / Tên riêng/viết hoa: `Chow` → `Chơ` (đúng: `Chow`)

**Ví dụ 4. Viwiki – sửa quá tay từ ngoài từ điển [viwiki-1949-1]**

- src : **Nero** được **chú** ông , hoàng đế Claudius nhận làm con nuôi và …
- pred: **Nào** được **cho** ông , hoàng đế Claudius nhận làm con nuôi và …
- tgt : Nero được chú ông , hoàng đế Claudius nhận làm con nuôi và …

  - Phân loại: Sửa quá tay / Ngoài từ điển (nước ngoài): `Nero` → `Nào` (đúng: `Nero`)

**Ví dụ 5. Viwiki – sửa quá tay từ ngoài từ điển [viwiki-1949-3]**

- src : **Nero** trị vì từ năm 54 đến 68 , tập trung phần lớn …
- pred: **Neo** trị vì từ năm 54 đến 68 , tập trung phần lớn …
- tgt : Nero trị vì từ năm 54 đến 68 , tập trung phần lớn …

  - Phân loại: Sửa quá tay / Ngoài từ điển (nước ngoài): `Nero` → `Neo` (đúng: `Nero`)

**Ví dụ 6. Viwiki – sửa quá tay từ thường [viwiki-10692-18]**

- src : … ra , khi xác chết bị mất nước ( bắt đầu " **khô** đi " ) , mô mềm co rút lại làm lộ ra …
- pred: … ra , khi xác chết bị mất nước ( bắt đầu " **khi** đi " ) , mô mềm co rút lại làm lộ ra …
- tgt : … ra , khi xác chết bị mất nước ( bắt đầu " khô đi " ) , mô mềm co rút lại làm lộ ra …

  - Phân loại: Sửa quá tay / Từ thường trong từ điển: `khô` → `khi` (đúng: `khô`)

**Ví dụ 7. VSEC – sửa sai ứng viên [vsec-8491]**

- src : **T** **ính** đến ngày 1 / 4 / 2019 , dân số 53 …
- pred: **Ở** **tính** đến ngày 1 / 4 / 2019 , dân số 53 …
- tgt : **Tính** **∅** đến ngày 1 / 4 / 2019 , dân số 53 …

  - Phân loại: Sửa sai ứng viên: `T` → `Ở` (đúng: `Tính`)

**Ví dụ 8. VSEC – sửa sai ứng viên [vsec-135]**

- src : … các pháp , đòi hỏi hằng ngày chúng ta phải luôn nhớ **ghĩ** đến những lời dạy , những bài pháp Đức Phật đã dạy …
- pred: … các pháp , đòi hỏi hằng ngày chúng ta phải luôn nhớ **cho** đến những lời dạy , những bài pháp Đức Phật đã dạy …
- tgt : … các pháp , đòi hỏi hằng ngày chúng ta phải luôn nhớ **ghi** đến những lời dạy , những bài pháp Đức Phật đã dạy …

  - Phân loại: Sửa sai ứng viên: `ghĩ` → `cho` (đúng: `ghi`)

**Ví dụ 9. VSEC – bỏ sót [vsec-1498]**

- src : … điện trở gây ra **dự** thay đổi điện thế của mạch cầu **điên** trở .
- pred: … điện trở gây ra **sự** thay đổi điện thế của mạch cầu điên trở .
- tgt : … điện trở gây ra **sự** thay đổi điện thế của mạch cầu **điện** trở .

  - Phân loại: Bỏ sót: `điên` → `điên` (đúng: `điện`)

**Ví dụ 10. VSEC – bỏ sót [vsec-4351]**

- src : … vì thế khi chơi dễ gây ra nguy hiểm cho trẻ như **dứt** tay , xước da .
- pred: … vì thế khi chơi dễ gây ra nguy hiểm cho trẻ như dứt tay , xước da .
- tgt : … vì thế khi chơi dễ gây ra nguy hiểm cho trẻ như **đứt** tay , xước da .

  - Phân loại: Bỏ sót: `dứt` → `dứt` (đúng: `đứt`)

**Ví dụ 11. VSEC – sửa quá tay từ thật [vsec-984]**

- src : **Hiên** nay , con người **dần** sử dụng internet là một công cụ hữu ích cho việc tìm …
- pred: **Hiện** nay , con người **dân** sử dụng internet là một công cụ hữu ích cho việc tìm …
- tgt : **Hiện** nay , con người dần sử dụng internet là một công cụ hữu ích cho việc tìm …

  - Phân loại: Sửa quá tay / Từ thường trong từ điển: `dần` → `dân` (đúng: `dần`)

**Ví dụ 12. Syn – sửa sai ứng viên [syn_test-271]**

- src : Bước ra từ thế **jơí** mạng , Sơn Tùng oanh tạc các bảng xếp hạng âm nhạc …
- pred: Bước ra từ thế **hơn** mạng , Sơn Tùng oanh tạc các bảng xếp hạng âm nhạc …
- tgt : Bước ra từ thế **giới** mạng , Sơn Tùng oanh tạc các bảng xếp hạng âm nhạc …

  - Phân loại: Sửa sai ứng viên: `jơí` → `hơn` (đúng: `giới`)

**Ví dụ 13. Syn – bỏ sót [syn_test-1083]**

- src : Về khoảng **thơif** gian giết bà Bông , **Lén** đi đâu , làm gì còn chưa được làm rõ . . …
- pred: Về khoảng **thời** gian giết bà Bông , Lén đi đâu , làm gì còn chưa được làm rõ . . …
- tgt : Về khoảng **thời** gian giết bà Bông , **Nén** đi đâu , làm gì còn chưa được làm rõ . . …

  - Phân loại: Bỏ sót: `Lén` → `Lén` (đúng: `Nén`)

**Ví dụ 14. Viwiki – sửa đúng (đối chứng) [viwiki-6666-106]**

- src : … hầu như mỗi người đều có một máy thu thanh bỏ túi **dể** nghe âm nhạc .
- pred: … hầu như mỗi người đều có một máy thu thanh bỏ túi **để** nghe âm nhạc .
- tgt : … hầu như mỗi người đều có một máy thu thanh bỏ túi **để** nghe âm nhạc .

  - Phân loại: Sửa đúng: `dể` → `để` (đúng: `để`)

## Kiểm chứng giả thuyết Viwiki

| chỉ số | giá trị | ghi chú |
|---|---|---|
| n_edits | 3404 | số vị trí A thay đổi |
| n_gold_errors | 1541 | số lỗi thật |
| sua_dung | 751 |  |
| sua_sai | 127 |  |
| bo_sot | 663 |  |
| sua_qua_tay | 2526 | 74.2% số vị trí bị sửa |
| cor_p | 0.2206 |  |
| cor_r | 0.4873 |  |
| cor_f1 | 0.3037 |  |
| over_so_ky_hieu | 0 | 0.0% ca sửa quá tay; tỉ lệ nền trong token src: 13.3%; trong lỗi thật: 1.8% |
| over_rate_so_ky_hieu | 0.0 | xác suất một token thuộc nhóm bị sửa quá tay |
| over_ten_rieng | 520 | 20.6% ca sửa quá tay; tỉ lệ nền trong token src: 10.8%; trong lỗi thật: 8.6% |
| over_rate_ten_rieng | 0.00909 | xác suất một token thuộc nhóm bị sửa quá tay |
| over_ngoai_tu_dien | 873 | 34.6% ca sửa quá tay; tỉ lệ nền trong token src: 1.4%; trong lỗi thật: 17.7% |
| over_rate_ngoai_tu_dien | 0.11984 | xác suất một token thuộc nhóm bị sửa quá tay |
| over_tu_thuong | 1133 | 44.9% ca sửa quá tay; tỉ lệ nền trong token src: 74.5%; trong lỗi thật: 72.0% |
| over_rate_tu_thuong | 0.00288 | xác suất một token thuộc nhóm bị sửa quá tay |
| syn_test_over_per_clean_token | 0.00279 | lỗi thật/token = 0.0373; sửa quá tay/lỗi thật = 0.07 |
| vsec_test_over_per_clean_token | 0.00329 | lỗi thật/token = 0.0351; sửa quá tay/lỗi thật = 0.09 |
| viwiki_test_over_per_clean_token | 0.0048 | lỗi thật/token = 0.0029; sửa quá tay/lỗi thật = 1.64 |
| neu_bo_qua_ten_rieng | 0.339 | F1 nếu không sửa nhóm này: P=0.261 R=0.483 (mất 6 sửa đúng) |
| neu_bo_qua_ten_rieng+ngoai_tu_dien | 0.3634 | F1 nếu không sửa nhóm này: P=0.338 R=0.393 (mất 145 sửa đúng) |
| neu_bo_qua_ten_rieng+ngoai_tu_dien+so_ky_hieu | 0.3634 | F1 nếu không sửa nhóm này: P=0.338 R=0.393 (mất 145 sửa đúng) |
| top_overcorrections | manga→mang (29); Nero→Nào (18); kỉ→kỷ (15); Gin→Giá (15); Nero→Neo (15); Khun→Khu (15); c→có (14); lb→lễ (13); and→anh (13); bạn→bàn (13); điốt→điểm (12); biến→tiến (10); trưởng→trường (10); Mao→Mai (10); Thự→Thọ (10) |  |

## So sánh A: n-gram (3-gram KN) và C: BARTpho

**Ví dụ 15. ngram đúng, bartpho sai [viwiki-6810-547]**

- src : Không những Voltaire , tác **gia** Thụy Điển là Frans G .
- ngram: Không những Voltaire , tác **giả** Thụy Điển là Frans G .
- bartpho: Không những Voltaire , tác gia Thụy Điển là Frans G .
- tgt : Không những Voltaire , tác **giả** Thụy Điển là Frans G .

**Ví dụ 16. ngram đúng, bartpho sai [vsec-404]**

- src : … có một quy định nào quy định cụ thể về công sức **đòng** góp của vợ , chồng trong việc tạo lập , duy trì …
- ngram: … có một quy định nào quy định cụ thể về công sức **đóng** góp của vợ , chồng trong việc tạo lập , duy trì …
- bartpho: … có một quy định nào quy định cụ thể về công sức đòng góp của vợ , chồng trong việc tạo lập , duy trì …
- tgt : … có một quy định nào quy định cụ thể về công sức **đóng** góp của vợ , chồng trong việc tạo lập , duy trì …

**Ví dụ 17. ngram đúng, bartpho sai [syn_test-4838]**

- src : **Buỗi** chiều , các đại biểu sẽ thảo luận theo tổ .
- ngram: **Buổi** chiều , các đại biểu sẽ thảo luận theo tổ .
- bartpho: Buỗi chiều , các đại biểu sẽ thảo luận theo tổ .
- tgt : **Buổi** chiều , các đại biểu sẽ thảo luận theo tổ .

**Ví dụ 18. bartpho đúng, ngram sai [syn_test-3260]**

- src : Người con trai khác của ông Đinh Văn **Fhu** **lpà** Đinh Văn Vũ , hiện đang giữ chức Phó Giám đốc …
- ngram: Người con trai khác của ông Đinh Văn Fhu **là** Đinh Văn Vũ , hiện đang giữ chức Phó Giám đốc …
- bartpho: Người con trai khác của ông Đinh Văn **Thu** **là** Đinh Văn Vũ , hiện đang giữ chức Phó Giám đốc …
- tgt : Người con trai khác của ông Đinh Văn **Thu** **là** Đinh Văn Vũ , hiện đang giữ chức Phó Giám đốc …

**Ví dụ 19. bartpho đúng, ngram sai [vsec-8988]**

- src : Học sinh có nhiều cách khác nhau để thực hiện biện **phpasp** này , chẳng hạn : Phân loại bài toán , phân dạng …
- ngram: Học sinh có nhiều cách khác nhau để thực hiện biện phpasp này , chẳng hạn : Phân loại bài toán , phân dạng …
- bartpho: Học sinh có nhiều cách khác nhau để thực hiện biện **pháp** này , chẳng hạn : Phân loại bài toán , phân dạng …
- tgt : Học sinh có nhiều cách khác nhau để thực hiện biện **pháp** này , chẳng hạn : Phân loại bài toán , phân dạng …

**Ví dụ 20. bartpho đúng, ngram sai [viwiki-1022-1]**

- src : … về tính ác liệt của cuộc chiến bảo vệ tổ quốc và **cơ** **ngợi** những chiến thắng được viết bởi nhà thơ Francisco González Bocanegra …
- ngram: … về tính ác liệt của cuộc chiến bảo vệ tổ quốc và cơ **ngơi** những chiến thắng được viết bởi nhà thơ Francisco González Bocanegra …
- bartpho: … về tính ác liệt của cuộc chiến bảo vệ tổ quốc và **ca** ngợi những chiến thắng được viết bởi nhà thơ Francisco González Bocanegra …
- tgt : … về tính ác liệt của cuộc chiến bảo vệ tổ quốc và **ca** ngợi những chiến thắng được viết bởi nhà thơ Francisco González Bocanegra …

**Ví dụ 21. cả hai sai [vsec-8889]**

- src : … tra có ít nhất 4000 hộ xin vay vốn tín **chất** để **phụ** vụ cho hoạt động nuôi trồng thủy sản .
- ngram: … tra có ít nhất 4000 hộ xin vay vốn tín **chấp** để phụ vụ cho hoạt động nuôi trồng thủy sản .
- bartpho: … tra có ít nhất 4000 hộ xin vay vốn tín chất để phụ vụ cho hoạt động nuôi trồng thủy sản .
- tgt : … tra có ít nhất 4000 hộ xin vay vốn tín **chấp** để **phục** vụ cho hoạt động nuôi trồng thủy sản .

**Ví dụ 22. cả hai sai [viwiki-12116-156]**

- src : … PL với 87 điểm ( HS : 80 - 22 ) , **Chel** 85 điểm ( 65 - 26 ) khi để Bolton cầm hòa …
- ngram: … PL với 87 điểm ( HS : 80 - 22 ) , Chel 85 điểm ( 65 - 26 ) khi để Bolton cầm hòa …
- bartpho: … PL với 87 điểm ( HS : 80 - 22 ) , Chel 85 điểm ( 65 - 26 ) khi để Bolton cầm hòa …
- tgt : … PL với 87 điểm ( HS : 80 - 22 ) , **chelsea** 85 điểm ( 65 - 26 ) khi để Bolton cầm hòa …

**Ví dụ 23. cả hai sai [syn_test-1468]**

- src : Tết thì **khoong** thể thiếu hoa đào , hoa **hai** , vẫn bánh chưng và không khí nhộn nhịp của ngày Tết …
- ngram: Tết thì **không** thể thiếu hoa đào , hoa **hồi** , vẫn bánh chưng và không khí nhộn nhịp của ngày Tết …
- bartpho: Tết thì **không** thể thiếu hoa đào , hoa hai , vẫn bánh chưng và không khí nhộn nhịp của ngày Tết …
- tgt : Tết thì **không** thể thiếu hoa đào , hoa **mai** , vẫn bánh chưng và không khí nhộn nhịp của ngày Tết …
