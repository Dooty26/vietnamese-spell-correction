# Ví dụ phân tích lỗi (tự sinh bởi `scripts/analyze_errors.py`, seed=7)

Câu được tách token như khi chấm (dấu câu tách riêng), cắt cửa sổ quanh vị trí lỗi.
In **đậm**: ở `src` là vị trí lỗi thật hoặc bị mô hình sửa; ở `pred`/`tgt` là chuỗi thay thế; `∅` là âm tiết bị xóa.

## Mô hình A: n-gram (3-gram KN)

**Ví dụ 1. Viwiki – sửa quá tay tên riêng/viết hoa [viwiki-11033-78]**

- src : Phần đầu tiên kể về những nỗ lực của Henry **V** trong việc đàn áp Lollard Heresy và phần thứ hai mô tả …
- pred: Phần đầu tiên kể về những nỗ lực của Henry **Vậy** trong việc đàn áp Lollard Heresy và phần thứ hai mô tả …
- tgt : Phần đầu tiên kể về những nỗ lực của Henry V trong việc đàn áp Lollard Heresy và phần thứ hai mô tả …

  - Phân loại: Sửa quá tay / Tên riêng/viết hoa: `V` → `Vậy` (đúng: `V`)

**Ví dụ 2. Viwiki – sửa quá tay tên riêng/viết hoa [viwiki-9953-46]**

- src : … Kramer mời cô đóng vai chính cùng Gary Cooper trong " High **Noon** " ( 1951 ) .
- pred: … Kramer mời cô đóng vai chính cùng Gary Cooper trong " High **Non** " ( 1951 ) .
- tgt : … Kramer mời cô đóng vai chính cùng Gary Cooper trong " High Noon " ( 1951 ) .

  - Phân loại: Sửa quá tay / Tên riêng/viết hoa: `Noon` → `Non` (đúng: `Noon`)

**Ví dụ 3. Viwiki – sửa quá tay tên riêng/viết hoa [viwiki-8390-142]**

- src : Malfoy đã nói với cụ Dumbledore rằng một số Tử Thần Thực **Tử** đã đột nhập vào trong lâu đài và bắn lên dấu hiệu …
- pred: Malfoy đã nói với cụ Dumbledore rằng một số Tử Thần Thực **Tế** đã đột nhập vào trong lâu đài và bắn lên dấu hiệu …
- tgt : Malfoy đã nói với cụ Dumbledore rằng một số Tử Thần Thực Tử đã đột nhập vào trong lâu đài và bắn lên dấu hiệu …

  - Phân loại: Sửa quá tay / Tên riêng/viết hoa: `Tử` → `Tế` (đúng: `Tử`)

**Ví dụ 4. Viwiki – sửa quá tay từ ngoài từ điển [viwiki-10081-27]**

- src : … cho người mắc hội chứng này chỉ được xác định một cách **tuơng** đối và số người mắc là nam thì nhiều hơn nữ .
- pred: … cho người mắc hội chứng này chỉ được xác định một cách **tương** đối và số người mắc là nam thì nhiều hơn nữ .
- tgt : … cho người mắc hội chứng này chỉ được xác định một cách tuơng đối và số người mắc là nam thì nhiều hơn nữ .

  - Phân loại: Sửa quá tay / Ngoài từ điển (nước ngoài): `tuơng` → `tương` (đúng: `tuơng`)

**Ví dụ 5. Viwiki – sửa quá tay từ ngoài từ điển [viwiki-1949-164]**

- src : **Nero** hứng thích với việc cưỡi xe ngựa không **mui** , hát cùng …
- pred: **Neo** hứng thích với việc cưỡi xe ngựa không **vui** , hát cùng …
- tgt : Nero hứng thích với việc cưỡi xe ngựa không mui , hát cùng …

  - Phân loại: Sửa quá tay / Ngoài từ điển (nước ngoài): `Nero` → `Neo` (đúng: `Nero`)

**Ví dụ 6. Viwiki – sửa quá tay từ thường [viwiki-12748-273]**

- src : Các kế hoạch đã được **phát** **thảo** cho các cuộc không kích nhắm vào các địa điểm có tên …
- pred: Các kế hoạch đã được phát **thải** cho các cuộc không kích nhắm vào các địa điểm có tên …
- tgt : Các kế hoạch đã được **phác** thảo cho các cuộc không kích nhắm vào các địa điểm có tên …

  - Phân loại: Sửa quá tay / Từ thường trong từ điển: `thảo` → `thải` (đúng: `thảo`)

**Ví dụ 7. VSEC – sửa sai ứng viên [vsec-7959]**

- src : **rong** đó bài tập 1 trang 144 yêu cầu **tìm** thời gian hai …
- pred: **trong** đó bài tập 1 trang 144 yêu cầu **tạm** thời gian hai …
- tgt : **Trong** đó bài tập 1 trang 144 yêu cầu tìm thời gian hai …

  - Phân loại: Sửa sai ứng viên: `rong` → `trong` (đúng: `Trong`)

**Ví dụ 8. VSEC – sửa sai ứng viên [vsec-216]**

- src : **T** **rường** Mầm non huyện **Y** **ên** Phong đã long trọng tổ chức **lê** khai giảng năm học …
- pred: **Ở** **trường** Mầm non huyện **Ý** **yên** Phong đã long trọng tổ chức **lễ** khai giảng năm học …
- tgt : **Trường** **∅** Mầm non huyện **Yên** **∅** Phong đã long trọng tổ chức **lễ** khai giảng năm học …

  - Phân loại: Sửa sai ứng viên: `Y` → `Ý` (đúng: `Yên`)

**Ví dụ 9. VSEC – bỏ sót [vsec-2411]**

- src : Do vậy **cẩn** đổi mới những **pháp** pháp khác nhau trong dạy học để nâng cao hiệu quả của …
- pred: Do vậy **cần** đổi mới những pháp pháp khác nhau trong dạy học để nâng cao hiệu quả của …
- tgt : Do vậy **cần** đổi mới những **phương** pháp khác nhau trong dạy học để nâng cao hiệu quả của …

  - Phân loại: Bỏ sót: `pháp` → `pháp` (đúng: `phương`)

**Ví dụ 10. VSEC – bỏ sót [vsec-6440]**

- src : a ) Đường **kinh** xi lanh công **tác** **banh** xe .
- pred: a ) Đường kinh xi lanh công tác banh xe .
- tgt : a ) Đường **kính** xi lanh công **tắc** **bánh** xe .

  - Phân loại: Bỏ sót: `banh` → `banh` (đúng: `bánh`)

**Ví dụ 11. VSEC – sửa quá tay từ thật [vsec-4364]**

- src : … cần bị xử lý bằng chế tài phù hợp , là chế **thi** hành chính hoặc chế tài hình sự .
- pred: … cần bị xử lý bằng chế tài phù hợp , là chế **tài** hành chính hoặc chế tài hình sự .
- tgt : … cần bị xử lý bằng chế tài phù hợp , là chế thi hành chính hoặc chế tài hình sự .

  - Phân loại: Sửa quá tay / Từ thường trong từ điển: `thi` → `tài` (đúng: `thi`)

**Ví dụ 12. Syn – sửa sai ứng viên [syn_test-4404]**

- src : … an đã thu giữ 5 con gà chọi , một cân lò **ao** , hai đồng hồ ( dùng để tính giờ ) , cân …
- pred: … an đã thu giữ 5 con gà chọi , một cân lò **cao** , hai đồng hồ ( dùng để tính giờ ) , cân …
- tgt : … an đã thu giữ 5 con gà chọi , một cân lò **xo** , hai đồng hồ ( dùng để tính giờ ) , cân …

  - Phân loại: Sửa sai ứng viên: `ao` → `cao` (đúng: `xo`)

**Ví dụ 13. Syn – bỏ sót [syn_test-3260]**

- src : Người con trai khác của ông Đinh Văn **Fhu** **lpà** Đinh Văn Vũ , hiện đang giữ chức Phó Giám đốc …
- pred: Người con trai khác của ông Đinh Văn Fhu **là** Đinh Văn Vũ , hiện đang giữ chức Phó Giám đốc …
- tgt : Người con trai khác của ông Đinh Văn **Thu** **là** Đinh Văn Vũ , hiện đang giữ chức Phó Giám đốc …

  - Phân loại: Bỏ sót: `Fhu` → `Fhu` (đúng: `Thu`)

**Ví dụ 14. Viwiki – sửa đúng (đối chứng) [viwiki-1796-49]**

- src : … Hán , **Khun** Sa đã mắc chứng nghẽn mạch máu , liệt **nữa** người .
- pred: … Hán , **Khu** Sa đã mắc chứng nghẽn mạch máu , liệt **nửa** người .
- tgt : … Hán , Khun Sa đã mắc chứng nghẽn mạch máu , liệt **nửa** người .

  - Phân loại: Sửa đúng: `nữa` → `nửa` (đúng: `nửa`)

## Kiểm chứng giả thuyết Viwiki

| chỉ số | giá trị | ghi chú |
|---|---|---|
| n_edits | 3393 | số vị trí A thay đổi |
| n_gold_errors | 1542 | số lỗi thật |
| sua_dung | 751 |  |
| sua_sai | 127 |  |
| bo_sot | 664 |  |
| sua_qua_tay | 2515 | 74.1% số vị trí bị sửa |
| cor_p | 0.2213 |  |
| cor_r | 0.487 |  |
| cor_f1 | 0.3044 |  |
| over_so_ky_hieu | 0 | 0.0% ca sửa quá tay; tỉ lệ nền trong token src: 13.3%; trong lỗi thật: 1.8% |
| over_rate_so_ky_hieu | 0.0 | xác suất một token thuộc nhóm bị sửa quá tay |
| over_ten_rieng | 526 | 20.9% ca sửa quá tay; tỉ lệ nền trong token src: 10.8%; trong lỗi thật: 8.6% |
| over_rate_ten_rieng | 0.00919 | xác suất một token thuộc nhóm bị sửa quá tay |
| over_ngoai_tu_dien | 858 | 34.1% ca sửa quá tay; tỉ lệ nền trong token src: 1.4%; trong lỗi thật: 17.6% |
| over_rate_ngoai_tu_dien | 0.11688 | xác suất một token thuộc nhóm bị sửa quá tay |
| over_tu_thuong | 1131 | 45.0% ca sửa quá tay; tỉ lệ nền trong token src: 74.5%; trong lỗi thật: 71.9% |
| over_rate_tu_thuong | 0.00288 | xác suất một token thuộc nhóm bị sửa quá tay |
| syn_test_over_per_clean_token | 0.00279 | lỗi thật/token = 0.0373; sửa quá tay/lỗi thật = 0.07 |
| vsec_test_over_per_clean_token | 0.00329 | lỗi thật/token = 0.0351; sửa quá tay/lỗi thật = 0.09 |
| viwiki_test_over_per_clean_token | 0.00478 | lỗi thật/token = 0.0029; sửa quá tay/lỗi thật = 1.63 |
| neu_bo_qua_ten_rieng | 0.3403 | F1 nếu không sửa nhóm này: P=0.263 R=0.483 (mất 6 sửa đúng) |
| neu_bo_qua_ten_rieng+ngoai_tu_dien | 0.3635 | F1 nếu không sửa nhóm này: P=0.338 R=0.393 (mất 145 sửa đúng) |
| neu_bo_qua_ten_rieng+ngoai_tu_dien+so_ky_hieu | 0.3635 | F1 nếu không sửa nhóm này: P=0.338 R=0.393 (mất 145 sửa đúng) |
| top_overcorrections | manga→mang (29); Nero→Nào (18); kỉ→kỷ (15); Gin→Giá (15); Nero→Neo (15); Khun→Khu (15); c→có (14); lb→lễ (13); and→anh (13); bạn→bàn (13); điốt→điểm (12); biến→tiến (10); trưởng→trường (10); Mao→Mai (10); Thự→Thọ (10) |  |

## So sánh A: n-gram (3-gram KN) và C: BARTpho v3

**Ví dụ 15. ngram đúng, bartpho sai [vsec-5393]**

- src : Tư tưởng Hồ Chí Minh là một trong những nhân tố quan **trong** quyết định thành công của sự nghiệp đổi mới .
- ngram: Tư tưởng Hồ Chí Minh là một trong những nhân tố quan **trọng** quyết định thành công của sự nghiệp đổi mới .
- bartpho: Tư tưởng Hồ Chí Minh là một trong những nhân tố quan trong quyết định thành công của sự nghiệp đổi mới .
- tgt : Tư tưởng Hồ Chí Minh là một trong những nhân tố quan **trọng** quyết định thành công của sự nghiệp đổi mới .

**Ví dụ 16. ngram đúng, bartpho sai [viwiki-2866-16]**

- src : Hệ thống tàu điện ngầm Luân Đôn do Cục **vân** tải Luân Đôn ( " Transport for London " ) quản lý …
- ngram: Hệ thống tàu điện ngầm Luân Đôn do Cục **vận** tải Luân Đôn ( " Transport for London " ) quản lý …
- bartpho: Hệ thống tàu điện ngầm Luân Đôn do Cục vân tải Luân Đôn ( " Transport for London " ) quản lý …
- tgt : Hệ thống tàu điện ngầm Luân Đôn do Cục **vận** tải Luân Đôn ( " Transport for London " ) quản lý …

**Ví dụ 17. ngram đúng, bartpho sai [syn_test-4838]**

- src : **Buỗi** chiều , các đại biểu sẽ thảo luận theo tổ .
- ngram: **Buổi** chiều , các đại biểu sẽ thảo luận theo tổ .
- bartpho: Buỗi chiều , các đại biểu sẽ thảo luận theo tổ .
- tgt : **Buổi** chiều , các đại biểu sẽ thảo luận theo tổ .

**Ví dụ 18. bartpho đúng, ngram sai [vsec-1489]**

- src : … phần học sinh chỉ được **àm** **uen** các dạng Toán tiếng việt **à** rất ít em có thể tham gia học tập các lớp Toán …
- ngram: … phần học sinh chỉ được **làm** **quen** các dạng Toán tiếng việt **là** rất ít em có thể tham gia học tập các lớp Toán …
- bartpho: … phần học sinh chỉ được **làm** **quen** các dạng Toán tiếng việt **và** rất ít em có thể tham gia học tập các lớp Toán …
- tgt : … phần học sinh chỉ được **làm** **quen** các dạng Toán tiếng việt **và** rất ít em có thể tham gia học tập các lớp Toán …

**Ví dụ 19. bartpho đúng, ngram sai [syn_test-1296]**

- src : Đây là bộ phim thiếu nhi **đẩm** chất hài hước nhưng có tinh thần giáo dục nghiêm túc và …
- ngram: Đây là bộ phim thiếu nhi đẩm chất hài hước nhưng có tinh thần giáo dục nghiêm túc và …
- bartpho: Đây là bộ phim thiếu nhi **đậm** chất hài hước nhưng có tinh thần giáo dục nghiêm túc và …
- tgt : Đây là bộ phim thiếu nhi **đậm** chất hài hước nhưng có tinh thần giáo dục nghiêm túc và …

**Ví dụ 20. bartpho đúng, ngram sai [viwiki-1949-153]**

- src : **Sauk** **hi** trận hỏa hoạn , Nero cho xây dựng lâu **đại** , cung …
- ngram: **Sau** **hai** trận hỏa hoạn , Nero cho xây dựng lâu **dài** , cung …
- bartpho: **Sau** **khi** trận hỏa hoạn , Nero cho xây dựng lâu đại , cung …
- tgt : Sauk **khi** trận hỏa hoạn , Nero cho xây dựng lâu **đài** , cung …

**Ví dụ 21. cả hai sai [vsec-1895]**

- src : Phân tích các số liệu , **do** **lường** để tìm ra các mức độ tác động của các nhân …
- ngram: Phân tích các số liệu , do **lượng** để tìm ra các mức độ tác động của các nhân …
- bartpho: Phân tích các số liệu , do lường để tìm ra các mức độ tác động của các nhân …
- tgt : Phân tích các số liệu , **đo** lường để tìm ra các mức độ tác động của các nhân …

**Ví dụ 22. cả hai sai [viwiki-12816-182]**

- src : … ngành giải trí thế giới với số tài sản hơn 130 triệu **USD** và luôn đứng đầu trên các phương tiện thông tin .
- ngram: … ngành giải trí thế giới với số tài sản hơn 130 triệu USD và luôn đứng đầu trên các phương tiện thông tin .
- bartpho: … ngành giải trí thế giới với số tài sản hơn 130 triệu USD và luôn đứng đầu trên các phương tiện thông tin .
- tgt : … ngành giải trí thế giới với số tài sản hơn 130 triệu **us** và luôn đứng đầu trên các phương tiện thông tin .

**Ví dụ 23. cả hai sai [syn_test-673]**

- src : … xuất hiện trên thảm đỏ LHP Cannes với bộ váy vô cùng **hk** **hênh** , người hâm mộ cho rằng cô đã vô ý .
- ngram: … xuất hiện trên thảm đỏ LHP Cannes với bộ váy vô cùng **hà** **hạnh** , người hâm mộ cho rằng cô đã vô ý .
- bartpho: … xuất hiện trên thảm đỏ LHP Cannes với bộ váy vô cùng **hở** hênh , người hâm mộ cho rằng cô đã vô ý .
- tgt : … xuất hiện trên thảm đỏ LHP Cannes với bộ váy vô cùng **hớ** hênh , người hâm mộ cho rằng cô đã vô ý .

## So sánh A: n-gram (3-gram KN) và C: BARTpho v4 (nhiễu dấu)

**Ví dụ 24. ngram đúng, bartpho_v4 sai [vsec-7217]**

- src : … cho từ " Bảo hiểm " tại Việt Nam , chúng tôi **nổ** lực mang lại cho bạn quyền được an tâm về cuộc sống …
- ngram: … cho từ " Bảo hiểm " tại Việt Nam , chúng tôi **nỗ** lực mang lại cho bạn quyền được an tâm về cuộc sống …
- bartpho_v4: … cho từ " Bảo hiểm " tại Việt Nam , chúng tôi nổ lực mang lại cho bạn quyền được an tâm về cuộc sống …
- tgt : … cho từ " Bảo hiểm " tại Việt Nam , chúng tôi **nỗ** lực mang lại cho bạn quyền được an tâm về cuộc sống …

**Ví dụ 25. ngram đúng, bartpho_v4 sai [viwiki-1949-47]**

- src : … đỡ Claudius trong vấn đề triều chính , Lucius chính thức được **nhập** làm con nuôi vào năm 50 và đổi tên thành " Nero …
- ngram: … đỡ Claudius trong vấn đề triều chính , Lucius chính thức được **nhận** làm con nuôi vào năm 50 và đổi tên thành " Nero …
- bartpho_v4: … đỡ Claudius trong vấn đề triều chính , Lucius chính thức được nhập làm con nuôi vào năm 50 và đổi tên thành " Nero …
- tgt : … đỡ Claudius trong vấn đề triều chính , Lucius chính thức được **nhận** làm con nuôi vào năm 50 và đổi tên thành " Nero …

**Ví dụ 26. ngram đúng, bartpho_v4 sai [syn_test-4193]**

- src : … uy tín am hiểu phong tục - tập quán tập trung khấn **cún** Thần linh .
- ngram: … uy tín am hiểu phong tục - tập quán tập trung khấn **cúng** Thần linh .
- bartpho_v4: … uy tín am hiểu phong tục - tập quán tập trung khấn cún Thần linh .
- tgt : … uy tín am hiểu phong tục - tập quán tập trung khấn **cúng** Thần linh .

**Ví dụ 27. bartpho_v4 đúng, ngram sai [vsec-672]**

- src : Ngoài ra , những người có trình độ học vấn cao **thưởng** bỏ ra nhiều thời gian và công sức hơn trong việc lựa …
- ngram: Ngoài ra , những người có trình độ học vấn cao thưởng bỏ ra nhiều thời gian và công sức hơn trong việc lựa …
- bartpho_v4: Ngoài ra , những người có trình độ học vấn cao **thường** bỏ ra nhiều thời gian và công sức hơn trong việc lựa …
- tgt : Ngoài ra , những người có trình độ học vấn cao **thường** bỏ ra nhiều thời gian và công sức hơn trong việc lựa …

**Ví dụ 28. bartpho_v4 đúng, ngram sai [syn_test-517]**

- src : Kết **wả** nghiên **cưus** cho thấy tất cả 30 mẫu đều bị **nhiễwm** nhôm …
- ngram: Kết **để** nghiên **cứu** cho thấy tất cả 30 mẫu đều bị **nhiễm** nhôm …
- bartpho_v4: Kết **quả** nghiên **cứu** cho thấy tất cả 30 mẫu đều bị **nhiễm** nhôm …
- tgt : Kết **quả** nghiên **cứu** cho thấy tất cả 30 mẫu đều bị **nhiễm** nhôm …

**Ví dụ 29. bartpho_v4 đúng, ngram sai [viwiki-1949-342]**

- src : **Nero** liền **cảnh** thấy hoảng sợ và tin rằng thần linh muốn đền thờ Jerusalem …
- ngram: **Neo** liền cảnh thấy hoảng sợ và tin rằng thần linh muốn đền thờ Jerusalem …
- bartpho_v4: Nero liền **cảm** thấy hoảng sợ và tin rằng thần linh muốn đền thờ Jerusalem …
- tgt : Nero liền **cảm** thấy hoảng sợ và tin rằng thần linh muốn đền thờ Jerusalem …

**Ví dụ 30. cả hai sai [vsec-819]**

- src : … cảm thấy mình bị bỏ rơi , dẫn đến ấm ức **dồn** **né** và tìm cách giải tỏa trong các mối quan hệ khác .
- ngram: … cảm thấy mình bị bỏ rơi , dẫn đến ấm ức **đòn** né và tìm cách giải tỏa trong các mối quan hệ khác .
- bartpho_v4: … cảm thấy mình bị bỏ rơi , dẫn đến ấm ức dồn né và tìm cách giải tỏa trong các mối quan hệ khác .
- tgt : … cảm thấy mình bị bỏ rơi , dẫn đến ấm ức dồn **nén** và tìm cách giải tỏa trong các mối quan hệ khác .

**Ví dụ 31. cả hai sai [viwiki-8331-8]**

- src : Những tên thánh của Clausewitz thường được nói **trong** không - nguồn gốc Đức như " Karl , " " Carl …
- ngram: Những tên thánh của Clausewitz thường được nói trong không - nguồn gốc Đức như " Karl , " " Carl …
- bartpho_v4: Những tên thánh của Clausewitz thường được nói trong không - nguồn gốc Đức như " Karl , " " Carl …
- tgt : Những tên thánh của Clausewitz thường được nói **trống** không - nguồn gốc Đức như " Karl , " " Carl …

**Ví dụ 32. cả hai sai [syn_test-2405]**

- src : … những bộ phim điện ảnh của anh mà tôi rất thích như **Dụ** án viên đạn lạc , Đời hát rong , Kế hoạch P76 …
- ngram: … những bộ phim điện ảnh của anh mà tôi rất thích như **Dự** án viên đạn lạc , Đời hát rong , Kế hoạch P76 …
- bartpho_v4: … những bộ phim điện ảnh của anh mà tôi rất thích như **Dự** án viên đạn lạc , Đời hát rong , Kế hoạch P76 …
- tgt : … những bộ phim điện ảnh của anh mà tôi rất thích như **Vụ** án viên đạn lạc , Đời hát rong , Kế hoạch P76 …
