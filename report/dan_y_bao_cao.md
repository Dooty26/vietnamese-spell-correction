# Dàn ý báo cáo (20–30 trang)

1. **Giới thiệu** – bài toán, ứng dụng, câu hỏi nghiên cứu: (i) LM n-gram và BARTpho khác nhau thế nào? (ii) mô hình train trên lỗi tổng hợp tổng quát sang lỗi thật đến đâu? *(TV2)*
2. **Dữ liệu** – Binhvq, VSEC, Viwiki; thống kê (số câu/âm tiết, độ dài, mật độ lỗi); làm sạch và chuẩn hóa Unicode/dấu thanh; sinh nhiễu (loại lỗi, tỉ lệ); chia tập theo bài báo; chống rò rỉ. *(TV1)*
3. **Phương pháp** – sơ đồ pipeline; A: từ điển, tập dễ nhầm, KenLM KN, beam search; C: BARTpho-syllable, tham số fine-tune. *(TV2, TV3)*
4. **Thực nghiệm** – môi trường, phiên bản thư viện, seed; chỉ số; quy trình chọn tham số trên Syn-val/VSEC-dev. *(TV2)*
5. **Kết quả** – bảng chính {A, C} × {Syn-test, VSEC-test, Viwiki-test}; ablation n-gram (bậc, smoothing) và mật độ nhiễu; recall theo loại lỗi; 5–10 ví dụ lỗi có phân tích. *(TV3, TV1)*
6. **Kết luận** – đạt/chưa đạt, hạn chế, hướng phát triển (BiLSTM, lỗi tách/ghép từ, teencode với ViLexNorm). *(TV3)*
7. **Phân công, tài liệu tham khảo, phụ lục sử dụng công cụ AI.**
