"""vspell: phát hiện và sửa lỗi chính tả tiếng Việt.

Định dạng bản ghi dùng chung cho mọi tập (JSONL), một dòng một câu:
    {"id": str, "src": str, "tgt": str, "dataset": str}
- src: câu có thể chứa lỗi (đầu vào mô hình)
- tgt: câu đúng (nhãn chuẩn); đã chuẩn hóa bằng vspell.normalize
"""

__version__ = "0.1.0"
