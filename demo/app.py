"""Demo Gradio: nhập câu -> chọn mô hình (n-gram / BARTpho) -> tô âm tiết sai và hiển thị gợi ý sửa.

Chạy: python demo/app.py --bartpho-checkpoint outputs/checkpoints/bartpho/best
Phụ trách: TV3 (ngày 10–11).
"""
import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bartpho-checkpoint", default=None)
    ap.add_argument("--ngram-dir", default="outputs/checkpoints/ngram")
    args = ap.parse_args()
    # TODO(TV3): gr.Interface với gr.HighlightedText; dùng vspell.eval.align.edit_positions để tô lỗi
    raise NotImplementedError


if __name__ == "__main__":
    main()
