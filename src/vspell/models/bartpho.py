"""Mô hình C: fine-tune vinai/bartpho-syllable (seq2seq câu lỗi -> câu đúng).

Chạy trên Kaggle/Colab Pro, lưu checkpoint thường xuyên. Phụ trách: TV3 (ngày 3–9).
"""
from __future__ import annotations


def build_datasets(train_path: str, val_path: str, tokenizer, config: dict):
    """Đọc JSONL -> HF Dataset đã tokenize (src -> input_ids, tgt -> labels)."""
    raise NotImplementedError


def train(config: dict, train_path: str, val_path: str) -> str:
    """Huấn luyện bằng Seq2SeqTrainer; chọn checkpoint tốt nhất theo Syn-val; trả về đường dẫn checkpoint."""
    raise NotImplementedError


class BartphoCorrector:
    def __init__(self, checkpoint: str, num_beams: int = 4, max_length: int = 96, device: str | None = None):
        self.checkpoint = checkpoint
        self.num_beams = num_beams
        self.max_length = max_length
        self.device = device

    def correct(self, sentences: list[str], batch_size: int = 64) -> list[str]:
        raise NotImplementedError
