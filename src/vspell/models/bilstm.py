"""Mô hình B (TÙY CHỌN, chỉ làm nếu dư thời gian): BiLSTM gắn nhãn chuỗi với char-CNN.

Mỗi âm tiết đầu vào -> biểu diễn char-CNN -> BiLSTM mức câu -> softmax trên từ vựng âm tiết đúng.
"""
from __future__ import annotations


class BiLSTMCorrector:
    def __init__(self, config: dict):
        self.config = config

    def fit(self, train_path: str, val_path: str) -> None:
        raise NotImplementedError

    def correct(self, sentences: list[str]) -> list[str]:
        raise NotImplementedError
