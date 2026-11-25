"""Binhvq news corpus (HF: vietgpt/binhvq_news_vi): đọc từng phần, làm sạch, tách câu, chia tập.

Đặc điểm bản HF: 9 file parquet (~19,4M dòng, mỗi file ~2.152 row group x 1.000 dòng), cột duy nhất "text".
Mỗi dòng là một ĐOẠN văn (~200–1.000 ký tự), KHÔNG có mã bài báo; các dòng có vẻ được sắp theo độ dài.
=> Lấy mẫu ngẫu nhiên các row group trên cả 9 file (không cần tải 4,8 GB),
   và chia tập theo ĐOẠN (mọi câu của một đoạn thuộc cùng một tập). Ghi rõ hạn chế này trong báo cáo.
Phụ trách: TV1.
"""
from __future__ import annotations

import html
import random
import re
from typing import Iterator

from vspell.data.viwiki import sentence_spans
from vspell.normalize import is_vietnamese_syllable, normalize, split_syllables

_BAD_MARKERS = re.compile(r"https?://|www\.|@|href|onmouse|javascript|[<>{}\\|]|\.(?:html?|php|aspx?)\b", re.I)
_TAG = re.compile(r"<[^>]*>")
_URL = re.compile(r"https?://\S+|www\.\S+")


def stream_paragraphs(hf_id: str, n_row_groups: int, seed: int, workers: int = 8) -> Iterator[tuple[str, str]]:
    """Đọc ngẫu nhiên n_row_groups row group (theo seed) trên mọi file parquet -> (paragraph_id, text).

    Tải song song bằng nhiều luồng nhưng vẫn trả về theo đúng thứ tự đã xáo (kết quả tái lập được).
    """
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{hf_id}/data/*.parquet"))
    n_groups = [pq.ParquetFile(fs.open(f, "rb")).metadata.num_row_groups for f in files]
    pairs = [(fi, rg) for fi in range(len(files)) for rg in range(n_groups[fi])]
    random.Random(seed).shuffle(pairs)

    def read(pair: tuple[int, int]) -> list[str]:
        fi, rg = pair
        with fs.open(files[fi], "rb") as fh:  # mỗi luồng mở file riêng (ParquetFile không an toàn đa luồng)
            return pq.ParquetFile(fh).read_row_group(rg, columns=["text"]).column(0).to_pylist()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        # Gửi theo lô để có thể dừng sớm mà không tải thừa quá nhiều
        for start in range(0, min(n_row_groups, len(pairs)), workers * 2):
            batch = pairs[start:min(start + workers * 2, n_row_groups)]
            for (fi, rg), texts in zip(batch, ex.map(read, batch)):
                for i, text in enumerate(texts):
                    if text:
                        yield f"{fi}-{rg}-{i}", text


def clean_text(text: str) -> str:
    """Giải mã HTML entity, bỏ thẻ HTML và URL, chuẩn hóa bằng vspell.normalize."""
    text = html.unescape(text)
    text = _TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    return normalize(text)


def split_sentences(text: str) -> list[str]:
    """Tách câu (dùng chung quy tắc với Viwiki)."""
    return [text[s:e].strip() for s, e in sentence_spans(text)]


def keep_sentence(sentence: str, min_syllables: int = 5, max_syllables: int = 80,
                  min_vi_ratio: float = 0.85) -> bool:
    """Giữ câu "sạch" để làm nhãn chuẩn:
    - độ dài trong [min_syllables, max_syllables] token;
    - không chứa dấu vết HTML/URL/email;
    - bắt đầu bằng chữ hoa/số/ngoặc/nháy và kết thúc bằng . ! ? hoặc nháy/ngoặc đóng;
    - >= min_vi_ratio số từ là âm tiết tiếng Việt hợp lệ; số token là số <= 20%; không viết hoa toàn bộ.
    """
    if _BAD_MARKERS.search(sentence):
        return False
    toks = split_syllables(sentence)
    if not (min_syllables <= len(toks) <= max_syllables):
        return False
    if not (sentence[0].isupper() or sentence[0].isdigit() or sentence[0] in "\"'“‘(["):
        return False
    if sentence[-1] not in ".!?\"'”’)…":
        return False
    words = [t for t in toks if t.isalpha()]
    if len(words) < min_syllables:
        return False
    if sum(t.isdigit() or t[0].isdigit() for t in toks) > 0.2 * len(toks):
        return False
    if sum(w.isupper() and len(w) > 1 for w in words) > 0.3 * len(words):
        return False
    # Viết tắt toàn chữ hoa (UBND, TP, ASEAN) không tính vào tỉ lệ tiếng Việt
    lower_words = [w for w in words if not (w.isupper() and len(w) > 1)]
    vi = sum(is_vietnamese_syllable(w) for w in lower_words)
    return vi >= min_vi_ratio * len(lower_words)


def split_by_paragraph(paragraphs: list[tuple[str, list[str]]], sizes: dict[str, int],
                       seed: int) -> dict[str, list[str]]:
    """Chia câu vào các tập theo thứ tự sizes (vd syn_test, syn_val, syn_train, lm_corpus) sao cho
    mọi câu của một đoạn thuộc cùng một tập. Tập cuối nhận phần còn lại đến khi đủ kích thước."""
    order = list(range(len(paragraphs)))
    random.Random(seed).shuffle(order)
    names = list(sizes)
    out: dict[str, list[str]] = {n: [] for n in names}
    k = 0
    for idx in order:
        while k < len(names) and len(out[names[k]]) >= sizes[names[k]]:
            k += 1
        if k == len(names):
            break
        out[names[k]].extend(paragraphs[idx][1])
    return out
