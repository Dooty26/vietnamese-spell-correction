"""Mô hình C: fine-tune vinai/bartpho-syllable (seq2seq câu lỗi -> câu đúng).

Huấn luyện chạy trên Kaggle/Colab (GPU), xem notebooks/04_bartpho.ipynb và scripts/train_bartpho.py.
Mọi câu (src, tgt, đầu ra) đều qua `vspell.normalize.normalize` để khớp quy ước chấm điểm.

`transformers`/`torch` chỉ được import bên trong hàm, để phần còn lại của package không phụ thuộc chúng.
"""
from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import re
import unicodedata

from vspell.normalize import normalize, parse_syllable, split_tone
from vspell.utils import get_logger, read_jsonl

log = get_logger()

# ---------------------------------------------------------------- kiểu đặt dấu cho tokenizer

_WORD_RE = re.compile(r"[^\W\d_]+")
_NEW_STYLE_NUCLEI = {"oa", "oe", "uy"}


def _new_style_word(word: str) -> str:
    base, tone = split_tone(word)
    if tone is None:
        return word
    parts = parse_syllable(base.lower())
    if parts is None or parts[2] or parts[1] not in _NEW_STYLE_NUCLEI:
        return word
    i = len(parts[0]) + 1  # nguyên âm thứ hai của "oa"/"oe"/"uy" (âm tiết mở)
    return base[:i] + unicodedata.normalize("NFC", unicodedata.normalize("NFD", base[i]) + tone) + base[i + 1:]


def to_bartpho_style(text: str) -> str:
    """Đổi dấu thanh của vần mở "oa/oe/uy" sang kiểu mới ("hòa" -> "hoà", "thủy" -> "thuỷ", "khỏe" -> "khoẻ").

    Từ điển sentencepiece của vinai/bartpho-syllable không có kiểu cũ: "hòa", "thỏa", "khóa", "tùy", "ủy",
    "khỏe"... đều thành <unk> (~0,35% âm tiết của VSEC/Syn-val), mô hình không sinh lại được và bị xóa khi
    giải mã. Chỉ dùng cho đầu vào tokenizer; đầu ra qua `normalize` để trở lại quy ước chấm điểm.
    """
    return _WORD_RE.sub(lambda m: _new_style_word(m.group(0)), text)


def postprocess(src: str, text: str) -> str:
    """Đầu ra mô hình -> quy ước chấm điểm. Tokenizer đổi "…" thành "...": trả lại nếu câu vào có "…"."""
    if "…" in src and "..." in text:
        text = text.replace("...", "…")
    return normalize(text)


# ---------------------------------------------------------------- dữ liệu

def load_pairs(path: str | Path, limit: int | None = None) -> list[dict]:
    """Đọc JSONL -> [{"src", "tgt"}] đã chuẩn hóa. `limit`: chỉ lấy N cặp đầu."""
    pairs = []
    for r in read_jsonl(path):
        pairs.append({"src": normalize(r["src"]), "tgt": normalize(r["tgt"])})
        if limit is not None and len(pairs) >= limit:
            break
    return pairs


def build_datasets(train_path: str, val_path: str, tokenizer, config: dict):
    """Đọc JSONL -> (train, val) là `datasets.Dataset` đã tokenize (src -> input_ids, tgt -> labels).

    Khóa config dùng: max_source_length, max_target_length (mặc định 128),
    train_limit / val_limit (None = lấy hết; `--limit` của script đặt cả hai).
    """
    from datasets import Dataset

    max_src = config.get("max_source_length", 128)
    max_tgt = config.get("max_target_length", 128)

    def tokenize(batch):
        enc = tokenizer([to_bartpho_style(x) for x in batch["src"]], max_length=max_src, truncation=True)
        enc["labels"] = tokenizer(text_target=[to_bartpho_style(x) for x in batch["tgt"]], max_length=max_tgt,
                                  truncation=True)["input_ids"]
        return enc

    out = []
    for path, limit in ((train_path, config.get("train_limit")), (val_path, config.get("val_limit"))):
        ds = Dataset.from_list(load_pairs(path, limit))
        ds = ds.map(tokenize, batched=True, remove_columns=["src", "tgt"], desc=f"tokenize {Path(path).name}")
        out.append(ds)
    log.info("Dữ liệu: train=%d cặp, val=%d cặp", len(out[0]), len(out[1]))
    return out[0], out[1]


# ---------------------------------------------------------------- huấn luyện

def _filter_kwargs(cls, kwargs: dict) -> dict:
    """Giữ các tham số mà `cls.__init__` nhận; đổi tên cho các bản transformers khác nhau."""
    params = inspect.signature(cls.__init__).parameters
    if "eval_strategy" in kwargs and "eval_strategy" not in params:  # transformers < 4.41
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    if "processing_class" in kwargs and "processing_class" not in params:  # transformers < 4.46
        kwargs["tokenizer"] = kwargs.pop("processing_class")
    if "warmup_ratio" in kwargs and "warmup_ratio" not in params:  # transformers 5: warmup_steps nhận tỉ lệ < 1
        ratio = kwargs.pop("warmup_ratio")
        if ratio and not kwargs.get("warmup_steps"):
            kwargs["warmup_steps"] = float(ratio)
    if "group_by_length" in kwargs and "group_by_length" not in params:  # transformers 5
        if kwargs.pop("group_by_length") and "train_sampling_strategy" in params:
            kwargs["train_sampling_strategy"] = "group_by_length"
    dropped = [k for k in kwargs if k not in params]
    if dropped:
        log.warning("%s không nhận tham số %s, bỏ qua", cls.__name__, dropped)
    return {k: v for k, v in kwargs.items() if k in params}


def _make_compute_metrics(tokenizer, val_srcs: list[str], val_tgts: list[str]):
    """compute_metrics cho metric_for_best_model = cor_f1 (cần predict_with_generate).

    Chấm với câu đích gốc `val_tgts`, không giải mã lại `labels`: nhãn đã bị tokenizer cắt ở
    max_target_length, nên nếu so với nhãn cắt thì một dự đoán mất đuôi câu vẫn được tính là đúng hoàn toàn.
    """
    import numpy as np

    from vspell.eval.metrics import evaluate

    def compute(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
        srcs = val_srcs[:len(preds)]
        hyp = [postprocess(src, text) for src, text in zip(srcs, tokenizer.batch_decode(preds, skip_special_tokens=True))]
        m = evaluate(srcs, hyp, val_tgts[:len(hyp)])
        return {k: m[k] for k in ("cor_f1", "det_f1", "sent_acc")}

    return compute


def train(config: dict, train_path: str, val_path: str) -> str:
    """Huấn luyện bằng Seq2SeqTrainer; chọn checkpoint tốt nhất theo Syn-val; lưu vào {output_dir}/best.

    Trả về đường dẫn thư mục `best/` (model + tokenizer + vspell_config.json).
    """
    import torch
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments)

    out_dir = Path(config["output_dir"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"])
    train_ds, val_ds = build_datasets(train_path, val_path, tokenizer, config)

    cuda = torch.cuda.is_available()
    metric = config.get("metric_for_best_model", "eval_loss")
    gen_eval = metric != "eval_loss"
    if gen_eval and val_ds.num_rows > config.get("gen_eval_size", 1000):
        # sinh câu trên cả 5k câu mỗi lần eval thì chậm; lấy tập con đầu Syn-val
        val_ds = val_ds.select(range(config.get("gen_eval_size", 1000)))

    args = dict(
        output_dir=str(out_dir),
        seed=config.get("seed", 42),
        num_train_epochs=config.get("num_train_epochs", 3),
        max_steps=config.get("max_steps") or -1,
        per_device_train_batch_size=config.get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 32),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        learning_rate=float(config.get("learning_rate", 5e-5)),
        warmup_ratio=config.get("warmup_ratio", 0.0),
        weight_decay=config.get("weight_decay", 0.0),
        lr_scheduler_type=config.get("lr_scheduler_type", "linear"),
        fp16=bool(config.get("fp16", True)) and cuda,
        eval_strategy="steps",
        eval_steps=config.get("eval_steps", 2000),
        save_strategy="steps",
        save_steps=config.get("save_steps", 2000),
        save_total_limit=config.get("save_total_limit", 2),
        # True: checkpoint chỉ gồm trọng số (~1,6 GB thay vì ~4,8 GB có optimizer); không resume được optimizer
        save_only_model=bool(config.get("save_only_model", False)),
        logging_steps=config.get("logging_steps", 100),
        load_best_model_at_end=True,
        metric_for_best_model=metric,
        greater_is_better=gen_eval,
        predict_with_generate=gen_eval,
        generation_max_length=config.get("max_target_length", 128),
        generation_num_beams=1,
        dataloader_num_workers=config.get("dataloader_num_workers", 2) if cuda else 0,
        group_by_length=config.get("group_by_length", True),
        report_to=config.get("report_to", "none"),
        use_cpu=not cuda,
    )
    if cuda:
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:  # vd Kaggle T4 x2: Trainer chạy DataParallel -> giữ batch hiệu dụng như config
            effective = args["per_device_train_batch_size"] * args["gradient_accumulation_steps"]
            args["gradient_accumulation_steps"] = max(1, effective // (args["per_device_train_batch_size"] * n_gpu))
        log.info("GPU: %d x %s, fp16=%s, batch hiệu dụng=%d", n_gpu, torch.cuda.get_device_name(0), args["fp16"],
                 args["per_device_train_batch_size"] * args["gradient_accumulation_steps"] * n_gpu)
    else:
        log.warning("Không có GPU: train trên CPU (chỉ để chạy thử)")
    training_args = Seq2SeqTrainingArguments(**_filter_kwargs(Seq2SeqTrainingArguments, args))

    trainer_kwargs = dict(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        processing_class=tokenizer,
    )
    if gen_eval:
        val_pairs = load_pairs(val_path, val_ds.num_rows)  # cùng thứ tự với val_ds (eval không xáo trộn)
        trainer_kwargs["compute_metrics"] = _make_compute_metrics(tokenizer, [p["src"] for p in val_pairs],
                                                                  [p["tgt"] for p in val_pairs])
    trainer = Seq2SeqTrainer(**_filter_kwargs(Seq2SeqTrainer, trainer_kwargs))

    resume = config.get("resume_from_checkpoint")
    if resume is True or resume == "auto":
        resume = any(out_dir.glob("checkpoint-*")) or None
    trainer.train(resume_from_checkpoint=resume)

    metrics = trainer.evaluate()
    log.info("Syn-val (mô hình tốt nhất): %s", metrics)
    best = out_dir / "best"
    trainer.save_model(str(best))
    tokenizer.save_pretrained(str(best))
    info = {"model_name": config["model_name"], "train_path": str(train_path), "val_path": str(val_path),
            "best_checkpoint": trainer.state.best_model_checkpoint, "best_metric": trainer.state.best_metric,
            "metric_for_best_model": metric, "global_step": trainer.state.global_step,
            "final_eval": {k: (v if not isinstance(v, float) or math.isfinite(v) else None)
                           for k, v in metrics.items()},
            "config": config}
    (best / "vspell_config.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str),
                                             encoding="utf-8")
    log.info("Đã lưu mô hình tốt nhất vào %s", best)
    return str(best)


# ---------------------------------------------------------------- suy luận

_CUT_AFTER = (".", ",", ";", ":", "!", "?", "…", ")", '"')


def split_long(text: str, n_tokens, max_tokens: int) -> list[str]:
    """Chia câu dài thành các đoạn (tách tại khoảng trắng) sao cho mỗi đoạn có `n_tokens(đoạn) <= max_tokens`.

    Ghép lại bằng " " ra đúng câu vào (câu đã `normalize` nên khoảng trắng là một dấu cách). Ưu tiên cắt sau
    dấu câu nếu điểm cắt đó nằm ở nửa sau của đoạn. Một từ đơn lẻ dài hơn giới hạn vẫn thành một đoạn riêng
    (tokenizer sẽ cắt nó). Câu ngắn trả về nguyên [text].
    """
    if n_tokens(text) <= max_tokens:
        return [text]
    chunks: list[str] = []
    cur: list[str] = []
    for w in text.split(" "):
        if cur and n_tokens(" ".join(cur + [w])) > max_tokens:
            cut = max((k for k in range(len(cur) // 2, len(cur)) if cur[k].endswith(_CUT_AFTER)), default=None)
            if cut is None or cut == len(cur) - 1:
                chunks.append(" ".join(cur))
                cur = []
            else:
                chunks.append(" ".join(cur[:cut + 1]))
                cur = cur[cut + 1:]
            # phần còn lại cộng từ mới vẫn có thể quá dài -> đẩy tiếp
            while cur and n_tokens(" ".join(cur + [w])) > max_tokens:
                chunks.append(" ".join(cur))
                cur = []
        cur.append(w)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


class BartphoCorrector:
    """Sửa lỗi theo batch bằng beam search. Đầu ra đã qua `normalize`.

    max_length: độ dài tối đa (token) của câu vào; câu ra được phép dài hơn câu vào một chút
    (`len_ratio`, `len_extra`). Câu vào dài hơn `max_length` được chia đoạn (`split_long`), sửa từng đoạn
    rồi ghép lại. Các đoạn được sắp theo độ dài để giảm padding, rồi trả về đúng thứ tự.
    """

    def __init__(self, checkpoint: str, num_beams: int = 4, max_length: int = 256, device: str | None = None,
                 len_ratio: float = 1.3, len_extra: int = 10, fp16: bool | None = None):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if checkpoint is None:
            raise ValueError("Cần --checkpoint cho bartpho (vd outputs/checkpoints/bartpho/best)")
        self.checkpoint = checkpoint
        self.num_beams = num_beams
        self.max_length = max_length
        self.len_ratio = len_ratio
        self.len_extra = len_extra
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        # Nạp fp32 rồi mới .half() khi chạy GPU: checkpoint có thể lưu fp16 (vd v4 xuất từ Colab cho nhẹ),
        # còn fp16 trên CPU vừa chậm vừa thiếu kernel.
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, dtype=torch.float32).to(self.device).eval()
        if fp16 if fp16 is not None else self.device.startswith("cuda"):
            self.model.half()

    def correct(self, sentences: list[str], batch_size: int = 32) -> list[str]:
        import torch
        from tqdm import tqdm

        srcs = [normalize(s) for s in sentences]
        # Câu dài hơn max_length token: chia thành đoạn, sửa từng đoạn rồi ghép lại (không cắt mất phần cuối).
        n_tok = lambda t: len(self.tokenizer(to_bartpho_style(t))["input_ids"])  # noqa: E731
        pieces: list[str] = []
        owner: list[int] = []
        for i, src in enumerate(srcs):
            for part in split_long(src, n_tok, self.max_length):
                pieces.append(part)
                owner.append(i)
        order = sorted(range(len(pieces)), key=lambda i: len(pieces[i]), reverse=True)
        fixed: list[str | None] = [None] * len(pieces)
        for start in tqdm(range(0, len(order), batch_size), desc="bartpho", unit="batch"):
            idx = order[start:start + batch_size]
            batch = [to_bartpho_style(pieces[i]) for i in idx]
            enc = self.tokenizer(batch, max_length=self.max_length, truncation=True,
                                 padding=True, return_tensors="pt").to(self.device)
            max_new = int(enc["input_ids"].shape[1] * self.len_ratio) + self.len_extra
            with torch.no_grad():
                gen = self.model.generate(**enc, num_beams=self.num_beams, max_new_tokens=max_new,
                                          early_stopping=True)
            for i, text in zip(idx, self.tokenizer.batch_decode(gen, skip_special_tokens=True)):
                fixed[i] = postprocess(pieces[i], text) if text.strip() else pieces[i]  # rỗng -> giữ đoạn vào
        parts: list[list[str]] = [[] for _ in srcs]
        for i, text in zip(owner, fixed):
            parts[i].append(text)  # type: ignore[arg-type]
        out = [" ".join(p) for p in parts]
        return out  # type: ignore[return-value]
