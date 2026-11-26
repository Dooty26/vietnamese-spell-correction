"""Test mô hình C bằng một BART tí hon khởi tạo ngẫu nhiên (không tải vinai/bartpho-syllable, không cần mạng)."""
import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("datasets")
transformers = pytest.importorskip("transformers")

from vspell.models.bartpho import BartphoCorrector, build_datasets, load_pairs, train  # noqa: E402

PAIRS = [
    ("Giá vàg hôm nay tăng mạnh.", "Giá vàng hôm nay tăng mạnh."),
    ("Chính phũ điều chỉnh giá xăng.", "Chính phủ điều chỉnh giá xăng."),
    ("Người dân đươc hỗ trợ.", "Người dân được hỗ trợ."),
    ("Hàng hoá được chuyển đi.", "Hàng hóa được chuyển đi."),
] * 4


def _write_jsonl(path, pairs):
    with open(path, "w", encoding="utf-8") as f:
        for s, t in pairs:
            f.write(json.dumps({"src": s, "tgt": t}, ensure_ascii=False) + "\n")
    return str(path)


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """Thư mục chứa tokenizer WordLevel + BartForConditionalGeneration rất nhỏ."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import BartConfig, BartForConditionalGeneration, PreTrainedTokenizerFast

    from vspell.normalize import normalize, split_syllables

    specials = ["<s>", "<pad>", "</s>", "<unk>"]
    vocab = {t: i for i, t in enumerate(specials)}
    for s, t in PAIRS:
        for w in split_syllables(normalize(s)) + split_syllables(normalize(t)):
            vocab.setdefault(w, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()])
    tok.post_processor = processors.TemplateProcessing(single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)])
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", eos_token="</s>",
                                     pad_token="<pad>", unk_token="<unk>")
    cfg = BartConfig(vocab_size=len(vocab), d_model=16, encoder_layers=1, decoder_layers=1,
                     encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32,
                     decoder_ffn_dim=32, max_position_embeddings=64, pad_token_id=1, bos_token_id=0,
                     eos_token_id=2, decoder_start_token_id=2, forced_eos_token_id=2)
    d = tmp_path_factory.mktemp("tiny_bart")
    hf_tok.save_pretrained(d)
    BartForConditionalGeneration(cfg).save_pretrained(d)
    return str(d)


def test_load_pairs_normalizes_and_limits(tmp_path):
    p = _write_jsonl(tmp_path / "a.jsonl", PAIRS)
    pairs = load_pairs(p, limit=5)
    assert len(pairs) == 5
    assert pairs[3]["src"] == "Hàng hóa được chuyển đi."  # "hoá" -> "hóa"


def test_build_datasets(tmp_path, tiny_model):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_model)
    tr = _write_jsonl(tmp_path / "tr.jsonl", PAIRS)
    va = _write_jsonl(tmp_path / "va.jsonl", PAIRS[:4])
    train_ds, val_ds = build_datasets(tr, va, tok, {"max_source_length": 5, "max_target_length": 6,
                                                    "train_limit": 10})
    assert len(train_ds) == 10 and len(val_ds) == 4
    assert set(train_ds.column_names) >= {"input_ids", "attention_mask", "labels"}
    assert max(len(x) for x in train_ds["input_ids"]) <= 5
    assert max(len(x) for x in train_ds["labels"]) <= 6
    assert train_ds[0]["labels"][-1] == tok.eos_token_id


def test_train_and_correct(tmp_path, tiny_model):
    tr = _write_jsonl(tmp_path / "tr.jsonl", PAIRS)
    va = _write_jsonl(tmp_path / "va.jsonl", PAIRS[:4])
    cfg = {"model_name": tiny_model, "output_dir": str(tmp_path / "ckpt"), "max_steps": 4,
           "per_device_train_batch_size": 4, "per_device_eval_batch_size": 4, "eval_steps": 2,
           "save_steps": 2, "learning_rate": 1e-3, "logging_steps": 1, "fp16": True}
    best = train(cfg, tr, va)
    info = json.loads((tmp_path / "ckpt" / "best" / "vspell_config.json").read_text(encoding="utf-8"))
    assert info["global_step"] == 4 and info["best_checkpoint"] is not None
    prov = info["provenance"]  # hồ sơ tái lập: commit, thư viện, hash dữ liệu
    assert prov["packages"]["transformers"] and prov["python"]
    from vspell.utils import sha256_file
    assert prov["files_sha256"] == {"train": sha256_file(tr), "val": sha256_file(va)}
    assert info["model_commit"] is None  # mô hình nền là thư mục cục bộ

    corr = BartphoCorrector(best, num_beams=2)
    sents = [s for s, _ in PAIRS[:4]] + ["Hàng hoá"]
    out = corr.correct(sents, batch_size=2)
    assert len(out) == len(sents) and all(isinstance(o, str) and o for o in out)
    from vspell.normalize import normalize
    assert all(o == normalize(o) for o in out)


def test_to_bartpho_style_and_postprocess():
    from vspell.models.bartpho import postprocess, to_bartpho_style
    from vspell.normalize import normalize

    s = "Hòa thỏa khỏe Thụy tùy ủy hoàng toán quý của mùa người khuỷu Khóa"
    assert to_bartpho_style(s) == "Hoà thoả khoẻ Thuỵ tuỳ uỷ hoàng toán quý của mùa người khuỷu Khoá"
    assert normalize(to_bartpho_style(s)) == s
    assert postprocess("a…", "a...") == "a…"
    assert postprocess("a...", "a...") == "a..."


def test_split_long_keeps_all_words():
    from vspell.models.bartpho import split_long

    n = lambda t: len(t.split()) + 2  # noqa: E731  (2 token đặc biệt)
    text = "một hai ba , bốn năm sáu bảy . tám chín mười mười một"
    assert split_long("ngắn thôi", n, 10) == ["ngắn thôi"]
    parts = split_long(text, n, 7)
    assert " ".join(parts) == text
    assert all(n(p) <= 7 for p in parts)
    assert parts[0].endswith(",") or parts[0].endswith(".")  # ưu tiên cắt sau dấu câu


def test_correct_long_sentence_not_truncated(tiny_model):
    """Câu vượt max_length phải được chia đoạn: với mô hình 'chép lại' đầu vào, đầu ra phải đủ cả câu."""
    corr = BartphoCorrector(tiny_model, num_beams=1, max_length=8)
    corr.model.generate = lambda input_ids, **kw: input_ids  # mô hình đồng nhất
    long = " ".join(s for s, _ in PAIRS[:3])  # 3 câu ~ 20 token, gấp ~2,5 lần giới hạn
    short = PAIRS[4][0]
    out = corr.correct([long, short], batch_size=2)
    from vspell.normalize import normalize, split_syllables
    # tokenizer tí hon giải mã có dấu cách trước dấu câu -> so theo âm tiết
    assert [split_syllables(o) for o in out] == [split_syllables(normalize(x)) for x in (long, short)]


def test_compute_metrics_uses_full_targets(tiny_model):
    """Nhãn bị cắt ở max_target_length + dự đoán mất đuôi câu không được chấm là đúng hoàn toàn."""
    import numpy as np
    from transformers import AutoTokenizer

    from vspell.models.bartpho import _make_compute_metrics

    tok = AutoTokenizer.from_pretrained(tiny_model)
    full = "Giá vàg hôm nay tăng mạnh."
    cut = tok([full], max_length=4, truncation=True)["input_ids"]  # "<s> Giá vàg </s>"
    preds = labels = np.array(cut)
    m = _make_compute_metrics(tok, [full], [full])((preds, labels))
    assert m["sent_acc"] == 0.0
    whole = np.array(tok([full])["input_ids"])
    assert _make_compute_metrics(tok, [full], [full])((whole, whole))["sent_acc"] == 1.0


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_load_pairs_rejects_non_positive_limit(tmp_path, bad):
    p = _write_jsonl(tmp_path / "a.jsonl", PAIRS)
    with pytest.raises(ValueError, match="limit"):
        load_pairs(p, limit=bad)
    assert len(load_pairs(p, limit=None)) == len(PAIRS)


def test_train_cli_rejects_zero_limit():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "-X", "utf8", str(root / "scripts" / "train_bartpho.py"), "--limit", "0"],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 2 and "--limit" in r.stderr
