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

    corr = BartphoCorrector(best, num_beams=2)
    sents = [s for s, _ in PAIRS[:4]] + ["Hàng hoá"]
    out = corr.correct(sents, batch_size=2)
    assert len(out) == len(sents) and all(isinstance(o, str) and o for o in out)
    from vspell.normalize import normalize
    assert all(o == normalize(o) for o in out)
