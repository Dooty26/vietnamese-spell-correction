import pytest

from vspell.models.ngram_corrector import CandidateGenerator, NgramCorrector, osa_distance, telex_decode, train
from vspell.models.ngram_lm import NgramLM, lm_tokens

CORPUS = [
    "Giá vàng hôm nay tăng mạnh.",
    "Giá vàng hôm qua giảm nhẹ.",
    "Chính phủ điều chỉnh giá xăng dầu.",
    "Bộ Công Thương điều chỉnh giá điện.",
    "Người dân được hỗ trợ trong mùa dịch.",
    "Ngân hàng điều chỉnh lãi suất cho người dân.",
    "Thủ đoạn của bọn lừa đảo thường rất tinh vi.",
    "Hàng hóa được chuyển đến các địa chỉ ở thành phố.",
] * 5
CFG = {"order": 3, "smoothing": "kneser_ney", "lexicon_min_count": 1,
       "decoding": {"beam_size": 5, "oov_penalty": -3.0, "real_word_penalty": -2.0}}


@pytest.mark.parametrize("smoothing", ["kneser_ney", "laplace"])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_lm_is_normalized(order, smoothing):
    lm = NgramLM(order, smoothing).fit(lm_tokens(s) for s in CORPUS)
    for ctx in [(), ("giá",), ("điều", "chỉnh"), ("không", "có")]:
        total = sum(10 ** lm.logprob(w, ctx) for w in lm.vocab if w != "<s>")
        assert total == pytest.approx(1.0, abs=1e-6)


def test_kneser_ney_beats_laplace_on_heldout():
    tr = [lm_tokens(s) for s in CORPUS]
    held = [lm_tokens("Giá xăng dầu hôm nay tăng nhẹ.")]
    kn = NgramLM(3, "kneser_ney").fit(tr).perplexity(held)
    lap = NgramLM(3, "laplace").fit(tr).perplexity(held)
    assert kn < lap


def test_save_load(tmp_path):
    lm = NgramLM(3).fit(lm_tokens(s) for s in CORPUS)
    lm.save(tmp_path / "lm.pkl")
    lm2 = NgramLM.load(tmp_path / "lm.pkl")
    assert lm2.logprob("vàng", ("giá",)) == lm.logprob("vàng", ("giá",))


def test_telex_and_distance():
    assert "được" in telex_decode("dduwowcj")
    assert "được" in telex_decode("đươcj")
    assert "giá" in telex_decode("gias")
    assert osa_distance("bắt", "bắy") == 1
    assert osa_distance("ab", "ba") == 1


def test_candidates():
    _, lex = train(CORPUS, CFG)
    gen = CandidateGenerator(lex)
    assert gen.candidates("vànv")[0][0] == "vàng"
    assert gen.candidates("ko") == () or all(c != "ko" for c, _, _ in gen.candidates("ko"))
    assert any(c == "được" for c, _, _ in gen.candidates("duoc"))


def test_corrector_fixes_and_keeps():
    lm, lex = train(CORPUS, CFG)
    corr = NgramCorrector(lm, lex, CFG)
    assert corr.correct_one("Giá vànv hôm nay tăng mạnh.") == "Giá vàng hôm nay tăng mạnh."
    assert corr.correct_one("Chính phủ đều chỉnh giá xăng dầu.") == "Chính phủ điều chỉnh giá xăng dầu."
    clean = "Giá vàng hôm qua giảm nhẹ."
    assert corr.correct_one(clean) == clean
    # Tên riêng/từ nước ngoài viết hoa không bị sửa
    assert "Michael" in corr.correct_one("Michael được hỗ trợ trong mùa dịch.")


def test_iy_variant():
    from vspell.models.ngram_corrector import is_iy_variant
    for a, b in [("lí", "lý"), ("kì", "kỳ"), ("kĩ", "kỹ"), ("quí", "quý"), ("Mĩ", "Mỹ")]:
        assert is_iy_variant(a, b)
    for a, b in [("tai", "tay"), ("í", "ý"), ("lí", "lỳ"), ("lí", "lí"), ("mi", "mai")]:
        assert not is_iy_variant(a, b)


def test_s4_rules_off_by_default_and_reduce_edits():
    lm, lex = train(CORPUS + ["Kỷ lục mới được lập ở thành phố."] * 5, CFG)
    base = NgramCorrector(lm, lex, CFG)
    s4cfg = {**CFG, "decoding": {**CFG["decoding"], "ignore_iy": True, "keep_non_vietnamese": True,
                                 "capital_penalty": -3.0, "edit_margin": -1.0}}
    s4 = NgramCorrector(lm, lex, s4cfg)
    # Lỗi thật vẫn được sửa
    assert s4.correct_one("Giá vànv hôm nay tăng mạnh.") == "Giá vàng hôm nay tăng mạnh."
    # i/y: mô hình cũ có ứng viên "kỷ" cho "Kỉ", luật mới thì không
    assert "kỷ" in [c for c, _ in base._options("Kỉ", True)]
    assert "kỷ" not in [c for c, _ in s4._options("Kỉ", True)]
    # Từ không dấu, sai cấu trúc âm tiết TV ("vàngs" kiểu Telex vẫn sửa được): "vanga" không bị sửa thành "vàng"
    assert "vàng" not in [c for c, _ in s4._options("vanga", False)]
    assert base.edit_margin == 0.0 and not base.ignore_iy


def test_candidates_deterministic_across_hash_seeds():
    """Ứng viên bằng chi phí và tần suất phải ra cùng thứ tự dù PYTHONHASHSEED khác (cắt ở max_candidates)."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    code = ("from vspell.models.ngram_corrector import CandidateGenerator\n"
            "lex = {w: 5 for w in ['cat', 'mat', 'hat', 'bot', 'but', 'ban', 'bao', 'bam', 'tat', 'nat']}\n"
            "print([c for c, _, _ in CandidateGenerator(lex, 2, 3).candidates('bat')])")
    src = str(Path(__file__).resolve().parents[1] / "src")
    outs = set()
    for seed in ["0", "1", "2", "3", "42"]:
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": src, "PYTHONUTF8": "1"}
        outs.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                                check=True).stdout)
    assert len(outs) == 1
