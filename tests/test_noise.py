import random

import pytest

from vspell.eval.metrics import per_error_type
from vspell.noise.generator import NoiseGenerator
from vspell.noise.rules import RULES, region_error, telex_error
from vspell.normalize import is_vietnamese_syllable, split_syllables
from vspell.utils import load_yaml

CFG = load_yaml("configs/noise.yaml")
SENT = "Chợ đêm Ninh Kiều mang nét đặc trưng của miền sông nước, giá 40.000 đồng theo UBND TP.HCM."


def test_keeps_token_count_and_is_reproducible():
    for seed in range(20):
        a, ea = NoiseGenerator(CFG, seed=seed).corrupt(SENT, force=True)
        b, eb = NoiseGenerator(CFG, seed=seed).corrupt(SENT, force=True)
        assert a == b and ea == eb
        assert len(split_syllables(a)) == len(split_syllables(SENT))
        assert ea, "câu bị ép làm nhiễu phải có ít nhất 1 lỗi"


def test_errors_are_at_reported_positions_and_skip_non_vietnamese():
    protected = {"40.000", "UBND", "TP", "HCM", ",", "."}
    for seed in range(50):
        src, errs = NoiseGenerator(CFG, seed=seed).corrupt(SENT, force=True)
        s, t = split_syllables(src), split_syllables(SENT)
        for e in errs:
            assert t[e["pos"]] == e["orig"] and s[e["pos"]] == e["noisy"]
        for i, tok in enumerate(t):
            if tok in protected:
                assert s[i] == tok


def test_clean_ratio_and_per_error_type_on_gold():
    pairs = NoiseGenerator(CFG, seed=0).make_pairs([SENT] * 400)
    clean = sum(not p["errors"] for p in pairs) / len(pairs)
    assert 0.08 < clean < 0.25
    out = per_error_type(pairs, [p["tgt"] for p in pairs])
    assert all(v["cor_r"] == 1.0 for v in out.values())


@pytest.mark.parametrize("name", list(RULES))
def test_rules_change_syllable_or_return_none(name):
    rng = random.Random(0)
    for w in ["được", "người", "Trường", "sông", "không", "nghiêng", "quá", "giữ", "Hà"]:
        out = RULES[name](w, rng)
        assert out is None or (out != w and out.strip() == out and " " not in out)


def test_region_makes_valid_syllables_and_telex_examples():
    rng = random.Random(1)
    for w in ["sông", "trường", "lúc", "tiếng", "mãi", "được"]:
        out = region_error(w, rng)
        assert out is None or is_vietnamese_syllable(out)
    outs = {telex_error("được", random.Random(s)) for s in range(30)}
    assert "đươcj" in outs


def test_tone_and_diacritic_rules():
    import random
    from vspell.normalize import remove_diacritics, split_tone
    from vspell.noise.rules import diacritic_error, tone_error
    rng = random.Random(0)
    for w in ["đóng", "Buổi", "giảm", "người", "hoà", "khác"]:
        t = tone_error(w, rng)
        assert t and t != w and split_tone(t)[0] == split_tone(w)[0]  # chỉ khác dấu thanh
        d = diacritic_error(w, rng)
        assert d and d != w and remove_diacritics(d) == remove_diacritics(w)
        assert split_tone(d)[1] == split_tone(w)[1]  # giữ dấu thanh
    assert tone_error("x", rng) is None
