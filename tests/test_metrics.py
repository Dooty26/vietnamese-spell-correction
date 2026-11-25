from vspell.eval.align import error_positions
from vspell.eval.metrics import evaluate, per_error_type
from vspell.normalize import split_syllables

SRCS = ["Chiến lược đều chỉnh giá", "Trời hôm nay đẹp"]
TGTS = ["Chiến lược điều chỉnh giá", "Trời hôm nay đẹp"]


def test_identity_model():
    m = evaluate(SRCS, SRCS, TGTS)
    assert m["cor_r"] == 0.0
    assert m["over_correction"] == 0.0
    assert m["n_errors"] == 1 and m["n_edits"] == 0


def test_perfect_model():
    m = evaluate(SRCS, TGTS, TGTS)
    assert m["cor_p"] == m["cor_r"] == m["cor_f1"] == 1.0
    assert m["sent_acc"] == 1.0


def test_detected_but_wrong_and_overcorrection():
    preds = ["Chiến lược đìu chỉnh giá", "Trời hôm này đẹp"]
    m = evaluate(SRCS, preds, TGTS)
    assert m["det_r"] == 1.0 and m["det_p"] == 0.5
    assert m["cor_r"] == 0.0
    assert m["over_correction"] == 1.0


def test_tone_placement_difference_is_not_error():
    m = evaluate(["hoà bình"], ["hòa bình"], ["hòa bình"])
    assert m["n_errors"] == 0 and m["n_edits"] == 0


def test_alignment_split_syllable():
    s, t = split_syllables("Tuy nh iên"), split_syllables("Tuy nhiên")
    assert error_positions(s, t) == {1: "nhiên", 2: ""}


def test_per_error_type():
    recs = [{"src": SRCS[0], "tgt": TGTS[0], "errors": [{"pos": 2, "type": "region"}]}]
    out = per_error_type(recs, [TGTS[0]])
    assert out == {"region": {"n": 1, "det_r": 1.0, "cor_r": 1.0}}
