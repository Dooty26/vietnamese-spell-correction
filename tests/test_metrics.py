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


def test_alignment_repeated_phrase_counts_one_error():
    """difflib khớp cụm "tôi học" dài nhất sai chỗ -> 3 thay đổi; căn chỉnh theo khoảng cách sửa chỉ 1."""
    s, t = split_syllables("tôi học tôi học"), split_syllables("tôi hỏi tôi học")
    assert error_positions(s, t) == {1: "hỏi"}
    m = evaluate(["tôi học tôi học"], ["tôi hỏi tôi học"], ["tôi hỏi tôi học"])
    assert m["n_errors"] == 1 and m["cor_f1"] == 1.0


def test_alignment_repeated_with_split_and_merge():
    s = split_syllables("anh đi anh đi nh iên")
    assert error_positions(s, split_syllables("anh đi ảnh đi nhiên")) == {2: "ảnh", 4: "nhiên", 5: ""}
    s2 = split_syllables("ba ba nhiên ba")
    assert error_positions(s2, split_syllables("ba ba nh iên ba")) == {2: "nh iên"}


def test_alignment_insert_delete_and_edges():
    from vspell.eval.align import opcodes
    s = split_syllables("một hai ba")
    assert error_positions(s, split_syllables("một hai ba bốn")) == {2: "ba bốn"}
    assert error_positions(s, split_syllables("không một hai ba")) == {0: "không một"}
    assert error_positions(s, split_syllables("một ba")) == {1: ""}
    assert error_positions(s, s) == {} and error_positions([], []) == {}
    assert opcodes(["a", "b"], ["a", "b"]) == [("equal", 0, 2, 0, 2)]
    assert opcodes(["x"], ["y", "z"]) == [("replace", 0, 1, 0, 2)]
