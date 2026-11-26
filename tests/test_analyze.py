import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "analyze_errors", Path(__file__).resolve().parents[1] / "scripts" / "analyze_errors.py")
ae = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ae)

LEX = {"chiến", "lược", "điều", "đều", "chỉnh", "giá", "trời", "hôm", "nay", "đẹp", "ông", "đi"}


def _cats(src, pred, tgt):
    return {(x["cat"], x["sub"]) for x in ae.classify_sentence(src, pred, tgt, LEX)}


def test_four_categories():
    src = "Chiến lược đều chỉnh giá trời hôm nay đẹp"
    tgt = "Chiến lược điều chỉnh giá trời hôm nay đẹp"
    assert _cats(src, tgt, tgt) == {("sua_dung", "")}
    assert _cats(src, src, tgt) == {("bo_sot", "")}
    assert _cats(src, "Chiến lược đầu chỉnh giá trời hôm nay đẹp", tgt) == {("sua_sai", "")}
    assert ("sua_qua_tay", "tu_thuong") in _cats(src, "Chiến lược điều chỉnh giá trời hôm nay đẹt", tgt)


def test_overcorrection_subgroups():
    toks = ["Ông", "Nero", "đi", "manga", "năm", "1990", "Hôm"]
    kinds = [ae.overcorrection_kind(toks, i, LEX) for i in range(len(toks))]
    # token đầu câu viết hoa nhưng có trong từ điển -> từ thường; "Hôm" giữa câu -> tên riêng/viết hoa
    assert kinds == ["tu_thuong", "ten_rieng", "tu_thuong", "ngoai_tu_dien", "ngoai_tu_dien",
                     "so_ky_hieu", "ten_rieng"]


def test_real_error_kind():
    assert ae.real_error_kind("đều", "điều") == "Sai 1 chữ cái (thêm/bớt/thay/đảo)"
    assert ae.real_error_kind("sanh", "xanh") == "Sai 1 chữ cái (thêm/bớt/thay/đảo)"
    assert ae.real_error_kind("chặn", "chận") == "Sai dấu phụ (ă â ê ô ơ ư đ)"
    assert ae.real_error_kind("bảo", "báo") == "Chỉ sai dấu thanh"
    assert ae.real_error_kind("c", "") == "Tách/ghép âm tiết"


def _per_sent(rows):
    recs = [dict(id=i, src=s, pred=p, tgt=t) for i, s, p, t in rows]
    return recs, [ae.classify_sentence(r["src"], r["pred"], r["tgt"], LEX) for r in recs]


def test_comparison_pairs_by_id_not_order():
    import pytest

    a = [(1, "đều chỉnh", "điều chỉnh", "điều chỉnh"), (2, "trời đẹp", "trời đẹp", "trời đẹp")]
    b = [(1, "đều chỉnh", "đều chỉnh", "điều chỉnh"), (2, "trời đẹp", "trời đẹp", "trời đẹp")]
    ps = {("A", "vsec_test"): _per_sent(a), ("B", "vsec_test"): _per_sent(b)}
    ps_rev = {("A", "vsec_test"): _per_sent(a), ("B", "vsec_test"): _per_sent(b[::-1])}

    def groups(p):
        return {r["group"]: r["n"] for r in ae.comparison_rows(p, ["A", "B"]) if r["n"]}

    assert groups(ps) == groups(ps_rev) == {"A_dung_B_sai": 1}
    with pytest.raises(ValueError, match="không cùng tập câu"):  # file B chạy với --limit
        ae.comparison_rows({("A", "vsec_test"): _per_sent(a), ("B", "vsec_test"): _per_sent(b[:1])}, ["A", "B"])
    bad = [(1, "đều chỉnh", "đều chỉnh", "đều chỉnh"), b[1]]
    with pytest.raises(ValueError, match="khác nhau"):
        ae.comparison_rows({("A", "vsec_test"): _per_sent(a), ("B", "vsec_test"): _per_sent(bad)}, ["A", "B"])


def test_viwiki_hypothesis_identity_model_no_zero_division(tmp_path):
    """Mô hình đồng nhất (không sửa gì) -> n_edits = 0; không được ZeroDivisionError."""
    rows = [(1, "đều chỉnh giá", "đều chỉnh giá", "điều chỉnh giá"), (2, "trời đẹp", "trời đẹp", "trời đẹp")]
    ps = {("identity", "viwiki_test"): _per_sent(rows)}
    out = {r["metric"]: r["value"] for r in ae.viwiki_hypothesis(ps, "identity", LEX, tmp_path)}
    assert out["n_edits"] == 0 and out["cor_p"] == out["cor_r"] == out["cor_f1"] == 0
    assert (tmp_path / "overcorrection_viwiki.csv").exists()
    ae.viwiki_hypothesis({("identity", "viwiki_test"): _per_sent(rows[1:])}, "identity", LEX, tmp_path)  # 0 lỗi thật


def test_breakdown_checks_predictions_against_gold(tmp_path):
    import json

    import pytest

    gold = [dict(id=i, src="đều chỉnh", tgt="điều chỉnh") for i in range(3)]
    data, pred = tmp_path / "data", tmp_path / "pred"
    data.mkdir(), pred.mkdir()

    def write(path, recs):
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")

    write(data / "vsec_test.jsonl", gold)
    full = [dict(r, pred=r["tgt"]) for r in gold]
    write(pred / "A__vsec_test.jsonl", full[:2])   # A và B cùng chạy --limit 2: khớp nhau nhưng thiếu câu
    write(pred / "B__vsec_test.jsonl", full[:2])
    with pytest.raises(ValueError, match="thiếu 1"):
        ae.run_breakdown(pred, ["A", "B"], LEX, data)
    ae.run_breakdown(pred, ["A", "B"], LEX, None)  # --allow-mismatch
    write(pred / "A__vsec_test.jsonl", full)
    write(pred / "B__vsec_test.jsonl", full[::-1])
    rows, per_sent = ae.run_breakdown(pred, ["A", "B"], LEX, data)
    assert set(per_sent) == {("A", "vsec_test"), ("B", "vsec_test")}


def test_missing_lexicon_is_rejected(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="--lexicon"):
        ae.load_lexicon(tmp_path / "lexicon.json")
    # có từ điển thì "học" là từ thường, không phải "ngoài từ điển"
    assert ("sua_qua_tay", "tu_thuong") in {(x["cat"], x["sub"]) for x in
                                            ae.classify_sentence("tôi học", "tôi hỏi", "tôi học", {"tôi", "học"})}
