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
