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
