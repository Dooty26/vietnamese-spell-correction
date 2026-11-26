from vspell.data.viwiki import parse_viwiki_doc, sentence_spans
from vspell.data.vsec import parse_vsec_record, split_dev_test


def _ann(tokens_fixes):
    return {
        "text": " ".join(t for t, _ in tokens_fixes),
        "annotations": [
            {"id": i + 1, "current_syllable": t, "is_correct": f is None,
             "alternative_syllables": [] if f is None else (f if isinstance(f, list) else [f])}
            for i, (t, f) in enumerate(tokens_fixes)
        ],
    }


def test_vsec_basic_and_split_syllable():
    raw = _ann([("Tuy", None), ("nh", ""), ("iên,", "nhiên,"), ("chiến", None), ("lược", None),
                ("đều", "điều"), ("chỉnh", None)])
    r = parse_vsec_record(raw, 0)
    assert r["src"] == "Tuy nh iên, chiến lược đều chỉnh"
    assert r["tgt"] == "Tuy nhiên, chiến lược điều chỉnh"


def test_vsec_repairs_lost_merge_but_keeps_real_deletion():
    raw = _ann([("Thang", None), ("đo", None), ("lư", ""), ("ờng", ""), ("chất", None),
                ("về", ""), ("dịch", None), ("vụ", None)])
    r = parse_vsec_record(raw, 0)
    assert r["tgt"] == "Thang đo lường chất dịch vụ"
    assert r["n_repaired"] == 1


def test_vsec_multiple_alternatives_keep_first():
    r = parse_vsec_record(_ann([("sơ", ["cơ", ""]), ("sở", None)]), 0)
    assert r["tgt"] == "cơ sở"
    assert r["alternatives"] == {"0": ["cơ", ""]}


def test_split_dev_test_deterministic_and_disjoint():
    recs = [{"id": str(i), "tgt": f"câu {i}"} for i in range(100)]
    d1, t1 = split_dev_test(recs, 0.2, 42)
    d2, _ = split_dev_test(recs, 0.2, 42)
    assert d1 == d2 and len(d1) == 20 and len(t1) == 80
    assert not {r["id"] for r in d1} & {r["id"] for r in t1}


def test_viwiki_sentence_split_and_apply_mistakes():
    text = "Minh Hoàng địng bắt Canh. Năm này hạn hát.\nDòng mới đượcmột câu."
    doc = {"_id": "1", "text": text, "mistakes": [
        {"text": "địng", "start_offset": str(text.index("địng")), "suggest": ["định"]},
        {"text": "hát", "start_offset": str(text.index("hát")), "suggest": ["hán"]},
        {"text": "đượcmột", "start_offset": str(text.index("đượcmột")), "suggest": ["được một"]},
    ]}
    assert len(sentence_spans(text)) == 3
    recs, dropped = parse_viwiki_doc(doc)
    assert dropped == 0
    assert [r["tgt"] for r in recs] == ["Minh Hoàng định bắt Canh.", "Năm này hạn hán.", "Dòng mới được một câu."]
    assert [r["n_errors"] for r in recs] == [1, 1, 1]


def test_viwiki_split_unicode_lowercase_and_abbreviations():
    assert len(sentence_spans("Anh ấy nói v.v. ấy là ví dụ.")) == 1  # "ấ" là chữ thường
    assert len(sentence_spans("Họ đến. Ếch kêu.")) == 2              # "Ế" là chữ hoa
    assert len(sentence_spans("Ông sống ở TP. Hồ Chí Minh. Bà ở Hà Nội.")) == 2
    assert len(sentence_spans("John F. Kennedy là tổng thống. Ông mất năm 1963.")) == 2


def test_viwiki_drops_sentences_with_unusable_labels():
    text = "Đây là lổi\nchính của bài. Câu sau hoàn toàn đúng nhé."
    text = text.replace("\n", "\n")
    doc = {"_id": "2", "text": text, "mistakes": [
        {"text": "lổi\nchính", "start_offset": str(text.index("lổi")), "suggest": ["lỗi chính"]}]}
    stats = {}
    recs, dropped = parse_viwiki_doc(doc, stats=stats)
    assert dropped == 1 and stats["dropped_cross_sentence"] == 1 and stats["dropped_sentences"] == 2
    assert [r["src"] for r in recs] == ["Câu sau hoàn toàn đúng nhé."]  # không còn câu lỗi mang nhãn "sạch"


def test_split_dev_test_keeps_same_target_together():
    recs = [{"id": str(i), "tgt": f"câu {i % 30}."} for i in range(90)]  # mỗi câu đích lặp 3 lần
    recs.append({"id": "x", "tgt": "Câu  1 !"})  # khác hoa thường/khoảng trắng/dấu câu vẫn là cùng câu đích
    dev, test = split_dev_test(recs, 0.2, 42)
    from vspell.data.vsec import target_key
    norm = lambda r: target_key(r["tgt"])  # noqa: E731
    assert not {norm(r) for r in dev} & {norm(r) for r in test}
    assert len(dev) + len(test) == len(recs) and dev and test


def test_interrupted_download_leaves_no_file(tmp_path, monkeypatch):
    import urllib.request
    from pathlib import Path

    import pytest

    from vspell.data.vsec import download_vsec

    dest = tmp_path / "VSEC.jsonl"
    calls = []

    def broken(url, path):
        calls.append(path)
        Path(path).write_text('{"text": "dở', encoding="utf-8")
        raise urllib.error.ContentTooShortError("ngắt giữa chừng", None)

    monkeypatch.setattr(urllib.request, "urlretrieve", broken)
    with pytest.raises(urllib.error.ContentTooShortError):
        download_vsec(dest, url="http://x")
    assert not dest.exists() and not list(tmp_path.iterdir())

    def ok(url, path):
        Path(path).write_text('{"text": "đủ"}\n', encoding="utf-8")

    monkeypatch.setattr(urllib.request, "urlretrieve", ok)
    assert download_vsec(dest, url="http://x").read_text(encoding="utf-8") == '{"text": "đủ"}\n'
    assert len(calls) == 1
