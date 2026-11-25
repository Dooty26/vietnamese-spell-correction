from vspell.data.binhvq import clean_text, keep_sentence, split_by_paragraph, split_sentences
from vspell.data.dedup import build_blocklist, is_leaked, sentence_key


def test_clean_and_split():
    text = 'Giá vàng hôm nay giảm mạnh.<a href="x">Link</a> Theo chuyên gia, nguyên nhân là do đồng USD tăng.'
    sents = split_sentences(clean_text(text))
    assert sents[0] == "Giá vàng hôm nay giảm mạnh."
    assert all("href" not in s for s in sents)


def test_keep_sentence():
    assert keep_sentence("Thủ đoạn của bọn lừa đảo thường bắt các shipper đặt cọc trước khi giao hàng.")
    assert keep_sentence("UBND TP Hà Nội vừa ban hành quyết định mới về quản lý trật tự đô thị.")
    assert not keep_sentence("Xem thêm tại http://vnexpress.net/abc.html ngay hôm nay nhé các bạn.")
    assert not keep_sentence("ngắn quá.")
    assert not keep_sentence("chữ thường ở đầu câu thì bị loại khỏi tập dữ liệu sạch.")
    assert not keep_sentence("Michael Jackson Backstreet Boys Music Total War Bridge Steve concert tour.")


def test_split_by_paragraph_keeps_paragraphs_together():
    paras = [(f"p{i}", [f"S{i}a", f"S{i}b"]) for i in range(50)]
    out = split_by_paragraph(paras, {"test": 10, "train": 60}, seed=0)
    assert len(out["test"]) == 10 and len(out["train"]) == 60
    test_p = {s[1:-1] for s in out["test"]}
    train_p = {s[1:-1] for s in out["train"]}
    assert not test_p & train_p


def test_dedup_exact_and_near():
    recs = [{"src": "Chiến lược đều chỉnh giá xăng dầu trong tháng tới của bộ công thương",
             "tgt": "Chiến lược điều chỉnh giá xăng dầu trong tháng tới của bộ công thương"}]
    keys, grams = build_blocklist(recs, ngram=5)
    assert sentence_key("chiến lược điều chỉnh giá xăng dầu trong tháng tới của Bộ Công Thương!") in keys
    near = "Theo đó chiến lược điều chỉnh giá xăng dầu trong tháng tới của bộ công thương"
    assert is_leaked(near, keys, grams, 5, 0.8)
    assert not is_leaked("Đội tuyển Việt Nam thắng đậm trong trận đấu tối qua tại sân Mỹ Đình.", keys, grams, 5, 0.8)
