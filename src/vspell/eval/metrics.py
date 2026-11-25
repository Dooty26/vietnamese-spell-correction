"""Chỉ số đánh giá mức âm tiết.

- Detection P/R/F1: vị trí mô hình thay đổi so với vị trí lỗi thật (theo chỉ số âm tiết src).
- Correction P/R/F1: chỉ tính đúng khi phát hiện đúng VÀ chuỗi sửa khớp tgt.
    P = số lỗi sửa đúng / số vị trí mô hình thay đổi;  R = số lỗi sửa đúng / số lỗi thật.
- Sentence accuracy: dự đoán == tgt (sau normalize, so theo token).
- Over-correction rate: tỉ lệ câu sạch (src == tgt) bị mô hình thay đổi.

Mọi chuỗi đều được normalize trước khi so (khác biệt vị trí dấu thanh không tính là lỗi).
Kiểm tra nhanh: "mô hình đồng nhất" (pred = src) cho Correction Recall = 0 và over-correction = 0.
Phụ trách: TV2.
"""
from __future__ import annotations

from collections import defaultdict

from vspell.eval.align import edit_positions, error_positions
from vspell.normalize import normalize, split_syllables


def _div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _f1(p: float, r: float) -> float:
    return _div(2 * p * r, p + r)


def _tok(text: str) -> list[str]:
    return split_syllables(normalize(text))


def evaluate(srcs: list[str], preds: list[str], tgts: list[str]) -> dict[str, float]:
    """-> {det_p, det_r, det_f1, cor_p, cor_r, cor_f1, sent_acc, over_correction, n_sent, n_errors, n_edits}."""
    if not (len(srcs) == len(preds) == len(tgts)):
        raise ValueError(f"Số câu không khớp: src={len(srcs)} pred={len(preds)} tgt={len(tgts)}")
    n_gold = n_pred = det_tp = cor_tp = 0
    sent_ok = clean = over = 0
    for src, pred, tgt in zip(srcs, preds, tgts):
        s, p, t = _tok(src), _tok(pred), _tok(tgt)
        gold = error_positions(s, t)
        edits = edit_positions(s, p)
        n_gold += len(gold)
        n_pred += len(edits)
        hit = gold.keys() & edits.keys()
        det_tp += len(hit)
        cor_tp += sum(1 for i in hit if edits[i] == gold[i])
        sent_ok += p == t
        if s == t:
            clean += 1
            over += p != s
    det_p, det_r = _div(det_tp, n_pred), _div(det_tp, n_gold)
    cor_p, cor_r = _div(cor_tp, n_pred), _div(cor_tp, n_gold)
    return {
        "det_p": det_p, "det_r": det_r, "det_f1": _f1(det_p, det_r),
        "cor_p": cor_p, "cor_r": cor_r, "cor_f1": _f1(cor_p, cor_r),
        "sent_acc": _div(sent_ok, len(srcs)),
        "over_correction": _div(over, clean),
        "n_sent": len(srcs), "n_errors": n_gold, "n_edits": n_pred,
    }


def per_error_type(records: list[dict], preds: list[str]) -> dict[str, dict[str, float]]:
    """Recall phát hiện/sửa theo từng loại lỗi (chỉ cho tập tổng hợp có trường errors[] với "pos" và "type").

    "pos" là chỉ số âm tiết trong split_syllables(normalize(src)) — generator phải dùng cùng cách tách.
    """
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "detected": 0, "corrected": 0})
    for rec, pred in zip(records, preds):
        s, p, t = _tok(rec["src"]), _tok(pred), _tok(rec["tgt"])
        gold = error_positions(s, t)
        edits = edit_positions(s, p)
        for err in rec.get("errors", []):
            pos, typ = err["pos"], err["type"]
            st = stats[typ]
            st["n"] += 1
            if pos in edits:
                st["detected"] += 1
                if pos in gold and edits[pos] == gold[pos]:
                    st["corrected"] += 1
    return {
        typ: {"n": st["n"], "det_r": _div(st["detected"], st["n"]), "cor_r": _div(st["corrected"], st["n"])}
        for typ, st in sorted(stats.items())
    }
