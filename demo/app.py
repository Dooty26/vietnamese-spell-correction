"""Demo Gradio: nhập câu/đoạn văn -> chọn mô hình (n-gram / BARTpho / cả hai) -> tô âm tiết bị sửa + bảng ứng viên.

Chạy (từ thư mục gốc dự án):
    python -X utf8 demo/app.py                                   # A + C (tự nạp BARTpho v4, không có thì v3)
    python -X utf8 demo/app.py --no-bartpho                      # chỉ mô hình A (n-gram), khởi động nhanh
    python -X utf8 demo/app.py --bartpho-checkpoint path/to/best # checkpoint BARTpho khác
    python -X utf8 demo/app.py --port 7861 --share               # đổi cổng / tạo link công khai
Phụ trách: TV3.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # chạy được mà không cần pip install -e / PYTHONPATH

from vspell.data.viwiki import sentence_spans  # noqa: E402
from vspell.eval.align import edit_positions  # noqa: E402
from vspell.models.ngram_lm import BOS, EOS, lm_token  # noqa: E402
from vspell.normalize import _TOKEN_RE, normalize  # noqa: E402

DEFAULT_NGRAM_DIR = str(ROOT / "outputs/checkpoints/ngram/o3_kneser_ney")
# Ưu tiên v4 (nhiễu giàu lỗi dấu: tốt nhất trên lỗi thật), không có thì v3 (150k cặp × 3 epoch).
BARTPHO_DIRS = [ROOT / "outputs/checkpoints/bartpho_v4/best", ROOT / "outputs/checkpoints/bartpho/best"]
DEFAULT_BARTPHO_DIR = next((d for d in BARTPHO_DIRS if (d / "model.safetensors").exists()), BARTPHO_DIRS[0])

# Mỗi câu một loại lỗi chính; câu cuối kiểm tra tên riêng không bị "sửa" nhầm.
EXAMPLES = [
    ["Hôm nay tooi ddi hojc ở trường đại học."],                                   # Telex chưa chuyển
    ["Chúng tôi sẽ cố gắng giải quyết vấn đề này một cách nhanh chống."],           # vùng miền (hỏi/ngã, sắc/ngã)
    ["Học sinh cần chuẩn bị bài trước khi đến nớp học."],                           # vùng miền l/n
    ["Chính phủ đã ban hành nghị địng mới về thuế thu nhập cá nhâb."],              # trượt phím
    ["Mình ko bít hôm nay có đc đi học hay ko."],                                   # teencode
    ["Sinh vien can nop bai tap truoc ngay mai."],                                  # mất dấu
    ["Ông Nguyễn Văn Thắng đã đến thăm Đà Nẵng và Hội An vào thág trước."],         # tên riêng giữ nguyên, chỉ sửa "thág"
    ["Bài báo nói về kinh tees Việt Nam. Các chuyen gia dự báo tăng trưởng sẽ chậm lại trong năm nay."],  # đoạn văn
    # So sánh A và C (xem report/phan_tich_A_vs_C.md):
    ["Pháp luật cần ghi nhận công sức đòng góp của vợ, chồng trong việc tạo lập tài sản chung."],  # sai dấu thanh: A đúng, C bỏ qua
    ["Học sinh có nhiều cách khác nhau để thực hiện biện phpasp này."],             # sai nhiều ký tự: C đúng, A không có ứng viên
]

COLOR_MAP = {"sửa": "#f59e0b", "n-gram": "#f59e0b", "BARTpho": "#8b5cf6"}
CAND_HEADERS = ["Câu", "Vị trí", "Âm tiết gốc", "Ứng viên", "Nguồn", "Điểm kênh", "Điểm LM", "Tổng", "Chọn"]
MAX_CANDS_SHOWN = 8


# ---------------------------------------------------------------------- nạp mô hình
def load_ngram(d: str | Path):
    from vspell.models.ngram_corrector import NgramCorrector
    from vspell.models.ngram_lm import NgramLM

    d = Path(d)
    lm = NgramLM.load(d / "lm.pkl")
    lexicon = json.loads((d / "lexicon.json").read_text(encoding="utf-8"))
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    return NgramCorrector(lm, lexicon, cfg)


def load_bartpho(checkpoint: str):
    from vspell.models.bartpho import BartphoCorrector  # import lười: torch/transformers nặng
    return BartphoCorrector(checkpoint)


# ---------------------------------------------------------------------- xử lý văn bản
def split_sentences(text: str) -> tuple[list[str], list[str]]:
    """-> (các câu, các khoảng nối) với len(gaps) == len(sents) + 1, để ghép lại đúng văn bản gốc."""
    spans = sentence_spans(text)
    sents, gaps, last = [], [], 0
    for s, e in spans:
        # khoảng trắng đầu/cuối câu thuộc về phần nối (normalize() sẽ cắt mất nó)
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        gaps.append(text[last:s])
        sents.append(text[s:e])
        last = e
    gaps.append(text[last:])
    return sents, gaps


def highlight(src: str, pred: str, label: str) -> tuple[list[tuple[str, str | None]], dict[int, str]]:
    """Câu src + câu sửa -> các đoạn cho gr.HighlightedText; âm tiết bị sửa hiện "sai → đúng"."""
    text = normalize(src)
    matches = list(_TOKEN_RE.finditer(text))
    src_toks = [m.group() for m in matches]
    pred_toks = [m.group() for m in _TOKEN_RE.finditer(normalize(pred))]
    edits = edit_positions(src_toks, pred_toks)
    segs, last = [], 0
    for i, m in enumerate(matches):
        if i in edits:
            segs.append((text[last:m.start()], None))
            new = edits[i] if edits[i] else "∅"
            segs.append((f"{m.group()} → {new}", label))
            last = m.end()
    segs.append((text[last:], None))
    return [s for s in segs if s[0]], edits


def ngram_candidates(corrector, src: str, pred: str, edits: dict[int, str], sent_no: int) -> list[list]:
    """Bảng ứng viên cho từng âm tiết bị sửa: điểm kênh, điểm LM cục bộ (ngữ cảnh lấy từ câu đã sửa), tổng."""
    matches = list(_TOKEN_RE.finditer(normalize(src)))
    pred_toks = [m.group() for m in _TOKEN_RE.finditer(normalize(pred))]
    if len(pred_toks) != len(matches):  # n-gram luôn sửa 1-1; phòng hờ
        return []
    first_word = next((i for i, m in enumerate(matches) if m.group().isalpha()), -1)
    ctx_tokens = [BOS] * (corrector.lm.order - 1) + [lm_token(t) for t in pred_toks] + [EOS]
    off = corrector.lm.order - 1
    n = corrector.lm.order
    w = corrector.lm_weight
    rows = []
    for i in sorted(edits):
        tok = matches[i].group()
        sources = {c: s for c, s, _ in corrector.gen.candidates(tok.lower())}
        opts = corrector._options(tok, i == first_word)
        scored = []
        for k, (cand, channel) in enumerate(opts):
            seq = list(ctx_tokens)
            seq[off + i] = cand
            # log P của cand và (n-1) token sau nó: phần điểm LM thay đổi khi đổi âm tiết i
            lm_score = sum(corrector.lm.logprob(seq[j], tuple(seq[j - n + 1:j]))
                           for j in range(off + i, min(off + i + n, len(seq))))
            src_name = "giữ nguyên" if k == 0 else sources.get(cand, "?")
            scored.append((cand, src_name, channel, lm_score, channel + w * lm_score))
        scored.sort(key=lambda r: -r[4])
        chosen = pred_toks[i].lower()
        for cand, src_name, channel, lm_score, total in scored[:MAX_CANDS_SHOWN]:
            rows.append([sent_no, i + 1, tok, cand, src_name, round(channel, 2), round(lm_score, 2),
                         round(total, 2), "✔" if cand == chosen else ""])
    return rows


# ---------------------------------------------------------------------- giao diện
def build_app(ngram, bartpho):
    import gradio as gr

    choices = ["n-gram"] + (["BARTpho", "Cả hai"] if bartpho is not None else [])

    def run(text: str, model: str):
        text = (text or "").strip("\n")
        use_ng = model in ("n-gram", "Cả hai")
        use_bp = model in ("BARTpho", "Cả hai") and bartpho is not None
        sents, gaps = split_sentences(text)
        outs, times, cand_rows = {}, [], []
        for name, enabled, fn in [("n-gram", use_ng, lambda ss: [ngram.correct_one(s) for s in ss]),
                                  ("BARTpho", use_bp, lambda ss: bartpho.correct(ss))]:
            if not enabled:
                continue
            t0 = time.perf_counter()
            try:
                preds = fn(sents) if sents else []
            except Exception as e:  # noqa: BLE001  (vd BARTpho chưa cài xong)
                outs[name] = ([(f"Lỗi khi chạy {name}: {type(e).__name__}: {e}", None)], "")
                times.append(f"**{name}**: lỗi")
                continue
            dt = time.perf_counter() - t0
            segs, plain, n_edit = [], [], 0
            for k, (s, p) in enumerate(zip(sents, preds)):
                segs.append((gaps[k], None))
                hs, edits = highlight(s, p, name)
                segs += hs
                plain.append(gaps[k] + p)
                n_edit += len(edits)
                if name == "n-gram":
                    cand_rows += ngram_candidates(ngram, s, p, edits, k + 1)
            segs.append((gaps[len(sents)], None))
            plain.append(gaps[len(sents)])
            outs[name] = ([x for x in segs if x[0]], "".join(plain))
            times.append(f"**{name}**: {dt * 1000:.0f} ms · {len(sents)} câu · {n_edit} âm tiết bị sửa")
        ng_hl, ng_txt = outs.get("n-gram", (None, ""))
        bp_hl, bp_txt = outs.get("BARTpho", (None, ""))
        return (gr.update(value=ng_hl, visible=use_ng), gr.update(value=ng_txt, visible=use_ng),
                gr.update(value=bp_hl, visible=use_bp), gr.update(value=bp_txt, visible=use_bp),
                gr.update(value=cand_rows, visible=use_ng),
                "  \n".join(times) or "_Chưa chạy_")

    with gr.Blocks(title="Sửa lỗi chính tả tiếng Việt") as app:
        gr.Markdown("# Phát hiện và sửa lỗi chính tả tiếng Việt\n"
                    "Mô hình A: noisy channel + LM 3-gram Kneser-Ney."
                    + (f" Mô hình C: BARTpho fine-tune (`{Path(bartpho.checkpoint).parent.name}`)." if bartpho is not None else
                       " _(BARTpho chưa nạp: chạy với `--bartpho-checkpoint` để bật.)_"))
        with gr.Row():
            with gr.Column(scale=3):
                inp = gr.Textbox(label="Câu hoặc đoạn văn", lines=4,
                                 placeholder="Nhập câu có lỗi chính tả, vd: Hôm nay tooi ddi hojc...")
            with gr.Column(scale=1):
                model = gr.Radio(choices, value="Cả hai" if bartpho is not None else "n-gram", label="Mô hình")
                btn = gr.Button("Sửa lỗi", variant="primary")
                timing = gr.Markdown("_Chưa chạy_")
        with gr.Row():
            with gr.Column():
                ng_hl = gr.HighlightedText(label="n-gram: âm tiết bị sửa (sai → đúng)", color_map=COLOR_MAP,
                                           combine_adjacent=False, show_legend=False)
                ng_txt = gr.Textbox(label="n-gram: câu đã sửa", interactive=False)
            with gr.Column(visible=bartpho is not None):
                bp_hl = gr.HighlightedText(label="BARTpho: âm tiết bị sửa (sai → đúng)", color_map=COLOR_MAP,
                                           combine_adjacent=False, show_legend=False, visible=False)
                bp_txt = gr.Textbox(label="BARTpho: câu đã sửa", interactive=False, visible=False)
        cands = gr.Dataframe(headers=CAND_HEADERS, label="Ứng viên của mô hình A cho từng âm tiết bị sửa "
                             "(Tổng = điểm kênh + λ·điểm LM, log10; ✔ = ứng viên được chọn)",
                             interactive=False, wrap=True)
        gr.Examples(EXAMPLES, inputs=[inp], label="Câu ví dụ (bấm để nạp, rồi bấm Sửa lỗi)")
        outputs = [ng_hl, ng_txt, bp_hl, bp_txt, cands, timing]
        btn.click(run, [inp, model], outputs)
        inp.submit(run, [inp, model], outputs)
    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bartpho-checkpoint", default=None,
                    help="mặc định: thư mục đầu tiên có model.safetensors trong "
                         + ", ".join(str(d.relative_to(ROOT)) for d in BARTPHO_DIRS))
    ap.add_argument("--no-bartpho", action="store_true", help="không nạp BARTpho (chỉ n-gram)")
    ap.add_argument("--ngram-dir", default=DEFAULT_NGRAM_DIR)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    ngram = load_ngram(args.ngram_dir)
    print(f"Đã nạp n-gram từ {args.ngram_dir} ({time.perf_counter() - t0:.1f}s)")
    bartpho = None
    ckpt = args.bartpho_checkpoint
    if ckpt is None and (DEFAULT_BARTPHO_DIR / "model.safetensors").exists():
        ckpt = str(DEFAULT_BARTPHO_DIR)
    if ckpt and not args.no_bartpho:
        t0 = time.perf_counter()
        bartpho = load_bartpho(ckpt)
        print(f"Đã nạp BARTpho từ {ckpt} trên {bartpho.device} ({time.perf_counter() - t0:.1f}s)")
    elif not args.no_bartpho:
        print(f"Không thấy checkpoint BARTpho ({DEFAULT_BARTPHO_DIR}): chỉ chạy n-gram")
    build_app(ngram, bartpho).launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
