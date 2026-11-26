"""S5a: hình EDA, hình kết quả và phân tích lỗi tự động cho các mô hình.

Một lệnh chạy lại toàn bộ (từ thư mục gốc repo):
    PYTHONPATH=src python -X utf8 scripts/analyze_errors.py
Khi đã có outputs/predictions/bartpho__*.jsonl:
    PYTHONPATH=src python -X utf8 scripts/analyze_errors.py --models ngram bartpho
-> thêm bảng A vs C (model_comparison.csv) và các nhóm ví dụ "A đúng C sai" / "C đúng A sai".

Phân loại mỗi vị trí âm tiết (theo chỉ số src, cùng cách căn chỉnh với vspell.eval.metrics):
    sua_dung     : vị trí lỗi thật, mô hình sửa đúng chuỗi tgt
    sua_sai      : vị trí lỗi thật, mô hình có sửa nhưng chọn sai ứng viên
    bo_sot       : vị trí lỗi thật, mô hình giữ nguyên
    sua_qua_tay  : vị trí không có lỗi, mô hình lại thay đổi
Sửa quá tay được tách theo token src (ưu tiên từ trên xuống):
    so_ky_hieu   : token có chữ số hoặc không có chữ cái
    ten_rieng    : viết hoa (không phải token đầu câu) -> nghi tên riêng
    ngoai_tu_dien: chữ thường (hoặc đầu câu) nhưng không có trong từ điển âm tiết của A -> từ nước ngoài/hiếm
    tu_thuong    : âm tiết có trong từ điển (lỗi "từ thật" do LM)

Đầu ra:
    outputs/figures/*.png                 (150 dpi)
    outputs/tables/error_breakdown.csv    model, dataset, category, subcategory, n, ratio
    outputs/tables/overcorrection_viwiki.csv  kiểm chứng giả thuyết sửa quá tay trên Viwiki
    outputs/tables/eda_summary.csv, real_error_kinds.csv
    outputs/tables/model_comparison.csv   (khi có ≥2 mô hình)
    outputs/tables/error_examples.md      10–15 ví dụ ứng viên (+ ví dụ A vs C)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from vspell.eval.align import edit_positions, error_positions
from vspell.models.ngram_corrector import osa_distance
from vspell.normalize import normalize, remove_diacritics, split_syllables, split_tone
from vspell.utils import get_logger, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["syn_test", "vsec_test", "viwiki_test"]
DS_LABEL = {"syn_test": "Syn-test", "vsec_test": "VSEC-test", "viwiki_test": "Viwiki-test",
            "syn_val": "Syn-val", "syn_train": "Syn-train"}
MODEL_LABEL = {"ngram": "A: n-gram (3-gram KN)", "bartpho": "C: BARTpho v3", "identity": "Đồng nhất",
               "bilstm": "B: BiLSTM", "bartpho_v1unk": "C: BARTpho v2 (lỗi <unk>)",
               "bartpho_v4": "C: BARTpho v4 (nhiễu dấu)", "bartpho_d12": "C: BARTpho d12 (50k, 1 ep)"}
# Ablation C (mật độ nhiễu khi sinh dữ liệu train): tag -> (mật độ, nhãn)
ABLATION_C = {"bartpho_d05": (0.05, "5% · 50k · 1 ep"), "bartpho_d12": (0.12, "12% · 50k · 1 ep"),
              "bartpho_d25": (0.25, "25% · 50k · 1 ep"), "bartpho": (0.05, "5% · 150k · 3 ep\n(mô hình chính)")}
CAT_LABEL = {"sua_dung": "Sửa đúng", "sua_sai": "Sửa sai ứng viên", "bo_sot": "Bỏ sót",
             "sua_qua_tay": "Sửa quá tay"}
SUB_LABEL = {"so_ky_hieu": "Số/ký hiệu", "ten_rieng": "Tên riêng/viết hoa",
             "ngoai_tu_dien": "Ngoài từ điển (nước ngoài)", "tu_thuong": "Từ thường trong từ điển"}
TYPE_LABEL = {"fat_finger": "Gõ nhầm phím", "telex": "Telex", "region": "Vùng miền",
              "edit_distance": "Sai ký tự", "teencode": "Teencode", "acronym": "Viết tắt"}
# Ablation bậc × smoothing: tên file dự đoán -> (bậc, smoothing). "ngram" là cấu hình chính o3 KN.
ABLATION = {"ngram_o2_laplace": (2, "laplace"), "ngram_o3_laplace": (3, "laplace"),
            "ngram_o2_kneser_ney": (2, "kneser_ney"), "ngram": (3, "kneser_ney"),
            "ngram_o4_kneser_ney": (4, "kneser_ney")}
SMOOTH_LABEL = {"laplace": "Laplace", "kneser_ney": "Kneser-Ney"}
DPI = 150
_HAS_LETTER = re.compile(r"[^\W\d_]")
log = get_logger()


def _tok(text: str) -> list[str]:
    return split_syllables(normalize(text))


# ---------------------------------------------------------------------------- phân loại
def overcorrection_kind(tokens: list[str], i: int, lexicon: set[str]) -> str:
    tok = tokens[i]
    if any(c.isdigit() for c in tok) or not _HAS_LETTER.search(tok):
        return "so_ky_hieu"
    sentence_start = i == 0 or tokens[i - 1] in {".", "!", "?", ":", "\"", "“", "(", "-", "–", "…"}
    if tok[0].isupper() and (not sentence_start or (len(tok) > 1 and tok.isupper())):
        return "ten_rieng"
    if tok.lower() not in lexicon:
        return "ngoai_tu_dien"
    return "tu_thuong"


def classify_sentence(src: str, pred: str, tgt: str, lexicon: set[str]) -> list[dict]:
    """-> danh sách vị trí cần quan tâm: {pos, cat, sub, src_tok, pred_str, gold_str}."""
    s, p, t = _tok(src), _tok(pred), _tok(tgt)
    gold, edits = error_positions(s, t), edit_positions(s, p)
    out = []
    for i in sorted(gold.keys() | edits.keys()):
        if i in gold:
            cat = "bo_sot" if i not in edits else ("sua_dung" if edits[i] == gold[i] else "sua_sai")
            sub = ""
        else:
            cat, sub = "sua_qua_tay", overcorrection_kind(s, i, lexicon)
        out.append({"pos": i, "cat": cat, "sub": sub, "src_tok": s[i],
                    "pred_str": edits.get(i, s[i]), "gold_str": gold.get(i, s[i])})
    return out


# ---------------------------------------------------------------------------- tiện ích I/O
def load_preds(pred_dir: Path, model: str, dataset: str) -> list[dict] | None:
    path = pred_dir / f"{model}__{dataset}.jsonl"
    return list(read_jsonl(path)) if path.exists() else None


def load_lexicon(path: Path) -> set[str]:
    if not path.exists():
        log.warning("Không thấy từ điển %s: nhóm 'ngoài từ điển' sẽ rỗng", path)
        return set()
    with open(path, encoding="utf-8") as f:
        return set(json.load(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log.info("Đã ghi %s", path)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.axisbelow": True})
    return plt


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    fig.clf()
    log.info("Đã vẽ %s", path)


COLORS = ["#2a6fbb", "#e07b39", "#3a9d5d", "#c0392b", "#8e6bbf", "#7f7f7f"]


def _bar_labels(ax, bars, fmt="{:.2f}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), ha="center", va="bottom",
                    fontsize=7, xytext=(0, 1), textcoords="offset points")


# ---------------------------------------------------------------------------- 1. EDA
def real_error_kind(wrong: str, right: str) -> str:
    """Kiểu lỗi thật (VSEC/Viwiki không có nhãn loại lỗi) suy từ cặp (sai, đúng)."""
    if " " in right or right == "":
        return "Tách/ghép âm tiết"
    if wrong.lower() == right.lower():
        return "Chỉ sai hoa/thường"
    if remove_diacritics(wrong).lower() == remove_diacritics(right).lower():
        bw, tw = split_tone(wrong.lower())
        br, tr = split_tone(right.lower())
        return "Chỉ sai dấu thanh" if bw == br else "Sai dấu phụ (ă â ê ô ơ ư đ)"
    if osa_distance(remove_diacritics(wrong).lower(), remove_diacritics(right).lower()) == 1:
        return "Sai 1 chữ cái (thêm/bớt/thay/đảo)"
    return "Sai ≥2 ký tự"


def run_eda(data_dir: Path, fig_dir: Path, tab_dir: Path, n_train: int) -> None:
    plt = _plt()
    sets: dict[str, list[dict]] = {}
    for ds in DATASETS:
        p = data_dir / f"{ds}.jsonl"
        if p.exists():
            sets[ds] = list(read_jsonl(p))
    train_p = data_dir / "syn_train.jsonl"
    if train_p.exists() and n_train > 0:
        recs = []
        for r in read_jsonl(train_p):
            recs.append(r)
            if len(recs) >= n_train:
                break
        sets["syn_train"] = recs
    lengths, n_err, kinds, summary = {}, {}, {}, []
    for ds, recs in sets.items():
        L, E, K = [], [], Counter()
        n_tok = 0
        for r in recs:
            s, t = _tok(r["src"]), _tok(r["tgt"])
            gold = error_positions(s, t)
            L.append(len(t))
            E.append(len(gold))
            n_tok += len(s)
            for i, right in gold.items():
                K[real_error_kind(s[i], right)] += 1
        lengths[ds], n_err[ds], kinds[ds] = L, E, K
        L_sorted = sorted(L)
        summary.append({"dataset": ds, "n_sent": len(recs), "len_mean": round(sum(L) / len(L), 2),
                        "len_median": L_sorted[len(L) // 2], "len_max": L_sorted[-1],
                        "n_errors": sum(E), "sent_with_error": round(sum(e > 0 for e in E) / len(E), 4),
                        "errors_per_token": round(sum(E) / n_tok, 4),
                        "errors_per_sent": round(sum(E) / len(E), 3)})
    write_csv(tab_dir / "eda_summary.csv", summary)
    kind_names = sorted({k for K in kinds.values() for k in K})
    write_csv(tab_dir / "real_error_kinds.csv",
              [{"dataset": ds, "kind": k, "n": kinds[ds][k],
                "ratio": round(kinds[ds][k] / max(1, sum(kinds[ds].values())), 4)}
               for ds in kinds for k in kind_names])
    order = [d for d in ["syn_train", *DATASETS] if d in sets]

    # Độ dài câu
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bins = list(range(0, 101, 4))
    for k, ds in enumerate(order):
        ax.hist([min(x, 100) for x in lengths[ds]], bins=bins, density=True, histtype="step", lw=1.8,
                color=COLORS[k], label=f"{DS_LABEL[ds]} (TB {sum(lengths[ds]) / len(lengths[ds]):.1f})")
    ax.set_xlabel("Độ dài câu (số token: âm tiết, số, dấu câu; cắt ở 100)")
    ax.set_ylabel("Tỉ lệ câu (mật độ)")
    ax.set_title("Phân bố độ dài câu của các tập dữ liệu")
    ax.legend()
    _save(fig, fig_dir, "eda_sentence_length.png")

    # Mật độ lỗi: số lỗi mỗi câu + lỗi/token
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [2.2, 1]})
    buckets = ["0", "1", "2", "3", "4", "≥5"]
    w = 0.8 / len(order)
    for k, ds in enumerate(order):
        c = Counter(min(e, 5) for e in n_err[ds])
        vals = [c[i] / len(n_err[ds]) for i in range(6)]
        ax1.bar([i + (k - (len(order) - 1) / 2) * w for i in range(6)], vals, w, color=COLORS[k],
                label=DS_LABEL[ds])
    ax1.set_xticks(range(6), buckets)
    ax1.set_xlabel("Số âm tiết lỗi trong câu")
    ax1.set_ylabel("Tỉ lệ câu")
    ax1.set_title("Phân bố số lỗi mỗi câu")
    ax1.legend()
    rates = [next(s["errors_per_token"] for s in summary if s["dataset"] == ds) * 100 for ds in order]
    bars = ax2.bar([DS_LABEL[d] for d in order], rates, color=COLORS[:len(order)])
    _bar_labels(ax2, bars, "{:.2f}%")
    ax2.set_ylabel("Lỗi / token (%)")
    ax2.set_title("Mật độ lỗi")
    ax2.tick_params(axis="x", rotation=20)
    _save(fig, fig_dir, "eda_error_density.png")

    # Loại lỗi: (a) nhãn sinh nhiễu của tập tổng hợp, (b) kiểu lỗi suy ra cho cả 3 tập test
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    syn_stats = {}
    for split in ["syn_train", "syn_val", "syn_test"]:
        p = data_dir / f"{split}.stats.json"
        if p.exists():
            syn_stats[split] = json.load(open(p, encoding="utf-8"))["error_types"]
    types = list(TYPE_LABEL)
    w = 0.8 / max(1, len(syn_stats))
    for k, (split, st) in enumerate(syn_stats.items()):
        ax1.bar([i + (k - (len(syn_stats) - 1) / 2) * w for i in range(len(types))],
                [st.get(t, 0) * 100 for t in types], w, color=COLORS[k], label=DS_LABEL[split])
    ax1.set_xticks(range(len(types)), [TYPE_LABEL[t] for t in types], rotation=25)
    ax1.set_ylabel("Tỉ lệ lỗi (%)")
    ax1.set_title("(a) Loại lỗi sinh nhiễu – dữ liệu tổng hợp")
    ax1.legend()
    test_sets = [d for d in DATASETS if d in kinds]
    w = 0.8 / max(1, len(test_sets))
    for k, ds in enumerate(test_sets):
        tot = max(1, sum(kinds[ds].values()))
        ax2.bar([i + (k - (len(test_sets) - 1) / 2) * w for i in range(len(kind_names))],
                [kinds[ds][n] / tot * 100 for n in kind_names], w, color=COLORS[k], label=DS_LABEL[ds])
    ax2.set_xticks(range(len(kind_names)), kind_names, rotation=30, ha="right")
    ax2.set_ylabel("Tỉ lệ lỗi (%)")
    ax2.set_title("(b) Kiểu khác biệt sai→đúng – 3 tập test")
    ax2.legend()
    _save(fig, fig_dir, "eda_error_types.png")
    plt.close("all")


# ---------------------------------------------------------------------------- 2. hình kết quả
def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        log.warning("Thiếu %s", path)
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_results(tab_dir: Path, fig_dir: Path, breakdown: list[dict], models: list[str]) -> None:
    plt = _plt()
    main = _read_csv(tab_dir / "main_results.csv")
    ppl = {r["name"]: float(r["perplexity"]) for r in _read_csv(tab_dir / "ngram_perplexity.csv")}
    res = {(r["model"], r["dataset"]): r for r in main}
    abl = [m for m in ABLATION if (m, "syn_test") in res]

    # Ablation F1
    if abl:
        fig, ax = plt.subplots(figsize=(8, 4))
        names = [f"{ABLATION[m][0]}-gram\n{SMOOTH_LABEL[ABLATION[m][1]]}" for m in abl]
        for k, ds in enumerate(["syn_test", "vsec_test"]):
            vals = [float(res[(m, ds)]["cor_f1"]) if (m, ds) in res else 0 for m in abl]
            bars = ax.bar([i + (k - 0.5) * 0.38 for i in range(len(abl))], vals, 0.38, color=COLORS[k],
                          label=DS_LABEL[ds])
            _bar_labels(ax, bars)
        ax.set_xticks(range(len(abl)), names)
        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel("Correction F1")
        ax.set_title("Ablation mô hình A: bậc n-gram × phương pháp làm trơn")
        ax.legend(loc="upper left", ncol=2)
        _save(fig, fig_dir, "ablation_order_smoothing_f1.png")

        # Perplexity vs F1
        fig, ax = plt.subplots(figsize=(7, 4.2))
        markers = {"laplace": "s", "kneser_ney": "o"}
        for k, ds in enumerate(["syn_test", "vsec_test"]):
            xs, ys = [], []
            for m in abl:
                o, sm = ABLATION[m]
                key = f"o{o}_{sm}"
                if key not in ppl or (m, ds) not in res:
                    continue
                x, y = ppl[key], float(res[(m, ds)]["cor_f1"])
                xs.append(x), ys.append(y)
                ax.scatter(x, y, color=COLORS[k], marker=markers[sm], s=55, zorder=3)
                ax.annotate(f"{o}-gram {SMOOTH_LABEL[sm]}", (x, y), fontsize=7, xytext=(6, -11 if o == 3 else 3),
                            textcoords="offset points")
            pts = sorted(zip(xs, ys))
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLORS[k], alpha=0.4, label=DS_LABEL[ds])
        ax.set_xscale("log")
        ax.set_xlabel("Perplexity trên tập giữ lại (thang log, thấp hơn là tốt hơn)")
        ax.set_ylabel("Correction F1")
        ax.set_title("Perplexity của LM so với F1 sửa lỗi")
        ax.scatter([], [], marker="o", color="gray", label="Kneser-Ney")
        ax.scatter([], [], marker="s", color="gray", label="Laplace")
        ax.legend(fontsize=8)
        _save(fig, fig_dir, "ablation_perplexity_vs_f1.png")

    # Recall theo loại lỗi
    per = _read_csv(tab_dir / "per_error_type.csv")
    rec_models = [m for m in [*models, *[a for a in abl if a not in models]]
                  if any(r["model"] == m for r in per)]
    if per and rec_models:
        types = list(TYPE_LABEL)
        fig, ax = plt.subplots(figsize=(9, 5))
        w = 0.8 / len(rec_models)
        for k, m in enumerate(rec_models):
            d = {r["error_type"]: float(r["cor_r"]) for r in per if r["model"] == m and r["dataset"] == "syn_test"}
            lab = MODEL_LABEL.get(m) if m not in ABLATION or m in models else None
            lab = lab or f"{ABLATION[m][0]}-gram {SMOOTH_LABEL[ABLATION[m][1]]}"
            bars = ax.bar([i + (k - (len(rec_models) - 1) / 2) * w for i in range(len(types))],
                          [d.get(t, 0) for t in types], w, color=COLORS[k % len(COLORS)], label=lab)
            if k == 0:
                _bar_labels(ax, bars)
        n_by_type = {r["error_type"]: r["n"] for r in per if r["dataset"] == "syn_test"}
        ax.set_xticks(range(len(types)), [f"{TYPE_LABEL[t]}\n(n={n_by_type.get(t, '?')})" for t in types])
        ax.set_ylabel("Correction recall")
        ax.set_ylim(0, 1.08)
        ax.set_title("Recall sửa lỗi theo loại lỗi (Syn-test)")
        ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
        _save(fig, fig_dir, "recall_by_error_type.png")

    # So sánh mô hình trên 3 tập test (tính từ breakdown để không phụ thuộc main_results.csv)
    cnt = defaultdict(Counter)
    for r in breakdown:
        if r["subcategory"] == "":
            cnt[(r["model"], r["dataset"])][r["category"]] += r["n"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, metric, title in zip(axes, ["P", "R", "F1"], ["Precision", "Recall", "F1"]):
        w = 0.8 / max(1, len(models))
        for k, m in enumerate(models):
            vals = []
            for ds in DATASETS:
                c = cnt.get((m, ds))
                if not c:
                    vals.append(0)
                    continue
                n_edit = c["sua_dung"] + c["sua_sai"] + c["sua_qua_tay"]
                n_gold = c["sua_dung"] + c["sua_sai"] + c["bo_sot"]
                p = c["sua_dung"] / n_edit if n_edit else 0
                rr = c["sua_dung"] / n_gold if n_gold else 0
                vals.append({"P": p, "R": rr, "F1": 2 * p * rr / (p + rr) if p + rr else 0}[metric])
            bars = ax.bar([i + (k - (len(models) - 1) / 2) * w for i in range(len(DATASETS))], vals, w,
                          color=COLORS[k], label=MODEL_LABEL.get(m, m))
            _bar_labels(ax, bars)
        ax.set_xticks(range(len(DATASETS)), [DS_LABEL[d] for d in DATASETS])
        ax.set_title(f"Correction {title}")
        ax.set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    fig.suptitle("So sánh mô hình trên 3 tập test")
    _save(fig, fig_dir, "model_comparison_test_sets.png")

    # Cơ cấu vị trí sai + nhóm sửa quá tay
    keys = [(m, ds) for m in models for ds in DATASETS if (m, ds) in cnt]
    if keys:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))
        labels = [f"{DS_LABEL[ds]}\n{m}" if len(models) > 1 else DS_LABEL[ds] for m, ds in keys]
        bottom = [0.0] * len(keys)
        for k, cat in enumerate(CAT_LABEL):
            vals = [cnt[key][cat] / max(1, sum(cnt[key].values())) for key in keys]
            ax1.bar(labels, vals, bottom=bottom, color=COLORS[k], label=CAT_LABEL[cat])
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax1.set_ylabel("Tỉ lệ trong các vị trí (lỗi thật ∪ vị trí bị sửa)")
        ax1.set_title("Cơ cấu kết quả theo vị trí")
        ax1.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.1))
        sub = defaultdict(Counter)
        for r in breakdown:
            if r["category"] == "sua_qua_tay" and r["subcategory"]:
                sub[(r["model"], r["dataset"])][r["subcategory"]] += r["n"]
        bottom = [0.0] * len(keys)
        for k, sc in enumerate(SUB_LABEL):
            vals = [sub[key][sc] / max(1, sum(sub[key].values())) for key in keys]
            bars = ax2.bar(labels, vals, bottom=bottom, color=COLORS[k + 2 if k < 4 else k], label=SUB_LABEL[sc])
            for b, v, bt in zip(bars, vals, bottom):
                if v > 0.06:
                    ax2.text(b.get_x() + b.get_width() / 2, bt + v / 2, f"{v:.0%}", ha="center", va="center",
                             fontsize=7, color="white")
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax2.set_ylabel("Tỉ lệ trong các ca sửa quá tay")
        ax2.set_title("Sửa quá tay tách theo loại token gốc")
        ax2.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.1))
        _save(fig, fig_dir, "error_breakdown.png")
    plt.close("all")


# ---------------------------------------------------------------------------- 3. phân tích lỗi
def run_breakdown(pred_dir: Path, models: list[str], lexicon: set[str]):
    rows, per_sent = [], {}
    for m in models:
        for ds in DATASETS:
            recs = load_preds(pred_dir, m, ds)
            if recs is None:
                log.warning("Bỏ qua %s__%s (không có file dự đoán)", m, ds)
                continue
            items = [classify_sentence(r["src"], r["pred"], r["tgt"], lexicon) for r in recs]
            per_sent[(m, ds)] = (recs, items)
            cat = Counter(x["cat"] for it in items for x in it)
            sub = Counter(x["sub"] for it in items for x in it if x["cat"] == "sua_qua_tay")
            total = sum(cat.values())
            for c in CAT_LABEL:
                rows.append({"model": m, "dataset": ds, "category": c, "subcategory": "", "n": cat[c],
                             "ratio": round(cat[c] / total, 4) if total else 0.0})
            n_over = cat["sua_qua_tay"]
            for sc in SUB_LABEL:
                rows.append({"model": m, "dataset": ds, "category": "sua_qua_tay", "subcategory": sc,
                             "n": sub[sc], "ratio": round(sub[sc] / n_over, 4) if n_over else 0.0})
            log.info("%s %s: %s | quá tay: %s", m, ds, dict(cat), dict(sub))
    return rows, per_sent


def viwiki_hypothesis(per_sent, model: str, lexicon: set[str], tab_dir: Path) -> list[dict]:
    """Kiểm chứng: precision thấp trên Viwiki có phải do sửa quá tay tên riêng/từ nước ngoài?"""
    if (model, "viwiki_test") not in per_sent:
        return []
    recs, items = per_sent[(model, "viwiki_test")]
    flat = [x for it in items for x in it]
    c = Counter(x["cat"] for x in flat)
    over = [x for x in flat if x["cat"] == "sua_qua_tay"]
    sub = Counter(x["sub"] for x in over)
    # Tỉ lệ nền: các token trong src Viwiki thuộc từng nhóm
    base = Counter()
    gold_sub = Counter()
    for r, it in zip(recs, items):
        s = _tok(r["src"])
        for i in range(len(s)):
            base[overcorrection_kind(s, i, lexicon)] += 1
        for x in it:
            if x["cat"] != "sua_qua_tay":
                gold_sub[overcorrection_kind(s, x["pos"], lexicon)] += 1
    n_tok = sum(base.values())
    n_edit, n_gold = c["sua_dung"] + c["sua_sai"] + c["sua_qua_tay"], c["sua_dung"] + c["sua_sai"] + c["bo_sot"]
    rows = []

    def add(name, value, note=""):
        rows.append({"metric": name, "value": value, "note": note})

    p = c["sua_dung"] / n_edit
    r_ = c["sua_dung"] / n_gold
    add("n_edits", n_edit, "số vị trí A thay đổi")
    add("n_gold_errors", n_gold, "số lỗi thật")
    add("sua_dung", c["sua_dung"]), add("sua_sai", c["sua_sai"]), add("bo_sot", c["bo_sot"])
    add("sua_qua_tay", c["sua_qua_tay"], f"{c['sua_qua_tay'] / n_edit:.1%} số vị trí bị sửa")
    add("cor_p", round(p, 4)), add("cor_r", round(r_, 4)), add("cor_f1", round(2 * p * r_ / (p + r_), 4))
    for sc in SUB_LABEL:
        add(f"over_{sc}", sub[sc], f"{sub[sc] / max(1, len(over)):.1%} ca sửa quá tay; "
            f"tỉ lệ nền trong token src: {base[sc] / n_tok:.1%}; "
            f"trong lỗi thật: {gold_sub[sc] / max(1, n_gold):.1%}")
        add(f"over_rate_{sc}", round(sub[sc] / max(1, base[sc]), 5),
            "xác suất một token thuộc nhóm bị sửa quá tay")
    # So với các tập khác: tỉ lệ sửa quá tay trên mỗi token sạch và mật độ lỗi thật
    for ds in DATASETS:
        if (model, ds) not in per_sent:
            continue
        rr_, it_ = per_sent[(model, ds)]
        cc = Counter(x["cat"] for it in it_ for x in it)
        ntok = sum(len(_tok(r["src"])) for r in rr_)
        ng = cc["sua_dung"] + cc["sua_sai"] + cc["bo_sot"]
        add(f"{ds}_over_per_clean_token", round(cc["sua_qua_tay"] / (ntok - ng), 5),
            f"lỗi thật/token = {ng / ntok:.4f}; sửa quá tay/lỗi thật = {cc['sua_qua_tay'] / ng:.2f}")
    # Phản thực tế: nếu A không động vào token của nhóm -> precision/recall mới
    for excl in [("ten_rieng",), ("ten_rieng", "ngoai_tu_dien"), ("ten_rieng", "ngoai_tu_dien", "so_ky_hieu")]:
        lost_tp = sum(1 for r, it in zip(recs, items) for x in it if x["cat"] == "sua_dung"
                      and overcorrection_kind(_tok(r["src"]), x["pos"], lexicon) in excl)
        lost_wrong = sum(1 for r, it in zip(recs, items) for x in it if x["cat"] == "sua_sai"
                         and overcorrection_kind(_tok(r["src"]), x["pos"], lexicon) in excl)
        removed_over = sum(sub[e] for e in excl)
        ne = n_edit - removed_over - lost_tp - lost_wrong
        tp = c["sua_dung"] - lost_tp
        pp, rr = tp / ne, tp / n_gold
        add("neu_bo_qua_" + "+".join(excl), round(2 * pp * rr / (pp + rr), 4),
            f"F1 nếu không sửa nhóm này: P={pp:.3f} R={rr:.3f} (mất {lost_tp} sửa đúng)")
    # Các token hay bị sửa quá tay nhất
    top = Counter((x["src_tok"], x["pred_str"]) for x in over).most_common(15)
    add("top_overcorrections", "; ".join(f"{a}→{b} ({n})" for (a, b), n in top))
    write_csv(tab_dir / "overcorrection_viwiki.csv", rows)
    return rows


# ---------------------------------------------------------------------------- 4. ví dụ
def _highlight(tokens: list[str], marks: dict[int, str], lo: int, hi: int) -> str:
    out = []
    for i in range(lo, hi):
        if i in marks:
            rep = marks[i]
            out.append(f"**{rep}**" if rep else "**∅**")
        else:
            out.append(tokens[i])
    return ("… " if lo > 0 else "") + " ".join(out) + (" …" if hi < len(tokens) else "")


def render_example(rec: dict, pos: list[int], title: str, preds: dict[str, str] | None = None, window=12) -> str:
    s, t = _tok(rec["src"]), _tok(rec["tgt"])
    gold = error_positions(s, t)
    lo, hi = max(0, min(pos) - window), min(len(s), max(pos) + window + 1)
    preds = preds or {"pred": rec["pred"]}
    edits = {k: edit_positions(s, _tok(v)) for k, v in preds.items()}
    show = set(pos) | {i for e in edits.values() for i in e if lo <= i < hi} | {i for i in gold if lo <= i < hi}
    lines = [f"**{title}**", "",
             f"- src : {_highlight(s, {i: s[i] for i in show}, lo, hi)}"]
    for k, e in edits.items():
        lines.append(f"- {k:<4}: {_highlight(s, {i: e[i] for i in e}, lo, hi)}")
    lines.append(f"- tgt : {_highlight(s, {i: gold[i] for i in gold}, lo, hi)}")
    return "\n".join(lines) + "\n"


def pick(per_sent, model, ds, cat, sub=None, k=1, rng=None, max_len=45, exclude=()):
    if (model, ds) not in per_sent:
        return []
    recs, items = per_sent[(model, ds)]
    cands = [(r, x) for r, it in zip(recs, items) for x in it
             if x["cat"] == cat and (sub is None or x["sub"] == sub) and len(_tok(r["src"])) <= max_len
             and r.get("id") not in exclude and _HAS_LETTER.search(x["src_tok"])
             and _HAS_LETTER.search(x["gold_str"] or "-")]
    rng.shuffle(cands)
    out, seen = [], set()
    for r, x in cands:
        if r.get("id") in seen:
            continue
        seen.add(r.get("id"))
        out.append((r, x))
        if len(out) >= k:
            break
    return out


def run_examples(per_sent, models: list[str], tab_dir: Path, hyp: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    A = models[0]
    plan = [("viwiki_test", "sua_qua_tay", "ten_rieng", 3, "Viwiki – sửa quá tay tên riêng/viết hoa"),
            ("viwiki_test", "sua_qua_tay", "ngoai_tu_dien", 2, "Viwiki – sửa quá tay từ ngoài từ điển"),
            ("viwiki_test", "sua_qua_tay", "tu_thuong", 1, "Viwiki – sửa quá tay từ thường"),
            ("vsec_test", "sua_sai", None, 2, "VSEC – sửa sai ứng viên"),
            ("vsec_test", "bo_sot", None, 2, "VSEC – bỏ sót"),
            ("vsec_test", "sua_qua_tay", "tu_thuong", 1, "VSEC – sửa quá tay từ thật"),
            ("syn_test", "sua_sai", None, 1, "Syn – sửa sai ứng viên"),
            ("syn_test", "bo_sot", None, 1, "Syn – bỏ sót"),
            ("viwiki_test", "sua_dung", None, 1, "Viwiki – sửa đúng (đối chứng)")]
    md = [f"# Ví dụ phân tích lỗi (tự sinh bởi `scripts/analyze_errors.py`, seed={seed})", "",
          "Câu được tách token như khi chấm (dấu câu tách riêng), cắt cửa sổ quanh vị trí lỗi.",
          "In **đậm**: ở `src` là vị trí lỗi thật hoặc bị mô hình sửa; ở `pred`/`tgt` là chuỗi thay thế; "
          "`∅` là âm tiết bị xóa.", ""]
    chosen = []
    n = 0
    md.append(f"## Mô hình {MODEL_LABEL.get(A, A)}\n")
    used = set()
    for ds, cat, sub, k, title in plan:
        for r, x in pick(per_sent, A, ds, cat, sub, k, rng, exclude=used):
            used.add(r.get("id"))
            n += 1
            note = f"{CAT_LABEL[cat]}{' / ' + SUB_LABEL[sub] if sub else ''}: `{x['src_tok']}` → " \
                   f"`{x['pred_str']}` (đúng: `{x['gold_str']}`)"
            md.append(render_example(r, [x["pos"]], f"Ví dụ {n}. {title} [{r.get('id')}]"))
            md.append(f"  - Phân loại: {note}\n")
            chosen.append({"n": n, "title": title, "id": r.get("id"), **x})
    if hyp:
        md.append("## Kiểm chứng giả thuyết Viwiki\n")
        md.append("| chỉ số | giá trị | ghi chú |\n|---|---|---|")
        md += [f"| {h['metric']} | {h['value']} | {h['note']} |" for h in hyp]
        md.append("")
    if len(models) >= 2:
        md += comparison_examples(per_sent, models, rng, n)
    out = tab_dir / "error_examples.md"
    out.write_text("\n".join(md), encoding="utf-8")
    log.info("Đã ghi %s (%d ví dụ cho %s)", out, n, A)
    return chosen


# ---------------------------------------------------------------------------- 5. so sánh mô hình
def _status(items) -> dict[int, str]:
    return {x["pos"]: x["cat"] for x in items}


def comparison_rows(per_sent, models: list[str]) -> list[dict]:
    rows = []
    A = models[0]
    for B in models[1:]:
        for ds in DATASETS:
            if (A, ds) not in per_sent or (B, ds) not in per_sent:
                continue
            ra, ia = per_sent[(A, ds)]
            rb, ib = per_sent[(B, ds)]
            c = Counter()
            for x, y in zip(ia, ib):
                sa, sb = _status(x), _status(y)
                for pos in sa.keys() | sb.keys():
                    a, b = sa.get(pos), sb.get(pos)
                    if a in ("sua_dung", "sua_sai", "bo_sot") or b in ("sua_dung", "sua_sai", "bo_sot"):
                        ok_a, ok_b = a == "sua_dung", b == "sua_dung"
                        c[{(True, True): "ca_hai_dung", (True, False): "A_dung_B_sai",
                           (False, True): "B_dung_A_sai", (False, False): "ca_hai_sai"}[(ok_a, ok_b)]] += 1
                    else:
                        c[{(True, True): "ca_hai_qua_tay", (True, False): "chi_A_qua_tay",
                           (False, True): "chi_B_qua_tay"}[(a == "sua_qua_tay", b == "sua_qua_tay")]] += 1
            for k in ["ca_hai_dung", "A_dung_B_sai", "B_dung_A_sai", "ca_hai_sai",
                      "ca_hai_qua_tay", "chi_A_qua_tay", "chi_B_qua_tay"]:
                rows.append({"model_A": A, "model_B": B, "dataset": ds, "group": k, "n": c[k]})
    return rows


def comparison_examples(per_sent, models, rng, start) -> list[str]:
    A = models[0]
    md, n = [], start
    for B in models[1:]:
        md.append(f"## So sánh {MODEL_LABEL.get(A, A)} và {MODEL_LABEL.get(B, B)}\n")
        for want, title in [(("sua_dung", "!"), f"{A} đúng, {B} sai"), (("!", "sua_dung"), f"{B} đúng, {A} sai"),
                            (("!", "!"), "cả hai sai")]:
            cands = []
            for ds in DATASETS:
                if (A, ds) not in per_sent or (B, ds) not in per_sent:
                    continue
                ra, ia = per_sent[(A, ds)]
                _, ib = per_sent[(B, ds)]
                for r, x, y in zip(ra, ia, ib):
                    if len(_tok(r["src"])) > 45:
                        continue
                    sa, sb = _status(x), _status(y)
                    s_tok = _tok(r["src"])
                    for pos in sa.keys() | sb.keys():
                        a, b = sa.get(pos), sb.get(pos)
                        if not ({a, b} & {"sua_dung", "sua_sai", "bo_sot"}) or not _HAS_LETTER.search(s_tok[pos]):
                            continue
                        ok = [(v == w) if w != "!" else (v != "sua_dung") for w, v in zip(want, (a, b))]
                        if all(ok):
                            cands.append((ds, r, pos))
            rng.shuffle(cands)
            per_ds = Counter()
            k = 0
            for ds, r, pos in cands:
                if per_ds[ds] >= 1 and k < 3 and len({d for d, _, _ in cands}) > 1:
                    continue
                per_ds[ds] += 1
                k += 1
                n += 1
                preds = {A: r["pred"], B: next(q["pred"] for q in per_sent[(B, ds)][0] if q.get("id") == r.get("id"))}
                md.append(render_example(r, [pos], f"Ví dụ {n}. {title} [{r.get('id')}]", preds))
                if k >= 3:
                    break
    return md


# ---------------------------------------------------------------------------- 6. A vs C theo kiểu lỗi thật
def ab_recall_rows(per_sent, models: list[str], lexicon: set[str]) -> list[dict]:
    """Recall sửa lỗi trên lỗi thật, tách theo (a) lỗi tạo ra từ không có thật / từ có thật trong từ điển,
    (b) kiểu lỗi suy từ cặp (sai, đúng). Lỗi từ có thật (vd giảm→giam) chỉ phát hiện được nhờ ngữ cảnh."""
    rows = []
    for m in models:
        for ds in ["vsec_test", "viwiki_test"]:
            if (m, ds) not in per_sent:
                continue
            tot, ok = Counter(), Counter()
            for it in per_sent[(m, ds)][1]:
                for x in it:
                    if x["cat"] not in ("sua_dung", "sua_sai", "bo_sot") or not _HAS_LETTER.search(x["src_tok"]):
                        continue
                    word = "Lỗi ra từ có thật" if x["src_tok"].lower() in lexicon else "Lỗi ra từ không có thật"
                    for key in [("loai_tu", word), ("kieu_loi", real_error_kind(x["src_tok"], x["gold_str"]))]:
                        tot[key] += 1
                        ok[key] += x["cat"] == "sua_dung"
            for (group, kind), n in sorted(tot.items()):
                rows.append({"model": m, "dataset": ds, "group": group, "kind": kind, "n": n,
                             "cor_r": round(ok[(group, kind)] / n, 4)})
    return rows


def run_ab_figures(rows: list[dict], models: list[str], tab_dir: Path, fig_dir: Path) -> None:
    plt = _plt()
    # (1) Recall theo loại từ gốc (không có thật / có thật), VSEC và Viwiki
    kinds = ["Lỗi ra từ không có thật", "Lỗi ra từ có thật"]
    d = {(r["model"], r["dataset"], r["kind"]): r for r in rows if r["group"] == "loai_tu"}
    if d:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
        for ax, ds in zip(axes, ["vsec_test", "viwiki_test"]):
            w = 0.8 / len(models)
            for k, m in enumerate(models):
                vals = [float(d[(m, ds, kd)]["cor_r"]) if (m, ds, kd) in d else 0 for kd in kinds]
                bars = ax.bar([i + (k - (len(models) - 1) / 2) * w for i in range(len(kinds))], vals, w,
                              color=COLORS[k], label=MODEL_LABEL.get(m, m))
                _bar_labels(ax, bars)
            ns = [d.get((models[0], ds, kd), {}).get("n", "?") for kd in kinds]
            ax.set_xticks(range(len(kinds)), [f"{kd}\n(n={n})" for kd, n in zip(kinds, ns)])
            ax.set_title(DS_LABEL[ds])
            ax.set_ylim(0, 1.05)
        axes[0].set_ylabel("Correction recall")
        axes[0].legend(fontsize=8)
        fig.suptitle("Recall trên lỗi thật: lỗi tạo ra từ không có thật và lỗi tạo ra từ có thật")
        _save(fig, fig_dir, "ab_recall_word_kind.png")

    # (2) Ablation C: mật độ nhiễu và quy mô train, trên VSEC-test
    main = {(r["model"], r["dataset"]): r for r in _read_csv(tab_dir / "main_results.csv")}
    abl = [m for m in ABLATION_C if (m, "vsec_test") in main]
    if len(abl) >= 2:
        out = []
        for m in abl:
            r = main[(m, "vsec_test")]
            out.append({"model": m, "density": ABLATION_C[m][0], "setting": ABLATION_C[m][1].replace("\n", " "),
                        **{k: r[k] for k in ["cor_p", "cor_r", "cor_f1", "over_correction"]}})
        write_csv(tab_dir / "ablation_noise_density.csv", out)
        metrics = [("cor_p", "Precision"), ("cor_r", "Recall"), ("cor_f1", "F1"), ("over_correction", "Sửa quá tay")]
        fig, ax = plt.subplots(figsize=(9, 4))
        w = 0.8 / len(abl)
        for k, m in enumerate(abl):
            vals = [float(main[(m, "vsec_test")][key]) for key, _ in metrics]
            bars = ax.bar([i + (k - (len(abl) - 1) / 2) * w for i in range(len(metrics))], vals, w,
                          color=COLORS[k], label=ABLATION_C[m][1].replace("\n", " "))
            _bar_labels(ax, bars)
        ax.set_xticks(range(len(metrics)), [lab for _, lab in metrics])
        ax.set_ylim(0, 1.05)
        ax.set_title("Ablation C trên VSEC-test: mật độ nhiễu khi sinh dữ liệu train")
        ax.legend(fontsize=8, ncol=2)
        _save(fig, fig_dir, "ablation_noise_density.png")
    plt.close("all")


# ---------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["ngram"],
                    help="danh sách mô hình (tên trước '__' trong file dự đoán); mô hình đầu là mốc so sánh")
    ap.add_argument("--pred-dir", default=None, help="mặc định outputs/predictions (hoặc thư mục repo chính)")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--lexicon", default=None, help="mặc định outputs/checkpoints/ngram/o3_kneser_ney/lexicon.json")
    ap.add_argument("--fig-dir", default=str(ROOT / "outputs/figures"))
    ap.add_argument("--tab-dir", default=str(ROOT / "outputs/tables"))
    ap.add_argument("--steps", default="eda,errors,results", help="tập con của eda,errors,results")
    ap.add_argument("--n-train", type=int, default=20000, help="số câu syn_train đầu tiên đưa vào EDA")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    def find(rel: str, given: str | None) -> Path:
        if given:
            return Path(given)
        for base in [ROOT, *ROOT.parents]:  # worktree nằm trong repo chính: tìm ngược lên
            p = base / rel
            if p.is_file() or (p.is_dir() and any(f.name != ".gitkeep" for f in p.iterdir())):
                return p
        return ROOT / rel

    pred_dir = find("outputs/predictions", args.pred_dir)
    data_dir = find("data/processed", args.data_dir)
    lex_path = find("outputs/checkpoints/ngram/o3_kneser_ney/lexicon.json", args.lexicon)
    fig_dir, tab_dir = Path(args.fig_dir), Path(args.tab_dir)
    steps = set(args.steps.split(","))
    log.info("predictions=%s data=%s lexicon=%s", pred_dir, data_dir, lex_path)

    if "eda" in steps:
        run_eda(data_dir, fig_dir, tab_dir, args.n_train)
    breakdown, per_sent = [], {}
    if steps & {"errors", "results"}:
        lexicon = load_lexicon(lex_path)
        breakdown, per_sent = run_breakdown(pred_dir, args.models, lexicon)
        write_csv(tab_dir / "error_breakdown.csv", breakdown)
    if "errors" in steps:
        hyp = viwiki_hypothesis(per_sent, args.models[0], lexicon, tab_dir)
        run_examples(per_sent, args.models, tab_dir, hyp, args.seed)
        if len(args.models) >= 2:
            write_csv(tab_dir / "model_comparison.csv", comparison_rows(per_sent, args.models))
            ab_rows = ab_recall_rows(per_sent, args.models, lexicon)
            write_csv(tab_dir / "ab_recall_by_kind.csv", ab_rows)
            run_ab_figures(ab_rows, args.models, tab_dir, fig_dir)
    if "results" in steps:
        run_results(tab_dir, fig_dir, breakdown, args.models)


if __name__ == "__main__":
    main()
