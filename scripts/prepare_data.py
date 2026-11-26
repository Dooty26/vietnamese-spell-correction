"""Bước 1: chuẩn bị dữ liệu.

(a) Tập lỗi thật (đã cài): tải VSEC + Viwiki -> định dạng chung -> chia VSEC dev/test.
(b) Binhvq: đọc ngẫu nhiên row group, làm sạch, loại trùng và loại câu rò rỉ từ (a), chia theo đoạn văn.

Đầu ra (data/processed/):
    vsec_dev.jsonl, vsec_test.jsonl, viwiki_test.jsonl, stats_real.json
    lm_corpus.txt, clean_syn_{train,val,test}.txt                (phần b)

Chạy:
    python scripts/prepare_data.py --config configs/data.yaml --skip-binhvq   # chỉ phần (a)
    python scripts/prepare_data.py --config configs/data.yaml --scale 0.01    # chạy thử nhanh (~5,6k câu)
    python scripts/prepare_data.py --config configs/data.yaml                 # đầy đủ (~560k câu, 20–40 phút)
"""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from vspell.data.binhvq import clean_text, keep_sentence, split_by_paragraph, split_sentences, stream_paragraphs
from vspell.data.dedup import build_blocklist, is_leaked, sentence_key
from vspell.data.viwiki import download_viwiki, load_viwiki
from vspell.data.vsec import download_vsec, load_vsec, split_dev_test
from vspell.eval.align import error_positions
from vspell.normalize import split_syllables
from vspell.utils import (get_logger, load_yaml, read_jsonl, resolve_hf_revision, run_provenance, set_seed,
                          sha256_file, write_jsonl)


def describe(records: list[dict]) -> dict:
    """Thống kê nhanh cho EDA: số câu, độ dài, số lỗi (mức âm tiết, sau chuẩn hóa)."""
    lengths, n_err, with_err = [], 0, 0
    for r in records:
        s, t = split_syllables(r["src"]), split_syllables(r["tgt"])
        e = len(error_positions(s, t))
        lengths.append(len(s))
        n_err += e
        with_err += e > 0
    return {
        "n_sentences": len(records),
        "n_sentences_with_errors": with_err,
        "n_syllable_errors": n_err,
        "error_rate_per_token": round(n_err / max(sum(lengths), 1), 4),
        "tokens_mean": round(statistics.mean(lengths), 2) if lengths else 0,
        "tokens_median": statistics.median(lengths) if lengths else 0,
        "tokens_max": max(lengths, default=0),
    }


def prepare_real_sets(cfg: dict, log) -> dict:
    raw, out = Path(cfg["paths"]["raw"]), Path(cfg["paths"]["processed"])
    vsec_path = download_vsec(raw / "VSEC.jsonl", cfg["vsec"]["github_raw"])
    viwiki_path = download_viwiki(raw / "viwiki_spelling_test.json")

    vsec_drop: dict = {}
    vsec = load_vsec(vsec_path, stats=vsec_drop)
    dev, test = split_dev_test(vsec, cfg["vsec"]["dev_ratio"], cfg["seed"])
    viwiki_drop: dict = {}
    viwiki = load_viwiki(viwiki_path, stats=viwiki_drop)

    write_jsonl(dev, out / "vsec_dev.jsonl")
    write_jsonl(test, out / "vsec_test.jsonl")
    write_jsonl(viwiki, out / "viwiki_test.jsonl")

    stats = {
        "vsec_all": {**describe(vsec), "n_repaired_annotations": sum(r["n_repaired"] for r in vsec),
                     **vsec_drop},
        "vsec_dev": describe(dev),
        "vsec_test": describe(test),
        "viwiki_test": {**describe(viwiki), **dict(sorted(viwiki_drop.items()))},
        "sources_sha256": {p.name: sha256_file(p) for p in (vsec_path, viwiki_path)},
    }
    (out / "stats_real.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, st in stats.items():
        log.info("%-12s %s", name, st)
    return stats


def prepare_binhvq(cfg: dict, log, scale: float = 1.0) -> dict:
    """Đọc ngẫu nhiên row group của Binhvq -> làm sạch -> lọc câu -> loại trùng/rò rỉ -> chia theo đoạn."""
    bcfg, dcfg = cfg["binhvq"], cfg["dedup"]
    out = Path(cfg["paths"]["processed"])
    # Thứ tự chia: tập đánh giá trước để luôn đủ kích thước
    order = ["syn_test", "syn_val", "syn_train", "lm_corpus"]
    sizes = {k: max(1, int(bcfg["split_sentences"][k] * scale)) for k in order}
    need = int(sum(sizes.values()) * 1.02)

    test_recs = [r for name in ("vsec_dev", "vsec_test", "viwiki_test")
                 for r in read_jsonl(out / f"{name}.jsonl")]
    keys, grams = build_blocklist(test_recs, dcfg["ngram"])
    log.info("Blocklist: %d câu, %d n-gram từ VSEC/Viwiki", len(keys), len(grams))

    seen: set[str] = set()
    paragraphs: list[tuple[str, list[str]]] = []
    c = Counter()
    pbar = tqdm(total=need, desc="Câu sạch Binhvq", unit="câu")
    # Cố định phiên bản Binhvq: revision trong config, hoặc commit hiện tại của nhánh main (ghi vào stats)
    revision = resolve_hf_revision(bcfg["hf_id"], bcfg.get("revision"), repo_type="dataset") or bcfg.get("revision")
    log.info("Binhvq %s@%s", bcfg["hf_id"], revision)
    for pid, text in stream_paragraphs(bcfg["hf_id"], bcfg["max_row_groups"], cfg["seed"], revision=revision):
        c["paragraphs"] += 1
        kept = []
        for sent in split_sentences(clean_text(text)):
            c["sentences_seen"] += 1
            if not keep_sentence(sent, bcfg["min_syllables"], bcfg["max_syllables"]):
                c["filtered"] += 1
                continue
            key = sentence_key(sent)
            if key in seen:
                c["duplicate"] += 1
                continue
            if is_leaked(sent, keys, grams, dcfg["ngram"], dcfg["overlap_threshold"]):
                c["leaked_test"] += 1
                continue
            seen.add(key)
            kept.append(sent)
        if kept:
            paragraphs.append((pid, kept))
            c["kept"] += len(kept)
            pbar.update(len(kept))
        if c["kept"] >= need:
            break
    pbar.close()
    if c["kept"] < sum(sizes.values()):
        log.warning("Chỉ có %d câu, cần %d: tăng binhvq.max_row_groups", c["kept"], sum(sizes.values()))

    splits = split_by_paragraph(paragraphs, sizes, cfg["seed"])
    files = {"lm_corpus": "lm_corpus.txt", "syn_train": "clean_syn_train.txt",
             "syn_val": "clean_syn_val.txt", "syn_test": "clean_syn_test.txt"}
    for name, fname in files.items():
        (out / fname).write_text("\n".join(splits[name]) + "\n", encoding="utf-8")
    stats = {**c, "split_sizes": {k: len(v) for k, v in splits.items()}, "scale": scale,
             "hf_id": bcfg["hf_id"], "revision": revision,
             "provenance": run_provenance({fname: out / fname for fname in files.values()})}
    (out / "stats_binhvq.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Binhvq: %s", stats)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--skip-binhvq", action="store_true", help="chỉ chuẩn bị VSEC/Viwiki")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="nhân kích thước các tập Binhvq (vd 0.01 để chạy thử nhanh)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    log = get_logger()

    prepare_real_sets(cfg, log)
    if not args.skip_binhvq:
        prepare_binhvq(cfg, log, args.scale)


if __name__ == "__main__":
    main()
