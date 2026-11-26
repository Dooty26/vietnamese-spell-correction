"""Tiện ích chung: seed, logging, đọc/ghi JSONL và YAML."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import urllib.request
from pathlib import Path
from typing import Iterable, Iterator


def set_seed(seed: int = 42) -> None:
    """Cố định seed cho random, numpy và torch (nếu có)."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_logger(name: str = "vspell") -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "fsspec", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(name)


def load_yaml(path: str | Path) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(records: Iterable[dict], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def download_atomic(url: str, dest: str | Path) -> Path:
    """Tải `url` về `dest` nếu chưa có. Tải vào file tạm cùng thư mục rồi mới đổi tên, nên lần tải bị ngắt
    không để lại `dest` dở dang (lần chạy sau sẽ tải lại thay vì đọc file hỏng)."""
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)  # ContentTooShortError nếu thiếu byte so với Content-Length
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def run_provenance(files: dict[str, str | Path] | None = None,
                   packages: Iterable[str] = ("torch", "transformers", "datasets", "accelerate", "sentencepiece",
                                              "huggingface_hub", "pyarrow", "numpy")) -> dict:
    """Hồ sơ một lần chạy để tái lập: commit repo (+ có sửa đổi chưa commit), phiên bản Python/thư viện thực tế,
    SHA-256 của các file đầu vào. Mục nào không lấy được thì ghi None thay vì làm hỏng lần chạy."""
    from importlib import metadata

    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True,
                                  cwd=Path(__file__).resolve().parent).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    status = git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python": platform.python_version(),
        "packages": versions,
        "files_sha256": {k: sha256_file(v) if Path(v).exists() else None for k, v in (files or {}).items()},
    }


def resolve_hf_revision(repo_id: str, revision: str | None, repo_type: str = "model") -> str | None:
    """Đổi revision (nhánh/tag/None = main) thành mã commit cố định trên Hugging Face Hub; None nếu repo_id là
    thư mục cục bộ hoặc không truy cập được Hub."""
    if Path(repo_id).exists():
        return None
    try:
        from huggingface_hub import HfApi
        info = (HfApi().model_info if repo_type == "model" else HfApi().dataset_info)(repo_id, revision=revision)
        return info.sha
    except Exception as e:  # offline, repo riêng tư...
        get_logger().warning("Không xác định được commit của %s@%s trên Hub: %s", repo_id, revision, e)
        return None
