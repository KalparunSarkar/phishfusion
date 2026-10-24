"""Real-dataset acquisition and preparation.

Primary dataset: **PhreshPhish** (HuggingFace ``phreshphish/phreshphish``), a
recent, high-quality corpus that - unlike most public phishing datasets - ships
the **raw URL and the raw rendered HTML** for every sample (Plus labels and
metadata).  This lets the framework's own Stage I/II extractors run end-to-end on
real pages, exactly as intended, without any live crawling.

The loader streams Parquet shards with PyArrow (so a 300 MB shard is never fully
materialised), samples a bounded number of rows per shard, and truncates HTML to
``HTML_MAX_CHARS`` to keep memory/time predictable.

A generic CSV/JSONL path already exists in ``data_loader.py``; this module only
adds the networked, shard-based sources.
"""

import argparse
import os
import random
import urllib.request
from pathlib import Path

from config import DATA_DIR, HTML_MAX_CHARS, RANDOM_SEED

HF_BASE = "https://huggingface.co/datasets/phreshphish/phreshphish/resolve/main/"
PH_DIR = DATA_DIR / "phreshphish"

PH_SPLITS = {
    "train": [f"data/train-{i:03d}.parquet" for i in range(56)],
    "test": [f"data/test-{i:03d}.parquet" for i in range(21)],
}


def _download(url, dest, retries=3):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, dest)
            return dest
        except Exception as exc:  # pragma: no cover - network dependent
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1}/{retries} for {url}: {exc}")


def download_phreshphish_shards(split, n_shards, dest_dir=PH_DIR):
    """Download the first ``n_shards`` Parquet files of a split."""
    paths = []
    for remote in PH_SPLITS[split][:n_shards]:
        local = Path(dest_dir) / os.path.basename(remote)
        if not (local.exists() and local.stat().st_size > 1_000_000):
            print(f"  downloading {remote} ...")
            _download(HF_BASE + remote, local)
        paths.append(local)
    return paths


def _label_to_int(value):
    return 1 if str(value).strip().lower() in ("phish", "phishing", "1", "true") else 0


def _iter_parquet(path, max_html):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(path))
    for batch in pf.iter_batches(batch_size=128, columns=["url", "label", "html"]):
        cols = batch.to_pydict()
        for url, label, html in zip(cols["url"], cols["label"], cols["html"]):
            if not url:
                continue
            html = html or ""
            if max_html and len(html) > max_html:
                html = html[:max_html]
            yield {"url": url, "label": _label_to_int(label), "html": html}


def load_phreshphish(split="train", n_shards=1, per_shard=None,
                     max_html=HTML_MAX_CHARS, seed=RANDOM_SEED, balance=True):
    """Load records from local PhreshPhish shards (download if absent).

    ``per_shard`` bounds how many rows are drawn from each shard (randomly, with
    a fixed seed for reproducibility).  ``balance`` equalises the two classes.
    """
    paths = download_phreshphish_shards(split, n_shards)
    rng = random.Random(seed)
    records = []
    for p in paths:
        rows = list(_iter_parquet(p, max_html))
        if per_shard and len(rows) > per_shard:
            rows = rng.sample(rows, per_shard)
        records.extend(rows)
        print(f"  {Path(p).name}: {len(rows)} rows (kept)")
    if balance:
        records = balance_classes(records, seed=seed)
    return records


def balance_classes(records, seed=RANDOM_SEED):
    rng = random.Random(seed)
    pos = [r for r in records if r["label"] == 1]
    neg = [r for r in records if r["label"] == 0]
    n = min(len(pos), len(neg))
    out = rng.sample(pos, n) + rng.sample(neg, n)
    rng.shuffle(out)
    return out


def load_real_dataset(name="phreshphish", split="train", n_shards=1, per_shard=None,
                      max_html=HTML_MAX_CHARS, seed=RANDOM_SEED):
    if name == "phreshphish":
        return load_phreshphish(split, n_shards, per_shard, max_html, seed)
    raise ValueError(f"unknown dataset: {name}")


def build_csv(out_path=None, train_shards=3, test_shards=1, per_shard=1000,
              max_html=HTML_MAX_CHARS, seed=RANDOM_SEED):
    """Download shards and write train/test CSVs ready for ``pipeline.py train``."""
    from data_loader import save_records

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_recs = load_phreshphish("train", train_shards, per_shard, max_html, seed)
    test_recs = load_phreshphish("test", test_shards, per_shard, max_html, seed)
    tpath = DATA_DIR / "phreshphish_train.csv"
    vpath = DATA_DIR / "phreshphish_test.csv"
    save_records(train_recs, tpath)
    save_records(test_recs, vpath)
    print(f"train={len(train_recs)} -> {tpath}")
    print(f"test ={len(test_recs)} -> {vpath}")
    return tpath, vpath


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download and prepare PhreshPhish")
    ap.add_argument("--train-shards", type=int, default=3)
    ap.add_argument("--test-shards", type=int, default=1)
    ap.add_argument("--per-shard", type=int, default=1000)
    ap.add_argument("--max-html", type=int, default=HTML_MAX_CHARS)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    a = ap.parse_args()
    build_csv(train_shards=a.train_shards, test_shards=a.test_shards,
              per_shard=a.per_shard, max_html=a.max_html, seed=a.seed)
