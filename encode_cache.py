"""Pre-compute LLM semantic features for a subset and cache them.

LLM inference is the expensive stage (~10 s/sample on CPU), so it is decoupled:
this script builds a balanced subset, encodes every (URL, page-text) pair once,
and writes the cache to JSON.  ``pipeline.py``/``run_experiments.py`` then read
the cache via ``--cache`` and never re-encode the same URL.

Usage:
    python encode_cache.py --train-n 100 --test-n 60 --model qwen2.5-coder:7b
"""

import argparse
import json
import random
import time
from pathlib import Path

from config import DATA_DIR, RANDOM_SEED
from data_loader import load_records, save_records
from html_feature_extractor import extract_html_features
from llm_semantic_encoder import LLMSemanticEncoder


def _balanced(records, n, seed):
    rng = random.Random(seed)
    pos = [r for r in records if r["label"] == 1]
    neg = [r for r in records if r["label"] == 0]
    k = min(n // 2, len(pos), len(neg))
    out = rng.sample(pos, k) + rng.sample(neg, k)
    rng.shuffle(out)
    return out


def build_subset(train_path, test_path, train_n, test_n, seed):
    train = _balanced(load_records(train_path), train_n, seed)
    test = _balanced(load_records(test_path), test_n, seed)
    tp = DATA_DIR / "phreshphish_llm_train.csv"
    vp = DATA_DIR / "phreshphish_llm_test.csv"
    save_records(train, tp)
    save_records(test, vp)
    return train, test, tp, vp


def encode_all(records, model, cache_path, log_every=10):
    cache = {}
    if Path(cache_path).exists():
        cache = json.loads(Path(cache_path).read_text())
    enc = LLMSemanticEncoder(model=model)
    if not enc.client.is_available():
        raise SystemExit("Ollama not available at http://localhost:11434")
    todo = [r for r in records if r["url"] not in cache]
    print(f"{len(records)} records, {len(todo)} to encode, {len(cache)} cached")
    t0 = time.time()
    for i, rec in enumerate(todo, 1):
        _, page_text = extract_html_features(rec["html"], rec["url"])
        vec, _ = enc.encode(rec["url"], page_text)
        cache[rec["url"]] = vec
        if i % log_every == 0 or i == len(todo):
            Path(cache_path).write_text(json.dumps(cache))
            rate = (time.time() - t0) / i
            eta = rate * (len(todo) - i)
            print(f"  {i}/{len(todo)} encoded  ({rate:.1f}s/sample, ETA {eta/60:.1f} min)",
                  flush=True)
    Path(cache_path).write_text(json.dumps(cache))
    print(f"cache written: {cache_path} ({len(cache)} entries)")
    return cache


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(DATA_DIR / "phreshphish_train.csv"))
    ap.add_argument("--test", default=str(DATA_DIR / "phreshphish_test.csv"))
    ap.add_argument("--train-n", type=int, default=100)
    ap.add_argument("--test-n", type=int, default=60)
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--cache", default=str(DATA_DIR / "llm_cache.json"))
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    a = ap.parse_args()
    train, test, tp, vp = build_subset(a.train, a.test, a.train_n, a.test_n, a.seed)
    print(f"subset: train={len(train)} ({tp}) test={len(test)} ({vp})")
    encode_all(train + test, a.model, a.cache)
