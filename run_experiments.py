"""Ablation study: quantify the contribution of each modality and of fusion.

Runs URL-only, HTML-only, LLM-only, every pairwise combination and the full
tri-modal fusion, using the same train/test split, and reports accuracy, F1,
ROC-AUC and false-positive rate.  The base-paper configuration (URL+HTML, naive
concatenation, XGBoost) is included explicitly as the reference point.

Usage::

    python run_experiments.py --data data/synthetic.csv --mode attention
    python run_experiments.py            # synthetic data, offline
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from config import ARTIFACT_DIR, LLM_MODEL, RANDOM_SEED
from data_loader import load_records, synthesize_dataset
from fusion_classifier import FusionClassifier
from pipeline import PhishingPipeline


ABLATIONS = [
    ("url", ["url"]),
    ("html", ["html"]),
    ("llm", ["llm"]),
    ("url+html (base paper)", ["url", "html"]),
    ("url+llm", ["url", "llm"]),
    ("html+llm", ["html", "llm"]),
    ("url+html+llm (proposed)", ["url", "html", "llm"]),
]


def _mask(block, enabled, name):
    """Return the modality block, or a single zero column when disabled."""
    if name in enabled:
        return block
    return np.zeros((block.shape[0], 1), dtype=np.float32)


def _prepare(data_path, test_path, use_llm, n, seed, cache=None, llm_model=LLM_MODEL):
    """Return (Xu_tr, Xh_tr, Xl_tr, y_tr, Xu_te, Xh_te, Xl_te, y_te)."""
    if data_path:
        records = load_records(data_path)
    else:
        records = synthesize_dataset(n=n, seed=seed)

    pipe = PhishingPipeline(seed=seed, llm_model=llm_model)
    bundle = pipe._raw_features(records, use_llm=use_llm, fetch_missing=False,
                                cache_path=cache)

    if test_path:
        test_records = load_records(test_path)
        test_bundle = pipe._raw_features(test_records, use_llm=use_llm,
                                         fetch_missing=False, cache_path=cache)
        train_idx = np.arange(len(records))
        test_idx = np.arange(len(test_records))
        pipe.text_builder.fit([bundle["url_texts"][i] for i in train_idx],
                              [bundle["page_texts"][i] for i in train_idx])
        tr = pipe._compose_from(bundle, train_idx, refit_text=False)
        te = pipe._compose_from(test_bundle, test_idx, refit_text=False)
        y_tr = bundle["labels"][train_idx]
        y_te = test_bundle["labels"][test_idx]
    else:
        idx = np.arange(len(records))
        strat = bundle["labels"] if len(np.unique(bundle["labels"])) > 1 else None
        train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=seed,
                                               stratify=strat)
        (tr, te) = pipe._compose(bundle, train_idx, test_idx)
        y_tr = bundle["labels"][train_idx]
        y_te = bundle["labels"][test_idx]
    return (*tr, y_tr, *te, y_te)


def run(data_path=None, mode="attention", head="xgb", use_llm=False, n=500, seed=RANDOM_SEED,
        test_path=None, cache=None, llm_model=LLM_MODEL, out_name="ablation.json"):
    Xu_tr, Xh_tr, Xl_tr, y_tr, Xu_te, Xh_te, Xl_te, y_te = _prepare(
        data_path, test_path, use_llm, n, seed, cache, llm_model)

    results = {}
    split_desc = f"official split ({len(y_tr)}/{len(y_te)})" if test_path else "80/20 split"
    print(f"\nAblation study (fusion={mode}, head={head}, llm_features={use_llm}, {split_desc})")
    print("-" * 78)
    print(f"{'configuration':30} {'acc':>7} {'f1':>7} {'auc':>7} {'fpr':>7}")
    print("-" * 78)
    for name, enabled in ABLATIONS:
        u_tr, h_tr, l_tr = (_mask(Xu_tr, enabled, "url"),
                            _mask(Xh_tr, enabled, "html"),
                            _mask(Xl_tr, enabled, "llm"))
        u_te, h_te, l_te = (_mask(Xu_te, enabled, "url"),
                            _mask(Xh_te, enabled, "html"),
                            _mask(Xl_te, enabled, "llm"))
        model = FusionClassifier(fusion_mode=mode, head=head, seed=seed, epochs=60)
        model.fit(u_tr, h_tr, l_tr, y_tr, (u_te, h_te, l_te), y_te)
        m = model.evaluate(u_te, h_te, l_te, y_te)
        results[name] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                         for k, v in m.items()}
        print(f"{name:30} {m['accuracy']:7.3f} {m['f1']:7.3f} "
              f"{m['roc_auc']:7.3f} {m['fpr']:7.3f}")
    print("-" * 78)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACT_DIR / out_name
    out.write_text(json.dumps({
        "config": {"mode": mode, "head": head, "llm": use_llm,
                   "dataset": data_path or "synthetic", "test": test_path,
                   "n_train": int(len(y_tr)), "n_test": int(len(y_te))},
        "results": results}, indent=2))
    print(f"Saved -> {out}\n")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--test-data", default=None)
    ap.add_argument("--mode", choices=["concat", "attention"], default="attention")
    ap.add_argument("--head", choices=["xgb", "mlp"], default="xgb")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--llm-model", default=LLM_MODEL)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--out", default="ablation.json")
    a = ap.parse_args()
    run(a.data, a.mode, a.head, a.llm, a.n, a.seed, a.test_data, a.cache, a.llm_model, a.out)
