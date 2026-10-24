import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import synthesize_dataset  # noqa: E402
from explainer import ExplanationGenerator, FeatureAttributor  # noqa: E402
from fusion_classifier import FusionClassifier  # noqa: E402
from pipeline import PhishingPipeline  # noqa: E402


def test_synthetic_dataset_shape_and_balance():
    recs = synthesize_dataset(n=100, seed=1)
    assert len(recs) == 100
    labels = {r["label"] for r in recs}
    assert labels == {0, 1}
    assert all(r["url"] and r["html"] for r in recs)


def test_pipeline_end_to_end_offline():
    recs = synthesize_dataset(n=120, seed=2)
    pipe = PhishingPipeline(seed=2)
    metrics = pipe.fit(recs, mode="attention", head="mlp", use_llm=False, verbose=False)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["n"] > 0
    # feature names align with model dims
    assert len(pipe.feature_names) == sum(pipe.dims)


def test_pipeline_predict_returns_explanation():
    recs = synthesize_dataset(n=120, seed=3)
    pipe = PhishingPipeline(seed=3)
    pipe.fit(recs, mode="concat", head="xgb", use_llm=False, verbose=False)
    result = pipe.predict("http://secure-login.paypal.verify.xyz/webscr?cmd=login",
                          html="<html><body><form action=''><input type='password'></form></body></html>",
                          fetch=False, explain=True, use_llm=False)
    assert result["label"] in ("phishing", "legitimate")
    assert 0.0 <= result["phishing_probability"] <= 1.0
    assert "explanation" in result and result["explanation"]
    assert "top_features" in result


def test_ablation_masking_runs():
    from run_experiments import run

    results = run(data_path=None, mode="concat", head="xgb", use_llm=False, n=150, seed=4)
    assert "url+html+llm (proposed)" in results
    assert "url+html (base paper)" in results


def test_evaluate_records_returns_metrics_and_probs():
    recs = synthesize_dataset(n=100, seed=5)
    pipe = PhishingPipeline(seed=5)
    pipe.fit(recs[:80], mode="concat", head="xgb", use_llm=False, verbose=False)
    m = pipe.evaluate_records(recs[80:], use_llm=False)
    assert 0.0 <= m["accuracy"] <= 1.0
    assert "roc_auc" in m and "probs" in m and "labels" in m
    assert len(m["probs"]) == len(m["labels"]) == m["n"]
