"""End-to-end pipeline: data -> features -> fusion -> train/eval -> explain.

Orchestrates the six stages of the proposal:

    Stage I     URL lexical + char-level TF-IDF features        -> f_url
    Stage II    HTML/DOM features + page-text char TF-IDF       -> f_html
    Stage III   LLM semantic signals                            -> f_llm
    Stage IV    gated-attention (or concatenation) fusion       -> z
    Stage V     classification head                             -> label, proba
    Stage VI    SHAP-grounded LLM explanation                   -> rationale

The pipeline can be driven from the command line::

    python pipeline.py demo            # synthetic data, offline, full run
    python pipeline.py train --data data.csv --mode attention --head xgb
    python pipeline.py predict --url http://secure-login.paypal.verify.xyz
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from config import (
    ARTIFACT_DIR,
    DATA_DIR,
    HTML_NUMERIC_KEYS,
    LLM_MODEL,
    LLM_TIMEOUT,
    OLLAMA_HOST,
    RANDOM_SEED,
    SEMANTIC_SCORE_KEYS,
    URL_NUMERIC_KEYS,
)
from data_loader import fetch_html, fetch_html_ex, load_records, save_records, synthesize_dataset
from explainer import ExplanationGenerator, FeatureAttributor
from fusion_classifier import FusionClassifier
from html_feature_extractor import extract_html_features, html_feature_vector
from llm_semantic_encoder import LLMSemanticEncoder
from text_features import TextFeatureBuilder, to_dense
from url_feature_extractor import extract_url_features, url_feature_vector


def _log(msg):
    print(msg, flush=True)


class PhishingPipeline:
    def __init__(self, text_kind="tfidf", use_network_features=False, seed=RANDOM_SEED,
                 llm_model=LLM_MODEL, llm_host=OLLAMA_HOST, llm_timeout=LLM_TIMEOUT):
        self.text_kind = text_kind
        self.use_network_features = use_network_features
        self.seed = seed
        self.text_builder = TextFeatureBuilder(kind=text_kind)
        self.llm_encoder = LLMSemanticEncoder(model=llm_model, host=llm_host,
                                              timeout=llm_timeout)
        self.model = None
        self.attributor = None
        self.explainer = ExplanationGenerator()
        self.feature_names = None
        self.dims = None

    # ------------------------------------------------------------- feature build
    def _raw_features(self, records, use_llm=True, fetch_missing=False, cache_path=None,
                      verbose=True):
        cache = {}
        if cache_path and Path(cache_path).exists():
            cache = json.loads(Path(cache_path).read_text())
        urls, labels = [], []
        url_num, html_num = [], []
        url_texts, page_texts = [], []
        llm_feats = []
        n = len(records)
        for i, rec in enumerate(records):
            url = rec["url"]
            html = rec.get("html") or ""
            if not html and fetch_missing:
                html, ok = fetch_html(url)
                if not ok:
                    html = ""
            urls.append(url)
            labels.append(int(rec["label"]))

            uf = extract_url_features(url, self.use_network_features)
            hf, page_text = extract_html_features(html, url)
            url_num.append(url_feature_vector(uf, self.use_network_features))
            html_num.append(html_feature_vector(hf))
            url_texts.append(url)
            page_texts.append(page_text)

            if use_llm:
                if url in cache:
                    vec = cache[url]
                else:
                    vec, _why = self.llm_encoder.encode(url, page_text)
                    cache[url] = vec
                llm_feats.append(vec)
            else:
                llm_feats.append([0.0] * len(SEMANTIC_SCORE_KEYS))

            if verbose and ((i + 1) % 50 == 0 or (i + 1) == n):
                _log(f"  features {i + 1}/{n}")

        if cache_path and use_llm:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_path).write_text(json.dumps(cache))

        return {
            "urls": urls,
            "labels": np.asarray(labels),
            "url_num": np.asarray(url_num, dtype=np.float32),
            "html_num": np.asarray(html_num, dtype=np.float32),
            "url_texts": url_texts,
            "page_texts": page_texts,
            "llm": np.asarray(llm_feats, dtype=np.float32),
        }

    def _compose_from(self, bundle, idx, refit_text=False):
        if refit_text:
            self.text_builder.fit(
                [bundle["url_texts"][i] for i in idx],
                [bundle["page_texts"][i] for i in idx],
            )
        url_char = to_dense(self.text_builder.url_vec.transform(
            [bundle["url_texts"][i] for i in idx]
        ))
        text_char = to_dense(self.text_builder.text_vec.transform(
            [bundle["page_texts"][i] for i in idx]
        ))
        X_url = np.hstack([bundle["url_num"][idx], url_char])
        X_html = np.hstack([bundle["html_num"][idx], text_char])
        X_llm = bundle["llm"][idx]
        return X_url, X_html, X_llm

    def _compose(self, bundle, train_idx, test_idx, refit_text=True):
        if refit_text:
            self.text_builder.fit(
                [bundle["url_texts"][i] for i in train_idx],
                [bundle["page_texts"][i] for i in train_idx],
            )
        return (self._compose_from(bundle, train_idx, refit_text=False),
                self._compose_from(bundle, test_idx, refit_text=False))

    def _make_feature_names(self):
        url_names = list(URL_NUMERIC_KEYS)
        if self.use_network_features:
            url_names = url_names + ["has_dns_record", "domain_age_days"]
        url_terms = self.text_builder.url_vec.feature_names()
        if url_terms:
            url_names += [f"url_tfidf[{t}]" for t in url_terms]
        else:
            url_names += [f"url_ngram_{i}" for i in range(self.text_builder.url_vec.n_features())]

        html_names = list(HTML_NUMERIC_KEYS)
        text_terms = self.text_builder.text_vec.feature_names()
        if text_terms:
            html_names += [f"text_tfidf[{t}]" for t in text_terms]
        else:
            html_names += [f"text_ngram_{i}"
                           for i in range(self.text_builder.text_vec.n_features())]

        llm_names = list(SEMANTIC_SCORE_KEYS)
        return url_names, html_names, llm_names

    # ---------------------------------------------------------------- fit/eval
    def fit(self, records, test_size=0.2, mode="attention", head="xgb",
            use_llm=True, fetch_missing=False, cache_path=None, verbose=True,
            test_records=None):
        if verbose:
            _log(f"Building raw features for {len(records)} records "
                 f"(llm={use_llm}, fetch_missing={fetch_missing})")
        bundle = self._raw_features(records, use_llm=use_llm,
                                    fetch_missing=fetch_missing, cache_path=cache_path)

        if test_records is not None:
            # Honor an external (official) train/test split.
            if verbose:
                _log(f"Building raw features for {len(test_records)} held-out records")
            test_bundle = self._raw_features(test_records, use_llm=use_llm,
                                             fetch_missing=fetch_missing,
                                             cache_path=cache_path)
            train_idx = np.arange(len(records))
            test_idx = np.arange(len(test_records))
            self.text_builder.fit(
                [bundle["url_texts"][i] for i in train_idx],
                [bundle["page_texts"][i] for i in train_idx],
            )
            tr = self._compose_from(bundle, train_idx, refit_text=False)
            te = self._compose_from(test_bundle, test_idx, refit_text=False)
            y_tr = bundle["labels"][train_idx]
            y_te = test_bundle["labels"][test_idx]
            self._train_bundle = bundle
            self._test_idx = test_idx
        else:
            idx = np.arange(len(records))
            strat = bundle["labels"] if len(np.unique(bundle["labels"])) > 1 else None
            train_idx, test_idx = train_test_split(
                idx, test_size=test_size, random_state=self.seed, stratify=strat
            )
            (tr, te) = self._compose(bundle, train_idx, test_idx)
            y_tr, y_te = bundle["labels"][train_idx], bundle["labels"][test_idx]
            self._train_bundle = bundle
            self._test_idx = test_idx

        Xu_tr, Xh_tr, Xl_tr = tr
        Xu_te, Xh_te, Xl_te = te

        if verbose:
            _log(f"Feature dims -> url={Xu_tr.shape[1]} html={Xh_tr.shape[1]} llm={Xl_tr.shape[1]}")
        self.dims = (Xu_tr.shape[1], Xh_tr.shape[1], Xl_tr.shape[1])

        self.model = FusionClassifier(fusion_mode=mode, head=head, seed=self.seed)
        self.model.fit(Xu_tr, Xh_tr, Xl_tr, y_tr,
                       (Xu_te, Xh_te, Xl_te), y_te)

        # feature names for explainability (order must match [X_url | X_html | X_llm])
        url_names, html_names, llm_names = self._make_feature_names()
        self.feature_names = url_names + html_names + llm_names

        Xc_train = np.hstack([Xu_tr, Xh_tr, Xl_tr])
        self.attributor = FeatureAttributor(self.model, self.feature_names, Xc_train)
        self._background = Xc_train[:200]
        self._train_bundle = bundle
        self._test_idx = test_idx
        self._split = (tr, te, y_tr, y_te)

        metrics = self.model.evaluate(Xu_te, Xh_te, Xl_te, y_te)
        if verbose:
            _log("Test metrics: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                                for k, v in metrics.items()}))
        return metrics

    def evaluate(self):
        (tr, te, y_tr, y_te) = self._split
        return self.model.evaluate(*te, y_te)

    def evaluate_records(self, records, use_llm=False, limit=None):
        """Score a set of labeled records with the loaded model (no refit).

        Returns the metric dict from :meth:`FusionClassifier.evaluate` plus
        ``probs``/``labels`` for plotting.
        """
        records = records[:limit] if limit else records
        raw = self._raw_features(records, use_llm=use_llm, verbose=False)
        idx = np.arange(len(records))
        X_url, X_html, X_llm = self._compose_from(raw, idx, refit_text=False)
        y = raw["labels"][idx]
        probs = self.model.predict_proba(X_url, X_html, X_llm)
        metrics = self.model.evaluate(X_url, X_html, X_llm, y)
        metrics["probs"] = probs
        metrics["labels"] = y
        return metrics

    # ---------------------------------------------------------------- prediction
    def predict(self, url, html=None, fetch=True, explain=True, use_llm=None,
                max_text_chars=800):
        fetch_info = None
        if html is None and fetch:
            fetch_info = fetch_html_ex(url)
            html = fetch_info["html"] if fetch_info["ok"] else ""
        html = html or ""
        uf = extract_url_features(url, self.use_network_features)
        hf, page_text = extract_html_features(html, url)
        url_num = np.array([url_feature_vector(uf, self.use_network_features)], dtype=np.float32)
        html_num = np.array([html_feature_vector(hf)], dtype=np.float32)
        url_char = to_dense(self.text_builder.url_vec.transform([url]))
        text_char = to_dense(self.text_builder.text_vec.transform([page_text]))
        X_url = np.hstack([url_num, url_char])
        X_html = np.hstack([html_num, text_char])
        if use_llm is None:
            use_llm = self.llm_encoder.client.is_available()
        if use_llm:
            vec, _ = self.llm_encoder.encode(url, page_text)
        else:
            vec = [0.0] * len(SEMANTIC_SCORE_KEYS)
        X_llm = np.array([vec], dtype=np.float32)

        proba = float(self.model.predict_proba(X_url, X_html, X_llm)[0])
        verdict = int(proba >= 0.5)

        result = {"url": url, "label": "phishing" if verdict else "legitimate",
                  "verdict": verdict, "phishing_probability": round(proba, 4),
                  "html_available": bool(html)}
        if fetch_info is not None:
            result["fetch"] = {
                "ok": fetch_info["ok"], "status": fetch_info["status"],
                "reason": fetch_info["reason"], "final_url": fetch_info.get("final_url"),
                "client": fetch_info.get("client"), "bytes": fetch_info.get("bytes"),
            }
        if self.model.fusion_mode == "attention":
            w = self.model.modality_weights(X_url, X_html, X_llm)
            if w is not None:
                result["modality_weights"] = {
                    "url": round(float(w[0, 0]), 3),
                    "html": round(float(w[0, 1]), 3),
                    "llm": round(float(w[0, 2]), 3),
                }
        if explain and self.attributor is not None:
            _, top = self.attributor.top_features(np.hstack([X_url, X_html, X_llm])[0], k=8)
            result["top_features"] = [{"feature": n_, "contribution": round(c, 4)} for n_, c in top]
            result["explanation"] = self.explainer.generate(
                url, verdict, proba, top, page_text=page_text[:max_text_chars],
                use_llm=bool(use_llm),
            )
        return result

    # ------------------------------------------------------------- persistence
    def save(self, prefix="phish_model"):
        import joblib

        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        base = ARTIFACT_DIR / prefix
        self.model.save(str(base) + ".model")
        joblib.dump({
            "text_builder": self.text_builder,
            "feature_names": self.feature_names,
            "dims": self.dims,
            "text_kind": self.text_kind,
            "use_network_features": self.use_network_features,
            "background": getattr(self, "_background", None),
        }, str(base) + ".meta")
        return str(base)

    @classmethod
    def load(cls, prefix="phish_model"):
        import joblib

        base = ARTIFACT_DIR / prefix
        meta = joblib.load(str(base) + ".meta")
        obj = cls(text_kind=meta["text_kind"],
                  use_network_features=meta["use_network_features"])
        obj.text_builder = meta["text_builder"]
        obj.feature_names = meta["feature_names"]
        obj.dims = meta["dims"]
        obj.model = FusionClassifier.load(str(base) + ".model")
        bg = meta.get("background")
        if bg is not None:
            obj.attributor = FeatureAttributor(obj.model, obj.feature_names, bg)
        else:
            obj.attributor = None
        return obj


# --------------------------------------------------------------------------- CLI
def _cmd_demo(args):
    records = synthesize_dataset(n=args.n, seed=args.seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "synthetic.csv"
    save_records(records, path)
    _log(f"Wrote synthetic dataset ({len(records)} rows) -> {path}")
    pipe = PhishingPipeline(seed=args.seed)
    metrics = pipe.fit(records, mode=args.mode, head=args.head, use_llm=args.llm)
    if args.save:
        loc = pipe.save()
        _log(f"Saved model artifacts -> {loc}")
    return metrics


def _cmd_train(args):
    records = load_records(args.data)
    _log(f"Loaded {len(records)} records from {args.data}")
    test_records = None
    if args.test_data:
        test_records = load_records(args.test_data)
        _log(f"Loaded {len(test_records)} held-out records from {args.test_data}")
    pipe = PhishingPipeline(llm_model=args.llm_model)
    metrics = pipe.fit(records, mode=args.mode, head=args.head, use_llm=args.llm,
                       fetch_missing=args.fetch, cache_path=args.cache,
                       test_records=test_records)
    if args.save:
        loc = pipe.save()
        _log(f"Saved model artifacts -> {loc}")
    return metrics


def _cmd_prepare(args):
    import datasets

    return datasets.build_csv(train_shards=args.train_shards, test_shards=args.test_shards,
                              per_shard=args.per_shard, max_html=args.max_html)


def _cmd_predict(args):
    pipe = PhishingPipeline.load(args.model)
    result = pipe.predict(args.url, use_llm=args.llm)
    print(json.dumps(result, indent=2))
    return result


def build_argparser():
    p = argparse.ArgumentParser(description="URL-HTML-LLM phishing detection pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="run on synthetic data")
    d.add_argument("--n", type=int, default=400)
    d.add_argument("--seed", type=int, default=RANDOM_SEED)
    d.add_argument("--mode", choices=["concat", "attention"], default="attention")
    d.add_argument("--head", choices=["xgb", "mlp"], default="xgb")
    d.add_argument("--llm", action="store_true", help="use local Ollama LLM features")
    d.add_argument("--save", action="store_true")
    d.set_defaults(func=_cmd_demo)

    t = sub.add_parser("train", help="train on a CSV/JSONL dataset")
    t.add_argument("--data", required=True)
    t.add_argument("--test-data", default=None, help="optional held-out split file")
    t.add_argument("--mode", choices=["concat", "attention"], default="attention")
    t.add_argument("--head", choices=["xgb", "mlp"], default="xgb")
    t.add_argument("--llm", action="store_true")
    t.add_argument("--llm-model", default=LLM_MODEL)
    t.add_argument("--fetch", action="store_true", help="fetch HTML for URL-only datasets")
    t.add_argument("--cache", default=None, help="path to cache LLM features (json)")
    t.add_argument("--save", action="store_true")
    t.set_defaults(func=_cmd_train)

    prep = sub.add_parser("prepare", help="download and prepare PhreshPhish dataset")
    prep.add_argument("--train-shards", type=int, default=3)
    prep.add_argument("--test-shards", type=int, default=1)
    prep.add_argument("--per-shard", type=int, default=1000)
    prep.add_argument("--max-html", type=int, default=120000)
    prep.set_defaults(func=_cmd_prepare)

    pr = sub.add_parser("predict", help="classify a single URL")
    pr.add_argument("--url", required=True)
    pr.add_argument("--model", default="phish_model")
    pr.add_argument("--llm", action="store_true")
    pr.set_defaults(func=_cmd_predict)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    args.func(args)
