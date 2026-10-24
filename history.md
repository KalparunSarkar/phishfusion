# Change History

All modifications to the **URL–HTML–LLM Fusion Framework** codebase, newest
first. This file records *what changed, why, and how it was verified* so the
paper's implementation section can be traced back to concrete commits.

---

## 2026‑10‑14 — Web demo: monochrome theme + default URL

### Changed (`demo.py`)
* Restyled the web UI to a **black-and-white / monochrome** theme: white
  background, black text and 1px black borders, black buttons with white text,
  gray track bars. Phishing vs legitimate is shown by inverted badges
  (black-on-white vs white-on-black) instead of red/green. Positive vs negative
  SHAP contributions are distinguished with black vs gray bars.
* Default value of the "URL to analyse" field changed from
  `https://www.google.com/` to **`example.com`**.

### Verification
* `GET /` → white background, black foreground, no colored variables, default
  field = `example.com`.
* `POST /predict (url=example.com)` → **LEGITIMATE (1.6%)**, explanation rendered.
* `pytest tests/ -q` → **15 passed**.

---

## 2026‑10‑14 — Robust live HTML fetching

Triggered by a demo observation: `https://www.amazon.in/` was shown as
*"HTML fetched: False"* and classified phishing. Investigation showed the old
fetcher required HTTP **200** and was blocked (Amazon returns **202** with a
bot-verification page).

### Changed
* `data_loader.py` — replaced `fetch_html` with a robust implementation
  (`fetch_html_ex`, with `fetch_html` kept as a `(html, ok)` wrapper):
  * browser-like headers + persistent session;
  * `https`→`http` fallback and retries;
  * accepts any HTML body (not just 200), correct decoding;
  * **anti-bot / JS-challenge detection** (AWS WAF, Cloudflare, `bm-verify`,
    captcha, …) → reported as not-fetched instead of scored as the real page;
  * optional `curl_cffi` / `cloudscraper` clients (used if installed).
* `pipeline.py` — `predict()` now uses `fetch_html_ex` and returns a `fetch`
  block (`ok`, `status`, `reason`, `client`, `bytes`, `final_url`).
* `demo.py` — web result card and CLI show the fetch status/reason and warn when
  the verdict is URL-only.

### Reverted
* The missing-modality augmentation briefly added to `fusion_classifier.py` was
  **removed** at the user's request (no retraining); the trained model is
  unchanged.

### Findings (recorded, not fixed)
* Fetching now succeeds for most sites (`google.com`, `github.com`,
  `flipkart.com`) and correctly flags Amazon's challenge page.
* However, **fetching the HTML does not by itself fix false positives on live,
  JS-heavy sites**: `google.com` / `amazon.in` are out-of-distribution relative
  to the PhreshPhish training corpus, so the model can still misclassify them
  (`github.com` / `flipkart.com` are correct). Root causes: char-n-gram text
  features overfit to the training distribution, and a missing HTML block is
  seen as out-of-distribution because `StandardScaler` does not map all-zeros to
  all-zeros — exactly what the reverted augmentation was meant to address.

### Verification
* `pytest tests/ -q` → **15 passed**.
* `fetch_html_ex`: google/github `ok=True`, amazon `ok=False, reason=anti-bot`.

---

## 2026‑10‑14 — Presentation demo added

### Added
* `demo.py` — a self-contained demonstration with two modes:
  * **terminal tour** (`python demo.py`): loads the trained model, prints the
    architecture, runs a live evaluation on held-out PhreshPhish pages, then
    walks through correctly-classified phishing/benign cases showing verdict,
    attention weights, top SHAP features and the explanation.
  * **web UI** (`python demo.py --web`): Flask page to paste a URL (live fetch
    with URL-only fallback) or click a bundled real example; renders verdict
    badge, probability gauge, per-modality attention bars, SHAP feature bars and
    the explanation. Optional `--llm` adds the LLM rationale.
* One new test (`evaluate_records`) — suite now **15 tests**.

### Changed
* `pipeline.py` — added `evaluate_records()` (batch score labeled records with a
  loaded model) and a `verbose` flag on `_raw_features()`.
* `explainer.py` — passes `silent=True` to SHAP so no `tqdm` bars leak into demo
  output.

### Verification
* `python demo.py --fast` → live eval (300 held-out: acc 0.923, AUC 0.980) and
  correct case-by-case explanations.
* `python demo.py --web --port 5055` → `GET /` renders form + 4 example buttons;
  `POST /predict` returns the result card (verified via curl).
* `pytest tests/ -q` → **15 passed**.

---

## 2026‑10‑14 — Real dataset integration, LLM run & experiments

### Dataset selection

Surveyed public phishing corpora. The base paper's own corpus is on Kaggle
(`aljofey/phishing-data`, needs credentials); most public datasets are
**URL-only** or precomputed-feature tables. Chose **PhreshPhish**
(`phreshphish/phreshphish`, HuggingFace) because it ships raw **URL + rendered
HTML + label** at scale (372k pages), which is exactly what Stages I/II/III need
and avoids live crawling. Verified availability with HTTP HEAD; shards are
~50–350 MB Parquet.

### Added files

| File | Purpose |
|------|---------|
| `datasets.py` | Streams PhreshPhish Parquet shards with PyArrow (bounded sampling, 120 KB HTML truncation, class balancing); `build_csv()` writes ready-to-train CSVs. |
| `encode_cache.py` | Builds a balanced subset and pre-computes LLM semantic features once, caching to `data/llm_cache.json`. |
| `summarize_results.py` | Aggregates `artifacts/*ablation*.json` into `artifacts/RESULTS.md`. |

### Modified files

* `config.py` — added `HTML_MAX_CHARS = 120_000` to cap per-page memory/parse cost.
* `pipeline.py`
  * `fit(..., test_records=None)` now honors an **external/official train-test
    split** (text vectorizers still fit on train only → no leakage).
  * Added `_compose_from()` helper; refactored `_compose()` onto it.
  * `PhishingPipeline.__init__` accepts `llm_model` / `llm_host` / `llm_timeout`.
  * CLI: new **`prepare`** subcommand; `train` gained `--test-data` and
    `--llm-model`.
  * Feature names for char features are now **human-readable** (e.g.
    `url_tfidf['ob']`) via new `feature_names()` on the text vectorizers.
* `text_features.py` — `CharTfidf.feature_names()` exposes the vocabulary;
  hashing variant returns `None`.
* `run_experiments.py` — accepts `--test-data`, `--cache`, `--llm-model`,
  `--out`; writes config metadata (dataset, split sizes) into the JSON.
* `explainer.py` — SHAP path tuned for high-dimensional inputs (50 background
  rows, `nsamples=128`) and its numerical warnings suppressed; SHAP is now the
  default attributor when installed.

### Environment

* `shap==0.53.0` installed → Stage VI now uses real **KernelSHAP** (fallback to
  occlusion preserved).
* Confirmed local **Ollama** with `qwen2.5-coder:7b` (and `qwen2.5:3b`); the LLM
  branch runs for real.

### Runs performed

* `pipeline.py prepare` → `data/phreshphish_train.csv` (2,624, balanced) and
  `data/phreshphish_test.csv` (906), using official PhreshPhish shards.
* Non-LLM ablation, attention + XGBoost head, official split →
  `artifacts/real_attention_ablation.json` (**tri-modal 0.916 acc / 0.915 F1 /
  0.972 AUC / 0.068 FPR**, beating URL+HTML 0.898/0.896/0.971/0.082).
* Non-LLM ablation, concat + XGBoost → `artifacts/real_concat_ablation.json`.
* Tri-modal ablation on a 100/60 cached subset (`qwen2.5-coder:7b`, ~10 s/sample,
  cached) → `real_concat_llm_ablation.json`, `real_attention_llm_ablation.json`.
  LLM-only AUC rises from 0.50 (chance) to **0.70**; tri-modal ROC-AUC 0.941.
* Trained + saved the deployable real model to `artifacts/phish_model`
  (attention+XGB, acc 0.912, AUC 0.974).
* `python summarize_results.py` → `artifacts/RESULTS.md`.

### Verification

* `pytest tests/ -q` → **14 passed**.
* Live prediction on held-out real pages returns correct verdicts plus SHAP
  attributions over human-readable features and, when available, an LLM rationale.

### Notes / limitations

* The LLM subset (100/60) is small because 7B inference is ~10 s/sample on CPU;
  the large-scale ablation therefore runs **without** LLM features. Scaling the
  LLM-annotated evaluation is the top follow-up.

---

## 2026‑10‑14 — Full implementation pass (Review‑2 completion)

### Context / starting point

Before this pass the `Code/` folder contained four partial files plus an empty
README:

| Old file | State |
|----------|-------|
| `url_feature_extractor.py` | hand‑crafted URL features; brand distance used the wrong host label |
| `html_feature_extractor.py` | DOM features; miscounted `frameborder=0` iframes as invisible |
| `llm_semantic_encoder.py` | single Ollama call, zero‑on‑failure |
| `fusion_classifier.py` | "attention" was a **fixed random projection**, never trained |
| `readme.md` | empty |
| `feature_extractor_checklist.md` | feature wish‑list (mostly unimplemented) |

There was no dataset path, no training script, no explanation module, no tests,
and no way to reproduce the base paper. Goal: implement the entire proposal.

### Added files

| File | Purpose |
|------|---------|
| `config.py` | **Single source of truth** for constants and the *canonical feature schemas* (`URL_NUMERIC_KEYS`, `HTML_NUMERIC_KEYS`, `SEMANTIC_SCORE_KEYS`, text/fusion/training hyper‑parameters). Also pins OpenMP env vars (see the macOS fix below). |
| `text_features.py` | Char‑level text representations — `CharTfidf` (sklearn, base‑paper TF‑IDF) and `HashingCharEmbedding` (vocabulary‑free). `TextFeatureBuilder` keeps separate URL and page‑text vectorizers. |
| `attention.py` | **Real trainable fusion** (torch): `GatedAttentionFusion` (learned per‑sample softmax modality weights) and `FusionNet` (attention + MLP head, end‑to‑end). Includes modality dropout. |
| `data_loader.py` | CSV/JSONL loading, `fetch_html` with timeout/size cap + graceful `(html, ok)` fallback, and a reproducible `synthesize_dataset`. |
| `explainer.py` | Stage VI: `FeatureAttributor` (SHAP `KernelExplainer` when available, else dependency‑free local occlusion) and `ExplanationGenerator` (LLM narrative with deterministic template fallback). |
| `pipeline.py` | End‑to‑end orchestration of Stages I–VI + CLI (`demo` / `train` / `predict`) + artifact persistence. |
| `run_experiments.py` | Ablation harness (URL‑only … tri‑modal) with accuracy/F1/AUC/FPR, saving `artifacts/ablation.json`. |
| `api.py` | FastAPI service (`/health`, `/predict`) with a Flask fallback. |
| `requirements.txt` | Dependencies, with optional packages marked and their fallbacks documented. |
| `tests/test_extractors.py` | 7 tests for URL/HTML extraction. |
| `tests/test_attention.py` | 3 tests: softmax weights sum to 1, gradients flow, classifier shapes. |
| `tests/test_pipeline.py` | 4 tests: dataset, end‑to‑end fit, prediction+explanation, ablation. |
| `readme.md` | Rewritten: architecture, layout, quickstart, schemas, status, pending work. |
| `history.md` | This changelog. |

### Modified files

#### `url_feature_extractor.py`
* Reworked around `config.URL_NUMERIC_KEYS` via new `url_feature_vector()` → **stable, canonical column order** between train and inference.
* Added `split_host()` / `registered_domain()` using `tldextract` (offline suffix list) with a manual fallback.
* **Fixed `min_brand_edit_distance`**: now compares the *registrable* domain (e.g. `paypa1`) against brands, instead of an arbitrary label such as `verify`. Added more brands.
* **Fixed `has_ip_address`**: rejects malformed octets (`999.1.1.1`).
* **Fixed `is_shortened`**: exact/`endswith(".shortener")` match instead of substring.
* **Fixed `has_port`**: `_safe_port()` no longer raises on ambiguous ports.
* Added more suspicious words/TLDs; `ends_with_known_ext` gained `.scr/.apk`.
* Kept the pure‑python edit‑distance fallback (used when `python-Levenshtein` absent).

#### `html_feature_extractor.py`
* **Fixed iframe invisibility**: `frameborder="0"` is no longer treated as invisible; a separate `frameborder_zero_count` feature was added, and invisibility is judged from `display:none`, `visibility:hidden`, `hidden`, or zero width/height.
* **Fixed external/internal classification**: compares *registered* domains, so `www.paypal.com` from `paypal.com` is not flagged external. Applied to links, forms, scripts, CSS, media, favicon.
* **Fixed `_copyright_mismatch`**: compares against the registered domain and also recognises `(c)`.
* Added `html_feature_vector()` for canonical ordering; returned dict now covers every key in `HTML_NUMERIC_KEYS`.
* Normalised `mail_submission` and right‑click/status‑bar detection.

#### `llm_semantic_encoder.py`
* Split out a reusable `LLMClient` (`chat`, `is_available`, `parse_json`) shared with the explainer.
* Added `encode_batch(..., logger)` and an `is_available()` health probe.
* Kept and hardened the zero‑vector fallback so a missing LLM never breaks the pipeline.
* Added `EXPLAIN_SYSTEM_PROMPT` for Stage VI.

#### `fusion_classifier.py`
* **Replaced the fake random‑projection attention** with the trainable `FusionNet` from `attention.py`; supports `fusion_mode ∈ {concat, attention}` and `head ∈ {xgb, mlp}` (attention+XGB embeds the learned representation then fits XGBoost).
* Full training loop: Adam, `BCEWithLogitsLoss` with class‑imbalance `pos_weight`, mini‑batching, best‑val‑AUC checkpointing, modality dropout.
* Per‑modality `StandardScaler`s; added `predict_concat()` (uniform black‑box interface for the explainer) and `modality_weights()`.
* Richer `evaluate()`: accuracy, precision, recall, F1, ROC‑AUC, **FPR**, confusion‑matrix counts.
* Added `save()` / `load()` (joblib + torch `state_dict`).
* Import order fixed so `config` (OpenMP env) loads before xgboost/torch.

#### `pipeline.py` (new file; also fixed during development)
* Composes Stage I/II/III vectors, fits text vectorizers on the **training split only** (no leakage), trains/evals the fusion model, builds feature names, and constructs the attributor.
* `predict()` performs a full single‑URL pass: fetch (with cloaking/JS fallback), extract, optional LLM, scale, classify, attention weights, top features, and generated explanation.
* **Bug fixed during build**: page‑text n‑grams were briefly placed in the URL block; corrected so `[X_url | X_html | X_llm]` matches the feature‑name order exactly.
* **Bug fixed during build**: saved models could not explain until `background` was persisted; `save`/`load` now carry a background sample and rebuild the attributor.
* Guarded against storing the explainer with a stale attributor.

### Bugs fixed (summary)

| # | Where | Bug | Fix |
|---|-------|-----|-----|
| 1 | `url_feature_extractor` | brand edit distance used `host.split('.')[-2]` (wrong label) | registrable domain via tldextract |
| 2 | `html_feature_extractor` | `frameborder=0` counted as invisible iframe | explicit invisibility heuristics |
| 3 | `html_feature_extractor` | `_copyright_mismatch` wrong label | registrable domain |
| 4 | `url_feature_extractor` | substring shortener match → false positives | exact/suffix match |
| 5 | `url_feature_extractor` | `has_ip` accepted `999.1.1.1` | octet range check |
| 6 | `url_feature_extractor` | `parsed.port` could raise | `_safe_port` |
| 7 | `fusion_classifier` | "attention" was random and untrained | learned `FusionNet` |
| 8 | `fusion_classifier` | non‑numeric keys silently dropped / order‑dependent vectors | canonical `*_KEYS` + `*_feature_vector` |
| 9 | `pipeline` | page‑text n‑grams in URL block | separated URL vs text vectorizers |
| 10 | `pipeline` | explanation missing after `load` | persist background, rebuild attributor |

### Environment fix (important)

On macOS, importing **PyTorch and XGBoost into the same process segfaults**
(duplicate OpenMP runtimes; reproduced: `import torch; xgboost.fit()` → exit 139).
Mitigation, verified to resolve it:

* `config.py` now does `os.environ.setdefault("OMP_NUM_THREADS", "1")` and
  `KMP_DUPLICATE_LIB_OK=TRUE` before any heavy import;
* `config` is imported before `xgboost`/`torch` in every module;
* `xgboost` is imported before `torch` in `fusion_classifier.py`.

Confirmed working variants: pinning `OMP_NUM_THREADS=1`, or importing xgboost
before torch. `KMP_DUPLICATE_LIB_OK=TRUE` alone was **not** sufficient.

### Verification

* `python -m pytest tests/ -q` → **14 passed**.
* `python pipeline.py demo --n 400` → trains, test metrics reported (synthetic
  data is separable → ~1.00 accuracy).
* `python pipeline.py demo --save` + `predict` → returns label, probability,
  **attention modality weights**, top features and an explanation.
* `python run_experiments.py --n 500` → full ablation table written to
  `artifacts/ablation.json`.
* `python fusion_classifier.py` → concat/xgb ≈ 0.96 acc, attention/mlp ≈ 0.90,
  attention/xgb ≈ 0.83 on the random demo split.
* `python api.py` → builds the FastAPI app and classifies a sample URL.
* Every module's `__main__` smoke test passes.

### Not yet done (carried forward)

* Run on a real dataset (PhishTank/APWG + benign corpus) and fill the paper's
  Table 3 with measured numbers.
* Paper reference fixes: correct reference **[3]** to **Nasution, Monika, Onan &
  Murakami (MDPI *Information* 2025)** (the paper currently says "Setiawan et al.");
  add authors to the reference entry.
* Add XAI/SHAP and dataset‑provenance references to the literature survey.
* Optional: replace char TF‑IDF with a small CharCNN embedding.
* The top‑level `Code.zip` and `Zips/Extractors.zip` are now stale duplicates of
  the (much older) source; regenerate or remove them.
