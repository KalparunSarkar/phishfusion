# PhishFusion

A hybrid **URL + HTML + LLM** framework for **explainable phishing detection**.

PhishFusion classifies a web page as **phishing** or **legitimate** by combining
three independent signal streams:

- **URL features** — lexical/statistical signals from the raw link
- **HTML features** — structural signals from the rendered page (links, forms, iframes, scripts …)
- **LLM features** — semantic cues such as urgency, brand impersonation, and credential harvesting

The three streams are fused with a **learned gated-attention layer**, classified
with XGBoost, and explained with **SHAP attributions** plus a natural-language
rationale.

---

## Why it matters

Traditional detectors use URLs alone or concatenate URL + HTML features. PhishFusion
adds an LLM semantic branch, learns **per-page attention weights** over each
modality, and produces a human-readable explanation for every verdict.

---

## Results

Measured on the **PhreshPhish** dataset (2,624 train / 906 test, official split):

| Configuration | Accuracy | F1 | ROC-AUC | FPR |
|---|---:|---:|---:|---:|
| URL only | 0.898 | 0.898 | 0.966 | 0.099 |
| HTML only | 0.762 | 0.736 | 0.869 | 0.141 |
| URL + HTML (baseline) | 0.898 | 0.896 | 0.971 | 0.082 |
| **URL + HTML + LLM (ours)** | **0.916** | **0.915** | **0.972** | **0.068** |

The tri-modal model beats the URL+HTML baseline on every metric while cutting the
false-positive rate from **8.2% → 6.8%**.

Full ablation tables: [`artifacts/RESULTS.md`](artifacts/RESULTS.md).

---

## Installation

Requires Python 3.10+.

```bash
python -m pip install -r requirements.txt
```

Optional components degrade gracefully if absent:

- `torch` — needed for attention fusion (falls back to concatenation)
- `xgboost` — falls back to logistic regression
- `shap` — falls back to local occlusion explanations
- A local [Ollama](https://ollama.com) server — supplies LLM semantic features
  and written rationales (the pipeline still works without it)

---

## How to run

### 1. Offline demo (no network, no LLM)

```bash
python pipeline.py demo --n 400
```

### 2. Train on your own CSV

Expected columns: `url,label,html` (`0` = benign, `1` = phishing; `html` optional).

```bash
python pipeline.py train --data data/mydata.csv --mode attention --head xgb --save
```

### 3. Classify a URL with an explanation

```bash
python pipeline.py predict --url "http://secure-login.paypal.com.verify.xyz/webscr?cmd=login"
```

### 4. Presentation demo

```bash
python demo.py          # guided terminal walkthrough on real held-out pages
python demo.py --web    # local web UI at http://127.0.0.1:5000
```

### 5. REST API

```bash
uvicorn api:app --reload
curl "http://127.0.0.1:8000/predict?url=https://example.com"
```

### 6. Tests

```bash
python -m pytest tests/ -q
```

---

## Notes

- Training data (the 1.3 GB PhreshPhish corpus) is **not** committed. Regenerate it with
  `python pipeline.py prepare`.
- The model is trained on PhreshPhish pages; some live JavaScript-heavy sites may be
  out-of-distribution. This is a generalisation limitation, not a fetching bug.
- See [`history.md`](history.md) for the full changelog and [`artifacts/RESULTS.md`](artifacts/RESULTS.md)
  for all experiment numbers.
