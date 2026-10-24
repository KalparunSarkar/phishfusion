"""Minimal REST API + CLI wrapper exposing the trained detector.

Endpoints
---------
``GET /health``                 -> service/diagnostic info
``GET /predict?url=...&llm=``   -> {label, phishing_probability, explanation, ...}

Serves the FastAPI app when FastAPI is available, otherwise falls back to Flask.
Run with::

    python api.py                 # auto-trains a demo model if none is saved
    uvicorn api:app --reload      # if FastAPI/uvicorn are installed
"""

import json

from config import ARTIFACT_DIR
from pipeline import PhishingPipeline

_PIPELINE = None


def get_pipeline():
    """Load saved artifacts, or train a quick demo model if none exist."""
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE
    if (ARTIFACT_DIR / "phish_model.meta").exists():
        _PIPELINE = PhishingPipeline.load("phish_model")
    else:
        from data_loader import synthesize_dataset

        pipe = PhishingPipeline()
        pipe.fit(synthesize_dataset(n=300), mode="attention", head="xgb", use_llm=False)
        _PIPELINE = pipe
    return _PIPELINE


def predict(url, llm=False):
    return get_pipeline().predict(url, use_llm=llm)


# --------------------------------------------------------------------- FastAPI
try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI(title="Phishing Detection API", version="1.0")

    @app.get("/health")
    def health():
        pipe = get_pipeline()
        return {
            "status": "ok",
            "fusion_mode": pipe.model.fusion_mode,
            "head": pipe.model.head,
            "llm_available": pipe.llm_encoder.client.is_available(),
        }

    @app.get("/predict")
    def predict_endpoint(url: str, llm: bool = False):
        return JSONResponse(predict(url, llm))

    HAS_FASTAPI = True
except Exception:  # pragma: no cover
    HAS_FASTAPI = False
    app = None


# ----------------------------------------------------------------------- Flask
def build_flask_app():
    from flask import Flask, jsonify, request

    flask_app = Flask("phishing_api")

    @flask_app.route("/health")
    def _health():
        pipe = get_pipeline()
        return jsonify({"status": "ok", "fusion_mode": pipe.model.fusion_mode,
                        "head": pipe.model.head})

    @flask_app.route("/predict")
    def _predict():
        url = request.args.get("url", "")
        llm = request.args.get("llm", "false").lower() in ("1", "true", "yes")
        if not url:
            return jsonify({"error": "missing url parameter"}), 400
        return jsonify(predict(url, llm))

    return flask_app


def main():
    try:
        import uvicorn

        uvicorn.run(app, host="127.0.0.1", port=8000)
    except ImportError:
        build_flask_app().run(host="127.0.0.1", port=8000)


if __name__ == "__main__":
    if HAS_FASTAPI and app is not None:
        print("FastAPI app ready. Run: uvicorn api:app --reload")
        print("Example:", 'curl "http://127.0.0.1:8000/predict?url=https://example.com"')
        print(json.dumps(get_pipeline().predict("https://www.google.com/", use_llm=False), indent=2))
    else:
        main()
