"""LLM semantic encoder and shared Ollama chat client.

The encoder turns (URL, visible page text) into six semantic signals that the
fusion layer consumes.  A second entry point, :meth:`LLMClient.chat`, is reused
by the explanation module to translate SHAP attributions into natural language.

The client degrades gracefully: if Ollama is not running, :meth:`encode` returns
an all-zero semantic vector and :meth:`chat` returns ``None`` rather than raising,
so the rest of the pipeline keeps working.
"""

import json
import re

import requests

from config import (
    LLM_MAX_TEXT_CHARS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    OLLAMA_HOST,
    SEMANTIC_SCORE_KEYS,
)

SYSTEM_PROMPT = """You are a phishing-detection semantic analyzer.
Given a URL and the visible text extracted from its page, score the
following six signals from 0.0 (absent) to 1.0 (strongly present):
urgency_language, brand_impersonation, credential_harvesting,
generic_greeting, grammar_anomaly_score, mismatched_domain_claim.

Respond with ONLY a JSON object, no preamble, no markdown fences, in
exactly this shape:
{"urgency_language": 0.0, "brand_impersonation": 0.0,
 "credential_harvesting": 0.0, "generic_greeting": 0.0,
 "grammar_anomaly_score": 0.0, "mismatched_domain_claim": 0.0,
 "rationale": "one short sentence explaining the scores"}
"""

EXPLAIN_SYSTEM_PROMPT = """You are a security analyst assistant.
You are given a phishing verdict produced by a machine-learning model and the
model's top feature attributions (SHAP values). Explain, in 2-4 plain-language
sentences, WHY the page was classified as it was. Reference the specific signals
(e.g. mismatched domain, urgency language, credential-harvesting form) and avoid
jargon. Do not contradict the verdict. Respond with only the explanation text.
"""


class LLMClient:
    def __init__(self, model=LLM_MODEL, host=OLLAMA_HOST, timeout=LLM_TIMEOUT,
                 temperature=LLM_TEMPERATURE):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature

    def chat(self, system, user, want_json=False):
        """Low-level chat call. Returns response text or ``None`` on failure."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if want_json:
            payload["format"] = "json"
        try:
            resp = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "")
        except Exception:
            return None

    def is_available(self):
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    @staticmethod
    def parse_json(text):
        text = (text or "").strip()
        text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise


class LLMSemanticEncoder:
    def __init__(self, model=LLM_MODEL, host=OLLAMA_HOST, max_text_chars=LLM_MAX_TEXT_CHARS,
                 timeout=LLM_TIMEOUT):
        self.max_text_chars = max_text_chars
        self.client = LLMClient(model=model, host=host, timeout=timeout)

    def _call_llm(self, url, page_text):
        page_text = (page_text or "")[: self.max_text_chars]
        user_content = f"URL: {url}\n\nVisible page text:\n{page_text}"
        text = self.client.chat(SYSTEM_PROMPT, user_content, want_json=True)
        if text is None:
            raise RuntimeError("LLM unavailable")
        return self.client.parse_json(text)

    def encode(self, url, page_text):
        """Return ``(f_llm, rationale)``; zeros + reason on failure."""
        try:
            parsed = self._call_llm(url, page_text)
        except Exception as exc:
            return [0.0] * len(SEMANTIC_SCORE_KEYS), f"LLM unavailable: {exc}"
        f_llm = [float(parsed.get(k, 0.0) or 0.0) for k in SEMANTIC_SCORE_KEYS]
        rationale = parsed.get("rationale", "")
        return f_llm, rationale

    def encode_batch(self, items, logger=None):
        """Encode many ``(url, page_text)`` pairs. Returns (vectors, rationales)."""
        vectors, rationales = [], []
        for i, (url, page_text) in enumerate(items):
            vec, why = self.encode(url, page_text)
            vectors.append(vec)
            rationales.append(why)
            if logger and (i + 1) % 25 == 0:
                logger(f"  LLM encoded {i + 1}/{len(items)}")
        return vectors, rationales


if __name__ == "__main__":
    sample_url = "http://secure-login.paypal.com.verify.xyz"
    sample_text = (
        "Dear user, your account has been limited. Verify your identity "
        "within 24 hours by confirming your password and card details, "
        "or your PayPal account will be permanently suspended."
    )
    encoder = LLMSemanticEncoder()
    print("LLM available:", encoder.client.is_available())
    vec, why = encoder.encode(sample_url, sample_text)
    print(dict(zip(SEMANTIC_SCORE_KEYS, vec)))
    print("rationale:", why)
