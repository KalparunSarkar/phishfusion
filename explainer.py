"""Explainability: local feature attributions + LLM natural-language rationale.

Attribution strategy
--------------------
The fusion model is treated as a black box over the *concatenated* feature vector
``[url || html || llm]`` via :meth:`FusionClassifier.predict_concat`.  This makes
attributions comparable across both fusion modes and keeps them expressed in terms
of the original, human-meaningful feature names.

* If ``shap`` is installed, a ``KernelExplainer`` is used (model-agnostic).
* Otherwise a fast, dependency-free *local occlusion* attribution is computed:
  each feature is reset to its training-population baseline and the resulting
  drop in the phishing probability becomes that feature's contribution.

The :class:`ExplanationGenerator` then feeds the top attributions to the LLM to
produce a plain-language rationale (Stage VI of the proposal), with a deterministic
template fallback when the LLM is unavailable.
"""

import warnings

import numpy as np

from llm_semantic_encoder import EXPLAIN_SYSTEM_PROMPT, LLMClient

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def local_occlusion_attributions(predict_fn, x, baseline, batch=True):
    """One-at-a-time occlusion attributions for a single instance.

    ``contribution_j = p(x) - p(x with feature j set to baseline)``.  A positive
    value means feature ``j`` increased the phishing probability.
    """
    x = np.asarray(x, dtype=np.float32).ravel()
    baseline = np.asarray(baseline, dtype=np.float32).ravel()
    p0 = float(predict_fn(x.reshape(1, -1))[0])
    d = x.shape[0]
    if batch:
        perturbed = np.tile(x, (d, 1))
        perturbed[np.arange(d), np.arange(d)] = baseline
        probs = predict_fn(perturbed)
        contrib = p0 - probs
    else:
        contrib = np.array([p0 - float(predict_fn(np.where(np.arange(d) == j, baseline, x).reshape(1, -1))[0])
                            for j in range(d)])
    return p0, contrib


class FeatureAttributor:
    def __init__(self, model, feature_names, background, method="auto"):
        self.model = model
        self.feature_names = list(feature_names)
        self.background = np.asarray(background, dtype=np.float32)
        self.baseline = self.background.mean(axis=0)
        if method == "auto":
            method = "shap" if HAS_SHAP else "occlusion"
        self.method = method
        self._explainer = None

    def _predict(self, X):
        return self.model.predict_concat(np.atleast_2d(X))

    def _ensure_shap(self):
        if self._explainer is None:
            bg = self.background
            if len(bg) > 50:
                idx = np.random.default_rng(0).choice(len(bg), 50, replace=False)
                bg = bg[idx]
            self._explainer = shap.KernelExplainer(self._predict, bg)

    def attribute(self, x):
        if self.method == "shap" and HAS_SHAP:
            self._ensure_shap()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                values = self._explainer.shap_values(
                    np.asarray(x, dtype=np.float32).reshape(1, -1), nsamples=128,
                    silent=True)
            values = np.asarray(values).ravel()
            p0 = float(self._predict(x.reshape(1, -1))[0])
            return p0, values
        return local_occlusion_attributions(self._predict, x, self.baseline)

    def top_features(self, x, k=8):
        p0, contrib = self.attribute(x)
        order = np.argsort(np.abs(contrib))[::-1][:k]
        pairs = [(self.feature_names[i], float(contrib[i])) for i in order]
        return p0, pairs


class ExplanationGenerator:
    def __init__(self, client=None):
        self.client = client or LLMClient()

    def _template(self, verdict, proba, attributions):
        toward = [name for name, c in attributions if c > 0.01]
        against = [name for name, c in attributions if c < -0.01]
        label = "phishing" if verdict == 1 else "legitimate"
        conf = proba if verdict == 1 else 1 - proba
        parts = [f"This page was classified as {label} with {conf:.0%} confidence."]
        if toward:
            parts.append("Signals that increased the phishing score: "
                         + ", ".join(toward[:5]) + ".")
        if against:
            parts.append("Signals that argued for legitimacy: "
                         + ", ".join(against[:5]) + ".")
        if not toward and not against:
            parts.append("No single feature dominated the decision.")
        return " ".join(parts)

    def generate(self, url, verdict, proba, attributions, page_text="", use_llm=True):
        attrib_txt = "\n".join(
            f"- {name}: {contrib:+.4f}" for name, contrib in attributions
        ) or "- (no dominant features)"
        verdict_str = "PHISHING" if verdict == 1 else "LEGITIMATE"
        user = (
            f"URL: {url}\n"
            f"Verdict: {verdict_str} (phishing probability = {proba:.3f})\n\n"
            f"Top model feature attributions (positive = pushes toward phishing):\n"
            f"{attrib_txt}\n\n"
            f"Excerpt of page text:\n{(page_text or '')[:500]}\n"
        )
        if use_llm:
            text = self.client.chat(EXPLAIN_SYSTEM_PROMPT, user, want_json=False)
            if text and text.strip():
                return text.strip()
        return self._template(verdict, proba, attributions)


if __name__ == "__main__":
    from fusion_classifier import FusionClassifier

    rng = np.random.default_rng(0)
    n = 300
    Xu, Xh, Xl = (rng.normal(size=(n, 6)), rng.normal(size=(n, 5)), rng.normal(size=(n, 3)))
    y = (Xu[:, 0] - Xh[:, 1] + Xl[:, 0] > 0).astype(int)
    names = [f"url_{i}" for i in range(6)] + [f"html_{i}" for i in range(5)] + \
            [f"llm_{i}" for i in range(3)]
    model = FusionClassifier(fusion_mode="concat", head="xgb")
    model.fit(Xu, Xh, Xl, y)
    Xc = np.hstack([Xu, Xh, Xl])
    attrib = FeatureAttributor(model, names, Xc[:100])
    print("method:", attrib.method)
    p0, top = attrib.top_features(Xc[0])
    print("p0:", round(p0, 3), "top:", [(n_, round(c, 3)) for n_, c in top])
    print(ExplanationGenerator().generate("http://x.y", int(p0 >= .5), p0, top, use_llm=False))
