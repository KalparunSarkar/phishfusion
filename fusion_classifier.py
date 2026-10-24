"""Fusion classifier: combines URL, HTML and LLM feature blocks.

Two fusion strategies, both selectable:

* ``concat``    - standardise each block and concatenate (base-paper baseline),
                  classified by XGBoost or an sklearn MLP.
* ``attention`` - a *learned* gated-attention fusion network (see ``attention.py``)
                  with either an end-to-end MLP head or an XGBoost head fitted on
                  the learned fused embeddings.

The class exposes ``predict_concat`` so the explanation module can attribute a
prediction to individual features using a single, uniform interface.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from config import (  # imported first: pins OpenMP before xgboost/torch load
    RANDOM_SEED,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_LR,
)

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False

try:
    import torch
    import torch.nn as nn

    from attention import FusionNet

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


def _make_xgb(seed):
    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=seed,
    )


class FusionClassifier:
    def __init__(self, fusion_mode="concat", head="xgb", seed=RANDOM_SEED,
                 epochs=TRAIN_EPOCHS, batch_size=TRAIN_BATCH_SIZE, lr=TRAIN_LR):
        assert fusion_mode in ("concat", "attention")
        assert head in ("xgb", "mlp")
        if fusion_mode == "attention" and not HAS_TORCH:
            raise RuntimeError("attention fusion requires torch")
        self.fusion_mode = fusion_mode
        self.head = head
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        self.scaler_url = StandardScaler()
        self.scaler_html = StandardScaler()
        self.scaler_llm = StandardScaler()
        self.dims = None
        self.net = None
        self.head_model = None

    # ------------------------------------------------------------------ utils
    def _scale(self, X_url, X_html, X_llm, fit=False):
        if fit:
            X_url = self.scaler_url.fit_transform(X_url)
            X_html = self.scaler_html.fit_transform(X_html)
            X_llm = self.scaler_llm.fit_transform(X_llm)
        else:
            X_url = self.scaler_url.transform(X_url)
            X_html = self.scaler_html.transform(X_html)
            X_llm = self.scaler_llm.transform(X_llm)
        return (np.asarray(X_url, dtype=np.float32),
                np.asarray(X_html, dtype=np.float32),
                np.asarray(X_llm, dtype=np.float32))

    def _fit_head(self, Z, y, sample_weight=None):
        if self.head == "xgb" and HAS_XGB:
            self.head_model = _make_xgb(self.seed)
            self.head_model.fit(Z, y)
        elif self.head == "mlp":
            self.head_model = MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=600, random_state=self.seed,
            )
            self.head_model.fit(Z, y)
        else:  # head == "xgb" but xgboost missing
            self.head_model = LogisticRegression(max_iter=2000)
            self.head_model.fit(Z, y)

    # ------------------------------------------------------------------- train
    def fit(self, X_url, X_html, X_llm, y, X_val=None, y_val=None):
        y = np.asarray(y).astype(int)
        self.dims = (X_url.shape[1], X_html.shape[1], X_llm.shape[1])
        Xu, Xh, Xl = self._scale(X_url, X_html, X_llm, fit=True)

        if self.fusion_mode == "concat":
            Z = np.hstack([Xu, Xh, Xl])
            self._fit_head(Z, y)
            return self

        # attention mode -----------------------------------------------------
        Xu_t = torch.tensor(Xu)
        Xh_t = torch.tensor(Xh)
        Xl_t = torch.tensor(Xl)
        y_t = torch.tensor(y.astype(np.float32))

        self.net = FusionNet(list(self.dims))
        optim = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        pos = max((y == 1).sum(), 1)
        neg = max((y == 0).sum(), 1)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, dtype=torch.float32))

        n = len(y)
        best_state, best_auc = None, -1.0
        val_tensors = None
        if X_val is not None:
            Xu_v, Xh_v, Xl_v = self._scale(X_val[0], X_val[1], X_val[2])
            val_tensors = (torch.tensor(Xu_v), torch.tensor(Xh_v), torch.tensor(Xl_v),
                           np.asarray(y_val).astype(int))

        for epoch in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                optim.zero_grad()
                logits = self.net([Xu_t[idx], Xh_t[idx], Xl_t[idx]])
                loss = loss_fn(logits, y_t[idx])
                loss.backward()
                optim.step()

            if val_tensors is not None:
                with torch.no_grad():
                    self.net.eval()
                    logits, _ = self.net(list(val_tensors[:3]), return_weights=True)
                    probs = torch.sigmoid(logits).numpy()
                try:
                    auc = roc_auc_score(val_tensors[3], probs)
                except ValueError:
                    auc = 0.0
                if auc > best_auc:
                    best_auc = auc
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
        if best_state is not None:
            self.net.load_state_dict(best_state)

        if self.head == "xgb" and HAS_XGB:
            Z = self._embed(Xu_t, Xh_t, Xl_t)
            self._fit_head(Z, y)
        return self

    def _embed(self, Xu_t, Xh_t, Xl_t):
        with torch.no_grad():
            self.net.eval()
            return self.net.embed([Xu_t, Xh_t, Xl_t]).numpy()

    # ---------------------------------------------------------------- predict
    def predict_proba(self, X_url, X_html, X_llm):
        Xu, Xh, Xl = self._scale(X_url, X_html, X_llm)
        if self.fusion_mode == "concat":
            Z = np.hstack([Xu, Xh, Xl])
            return self.head_model.predict_proba(Z)[:, 1]

        Xu_t, Xh_t, Xl_t = (torch.tensor(Xu), torch.tensor(Xh), torch.tensor(Xl))
        if self.head == "xgb" and HAS_XGB:
            Z = self._embed(Xu_t, Xh_t, Xl_t)
            return self.head_model.predict_proba(Z)[:, 1]
        with torch.no_grad():
            self.net.eval()
            logits = self.net([Xu_t, Xh_t, Xl_t])
            return torch.sigmoid(logits).numpy()

    def predict_concat(self, X_concat):
        """Predict from a single concatenated matrix (url||html||llm)."""
        du, dh, dl = self.dims
        X_concat = np.asarray(X_concat, dtype=np.float32)
        return self.predict_proba(
            X_concat[:, :du], X_concat[:, du:du + dh], X_concat[:, du + dh:du + dh + dl]
        )

    def modality_weights(self, X_url, X_html, X_llm):
        """Return learned per-sample modality weights (attention mode only)."""
        if self.fusion_mode != "attention" or not HAS_TORCH:
            return None
        Xu, Xh, Xl = self._scale(X_url, X_html, X_llm)
        with torch.no_grad():
            self.net.eval()
            _, w = self.net([torch.tensor(Xu), torch.tensor(Xh), torch.tensor(Xl)],
                            return_weights=True)
            return w.numpy()

    # --------------------------------------------------------------- evaluate
    def evaluate(self, X_url, X_html, X_llm, y_true, threshold=0.5):
        probs = self.predict_proba(X_url, X_html, X_llm)
        preds = (probs >= threshold).astype(int)
        y_true = np.asarray(y_true).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        try:
            auc = roc_auc_score(y_true, probs)
        except ValueError:
            auc = float("nan")
        return {
            "accuracy": accuracy_score(y_true, preds),
            "precision": precision_score(y_true, preds, zero_division=0),
            "recall": recall_score(y_true, preds, zero_division=0),
            "f1": f1_score(y_true, preds, zero_division=0),
            "roc_auc": auc,
            "fpr": fpr,
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "n": int(len(y_true)),
        }

    # ------------------------------------------------------------- persistence
    def save(self, path):
        import joblib

        payload = {
            "fusion_mode": self.fusion_mode,
            "head": self.head,
            "seed": self.seed,
            "dims": self.dims,
            "scalers": (self.scaler_url, self.scaler_html, self.scaler_llm),
            "head_model": self.head_model,
        }
        joblib.dump(payload, path)
        if self.net is not None and HAS_TORCH:
            torch.save(self.net.state_dict(), str(path) + ".net.pt")

    @classmethod
    def load(cls, path):
        import joblib

        payload = joblib.load(path)
        obj = cls(fusion_mode=payload["fusion_mode"], head=payload["head"], seed=payload["seed"])
        obj.dims = payload["dims"]
        obj.scaler_url, obj.scaler_html, obj.scaler_llm = payload["scalers"]
        obj.head_model = payload["head_model"]
        if obj.fusion_mode == "attention" and HAS_TORCH:
            obj.net = FusionNet(list(obj.dims))
            obj.net.load_state_dict(torch.load(str(path) + ".net.pt"))
        return obj


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 600
    X_url = rng.normal(size=(n, 24))
    X_html = rng.normal(size=(n, 20))
    X_llm = rng.normal(size=(n, 6))
    y = (X_url[:, 0] + X_llm[:, 2] > 0.3).astype(int)
    split = int(n * 0.8)
    for mode, head in (("concat", "xgb"), ("attention", "mlp"), ("attention", "xgb")):
        model = FusionClassifier(fusion_mode=mode, head=head, epochs=40)
        model.fit(X_url[:split], X_html[:split], X_llm[:split], y[:split],
                  (X_url[split:], X_html[split:], X_llm[split:]), y[split:])
        metrics = model.evaluate(X_url[split:], X_html[split:], X_llm[split:], y[split:])
        print(f"{mode:9} + {head:3} -> acc={metrics['accuracy']:.3f} "
              f"f1={metrics['f1']:.3f} auc={metrics['roc_auc']:.3f}")
