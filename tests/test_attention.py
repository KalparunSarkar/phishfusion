import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention import FusionNet, GatedAttentionFusion  # noqa: E402
from fusion_classifier import FusionClassifier  # noqa: E402

import torch  # noqa: E402


def test_gated_attention_weights_sum_to_one():
    torch.manual_seed(0)
    dims = [10, 8, 5]
    xs = [torch.randn(16, d) for d in dims]
    fusion = GatedAttentionFusion(dims)
    fused, weights = fusion(xs, return_weights=True)
    assert fused.shape == (16, 64)
    assert weights.shape == (16, 3)
    sums = weights.sum(dim=1).detach().numpy()
    assert np.allclose(sums, 1.0, atol=1e-5)


def test_fusion_net_is_trainable():
    torch.manual_seed(0)
    dims = [6, 5, 3]
    net = FusionNet(dims)
    xs = [torch.randn(32, d) for d in dims]
    logits = net(xs)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.randint(0, 2, (32,)).float()
    )
    loss.backward()
    # at least one gradient must be non-zero -> the gate learns something
    grads = [p.grad.abs().sum().item() for p in net.parameters() if p.grad is not None]
    assert any(g > 0 for g in grads)


def test_classifier_attention_and_concat_shapes():
    rng = np.random.default_rng(0)
    n = 200
    Xu, Xh, Xl = (rng.normal(size=(n, 8)), rng.normal(size=(n, 6)), rng.normal(size=(n, 4)))
    y = (Xu[:, 0] + Xl[:, 0] > 0).astype(int)
    split = 150
    for mode, head in (("concat", "xgb"), ("attention", "mlp")):
        model = FusionClassifier(fusion_mode=mode, head=head, epochs=20)
        model.fit(Xu[:split], Xh[:split], Xl[:split], y[:split])
        probs = model.predict_proba(Xu[split:], Xh[split:], Xl[split:])
        assert probs.shape == (n - split,)
        assert np.all((probs >= 0) & (probs <= 1))
