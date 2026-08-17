"""NumPy-only inference for the HepatoCryoAI neural network.

Loads ``models/nn_weights.npz`` (StandardScaler parameters + Linear-layer
weights, with any BatchNorm folded into the adjacent Linear layer at export
time) and runs a forward pass using nothing but NumPy. No torch, no sklearn.
"""
import os
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXPORT_PATH = os.path.join(_REPO_ROOT, "models", "nn_weights.npz")

_cache = None


def _load():
    global _cache
    if _cache is None:
        data = np.load(_EXPORT_PATH)
        n_layers = int(data["n_layers"])
        layers = [(data[f"W{i}"], data[f"b{i}"]) for i in range(n_layers)]
        _cache = {
            "mean": data["mean"],
            "scale": data["scale"],
            "layers": layers,
        }
    return _cache


def _forward(x, layers):
    a = x
    n = len(layers)
    for i, (W, b) in enumerate(layers):
        a = a @ W.T + b
        if i < n - 1:
            a = np.maximum(a, 0.0)
    return a


def predict(dmso, trehalose):
    """Predict post-thaw viability (%) for the given DMSO/trehalose concentrations (%).

    Parameters
    ----------
    dmso, trehalose : float
        Cryoprotectant concentrations in percent.

    Returns
    -------
    float
        Predicted cell viability (%).
    """
    state = _load()
    x = np.array([[float(dmso), float(trehalose)]], dtype=np.float64)
    x_scaled = (x - state["mean"]) / state["scale"]
    y = _forward(x_scaled, state["layers"])
    return float(y[0, 0])


if __name__ == "__main__":
    print(f"predict(5, 10) = {predict(5, 10):.4f}")
