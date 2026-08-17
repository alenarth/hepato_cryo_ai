"""NumPy-only inference for the HepatoCryoAI Random Forest.

Loads ``models/rf_trees.npz`` (200 decision trees flattened into concatenated
arrays) and reproduces scikit-learn's RandomForestRegressor.predict exactly
(bit-for-bit), using nothing but NumPy. No scikit-learn, no joblib.

Array layout
------------
Each tree's ``children_left``/``children_right`` entries are scikit-learn's
original LOCAL (per-tree) node indices, left untouched. ``offsets[k]`` marks
where tree k's nodes start in the flat arrays, so the child of local node
``n`` in tree ``k`` lives at ``offsets[k] + children_left[offsets[k] + n]``
(the offset is re-added at every step, never subtracted).
"""
import os
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXPORT_PATH = os.path.join(_REPO_ROOT, "models", "rf_trees.npz")

_cache = None


def _load():
    global _cache
    if _cache is None:
        data = np.load(_EXPORT_PATH)
        _cache = {
            "feature": data["feature"],
            "threshold": data["threshold"],
            "children_left": data["children_left"],
            "children_right": data["children_right"],
            "value": data["value"],
            "offsets": data["offsets"],
            "n_estimators": int(data["n_estimators"]),
        }
    return _cache


def _predict_tree_batch(X, offset, feature, threshold, children_left, children_right, value):
    """Vectorized traversal of one tree for every row of X at once."""
    n = X.shape[0]
    node_local = np.zeros(n, dtype=np.int64)  # local index within this tree; root = 0
    active = np.arange(n)

    while active.size:
        global_idx = offset + node_local[active]
        is_leaf = feature[global_idx] == -2
        if np.all(is_leaf):
            break

        internal = active[~is_leaf]
        active = internal  # leaves drop out of further traversal

        gi = offset + node_local[internal]
        f = feature[gi]
        th = threshold[gi]
        go_left = X[internal, f] <= th

        cl = children_left[gi]
        cr = children_right[gi]
        node_local[internal] = np.where(go_left, cl, cr)

    leaf_global = offset + node_local
    return value[leaf_global]


def predict(X):
    """Predict viability (%) for an (n_samples, 2) array of [DMSO, Trehalose].

    Returns
    -------
    numpy.ndarray, shape (n_samples,)
        Mean prediction across all 200 trees, identical to
        RandomForestRegressor.predict.
    """
    state = _load()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    n_trees = state["n_estimators"]
    offsets = state["offsets"]
    preds = np.zeros((n_trees, X.shape[0]), dtype=np.float64)

    for k in range(n_trees):
        preds[k] = _predict_tree_batch(
            X, int(offsets[k]),
            state["feature"], state["threshold"],
            state["children_left"], state["children_right"], state["value"],
        )

    return preds.mean(axis=0)


def predict_one(dmso, trehalose):
    """Convenience wrapper: predict viability (%) for a single (dmso, trehalose) pair."""
    return float(predict(np.array([[dmso, trehalose]], dtype=np.float64))[0])


if __name__ == "__main__":
    print(f"predict_one(5, 10) = {predict_one(5, 10):.4f}")
