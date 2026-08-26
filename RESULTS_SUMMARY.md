# Results Summary

This file is the bridge between the code (notebooks, `metrics.json`) and the paper.
All numbers below are read directly from `metrics.json`, generated on 2026-08-16.
Do not hand-edit numbers here without regenerating `metrics.json` from the notebooks first.

Dataset: 216 samples (HepG2). Split: 172 train+validation / 44 test (`test_size=0.2,
random_state=42`), identical across every notebook. For the neural network, the 172
train+validation samples are further split into 129 fit / 43 validation
(`test_size=0.25, random_state=42`).

## Anchor values (must not change without investigation)

| Item | Value |
|---|---|
| Random Forest — R² test | 0.9797 |
| Random Forest — RMSE test | 5.0245 |
| Random Forest — CV R² | 0.9602 ± 0.0196 |
| Random Forest — feature importance | DMSO ≈ 0.777 / Trehalose ≈ 0.223 |
| Linear regression baseline — R² test | 0.4775 |
| Polynomial regression baseline — R² test | 0.6096 |
| Data split | 129 / 43 / 44 |

## Random Forest (features: raw `% DMSO`, `TREHALOSE`)

| Metric | Value |
|---|---|
| R² (test, 44 samples) | 0.9797 |
| RMSE (test) | 5.0245 |
| CV R² (5-fold, mean ± std) | 0.9602 ± 0.0196 |
| CV RMSE (5-fold, mean ± std) | 6.9381 ± 1.2493 |
| Feature importance — % DMSO | 0.7768 |
| Feature importance — TREHALOSE | 0.2232 |
| Hyperparameters | `n_estimators=200, random_state=42` |

**Best combinations (exhaustive grid search, DMSO/Trehalose 0–100% step 1%, sum ≤ 100%):**

| Strategy | DMSO % | Trehalose % | Predicted viability % |
|---|---|---|---|
| Global optimum | 2.0 | 1.0 | 95.19 |
| Best DMSO-only (no trehalose) | 2.0 | 0.0 | 95.19 |
| Best trehalose-only (no DMSO) | 0.0 | 23.0 | 78.48 |

## Neural Network (PyTorch → NumPy export; features: raw `% DMSO`, `TREHALOSE`)

Architecture: `2 → 128 → 64 → 32 → 1`, ReLU, Adam (lr=1e-3),
`ReduceLROnPlateau(factor=0.5, patience=30, min_lr=1e-6)`, MSE loss, batch_size=32,
max 2000 epochs, early stopping on validation loss. Polynomial feature expansion
(previously DMSO², DMSO×Trehalose, Trehalose²) was removed from the pipeline.

**Hyperparameter search** (24 configs × 3 seeds [0, 1, 42], selected by mean
validation MSE — never by test):

| batch_norm | dropout | weight_decay | patience | mean val loss | std val loss |
|---|---|---|---|---|---|
| **False** | **0.0** | **1e-4** | **100** | **45.77** | **4.20** |
| False | 0.1 | 1e-4 | 200 | 47.23 | 4.68 |
| False | 0.1 | 1e-4 | 100 | 47.54 | 4.94 |
| False | 0.0 | 0 | 100 | 48.93 | 5.60 |
| True | 0.0 | 1e-4 | 200 | 61.61 | 9.34 |
| True | 0.2 | 1e-4 | 100 | 92.79 | 7.76 |

*(full 24-row table is printed in `notebooks/neural_network.ipynb`; winning row bolded above)*

**Winning configuration:** `batch_norm=False, dropout=0.0, weight_decay=1e-4, patience=100`

| Metric | Value |
|---|---|
| R² — fit (129) | 0.9795 |
| RMSE — fit (129) | 5.4652 |
| R² — validation (43) | 0.9576 |
| RMSE — validation (43) | 6.5839 |
| R² — test (44) | 0.9788 |
| RMSE — test (44) | 5.1370 |
| Best epoch (early stopping) | 441 |
| Test residuals — mean | 0.3362 |
| Test residuals — std | 5.1260 |
| CV R² (5-fold, mean ± std) | 0.9680 ± 0.0124 |
| CV RMSE (5-fold, mean ± std) | 6.3072 ± 0.9676 |

**Seed-stability check** (control only, not a headline result; seeds 0, 1, 2, 42, 123):

Test R² = 0.9771 ± 0.0019 — well under the 0.05 sensitivity threshold, result is stable.

**PyTorch → NumPy export validation** (dense grid, DMSO/Trehalose 0–100% step 1%, sum ≤ 100%,
5151 points): max absolute difference = 7.22e-05 (< 1e-4 required) — **PASS**.

## Baselines

| Model | R² test | RMSE test | R² CV (5-fold) |
|---|---|---|---|
| Linear regression (raw features) | 0.4775 | 25.5001 | 0.3399 |
| Polynomial regression (degree 2) | 0.6096 | 22.0430 | 0.3787 |

## Model comparison

| Model | R² test | RMSE test |
|---|---|---|
| Random Forest | 0.9797 | 5.0245 |
| Neural Network (raw features) | 0.9788 | 5.1370 |
| Polynomial regression (degree 2) | 0.6096 | 22.0430 |
| Linear regression | 0.4775 | 25.5001 |

## Example prediction — 5% DMSO / 10% Trehalose

| Model | Predicted viability % |
|---|---|
| Random Forest | 65.76 |
| Neural Network (PyTorch) | 69.23 |
| Neural Network (NumPy export) | 69.23 |

## Files produced by this pass

- `notebooks/neural_network.ipynb` — retrained on raw features, hyperparameter search, final
  model, 5-fold CV, seed-stability check, NumPy export, acceptance test.
- `nn_model.pth` — retrained PyTorch weights (2-feature input; reproducibility/provenance).
- `nn_export.npz` — StandardScaler parameters + Linear-layer weights for NumPy-only inference.
- `nn_inference.py` — `predict(dmso, trehalose)`, NumPy only, no torch/sklearn.
- `metrics.json` — single source of truth for all metrics on this page.
- `RESULTS_SUMMARY.md` — this file.

**Not touched in this pass** (by design, deferred to a later step): `app.py`, HTML templates,
figures in `static/images/`, the legacy polynomial-feature
NN pipeline artifact, `random_forest_model.pkl`, `random_forest.ipynb`.

> **Note (reorganization pass):** the legacy polynomial-feature
> pipeline artifact mentioned above has since been removed; `nn_export.npz`/`nn_inference.py`
> now live at `models/nn_weights.npz`/`src/nn_inference.py`; `random_forest.ipynb` is now
> `notebooks/01_random_forest.ipynb`. This file otherwise reflects the state at the time it was written.
