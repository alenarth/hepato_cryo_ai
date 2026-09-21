# Results Summary

This file is the bridge between the code (notebooks, `metrics.json`) and the paper.
All numbers below are read directly from `metrics.json`, generated on 2026-09-04.
Do not hand-edit numbers here without regenerating `metrics.json` from the notebooks first.

Dataset: **206 samples** (HepG2). Split: 164 train+validation /
42 test (`test_size=0.2, random_state=42`), identical across every
notebook. For the neural network, the 164 train+validation samples are further
split into 123 fit / 41 validation
(`test_size=0.25, random_state=42`).

## Anchor values (must not change without investigation)

| Item | Value |
|---|---|
| Total samples | 206 |
| Splits | 164 train+val / 42 test; 123 fit / 41 validation |
| Random Forest — R² test | 0.9840 |
| Random Forest — RMSE test (pp) | 4.6043 |
| Random Forest — CV R² | 0.9598 ± 0.0131 |
| Random Forest — feature importance | DMSO ≈ 0.7346 / Trehalose ≈ 0.2654 |
| Linear regression baseline — R² test | 0.5217 |
| Polynomial regression baseline — R² test | 0.5823 |
| SVR — R² test | 0.6485 |
| Highest predicted viability, RF (plateau) | 15 combinations at 92.67%, DMSO 13–17%, trehalose 0–2% |
| Highest predicted viability, trehalose alone (RF) | 16–25% at 77.57% |
| Data split | 123 / 41 / 42 |

## Random Forest (features: raw `% DMSO`, `TREHALOSE`)

| Metric | Value |
|---|---|
| R² (test, 42 samples) | 0.9840 |
| RMSE (test, pp) | 4.6043 |
| CV R² (5-fold, mean ± std) | 0.9598 ± 0.0131 |
| CV RMSE (5-fold, mean ± std, pp) | 6.9691 ± 1.0810 |
| Feature importance — % DMSO | 0.7346 |
| Feature importance — TREHALOSE | 0.2654 |
| Permutation importance — % DMSO | 1.9637 ± 0.1637 |
| Permutation importance — TREHALOSE | 0.6033 ± 0.0637 |
| Hyperparameters | `n_estimators=200, random_state=42` |

**Highest predicted viability (Random Forest; grid of 5,151 points, DMSO/Trehalose 0–100% step 1%,
sum ≤ 100%; 29 grid points coincide with formulations tested experimentally):**

The values below are model predictions. The Random Forest predicts by piecewise-constant regions,
so each maximum is a **plateau** of grid points tied at exactly the same predicted value, not a
single point. Each row below is therefore reported as a range.

| Strategy | DMSO % | Trehalose % | Predicted viability % | Tied combinations |
|---|---|---|---|---|
| Highest predicted — combined | 13–17 | 0–2 | 92.67 | 15 |
| Highest predicted — DMSO alone | 13–17 | 0 | 92.67 | 5 |
| Highest predicted — trehalose alone | 0 | 16–25 | 77.57 | 10 |

## Neural Network (PyTorch → NumPy export; features: raw `% DMSO`, `TREHALOSE`)

Architecture: `2 -> 128 -> 64 -> 32 -> 1 (ReLU nas camadas ocultas, saída linear)`, Adam (lr=1e-3),
`ReduceLROnPlateau(factor=0.5, patience=30, min_lr=1e-6)`, MSE loss, batch_size=32,
max 2000 epochs, early stopping on validation loss. The output layer is linear, so the raw
network output is not bounded; the application limits the displayed prediction to the physical
range of viability, 0–100%. No feature engineering: the two raw concentrations are the only inputs.

**Hyperparameter search** (8 configs × 2 seeds
[0, 42], 16 runs, selected by mean validation MSE — never by test; early-stopping patience was
fixed at 100 before the search and is identical for every configuration).

| batch_norm | dropout | weight_decay | patience | mean val loss | std val loss | mean best epoch |
|---|---|---|---|---|---|---|
| **True** | **0.0** | **0** | **100** | **64.94** | **12.53** | **523** |
| True | 0.0 | 1e-4 | 100 | 65.40 | 6.62 | 510 |
| False | 0.0 | 1e-4 | 100 | 75.11 | 52.17 | 451 |
| False | 0.0 | 0 | 100 | 83.73 | 63.18 | 591 |
| True | 0.2 | 0 | 100 | 101.99 | 4.99 | 410 |
| True | 0.2 | 1e-4 | 100 | 104.25 | 8.41 | 410 |
| False | 0.2 | 0 | 100 | 105.95 | 24.17 | 511 |
| False | 0.2 | 1e-4 | 100 | 153.30 | 9.39 | 266 |

**Winning configuration:** `batch_norm=True, dropout=0.0, weight_decay=0, patience=100` (mean validation loss 64.9415)

BatchNorm is selected, and weight decay is not.
The two BatchNorm configurations are also markedly more stable across seeds (std of validation
loss ~6–13) than the BatchNorm-free ones (std ~52–63).

| Metric | Value |
|---|---|
| R² — fit (123) | 0.9707 |
| RMSE — fit (123), pp | 6.1650 |
| R² — validation (41) | 0.9538 |
| RMSE — validation (41), pp | 7.4888 |
| R² — test (42) | 0.9818 |
| RMSE — test (42), pp | 4.9177 |
| Best epoch (early stopping) | 523 |
| Test residuals — mean (pp) | -0.6783 |
| Test residuals — std (pp) | 4.8707 |
| CV R² (5-fold, mean ± std) | 0.9643 ± 0.0105 |
| CV RMSE (5-fold, mean ± std, pp) | 6.6049 ± 0.9289 |

Per-fold CV R²: 0.9488 / 0.9595 / 0.9615 / 0.9731 / 0.9785. The `StandardScaler` is refit inside each fold.

The configuration was fixed on validation loss before the test set was scored. The same held-out
test partition, split by observation (42 samples), was used to report the performance of all
compared models and in the seed-stability check below. Test R² = 0.9818 is above the 0.85 sanity threshold.

**Seed-stability check** (control only, not a headline result; seeds
[0, 1, 2, 42, 123]): 0: 0.9825, 1: 0.9741, 2: 0.9757, 42: 0.9818, 123: 0.9772.

Test R² = 0.9782 ± 0.0033
— well under the 0.05 sensitivity threshold, so the result is stable across seeds.

## NumPy export acceptance tests

Both deployed models run on NumPy-only exports. Both exports are validated on a dense grid
(DMSO/Trehalose 0–100%, step 1%, sum ≤ 100% — 5151 points).

| Export | Criterion | Result | Status |
|---|---|---|---|
| `models/rf_trees.npz` vs scikit-learn | difference exactly 0 | max abs diff = 0, 0 mismatching points | **PASS** |
| `models/nn_weights.npz` vs PyTorch | max abs diff < 1e-4 | max abs diff = 7.714e-05 | **PASS** |

The Random Forest export is checked both batched and one row at a time. `src/rf_inference.py`
accumulates tree predictions sequentially and divides once, mirroring scikit-learn's own
accumulation order: an `np.stack(...).mean(axis=0)` matches on a wide batch but drifts by ~1e-14
on the single-row calls the web application actually makes.

The winning configuration now uses BatchNorm, which the exporter folds into the adjacent Linear
layer, so NumPy inference remains a plain sequence of affine layers with ReLU.

## Baselines

| Model | R² test | RMSE test (pp) | R² CV (5-fold) |
|---|---|---|---|
| Linear regression (raw features) | 0.5217 | 25.1770 | 0.2890 |
| Polynomial regression (degree 2) | 0.5823 | 23.5275 | 0.3654 |

## Model comparison

| Model | R² test | RMSE test (pp) | R² CV (5-fold) |
|---|---|---|---|
| Random Forest | 0.9840 | 4.6043 | 0.9598 |
| Neural Network (ANN) | 0.9818 | 4.9177 | 0.9643 |
| XGBoost | 0.9814 | 4.9602 | 0.9627 |
| SVR (raw) | 0.6485 | 21.5835 | 0.3460 |
| Polynomial regression (degree 2) | 0.5823 | 23.5275 | 0.3654 |
| Linear regression | 0.5217 | 25.1770 | 0.2890 |

Also exported to `data/comparison_table.csv`.

## Example prediction — 10% DMSO / 0% Trehalose

The combination quoted in a figure legend of the paper.

| Model | Predicted viability % |
|---|---|
| Random Forest (scikit-learn) | 91.22 |
| Random Forest (NumPy export) | 91.22 |
| Neural Network (PyTorch) | 89.69 |
| Neural Network (NumPy export) | 89.69 |

## Reproducing this pass

Run the notebooks in `notebooks/` in numeric order. `metrics.json` is created by notebook 01 and
extended by 02, 03 and 04; `data/comparison_table.csv` and `data/hyperparameters.csv` are derived
from it in notebook 04, so the three files cannot drift apart.

1. `01_random_forest.ipynb` — Random Forest, exhaustive grid search, `rf_trees.npz` export and
   its exactness test; creates `metrics.json`.
2. `02_neural_network.ipynb` — EDA figures, regularization search, final model, 5-fold CV,
   seed-stability check, `nn_weights.npz` export and its acceptance test.
3. `03_model_comparison.ipynb` — XGBoost and SVR, comparison figures, learning curves.
4. `04_baselines.ipynb` — linear and polynomial baselines; final assembly of `metrics.json`,
   `comparison_table.csv` and `hyperparameters.csv`.

## Files generated by the notebooks

- All four notebooks in `notebooks/` — executed end to end with stored outputs.
- `models/random_forest_model.pkl`, `models/rf_trees.npz`, `models/nn_model.pth`,
  `models/nn_weights.npz`.
- `metrics.json`, `data/comparison_table.csv`, `data/hyperparameters.csv`.
- The 14 code-generated figures in `static/images/`.

The three `wetlab_*.png` figures in `static/images/` are bench figures produced in GraphPad and
have no generating code in this repository.
