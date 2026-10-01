# Results Summary

This file is the bridge between the code (notebooks, `metrics.json`) and the paper.
All numbers below are read directly from `metrics.json`, generated on 2026-10-01.
Do not hand-edit numbers here without regenerating `metrics.json` from the notebooks first.

Dataset: **235 HepG2 observations**, all used. Split: 188 train+validation /
47 test (`test_size=0.2, random_state=42`), identical across every
notebook. For the neural network, the 188 train+validation samples are further
split into 141 fit / 47 validation
(`test_size=0.25, random_state=42`).

## Dataset

`data/raw/hepg2.csv` is built from the laboratory spreadsheet by `data/build_dataset.py`,
which preserves the spreadsheet `INDEX` and records each observation's origin in the `ORIGEM`
column. It holds 235 observations of 35 distinct DMSO/trehalose formulations:

| Origin (`ORIGEM`) | Description | Spreadsheet `INDEX` | Observations |
|---|---|---|---|
| `base_original` | original laboratory database | 118–251 | 147 |
| `maio_2025` | DMSO experiments, May 2025 | 252–284 | 33 |
| `dez_2025_trealose` | trehalose experiments, December 2025 | 285–323 | 39 |
| `dez_2025_dmso_trealose` | DMSO + trehalose experiments, December 2025 | 324–339 | 16 |
| **Total** | | | **235** |

The models use only `% DMSO` and `TREHALOSE` (both in %) as inputs and `VIABILIDADE`
(post-thaw viability, %) as the target.

## Anchor values (must not change without investigation)

| Item | Value |
|---|---|
| Total samples | 235 (147 / 33 / 39 / 16 by origin) |
| Splits | 188 train+val / 47 test; 141 fit / 47 validation |
| Random Forest — R² test | 0.9266 |
| Random Forest — RMSE test (pp) | 10.4366 |
| Random Forest — CV R² | 0.8121 ± 0.0542 |
| Random Forest — CV RMSE (pp) | 15.4086 ± 1.8908 |
| Random Forest — feature importance | DMSO ≈ 0.8411 / Trehalose ≈ 0.1589 |
| Linear regression baseline — R² test / CV | 0.2184 / 0.2473 |
| Polynomial regression baseline — R² test / CV | 0.2675 / 0.2573 |
| SVR — R² test / CV | 0.0504 / 0.0817 |
| Highest predicted viability, RF (plateau) | 6 combinations at 95.20%, DMSO 2–3%, trehalose 0–2% |
| Highest predicted viability, trehalose alone (RF) | 16–25% at 43.71% |
| RF prediction, 10% DMSO / 0% trehalose | 91.48% |

The CV RMSE was computed with scikit-learn 1.9.0, the version pinned in `requirements-dev.txt`.
scikit-learn 1.8.0 gives a slightly different tree in one fold (fold 3 RMSE 13.8834 instead of
13.8847), which moves the mean to 15.4083 ± 1.8910; every other anchor is identical across the
two versions.

## Random Forest (features: raw `% DMSO`, `TREHALOSE`)

| Metric | Value |
|---|---|
| R² (test, 47 samples) | 0.9266 |
| RMSE (test, pp) | 10.4366 |
| CV R² (5-fold, mean ± std) | 0.8121 ± 0.0542 |
| CV RMSE (5-fold, mean ± std, pp) | 15.4086 ± 1.8908 |
| Feature importance — % DMSO | 0.8411 |
| Feature importance — TREHALOSE | 0.1589 |
| Permutation importance — % DMSO | 1.7236 ± 0.1708 |
| Permutation importance — TREHALOSE | 0.3159 ± 0.0290 |
| Test residuals — mean / std (pp) | -0.4892 / 10.5378 |
| Learning curve at 150 samples — training / CV R² | 0.8831 / 0.8111 |
| Hyperparameters | `n_estimators=200, random_state=42` |

**Highest predicted viability (Random Forest; grid of 5,151 points, DMSO/Trehalose 0–100% step 1%,
sum ≤ 100%; 35 grid points coincide with formulations tested experimentally):**

The values below are model predictions. The Random Forest predicts by piecewise-constant regions,
so each maximum is a **plateau** of grid points tied at exactly the same predicted value, not a
single point. Each row below is therefore reported as a range.

| Strategy | DMSO % | Trehalose % | Predicted viability % | Tied combinations |
|---|---|---|---|---|
| Highest predicted — overall | 2–3 | 0–2 | 95.20 | 6 |
| Highest predicted — DMSO alone | 2–3 | 0 | 95.20 | 2 |
| Highest predicted — trehalose alone | 0 | 16–25 | 43.71 | 10 |

Tied combinations (DMSO %, trehalose %):

- Overall: (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2).
- DMSO alone: (2, 0), (3, 0).
- Trehalose alone: (0, 16) through (0, 25), every integer step.

## Neural Network (PyTorch → NumPy export; features: raw `% DMSO`, `TREHALOSE`)

Architecture: `2 -> 128 -> 64 -> 32 -> 1 (ReLU on hidden layers, linear output)`, Adam (lr=1e-3),
`ReduceLROnPlateau(factor=0.5, patience=30, min_lr=1e-6)`, MSE loss, batch_size=32,
max 2000 epochs, early stopping on validation loss. The output layer is linear, so the raw
network output is not bounded; the application limits the displayed prediction to the physical
range of viability, 0–100%. No feature engineering: the two raw concentrations are the only inputs.
The `StandardScaler` is fitted on the 141-sample fit set only.

**Hyperparameter search**, redone on this dataset without assuming the previous winner
(8 configs × 2 seeds [0, 42], 16 runs, selected by mean validation MSE — never by test;
early-stopping patience was fixed at 100 before the search and is identical for every configuration).

| batch_norm | dropout | weight_decay | patience | mean val loss | std val loss | mean best epoch |
|---|---|---|---|---|---|---|
| **True** | **0.0** | **0** | **100** | **308.92** | **3.19** | **250** |
| True | 0.0 | 1e-4 | 100 | 313.21 | 3.96 | 284 |
| True | 0.2 | 1e-4 | 100 | 396.53 | 10.89 | 279 |
| True | 0.2 | 0 | 100 | 396.69 | 9.44 | 312 |
| False | 0.2 | 1e-4 | 100 | 441.28 | 7.56 | 276 |
| False | 0.0 | 1e-4 | 100 | 445.68 | 3.85 | 208 |
| False | 0.0 | 0 | 100 | 451.61 | 2.69 | 232 |
| False | 0.2 | 0 | 100 | 451.86 | 8.35 | 276 |

**Winning configuration:** `batch_norm=True, dropout=0.0, weight_decay=0, patience=100` (mean validation loss 308.9223)

The four BatchNorm configurations have the four lowest validation losses; within them, dropout
raises the validation loss and weight decay changes it little.

| Metric | Value |
|---|---|
| R² — fit (141) | 0.7789 |
| RMSE — fit (141), pp | 17.7065 |
| R² — validation (47) | 0.7314 |
| RMSE — validation (47), pp | 17.6402 |
| R² — test (47) | 0.8735 |
| RMSE — test (47), pp | 13.6988 |
| Best epoch (early stopping) | 264 |
| Test residuals — mean (pp) | -0.2621 |
| Test residuals — std (pp) | 13.6963 |
| CV R² (5-fold, mean ± std) | 0.7707 ± 0.0413 |
| CV RMSE (5-fold, mean ± std, pp) | 17.2208 ± 1.2983 |

Per-fold CV R²: 0.7239 / 0.8267 / 0.8111 / 0.7584 / 0.7333. The 5-fold CV runs over the 188
train+validation samples, and the `StandardScaler` is refit inside each fold.

The configuration was fixed on validation loss before the test set was scored. The same held-out
test partition, split by observation (47 samples), was used to report the performance of all
compared models and in the seed-stability check below. Test R² = 0.8735 is in the same range as
the Random Forest (0.9266) and within the 0.85–0.97 sanity range; a value above 0.97 would point to
leakage, given how much replicates of the same formulation disagree (notably trehalose alone).

**Seed-stability check** (control only, not a headline result; seeds
[0, 1, 2, 42, 123]): 0: 0.8612, 1: 0.8488, 2: 0.8604, 42: 0.8735, 123: 0.8656.

Test R² = 0.8619 ± 0.0080
— well under the 0.05 sensitivity threshold, so the result is stable across seeds.

## NumPy export acceptance tests

Both deployed models run on NumPy-only exports. Both exports are validated on a dense grid
(DMSO/Trehalose 0–100%, step 1%, sum ≤ 100% — 5151 points).

| Export | Criterion | Result | Status |
|---|---|---|---|
| `models/rf_trees.npz` vs scikit-learn | difference exactly 0 | max abs diff = 0, 0 mismatching points | **PASS** |
| `models/nn_weights.npz` vs PyTorch | max abs diff < 1e-4 | max abs diff = 7.398e-05 | **PASS** |

The Random Forest export is checked both batched and one row at a time. `src/rf_inference.py`
accumulates tree predictions sequentially and divides once, mirroring scikit-learn's own
accumulation order: an `np.stack(...).mean(axis=0)` matches on a wide batch but drifts by ~1e-14
on the single-row calls the web application actually makes.

The winning configuration uses BatchNorm, which the exporter folds into the adjacent Linear
layer, so NumPy inference remains a plain sequence of affine layers with ReLU.

## Baselines

| Model | R² test | RMSE test (pp) | R² CV (5-fold) |
|---|---|---|---|
| Linear regression (raw features) | 0.2184 | 34.0579 | 0.2473 |
| Polynomial regression (degree 2) | 0.2675 | 32.9698 | 0.2573 |

## Model comparison

| Model | R² test | RMSE test (pp) | R² CV (5-fold) |
|---|---|---|---|
| Random Forest | 0.9266 | 10.4366 | 0.8121 |
| XGBoost | 0.9255 | 10.5150 | 0.8074 |
| Neural Network (ANN) | 0.8735 | 13.6988 | 0.7707 |
| Polynomial regression (degree 2) | 0.2675 | 32.9698 | 0.2573 |
| Linear regression | 0.2184 | 34.0579 | 0.2473 |
| SVR (raw) | 0.0504 | 37.5398 | 0.0817 |

Also exported to `data/comparison_table.csv`.

## Example prediction — 10% DMSO / 0% Trehalose

The combination quoted in a figure legend of the paper.

| Model | Predicted viability % |
|---|---|
| Random Forest (scikit-learn) | 91.48 |
| Random Forest (NumPy export) | 91.48 |
| Neural Network (PyTorch) | 89.44 |
| Neural Network (NumPy export) | 89.44 |

## Reproducing this pass

`data/raw/hepg2.csv` is rebuilt from the laboratory sources (kept out of version control in
`data/source/`) with:

```bash
python data/build_dataset.py --planilha data/source/planilha_criopreservacao_dez2025.xlsx \
    --base-original data/source/hepg2_base_original.csv --saida data/raw/hepg2.csv
```

The expected MD5 of the generated file is `89458bc9b0d8a16285501ab066c2a722`.

Then run the notebooks in `notebooks/` in numeric order. `metrics.json` is created by notebook 01 and
extended by 02, 03 and 04; `data/comparison_table.csv` and `data/hyperparameters.csv` are derived
from it in notebook 04, so the three files cannot drift apart.

1. `01_random_forest.ipynb` — Random Forest, exhaustive grid search, `rf_trees.npz` export and
   its exactness test; creates `metrics.json` (including the dataset composition by origin).
2. `02_neural_network.ipynb` — EDA figures and summary, regularization search, final model,
   5-fold CV, seed-stability check, `nn_weights.npz` export and its acceptance test.
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
have no generating code in this repository. `wetlab_trehalose_viability.png` shows only the
trehalose-alone observations of the original laboratory database; the trehalose experiments of
December 2025 are not in it.
