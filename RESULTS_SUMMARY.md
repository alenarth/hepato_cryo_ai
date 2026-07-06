# HepatoCryoAI — Results Summary

This summary was assembled from executed notebook outputs and the serialized model artifacts produced during the analysis.  
All numbers below come from **executed notebook outputs** rather than manual transcription.

---

## 1. Full Model Comparison

**Split**: 80/20 (seed=42, 216 samples → 172 train, 44 test)  
**CV**: 5-fold KFold (seed=42, shuffle=True) over the 172-sample training set  
Comparison note: the PyTorch neural network uses a nested 60/20/20 split, so its cross-validated R² is not directly comparable to the single-split metrics reported for the other models.

| Model | R²(test) | RMSE(test) % | R²(CV 5-fold) |
|-------|---------|-------------|--------------|
| **Random Forest** | **0.9797** | **5.02** | **0.9602** |
| XGBoost | 0.9773 | 5.32 | 0.9605 |
| PyTorch NN (poly features) | 0.9278 | 9.48 | — |
| MLP sklearn (raw) | 0.6122 | 21.97 | 0.5849 |
| Polynomial Regression (deg=2) | 0.6096 | 22.04 | 0.3787 |
| SVR (raw) | 0.5441 | 23.82 | 0.2878 |
| **Linear Regression (raw)** | **0.4775** | **25.50** | **0.3399** |

Source: `notebooks/baselines_and_ablation.ipynb` → `data/comparison_table.csv`

---

## 2. Neural Network — Final Metrics (single deterministic run)

Seeds fixed: `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`,  
`torch.use_deterministic_algorithms(True)`, `DEVICE=cpu`, DataLoader generator seed=42.

| Metric | Value |
|--------|-------|
| R²(test) | 0.9278 |
| RMSE(test) | 9.48% |
| R²(train) | 0.8775 |
| RMSE(train) | 13.34% |
| CV R² mean | 0.8835 |
| CV R² std | ±0.0573 |
| CV RMSE mean | 12.13% |
| CV RMSE std | ±3.58% |
| Early stopping epoch | 381 |

These metrics are **identical** in `neural_network.ipynb` output, `nn_pipeline.pkl`, and the web app.

**Note**: Previous non-deterministic run gave R²(test)=0.9091 (different random seed handling).  
The deterministic run (above) is the authoritative value.

---

## 3. Random Forest — Final Metrics

| Metric | Value |
|--------|-------|
| R²(test) | 0.9797 |
| RMSE(test) | 5.02% |
| R²(CV mean) | 0.9602 |
| R²(CV std) | ±0.0196 |
| RMSE(CV mean) | 6.94% |
| RMSE(CV std) | ±1.25% |

Source: `rf_metrics.json` (written by `random_forest.ipynb`, loaded dynamically by `app.py`).

### Best Combinations (exhaustive grid search, 0–100% in 1% steps)

| Strategy | DMSO % | Trehalose % | Predicted Viability |
|----------|--------|------------|-------------------|
| Optimal Global | 2.0 | 0.0 | 95.19% |
| DMSO Alone | 2.0 | 0.0 | 95.19% |
| Trehalose Alone | 0.0 | 23.0 | 78.48% |

> **FLAG FOR ARTICLE**: The optimal trehalose-alone concentration from the new run is 23.0%,
> not "16–20%" as stated in the article. Both give the same predicted viability (78.48%).
> The RF with sklearn 1.4.0 distributes probability mass slightly differently from the
> original sklearn 1.8.0. The article range "16–20%" should be verified against the
> current model execution or updated to reflect the actual best concentration.

---

## 4. Feature Ablation Tables

### 4a — Random Forest: Raw vs. Polynomial Features

| Features | R²(test) | R²(CV) |
|---------|---------|-------|
| Raw [DMSO, Trehalose] | 0.9797 | 0.9602 |
| Polynomial (deg=2, 5 features) | 0.9800 | 0.9561 |

**Conclusion**: RF does NOT need polynomial features. The marginal test gain (+0.0003)
comes at the cost of CV stability (−0.0041), confirming trees handle non-linearity natively
through axis-aligned splits.

### 4b — Neural Network: Raw vs. Polynomial Features

| Features | R²(test) | RMSE(test) % | Early-stop epoch |
|---------|---------|-------------|----------------|
| Raw [DMSO, Trehalose] (2 features) | 0.8830 | 12.07 | 354 |
| Polynomial (deg=2, 5 features) | 0.9278 | 9.48 | 381 |

**Conclusion (with default regularization)**: Polynomial features improve NN test R² by 0.045.

---

## 5. Raw-Feature NN Underfitting Investigation

Tests whether the NN gap with raw features is due to over-regularization vs. architecture.

| Variant | R²(test) | RMSE(test) % | Early-stop ep |
|---------|---------|-------------|--------------|
| Default (BN, drop=0.2, wd=1e-4, pat=100) | 0.8830 | 12.07 | 354 |
| No dropout (BN, drop=0, wd=1e-4, pat=100) | 0.9304 | 9.31 | 365 |
| No BatchNorm (drop=0.2, wd=1e-4, pat=100) | 0.8437 | 13.95 | 298 |
| No weight decay (BN, drop=0.2, wd=0, pat=100) | 0.8828 | 12.08 | 356 |
| More patience (BN, drop=0.2, wd=1e-4, pat=200) | 0.8830 | 12.07 | 354 |
| **Combined (no BN, drop=0, wd=0, pat=200)** | **0.9783** | **5.20** | **500** |

**Poly-feature NN (reference)**: R²(test) = 0.9278

**Conclusion**: The raw-feature NN with reduced regularization (no BN, no dropout, no
weight decay, extended patience) achieves R²=0.9783, **surpassing** the polynomial NN
(R²=0.9278). This indicates the original poly-feature advantage was due to
**over-regularization of the raw-feature NN**, not an architectural limitation.  
The polynomial features are NOT architecturally necessary for the NN to fit this dataset.

---

## 6. SFB Verification Numbers

```
Sum of DMSO + Trehalose + SFB:
  Rows summing to 100: 208 / 216
  Rows summing to 10:   8 / 216

Rank of [1, DMSO, Trehalose, SFB] matrix (216×4): 4 (FULL RANK — not rank-deficient)

R²(SFB ~ DMSO + Trehalose, linear): 0.7083
  → SFB is ~70.8% linearly predictable from DMSO + Trehalose (NOT ~71% as originally stated)

RF without SFB: R²(test) = 0.9797
RF with SFB:    R²(test) = 0.9798   (Delta = +0.0001 — negligible)

Feature importances WITHOUT SFB: DMSO=0.777, Trehalose=0.223
Feature importances WITH SFB:    DMSO=0.655, Trehalose=0.097, SFB=0.248
```

**Correct justification for SFB exclusion** (replacing the incorrect "linear dependence" claim):
1. SFB adds negligible predictive value (ΔR²=+0.0001).
2. Including SFB inflates its apparent importance (SFB takes ~24.8% Gini credit) at the
   expense of DMSO (drops from 77.7% to 65.5%), because SFB acts as a balance/diluent
   variable, not an independently optimized cryoprotective factor.
3. The design matrix has full rank — the exact linear dependency (SFB=100−DMSO−Trehalose)
   does NOT hold for 8/216 rows (which sum to 10, not 100).

---

## 7. Polynomial Feature Correlation

```
corr(DMSO, DMSO²) = 0.9586
R²(DMSO² ~ DMSO, linear) = 0.9189
```

This confirms that Gini (MDI) importance distributes credit arbitrarily between DMSO and DMSO²
(Strobl et al., 2008, BMC Bioinformatics 9:307), making the polynomial feature importance
plot **uninterpretable**. Feature importance now uses raw features only.

---

## 8. Hyperparameter Table

| Model | Key Hyperparameters |
|-------|---------------------|
| **Random Forest** | n_estimators=200, criterion=squared_error (default), max_features=1.0 (sklearn 1.4 default), random_state=42 |
| **XGBoost** | n_estimators=100, learning_rate=0.1, random_state=42, all other params default |
| **MLP sklearn** | hidden_layer_sizes=(64,32), max_iter=2000, early_stopping=True, random_state=42, solver=adam (default), alpha=0.0001 (default) |
| **SVR** | kernel='rbf', C=100, gamma=0.1, epsilon=0.1 (default) |
| **Linear Regression** | no hyperparameters |
| **Polynomial Regression deg=2** | degree=2, include_bias=False + LinearRegression |
| **PyTorch NN (poly)** | layers=128→64→32→1, BatchNorm1d, Dropout=0.2, Adam lr=0.001, weight_decay=1e-4, ReduceLROnPlateau (factor=0.5, patience=30), early_stopping patience=100, epochs_max=2000, batch_size=32, features=5 poly deg=2 |

Source: `notebooks/baselines_and_ablation.ipynb`, `notebooks/neural_network.ipynb`

---

## 9. Files Modified

| File | Changes |
|------|---------|
| `notebooks/random_forest.ipynb` | Added seed, permutation importance, `../static/images/` save paths, rf_metrics.json export, RMSE CV |
| `notebooks/neural_network.ipynb` | Full rewrite: deterministic seeds, CPU device, fixed feature importance (raw+train-only+permutation), removed polynomial FI, fixed SFB cell (correct reason), added split documentation, fixed figure save paths |
| `notebooks/alternatives.ipynb` | Fixed legend "Rede Neural" → "Neural Network (MLP)", removed Portuguese comments, save to correct paths |
| `notebooks/baselines_and_ablation.ipynb` | **New file**: linear/poly baselines, full comparison table, feature ablation (RF+NN), underfitting investigation, SFB verification, polynomial correlation |
| `templates/lab_data.html` | Fixed all mojibake (UTF-8 BOM removed, 24 special-char sequences corrected), updated Feature Importance description (polynomial → raw+permutation), corrected SFB description |
| `templates/simulator.html` | RF metrics now dynamic (loaded from rf_metrics.json via app.py) |
| `app.py` | Added `get_rf_metrics()` function loading `rf_metrics.json`, passes `rf_metrics` to both templates |
| `rf_metrics.json` | **New file**: RF test metrics + CV metrics + best combinations (written by notebook, read by app) |
| `data/comparison_table.csv` | **New file**: Full model comparison table |
| `static/images/*.png` | Regenerated: feature_importance.png (raw+permutation), nn_feature_importance.png, nn_distribuicao.png (with fitted curves), nn_correlacao.png, real_vs_pred.png, heatmap_viability.png, residuals_plot.png, learning_curve.png, nn_real_vs_previsto.png, nn_residuos.png, nn_comparativo_rf_vs_nn.png, baselines_comparison.png |

---

## 10. FLAGS FOR ARTICLE

Issues where the new evidence **contradicts or updates** the current article narrative.
Each flag requires a human decision before updating the article text.

| # | Location in Article | Current claim | Evidence from code | Decision needed |
|---|---------------------|--------------|-------------------|----------------|
| F1 | SFB exclusion justification | "SFB ≈ 100% − DMSO% − Trehalose%, introducing multicollinearity" | The sum equals 100 for only 208/216 rows; 8 rows sum to 10. Matrix is full rank. Correct reason is negligible predictive value + confounded importance | Replace justification with correct reason (see §6) |
| F2 | Optimal trehalose-alone concentration | "16–20% trehalose" | Current model (sklearn 1.4.0) gives 23% trehalose as optimal, same predicted viability 78.48% | Verify experimentally; update range or note model version dependency |
| F3 | NN vs RF comparison narrative | RF superiority attributed to its "non-parametric advantage" | Ablation shows raw-feature NN with minimal regularization reaches R²=0.9783 > RF=0.9797 ≈ RF. The poly-feature NN (R²=0.9278) underperforms due to over-regularization, NOT inherent NN limitation | The narrative about NN needing poly features to be competitive may need qualification |
| F4 | NN R² value reported | R²=0.9091 (or similar from non-deterministic run) | Deterministic run gives R²=0.9278. Seeds were not fully fixed in the original training | Update NN metrics in article to R²=0.9278, RMSE=9.48%, epoch=381 |
| F5 | Feature importance discussion | DMSO²  dominates (~42%) feature importance | Polynomial feature importance is unreliable due to corr(DMSO,DMSO²)=0.96 (Strobl 2008). Raw feature importance: DMSO~77.7%, Trehalose~22.3% | Remove polynomial FI discussion; use raw feature importance values |
| F6 | SFB R² predictability | "~71% predictable" | Actual R²=0.7083 (~70.8%); not a meaningful discrepancy but good to be precise | Minor: update to 70.8% |
