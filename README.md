# HepatoCryoAI

A web application that predicts post-thaw viability of cryopreserved HepG2
cells from the concentrations of two cryoprotectants, DMSO and trehalose.

## Scientific context

Cryopreservation of human hepatocytes is a bottleneck for cell-based
alternatives to liver transplantation: cells routinely lose viability and
function after thawing, and the choice of cryoprotectant concentrations has
a large, non-linear effect on the outcome. This project is built on
experimental observations of HepG2 cell viability across a grid of DMSO and
trehalose concentrations. The file `data/raw/hepg2.csv` holds 235
observations, all of which are used: 147 from the original laboratory
database, 33 from DMSO experiments of May 2025, 39 from trehalose experiments
of December 2025 and 16 from DMSO + trehalose experiments of December 2025.
It is built from the laboratory spreadsheet by `data/build_dataset.py`, which
preserves the spreadsheet `INDEX` and records each observation's origin in the
`ORIGEM` column. From that data, the project trains two distinct regression
algorithms on the same samples -- a Random Forest and a small neural
network -- and serves both through a web interface so that a given combination of
concentrations can be evaluated before running a wet-lab experiment.

## Models and performance

Both models take only the two raw concentrations, `% DMSO` and `TREHALOSE`,
as input. No polynomial or other hand-engineered features are used. The
neural network's output layer is linear, so its raw output is not bounded;
the application limits the displayed prediction to the physical range of
viability, 0-100%. RMSE values are in percentage points (pp).

| Model | R² (test) | RMSE (test, pp) | R² (5-fold CV) |
|---|---|---|---|
| Random Forest | 0.9266 | 10.44 | 0.8121 |
| XGBoost | 0.9255 | 10.52 | 0.8074 |
| Neural Network (ANN) | 0.8735 | 13.70 | 0.7707 |
| Polynomial Regression (deg 2) | 0.2675 | 32.97 | 0.2573 |
| Linear Regression | 0.2184 | 34.06 | 0.2473 |
| SVR | 0.0504 | 37.54 | 0.0817 |

XGBoost, Polynomial Regression, SVR and Linear Regression are included as
baselines / alternative-algorithm comparisons, not as deployed models. All
numbers above are read directly from `metrics.json`.

## Methodology

The 235 samples are split 80/20 with a fixed seed into 188 train+validation
samples and 47 test samples, held out from training and model selection.
The same held-out test partition, split by observation, was used to report
the performance of every model compared and in the neural network's
seed-stability check. For the
neural network, the 188 train+validation samples are further split into 141
for fitting and 47 for validation (early stopping and learning-rate
scheduling). Five-fold cross-validation is run over the 188 train+validation
samples for both models. The neural network's regularization configuration
(batch normalization, dropout, weight decay) was selected by a small grid
search scored on validation loss only -- the test set is never used for
model selection.

## Running the application

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000`. The runtime environment needs only
`flask` and `numpy` (about 45 MB installed): inference for both models runs
on NumPy alone, from weights exported ahead of time (`models/nn_weights.npz`,
`models/rf_trees.npz`) -- no scikit-learn and no PyTorch are required to
serve predictions.

## Reproducing the analysis

The full analysis -- training, cross-validation, and figure generation --
requires the development dependencies:

```bash
pip install -r requirements-dev.txt
```

`data/raw/hepg2.csv` is committed. To rebuild it from the laboratory sources
(kept out of version control in `data/source/`):

```bash
python data/build_dataset.py --planilha data/source/planilha_criopreservacao_dez2025.xlsx \
    --base-original data/source/hepg2_base_original.csv --saida data/raw/hepg2.csv
```

Then run the notebooks in `notebooks/` in numeric order:

1. `01_random_forest.ipynb` -- trains and validates the Random Forest.
2. `02_neural_network.ipynb` -- hyperparameter search, final training,
   cross-validation, and the NumPy export of the neural network.
3. `03_model_comparison.ipynb` -- compares Random Forest, XGBoost, the
   neural network, and SVR under an identical protocol.
4. `04_baselines.ipynb` -- Linear Regression and Polynomial Regression
   baselines.

## Project structure

```
hepato_cryo_ai/
├── README.md                 # this file
├── requirements.txt          # runtime dependencies (flask, numpy)
├── requirements-dev.txt      # + dependencies to reproduce the analysis
├── app.py                    # Flask application
├── metrics.json              # single source of truth for all reported numbers
├── src/
│   ├── nn_inference.py       # NumPy-only neural network inference
│   └── rf_inference.py       # NumPy-only Random Forest inference
├── models/
│   ├── nn_weights.npz        # exported neural network weights (used by the app)
│   ├── rf_trees.npz          # exported Random Forest trees (used by the app)
│   ├── nn_model.pth          # PyTorch weights (reference/audit only)
│   └── random_forest_model.pkl  # scikit-learn model (reference/audit only)
├── data/
│   ├── build_dataset.py      # builds raw/hepg2.csv from the laboratory spreadsheet
│   ├── source/               # laboratory sources (not version-controlled)
│   ├── raw/hepg2.csv         # dataset: 235 observations, all analyzed
│   ├── comparison_table.csv  # model comparison table
│   └── hyperparameters.csv   # hyperparameters for every model
├── notebooks/                # analysis notebooks, numbered by execution order
├── static/images/            # all figures (single location)
├── templates/                # Flask/Jinja2 HTML templates
└── tests/                    # regression tests
```

## Reproducibility notes

All random seeds are fixed (Python, NumPy, PyTorch, and scikit-learn where
applicable), and training runs on CPU with PyTorch's deterministic algorithms
enabled. The fixed seeds determine the train/test split, the fit/validation
split, the cross-validation folds, the Random Forest's bootstrap samples, the
network's weight initialization and its mini-batch order, so every run of the
notebooks trains on the same partitions from the same starting point.
Dependency versions are pinned in `requirements.txt` and
`requirements-dev.txt`. `metrics.json` is the single source of truth: every number
shown by the web application and reported in the paper is read from it
directly, with no hardcoded fallback values in the application code or
templates.

## Citation

*Citation to the associated paper will be added here upon publication.*

## License

*No license has been declared for this repository yet.*
