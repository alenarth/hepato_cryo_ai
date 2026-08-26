# HepatoCryoAI

A web application that predicts post-thaw viability of cryopreserved HepG2
hepatocytes from the concentrations of two cryoprotectants, DMSO and
trehalose.

## Scientific context

Cryopreservation of human hepatocytes is a bottleneck for cell-based
alternatives to liver transplantation: cells routinely lose viability and
function after thawing, and the choice of cryoprotectant concentrations has
a large, non-linear effect on the outcome. This project is built on 216
experimental observations of HepG2 cell viability across a grid of DMSO and
trehalose concentrations. From that data, it trains two independent
regression models -- a Random Forest and a small neural network -- and
serves both through a web interface so that a given combination of
concentrations can be evaluated before running a wet-lab experiment.

## Models and performance

Both models take only the two raw concentrations, `% DMSO` and `TREHALOSE`,
as input. No polynomial or other hand-engineered features are used.

| Model | R² (test) | RMSE (test) | R² (5-fold CV) |
|---|---|---|---|
| Random Forest | 0.9797 | 5.02 | 0.9602 |
| Neural Network (ANN) | 0.9788 | 5.14 | 0.9680 |
| XGBoost | 0.9773 | 5.32 | 0.9605 |
| Polynomial Regression (deg 2) | 0.6096 | 22.04 | 0.3787 |
| SVR | 0.5441 | 23.82 | 0.2878 |
| Linear Regression | 0.4775 | 25.50 | 0.3399 |

XGBoost, Polynomial Regression, SVR and Linear Regression are included as
baselines / alternative-algorithm comparisons, not as deployed models. All
numbers above are read directly from `metrics.json`.

## Methodology

The 216 samples are split 80/20 with a fixed seed into 172 train+validation
samples and 44 test samples, held out and evaluated exactly once. For the
neural network, the 172 train+validation samples are further split into 129
for fitting and 43 for validation (early stopping and learning-rate
scheduling). Five-fold cross-validation is run over the 172 train+validation
samples for both models. The neural network's regularization configuration
(batch normalization, dropout, weight decay, patience) was selected by a
small grid search scored on validation loss only -- the test set is never
used for model selection.

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
│   ├── raw/hepg2.csv         # the 216 experimental observations
│   ├── comparison_table.csv  # model comparison table
│   └── hyperparameters.csv   # hyperparameters for every model
├── notebooks/                # analysis notebooks, numbered by execution order
├── static/images/            # all figures (single location)
├── templates/                # Flask/Jinja2 HTML templates
└── tests/                    # regression tests
```

## Reproducibility notes

All random seeds are fixed (Python, NumPy, PyTorch, and scikit-learn where
applicable), and training runs on CPU rather than GPU for bit-for-bit
reproducibility. `metrics.json` is the single source of truth: every number
shown by the web application and reported in the paper is read from it
directly, with no hardcoded fallback values in the application code or
templates.

## Citation

*Citation to the associated paper will be added here upon publication.*

## License

*No license has been declared for this repository yet.*
