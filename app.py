import os
import sys
import json
from flask import Flask, render_template, request
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import nn_inference
import rf_inference

app = Flask(__name__)

# Metric artifacts are loaded lazily and cached at module scope so the app
# only pays the loading cost once, on first use. Both models now run on
# NumPy-only exports (models/rf_trees.npz / models/nn_weights.npz) -- no
# scikit-learn or PyTorch in the request path. nn_model.pth and
# random_forest_model.pkl stay in models/ as reference/audit artifacts used
# by the notebooks, but the app no longer reads them.

_metrics_cache = None

_NN_EXPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "nn_weights.npz")
_METRICS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics.json")


def get_metrics():
    """Load and cache metrics.json, the single source of truth for every
    number shown by the templates (Random Forest, neural network, baselines,
    best combinations, example predictions).

    Raises
    ------
    FileNotFoundError
        If metrics.json is missing.
    json.JSONDecodeError
        If metrics.json exists but is not valid JSON.

    Both are handled by the app-wide error handlers below, which log the
    failure and render a visible error page -- this file must never fall
    back to hardcoded numbers.
    """
    global _metrics_cache
    if _metrics_cache is None:
        with open(_METRICS_PATH, "r") as f:
            _metrics_cache = json.load(f)
    return _metrics_cache


def nn_ready():
    """The neural network is available whenever its exported NumPy weights exist.

    Inference no longer depends on PyTorch; nn_inference.py loads and caches
    nn_export.npz on its own.
    """
    return os.path.exists(_NN_EXPORT_PATH)


@app.errorhandler(FileNotFoundError)
@app.errorhandler(json.JSONDecodeError)
def handle_missing_artifact(e):
    """Fail visibly when a required artifact is missing or malformed.

    Silently falling back to stale hardcoded numbers previously caused the
    site to diverge from the published results, so any such failure is
    logged and surfaced to the user instead of being swallowed.
    """
    app.logger.error(f"Required artifact missing or invalid: {e}")
    return render_template("error.html", error=str(e)), 500


# The routes below keep the presentation layer thin and centralize the model
# loading logic in the helper functions defined above.

@app.route("/")
def home():
    return render_template("index.html", nn_available=nn_ready())


@app.route("/application", methods=["GET", "POST"])
def application():
    nn_avail = nn_ready()
    metrics = get_metrics()

    rf_result = None
    nn_result = None
    dmso_value = None
    trehalose_value = None
    error_msg = None
    selected_model = request.form.get("model", "both") if request.method == "POST" else "both"

    if request.method == "POST":
        try:
            dmso_value = float(request.form.get("dmso", 0))
            trehalose_value = float(request.form.get("trehalose", 0))

            if dmso_value < 0 or trehalose_value < 0:
                error_msg = "Concentrations cannot be negative."
            elif dmso_value + trehalose_value > 100:
                error_msg = (
                    f"The sum of DMSO ({dmso_value}%) and "
                    f"trehalose ({trehalose_value}%) exceeds 100%. "
                    f"This has no biological meaning."
                )
            else:
                X = np.array([[dmso_value, trehalose_value]])

                # NumPy-only inference for both models: no scikit-learn, no PyTorch.
                if selected_model in ("rf", "both"):
                    rf_result = round(float(rf_inference.predict(X)[0]), 2)

                # NumPy-only inference: no PyTorch involved in production.
                if selected_model in ("nn", "both") and nn_avail:
                    nn_result = round(nn_inference.predict(dmso_value, trehalose_value), 2)

        except (ValueError, TypeError):
            error_msg = "Invalid input. Please enter valid numbers."

    return render_template(
        "simulator.html",
        rf_result=rf_result,
        nn_result=nn_result,
        nn_available=nn_avail,
        nn_metrics=metrics["neural_network"],
        rf_metrics=metrics["random_forest"],
        dmso=dmso_value,
        trehalose=trehalose_value,
        error=error_msg,
        selected_model=selected_model,
    )


@app.route("/lab-data")
def lab_data():
    metrics = get_metrics()
    return render_template(
        "lab_data.html",
        nn_available=nn_ready(),
        nn_metrics=metrics["neural_network"],
        rf_metrics=metrics["random_forest"],
    )


# Local development entry point. The debug flag is read from the environment so
# production can stay conservative while local testing remains convenient.

if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode)
