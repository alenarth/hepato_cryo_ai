import os
import json
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# The trained models are loaded lazily so the web app can start even when the
# optional neural-network assets are not available in the current environment.

rf_model = None
nn_model = None
nn_pipeline = None
_nn_load_attempted = False
_rf_metrics_cache = None


def get_rf():
    """Load and cache the trained Random Forest regressor.

    Returns
    -------
    object
        The fitted scikit-learn model used for viability predictions.
    """
    global rf_model
    if rf_model is None:
        rf_model = joblib.load("random_forest_model.pkl")
    return rf_model


def get_rf_metrics():
    """Return the Random Forest metrics used by the web templates.

    The metrics are read from the JSON file produced by the analysis notebook when
    available. If that artifact is missing, a small fallback dictionary preserves
    a stable view of the last known results for local testing.

    Returns
    -------
    dict
        Test metrics, cross-validation summaries, and the best concentration
        combinations reported in the analysis notebook.
    """
    global _rf_metrics_cache
    if _rf_metrics_cache is None:
        metrics_path = os.path.join(os.path.dirname(__file__), "rf_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                _rf_metrics_cache = json.load(f)
        else:
            # Preserve a stable fallback view when the metrics file is absent.
            _rf_metrics_cache = {
                "r2_test": 0.9797,
                "rmse_test": 5.0247,
                "cv_r2_mean": 0.9602,
                "best_combinations": {
                    "global":    {"dmso": 3.0, "trehalose": 1.0, "viability": 95.19},
                    "dmso_only": {"dmso": 2.0, "trehalose": 0.0, "viability": 95.19},
                    "treh_only": {"dmso": 0.0, "trehalose": 16.0, "viability": 78.48},
                },
            }
    return _rf_metrics_cache


def get_nn():
    """Load the optional neural-network model and preprocessing pipeline.

    The app keeps working even when PyTorch or the serialized model files are not
    available. In that case it returns ``(None, None)`` and the UI falls back to
    the Random Forest model.

    Returns
    -------
    tuple
        A pair ``(model, pipeline)`` when the neural network is available, or
        ``(None, None)`` otherwise.
    """
    global nn_model, nn_pipeline, _nn_load_attempted

    if _nn_load_attempted:
        return nn_model, nn_pipeline

    _nn_load_attempted = True

    try:
        import torch
        from torch import nn as tnn

        class HepatoCryoNN(tnn.Module):
            def __init__(self, input_dim=5, dropout_rate=0.2):
                super().__init__()
                self.network = tnn.Sequential(
                    tnn.Linear(input_dim, 128), tnn.BatchNorm1d(128), tnn.ReLU(), tnn.Dropout(dropout_rate),
                    tnn.Linear(128, 64),  tnn.BatchNorm1d(64),  tnn.ReLU(), tnn.Dropout(dropout_rate),
                    tnn.Linear(64, 32),   tnn.BatchNorm1d(32),  tnn.ReLU(), tnn.Dropout(dropout_rate),
                    tnn.Linear(32, 1)
                )

            def forward(self, x):
                return self.network(x)

        model_path = os.path.join(os.path.dirname(__file__), "nn_model.pth")
        pipe_path = os.path.join(os.path.dirname(__file__), "nn_pipeline.pkl")

        if os.path.exists(model_path) and os.path.exists(pipe_path):
            nn_pipeline = joblib.load(pipe_path)
            nn_model = HepatoCryoNN(input_dim=nn_pipeline["input_dim"])
            nn_model.load_state_dict(
                torch.load(model_path, map_location="cpu", weights_only=True)
            )
            nn_model.eval()
            print("[OK] Neural Network loaded.")
        else:
            print("[INFO] NN model files not found. RF only.")

    except ImportError:
        print("[INFO] PyTorch not installed. RF only.")
    except Exception as e:
        print(f"[WARN] NN load failed: {e}. RF only.")

    return nn_model, nn_pipeline


# The routes below keep the presentation layer thin and centralize the model
# loading logic in the helper functions defined above.

@app.route("/")
def home():
    model, _ = get_nn()
    return render_template("index.html", nn_available=model is not None)


@app.route("/application", methods=["GET", "POST"])
def application():
    nn_mod, nn_pipe = get_nn()
    nn_avail = nn_mod is not None

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

                # The Random Forest expects the same two-feature input used in the
                # analysis notebook.
                if selected_model in ("rf", "both"):
                    rf_result = round(float(get_rf().predict(X)[0]), 2)

                # The neural-network pathway uses the same preprocessing pipeline
                # that was fitted during training, including polynomial expansion
                # and standardization.
                if selected_model in ("nn", "both") and nn_avail:
                    import torch
                    entrada_poly = nn_pipe["poly"].transform(X)
                    entrada_scaled = nn_pipe["scaler"].transform(entrada_poly)
                    entrada_tensor = torch.FloatTensor(entrada_scaled)
                    with torch.no_grad():
                        nn_result = round(nn_mod(entrada_tensor).item(), 2)

        except (ValueError, TypeError):
            error_msg = "Invalid input. Please enter valid numbers."
        except Exception as e:
            error_msg = f"Prediction error: {str(e)}"

    nn_metrics = nn_pipe.get("metrics", {}) if nn_pipe else {}

    return render_template(
        "simulator.html",
        rf_result=rf_result,
        nn_result=nn_result,
        nn_available=nn_avail,
        nn_metrics=nn_metrics,
        rf_metrics=get_rf_metrics(),
        dmso=dmso_value,
        trehalose=trehalose_value,
        error=error_msg,
        selected_model=selected_model,
    )
    


@app.route("/lab-data")
def lab_data():
    nn_mod, nn_pipe = get_nn()
    nn_metrics = nn_pipe.get("metrics", {}) if nn_pipe else {}
    return render_template(
        "lab_data.html",
        nn_available=nn_mod is not None,
        nn_metrics=nn_metrics,
        rf_metrics=get_rf_metrics(),
    )


# Local development entry point. The debug flag is read from the environment so
# production can stay conservative while local testing remains convenient.

if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode)