import os
from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# ============================================================
# MODEL LOADING
# ============================================================

# Random Forest (always available)
rf_model = joblib.load('random_forest_model.pkl')

# Neural Network (optional — requires PyTorch)
nn_model = None
nn_pipeline = None

try:
    import torch
    from torch import nn

    class HepatoCryoNN(nn.Module):
        def __init__(self, input_dim=5, dropout_rate=0.2):
            super(HepatoCryoNN, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate),
                nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout_rate),
                nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout_rate),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            return self.network(x)

    if os.path.exists('nn_model.pth') and os.path.exists('nn_pipeline.pkl'):
        nn_pipeline = joblib.load('nn_pipeline.pkl')
        nn_model = HepatoCryoNN(input_dim=nn_pipeline['input_dim'])
        nn_model.load_state_dict(torch.load('nn_model.pth', map_location='cpu', weights_only=True))
        nn_model.eval()
        print('[OK] Neural Network model loaded.')
    else:
        print('[INFO] Neural Network model files not found. Running with Random Forest only.')

except ImportError:
    print('[INFO] PyTorch not installed. Running with Random Forest only.')

# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def home():
    return render_template('index.html', nn_available=nn_model is not None)

@app.route('/application', methods=['GET', 'POST'])
def application():
    rf_result = None
    nn_result = None
    dmso_value = None
    trehalose_value = None
    error_msg = None

    if request.method == 'POST':
        try:
            dmso_value = float(request.form.get('dmso', 0))
            trehalose_value = float(request.form.get('trehalose', 0))

            # Input validation
            if dmso_value < 0 or trehalose_value < 0:
                error_msg = 'Concentrations cannot be negative.'
            elif dmso_value + trehalose_value > 100:
                error_msg = (
                    f'The sum of DMSO ({dmso_value}%) and trehalose ({trehalose_value}%) '
                    f'exceeds 100%. This has no biological meaning.'
                )
            else:
                # Random Forest prediction
                input_df = pd.DataFrame({'% DMSO': [dmso_value], 'TREHALOSE': [trehalose_value]})
                rf_result = round(rf_model.predict(input_df)[0], 2)

                # Neural Network prediction (if available)
                if nn_model is not None:
                    import torch
                    entrada = np.array([[dmso_value, trehalose_value]])
                    entrada_poly = nn_pipeline['poly'].transform(entrada)
                    entrada_scaled = nn_pipeline['scaler'].transform(entrada_poly)
                    entrada_tensor = torch.FloatTensor(entrada_scaled)
                    with torch.no_grad():
                        nn_result = round(nn_model(entrada_tensor).item(), 2)

        except (ValueError, TypeError):
            error_msg = 'Invalid input. Please enter valid numbers.'

    nn_metrics = nn_pipeline.get('metrics', {}) if nn_pipeline else {}
    return render_template('simulator.html',
                           rf_result=rf_result,
                           nn_result=nn_result,
                           nn_available=nn_model is not None,
                           nn_metrics=nn_metrics,
                           dmso=dmso_value,
                           trehalose=trehalose_value,
                           error=error_msg)

@app.route('/lab-data')
def lab_data():
    nn_metrics = nn_pipeline.get('metrics', {}) if nn_pipeline else {}
    return render_template('lab_data.html',
                           nn_available=nn_model is not None,
                           nn_metrics=nn_metrics)

if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=debug_mode)
