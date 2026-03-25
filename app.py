from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

rf_model = joblib.load('random_forest_model.pkl')

@app.route('/')
def home():
    # First page: background and disclaimer
    return render_template('index.html')

@app.route('/application', methods=['GET', 'POST'])
def application():
    # Simulator page with AI results
    viability_result = None
    dmso_value = None
    trehalose_value = None

    if request.method == 'POST':
        dmso_value = float(request.form.get('dmso', 0))
        trehalose_value = float(request.form.get('trehalose', 0))
        input_df = pd.DataFrame({'% DMSO': [dmso_value], 'TREHALOSE': [trehalose_value]})
        prediction = rf_model.predict(input_df)[0]
        viability_result = round(prediction, 2)

    return render_template('simulator.html',
                           result=viability_result,
                           dmso=dmso_value,
                           trehalose=trehalose_value)

@app.route('/lab-data')
def lab_data():
    # Third page: description of real experiments and graphics
    return render_template('lab_data.html')

if __name__ == '__main__':
    app.run(debug=True)