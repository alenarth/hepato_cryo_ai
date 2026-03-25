# HepatoCryoAI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple)

**HepatoCryoAI** is a predictive web application and machine‑learning model designed to optimize cryopreservation protocols for human hepatocytes. Developed as a scientific research initiative, the tool acts as an intelligent calculator that estimates post‑thaw cell viability based on varying concentrations of cryoprotectants.

---

## Scientific background

Chronic liver diseases (CLD) are a major public‑health issue, and cirrhosis is a leading cause of hospitalization and mortality. Although liver transplantation is the only definitive therapy, the severe shortage of organs has driven interest in cell‑based alternatives using human hepatocytes.

The primary limitation of this approach is loss of viability and functionality after thawing. **HepatoCryoAI** addresses this bottleneck with a Random‑Forest regressor trained on *in vitro* HepG2 data, predicting survival rates when using dimethyl sulfoxide (DMSO) and trehalose as cryoprotectants.

---

## Key features

1. **Machine‑learning prediction** – Highly accurate Random‑Forest model (R² ≈ 0.98) for viability estimation.
2. **Interactive web simulator** – User‑friendly interface for researchers to enter cryoprotectant parameters and receive instant predictions.
3. **Data visualization** – Heatmaps and scatter plots that compare computational output with laboratory results.
4. **Laboratory data section** – Displays foundational experiments (Trypan‑blue and MTT assays).

---

## 🛠 Technology stack

* **Backend & modeling:** Python, pandas, NumPy, scikit‑learn
* **Web framework:** Flask
* **Frontend:** HTML5, CSS3, Bootstrap 5
* **Model persistence:** Joblib

---

## Running locally

Steps to set up the environment and run the application:

```bash
# clone the repository
git clone https://github.com/alenarth/HepatoCryoAI.git
cd HepatoCryoAI

# create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
# source venv/Scripts/activate

# install dependencies
pip install flask pandas numpy scikit-learn joblib

# start the Flask app
python app.py
```

Open a browser and visit <http://127.0.0.1:5000> to access the simulator.

---

## ⚠️ Disclaimer & liability

HepatoCryoAI is provided solely as a complementary research tool. Simulation results **must not** replace rigorous *in vitro* and *in vivo* laboratory validation. Users accept full responsibility for decisions made based on this software.

---

## Acknowledgments

This project contributes to a wider initiative for creating hepatocyte banks and expanding cell therapy as an alternative to liver transplantation, developed in accordance with institutional research guidelines.

---
