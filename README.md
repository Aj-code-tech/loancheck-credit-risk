# LoanCheck — Explainable Credit Risk Underwriting Console

**LoanCheck** is an end-to-end machine learning underwriting application that evaluates borrower default probability in real time and explains individual risk decisions using **SHAP (SHapley Additive exPlanations)** feature attributions.

Built with **FastAPI**, **XGBoost**, **Scikit-learn**, and an industrial dark-slate underwriting console, the system pairs calibrated probability estimation with policy threshold controls and per-applicant risk factor diagnostics.

---

## Key Capabilities

- **Calibrated Default Risk Scoring:** Trains and compares regularized Logistic Regression and hyperparameter-tuned XGBoost pipelines on 32,400+ deduplicated borrower records (`ROC-AUC: 0.94`, `F1: 0.82`).
- **Policy Cutoff Thresholding:** Evaluates borrower risk against a configurable underwriting cutoff (`38.0%`) and classifies applications across four risk tiers (`Low Risk`, `Moderate Risk`, `High Risk`, `Critical Risk`).
- **Real-Time SHAP Feature Attribution:** Computes per-applicant SHAP contributions on every `/predict` call to surface the top 5 borrower features pushing default risk higher (`+`) or lower (`−`).
- **Institutional Underwriting Console:** Responsive split-pane interface featuring tabular numeric readouts, live loan-to-income ratio calculation, one-click borrower scenario presets (`Prime Case`, `Borderline`, `High Risk`), and API health telemetry.

---

## Tech Stack

- **Machine Learning & Explainability:** Python, Scikit-learn, XGBoost, SHAP, Pandas, NumPy, Joblib
- **Backend API:** FastAPI, Uvicorn, Pydantic v2
- **Frontend Console:** Semantic HTML5, Industrial Slate CSS3, Vanilla JavaScript (Fetch API)

---

## Project Structure

```text
loancheck-credit-risk/
├── main.py                  # FastAPI server, Pydantic schemas, SHAP explainer & /predict endpoint
├── Credit_Risk.ipynb        # EDA, preprocessing pipeline, model benchmarking & calibration
├── credit_risk_dataset.csv  # Borrower credit dataset (32,581 records)
├── credit_risk_model.pkl    # Serialized Scikit-learn + XGBoost calibrated pipeline
├── best_threshold.pkl       # Decision threshold artifact
├── requirements.txt         # Python dependencies
├── render.yaml              # Render deployment blueprint
└── static/
    ├── index.html           # Underwriting console layout
    ├── style.css            # Industrial dark slate UI styling
    └── script.js            # Preset loader, threshold gauge & SHAP chart renderer