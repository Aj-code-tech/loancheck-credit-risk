from contextlib import asynccontextmanager
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "credit_risk_model.pkl"
THRESHOLD_PATH = BASE_DIR / "best_threshold.pkl"

artifacts = {}

FEATURE_LABELS = {
    "person_age": "Applicant Age",
    "person_income": "Annual Income",
    "person_home_ownership": "Home Ownership",
    "person_emp_length": "Work Experience",
    "loan_intent": "Loan Purpose",
    "loan_grade": "Loan Grade",
    "loan_amnt": "Loan Amount",
    "loan_int_rate": "Interest Rate",
    "loan_percent_income": "Loan-to-Income Ratio",
    "cb_person_default_on_file": "Prior Default Record",
    "cb_person_cred_hist_length": "Credit History Length",
}


def build_shap_explainer(model_pipeline):
    """Extract underlying tree estimator and preprocessor for real-time SHAP attributions."""
    try:
        estimator = model_pipeline
        if hasattr(estimator, "calibrated_classifiers_"):
            estimator = estimator.calibrated_classifiers_[0].estimator

        if hasattr(estimator, "named_steps"):
            steps = list(estimator.named_steps.values())
            preprocessor = steps[0] if len(steps) > 1 else None
            classifier = steps[-1]
            explainer = shap.TreeExplainer(classifier)
            return preprocessor, explainer
    except Exception:
        pass
    return None, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = joblib.load(MODEL_PATH)
    artifacts["model"] = model
    artifacts["threshold"] = float(joblib.load(THRESHOLD_PATH))
    preprocessor, explainer = build_shap_explainer(model)
    artifacts["preprocessor"] = preprocessor
    artifacts["explainer"] = explainer
    yield
    artifacts.clear()


app = FastAPI(
    title="LoanCheck Underwriting Engine",
    description="Institutional credit default probability & SHAP attribution API",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoanApplication(BaseModel):
    person_age: int = Field(..., ge=18, le=100)
    person_income: float = Field(..., gt=0)
    person_home_ownership: str
    person_emp_length: float = Field(..., ge=0, le=60)
    loan_intent: str
    loan_grade: str
    loan_amnt: float = Field(..., gt=0)
    loan_int_rate: float = Field(..., gt=0, le=40)
    loan_percent_income: float = Field(..., ge=0, le=1.0)
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int = Field(..., ge=0, le=60)


def compute_feature_impacts(applicant_df: pd.DataFrame) -> list[dict]:
    preprocessor = artifacts.get("preprocessor")
    explainer = artifacts.get("explainer")
    if preprocessor is None or explainer is None:
        return []

    try:
        transformed = preprocessor.transform(applicant_df)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        shap_vals = explainer.shap_values(transformed)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[-1]
        if shap_vals.ndim == 3:
            shap_vals = shap_vals[0, :, -1]
        else:
            shap_vals = shap_vals[0]

        feature_names = preprocessor.get_feature_names_out()
        aggregated = {label: 0.0 for label in FEATURE_LABELS.values()}

        for raw_name, val in zip(feature_names, shap_vals):
            clean_name = raw_name.split("__", 1)[-1]
            matched_label = None
            for col_key, label in FEATURE_LABELS.items():
                if clean_name == col_key or clean_name.startswith(f"{col_key}_"):
                    matched_label = label
                    break
            if matched_label:
                aggregated[matched_label] += float(val)

        ranked = sorted(aggregated.items(), key=lambda item: abs(item[1]), reverse=True)[:5]
        return [{"feature": feat, "impact": round(val, 4)} for feat, val in ranked]
    except Exception:
        return []


def classify_risk_band(prob: float, threshold: float) -> str:
    if prob >= max(threshold + 0.25, 0.70):
        return "Critical Risk"
    if prob >= threshold:
        return "High Risk"
    if prob >= threshold * 0.65:
        return "Moderate Risk"
    return "Low Risk"


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": "model" in artifacts,
        "decision_threshold": artifacts.get("threshold"),
    }


@app.post("/predict")
def assess_loan_risk(application: LoanApplication):
    if "model" not in artifacts:
        raise HTTPException(status_code=503, detail="Risk model is not initialized.")

    applicant_df = pd.DataFrame([application.model_dump()])
    raw_prob = artifacts["model"].predict_proba(applicant_df)[:, 1][0]

    default_prob = round(float(raw_prob), 4)
    threshold = round(float(artifacts["threshold"]), 4)
    is_default = int(default_prob >= threshold)
    risk_band = classify_risk_band(default_prob, threshold)
    shap_impacts = compute_feature_impacts(applicant_df)

    return {
        "default_probability": default_prob,
        "default_prediction": is_default,
        "threshold": threshold,
        "risk_tier": risk_band,
        "Result": "High Risk" if is_default == 1 else "Low Risk",
        "shap_impacts": shap_impacts,
    }


app.mount("/", StaticFiles(directory=str(BASE_DIR / "static"), html=True), name="static")