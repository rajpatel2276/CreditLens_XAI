import json
import os
import time
import joblib
import numpy as np
import pandas as pd
import sklearn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import xgboost as xgb
import shap

app = FastAPI(title="CreditLens XAI")

# Load trained models and metadata
preprocessor = joblib.load("artifacts/preprocessor.joblib")
logreg_model = joblib.load("artifacts/logreg_model.joblib")
xgb_model = joblib.load("artifacts/xgb_model.joblib")
explainer = shap.TreeExplainer(xgb_model)

with open("artifacts/schema.json", "r") as f:
    schema_data = json.load(f)

with open("artifacts/metrics.json", "r") as f:
    metrics_data = json.load(f)

ENCODED_NAMES = list(preprocessor.get_feature_names_out())
ALL_FEATURES = schema_data["num_features"] + schema_data["cat_features"]
FEATURE_LABELS = schema_data["feature_labels"]

# Readable labels for the dataset categories
READABLE_LABELS = {
    "checking_status": {
        "<0": "Overdrawn (< $0)",
        "0<=X<200": "Low Balance ($0 – $200)",
        ">=200": "Healthy Balance ($200+)",
        "no checking": "No Checking Account",
    },
    "credit_history": {
        "no credits/all paid": "No Prior Loans / All Paid",
        "all paid": "All Existing Loans Paid on Time",
        "existing paid": "Current Loans Paid on Time",
        "delayed previously": "Past Payment Delays",
        "critical/other existing credit": "Other Active Credit Accounts",
    },
    "purpose": {
        "new car": "New Car",
        "used car": "Used Car",
        "furniture/equipment": "Furniture / Equipment",
        "radio/tv": "Electronics / Appliances",
        "domestic appliance": "Home Appliance",
        "repairs": "Home Repairs",
        "education": "Education",
        "retraining": "Job Training",
        "business": "Small Business",
        "other": "Other Purpose",
    },
    "savings_status": {
        "<100": "Low Savings (< $100)",
        "100<=X<500": "Moderate Savings ($100 – $500)",
        "500<=X<1000": "Good Savings ($500 – $1,000)",
        ">=1000": "High Savings ($1,000+)",
        "no known savings": "No Savings Account",
    },
    "employment": {
        "unemployed": "Unemployed",
        "<1": "Less than 1 Year",
        "1<=X<4": "1 to 4 Years",
        "4<=X<7": "4 to 7 Years",
        ">=7": "7+ Years",
    },
    "housing": {
        "rent": "Renting",
        "own": "Own House",
        "for free": "Living Rent-Free",
    },
}


class LoanApplication(BaseModel):
    duration: int = 24
    credit_amount: int = 3500
    installment_commitment: int = 3
    age: int = 32
    checking_status: str = "0<=X<200"
    credit_history: str = "existing paid"
    purpose: str = "new car"
    savings_status: str = "<100"
    employment: str = "1<=X<4"
    housing: str = "own"
    threshold: float = 0.50


def format_display_value(feat: str, val):
    if feat in READABLE_LABELS:
        return READABLE_LABELS[feat].get(str(val), str(val))
    if feat == "duration":
        return f"{val} months"
    if feat == "credit_amount":
        return f"${int(val):,}"
    if feat == "installment_commitment":
        return f"Level {val} of 4"
    if feat == "age":
        return f"{val} years"
    return str(val)


def get_improvement_tips(app_dict: dict, current_risk: float, threshold: float):
    """Test simple changes to show how the applicant can reduce default risk."""
    tests = []
    if app_dict["duration"] > 18:
        tests.append(("Shorten loan duration to 18 months", {**app_dict, "duration": 18}))
    if app_dict["credit_amount"] > 2500:
        smaller = max(1000, int(round(app_dict["credit_amount"] * 0.7, -2)))
        tests.append((f"Lower loan amount to ${smaller:,}", {**app_dict, "credit_amount": smaller}))
    if app_dict["checking_status"] != ">=200":
        tests.append(("Maintain $200+ checking balance", {**app_dict, "checking_status": ">=200"}))
    if app_dict["savings_status"] in ["<100", "no known savings"]:
        tests.append(("Build $500–$1,000 in savings", {**app_dict, "savings_status": "500<=X<1000"}))

    if not tests:
        return []

    df_test = pd.DataFrame([t[1] for t in tests])[ALL_FEATURES]
    X_test = preprocessor.transform(df_test)
    new_probs = xgb_model.predict_proba(X_test)[:, 1]

    suggestions = []
    for (label, _), new_p in zip(tests, new_probs):
        new_p = float(new_p)
        if new_p < current_risk - 0.01:
            suggestions.append({
                "action": label,
                "new_risk_pct": round(new_p * 100, 1),
                "approved_now": bool(new_p < threshold),
            })
    return sorted(suggestions, key=lambda x: x["new_risk_pct"])[:3]


@app.get("/api/meta")
def get_meta():
    return {
        "schema": schema_data,
        "code_labels": READABLE_LABELS,
        "metrics": metrics_data,
    }


@app.post("/api/predict")
def predict_loan(req: LoanApplication):
    app_dict = req.model_dump(exclude={"threshold"})
    df = pd.DataFrame([app_dict])[ALL_FEATURES]
    X_proc = preprocessor.transform(df)

    # 1. Predictions from both models
    lr_prob = float(logreg_model.predict_proba(X_proc)[0][1])
    xgb_prob = float(xgb_model.predict_proba(X_proc)[0][1])

    # 2. SHAP explanation for XGBoost (scaled to intuitive impact points)
    raw_shap = explainer.shap_values(X_proc)[0]
    feature_impacts = {feat: 0.0 for feat in ALL_FEATURES}
    for idx, enc_name in enumerate(ENCODED_NAMES):
        col_part = enc_name.split("__", 1)[1]
        matched = next((f for f in ALL_FEATURES if col_part == f or col_part.startswith(f + "_")), None)
        if matched:
            feature_impacts[matched] += float(raw_shap[idx])

    contributions = []
    for feat in ALL_FEATURES:
        # Scale SHAP value by 100 so beginners see clean "+24 pts" instead of "+0.2413 log-odds"
        pts = round(feature_impacts[feat] * 100, 1)
        contributions.append({
            "label": FEATURE_LABELS[feat],
            "display_value": format_display_value(feat, app_dict[feat]),
            "impact_pts": pts,
        })
    contributions.sort(key=lambda x: abs(x["impact_pts"]), reverse=True)

    is_high_risk = xgb_prob >= req.threshold
    suggestions = get_improvement_tips(app_dict, xgb_prob, req.threshold)

    return {
        "decision": "High Risk — Decline / Review" if is_high_risk else "Low Risk — Approve Loan",
        "is_high_risk": bool(is_high_risk),
        "xgb_risk_pct": round(xgb_prob * 100, 1),
        "logreg_risk_pct": round(lr_prob * 100, 1),
        "summary": (
            f"Predicted default risk ({xgb_prob*100:.1f}%) is above the {int(req.threshold*100)}% limit. "
            "Check the red bars below to see which factors raised the risk."
            if is_high_risk
            else f"Predicted default risk ({xgb_prob*100:.1f}%) is safely below the {int(req.threshold*100)}% limit."
        ),
        "contributions": contributions,
        "suggestions": suggestions,
    }


os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def read_index():
    return FileResponse("static/index.html")