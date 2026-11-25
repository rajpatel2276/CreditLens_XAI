import json
import os
import time
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb
import shap

os.makedirs("artifacts", exist_ok=True)

# 10 high-signal financial & applicant features (1-to-1 with UI controls)
NUM_FEATURES = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "age",
]

CAT_FEATURES = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "housing",
]

ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

FEATURE_LABELS = {
    "duration": "Loan Duration (Months)",
    "credit_amount": "Loan Amount",
    "installment_commitment": "Installment Rate (% of Income)",
    "age": "Applicant Age (Years)",
    "checking_status": "Checking Account Status",
    "credit_history": "Credit History",
    "purpose": "Loan Purpose",
    "savings_status": "Savings Account Balance",
    "employment": "Employment Tenure",
    "housing": "Housing Status",
}


def load_credit_data():
    print("Fetching Statlog German Credit dataset (credit-g) from OpenML...")
    data = fetch_openml("credit-g", version=1, as_frame=True)
    df = data.frame.copy()

    X = df[ALL_FEATURES].copy()
    for col in NUM_FEATURES:
        X[col] = pd.to_numeric(X[col])
    for col in CAT_FEATURES:
        X[col] = X[col].astype(str)

    # Target: 1 = High Default Risk ("bad"), 0 = Low Default Risk ("good")
    y = (df["class"].astype(str) == "bad").astype(int).values
    print(f"Loaded {len(X)} applications | Default Rate: {y.mean()*100:.1f}% ({y.sum()} defaults)")
    return X, y


def evaluate_model(name, y_true, y_prob, infer_ms):
    y_pred = (y_prob >= 0.50).astype(int)
    return {
        "model": name,
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "f1_default": round(float(f1_score(y_true, y_pred)), 4),
        "recall_default": round(float(recall_score(y_true, y_pred)), 4),
        "precision_default": round(float(precision_score(y_true, y_pred)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "latency_ms": round(float(infer_ms), 3),
    }


def main():
    X, y = load_credit_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
        ]
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    encoded_names = list(preprocessor.get_feature_names_out())

    # ---------------------------------------------------------
    # MODEL A: Logistic Regression Baseline (Class-Balanced)
    # ---------------------------------------------------------
    print("\n[1/2] Training Logistic Regression baseline...")
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    logreg.fit(X_train_proc, y_train)

    t0 = time.perf_counter()
    lr_probs = logreg.predict_proba(X_test_proc)[:, 1]
    lr_ms = ((time.perf_counter() - t0) / len(X_test)) * 1000
    lr_metrics = evaluate_model("Logistic Regression", y_test, lr_probs, lr_ms)
    print(
        f"LogReg  -> PR-AUC: {lr_metrics['pr_auc']:.4f} | ROC-AUC: {lr_metrics['roc_auc']:.4f} | "
        f"Default F1: {lr_metrics['f1_default']:.4f}"
    )

    # ---------------------------------------------------------
    # MODEL B: XGBoost Classifier (Imbalance-Weighted)
    # ---------------------------------------------------------
    print("\n[2/2] Training XGBoost Classifier with scale_pos_weight...")
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    imbalance_ratio = neg_count / pos_count

    xgb_clf = xgb.XGBClassifier(
        n_estimators=160,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=imbalance_ratio,
        eval_metric="aucpr",
        random_state=42,
    )
    xgb_clf.fit(X_train_proc, y_train)

    t0 = time.perf_counter()
    xgb_probs = xgb_clf.predict_proba(X_test_proc)[:, 1]
    xgb_ms = ((time.perf_counter() - t0) / len(X_test)) * 1000
    xgb_metrics = evaluate_model("XGBoost", y_test, xgb_probs, xgb_ms)
    print(
        f"XGBoost -> PR-AUC: {xgb_metrics['pr_auc']:.4f} | ROC-AUC: {xgb_metrics['roc_auc']:.4f} | "
        f"Default F1: {xgb_metrics['f1_default']:.4f}"
    )

    # ---------------------------------------------------------
    # Compute Global SHAP Feature Importance on Test Set
    # ---------------------------------------------------------
    explainer = shap.TreeExplainer(xgb_clf)
    shap_values = explainer.shap_values(X_test_proc)

    # Aggregate absolute SHAP values back to the 10 original features
    global_imp = {feat: 0.0 for feat in ALL_FEATURES}
    for idx, enc_name in enumerate(encoded_names):
        raw_col = enc_name.split("__", 1)[1]
        matched_feat = next((f for f in ALL_FEATURES if raw_col == f or raw_col.startswith(f + "_")), None)
        if matched_feat:
            global_imp[matched_feat] += float(np.mean(np.abs(shap_values[:, idx])))

    sorted_global = sorted(
        [{"feature": k, "label": FEATURE_LABELS[k], "mean_abs_shap": round(v, 4)} for k, v in global_imp.items()],
        key=lambda x: x["mean_abs_shap"],
        reverse=True,
    )

    # Save schema & valid dropdown categories for the frontend
    schema = {
        "num_features": NUM_FEATURES,
        "cat_features": CAT_FEATURES,
        "feature_labels": FEATURE_LABELS,
        "categories": {col: sorted(X[col].unique().tolist()) for col in CAT_FEATURES},
        "num_ranges": {
            col: {
                "min": int(X[col].min()),
                "max": int(X[col].max()),
                "median": int(X[col].median()),
            }
            for col in NUM_FEATURES
        },
        "base_value": round(float(explainer.expected_value), 4),
    }

    joblib.dump(preprocessor, "artifacts/preprocessor.joblib")
    joblib.dump(logreg, "artifacts/logreg_model.joblib")
    joblib.dump(xgb_clf, "artifacts/xgb_model.joblib")

    with open("artifacts/schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    with open("artifacts/metrics.json", "w") as f:
        json.dump(
            {
                "train_size": len(X_train),
                "test_size": len(X_test),
                "default_rate_pct": round(float(y.mean() * 100), 1),
                "scale_pos_weight": round(float(imbalance_ratio), 2),
                "logreg": lr_metrics,
                "xgboost": xgb_metrics,
                "global_shap": sorted_global,
            },
            f,
            indent=2,
        )

    print("\nSaved models, schema, and SHAP benchmark artifacts to ./artifacts/ successfully!")


if __name__ == "__main__":
    main()