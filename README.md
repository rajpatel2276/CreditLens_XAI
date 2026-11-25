# CreditLens XAI — Explainable Loan Default Predictor

An interpretable credit-risk scoring and underwriting workbench that evaluates loan applications using **class-balanced Logistic Regression**, **imbalance-weighted XGBoost**, **SHAP (SHapley Additive exPlanations)** feature attributions, and an automated **What-If Counterfactual Simulator**.

---

## 1. Problem Statement & Motivation
In financial underwriting, predictive accuracy alone is insufficient—regulators, loan officers, and applicants require transparent explanations for *why* a credit application is approved or flagged for default risk. Furthermore, credit datasets suffer from class imbalance (30% default prevalence vs. 70% non-default), making raw accuracy misleading. **CreditLens XAI** solves both problems by pairing Precision-Recall AUC evaluation with real-time local SHAP explanations and actionable risk-reduction simulations.

---

## 2. Dataset & Feature Engineering
* **Source:** Statlog German Credit Dataset (`fetch_openml("credit-g", version=1)`), comprising **1,000 real loan applications** split 80/20 stratified (**800 training / 200 test samples**) with a **30.0% default prevalence** (`300` defaults, `700` non-defaults).
* **Target Encoding:**
  * `1` (**High Default Risk**): Original label `"bad"`
  * `0` (**Low Default Risk**): Original label `"good"`
* **Preprocessing (`ColumnTransformer`):**
  * **Numeric Features (4 — standardized via `StandardScaler`):** `duration` (months), `credit_amount`, `installment_commitment` (1–4 income burden tier), `age` (years).
  * **Categorical Features (6 — encoded via `OneHotEncoder(handle_unknown="ignore")`)**: `checking_status`, `credit_history`, `purpose`, `savings_status`, `employment`, `housing`.

---

## 3. Model Architecture & Evaluation Benchmark
All metrics are evaluated on the **200 stratified test applications** at a `0.50` decision cutoff:

| Model | Class Imbalance Strategy | PR-AUC (Primary) | ROC-AUC | Default F1 | Avg Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | `class_weight="balanced"` | **0.6816** | **0.8154** | **0.6667** | ~0.05 ms |
| **XGBoost Classifier** | `scale_pos_weight=2.33` | 0.6550 | 0.7856 | 0.6187 | ~0.40 ms |

### Key Engineering & Viva Insights
1. **Why PR-AUC over Accuracy:** Under a 70/30 class split, a dummy classifier predicting "approve" for everyone achieves 70% accuracy while catching 0% of defaults. **Precision-Recall AUC (PR-AUC)** directly measures performance on the minority default class.
2. **Why Logistic Regression Beats XGBoost Here:** On smaller tabular datasets (`1,000` rows) dominated by strong monotonic linear signals (loan amount, duration, liquidity tiers), L2-regularized Logistic Regression generalizes better without overfitting compared to gradient-boosted trees.
3. **SHAP Aggregation:** `shap.TreeExplainer` computes exact log-odds Shapley values per one-hot encoded column; the backend sums dummy-variable SHAP contributions back into the 10 human-readable applicant attributes.

---

## 4. Application Features
* **Live Risk & SHAP Simulator:** Synced sliders and dropdowns with instant auto-recalculation, adjustable bank cutoff threshold (`30%–70%`), model mode selector (`50/50 Hybrid Blend`, `Logistic Regression`, or `XGBoost`), and horizontal SHAP attribution bars showing exact positive (`+` increases risk) and negative (`-` lowers risk) drivers.
* **What-If Counterfactual Simulator:** Automatically tests realistic financial adjustments (shortening loan duration to 18 months, reducing principal by 30%, increasing checking/savings liquidity) and highlights actions that flip a declined application to **Approved**.
* **Batch Portfolio Screener:** Score multiple applicants simultaneously via a built-in 5-applicant portfolio or uploaded `.csv` file, displaying each row's primary SHAP driver with one-click CSV export.

---

## 5. Project Structure
```text
explainable-loan-predictor/
├── artifacts/
│   ├── preprocessor.joblib   # Fitted ColumnTransformer (Scaler + OneHotEncoder)
│   ├── logreg_model.joblib   # Trained Logistic Regression model
│   ├── xgb_model.joblib      # Trained XGBoost model
│   ├── schema.json           # Feature metadata, categories & SHAP baseline
│   └── metrics.json          # Offline evaluation metrics & global SHAP rankings
├── static/
│   └── index.html            # Tailwind CSS + Vanilla JS interactive UI
├── train.py                  # Data ingestion, preprocessing, training & SHAP benchmark
├── server.py                 # FastAPI backend with live SHAP & counterfactual engine
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # Project documentation

---
## 6. Setup & Execution
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models and generate artifacts (optional if artifacts/ already exists)
python train.py

# 4. Start the FastAPI server
uvicorn server:app --host 0.0.0.0 --port 8001
# random2
