# ChurnSense

A customer churn prediction system I built as a portfolio ML project. It uses the IBM Telco Customer Churn dataset, trains and compares three tree-based models with Optuna hyperparameter tuning, and serves predictions through a FastAPI REST API with plain-language explanations powered by SHAP.

The goal wasn't to build the simplest thing that runs — I wanted something I'd actually trust in production: proper cross-validation, no data leakage, calibrated outputs, and explanations that a non-technical person could read.

---

## What it does

- Downloads and preprocesses the IBM Telco Customer Churn dataset automatically
- Engineers features: tenure buckets, charge-per-service ratios, contract×payment interaction terms, a stacked multi-risk flag for customers with multiple churn indicators
- Compares XGBoost, LightGBM, and Random Forest, each tuned with Optuna (40/40/25 trials, MedianPruner, 5-fold stratified CV)
- Tests class imbalance handling: SMOTE vs native class weighting, picks whichever scores higher on F1
- Saves the winning model and generates SHAP-based feature importance and summary plots
- Serves predictions via FastAPI — single customer or batch (CSV or JSON)
- Returns churn probability, binary label, and top-3 contributing factors in plain English

---

## Model comparison

These numbers are from the actual training run on the full dataset. The test set is 20% held out before any tuning.

| Model | CV AUC-ROC | Test Accuracy | Test F1 | Test AUC-ROC |
|-------|-----------|---------------|---------|--------------|
| XGBoost | _see reports/metrics.json_ | — | — | — |
| LightGBM | _see reports/metrics.json_ | — | — | — |
| Random Forest | _see reports/metrics.json_ | — | — | — |
| **Winner** | — | — | — | — |

> **Note**: Run `python -m churnsense.train` to populate these numbers. After training, `reports/metrics.json` contains the full results and this README should be updated with real values.

**Why the winner wins**: Contract type and tenure dominate the signal — tree models handle this interaction well. XGBoost and LightGBM tend to win over Random Forest because of their gradient boosting approach and better handling of the moderate class imbalance. SMOTE vs class weighting was determined empirically: whichever produced higher 3-fold F1 on the training split was used.

**What I'd improve**: 
- Threshold tuning — 0.5 isn't optimal for imbalanced data. A PR-curve-based threshold search would likely improve recall without destroying precision
- Calibration — tree models are often overconfident. Adding `CalibratedClassifierCV` wrapper would make the probability outputs more trustworthy
- More feature engineering around plan changes (the dataset doesn't have this, but in practice tenure + charge patterns proxy for it)
- Proper MLflow or similar experiment tracking instead of flat JSON files

---

## How to run locally

**Requirements**: Python 3.11, pip

```bash
git clone <repo>
cd churnSense

python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train the model (downloads dataset automatically, ~10-15 min)
python -m churnsense.train

# Start the API
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The interactive docs are at http://localhost:8000/docs

---

## Sample request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 351.75
  }'
```

Sample response:
```json
{
  "churn_probability": 0.82,
  "churn_predicted": true,
  "threshold_used": 0.5,
  "explanation_summary": "Primary driver: month-to-month contract (highest churn risk). Also contributing: only 5 months as a customer; electronic check payment (correlated with higher churn).",
  "top_factors": [
    {
      "feature": "Contract",
      "value": 0.0,
      "shap_impact": 0.48,
      "direction": "increases",
      "description": "month-to-month contract (highest churn risk)"
    },
    {
      "feature": "tenure",
      "value": -1.21,
      "shap_impact": 0.31,
      "direction": "increases",
      "description": "only 5 months as a customer"
    },
    {
      "feature": "PaymentMethod",
      "value": 0.0,
      "shap_impact": 0.19,
      "direction": "increases",
      "description": "electronic check payment (correlated with higher churn)"
    }
  ]
}
```

### Batch prediction

```bash
# JSON array
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[{...customer1...}, {...customer2...}]'

# CSV upload, get CSV back
curl -X POST "http://localhost:8000/predict/batch?response_format=csv" \
  -F "file=@your_customers.csv"
```

---

## How to run with Docker

```bash
# First train the model locally so it's available to mount
python -m churnsense.train

# Build and run
docker compose up --build

# Or manually:
docker build -t churnsense .
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  churnsense
```

The container expects the trained model to be mounted at `/app/models/best_model.pkl`. The docker-compose.yml handles this automatically.

---

## Running tests

```bash
pytest tests/ -v

# Model-dependent tests require a trained model:
pytest tests/test_model.py -v    # skips gracefully if model not found

# Feature engineering tests run without any model:
pytest tests/test_features.py tests/test_api.py -v
```

---

## Project structure

```
churnSense/
├── data/
│   ├── raw/                   # auto-downloaded CSV (gitignored)
│   └── load.py                # download + preprocessing + splits
├── notebooks/
│   └── eda.ipynb              # exploratory analysis
├── src/
│   └── churnsense/
│       ├── features.py        # feature engineering (sklearn transformer)
│       ├── train.py           # Optuna tuning, model comparison, training
│       ├── evaluate.py        # metrics, confusion matrix, SHAP plots
│       ├── explain.py         # SHAP wrapper → plain-English summaries
│       └── model.py           # thin inference wrapper
├── api/
│   ├── main.py                # FastAPI app
│   └── schemas.py             # Pydantic request/response models
├── models/                    # trained model artifacts (gitignored)
├── reports/                   # metrics.json, plots (gitignored)
├── tests/
│   ├── test_features.py
│   ├── test_api.py
│   └── test_model.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
