"""
Model sanity tests. These load the real trained model from models/best_model.pkl
and verify it behaves correctly on the test split.

Skip with: pytest tests/test_model.py -m "not requires_model"
Or run all: pytest tests/test_model.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

MODEL_PATH = Path(__file__).parents[1] / "models" / "best_model.pkl"
requires_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Trained model not found — run python -m churnsense.train first",
)

SAMPLE_CUSTOMER = {
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
    "MonthlyCharges": 75.50,
    "TotalCharges": 377.50,
}

LOW_RISK_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 72,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Credit card (automatic)",
    "MonthlyCharges": 65.00,
    "TotalCharges": 4680.00,
}


@pytest.fixture(scope="module")
def model():
    from churnsense.model import ChurnModel
    return ChurnModel()


@requires_model
def test_probability_in_valid_range(model):
    df = pd.DataFrame([SAMPLE_CUSTOMER])
    probs = model.predict_proba(df)
    assert len(probs) == 1
    assert 0.0 <= probs[0] <= 1.0


@requires_model
def test_high_risk_scores_higher_than_low_risk(model):
    df_high = pd.DataFrame([SAMPLE_CUSTOMER])
    df_low = pd.DataFrame([LOW_RISK_CUSTOMER])
    prob_high = model.predict_proba(df_high)[0]
    prob_low = model.predict_proba(df_low)[0]
    assert prob_high > prob_low, (
        f"Expected high-risk customer ({prob_high:.3f}) to score higher "
        f"than low-risk ({prob_low:.3f})"
    )


@requires_model
def test_batch_probabilities_all_valid(model):
    df = pd.DataFrame([SAMPLE_CUSTOMER] * 50)
    probs = model.predict_proba(df)
    assert len(probs) == 50
    assert np.all((probs >= 0.0) & (probs <= 1.0))


@requires_model
def test_predictions_deterministic(model):
    df = pd.DataFrame([SAMPLE_CUSTOMER])
    prob1 = model.predict_proba(df)[0]
    prob2 = model.predict_proba(df)[0]
    assert prob1 == prob2


@requires_model
def test_explanation_returns_three_factors(model):
    df = pd.DataFrame([SAMPLE_CUSTOMER])
    results = model.predict_with_explanation(df)
    assert len(results) == 1
    assert len(results[0]["top_factors"]) <= 3
    assert len(results[0]["top_factors"]) >= 1


@requires_model
def test_explanation_has_non_empty_summary(model):
    df = pd.DataFrame([SAMPLE_CUSTOMER])
    results = model.predict_with_explanation(df)
    assert isinstance(results[0]["explanation_summary"], str)
    assert len(results[0]["explanation_summary"]) > 10


@requires_model
def test_threshold_respected(model):
    df = pd.DataFrame([SAMPLE_CUSTOMER])
    # With threshold=0.0, everyone should be predicted as churn
    pred_low = model.predict(df, threshold=0.0)
    assert pred_low[0] == 1

    # With threshold=1.0, nobody should be predicted as churn
    pred_high = model.predict(df, threshold=1.0)
    assert pred_high[0] == 0


@requires_model
def test_test_set_auc_above_baseline(model):
    from data.load import get_splits
    from sklearn.metrics import roc_auc_score

    _, X_test, _, y_test = get_splits()
    probs = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, probs)
    assert auc > 0.80, f"AUC {auc:.4f} is below 0.80 — model may not have trained correctly"
