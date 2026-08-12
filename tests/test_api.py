import sys
import json
import io
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

# Import app once — module-level so the mock patches the right reference
from api.main import app

MOCK_PREDICTION = {
    "churn_probability": 0.78,
    "churn_predicted": True,
    "threshold_used": 0.5,
    "explanation_summary": "Primary driver: month-to-month contract (highest churn risk).",
    "top_factors": [
        {
            "feature": "Contract",
            "value": 0.0,
            "shap_impact": 0.45,
            "direction": "increases",
            "description": "month-to-month contract (highest churn risk)",
        },
        {
            "feature": "tenure",
            "value": 5.0,
            "shap_impact": 0.22,
            "direction": "increases",
            "description": "only 5 months as a customer",
        },
        {
            "feature": "MonthlyCharges",
            "value": 75.5,
            "shap_impact": 0.18,
            "direction": "increases",
            "description": "monthly charges of $75.50",
        },
    ],
}


@pytest.fixture
def mock_model():
    m = MagicMock()
    m.model_name = "XGBoost"
    m.metadata = {"winner": "XGBoost", "cv_results": {}}
    m.predict_with_explanation.return_value = [MOCK_PREDICTION]
    return m


@pytest.fixture
def client(mock_model):
    with patch("api.main._model", mock_model):
        with TestClient(app) as c:
            yield c


VALID_CUSTOMER = {
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


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "model_name" in data


def test_predict_valid_customer(client):
    resp = client.post("/predict", json=VALID_CUSTOMER)
    assert resp.status_code == 200
    data = resp.json()
    assert "churn_probability" in data
    assert "churn_predicted" in data
    assert "top_factors" in data
    assert len(data["top_factors"]) == 3
    assert "explanation_summary" in data


def test_predict_probability_bounds(client):
    resp = client.post("/predict", json=VALID_CUSTOMER)
    assert resp.status_code == 200
    prob = resp.json()["churn_probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_top_factors_have_required_fields(client):
    resp = client.post("/predict", json=VALID_CUSTOMER)
    for factor in resp.json()["top_factors"]:
        assert "feature" in factor
        assert "description" in factor
        assert "shap_impact" in factor
        assert "direction" in factor
        assert factor["direction"] in ("increases", "reduces")


def test_predict_invalid_contract(client):
    bad = {**VALID_CUSTOMER, "Contract": "Weekly"}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_invalid_gender(client):
    bad = {**VALID_CUSTOMER, "gender": "Unknown"}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_invalid_payment_method(client):
    bad = {**VALID_CUSTOMER, "PaymentMethod": "Bitcoin"}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_negative_tenure(client):
    bad = {**VALID_CUSTOMER, "tenure": -1}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_batch_json(mock_model):
    mock_model.predict_with_explanation.return_value = [MOCK_PREDICTION, MOCK_PREDICTION]
    with patch("api.main._model", mock_model):
        with TestClient(app) as c:
            resp = c.post("/predict/batch", json=[VALID_CUSTOMER, VALID_CUSTOMER])
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert "prediction" in data[0]


def test_predict_batch_single_row(mock_model):
    mock_model.predict_with_explanation.return_value = [MOCK_PREDICTION]
    with patch("api.main._model", mock_model):
        with TestClient(app) as c:
            resp = c.post("/predict/batch", json=[VALID_CUSTOMER])
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["prediction"]["churn_probability"] == MOCK_PREDICTION["churn_probability"]


def test_predict_batch_csv_response(mock_model):
    mock_model.predict_with_explanation.return_value = [MOCK_PREDICTION]
    csv_content = ",".join(VALID_CUSTOMER.keys()) + "\n"
    csv_content += ",".join(str(v) for v in VALID_CUSTOMER.values()) + "\n"

    with patch("api.main._model", mock_model):
        with TestClient(app) as c:
            resp = c.post(
                "/predict/batch?response_format=csv",
                files={"file": ("test.csv", csv_content.encode(), "text/csv")},
            )
    assert resp.status_code == 200
    assert "text/csv" in resp.headers["content-type"]


def test_predict_batch_empty_body(client):
    resp = client.post("/predict/batch")
    assert resp.status_code == 400


def test_predict_custom_threshold(client):
    resp = client.post("/predict?threshold=0.3", json=VALID_CUSTOMER)
    assert resp.status_code == 200
    data = resp.json()
    assert data["threshold_used"] == 0.3
