"""
Unit & integration tests for AutoML generic dataset support.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from data.load import get_splits, load_raw
from churnsense.features import build_generic_pipeline
from xgboost import XGBClassifier


@pytest.fixture
def dummy_csv(tmp_path):
    csv_file = tmp_path / "dummy_dataset.csv"
    df = pd.DataFrame({
        "user_id": [f"U{i}" for i in range(100)],
        "age": np.random.randint(18, 70, size=100),
        "income": np.random.uniform(20000, 120000, size=100),
        "tier": np.random.choice(["basic", "pro", "enterprise"], size=100),
        "is_churned": np.random.choice(["Yes", "No"], size=100),
    })
    df.to_csv(csv_file, index=False)
    return csv_file


def test_generic_load_and_splits(dummy_csv):
    df = load_raw(csv_path=dummy_csv, target_col="is_churned")
    assert "is_churned" in df.columns
    assert set(df["is_churned"].unique()).issubset({0, 1})

    X_train, X_test, y_train, y_test = get_splits(csv_path=dummy_csv, target_col="is_churned")
    assert "user_id" not in X_train.columns
    assert len(X_train) + len(X_test) == 100
    assert X_train.shape[1] == 3


def test_generic_pipeline_fit_predict(dummy_csv):
    X_train, X_test, y_train, y_test = get_splits(csv_path=dummy_csv, target_col="is_churned")
    
    clf = XGBClassifier(n_estimators=10, random_state=42)
    pipeline = build_generic_pipeline(clf, X_train)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    assert len(preds) == len(X_test)
    assert len(probs) == len(X_test)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
