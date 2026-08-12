import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from churnsense.features import ChurnFeatureEngineer


RAW_ROW = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No phone service",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No internet service",
    "DeviceProtection": "No internet service",
    "TechSupport": "No internet service",
    "StreamingTV": "No internet service",
    "StreamingMovies": "No internet service",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 75.50,
    "TotalCharges": 377.5,
}


@pytest.fixture
def engineer():
    return ChurnFeatureEngineer()


@pytest.fixture
def single_row_df():
    return pd.DataFrame([RAW_ROW])


@pytest.fixture
def multi_row_df():
    rows = []
    for i in range(20):
        r = RAW_ROW.copy()
        r["tenure"] = i * 5  # starts at 0 (valid — include_lowest handles it)
        r["MonthlyCharges"] = 30 + i * 3.5
        r["TotalCharges"] = r["MonthlyCharges"] * max(r["tenure"], 1)
        rows.append(r)
    return pd.DataFrame(rows)


def test_output_has_no_nans(engineer, multi_row_df):
    out = engineer.transform(multi_row_df)
    assert not out.isnull().any().any(), "NaNs found after feature engineering"


def test_tenure_bucket_ranges(engineer):
    cases = [
        (0, 0), (6, 0), (12, 0),
        (13, 1), (24, 1), (36, 1),
        (37, 2), (50, 2), (60, 2),
        (61, 3), (72, 3),
    ]
    for tenure, expected_bucket in cases:
        row = RAW_ROW.copy()
        row["tenure"] = tenure
        row["TotalCharges"] = row["MonthlyCharges"] * max(tenure, 1)
        df = pd.DataFrame([row])
        out = engineer.transform(df)
        assert out["tenure_bucket"].iloc[0] == expected_bucket, (
            f"tenure={tenure} → expected bucket {expected_bucket}, "
            f"got {out['tenure_bucket'].iloc[0]}"
        )


def test_multi_risk_flag_fires_correctly(engineer):
    # Should fire: month-to-month + electronic check + high charges + short tenure
    high_risk = RAW_ROW.copy()
    high_risk["tenure"] = 6
    high_risk["MonthlyCharges"] = 80.0
    high_risk["TotalCharges"] = 480.0
    df_risk = pd.DataFrame([high_risk])
    out_risk = engineer.transform(df_risk)
    assert out_risk["multi_risk"].iloc[0] == 1

    # Should NOT fire: two-year contract
    low_risk = RAW_ROW.copy()
    low_risk["Contract"] = "Two year"
    low_risk["tenure"] = 6
    df_low = pd.DataFrame([low_risk])
    out_low = engineer.transform(df_low)
    assert out_low["multi_risk"].iloc[0] == 0


def test_num_services_count(engineer, single_row_df):
    out = engineer.transform(single_row_df)
    # RAW_ROW: PhoneService=Yes (counted), MultipleLines=No phone service→No (not counted)
    # All INTERNET_COLS = No internet service → No (not counted)
    # So num_services should be 1 (PhoneService only)
    assert out["num_services"].iloc[0] == 1


def test_charge_per_service_positive(engineer, multi_row_df):
    out = engineer.transform(multi_row_df)
    assert (out["charge_per_service"] > 0).all()


def test_contract_encoding(engineer):
    for contract, expected_code in [("Month-to-month", 0), ("One year", 1), ("Two year", 2)]:
        row = RAW_ROW.copy()
        row["Contract"] = contract
        df = pd.DataFrame([row])
        out = engineer.transform(df)
        assert out["Contract"].iloc[0] == expected_code


def test_no_internet_service_normalisation(engineer):
    row = RAW_ROW.copy()
    row["OnlineSecurity"] = "No internet service"
    df = pd.DataFrame([row])
    out = engineer.transform(df)
    # After normalisation and encoding, should be 0 (No → binary 0)
    assert out["OnlineSecurity"].iloc[0] == 0


def test_output_row_count_matches_input(engineer, multi_row_df):
    out = engineer.transform(multi_row_df)
    assert len(out) == len(multi_row_df)


def test_all_services_active(engineer):
    row = RAW_ROW.copy()
    row["PhoneService"] = "Yes"
    row["MultipleLines"] = "Yes"
    row["OnlineSecurity"] = "Yes"
    row["OnlineBackup"] = "Yes"
    row["DeviceProtection"] = "Yes"
    row["TechSupport"] = "Yes"
    row["StreamingTV"] = "Yes"
    row["StreamingMovies"] = "Yes"
    df = pd.DataFrame([row])
    out = engineer.transform(df)
    # 8 service columns total (2 phone + 6 internet)
    assert out["num_services"].iloc[0] == 8


def test_zero_tenure_bucket(engineer):
    row = RAW_ROW.copy()
    row["tenure"] = 0
    row["TotalCharges"] = 0.0
    df = pd.DataFrame([row])
    out = engineer.transform(df)
    # tenure=0 should land in bucket 0 with include_lowest=True
    assert out["tenure_bucket"].iloc[0] == 0
    assert not out.isnull().any().any(), "tenure=0 should not produce NaNs"
