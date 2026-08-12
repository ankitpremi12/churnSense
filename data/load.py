import os
import requests
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parent / "raw"
CSV_PATH = DATA_DIR / "Telco-Customer-Churn.csv"
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)


def download_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {DATASET_URL} ...")
    try:
        resp = requests.get(DATASET_URL, timeout=30)
        resp.raise_for_status()
        CSV_PATH.write_bytes(resp.content)
        print(f"Saved to {CSV_PATH}")
    except requests.RequestException as e:
        raise RuntimeError(
            f"Failed to download dataset: {e}\n\n"
            "Please manually place Telco-Customer-Churn.csv in data/raw/ and retry."
        ) from e


def load_raw() -> pd.DataFrame:
    if not CSV_PATH.exists():
        download_dataset()

    df = pd.read_csv(CSV_PATH)

    # TotalCharges has a handful of blank strings for customers with 0 tenure
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    return df


def get_splits(test_size: float = 0.2, random_state: int = 42):
    df = load_raw()
    # Drop customerID — not a feature
    df = df.drop(columns=["customerID"])

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
