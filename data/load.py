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


def load_raw(csv_path: Path | str = None, target_col: str = "Churn") -> pd.DataFrame:
    if csv_path is None:
        if not CSV_PATH.exists():
            download_dataset()
        filepath = CSV_PATH
    else:
        filepath = Path(csv_path)
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    df = pd.read_csv(filepath)

    # Clean numeric columns with blank strings if Telco dataset
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Automatically normalize target column into binary (0 / 1)
    if target_col in df.columns:
        unique_vals = df[target_col].dropna().unique()
        if set(unique_vals).issubset({"Yes", "No", "yes", "no"}):
            df[target_col] = df[target_col].astype(str).str.strip().str.capitalize()
            df[target_col] = (df[target_col] == "Yes").astype(int)
        elif set(unique_vals).issubset({"True", "False", true, false} if 'true' in locals() else {True, False}):
            df[target_col] = df[target_col].astype(int)
        else:
            # Map top positive/active status or binary values
            pos_label = unique_vals[0] if len(unique_vals) == 2 else 1
            df[target_col] = (df[target_col] == pos_label).astype(int)

    return df


def get_splits(
    csv_path: Path | str = None,
    target_col: str = "Churn",
    test_size: float = 0.2,
    random_state: int = 42,
):
    df = load_raw(csv_path=csv_path, target_col=target_col)
    
    # Drop identifier columns if present
    id_cols = [c for c in ["customerID", "customer_id", "id", "ID", "user_id", "user_ID"] if c in df.columns]

    if id_cols:
        df = df.drop(columns=id_cols)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

