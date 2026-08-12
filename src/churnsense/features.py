import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


# Simple Yes/No binary columns (no special sentinel values)
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

# Service columns with sentinel strings — normalised to "No" before binary encoding.
# These also feed into num_services count.
INTERNET_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
# PhoneService and MultipleLines both use "No phone service" sentinel
PHONE_COLS = ["PhoneService", "MultipleLines"]

CONTRACT_ORDER = ["Month-to-month", "One year", "Two year"]
PAYMENT_ORDER = [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)",
]
INTERNET_SERVICE_ORDER = ["No", "DSL", "Fiber optic"]


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Stateless transformer — all derived features are computed from raw columns
    with no learned state, so fit() is a no-op. Keeps the sklearn pipeline
    interface so it slots cleanly into cross-validation.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # --- Normalise sentinel strings to "No" ---
        for col in INTERNET_COLS:
            df[col] = df[col].replace("No internet service", "No")
        for col in PHONE_COLS:
            df[col] = df[col].replace("No phone service", "No")

        # --- Simple binary columns (no sentinel values) ---
        for col, mapping in BINARY_MAP.items():
            df[col] = df[col].map(mapping)

        # --- Service binary flags (used for counting and as features) ---
        # Includes phone cols so PhoneService and MultipleLines are counted
        service_flag_cols = INTERNET_COLS + PHONE_COLS
        for col in service_flag_cols:
            df[col] = (df[col] == "Yes").astype(int)

        # --- Ordinal encodes ---
        df["Contract"] = pd.Categorical(
            df["Contract"], categories=CONTRACT_ORDER, ordered=True
        ).codes  # 0, 1, 2
        df["PaymentMethod"] = pd.Categorical(
            df["PaymentMethod"], categories=PAYMENT_ORDER, ordered=True
        ).codes
        df["InternetService"] = pd.Categorical(
            df["InternetService"], categories=INTERNET_SERVICE_ORDER, ordered=True
        ).codes

        # --- Derived features ---
        # include_lowest=True makes the first bin [0, 12] (closed on left),
        # so tenure=0 (new customers who haven't been billed) lands in bucket 0.
        df["tenure_bucket"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 36, 60, np.inf],
            labels=[0, 1, 2, 3],
            right=True,
            include_lowest=True,
        ).astype(int)

        df["num_services"] = df[service_flag_cols].sum(axis=1)

        # Charge per active service — avoids div-by-zero
        df["charge_per_service"] = df["MonthlyCharges"] / (df["num_services"] + 1)

        # How much of the expected lifetime spend has actually been collected
        # (low value = new or recently downgraded customer)
        expected = df["MonthlyCharges"] * df["tenure"]
        df["charge_fulfillment"] = df["TotalCharges"] / expected.replace(0, np.nan)
        df["charge_fulfillment"] = df["charge_fulfillment"].fillna(1.0).clip(0, 2)

        # Interaction: contract risk × payment risk
        # High contract code = longer contract (lower risk); lower payment code = riskier
        # So risk_score is high when contract is short and payment is via electronic check
        df["contract_x_payment"] = (2 - df["Contract"]) * (3 - df["PaymentMethod"])

        # Tenure × contract interaction — new customers on flexible contracts churn most
        df["new_on_monthly"] = (
            (df["tenure"] <= 12) & (df["Contract"] == 0)
        ).astype(int)

        # Stacked risk flag: month-to-month + electronic check + high charges + low tenure
        high_charge_threshold = 65.0  # ~median MonthlyCharges for churners
        df["multi_risk"] = (
            (df["Contract"] == 0)
            & (df["PaymentMethod"] == 0)
            & (df["MonthlyCharges"] > high_charge_threshold)
            & (df["tenure"] <= 12)
        ).astype(int)

        # Senior citizen is already binary int in the raw dataset
        return df


def build_preprocessor() -> ColumnTransformer:
    """
    Applies StandardScaler to continuous columns. Categorical/binary columns
    are already numeric after ChurnFeatureEngineer, so they pass through.
    """
    continuous = [
        "tenure", "MonthlyCharges", "TotalCharges",
        "charge_per_service", "charge_fulfillment", "contract_x_payment",
    ]
    passthrough = [
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
        "PaperlessBilling", "Contract", "PaymentMethod",
        "tenure_bucket", "num_services", "new_on_monthly", "multi_risk",
    ]
    return ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), continuous),
            ("pass", "passthrough", passthrough),
        ]
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    continuous = preprocessor.transformers_[0][2]
    passthrough = preprocessor.transformers_[1][2]
    return list(continuous) + list(passthrough)


def build_pipeline(classifier) -> Pipeline:
    return Pipeline([
        ("engineer", ChurnFeatureEngineer()),
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier),
    ])


def build_generic_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    cat_cols = list(X.select_dtypes(exclude=[np.number]).columns)

    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder

    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", num_transformer, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_transformer, cat_cols))

    return ColumnTransformer(transformers=transformers)


def build_generic_pipeline(classifier, X_sample: pd.DataFrame) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_generic_preprocessor(X_sample)),
        ("classifier", classifier),
    ])

