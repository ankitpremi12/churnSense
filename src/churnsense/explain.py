"""
Plain-language prediction explanations using SHAP.

For each prediction, returns the top 3 contributing factors as human-readable
sentences — referencing actual feature values and what they mean for churn risk.
"""

import numpy as np
import pandas as pd
import shap
from pathlib import Path


# Maps feature names to (high-risk description, low-risk description, value formatter)
EXPLANATION_TEMPLATES = {
    "Contract": {
        0: "month-to-month contract (highest churn risk)",
        1: "one-year contract",
        2: "two-year contract (lowest churn risk)",
        "default": "contract type code {val}",
    },
    "PaymentMethod": {
        0: "electronic check payment (correlated with higher churn)",
        1: "mailed check payment",
        2: "automatic bank transfer",
        3: "automatic credit card payment",
        "default": "payment method {val}",
    },
    "tenure": lambda val: (
        f"only {int(val)} month{'s' if val != 1 else ''} as a customer"
        if val <= 12
        else f"{int(val)} months as a customer"
    ),
    "MonthlyCharges": lambda val: f"monthly charges of ${val:.2f}",
    "TotalCharges": lambda val: f"total lifetime spend of ${val:.2f}",
    "charge_per_service": lambda val: f"${val:.2f} per active service per month",
    "num_services": lambda val: (
        f"only {int(val)} active service{'s' if val != 1 else ''}"
        if val <= 2
        else f"{int(val)} active services"
    ),
    "multi_risk": {
        1: "combination of high-risk factors (month-to-month + electronic check + high charges + short tenure)",
        0: None,
    },
    "new_on_monthly": {
        1: "new customer on a flexible month-to-month contract",
        0: None,
    },
    "InternetService": {
        0: "no internet service",
        1: "DSL internet service",
        2: "fiber optic internet (associated with higher churn rates)",
        "default": "internet service type {val}",
    },
    "SeniorCitizen": {
        1: "senior citizen status",
        0: None,
    },
    "tenure_bucket": {
        0: "very new customer (0–12 months)",
        1: "growing customer (13–36 months)",
        2: "established customer (37–60 months)",
        3: "long-term loyal customer (60+ months)",
        "default": "tenure segment {val}",
    },
    "charge_fulfillment": lambda val: (
        f"low charge fulfillment ratio ({val:.2f}) suggesting recent downgrade or upgrade"
        if val < 0.85 or val > 1.15
        else None
    ),
    "contract_x_payment": lambda val: (
        f"high contract-payment risk score ({val:.1f})" if val > 4 else None
    ),
    "PaperlessBilling": {1: "paperless billing enabled", 0: None},
    "OnlineSecurity": {0: "no online security add-on", 1: "online security enabled"},
    "TechSupport": {0: "no tech support add-on", 1: "tech support enabled"},
}


def _format_feature(name: str, value: float) -> str | None:
    template = EXPLANATION_TEMPLATES.get(name)
    if template is None:
        return None
    if callable(template):
        return template(value)
    key = int(round(value))
    if key in template:
        return template[key]
    return template.get("default", "").format(val=value) or None


class ChurnExplainer:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self._explainer = None
        self._feature_names = None

    def _setup(self, X_sample: pd.DataFrame):
        if self._explainer is not None:
            return

        from churnsense.features import get_feature_names

        try:
            preprocessor = self.pipeline.named_steps.get("preprocessor")
            if preprocessor is None:
                for name, step in self.pipeline.steps:
                    if "preprocessor" in name:
                        preprocessor = step
                        break
            self._feature_names = get_feature_names(preprocessor)
        except Exception:
            self._feature_names = [f"feature_{i}" for i in range(50)]

        classifier = self.pipeline[-1]
        self._explainer = shap.TreeExplainer(classifier)

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        X_t = X.copy()
        for step_name, step in self.pipeline.steps[:-1]:
            if not hasattr(step, "transform"):
                continue
            X_t = step.transform(X_t)
        if hasattr(X_t, "toarray"):
            X_t = X_t.toarray()
        return X_t

    def explain(self, X: pd.DataFrame, top_n: int = 3) -> list[dict]:
        self._setup(X)
        X_t = self._transform(X)
        n_features = X_t.shape[1]
        feat_names = self._feature_names[:n_features]

        shap_values = self._explainer.shap_values(
            pd.DataFrame(X_t, columns=feat_names)
        )
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        explanations = []
        for i in range(len(X)):
            row_shap = shap_values[i]
            row_vals = X_t[i]

            # Sort by absolute SHAP impact — keeps most influential regardless of direction
            idx_sorted = np.argsort(np.abs(row_shap))[::-1]

            factors = []
            for idx in idx_sorted:
                if len(factors) >= top_n:
                    break
                feat = feat_names[idx]
                val = row_vals[idx]
                impact = row_shap[idx]
                description = _format_feature(feat, val)
                if description is None:
                    continue
                direction = "increases" if impact > 0 else "reduces"
                factors.append({
                    "feature": feat,
                    "value": float(val),
                    "shap_impact": float(impact),
                    "direction": direction,
                    "description": description,
                })

            # Compose plain-language summary
            if factors:
                top_desc = factors[0]["description"]
                summary = f"Primary driver: {top_desc}."
                if len(factors) > 1:
                    others = "; ".join(f["description"] for f in factors[1:])
                    summary += f" Also contributing: {others}."
            else:
                summary = "No clear dominant risk factor identified."

            explanations.append({
                "factors": factors,
                "summary": summary,
            })

        return explanations
