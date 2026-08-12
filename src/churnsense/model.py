"""
Thin wrapper around the trained sklearn/imblearn pipeline.
Keeps inference logic in one place so the API doesn't need to know
about joblib or SHAP directly.
"""

import json
import joblib
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


class ChurnModel:
    def __init__(self, model_path: Path | None = None, threshold: float = 0.5):
        model_path = model_path or MODELS_DIR / "best_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {model_path}. "
                "Run `python -m churnsense.train` first."
            )
        self.pipeline = joblib.load(model_path)
        self.threshold = threshold
        self._load_metadata()
        self._explainer = None

    def _load_metadata(self):
        meta_path = MODELS_DIR / "metadata.json"
        if meta_path.exists():
            self.metadata = json.loads(meta_path.read_text())
        else:
            self.metadata = {}

    @property
    def model_name(self) -> str:
        return self.metadata.get("winner", "unknown")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        t = threshold if threshold is not None else self.threshold
        return (self.predict_proba(X) >= t).astype(int)

    def predict_with_explanation(
        self, X: pd.DataFrame, threshold: float | None = None
    ) -> list[dict]:
        from churnsense.explain import ChurnExplainer

        if self._explainer is None:
            self._explainer = ChurnExplainer(self.pipeline)

        probs = self.predict_proba(X)
        explanations = self._explainer.explain(X, top_n=3)
        t = threshold if threshold is not None else self.threshold

        results = []
        for i, (prob, expl) in enumerate(zip(probs, explanations)):
            results.append({
                "churn_probability": round(float(prob), 4),
                "churn_predicted": bool(prob >= t),
                "threshold_used": float(t),
                "top_factors": expl["factors"],
                "explanation_summary": expl["summary"],
            })
        return results
