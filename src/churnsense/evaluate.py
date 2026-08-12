"""
Evaluation: computes all metrics on the test set, saves confusion matrix,
feature importance plot, and SHAP summary plot.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import shap
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)

PROJECT_ROOT = Path(__file__).parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No churn", "Churn"],
        yticklabels=["No churn", "Churn"],
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13)
    fig.tight_layout()
    path = REPORTS_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_feature_importance(pipeline, feature_names: list[str], model_name: str):
    """Works for tree-based models that expose feature_importances_."""
    classifier = pipeline.named_steps.get("classifier")
    if classifier is None:
        # imblearn pipeline — different step name
        classifier = pipeline[-1]

    if not hasattr(classifier, "feature_importances_"):
        print("  (skipping feature importance — not available for this model type)")
        return

    importances = classifier.feature_importances_
    n = min(len(feature_names), len(importances))
    pairs = sorted(zip(feature_names[:n], importances[:n]), key=lambda x: x[1], reverse=True)
    names, vals = zip(*pairs[:20])

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(names)))
    ax.barh(list(reversed(names)), list(reversed(vals)), color=list(reversed(colors)))
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(f"Feature Importance — {model_name}", fontsize=13)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    fig.tight_layout()
    path = REPORTS_DIR / "feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_shap_summary(pipeline, X_test: pd.DataFrame, feature_names: list[str]):
    """Computes SHAP values on the transformed test set and saves summary plot."""
    # Walk pipeline steps manually so we skip SMOTE (no transform at inference)
    # and handle both sklearn and imblearn pipelines uniformly.
    X_t = X_test.copy()
    for step_name, step in pipeline.steps[:-1]:
        if not hasattr(step, "transform"):
            # SMOTE and similar resamplers don't have transform—skip
            continue
        X_t = step.transform(X_t)

    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()

    X_df = pd.DataFrame(X_t, columns=feature_names[:X_t.shape[1]])

    classifier = pipeline[-1]
    try:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_df)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    except Exception as e:
        print(f"  SHAP computation failed: {e}")
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    shap.summary_plot(shap_values, X_df, show=False, max_display=20)
    plt.title("SHAP Feature Impact — Churn Prediction", fontsize=13)
    plt.tight_layout()
    path = REPORTS_DIR / "shap_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def run_evaluation(pipeline, X_test, y_test, model_name: str, metadata: dict):
    from churnsense.features import get_feature_names, build_preprocessor

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = compute_metrics(y_test, y_pred, y_prob)

    print("\n  Test set metrics:")
    print(f"    Accuracy  : {metrics['accuracy']:.4f}")
    print(f"    Precision : {metrics['precision']:.4f}")
    print(f"    Recall    : {metrics['recall']:.4f}")
    print(f"    F1        : {metrics['f1']:.4f}")
    print(f"    AUC-ROC   : {metrics['roc_auc']:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["No churn", "Churn"]))

    # Save metrics JSON
    full_results = {**metadata, "test_metrics": metrics}
    metrics_path = REPORTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(full_results, indent=2))
    print(f"  Saved {metrics_path}")

    # Get feature names after preprocessing
    try:
        # Try to get feature names from the preprocessor step
        preprocessor = pipeline.named_steps.get("preprocessor")
        if preprocessor is None:
            # imblearn pipeline
            for name, step in pipeline.steps:
                if "preprocessor" in name:
                    preprocessor = step
                    break
        feature_names = get_feature_names(preprocessor) if preprocessor else []
    except Exception:
        feature_names = [f"feature_{i}" for i in range(50)]

    plot_confusion_matrix(y_test, y_pred, model_name)
    plot_feature_importance(pipeline, feature_names, model_name)
    plot_shap_summary(pipeline, X_test, feature_names)

    return metrics


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parents[3]))
    sys.path.insert(0, str(Path(__file__).parents[3] / "src"))
    from data.load import get_splits

    pipeline = joblib.load(Path(__file__).parents[3] / "models" / "best_model.pkl")
    meta = json.loads((Path(__file__).parents[3] / "models" / "metadata.json").read_text())
    _, X_test, _, y_test = get_splits()
    run_evaluation(pipeline, X_test, y_test, meta["winner"], meta)
