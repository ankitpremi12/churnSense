"""
Training pipeline for ChurnSense.

Runs Optuna hyperparameter search for XGBoost, LightGBM, and Random Forest,
compares them on stratified 5-fold CV AUC-ROC, then retrains the winner on the
full training set and evaluates on the held-out test set.

Usage:
    python -m churnsense.train
    python -m churnsense.train --no-smote   # skip SMOTE, use class weighting only
"""

import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import joblib
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Suppress LightGBM/XGBoost verbosity during Optuna trials
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Resolve project root regardless of where the script is called from
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.load import get_splits
from churnsense.features import build_pipeline, ChurnFeatureEngineer

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
RANDOM_STATE = 42


# ── Optuna objective functions ──────────────────────────────────────────────

def xgb_objective(trial: optuna.Trial, X_train, y_train, use_smote: bool) -> float:
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
        "scale_pos_weight": 1.0 if use_smote else scale_pos,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
    }
    clf = XGBClassifier(**params)

    if use_smote:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        from churnsense.features import ChurnFeatureEngineer, build_preprocessor
        pipe = ImbPipeline([
            ("engineer", ChurnFeatureEngineer()),
            ("preprocessor", build_preprocessor()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", clf),
        ])
    else:
        pipe = build_pipeline(clf)

    scores = cross_val_score(pipe, X_train, y_train, cv=CV, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def lgbm_objective(trial: optuna.Trial, X_train, y_train, use_smote: bool) -> float:
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "is_unbalance": not use_smote,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }
    clf = LGBMClassifier(**params)
    pipe = build_pipeline(clf)
    scores = cross_val_score(pipe, X_train, y_train, cv=CV, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def rf_objective(trial: optuna.Trial, X_train, y_train, use_smote: bool) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    clf = RandomForestClassifier(**params)
    pipe = build_pipeline(clf)
    scores = cross_val_score(pipe, X_train, y_train, cv=CV, scoring="roc_auc", n_jobs=-1)
    return scores.mean()



# ── Optuna study runner ─────────────────────────────────────────────────────

def run_study(name: str, objective_fn, X_train, y_train, n_trials: int, use_smote: bool):
    print(f"\n{'─' * 55}")
    print(f"  Tuning {name} ({n_trials} trials, {'SMOTE' if use_smote else 'class weighting'})")
    print(f"{'─' * 55}")

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    study = optuna.create_study(direction="maximize", pruner=pruner)

    def callback(study, trial):
        val_str = f"{trial.value:.4f}" if trial.value is not None else "pruned"
        print(
            f"  [{name}] Trial {trial.number:>3} | "
            f"AUC {val_str} | "
            f"Best so far: {study.best_value:.4f}"
        )

    study.optimize(
        lambda t: objective_fn(t, X_train, y_train, use_smote),
        n_trials=n_trials,
        callbacks=[callback],
        show_progress_bar=False,
    )

    print(f"  {name} best AUC (CV): {study.best_value:.4f}")
    return study.best_params, study.best_value


# ── SMOTE comparison helper ─────────────────────────────────────────────────

def _compare_imbalance_strategies(X_train, y_train) -> bool:
    """
    Runs a quick 3-fold comparison: SMOTE vs class-weight-only on XGBoost
    with default-ish params. Returns True if SMOTE wins.
    """
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    # Class weighting
    clf_weighted = XGBClassifier(
        n_estimators=300, scale_pos_weight=scale_pos,
        random_state=RANDOM_STATE, verbosity=0, n_jobs=-1
    )
    pipe_w = build_pipeline(clf_weighted)
    score_weighted = cross_val_score(pipe_w, X_train, y_train, cv=cv3, scoring="f1").mean()

    # SMOTE — needs imblearn pipeline to apply within each fold
    from churnsense.features import ChurnFeatureEngineer, build_preprocessor
    clf_smote = XGBClassifier(
        n_estimators=300, random_state=RANDOM_STATE, verbosity=0, n_jobs=-1
    )
    smote_pipe = ImbPipeline([
        ("engineer", ChurnFeatureEngineer()),
        ("preprocessor", build_preprocessor()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("classifier", clf_smote),
    ])
    score_smote = cross_val_score(smote_pipe, X_train, y_train, cv=cv3, scoring="f1").mean()

    print(f"\n  Imbalance strategy comparison (3-fold F1):")
    print(f"    Class weighting : {score_weighted:.4f}")
    print(f"    SMOTE           : {score_smote:.4f}")
    use_smote = score_smote > score_weighted
    print(f"  → Using {'SMOTE' if use_smote else 'class weighting'}")
    return use_smote


# ── Build final model from best params ─────────────────────────────────────

def build_best_model(model_name: str, best_params: dict[str, Any], use_smote: bool, X_sample: pd.DataFrame = None):
    is_generic = X_sample is not None and "Contract" not in X_sample.columns

    if model_name == "XGBoost":
        clf = XGBClassifier(**{**best_params, "eval_metric": "logloss",
                               "random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": 0})
    elif model_name == "LightGBM":
        clf = LGBMClassifier(**{**best_params, "random_state": RANDOM_STATE,
                                "n_jobs": -1, "verbose": -1})
    else:
        clf = RandomForestClassifier(**{**best_params, "random_state": RANDOM_STATE, "n_jobs": -1})

    if use_smote:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        from churnsense.features import ChurnFeatureEngineer, build_preprocessor, build_generic_preprocessor
        
        preproc = build_generic_preprocessor(X_sample) if is_generic else build_preprocessor()
        steps = []
        if not is_generic:
            steps.append(("engineer", ChurnFeatureEngineer()))
        steps.extend([
            ("preprocessor", preproc),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", clf),
        ])
        pipe = ImbPipeline(steps)
    else:
        if is_generic:
            pipe = build_generic_pipeline(clf, X_sample)
        else:
            pipe = build_pipeline(clf)

    return pipe


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train ChurnSense models")
    parser.add_argument("--no-smote", action="store_true", help="Force-disable SMOTE oversampling")
    parser.add_argument("--n-trials", type=int, default=30, help="Optuna trials per model")
    parser.add_argument("--csv", type=str, default=None, help="Path to custom CSV dataset")
    parser.add_argument("--target", type=str, default="Churn", help="Target column name")
    args = parser.parse_args()

    print(f"\n{'=' * 55}")
    print(f"  ChurnSense Model Training Pipeline")
    if args.csv:
        print(f"  Dataset: {args.csv} (target: {args.target})")
    else:
        print(f"  Dataset: Default IBM Telco Customer Churn")
    print(f"{'=' * 55}\n")

    X_train, X_test, y_train, y_test = get_splits(csv_path=args.csv, target_col=args.target)
    print(f"Train set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")
    print(f"Features : {X_train.shape[1]} columns")

    is_generic = "Contract" not in X_train.columns

    # Determine imbalance strategy
    if args.no_smote:
        use_smote = False
        print("SMOTE disabled via --no-smote flag.")
    else:
        print("Evaluating class imbalance strategy...")
        use_smote = _compare_imbalance_strategies(X_train, y_train)

    # Optuna study loop
    results = {}
    models_to_tune = [
        ("XGBoost", xgb_objective),
        ("LightGBM", lgbm_objective),
        ("RandomForest", rf_objective),
    ]

    for name, obj in models_to_tune:
        print(f"\nTuning {name} ({args.n_trials} trials)...")
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study.optimize(
            lambda t: obj(t, X_train, y_train, use_smote),
            n_trials=args.n_trials,
            show_progress_bar=False,
        )
        print(f"  Best {name} CV AUC: {study.best_value:.4f}")
        results[name] = {"params": study.best_params, "cv_auc": study.best_value}

    # Pick winner
    winner = max(results, key=lambda k: results[k]["cv_auc"])
    print(f"\n{'=' * 55}")
    print(f"  Model comparison (CV AUC-ROC):")
    for name, r in results.items():
        marker = " <-- winner" if name == winner else ""
        print(f"    {name:<15} {r['cv_auc']:.4f}{marker}")
    print(f"{'=' * 55}\n")

    # Retrain winner on full training set
    print(f"Retraining {winner} on full training set...")
    final_pipe = build_best_model(winner, results[winner]["params"], use_smote, X_sample=X_train)
    final_pipe.fit(X_train, y_train)

    # Save
    model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(final_pipe, model_path)
    print(f"Model saved to {model_path}")

    metadata = {
        "winner": winner,
        "use_smote": use_smote,
        "is_generic": is_generic,
        "target_col": args.target,
        "feature_names": list(X_train.columns),
        "cv_results": {k: {"cv_auc": v["cv_auc"]} for k, v in results.items()},
        "best_params": results[winner]["params"],
    }
    meta_path = MODELS_DIR / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"Metadata saved to {meta_path}")

    # Evaluate on test set
    print("\nRunning evaluation on test set...")
    from churnsense.evaluate import run_evaluation
    run_evaluation(final_pipe, X_test, y_test, winner, metadata)

    print("\nDone. Check reports/ for plots and metrics.")


if __name__ == "__main__":
    main()
