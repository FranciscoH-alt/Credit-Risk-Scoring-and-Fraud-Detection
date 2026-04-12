"""
src/train_model.py — Model Training (DuckDB)

Reads engineered_features from DuckDB, trains XGBoost with Optuna tuning,
computes SHAP values, and writes predictions back to DuckDB.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from src.db import get_conn

optuna.logging.set_verbosity(optuna.logging.WARNING)

_ROOT        = Path(__file__).parent.parent
_MODELS_DIR  = _ROOT / "models"
_EXPORTS_DIR = _ROOT / "exports"
_VISUALS_DIR = _ROOT / "visuals"

FRAUD_RATE       = 0.035
SCALE_POS_WEIGHT = (1 - FRAUD_RATE) / FRAUD_RATE  # ≈ 27.6

_DROP_COLS    = ["TransactionID", "isFraud", "card4_amt_bucket"]
_TARGET       = "isFraud"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_features(conn) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    print("  Loading engineered_features from DuckDB...")
    df = conn.execute("SELECT * FROM engineered_features").df()
    print(f"  {len(df):,} rows × {df.shape[1]} columns")

    tx_ids = df["TransactionID"].copy()
    y = df[_TARGET].astype(int)

    drop = [c for c in _DROP_COLS if c in df.columns]
    X = df.drop(columns=drop)

    # Drop any remaining non-numeric columns (safety net)
    obj_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if obj_cols:
        print(f"  Dropping {len(obj_cols)} string columns: {obj_cols[:5]}")
        X = X.drop(columns=obj_cols)

    print(f"  Feature matrix: {X.shape[0]:,} × {X.shape[1]}  |  fraud rate: {y.mean()*100:.2f}%")
    return X, y, tx_ids


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def _objective(trial, X_tr, y_tr):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": SCALE_POS_WEIGHT,
        "tree_method":      "hist",
        "eval_metric":      "auc",
        "random_state":     42,
        "n_jobs":           -1,
        "verbosity":        0,
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for tr_idx, val_idx in cv.split(X_tr, y_tr):
        m = XGBClassifier(**params, early_stopping_rounds=50)
        m.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx],
              eval_set=[(X_tr.iloc[val_idx], y_tr.iloc[val_idx])],
              verbose=False)
        scores.append(roc_auc_score(y_tr.iloc[val_idx],
                                     m.predict_proba(X_tr.iloc[val_idx])[:, 1]))
    return float(np.mean(scores))


def run_optuna_study(X_tr, y_tr, n_trials=50) -> dict:
    print(f"  Optuna: {n_trials} trials × 3-fold CV")
    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    t0 = time.time()
    study.optimize(lambda t: _objective(t, X_tr, y_tr),
                   n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    print(f"  Best CV AUC: {study.best_value:.4f}  ({(time.time()-t0)/60:.1f} min)")

    _EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(_EXPORTS_DIR / "optuna_trials.csv", index=False)
    return study.best_params


# ---------------------------------------------------------------------------
# Final model
# ---------------------------------------------------------------------------

def train_final_model(X_tr, y_tr, X_te, y_te, best_params) -> tuple:
    print("  Training final model...")
    params = {**best_params, "scale_pos_weight": SCALE_POS_WEIGHT,
              "tree_method": "hist", "random_state": 42, "n_jobs": -1}
    model = XGBClassifier(**params)
    model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_te)[:, 1]
    preds = (proba >= 0.5).astype(int)

    auc  = roc_auc_score(y_te, proba)
    ap   = average_precision_score(y_te, proba)
    rpt  = classification_report(y_te, preds, output_dict=True)

    metrics = {
        "auc_roc":          round(auc, 4),
        "avg_precision":    round(ap, 4),
        "precision_fraud":  round(rpt.get("1", {}).get("precision", 0), 4),
        "recall_fraud":     round(rpt.get("1", {}).get("recall", 0), 4),
        "f1_fraud":         round(rpt.get("1", {}).get("f1-score", 0), 4),
        "precision_legit":  round(rpt.get("0", {}).get("precision", 0), 4),
        "recall_legit":     round(rpt.get("0", {}).get("recall", 0), 4),
        "f1_legit":         round(rpt.get("0", {}).get("f1-score", 0), 4),
        "support_fraud":    int(rpt.get("1", {}).get("support", 0)),
        "support_legit":    int(rpt.get("0", {}).get("support", 0)),
        "confusion_matrix": confusion_matrix(y_te, preds).tolist(),
        "best_params":      best_params,
    }

    print(f"\n  === Evaluation ===")
    print(f"  AUC-ROC:       {auc:.4f}  (target ≥ 0.88)")
    print(f"  Avg Precision: {ap:.4f}")
    print(f"  F1 (fraud):    {metrics['f1_fraud']:.4f}")

    if auc < 0.80:
        print("  WARNING: AUC < 0.80. Check V-column threshold and encoding.")
    elif auc < 0.85:
        print("  NOTE: AUC 0.80-0.85. Verify card1_addr1_count feature is present.")
    elif auc < 0.88:
        print("  NOTE: AUC 0.85-0.88. Try --optuna-trials 100 to improve.")

    return model, metrics, proba


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def compute_shap_values(model, X_te, n_sample=5000) -> pd.DataFrame:
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _VISUALS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Computing SHAP (sample={n_sample:,})...")
    sample = X_te.sample(n=min(n_sample, len(X_te)), random_state=42)
    explainer   = shap.TreeExplainer(model)
    shap_vals   = explainer.shap_values(sample)

    mean_abs = np.abs(shap_vals).mean(axis=0)
    imp_df = pd.DataFrame({
        "feature":       X_te.columns.tolist(),
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    imp_df.head(20).to_csv(_EXPORTS_DIR / "feature_importance_shap.csv", index=False)

    shap.summary_plot(shap_vals, sample, plot_type="bar", max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(_VISUALS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Top feature: {imp_df.iloc[0]['feature']}  SHAP={imp_df.iloc[0]['mean_abs_shap']:.4f}")
    return imp_df


# ---------------------------------------------------------------------------
# Persist + write predictions
# ---------------------------------------------------------------------------

def save_model(model) -> None:
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    p = _MODELS_DIR / "fraud_model.pkl"
    joblib.dump(model, p)
    print(f"  Model saved: {p}")


def save_metrics(metrics: dict) -> None:
    _EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_EXPORTS_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    rows = [{"metric_name": k, "value": v}
            for k, v in metrics.items()
            if isinstance(v, (int, float))]
    pd.DataFrame(rows).to_csv(_EXPORTS_DIR / "model_performance_metrics.csv", index=False)
    print(f"  Metrics saved.")


def write_predictions(conn, tx_ids, y_te, proba) -> None:
    pred_df = pd.DataFrame({
        "TransactionID":   tx_ids.values,
        "isFraud":         y_te.values,
        "fraud_prob":      proba.astype("float32"),
        "predicted_label": (proba >= 0.5).astype("int8"),
        "model_version":   "xgb_v1",
    })
    conn.execute("DROP TABLE IF EXISTS model_predictions")
    conn.execute("CREATE TABLE model_predictions AS SELECT * FROM pred_df")
    print(f"  {len(pred_df):,} predictions written to model_predictions")


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------

def run_training(conn=None, use_smote=False, n_optuna_trials=50) -> dict:
    close = conn is None
    if conn is None:
        conn = get_conn()

    print("\n=== Step 4: Model Training ===")
    t0 = time.time()

    X, y, tx_ids = load_features(conn)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    _, tx_te = train_test_split(tx_ids, test_size=0.20, stratify=y, random_state=42)

    X_tr = X_tr.reset_index(drop=True)
    X_te = X_te.reset_index(drop=True)
    y_tr = y_tr.reset_index(drop=True)
    y_te = y_te.reset_index(drop=True)
    tx_te = tx_te.reset_index(drop=True)

    print(f"  Train: {len(X_tr):,}  |  Test: {len(X_te):,}")

    if use_smote:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(sampling_strategy=0.1, random_state=42, n_jobs=-1)
        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
        print(f"  After SMOTE: {len(X_tr):,} rows")

    best_params = run_optuna_study(X_tr, y_tr, n_trials=n_optuna_trials)
    model, metrics, proba = train_final_model(X_tr, y_tr, X_te, y_te, best_params)

    imp_df = compute_shap_values(model, X_te)
    print("\n  Top 5 features:")
    for _, row in imp_df.head(5).iterrows():
        print(f"    {row['feature']:40s}  {row['mean_abs_shap']:.4f}")

    save_model(model)
    save_metrics(metrics)
    write_predictions(conn, tx_te, y_te, proba)

    print(f"\n  Total: {(time.time()-t0)/60:.1f} min")
    print("=== Model Training complete ===\n")

    if close:
        conn.close()
    return metrics


if __name__ == "__main__":
    run_training()
