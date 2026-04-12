"""
src/risk_segmentation.py — Risk Segmentation (DuckDB)

Assigns Low / Medium / High risk tiers from fraud_prob scores.
Justifies thresholds via F1-optimal and cost-sensitive analysis.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.db import get_conn

_EXPORTS_DIR = Path(__file__).parent.parent / "exports"
_VISUALS_DIR = Path(__file__).parent.parent / "visuals"


def justify_thresholds(y_true, proba) -> dict:
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)

    f1 = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-9)
    f1_idx = int(np.argmax(f1))
    f1_thr = float(thresholds[f1_idx])

    costs = []
    for t in thresholds:
        p = (proba >= t).astype(int)
        fn = int(((p == 0) & (y_true == 1)).sum())
        fp = int(((p == 1) & (y_true == 0)).sum())
        costs.append(fn * 10 + fp)
    cost_idx = int(np.argmin(costs))
    cost_thr = float(thresholds[cost_idx])

    print(f"  F1-optimal threshold:   {f1_thr:.4f}  (F1={f1[f1_idx]:.4f})")
    print(f"  Cost-optimal threshold: {cost_thr:.4f}  (cost={costs[cost_idx]:,})")
    return {
        "f1_optimal_threshold":   round(f1_thr, 4),
        "cost_optimal_threshold": round(cost_thr, 4),
    }


def plot_precision_recall(y_true, proba) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-9)
    f1_idx = int(np.argmax(f1))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recalls, precisions, lw=2, label="Precision-Recall")
    ax.scatter([recalls[f1_idx]], [precisions[f1_idx]], color="orange", zorder=5,
               label=f"F1-optimal (t={thresholds[f1_idx]:.2f})")
    ax.axvline(x=0.30, color="red", linestyle=":", alpha=0.7, label="Business lower bound (0.30)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Fraud Detection")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = _VISUALS_DIR / "precision_recall_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def run_segmentation(conn=None) -> None:
    close = conn is None
    if conn is None:
        conn = get_conn()

    print("\n=== Step 5: Risk Segmentation ===")
    t0 = time.time()

    # Load predictions + context
    df = conn.execute("""
        SELECT
            mp.TransactionID,
            mp.isFraud,
            mp.fraud_prob,
            t.TransactionAmt,
            t.card4,
            ri.DeviceType,
            ef.tx_hour
        FROM model_predictions mp
        JOIN raw_transactions t    USING (TransactionID)
        LEFT JOIN raw_identity ri  USING (TransactionID)
        LEFT JOIN engineered_features ef USING (TransactionID)
    """).df()

    print(f"  {len(df):,} predictions loaded")

    # Threshold analysis
    analysis = justify_thresholds(df["isFraud"].values, df["fraud_prob"].values)
    _EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**analysis,
                   "business_low": 0.30,
                   "business_high": 0.70}]).to_csv(
        _EXPORTS_DIR / "threshold_analysis.csv", index=False
    )

    # Assign tiers
    df["risk_label"] = "Medium"
    df.loc[df["fraud_prob"] < 0.30, "risk_label"] = "Low"
    df.loc[df["fraud_prob"] > 0.70, "risk_label"] = "High"
    df["risk_score"] = df["risk_label"].map({"Low": 1, "Medium": 2, "High": 3}).astype("int8")

    total = len(df)
    for lbl in ["Low", "Medium", "High"]:
        n = (df["risk_label"] == lbl).sum()
        print(f"  {lbl:8s}: {n:6,} ({100*n/total:.1f}%)")

    # Write to DuckDB
    seg_df = df[["TransactionID", "fraud_prob", "risk_label", "risk_score",
                 "TransactionAmt", "card4", "DeviceType", "tx_hour"]]
    conn.execute("DROP TABLE IF EXISTS risk_segments")
    conn.execute("CREATE TABLE risk_segments AS SELECT * FROM seg_df")
    print(f"  {len(seg_df):,} rows written to risk_segments")

    plot_precision_recall(df["isFraud"].values, df["fraud_prob"].values)
    print(f"  Duration: {time.time()-t0:.0f}s")
    print("=== Risk Segmentation complete ===\n")

    if close:
        conn.close()


if __name__ == "__main__":
    run_segmentation()
