"""
src/export_powerbi.py — Power BI Export Layer (DuckDB)

Reads from DuckDB (final persisted state) and exports 7 CSVs to exports/.
"""

from pathlib import Path

import pandas as pd

from src.db import get_conn

_EXPORTS_DIR = Path(__file__).parent.parent / "exports"


def _save(df: pd.DataFrame, name: str) -> None:
    path = _EXPORTS_DIR / name
    df.to_csv(path, index=False)
    print(f"  Exported: {name} ({len(df)} rows)")


def export_fraud_summary_by_card_type(conn) -> None:
    df = conn.execute("""
        SELECT
            COALESCE(t.card4, 'unknown') AS card_type,
            rs.risk_label,
            COUNT(*)                     AS transaction_count,
            ROUND(SUM(t.TransactionAmt), 2) AS total_amount,
            ROUND(AVG(mp.fraud_prob), 4)    AS avg_fraud_prob,
            SUM(t.isFraud)               AS actual_fraud_count
        FROM model_predictions mp
        JOIN risk_segments rs    USING (TransactionID)
        JOIN raw_transactions t  USING (TransactionID)
        GROUP BY t.card4, rs.risk_label
        ORDER BY card_type, risk_label
    """).df()
    _save(df, "fraud_summary_by_card_type.csv")


def export_fraud_summary_by_device(conn) -> None:
    df = conn.execute("""
        SELECT
            COALESCE(rs.DeviceType, 'unknown') AS device_type,
            rs.risk_label,
            COUNT(*)                           AS transaction_count,
            ROUND(SUM(rs.TransactionAmt), 2)   AS total_amount,
            ROUND(AVG(rs.fraud_prob), 4)       AS avg_fraud_prob,
            SUM(mp.isFraud)                    AS actual_fraud_count
        FROM risk_segments rs
        JOIN model_predictions mp USING (TransactionID)
        GROUP BY rs.DeviceType, rs.risk_label
        ORDER BY device_type, risk_label
    """).df()
    _save(df, "fraud_summary_by_device.csv")


def export_fraud_rate_over_time(conn) -> None:
    df = conn.execute("""
        SELECT
            ef.tx_dow                 AS day_of_week,
            ef.tx_hour                AS hour_of_day,
            COUNT(*)                  AS transaction_count,
            SUM(t.isFraud)            AS fraud_count,
            ROUND(100.0 * SUM(t.isFraud) / COUNT(*), 2) AS fraud_rate_pct,
            ROUND(AVG(t.TransactionAmt), 2) AS avg_amount
        FROM engineered_features ef
        JOIN raw_transactions t USING (TransactionID)
        GROUP BY ef.tx_dow, ef.tx_hour
        ORDER BY ef.tx_dow, ef.tx_hour
    """).df()
    day_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    df["day_name"] = df["day_of_week"].map(day_names)
    _save(df, "fraud_rate_over_time.csv")


def export_risk_segments_summary(conn) -> None:
    df = conn.execute("""
        SELECT
            risk_label,
            risk_score,
            COUNT(*)                  AS transaction_count,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_total,
            ROUND(AVG(TransactionAmt), 2) AS avg_amount,
            ROUND(AVG(fraud_prob), 4)     AS avg_fraud_prob
        FROM risk_segments
        GROUP BY risk_label, risk_score
        ORDER BY risk_score
    """).df()
    _save(df, "risk_segments_summary.csv")


def verify_shap_export() -> None:
    path = _EXPORTS_DIR / "feature_importance_shap.csv"
    if not path.exists():
        print("  WARNING: feature_importance_shap.csv missing — run training step first.")
        return
    df = pd.read_csv(path)
    print(f"  Verified: feature_importance_shap.csv ({len(df)} features, top: {df.iloc[0]['feature']})")


def verify_metrics_export() -> None:
    path = _EXPORTS_DIR / "model_performance_metrics.csv"
    if not path.exists():
        print("  WARNING: model_performance_metrics.csv missing — run training step first.")
        return
    df = pd.read_csv(path)
    auc = df[df["metric_name"] == "auc_roc"]["value"].values
    print(f"  Verified: model_performance_metrics.csv  |  AUC-ROC = {auc[0]:.4f}" if len(auc) else
          f"  Verified: model_performance_metrics.csv ({len(df)} metrics)")


def export_transaction_amount_distribution(conn) -> None:
    df = conn.execute("""
        SELECT
            CASE
                WHEN TransactionAmt < 10   THEN '<$10'
                WHEN TransactionAmt < 50   THEN '$10-$50'
                WHEN TransactionAmt < 200  THEN '$50-$200'
                WHEN TransactionAmt < 1000 THEN '$200-$1000'
                ELSE '>$1000'
            END          AS amount_range,
            isFraud,
            COUNT(*)     AS transaction_count,
            ROUND(AVG(TransactionAmt), 2) AS avg_amount
        FROM raw_transactions
        GROUP BY amount_range, isFraud
        ORDER BY MIN(TransactionAmt), isFraud
    """).df()
    _save(df, "transaction_amount_distribution.csv")


def run_exports(conn=None, metrics: dict = None) -> None:
    close = conn is None
    if conn is None:
        conn = get_conn()

    _EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\n=== Step 6: Power BI Exports ===")

    export_fraud_summary_by_card_type(conn)
    export_fraud_summary_by_device(conn)
    export_fraud_rate_over_time(conn)
    export_risk_segments_summary(conn)
    verify_shap_export()
    verify_metrics_export()
    export_transaction_amount_distribution(conn)

    files = sorted(_EXPORTS_DIR.glob("*.csv"))
    print(f"\n  All exports in {_EXPORTS_DIR}:")
    for f in files:
        print(f"    {f.name:<50s}  {f.stat().st_size//1024:,} KB")

    print("=== Exports complete ===\n")

    if close:
        conn.close()


if __name__ == "__main__":
    run_exports()
