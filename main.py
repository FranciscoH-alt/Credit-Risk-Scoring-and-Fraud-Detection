"""
main.py — Master Pipeline Runner (DuckDB)

Runs the full IEEE-CIS Fraud Detection pipeline end-to-end.
Database: DuckDB file at data/fraud_detection.duckdb (no server needed).

Steps:
  1. ingest    — Load CSVs into DuckDB (raw_transactions, raw_identity)
  2. etl       — Merge tables, null-rate audit, parquet snapshot
  3. features  — Feature engineering → engineered_features
  4. train     — XGBoost + Optuna + SHAP → model_predictions
  5. segment   — Risk tier assignment → risk_segments
  6. export    — 7 Power BI CSVs → exports/

Usage:
  python main.py                          # full pipeline
  python main.py --skip ingest etl        # resume from feature engineering
  python main.py --only train export      # just train + export
  python main.py --optuna-trials 5        # smoke test (fast)
  python main.py --use-smote              # SMOTE instead of scale_pos_weight
  python main.py --data-dir /path/to/csv # override CSV folder
"""

import argparse
import sys
import time
from pathlib import Path

STEPS = ["ingest", "etl", "features", "train", "segment", "export"]


def parse_args():
    p = argparse.ArgumentParser(
        description="IEEE-CIS Fraud Detection Pipeline (DuckDB)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--skip",  nargs="+", choices=STEPS, default=[], metavar="STEP")
    p.add_argument("--only",  nargs="+", choices=STEPS, default=[], metavar="STEP")
    p.add_argument("--data-dir",      type=str, default=None)
    p.add_argument("--use-smote",     action="store_true", default=False)
    p.add_argument("--optuna-trials", type=int, default=50, metavar="N")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    steps = [s for s in STEPS if s in args.only] if args.only \
            else [s for s in STEPS if s not in args.skip]

    if not steps:
        print("No steps to run.")
        return 1

    print("=" * 58)
    print("  IEEE-CIS Fraud Detection Pipeline  (DuckDB)")
    print(f"  Steps:         {' → '.join(steps)}")
    print(f"  Optuna trials: {args.optuna_trials}")
    print(f"  SMOTE:         {'yes' if args.use_smote else 'no (scale_pos_weight)'}")

    # Resolve DB path and show it
    from src.db import DB_PATH, get_conn
    print(f"  Database:      {DB_PATH}")
    print("=" * 58)

    # Share one connection across all steps (faster, single write lock)
    conn = get_conn()
    t_pipeline = time.time()
    metrics = {}

    if "ingest" in steps:
        from src.ingest import run_ingest
        conn.close()            # ingest reopens its own connection per table
        run_ingest(data_dir=args.data_dir)
        conn = get_conn()

    if "etl" in steps:
        from src.etl_pipeline import run_etl
        run_etl(conn=conn)

    if "features" in steps:
        from src.feature_engineering import run_feature_engineering
        run_feature_engineering(conn=conn)

    if "train" in steps:
        from src.train_model import run_training
        metrics = run_training(conn=conn, use_smote=args.use_smote,
                               n_optuna_trials=args.optuna_trials)

    if "segment" in steps:
        from src.risk_segmentation import run_segmentation
        run_segmentation(conn=conn)

    if "export" in steps:
        from src.export_powerbi import run_exports
        run_exports(conn=conn, metrics=metrics)

    conn.close()

    elapsed  = time.time() - t_pipeline
    minutes  = int(elapsed // 60)
    seconds  = int(elapsed % 60)

    print("=" * 58)
    print(f"  Pipeline complete in {minutes}m {seconds}s")
    if metrics.get("auc_roc"):
        print(f"  AUC-ROC:       {metrics['auc_roc']:.4f}")
        print(f"  Avg Precision: {metrics['avg_precision']:.4f}")
        print(f"  F1 (fraud):    {metrics['f1_fraud']:.4f}")
    print(f"  Database: {DB_PATH}")
    print(f"  Exports:  {Path('exports').resolve()}")
    print(f"  Model:    {Path('models/fraud_model.pkl').resolve()}")
    print("=" * 58)
    return 0


if __name__ == "__main__":
    sys.exit(main())
