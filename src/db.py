"""
src/db.py — DuckDB connection factory.

DuckDB replaces PostgreSQL as the warehouse layer. No server required —
everything lives in a single file at DB_PATH (default: data/fraud_detection.duckdb).

DuckDB advantages for this project:
  - No installation or server setup
  - Native parallel CSV reading (read_csv_auto)
  - In-process: zero network overhead
  - Full SQL support including window functions and COPY
  - Parquet read/write built-in

Usage:
    from src.db import get_conn, DB_PATH

    conn = get_conn()
    conn.execute("SELECT COUNT(*) FROM raw_transactions").fetchone()
    conn.close()
"""

import os
from pathlib import Path

import duckdb
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Database path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent

def _resolve_db_path() -> str:
    raw = os.environ.get("DB_PATH", "data/fraud_detection.duckdb")
    p = Path(raw)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)

DB_PATH = _resolve_db_path()


def get_conn() -> duckdb.DuckDBPyConnection:
    """
    Return a DuckDB connection to the project database file.
    DuckDB supports multiple readers but only one writer at a time.
    For pipeline use (single process), this is always safe.
    """
    return duckdb.connect(DB_PATH)


def get_data_dir() -> str:
    """
    Return the path to the directory containing the raw CSV files.
    Reads DATA_DIR from .env; defaults to the ieee-fraud-detection 2 folder.
    """
    data_dir = os.environ.get(
        "DATA_DIR",
        "/Users/franciscohenriques/Downloads/ieee-fraud-detection 2",
    )
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"DATA_DIR '{data_dir}' does not exist.\n"
            "Update DATA_DIR in your .env file to point to the folder "
            "containing train_transaction.csv and train_identity.csv."
        )
    return data_dir
