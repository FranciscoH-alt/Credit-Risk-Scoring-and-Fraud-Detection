"""
src/ingest.py — Data Ingestion (DuckDB)

DuckDB's read_csv_auto() reads CSVs in parallel using all CPU cores and
infers schema automatically. For a 652 MB file this is dramatically faster
than chunked pandas loading — typically 30-90 seconds vs 15-25 minutes.

Memory note: DuckDB streams the CSV in blocks internally; it never loads
the full file into Python RAM. Peak Python heap stays under ~500 MB.

Tables created:
  - raw_transactions  (from train_transaction.csv)
  - raw_identity      (from train_identity.csv)
"""

import os
import time

from src.db import get_conn, get_data_dir


def ingest_transactions(conn, tx_path: str) -> int:
    """
    Load train_transaction.csv into raw_transactions using DuckDB's
    native parallel CSV reader. Replaces table on each run (idempotent).
    """
    print(f"  Loading transactions: {tx_path}")
    t0 = time.time()

    conn.execute("DROP TABLE IF EXISTS raw_transactions")
    conn.execute(f"""
        CREATE TABLE raw_transactions AS
        SELECT * FROM read_csv_auto(
            '{tx_path}',
            header = true,
            parallel = true,
            sample_size = 10000
        )
    """)

    n = conn.execute("SELECT COUNT(*) FROM raw_transactions").fetchone()[0]
    print(f"  Transactions: {n:,} rows in {time.time()-t0:.0f}s")
    return n


def ingest_identity(conn, id_path: str) -> int:
    """Load train_identity.csv into raw_identity."""
    print(f"  Loading identity: {id_path}")
    t0 = time.time()

    conn.execute("DROP TABLE IF EXISTS raw_identity")
    conn.execute(f"""
        CREATE TABLE raw_identity AS
        SELECT * FROM read_csv_auto(
            '{id_path}',
            header = true,
            parallel = true,
            sample_size = 5000
        )
    """)

    n = conn.execute("SELECT COUNT(*) FROM raw_identity").fetchone()[0]
    print(f"  Identity:     {n:,} rows in {time.time()-t0:.0f}s")
    return n


def _create_indexes(conn) -> None:
    """Create indexes on join key and frequently filtered columns."""
    print("  Creating indexes...")
    # DuckDB uses CREATE INDEX (not IF NOT EXISTS on older versions — drop first)
    for idx, tbl, col in [
        ("idx_tx_id",     "raw_transactions", "TransactionID"),
        ("idx_tx_fraud",  "raw_transactions", "isFraud"),
        ("idx_id_txid",   "raw_identity",     "TransactionID"),
    ]:
        conn.execute(f"DROP INDEX IF EXISTS {idx}")
        conn.execute(f"CREATE INDEX {idx} ON {tbl}({col})")
    print("  Indexes created.")


def run_ingest(data_dir: str = None) -> None:
    """
    Entry point called by main.py.
    Loads both CSVs into DuckDB and creates indexes.
    """
    if data_dir is None:
        data_dir = get_data_dir()

    tx_path = os.path.join(data_dir, "train_transaction.csv")
    id_path = os.path.join(data_dir, "train_identity.csv")

    for path in [tx_path, id_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"File not found: {path}\n"
                f"Set DATA_DIR in .env to the folder containing the CSVs."
            )

    print("\n=== Step 1: Data Ingestion ===")
    conn = get_conn()

    n_tx = ingest_transactions(conn, tx_path)
    n_id = ingest_identity(conn, id_path)
    _create_indexes(conn)
    conn.close()

    print(f"\n  Summary:")
    print(f"    raw_transactions: {n_tx:,} rows")
    print(f"    raw_identity:     {n_id:,} rows")
    print(f"    Identity coverage: {100*n_id/n_tx:.1f}% of transactions")
    print("=== Ingestion complete ===\n")


if __name__ == "__main__":
    run_ingest()
