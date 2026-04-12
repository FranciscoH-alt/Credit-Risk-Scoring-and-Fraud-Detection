"""
src/etl_pipeline.py — ETL Pipeline (DuckDB)

Three responsibilities:
  1. merge_tables()       — LEFT JOIN raw_transactions + raw_identity → merged_data
  2. compute_null_rates() — Audit V-column null percentages → v_null_rates.csv
  3. save_parquet()       — Snapshot merged_data → data/processed/merged_data.parquet

DuckDB does the JOIN entirely in-process with no memory pressure — it
streams from both tables in parallel. The full merge of 590k × 435 cols
completes in ~30-60 seconds.
"""

import time
from pathlib import Path

import pandas as pd

from src.db import get_conn

_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
V_NULL_THRESHOLD = 0.80


def merge_tables(conn) -> int:
    """
    Materialize LEFT JOIN of raw_transactions + raw_identity into merged_data.
    DuckDB executes this as a single parallel SQL query.
    """
    print("  Merging transactions + identity → merged_data...")
    t0 = time.time()

    conn.execute("DROP TABLE IF EXISTS merged_data")
    conn.execute("""
        CREATE TABLE merged_data AS
        SELECT t.*,
               i.* EXCLUDE (TransactionID)
        FROM raw_transactions t
        LEFT JOIN raw_identity i USING (TransactionID)
    """)

    n = conn.execute("SELECT COUNT(*) FROM merged_data").fetchone()[0]
    print(f"  merged_data: {n:,} rows in {time.time()-t0:.0f}s")
    return n


def compute_null_rates(conn) -> pd.DataFrame:
    """
    Compute null percentage for every column in merged_data.
    Saves to data/processed/v_null_rates.csv.
    Returns DataFrame with columns ['col', 'null_pct', 'total_rows'].
    """
    print("  Computing null rates...")
    t0 = time.time()

    # Get all column names
    cols = [r[0] for r in conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'merged_data' ORDER BY ordinal_position"
    ).fetchall()]

    total = conn.execute("SELECT COUNT(*) FROM merged_data").fetchone()[0]

    # Build one SELECT counting NULLs for all columns in batches of 50
    records = []
    batch_size = 50
    for i in range(0, len(cols), batch_size):
        batch = cols[i:i + batch_size]
        parts = ", ".join(
            f'SUM(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END) AS "{c}"'
            for c in batch
        )
        row = conn.execute(f"SELECT {parts} FROM merged_data").fetchone()
        for col, null_count in zip(batch, row):
            records.append({
                "col":       col,
                "null_count": null_count or 0,
                "total_rows": total,
                "null_pct":  round(100.0 * (null_count or 0) / total, 2),
            })

    null_df = pd.DataFrame(records)
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = _PROCESSED_DIR / "v_null_rates.csv"
    null_df.to_csv(out, index=False)

    v_cols   = null_df[null_df["col"].str.startswith("V")]
    high_null = v_cols[v_cols["null_pct"] > V_NULL_THRESHOLD * 100]
    print(f"  V columns: {len(v_cols)} total, {len(high_null)} to drop "
          f"(>{V_NULL_THRESHOLD*100:.0f}% null)  [{time.time()-t0:.0f}s]")
    print(f"  Saved: {out}")
    return null_df


def save_parquet(conn) -> None:
    """
    Export merged_data to parquet for use in the EDA notebook.
    DuckDB can write parquet directly — no pandas intermediate needed.
    """
    out = _PROCESSED_DIR / "merged_data.parquet"
    if out.exists():
        print(f"  Parquet snapshot already exists: {out} — skipping.")
        return

    print(f"  Exporting merged_data → {out}...")
    t0 = time.time()
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    conn.execute(f"COPY merged_data TO '{out}' (FORMAT PARQUET)")
    print(f"  Parquet saved in {time.time()-t0:.0f}s")


def run_etl(conn=None) -> None:
    """
    Entry point called by main.py as step 2.
    Sequence: merge → null rates → parquet snapshot.
    """
    close = conn is None
    if conn is None:
        conn = get_conn()

    print("\n=== Step 2: ETL Pipeline ===")
    merge_tables(conn)
    compute_null_rates(conn)
    save_parquet(conn)
    print("=== ETL complete ===\n")

    if close:
        conn.close()


if __name__ == "__main__":
    run_etl()
