"""
src/feature_engineering.py — Feature Engineering (DuckDB)

Reads merged_data from DuckDB, engineers features in pandas (velocity
features require row-level iteration), then writes the result back to
DuckDB as engineered_features and also saves a parquet snapshot.

Features:
  1. V-column null filtering (drop >80% null)
  2. Time features (hour, day-of-week, day-of-month from TransactionDT)
  3. Velocity (transaction count per card1 in 1h / 24h windows)
  4. Behavioral deviation (amount vs card1 mean / z-score)
  5. Identity aggregations (device, email domain frequency)
  6. card4 × amount bucket interaction
  7. card1+addr1 co-occurrence frequency (strong fraud signal)
  8. Log-transformed amount
  9. Categorical encoding (M cols, card4/card6/ProductCD, DeviceType)
  10. High-cardinality frequency encoding (email domains, DeviceInfo)
  11. Missing value imputation
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.db import get_conn

_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
V_NULL_THRESHOLD = 0.80


# ---------------------------------------------------------------------------
# 1. V-column filtering
# ---------------------------------------------------------------------------

def get_v_cols_to_drop() -> list[str]:
    path = _PROCESSED_DIR / "v_null_rates.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"v_null_rates.csv not found at {path}. Run ETL step first."
        )
    null_df = pd.read_csv(path)
    v_mask   = null_df["col"].str.startswith("V")
    high     = null_df[v_mask & (null_df["null_pct"] > V_NULL_THRESHOLD * 100)]
    return high["col"].tolist()


# ---------------------------------------------------------------------------
# 2. Time features
# ---------------------------------------------------------------------------

def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["TransactionDT"]
    df["tx_hour"] = ((dt // 3600) % 24).astype("int8")
    df["tx_dow"]  = ((dt // 86400) % 7).astype("int8")
    df["tx_dom"]  = ((dt // 86400) % 30 + 1).astype("int8")
    return df


# ---------------------------------------------------------------------------
# 3. Velocity features
# ---------------------------------------------------------------------------

def _count_in_window(times: np.ndarray, window_secs: int) -> np.ndarray:
    """O(n log n) windowed count using searchsorted. Input must be sorted."""
    counts = np.zeros(len(times), dtype=np.int32)
    for i in range(len(times)):
        lo = np.searchsorted(times, times[i] - window_secs, side="left")
        counts[i] = i - lo
    return counts


def engineer_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    # Sort by card1 + time, compute velocity arrays, then restore original order.
    # Avoids groupby.apply (pandas 2.2+ may drop the groupby key from the result).
    sorted_df = df.sort_values(["card1", "TransactionDT"])
    card1_vals = sorted_df["card1"].values
    dt_vals    = sorted_df["TransactionDT"].values

    counts_1h  = np.zeros(len(sorted_df), dtype=np.int32)
    counts_24h = np.zeros(len(sorted_df), dtype=np.int32)

    # Locate group boundaries (NaN card1 treated as its own group)
    boundaries = np.where(
        np.concatenate(([True], card1_vals[1:] != card1_vals[:-1]))
    )[0]
    boundaries = np.append(boundaries, len(card1_vals))

    for i in range(len(boundaries) - 1):
        s, e = int(boundaries[i]), int(boundaries[i + 1])
        times = dt_vals[s:e]
        counts_1h[s:e]  = _count_in_window(times, 3600)
        counts_24h[s:e] = _count_in_window(times, 86400)

    result = sorted_df.copy()
    result["tx_count_card1_1h"]  = counts_1h
    result["tx_count_card1_24h"] = counts_24h
    return result.reindex(df.index)


# ---------------------------------------------------------------------------
# 4. Behavioral deviation
# ---------------------------------------------------------------------------

def engineer_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    stats = (
        df.groupby("card1")["TransactionAmt"]
        .agg(card1_mean="mean", card1_std="std")
        .reset_index()
    )
    df = df.merge(stats, on="card1", how="left")
    df["amt_vs_card1_mean"]   = (df["TransactionAmt"] - df["card1_mean"]).astype("float32")
    df["amt_vs_card1_zscore"] = (
        (df["TransactionAmt"] - df["card1_mean"]) / (df["card1_std"].fillna(1.0) + 1e-8)
    ).astype("float32")
    df.drop(columns=["card1_mean", "card1_std"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# 5. Identity aggregations
# ---------------------------------------------------------------------------

def engineer_identity_features(df: pd.DataFrame) -> pd.DataFrame:
    for col, new_col in [("DeviceInfo", "tx_count_device"),
                          ("P_emaildomain", "tx_count_pdomain")]:
        if col in df.columns:
            df[new_col] = (
                df.groupby(col)["TransactionID"].transform("count")
                .fillna(0).astype("int32")
            )
    return df


# ---------------------------------------------------------------------------
# 6. Interaction feature
# ---------------------------------------------------------------------------

def engineer_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    bins   = [0, 10, 50, 200, 1000, float("inf")]
    labels = ["<10", "10-50", "50-200", "200-1000", ">1000"]
    bucket = pd.cut(df["TransactionAmt"], bins=bins, labels=labels, right=False).astype(str)
    df["card4_amt_bucket"] = df["card4"].fillna("unknown").astype(str) + "_" + bucket
    return df


# ---------------------------------------------------------------------------
# 7. card1+addr1 frequency
# ---------------------------------------------------------------------------

def engineer_card1_addr1_frequency(df: pd.DataFrame) -> pd.DataFrame:
    df["card1_addr1_count"] = (
        df.groupby(["card1", "addr1"])["TransactionID"]
        .transform("count").fillna(0).astype("int32")
    )
    return df


# ---------------------------------------------------------------------------
# 8. Log amount
# ---------------------------------------------------------------------------

def engineer_log_amount(df: pd.DataFrame) -> pd.DataFrame:
    df["log_transaction_amt"] = np.log1p(df["TransactionAmt"]).astype("float32")
    return df


# ---------------------------------------------------------------------------
# 9 & 10. Encoding
# ---------------------------------------------------------------------------

def encode_m_columns(df: pd.DataFrame) -> pd.DataFrame:
    for i in range(1, 10):
        col = f"M{i}"
        if col in df.columns:
            df[f"M{i}_enc"] = df[col].map({"T": 1, "F": 0}).fillna(-1).astype("int8")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    # Low-cardinality → label encode
    for col in ["card4", "card6", "ProductCD", "DeviceType"]:
        if col in df.columns:
            df[col] = pd.Categorical(df[col].fillna("__missing__")).codes.astype("int16")
    # High-cardinality → frequency encode
    for col, new_col in [("P_emaildomain", "P_emaildomain_freq"),
                          ("R_emaildomain", "R_emaildomain_freq"),
                          ("DeviceInfo",    "DeviceInfo_freq")]:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True).to_dict()
            df[new_col] = df[col].map(freq).fillna(0.0).astype("float32")
    return df


# ---------------------------------------------------------------------------
# 11. Imputation
# ---------------------------------------------------------------------------

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    # C cols → 0 (counts; NaN means 0)
    c_cols = [f"C{i}" for i in range(1, 15) if f"C{i}" in df.columns]
    df[c_cols] = df[c_cols].fillna(0)

    # D, V, id numeric → median
    d_cols = [f"D{i}" for i in range(1, 16) if f"D{i}" in df.columns]
    v_cols = [c for c in df.columns if c.startswith("V")]
    id_num = (
        [f"id_{str(i).zfill(2)}" for i in range(1, 12)]
        + ["id_13", "id_14", "id_17", "id_18", "id_19", "id_20", "id_32"]
    )
    id_present = [c for c in id_num if c in df.columns]

    for group in [d_cols, v_cols, id_present]:
        if group:
            df[group] = df[group].fillna(df[group].median())

    # Misc base numerics
    for col in ["card2", "card3", "card5", "addr1", "addr2", "dist1", "dist2"]:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_STRING_COLS_TO_DROP = [
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "P_emaildomain", "R_emaildomain", "DeviceInfo",
]


def run_feature_engineering(conn=None) -> None:
    """
    Entry point called by main.py as step 3.
    Reads merged_data from DuckDB, engineers features in pandas,
    writes back to DuckDB as engineered_features + parquet snapshot.
    """
    close = conn is None
    if conn is None:
        conn = get_conn()

    print("\n=== Step 3: Feature Engineering ===")
    t0 = time.time()

    # Determine V columns to drop
    v_drop = get_v_cols_to_drop()
    print(f"  Dropping {len(v_drop)} V columns (>{V_NULL_THRESHOLD*100:.0f}% null)")

    # Load full merged_data into pandas via DuckDB arrow
    # DuckDB → Arrow → pandas avoids double-copying large arrays
    print("  Loading merged_data...")
    df = conn.execute("SELECT * FROM merged_data").df()
    print(f"  Loaded {len(df):,} rows × {df.shape[1]} columns  "
          f"({df.memory_usage(deep=True).sum()/1e9:.2f} GB)")

    # Apply feature engineering pipeline
    df = df.drop(columns=[c for c in v_drop if c in df.columns])
    df = engineer_time_features(df)
    df = engineer_velocity_features(df)
    df = engineer_behavioral_features(df)
    df = engineer_identity_features(df)
    df = engineer_interaction_features(df)
    df = engineer_card1_addr1_frequency(df)
    df = engineer_log_amount(df)
    df = encode_m_columns(df)
    df = encode_categoricals(df)
    df = impute_missing(df)

    # Drop original string columns that have been encoded/replaced
    drop_these = [c for c in _STRING_COLS_TO_DROP if c in df.columns]
    drop_these += [f"M{i}" for i in range(1, 10) if f"M{i}" in df.columns]
    df.drop(columns=drop_these, inplace=True)

    # Write to DuckDB
    print("  Writing engineered_features to DuckDB...")
    conn.execute("DROP TABLE IF EXISTS engineered_features")
    conn.execute("CREATE TABLE engineered_features AS SELECT * FROM df")

    # Parquet snapshot
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = _PROCESSED_DIR / "engineered_features.parquet"
    conn.execute(f"COPY engineered_features TO '{parquet_path}' (FORMAT PARQUET)")

    n_cols = conn.execute(
        "SELECT COUNT(*) FROM information_schema.columns "
        "WHERE table_name = 'engineered_features'"
    ).fetchone()[0]

    print(f"\n  Feature engineering complete:")
    print(f"    Rows:     {len(df):,}")
    print(f"    Columns:  {n_cols}")
    print(f"    Parquet:  {parquet_path}")
    print(f"    Duration: {time.time()-t0:.0f}s")
    print("=== Feature Engineering complete ===\n")

    if close:
        conn.close()


if __name__ == "__main__":
    run_feature_engineering()
