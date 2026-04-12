# Credit Risk Scoring & Fraud Detection System

## Overview

End-to-end fraud detection system built on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset (~590k transactions, 3.5% fraud rate). The pipeline ingests raw data into PostgreSQL, engineers >50 features, trains an XGBoost model with Optuna hyperparameter tuning, and produces Power BI-ready exports with risk segmentation.

**Stack:** Python 3.13 · PostgreSQL · XGBoost · SHAP · Optuna · scikit-learn · Power BI

| Metric | Value |
|---|---|
| AUC-ROC | TBD (target ≥ 0.88) |
| Average Precision | TBD |
| Precision (Fraud) | TBD |
| Recall (Fraud) | TBD |
| F1 (Fraud) | TBD |

> Fill in the table above after running `python main.py`.

---

## Repository Structure

```
Credit-Risk-Proj/
├── data/
│   ├── raw/             # gitignored — symlink or copy CSVs here
│   └── processed/       # parquet snapshots, null rate audit
├── sql/
│   ├── schema.sql       # 6 PostgreSQL table definitions
│   └── etl_queries.sql  # Reference SQL for ETL and Power BI exports
├── notebooks/
│   └── eda.ipynb        # 8-section exploratory data analysis
├── src/
│   ├── db.py            # PostgreSQL connection factory (reads from .env)
│   ├── ingest.py        # Chunked CSV → PostgreSQL loader
│   ├── etl_pipeline.py  # Merge, null-rate audit, parquet snapshots
│   ├── feature_engineering.py  # 12 feature groups → engineered_features
│   ├── train_model.py   # Optuna + XGBoost + SHAP + predictions
│   ├── risk_segmentation.py    # Risk tier assignment + threshold analysis
│   └── export_powerbi.py       # 7 CSVs for Power BI
├── models/
│   └── fraud_model.pkl  # Trained XGBoost model (gitignored)
├── exports/             # Power BI CSVs (gitignored — generated at runtime)
├── visuals/             # EDA plots as PNG (gitignored — generated at runtime)
├── main.py              # Master pipeline runner
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd Credit-Risk-Proj
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 2. Get the dataset

The raw CSVs are **not included** in this repo (652 MB transaction file). Download from Kaggle:

```bash
# Option A: Kaggle CLI
pip install kaggle
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip -d /path/to/your/data/folder

# Option B: Manual download
# Go to: https://www.kaggle.com/c/ieee-fraud-detection/data
# Download: train_transaction.csv, train_identity.csv
# Place them anywhere accessible
```

Required files:
- `train_transaction.csv` (652 MB, 590k rows, 394 columns)
- `train_identity.csv` (25 MB, 144k rows, 41 columns)

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
PG_HOST=localhost
PG_PORT=5432
PG_DB=fraud_detection
PG_USER=your_postgres_user
PG_PASSWORD=your_postgres_password

# Path to folder containing train_transaction.csv and train_identity.csv
DATA_DIR=/path/to/your/ieee-fraud-detection/folder
```

### 4. Create the PostgreSQL database

```bash
psql -U your_postgres_user -d postgres -c "CREATE DATABASE fraud_detection;"
```

---

## Running the Pipeline

### Full run (first time)

```bash
python main.py
```

### Resume from a specific step

```bash
# Ingest + ETL already done — resume from feature engineering
python main.py --skip ingest etl

# Run only training and export
python main.py --only train export

# Smoke test (5 Optuna trials, much faster)
python main.py --only train --optuna-trials 5
```

### Options

```
--skip STEP [STEP ...]    Steps to skip (ingest etl features train segment export)
--only STEP [STEP ...]    Run only these steps
--data-dir PATH           Path to CSV folder (overrides DATA_DIR in .env)
--use-smote               Use SMOTE instead of scale_pos_weight for imbalance
--optuna-trials N         Number of hyperparameter trials (default: 50)
```

### Step durations (approximate, MacBook Pro M-series)

| Step | Duration |
|---|---|
| Ingest (652 MB CSV) | 15–25 min |
| ETL (merge + null rates) | 10–20 min |
| Feature Engineering | 20–40 min |
| Training (50 Optuna trials × 3-fold CV) | 2–5 hours |
| Segmentation | < 1 min |
| Export | < 1 min |

---

## Database Schema

Six PostgreSQL tables form a linear processing pipeline:

```
raw_transactions (590k rows, 394 cols)
        │
        ├── LEFT JOIN ──► raw_identity (144k rows, 41 cols)
        │
        ▼
   merged_data (590k rows, ~435 cols)  ← materialized join
        │
        ▼
engineered_features (590k rows, ~80 cols)  ← feature engineering output
        │
        ▼
  model_predictions (118k rows)  ← test set fraud probabilities
        │
        ▼
    risk_segments (118k rows)  ← Low / Medium / High tiers
```

Key design decisions:
- **`merged_data` is materialized** (not a view) to avoid repeated JOIN cost
- **Raw tables are never modified** — null-dropping and encoding only in `engineered_features`
- **V-column filtering** (drop >80% null) is determined at ETL time from `data/processed/v_null_rates.csv`

---

## Feature Engineering

| Feature Group | Features | Rationale |
|---|---|---|
| Time | `tx_hour`, `tx_dow`, `tx_dom` | Fraud peaks at certain hours |
| Velocity | `tx_count_card1_1h`, `tx_count_card1_24h` | Rapid sequential transactions = fraud signal |
| Behavioral | `amt_vs_card1_mean`, `amt_vs_card1_zscore` | Unusual amounts for a card = fraud signal |
| Identity | `tx_count_device`, `tx_count_pdomain` | Shared device/domain = fraud ring signal |
| Interaction | `card4_amt_bucket` | Card network × amount tier (20 values) |
| Card co-occurrence | `card1_addr1_count` | Strongest single engineered feature per community research |
| Amount transform | `log_transaction_amt` | Reduces right-skew |
| Frequency encoding | `P_emaildomain_freq`, `R_emaildomain_freq`, `DeviceInfo_freq` | High-cardinality strings as proportions |
| M columns | `M1_enc` … `M9_enc` | T=1, F=0, NaN=-1 |

---

## Model Details

**Algorithm:** XGBoost (`tree_method='hist'`)

**Class imbalance:** `scale_pos_weight ≈ 27.6` (primary). Optional SMOTE via `--use-smote`.

**Hyperparameter tuning:** Optuna TPE sampler, 50 trials, 3-fold stratified CV.

**Parameters tuned:** `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`

**SHAP:** TreeExplainer on 5,000-row test sample. Top 20 features exported to `exports/feature_importance_shap.csv`.

**AUC-ROC fallback strategy:**

| Achieved AUC | Action |
|---|---|
| < 0.80 | Check V-column null threshold (try 0.90), audit encoding |
| 0.80–0.85 | Verify `card1_addr1_count` feature is present; log(amt) is applied |
| 0.85–0.88 | Increase Optuna trials to 100 |
| ≥ 0.88 | Target met |

---

## Risk Segmentation

Fraud probability scores are bucketed into three tiers:

| Tier | Threshold | Risk Score | Interpretation |
|---|---|---|---|
| Low | fraud_prob < 0.30 | 1 | Routine processing |
| Medium | 0.30 – 0.70 | 2 | Flag for review |
| High | > 0.70 | 3 | Block / manual intervention |

Thresholds are validated by:
1. **F1-optimal threshold** (maximises F1 on precision-recall curve)
2. **Cost-optimal threshold** (FN = 10× cost of FP, reflecting real ops)

Both methods are documented in `exports/threshold_analysis.csv`.

---

## Power BI Exports

All 7 files are generated in `exports/` by `python main.py --only export`:

| File | Content | Suggested Visual |
|---|---|---|
| `fraud_summary_by_card_type.csv` | Fraud metrics by card network + risk tier | Stacked bar |
| `fraud_summary_by_device.csv` | Fraud metrics by device type + risk tier | Donut |
| `fraud_rate_over_time.csv` | Fraud rate by hour × day of week | Heatmap |
| `risk_segments_summary.csv` | Count, %, avg amount per tier | KPI cards |
| `feature_importance_shap.csv` | Top 20 SHAP features | Horizontal bar |
| `model_performance_metrics.csv` | AUC, precision, recall, F1 | KPI table |
| `transaction_amount_distribution.csv` | Fraud rate by amount range | Grouped bar |

---

## Key Findings

> *Fill in after running the full pipeline.*

- **Overall fraud rate:** ~3.5% (24,000 / 590,000 transactions)
- **Top predictive features:** See `exports/feature_importance_shap.csv`
- **Fraud concentration by hour:** See `visuals/temporal_patterns.png`
- **Highest-risk card network:** See `visuals/categorical_fraud_rates.png`
- **Risk tier distribution:** See `exports/risk_segments_summary.csv`

---

## Reproducing Results

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure .env with your PostgreSQL credentials and DATA_DIR

# 3. Create database
psql -c "CREATE DATABASE fraud_detection;"

# 4. Run full pipeline
python main.py

# 5. Open notebook for EDA
jupyter notebook notebooks/eda.ipynb
```

All random seeds are fixed at `42` for reproducibility.
