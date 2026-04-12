-- =============================================================================
-- IEEE-CIS Fraud Detection — DuckDB Schema
-- Tables are created by the ETL pipeline; this file documents the structure.
-- DuckDB uses CREATE OR REPLACE TABLE (not IF NOT EXISTS for all cases).
-- =============================================================================

-- raw_transactions and raw_identity are created directly from CSVs via
-- CREATE TABLE AS SELECT * FROM read_csv_auto(...) in ingest.py.
-- The schema adapts automatically to the CSV columns.

-- merged_data: LEFT JOIN of transactions + identity
-- engineered_features: output of feature engineering
-- model_predictions: fraud probabilities from XGBoost
-- risk_segments: Low / Medium / High risk tiers

-- See src/etl_pipeline.py and src/ingest.py for the actual CREATE statements.
-- Run sql/etl_queries.sql for ad-hoc analysis queries.
