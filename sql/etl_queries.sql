-- =============================================================================
-- IEEE-CIS Fraud Detection — ETL & Analytical SQL Reference
-- These queries document key transformations and can be run directly
-- in psql for ad-hoc analysis or debugging.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- SECTION 1: DATA QUALITY CHECKS (run after ingestion)
-- ---------------------------------------------------------------------------

-- Row counts across all tables
SELECT 'raw_transactions'    AS tbl, COUNT(*) AS rows FROM raw_transactions
UNION ALL
SELECT 'raw_identity',                COUNT(*) FROM raw_identity
UNION ALL
SELECT 'merged_data',                 COUNT(*) FROM merged_data
UNION ALL
SELECT 'engineered_features',         COUNT(*) FROM engineered_features
UNION ALL
SELECT 'model_predictions',           COUNT(*) FROM model_predictions
UNION ALL
SELECT 'risk_segments',               COUNT(*) FROM risk_segments;


-- Fraud rate verification (~3.5% expected)
SELECT
    "isFraud",
    COUNT(*)                                             AS transaction_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
FROM raw_transactions
GROUP BY "isFraud"
ORDER BY "isFraud";


-- Identity join coverage (% of transactions that have identity records)
SELECT
    ROUND(
        100.0 * COUNT(i."TransactionID") / COUNT(t."TransactionID"), 1
    ) AS identity_coverage_pct
FROM raw_transactions t
LEFT JOIN raw_identity i USING ("TransactionID");


-- ---------------------------------------------------------------------------
-- SECTION 2: NULL RATE AUDIT
-- Python's compute_null_rates() automates this across all V columns.
-- Use this for manual spot checks.
-- ---------------------------------------------------------------------------
SELECT
    'V1' AS col,
    ROUND(100.0 * SUM(CASE WHEN "V1" IS NULL THEN 1 ELSE 0 END) / COUNT(*), 1) AS null_pct
FROM raw_transactions;


-- ---------------------------------------------------------------------------
-- SECTION 3: POWER BI EXPORT QUERIES (mirroring src/export_powerbi.py)
-- ---------------------------------------------------------------------------

-- fraud_summary_by_card_type.csv
SELECT
    t.card4                                                      AS card_type,
    rs.risk_label,
    COUNT(*)                                                     AS transaction_count,
    ROUND(SUM(t."TransactionAmt")::NUMERIC, 2)                  AS total_amount,
    ROUND(AVG(mp.fraud_prob)::NUMERIC, 4)                       AS avg_fraud_prob,
    SUM(t."isFraud")                                            AS fraud_count
FROM model_predictions mp
JOIN risk_segments rs    USING ("TransactionID")
JOIN raw_transactions t  USING ("TransactionID")
GROUP BY t.card4, rs.risk_label
ORDER BY card_type, risk_label;


-- fraud_rate_over_time.csv
SELECT
    ef.tx_dow                                                    AS day_of_week,
    ef.tx_hour                                                   AS hour_of_day,
    COUNT(*)                                                     AS transaction_count,
    SUM(CASE WHEN t."isFraud" = 1 THEN 1 ELSE 0 END)           AS fraud_count,
    ROUND(
        100.0 * SUM(CASE WHEN t."isFraud" = 1 THEN 1 ELSE 0 END) / COUNT(*), 2
    )                                                            AS fraud_rate_pct
FROM engineered_features ef
JOIN raw_transactions t USING ("TransactionID")
GROUP BY ef.tx_dow, ef.tx_hour
ORDER BY ef.tx_dow, ef.tx_hour;


-- risk_segments_summary.csv
SELECT
    risk_label,
    risk_score,
    COUNT(*)                                                     AS transaction_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)         AS pct_of_total,
    ROUND(AVG("TransactionAmt")::NUMERIC, 2)                    AS avg_amount,
    ROUND(AVG(fraud_prob)::NUMERIC, 4)                          AS avg_fraud_prob
FROM risk_segments
GROUP BY risk_label, risk_score
ORDER BY risk_score;


-- transaction_amount_distribution.csv
SELECT
    CASE
        WHEN "TransactionAmt" < 10   THEN '<$10'
        WHEN "TransactionAmt" < 50   THEN '$10-$50'
        WHEN "TransactionAmt" < 200  THEN '$50-$200'
        WHEN "TransactionAmt" < 1000 THEN '$200-$1000'
        ELSE '>$1000'
    END             AS amount_range,
    "isFraud",
    COUNT(*)        AS transaction_count
FROM raw_transactions
GROUP BY amount_range, "isFraud"
ORDER BY
    CASE amount_range
        WHEN '<$10'       THEN 1
        WHEN '$10-$50'    THEN 2
        WHEN '$50-$200'   THEN 3
        WHEN '$200-$1000' THEN 4
        ELSE 5
    END,
    "isFraud";


-- ---------------------------------------------------------------------------
-- SECTION 4: MODEL PERFORMANCE SPOT CHECK
-- ---------------------------------------------------------------------------
SELECT
    CASE
        WHEN fraud_prob < 0.10 THEN '0.00-0.10'
        WHEN fraud_prob < 0.20 THEN '0.10-0.20'
        WHEN fraud_prob < 0.30 THEN '0.20-0.30'
        WHEN fraud_prob < 0.50 THEN '0.30-0.50'
        WHEN fraud_prob < 0.70 THEN '0.50-0.70'
        ELSE '0.70+'
    END                          AS prob_bucket,
    COUNT(*)                     AS count,
    SUM("isFraud")               AS actual_fraud,
    ROUND(
        100.0 * SUM("isFraud") / COUNT(*), 1
    )                            AS actual_fraud_rate_pct
FROM model_predictions
GROUP BY prob_bucket
ORDER BY prob_bucket;
