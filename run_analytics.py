"""
Credit Risk SQL Queries - Phase 1 Analytics
Key queries for: default rates by segment, risk analysis, cohort analysis
"""

import duckdb
import pandas as pd
from pathlib import Path

DB_PATH = "data/credit_risk.duckdb"
conn = duckdb.connect(DB_PATH)

print("="*80)
print("CREDIT RISK ANALYTICAL QUERIES - PHASE 1")
print("="*80)

# ==============================================================================
# QUERY 1: DEFAULT RATE BY CUSTOMER SEGMENT
# ==============================================================================
print("\n1️⃣  DEFAULT RATE BY CUSTOMER SEGMENT")
print("-" * 80)

query_1 = """
SELECT
    education_level,
    COUNT(*) as total_customers,
    SUM(CASE WHEN is_default = 1 THEN 1 ELSE 0 END) as default_count,
    ROUND(100.0 * SUM(CASE WHEN is_default = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as default_rate_pct,
    ROUND(AVG(income_total), 0) as avg_income,
    ROUND(AVG(debt_to_income_ratio), 2) as avg_dti_ratio
FROM AGG_CUSTOMER_RISK_PROFILE
WHERE education_level IS NOT NULL
GROUP BY education_level
ORDER BY default_rate_pct DESC
"""

result_1 = conn.execute(query_1).fetch_df()
print(result_1.to_string(index=False))
result_1.to_csv("sql/01_default_rate_by_education.csv", index=False)
print("\n✓ Saved to: sql/01_default_rate_by_education.csv")

# ==============================================================================
# QUERY 2: DEFAULT RATE BY INCOME TYPE
# ==============================================================================
print("\n2️⃣  DEFAULT RATE BY INCOME TYPE")
print("-" * 80)

query_2 = """
SELECT
    income_type,
    COUNT(*) as total_customers,
    SUM(CASE WHEN is_default = 1 THEN 1 ELSE 0 END) as default_count,
    ROUND(100.0 * SUM(CASE WHEN is_default = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as default_rate_pct,
    ROUND(AVG(income_total), 0) as avg_income,
    ROUND(AVG(credit_amount), 0) as avg_credit,
    ROUND(AVG(debt_to_income_ratio), 2) as avg_dti_ratio
FROM AGG_CUSTOMER_RISK_PROFILE
WHERE income_type IS NOT NULL
GROUP BY income_type
ORDER BY default_rate_pct DESC
"""

result_2 = conn.execute(query_2).fetch_df()
print(result_2.to_string(index=False))
result_2.to_csv("sql/02_default_rate_by_income_type.csv", index=False)
print("\n✓ Saved to: sql/02_default_rate_by_income_type.csv")

# ==============================================================================
# QUERY 3: DEBT-TO-INCOME RATIO RISK SEGMENTATION
# ==============================================================================
print("\n3️⃣  RISK SEGMENTATION BY DEBT-TO-INCOME RATIO")
print("-" * 80)

query_3 = """
SELECT
    dti_risk_level,
    COUNT(*) as total_customers,
    SUM(CASE WHEN is_default = 1 THEN 1 ELSE 0 END) as default_count,
    ROUND(100.0 * SUM(CASE WHEN is_default = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as default_rate_pct,
    ROUND(AVG(debt_to_income_ratio), 2) as avg_dti_ratio,
    ROUND(AVG(income_total), 0) as avg_income,
    ROUND(AVG(external_score_avg), 3) as avg_external_score
FROM AGG_CUSTOMER_RISK_PROFILE
GROUP BY dti_risk_level
ORDER BY
    CASE dti_risk_level WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END
"""

result_3 = conn.execute(query_3).fetch_df()
print(result_3.to_string(index=False))
result_3.to_csv("sql/03_risk_segmentation_dti.csv", index=False)
print("\n✓ Saved to: sql/03_risk_segmentation_dti.csv")

# ==============================================================================
# QUERY 4: GENDER & PROPERTY OWNERSHIP ANALYSIS
# ==============================================================================
print("\n4️⃣  DEFAULT RATE BY GENDER & PROPERTY OWNERSHIP")
print("-" * 80)

query_4 = """
SELECT
    gender,
    COUNT(*) as total_customers,
    SUM(CASE WHEN is_default = 1 THEN 1 ELSE 0 END) as default_count,
    ROUND(100.0 * SUM(CASE WHEN is_default = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as default_rate_pct,
    ROUND(AVG(income_total), 0) as avg_income,
    ROUND(AVG(credit_amount), 0) as avg_credit,
    SUM(CASE WHEN is_default = 1 AND num_prior_credits > 2 THEN 1 ELSE 0 END) as defaults_w_prior_credits
FROM AGG_CUSTOMER_RISK_PROFILE
WHERE gender IS NOT NULL AND gender IN ('M', 'F')
GROUP BY gender
ORDER BY default_rate_pct DESC
"""

result_4 = conn.execute(query_4).fetch_df()
print(result_4.to_string(index=False))
result_4.to_csv("sql/04_default_rate_by_gender_property.csv", index=False)
print("\n✓ Saved to: sql/04_default_rate_by_gender_property.csv")

# ==============================================================================
# QUERY 5: PRIOR CREDIT HISTORY IMPACT
# ==============================================================================
print("\n5️⃣  DEFAULT RATE BY PRIOR CREDIT HISTORY")
print("-" * 80)

query_5 = """
SELECT
    CASE
        WHEN num_prior_credits = 0 THEN 'No Prior Credits'
        WHEN num_prior_credits = 1 THEN '1 Prior Credit'
        WHEN num_prior_credits BETWEEN 2 AND 5 THEN '2-5 Prior Credits'
        ELSE '6+ Prior Credits'
    END as prior_credit_segment,
    COUNT(*) as total_customers,
    SUM(CASE WHEN is_default = 1 THEN 1 ELSE 0 END) as default_count,
    ROUND(100.0 * SUM(CASE WHEN is_default = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as default_rate_pct,
    ROUND(AVG(max_overdue_ever), 0) as avg_max_overdue,
    ROUND(AVG(total_prior_debt), 0) as avg_total_prior_debt
FROM AGG_CUSTOMER_RISK_PROFILE
GROUP BY
    CASE
        WHEN num_prior_credits = 0 THEN 'No Prior Credits'
        WHEN num_prior_credits = 1 THEN '1 Prior Credit'
        WHEN num_prior_credits BETWEEN 2 AND 5 THEN '2-5 Prior Credits'
        ELSE '6+ Prior Credits'
    END
ORDER BY
    CASE
        WHEN num_prior_credits = 0 THEN 'No Prior Credits'
        WHEN num_prior_credits = 1 THEN '1 Prior Credit'
        WHEN num_prior_credits BETWEEN 2 AND 5 THEN '2-5 Prior Credits'
        ELSE '6+ Prior Credits'
    END
"""

result_5 = conn.execute(query_5).fetch_df()
print(result_5.to_string(index=False))
result_5.to_csv("sql/05_default_rate_by_prior_credits.csv", index=False)
print("\n✓ Saved to: sql/05_default_rate_by_prior_credits.csv")

# ==============================================================================
# QUERY 6: EXTERNAL CREDIT SCORE IMPACT
# ==============================================================================
print("\n6️⃣  DEFAULT RATE BY EXTERNAL CREDIT SCORE")
print("-" * 80)

query_6 = """
SELECT
    CASE
        WHEN external_score_avg < 0.25 THEN 'Very Low (< 0.25)'
        WHEN external_score_avg < 0.50 THEN 'Low (0.25-0.50)'
        WHEN external_score_avg < 0.75 THEN 'Medium (0.50-0.75)'
        ELSE 'High (0.75+)'
    END as credit_score_segment,
    COUNT(*) as total_customers,
    SUM(CASE WHEN is_default = 1 THEN 1 ELSE 0 END) as default_count,
    ROUND(100.0 * SUM(CASE WHEN is_default = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as default_rate_pct,
    ROUND(AVG(external_score_avg), 3) as avg_external_score,
    ROUND(AVG(income_total), 0) as avg_income
FROM AGG_CUSTOMER_RISK_PROFILE
WHERE external_score_avg IS NOT NULL
GROUP BY
    CASE
        WHEN external_score_avg < 0.25 THEN 'Very Low (< 0.25)'
        WHEN external_score_avg < 0.50 THEN 'Low (0.25-0.50)'
        WHEN external_score_avg < 0.75 THEN 'Medium (0.50-0.75)'
        ELSE 'High (0.75+)'
    END
ORDER BY default_rate_pct DESC
"""

result_6 = conn.execute(query_6).fetch_df()
print(result_6.to_string(index=False))
result_6.to_csv("sql/06_default_rate_by_credit_score.csv", index=False)
print("\n✓ Saved to: sql/06_default_rate_by_credit_score.csv")

# ==============================================================================
# QUERY 7: HIGH RISK COHORT ANALYSIS
# ==============================================================================
print("\n7️⃣  HIGH-RISK CUSTOMER COHORTS")
print("-" * 80)

query_7 = """
SELECT
    'High DTI + Low External Score' as high_risk_cohort,
    COUNT(*) as cohort_size,
    SUM(CASE WHEN is_default = 1 THEN 1 ELSE 0 END) as default_count,
    ROUND(100.0 * SUM(CASE WHEN is_default = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as default_rate_pct,
    ROUND(AVG(income_total), 0) as avg_income,
    ROUND(AVG(credit_amount), 0) as avg_credit,
    ROUND(AVG(debt_to_income_ratio), 2) as avg_dti
FROM AGG_CUSTOMER_RISK_PROFILE
WHERE dti_risk_level = 'High' AND external_score_avg < 0.5

UNION ALL

SELECT
    'High DTI, Low External Score, Prior Defaults' as high_risk_cohort,
    COUNT(*) as cohort_size,
    SUM(CASE WHEN is_default = 1 THEN 1 ELSE 0 END) as default_count,
    ROUND(100.0 * SUM(CASE WHEN is_default = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as default_rate_pct,
    ROUND(AVG(income_total), 0) as avg_income,
    ROUND(AVG(credit_amount), 0) as avg_credit,
    ROUND(AVG(debt_to_income_ratio), 2) as avg_dti
FROM AGG_CUSTOMER_RISK_PROFILE
WHERE dti_risk_level = 'High' AND external_score_avg < 0.5 AND max_overdue_ever > 0
"""

result_7 = conn.execute(query_7).fetch_df()
print(result_7.to_string(index=False))
result_7.to_csv("sql/07_high_risk_cohorts.csv", index=False)
print("\n✓ Saved to: sql/07_high_risk_cohorts.csv")

# ==============================================================================
# QUERY 8: REGION RATING ANALYSIS
# ==============================================================================
print("\n8️⃣  DEFAULT RATE BY REGION")
print("-" * 80)

query_8 = """
SELECT
    region_rating,
    COUNT(*) as total_customers,
    SUM(CASE WHEN is_default = 1 THEN 1 ELSE 0 END) as default_count,
    ROUND(100.0 * SUM(CASE WHEN is_default = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as default_rate_pct,
    ROUND(AVG(income_total), 0) as avg_income,
    ROUND(AVG(debt_to_income_ratio), 2) as avg_dti_ratio
FROM AGG_CUSTOMER_RISK_PROFILE
WHERE region_rating IS NOT NULL
GROUP BY region_rating
ORDER BY region_rating
"""

result_8 = conn.execute(query_8).fetch_df()
print(result_8.to_string(index=False))
result_8.to_csv("sql/08_default_rate_by_region.csv", index=False)
print("\n✓ Saved to: sql/08_default_rate_by_region.csv")

# ==============================================================================
# SUMMARY STATISTICS
# ==============================================================================
print("\n" + "="*80)
print("📊 SUMMARY STATISTICS")
print("="*80)

summary_query = """
SELECT
    COUNT(*) as total_customers,
    SUM(CASE WHEN is_default = 1 THEN 1 ELSE 0 END) as total_defaults,
    ROUND(100.0 * SUM(CASE WHEN is_default = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as overall_default_rate_pct,
    ROUND(AVG(income_total), 0) as avg_income,
    ROUND(AVG(credit_amount), 0) as avg_credit_amount,
    ROUND(AVG(debt_to_income_ratio), 2) as avg_dti_ratio,
    ROUND(AVG(external_score_avg), 3) as avg_external_score,
    ROUND(AVG(num_prior_credits), 1) as avg_prior_credits
FROM AGG_CUSTOMER_RISK_PROFILE
"""

summary_result = conn.execute(summary_query).fetch_df()
print(summary_result.to_string(index=False))

print("\n" + "="*80)
print("✓ PHASE 1 SQL ANALYSIS COMPLETE")
print("="*80)
print("\nKey Findings:")
print("  • 8 analytical views generated")
print("  • All results saved to sql/ directory as CSV files")
print("  • Ready for Phase 2: Modeling with these analytics")

conn.close()
