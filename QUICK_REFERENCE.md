# Quick Reference - Credit Risk Warehouse

## Access the Database
```python
import duckdb
conn = duckdb.connect("data/credit_risk.duckdb")

# Query example:
result = conn.execute("""
    SELECT education_level, COUNT(*) as cnt,
           ROUND(100.0 * SUM(is_default) / COUNT(*), 2) as default_rate
    FROM AGG_CUSTOMER_RISK_PROFILE
    GROUP BY education_level
    ORDER BY default_rate DESC
""").fetch_df()
```

## Database Tables

### Staging (STG_*)
- `stg_application` - 307,511 loan applications
- `stg_bureau` - 1,716,428 credit bureau records
- `stg_previous_app` - 1,670,214 previous applications
- `stg_credit_card_balance` - 3,840,312 transactions

### Dimensions (DIM_*)
- `DIM_CUSTOMER` - Demographics, income, education, occupation
- `DIM_LOAN_APPLICATION` - Application timing, contact info, addresses

### Facts (FACT_*)
- `FACT_LOAN` - Loan amounts, income, scores, DTI ratio
- `FACT_BUREAU_HISTORY` - Prior credits, overdue status, credit types

### Analytics (AGG_*)
- `AGG_CUSTOMER_RISK_PROFILE` - Combined risk profile for each customer

## Key Columns for Phase 2 Modeling

**Target Variable**
- `is_default` (0/1) - Payment default indicator

**Strong Features** (from Phase 1 analysis)
- `external_score_avg` - Average of 3 external credit scores
- `debt_to_income_ratio` - Loan / Income
- `education_level` - Education attainment
- `income_type` - Employment type
- `gender` - M/F
- `region_rating` - 1=best, 3=worst
- `num_prior_credits` - Count of prior credits
- `max_overdue_ever` - Worst delinquency

**Context Features**
- `income_total` - Annual income
- `credit_amount` - Loan amount
- `num_family_members` - Family size
- `owns_car`, `owns_property` - Asset ownership

## SQL Query Templates

### Default Rate by Segment
```sql
SELECT education_level,
       COUNT(*) as cnt,
       SUM(is_default) as defaults,
       ROUND(100.0 * SUM(is_default) / COUNT(*), 2) as default_rate_pct
FROM AGG_CUSTOMER_RISK_PROFILE
GROUP BY education_level
ORDER BY default_rate_pct DESC
```

### Risk Cohort Analysis
```sql
SELECT
    CASE WHEN debt_to_income_ratio > 1.5 THEN 'High'
         WHEN debt_to_income_ratio > 1.0 THEN 'Medium'
         ELSE 'Low' END as dti_level,
    COUNT(*) as cnt,
    ROUND(100.0 * SUM(is_default) / COUNT(*), 2) as default_rate_pct,
    ROUND(AVG(income_total), 0) as avg_income
FROM AGG_CUSTOMER_RISK_PROFILE
GROUP BY dti_level
```

### High-Risk Segment
```sql
SELECT COUNT(*) as high_risk_count,
       ROUND(100.0 * SUM(is_default) / COUNT(*), 2) as default_rate_pct
FROM AGG_CUSTOMER_RISK_PROFILE
WHERE debt_to_income_ratio > 1.5
  AND external_score_avg < 0.5
  AND max_overdue_ever > 0
```

## Import for Python Modeling

```python
import pandas as pd
import duckdb

conn = duckdb.connect("data/credit_risk.duckdb")

# Load for sklearn/XGBoost
df = conn.execute("""
    SELECT
        is_default,
        income_total,
        credit_amount,
        debt_to_income_ratio,
        external_score_avg,
        num_prior_credits,
        max_overdue_ever,
        education_level,
        income_type,
        gender,
        region_rating,
        num_family_members
    FROM AGG_CUSTOMER_RISK_PROFILE
    WHERE is_default IS NOT NULL
""").fetch_df()

# One-hot encode categoricals
df = pd.get_dummies(df, columns=['education_level', 'income_type', 'gender'], drop_first=True)

# Ready for train_test_split & model training
```

## File Locations
- **Database**: `data/credit_risk.duckdb` (~500 MB)
- **Schema**: `sql/WAREHOUSE_SCHEMA.sql`
- **Reports**: `sql/0*_*.csv` (8 files)
- **Summary**: `PHASE_1_SUMMARY.md`
- **Scripts**: `setup_warehouse.py`, `run_analytics.py`
