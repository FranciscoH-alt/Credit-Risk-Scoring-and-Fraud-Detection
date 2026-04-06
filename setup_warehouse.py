"""
Credit Risk Warehouse Setup - Phase 1: Data & SQL Foundation
Loads Home Credit data into DuckDB with normalized fact/dimension schema
"""

import duckdb
import pandas as pd
import os
from pathlib import Path

# Configuration
DATA_DIR = Path("/Users/franciscohenriques/Downloads/home-credit-default-risk")
DB_PATH = "data/credit_risk.duckdb"
os.makedirs("data", exist_ok=True)

# Initialize DuckDB connection
conn = duckdb.connect(DB_PATH)
print(f"✓ Connected to DuckDB at {DB_PATH}")

# ==============================================================================
# 1. LOAD RAW DATA INTO STAGING TABLES
# ==============================================================================
print("\n1. Loading raw data into staging tables...")

# Application data (main fact table source)
print("  Loading application_train.csv...")
conn.execute(f"""
    CREATE TABLE IF NOT EXISTS stg_application AS
    SELECT * FROM read_csv_auto('{DATA_DIR}/application_train.csv',
                                sample_size=50000)
""")
app_count = conn.execute("SELECT COUNT(*) as cnt FROM stg_application").fetchall()[0][0]
print(f"    ✓ Loaded {app_count:,} records")

# Bureau data (credit history)
print("  Loading bureau.csv...")
conn.execute(f"""
    CREATE TABLE IF NOT EXISTS stg_bureau AS
    SELECT * FROM read_csv_auto('{DATA_DIR}/bureau.csv',
                                sample_size=50000)
""")
bureau_count = conn.execute("SELECT COUNT(*) as cnt FROM stg_bureau").fetchall()[0][0]
print(f"    ✓ Loaded {bureau_count:,} records")

# Previous applications
print("  Loading previous_application.csv...")
conn.execute(f"""
    CREATE TABLE IF NOT EXISTS stg_previous_app AS
    SELECT * FROM read_csv_auto('{DATA_DIR}/previous_application.csv',
                                sample_size=50000)
""")
prev_count = conn.execute("SELECT COUNT(*) as cnt FROM stg_previous_app").fetchall()[0][0]
print(f"    ✓ Loaded {prev_count:,} records")

# Credit card balance
print("  Loading credit_card_balance.csv...")
conn.execute(f"""
    CREATE TABLE IF NOT EXISTS stg_credit_card_balance AS
    SELECT * FROM read_csv_auto('{DATA_DIR}/credit_card_balance.csv',
                                sample_size=50000)
""")
cc_count = conn.execute("SELECT COUNT(*) as cnt FROM stg_credit_card_balance").fetchall()[0][0]
print(f"    ✓ Loaded {cc_count:,} records")

# ==============================================================================
# 2. BUILD NORMALIZED SCHEMA (FACT + DIMENSION TABLES)
# ==============================================================================
print("\n2. Building normalized data warehouse schema...")

# DIM_CUSTOMER: Customer demographics & attributes
print("  Creating DIM_CUSTOMER...")
conn.execute("""
    CREATE TABLE IF NOT EXISTS DIM_CUSTOMER AS
    SELECT
        SK_ID_CURR as customer_id,
        CODE_GENDER as gender,
        CNT_CHILDREN as num_children,
        CNT_FAM_MEMBERS as num_family_members,
        NAME_EDUCATION_TYPE as education_level,
        NAME_FAMILY_STATUS as family_status,
        NAME_HOUSING_TYPE as housing_type,
        OCCUPATION_TYPE as occupation_type,
        ORGANIZATION_TYPE as organization_type,
        NAME_INCOME_TYPE as income_type,
        REGION_RATING_CLIENT as region_rating,
        REGION_POPULATION_RELATIVE as region_population_pct,
        FLAG_OWN_CAR as owns_car,
        FLAG_OWN_REALTY as owns_property,
        DAYS_BIRTH as days_birth,
        DAYS_EMPLOYED as days_employed,
        OWN_CAR_AGE as car_age
    FROM stg_application
    WHERE SK_ID_CURR IS NOT NULL
""")
dim_customer_count = conn.execute("SELECT COUNT(*) FROM DIM_CUSTOMER").fetchall()[0][0]
print(f"    ✓ Created DIM_CUSTOMER with {dim_customer_count:,} records")

# DIM_LOAN_APPLICATION: Loan application details
print("  Creating DIM_LOAN_APPLICATION...")
conn.execute("""
    CREATE TABLE IF NOT EXISTS DIM_LOAN_APPLICATION AS
    SELECT
        SK_ID_CURR as application_id,
        NAME_CONTRACT_TYPE as contract_type,
        WEEKDAY_APPR_PROCESS_START as application_weekday,
        HOUR_APPR_PROCESS_START as application_hour,
        REG_REGION_NOT_LIVE_REGION as region_mismatch,
        FLAG_MOBIL as has_mobile,
        FLAG_EMP_PHONE as has_emp_phone,
        FLAG_WORK_PHONE as has_work_phone,
        FLAG_EMAIL as has_email
    FROM stg_application
    WHERE SK_ID_CURR IS NOT NULL
""")
dim_app_count = conn.execute("SELECT COUNT(*) FROM DIM_LOAN_APPLICATION").fetchall()[0][0]
print(f"    ✓ Created DIM_LOAN_APPLICATION with {dim_app_count:,} records")

# FACT_LOAN: Core loan facts
print("  Creating FACT_LOAN...")
conn.execute("""
    CREATE TABLE IF NOT EXISTS FACT_LOAN AS
    SELECT
        SK_ID_CURR as loan_id,
        TARGET as defaulted,
        AMT_INCOME_TOTAL as income_total,
        AMT_CREDIT as credit_amount,
        AMT_ANNUITY as annuity_amount,
        AMT_GOODS_PRICE as goods_price,
        CAST(AMT_CREDIT AS FLOAT) / NULLIF(AMT_INCOME_TOTAL, 0) as debt_to_income_ratio,
        EXT_SOURCE_1 as external_score_1,
        EXT_SOURCE_2 as external_score_2,
        EXT_SOURCE_3 as external_score_3,
        CAST((COALESCE(EXT_SOURCE_1, 0) + COALESCE(EXT_SOURCE_2, 0) + COALESCE(EXT_SOURCE_3, 0)) / 3.0 AS FLOAT) as external_score_avg
    FROM stg_application
    WHERE SK_ID_CURR IS NOT NULL AND AMT_INCOME_TOTAL > 0
""")
fact_loan_count = conn.execute("SELECT COUNT(*) FROM FACT_LOAN").fetchall()[0][0]
print(f"    ✓ Created FACT_LOAN with {fact_loan_count:,} records")

# FACT_BUREAU_HISTORY: Credit bureau history facts
print("  Creating FACT_BUREAU_HISTORY...")
conn.execute("""
    CREATE TABLE IF NOT EXISTS FACT_BUREAU_HISTORY AS
    SELECT
        SK_ID_CURR as customer_id,
        SK_ID_BUREAU as bureau_id,
        CREDIT_ACTIVE as credit_status,
        CREDIT_TYPE as credit_type,
        DAYS_CREDIT as days_since_credit_start,
        CREDIT_DAY_OVERDUE as days_overdue,
        AMT_CREDIT_MAX_OVERDUE as max_overdue_amount,
        CNT_CREDIT_PROLONG as times_prolonged,
        AMT_CREDIT_SUM as credit_amount,
        AMT_CREDIT_SUM_DEBT as current_debt,
        AMT_CREDIT_SUM_OVERDUE as current_overdue
    FROM stg_bureau
    WHERE SK_ID_CURR IS NOT NULL
""")
fact_bureau_count = conn.execute("SELECT COUNT(*) FROM FACT_BUREAU_HISTORY").fetchall()[0][0]
print(f"    ✓ Created FACT_BUREAU_HISTORY with {fact_bureau_count:,} records")

# ==============================================================================
# 3. BUILD AGGREGATED ANALYTICS TABLES
# ==============================================================================
print("\n3. Creating aggregated analytics tables...")

# AGG_CUSTOMER_RISK_PROFILE: Customer-level risk metrics
print("  Creating AGG_CUSTOMER_RISK_PROFILE...")
conn.execute("""
    CREATE TABLE IF NOT EXISTS AGG_CUSTOMER_RISK_PROFILE AS
    SELECT
        f.loan_id,
        c.customer_id,
        c.gender,
        c.education_level,
        c.income_type,
        c.region_rating,
        f.income_total,
        f.credit_amount,
        f.debt_to_income_ratio,
        f.external_score_avg,
        COALESCE(b.num_prior_credits, 0) as num_prior_credits,
        COALESCE(b.total_prior_debt, 0) as total_prior_debt,
        COALESCE(b.max_overdue_ever, 0) as max_overdue_ever,
        f.defaulted as is_default,
        CASE
            WHEN f.debt_to_income_ratio > 1.5 THEN 'High'
            WHEN f.debt_to_income_ratio > 1.0 THEN 'Medium'
            ELSE 'Low'
        END as dti_risk_level
    FROM FACT_LOAN f
    LEFT JOIN DIM_CUSTOMER c ON f.loan_id = c.customer_id
    LEFT JOIN (
        SELECT
            customer_id,
            COUNT(*) as num_prior_credits,
            SUM(credit_amount) as total_prior_debt,
            MAX(max_overdue_amount) as max_overdue_ever
        FROM FACT_BUREAU_HISTORY
        GROUP BY customer_id
    ) b ON f.loan_id = b.customer_id
""")
agg_customer_count = conn.execute("SELECT COUNT(*) FROM AGG_CUSTOMER_RISK_PROFILE").fetchall()[0][0]
print(f"    ✓ Created AGG_CUSTOMER_RISK_PROFILE with {agg_customer_count:,} records")

print("\n" + "="*70)
print("✓ WAREHOUSE SETUP COMPLETE")
print("="*70)
print(f"Database: {DB_PATH}")
print(f"\nTables created:")
print(f"  - DIM_CUSTOMER: {dim_customer_count:,} records")
print(f"  - DIM_LOAN_APPLICATION: {dim_app_count:,} records")
print(f"  - FACT_LOAN: {fact_loan_count:,} records")
print(f"  - FACT_BUREAU_HISTORY: {fact_bureau_count:,} records")
print(f"  - AGG_CUSTOMER_RISK_PROFILE: {agg_customer_count:,} records")

# Close connection
conn.close()
print("\n✓ Database connection closed")
