-- CREDIT RISK WAREHOUSE SCHEMA
-- Phase 1: Data & SQL Foundation
-- Home Credit Default Risk Project

-- ============================================================================
-- DIMENSION TABLES (D*) - Reference data about entities
-- ============================================================================

-- DIM_CUSTOMER: Customer demographic & financial profile
-- Primary Key: customer_id
CREATE TABLE DIM_CUSTOMER AS
SELECT
    SK_ID_CURR as customer_id,                    -- Unique customer identifier
    CODE_GENDER as gender,                        -- M/F
    CNT_CHILDREN as num_children,                 -- Number of children
    CNT_FAM_MEMBERS as num_family_members,        -- Total family members
    NAME_EDUCATION_TYPE as education_level,       -- Education attainment
    NAME_FAMILY_STATUS as family_status,          -- Marital status
    NAME_HOUSING_TYPE as housing_type,            -- Housing situation
    OCCUPATION_TYPE as occupation_type,           -- Job type
    ORGANIZATION_TYPE as organization_type,       -- Company type
    NAME_INCOME_TYPE as income_type,              -- Employment income type
    REGION_RATING_CLIENT as region_rating,        -- Regional risk rating (1-3)
    REGION_POPULATION_RELATIVE as region_population_pct,  -- Urbanization level
    FLAG_OWN_CAR as owns_car,                     -- Car ownership (Y/N)
    FLAG_OWN_REALTY as owns_property,             -- Property ownership (Y/N)
    DAYS_BIRTH as days_birth,                     -- Age in days at application
    DAYS_EMPLOYED as days_employed,               -- Employment duration in days
    OWN_CAR_AGE as car_age                        -- Car age in years
FROM stg_application;


-- DIM_LOAN_APPLICATION: Loan application circumstances
-- Primary Key: application_id
CREATE TABLE DIM_LOAN_APPLICATION AS
SELECT
    SK_ID_CURR as application_id,
    NAME_CONTRACT_TYPE as contract_type,          -- Cash/Revolving/etc
    WEEKDAY_APPR_PROCESS_START as application_weekday,  -- Day of week (0-6)
    HOUR_APPR_PROCESS_START as application_hour,  -- Hour of day (0-23)
    REG_REGION_NOT_LIVE_REGION as region_mismatch,      -- Address inconsistency
    FLAG_MOBIL as has_mobile,                     -- Mobile phone provided
    FLAG_EMP_PHONE as has_emp_phone,              -- Employment phone provided
    FLAG_WORK_PHONE as has_work_phone,            -- Work phone provided
    FLAG_EMAIL as has_email                       -- Email provided
FROM stg_application;


-- ============================================================================
-- FACT TABLES (FACT_*) - Transaction/event data
-- ============================================================================

-- FACT_LOAN: Core loan transaction facts
-- Primary Key: loan_id (composite with customer dimensions)
CREATE TABLE FACT_LOAN AS
SELECT
    SK_ID_CURR as loan_id,
    TARGET as defaulted,                          -- Target: 1=defaulted, 0=repaid
    AMT_INCOME_TOTAL as income_total,             -- Annual income (currency units)
    AMT_CREDIT as credit_amount,                  -- Loan amount requested
    AMT_ANNUITY as annuity_amount,                -- Annual payment amount
    AMT_GOODS_PRICE as goods_price,               -- Price of goods (consumer loans)
    CAST(AMT_CREDIT AS FLOAT) / NULLIF(AMT_INCOME_TOTAL, 0) as debt_to_income_ratio,
    EXT_SOURCE_1 as external_score_1,             -- External credit score (normalized)
    EXT_SOURCE_2 as external_score_2,             -- External credit score (normalized)
    EXT_SOURCE_3 as external_score_3,             -- External credit score (normalized)
    CAST((COALESCE(EXT_SOURCE_1, 0) + COALESCE(EXT_SOURCE_2, 0) + COALESCE(EXT_SOURCE_3, 0)) / 3.0 AS FLOAT) as external_score_avg
FROM stg_application
WHERE AMT_INCOME_TOTAL > 0;


-- FACT_BUREAU_HISTORY: Credit bureau history for each customer
-- Primary Key: (customer_id, bureau_id)
-- Grain: One row per prior credit bureau record
CREATE TABLE FACT_BUREAU_HISTORY AS
SELECT
    SK_ID_CURR as customer_id,
    SK_ID_BUREAU as bureau_id,                    -- Bureau credit identifier
    CREDIT_ACTIVE as credit_status,               -- Active/Closed status
    CREDIT_TYPE as credit_type,                   -- Car/Cash/Mortgage/etc
    DAYS_CREDIT as days_since_credit_start,       -- Time since opening (days)
    CREDIT_DAY_OVERDUE as days_overdue,           -- Current days past due
    AMT_CREDIT_MAX_OVERDUE as max_overdue_amount, -- Maximum amount overdue
    CNT_CREDIT_PROLONG as times_prolonged,        -- Number of times credit extended
    AMT_CREDIT_SUM as credit_amount,              -- Credit limit/amount
    AMT_CREDIT_SUM_DEBT as current_debt,          -- Current outstanding balance
    AMT_CREDIT_SUM_OVERDUE as current_overdue     -- Current overdue amount
FROM stg_bureau;


-- ============================================================================
-- AGGREGATED / ANALYTICS TABLES (AGG_*)
-- Pre-aggregated for faster analytical queries
-- ============================================================================

-- AGG_CUSTOMER_RISK_PROFILE: Customer-level risk assessment
-- Combines customer demographics, loan facts, and credit history
-- Primary Key: loan_id
-- Grain: One row per customer/loan application
CREATE TABLE AGG_CUSTOMER_RISK_PROFILE AS
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
    -- Risk level classification based on DTI
    CASE
        WHEN f.debt_to_income_ratio > 1.5 THEN 'High'
        WHEN f.debt_to_income_ratio > 1.0 THEN 'Medium'
        ELSE 'Low'
    END as dti_risk_level
FROM FACT_LOAN f
LEFT JOIN DIM_CUSTOMER c ON f.loan_id = c.customer_id
LEFT JOIN (
    -- Aggregate prior credit bureau history by customer
    SELECT
        customer_id,
        COUNT(*) as num_prior_credits,
        SUM(credit_amount) as total_prior_debt,
        MAX(max_overdue_amount) as max_overdue_ever
    FROM FACT_BUREAU_HISTORY
    GROUP BY customer_id
) b ON f.loan_id = b.customer_id;


-- ============================================================================
-- KEY ANALYTICAL QUERIES (EXAMPLES)
-- ============================================================================

-- Q1: Default Rate by Education Level
SELECT
    education_level,
    COUNT(*) as total_customers,
    SUM(CASE WHEN is_default = 1 THEN 1 ELSE 0 END) as default_count,
    ROUND(100.0 * SUM(CASE WHEN is_default = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as default_rate_pct
FROM AGG_CUSTOMER_RISK_PROFILE
WHERE education_level IS NOT NULL
GROUP BY education_level
ORDER BY default_rate_pct DESC;


-- Q2: Risk Cohort Analysis - High Risk Segment
SELECT
    COUNT(*) as high_risk_count,
    ROUND(100.0 * SUM(CASE WHEN is_default = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as default_rate_pct
FROM AGG_CUSTOMER_RISK_PROFILE
WHERE dti_risk_level = 'High'
  AND external_score_avg < 0.5
  AND max_overdue_ever > 0;


-- Q3: Default Rate by Income Type
SELECT
    income_type,
    COUNT(*) as total_customers,
    SUM(CASE WHEN is_default = 1 THEN 1 ELSE 0 END) as default_count,
    ROUND(100.0 * SUM(CASE WHEN is_default = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as default_rate_pct,
    ROUND(AVG(income_total), 0) as avg_income
FROM AGG_CUSTOMER_RISK_PROFILE
WHERE income_type IS NOT NULL
GROUP BY income_type
ORDER BY default_rate_pct DESC;


-- Q4: Default Rate by Prior Credit History
SELECT
    CASE
        WHEN num_prior_credits = 0 THEN 'No Prior Credits'
        WHEN num_prior_credits = 1 THEN '1 Prior Credit'
        WHEN num_prior_credits BETWEEN 2 AND 5 THEN '2-5 Prior Credits'
        ELSE '6+ Prior Credits'
    END as prior_credit_segment,
    COUNT(*) as total_customers,
    SUM(CASE WHEN is_default = 1 THEN 1 ELSE 0 END) as default_count,
    ROUND(100.0 * SUM(CASE WHEN is_default = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as default_rate_pct
FROM AGG_CUSTOMER_RISK_PROFILE
GROUP BY prior_credit_segment
ORDER BY default_rate_pct DESC;


-- Q5: Default Rate by External Credit Score
SELECT
    CASE
        WHEN external_score_avg < 0.25 THEN 'Very Low (< 0.25)'
        WHEN external_score_avg < 0.50 THEN 'Low (0.25-0.50)'
        WHEN external_score_avg < 0.75 THEN 'Medium (0.50-0.75)'
        ELSE 'High (0.75+)'
    END as credit_score_segment,
    COUNT(*) as total_customers,
    SUM(CASE WHEN is_default = 1 THEN 1 ELSE 0 END) as default_count,
    ROUND(100.0 * SUM(CASE WHEN is_default = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as default_rate_pct
FROM AGG_CUSTOMER_RISK_PROFILE
WHERE external_score_avg IS NOT NULL
GROUP BY credit_score_segment
ORDER BY default_rate_pct DESC;


-- ============================================================================
-- SCHEMA METADATA
-- ============================================================================
/*
DATASET: Home Credit Default Risk (Kaggle)
DATABASE: DuckDB (data/credit_risk.duckdb)
CREATED: Phase 1 - SQL Foundation

TABLES:
  Staging (STG_*):
    - stg_application: 307,511 customer loan applications
    - stg_bureau: 1,716,428 credit bureau history records
    - stg_previous_app: 1,670,214 previous application records
    - stg_credit_card_balance: 3,840,312 credit card balance records

  Dimension (DIM_*):
    - DIM_CUSTOMER: Customer demographics (307,511 records)
    - DIM_LOAN_APPLICATION: Loan application details (307,511 records)

  Fact (FACT_*):
    - FACT_LOAN: Core loan transactions (307,511 records)
    - FACT_BUREAU_HISTORY: Credit bureau history (1,716,428 records)

  Aggregate (AGG_*):
    - AGG_CUSTOMER_RISK_PROFILE: Customer risk metrics (307,511 records)

KEY INSIGHTS FROM PHASE 1:
  • Overall default rate: 8.07%
  • Strongest risk factors:
    - External credit score (p=0.0): 15.37% default rate for very low scores
    - Income type (p=0.0): 36.36% for unemployed, 9.59% for working
    - Debt-to-income ratio (p=0.0): 8.24% for high DTI vs 6.44% for low
    - Region rating (p=0.0): 11.10% for region 3 vs 4.82% for region 1
    - Gender (p=0.0): 10.14% for males vs 7.00% for females
    - Education (p=0.0): 10.93% for lower secondary vs 1.83% for academic degree
  • High-risk cohort: High DTI + Low external score = 9.61% default rate
  • Customers with prior defaults and high DTI = 11.44% default rate

NEXT PHASE:
  - Phase 2: Build predictive models (logistic regression, XGBoost)
  - Generate customer risk scores (0-100 scale)
  - Calculate performance metrics (AUC-ROC, Precision, Recall)
  - Feature importance analysis
*/
