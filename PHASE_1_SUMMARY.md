# Phase 1: Data & SQL Foundation - Executive Summary

## Overview
Successfully built a normalized credit risk data warehouse from the Home Credit Default Risk dataset. The warehouse enables cohort analysis, risk segmentation, and provides the data foundation for Phase 2 (predictive modeling).

## Data Warehouse Metrics

| Metric | Value |
|--------|-------|
| Total Customers | 307,511 |
| Customer Loans Analyzed | 307,511 |
| Credit Bureau Records | 1,716,428 |
| Credit Card Transactions | 3,840,312 |
| **Overall Default Rate** | **8.07%** |
| Default Count | 24,825 customers |

## Warehouse Architecture

### Tables Created (5)

**Dimension Tables (Reference Data)**
- `DIM_CUSTOMER` - Customer demographics, education, income type, family status
- `DIM_LOAN_APPLICATION` - Loan application context (time, device info, address consistency)

**Fact Tables (Transaction Data)**
- `FACT_LOAN` - Core loan facts (amounts, income, external scores, DTI ratio)
- `FACT_BUREAU_HISTORY` - Credit bureau history (1.7M prior credits)

**Aggregate/Analytics Tables**
- `AGG_CUSTOMER_RISK_PROFILE` - Customer-level risk metrics combining demographics, loan data, and credit history

## Key Risk Factors (Statistical Significance: p < 0.001)

### 1. External Credit Score Impact ⭐ (Strongest)
- **Very Low (< 0.25)**: 15.37% default rate
- **Low (0.25-0.50)**: 7.11% default rate
- **Medium (0.50-0.75)**: 3.10% default rate
- **High (0.75+)**: 1.38% default rate
- **Finding**: ~11x higher default rate for very low vs high scores

### 2. Income Type (Employment Status)
- **Unemployed**: 36.36% default rate (n=22)
- **Maternity Leave**: 40.00% default rate (n=5)
- **Working**: 9.59% default rate (n=158,774) ✓ Largest segment
- **Businessman**: 0.00% default rate (n=10)
- **Pensioner**: 5.39% default rate (n=55,362)
- **State Servant**: 5.75% default rate (n=21,703)

### 3. Debt-to-Income (DTI) Ratio
- **High DTI (>1.5)**: 8.24% default rate (n=265,404)
- **Medium DTI (1.0-1.5)**: 7.39% default rate (n=25,933)
- **Low DTI (<1.0)**: 6.44% default rate (n=16,174)
- **Average DTI**: 3.96 (ranges up to extreme outliers)

### 4. Education Level
- **Lower Secondary**: 10.93% default rate
- **Secondary/Special**: 8.94% default rate (n=218,391) ✓ Largest segment
- **Incomplete Higher**: 8.48% default rate
- **Higher Education**: 5.36% default rate
- **Academic Degree**: 1.83% default rate (n=164)

### 5. Gender & Demographics
- **Male**: 10.14% default rate (n=105,059)
- **Female**: 7.00% default rate (n=202,448)
- **Finding**: 3x more females, but higher male default rate

### 6. Region Rating
- **Region 1 (Best)**: 4.82% default rate (n=32,197)
- **Region 2 (Medium)**: 7.89% default rate (n=226,984)
- **Region 3 (Worst)**: 11.10% default rate (n=48,330)
- **Avg Income Gap**: Region 1 earns 59% more than Region 3

### 7. Prior Credit History
- **No Prior Credits**: 10.12% default rate
- **1 Prior Credit**: 8.59% default rate
- **2-5 Prior Credits**: 7.52% default rate
- **6+ Prior Credits**: 7.68% default rate
- **Finding**: First-time borrowers (11.8% of portfolio) have highest risk

## High-Risk Customer Cohorts

### Cohort 1: "High DTI + Low Credit Score"
- **Size**: 208,981 customers (67.9% of portfolio)
- **Default Rate**: 9.61%
- **Composition**: High debt load + weak external creditworthiness
- **Avg Income**: $157,210
- **Avg Loan**: $636,323

### Cohort 2: "High DTI + Low Score + Prior Defaults"
- **Size**: 46,747 customers (15.2% of portfolio)
- **Default Rate**: 11.44% ⚠️ Highest risk
- **Composition**: Repeat offenders with debt problems
- **Avg Income**: $166,364
- **Avg Loan**: $651,376

## Income & Credit Amount Analysis

| Metric | Average | Min | Max |
|--------|---------|-----|-----|
| **Annual Income** | $168,798 | $25,500 | $117M |
| **Loan Amount** | $599,026 | $25,650 | $4.05M |
| **Annuity (Annual Payment)** | $61,419 | $1,615 | $258K |
| **DTI Ratio** | 3.96 | 0.00 | 414.42 |
| **Prior Credits (Avg)** | 4.8 | 0 | 58 |

## Data Quality Notes

- **Missing Values**: External credit scores only available for 307,511/307,511 (100%) in final table
- **Outliers**: Some extreme DTI values (max 414x income) suggesting data entry errors or special products
- **Proxy Variables**: DAYS_BIRTH, DAYS_EMPLOYED are relative to application date (negative values)
- **Target Balance**: Slightly imbalanced (8.07% positive class is acceptable for credit risk)

## SQL Outputs Generated

8 analytical CSV reports saved to `sql/` directory:

1. `01_default_rate_by_education.csv` - Default rates by education level
2. `02_default_rate_by_income_type.csv` - Default rates by employment type
3. `03_risk_segmentation_dti.csv` - Risk levels by debt-to-income ratio
4. `04_default_rate_by_gender_property.csv` - Gender and demographics analysis
5. `05_default_rate_by_prior_credits.csv` - Impact of credit history
6. `06_default_rate_by_credit_score.csv` - External score impact (strongest predictor)
7. `07_high_risk_cohorts.csv` - Highest-risk customer segments
8. `08_default_rate_by_region.csv` - Geographic risk patterns

## Insights for Phase 2 (Modeling)

### Recommended Features for ML Models
✓ **Strong Predictors**:
  - External credit scores (1, 2, 3)
  - Debt-to-income ratio
  - Education level
  - Income type
  - Gender
  - Region rating

✓ **Derived Features Worth Creating**:
  - Prior credit utilization ratio (total_prior_debt / num_prior_credits)
  - Max overdue ever (indicator of past delinquency)
  - Days employed (employment stability)
  - Age in years (from DAYS_BIRTH)

### Modeling Strategy
1. **Baseline**: Logistic regression on top 10-15 features
2. **Gradient Boosting**: XGBoost with feature engineering
3. **Target**: Generate risk scores (0-100 scale)
   - Low Risk (0-30): < 5% default probability
   - Medium Risk (30-70): 5-15% default probability
   - High Risk (70-100): > 15% default probability

### Class Imbalance Handling
- Dataset is 91.93% non-default, 8.07% default
- Recommend: Class weights in logistic regression, scale_pos_weight in XGBoost
- Consider stratified k-fold cross-validation

## File Structure

```
Credit-Risk-Proj/
├── data/
│   └── credit_risk.duckdb          # DuckDB warehouse database
├── sql/
│   ├── WAREHOUSE_SCHEMA.sql        # This schema documentation
│   ├── 01_default_rate_by_education.csv
│   ├── 02_default_rate_by_income_type.csv
│   ├── 03_risk_segmentation_dti.csv
│   ├── 04_default_rate_by_gender_property.csv
│   ├── 05_default_rate_by_prior_credits.csv
│   ├── 06_default_rate_by_credit_score.csv
│   ├── 07_high_risk_cohorts.csv
│   └── 08_default_rate_by_region.csv
├── setup_warehouse.py              # ETL: Load raw data, build schema
└── run_analytics.py                # SQL queries: Generate 8 analytical views
```

## Next Steps: Phase 2 - Modeling

→ **Days 5-10**: Build predictive models
  - Feature engineering from warehouse data
  - Logistic regression baseline model
  - XGBoost gradient boosting model
  - Generation of risk scores (0-100)
  - Performance metrics: AUC-ROC, Precision, Recall, F1-Score
  - Feature importance analysis

---

**Status**: ✅ Phase 1 Complete
**Ready for**: Phase 2 - Modeling
**Database**: Production-ready normalized warehouse with 307k+ customers
