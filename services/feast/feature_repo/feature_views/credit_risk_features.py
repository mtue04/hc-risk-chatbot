"""
Feature view for Home Credit risk model features.

Contains all 170 engineered features used by the LightGBM credit risk model.
Features include:
- Base application features (demographics, amounts, flags)
- Engineered ratios and interactions
- Bureau credit history aggregations
- Previous application statistics
- POS cash balance and installment payment behavior
- KNN-based target encoding
- Advanced financial indicators
"""

from datetime import timedelta
from feast import Feature, FeatureView, Field
from feast.types import Float64, Int64

from entities.applicant import applicant
from data_sources.features_source import features_source


# Define all 170 model features
credit_risk_features = FeatureView(
    name="credit_risk_features",
    entities=[applicant],
    schema=[
        # Base application features (categorical encoded as floats)
        Field(name="NAME_CONTRACT_TYPE", dtype=Float64),
        Field(name="CODE_GENDER", dtype=Float64),
        Field(name="FLAG_OWN_CAR", dtype=Float64),
        Field(name="FLAG_OWN_REALTY", dtype=Float64),
        Field(name="CNT_CHILDREN", dtype=Float64),
        Field(name="AMT_INCOME_TOTAL", dtype=Float64),
        Field(name="AMT_CREDIT", dtype=Float64),
        Field(name="AMT_ANNUITY", dtype=Float64),
        Field(name="NAME_TYPE_SUITE", dtype=Float64),
        Field(name="NAME_INCOME_TYPE", dtype=Float64),
        Field(name="NAME_EDUCATION_TYPE", dtype=Float64),
        Field(name="NAME_FAMILY_STATUS", dtype=Float64),
        Field(name="NAME_HOUSING_TYPE", dtype=Float64),
        Field(name="REGION_POPULATION_RELATIVE", dtype=Float64),
        Field(name="DAYS_BIRTH", dtype=Float64),
        Field(name="DAYS_EMPLOYED", dtype=Float64),
        Field(name="DAYS_REGISTRATION", dtype=Float64),
        Field(name="DAYS_ID_PUBLISH", dtype=Float64),
        Field(name="FLAG_MOBIL", dtype=Float64),
        Field(name="FLAG_EMP_PHONE", dtype=Float64),
        Field(name="FLAG_WORK_PHONE", dtype=Float64),
        Field(name="FLAG_CONT_MOBILE", dtype=Float64),
        Field(name="FLAG_PHONE", dtype=Float64),
        Field(name="FLAG_EMAIL", dtype=Float64),
        Field(name="OCCUPATION_TYPE", dtype=Float64),
        Field(name="CNT_FAM_MEMBERS", dtype=Float64),
        Field(name="REGION_RATING_CLIENT", dtype=Float64),
        Field(name="WEEKDAY_APPR_PROCESS_START", dtype=Float64),
        Field(name="HOUR_APPR_PROCESS_START", dtype=Float64),
        Field(name="REG_REGION_NOT_LIVE_REGION", dtype=Float64),
        Field(name="REG_REGION_NOT_WORK_REGION", dtype=Float64),
        Field(name="LIVE_REGION_NOT_WORK_REGION", dtype=Float64),
        Field(name="REG_CITY_NOT_LIVE_CITY", dtype=Float64),
        Field(name="REG_CITY_NOT_WORK_CITY", dtype=Float64),
        Field(name="LIVE_CITY_NOT_WORK_CITY", dtype=Float64),
        Field(name="ORGANIZATION_TYPE", dtype=Float64),
        Field(name="EXT_SOURCE_2", dtype=Float64),
        Field(name="EXT_SOURCE_3", dtype=Float64),
        Field(name="YEARS_BEGINEXPLUATATION_AVG", dtype=Float64),
        Field(name="FLOORSMAX_AVG", dtype=Float64),
        Field(name="TOTALAREA_MODE", dtype=Float64),
        Field(name="EMERGENCYSTATE_MODE", dtype=Float64),
        Field(name="OBS_30_CNT_SOCIAL_CIRCLE", dtype=Float64),
        Field(name="DEF_30_CNT_SOCIAL_CIRCLE", dtype=Float64),
        Field(name="DEF_60_CNT_SOCIAL_CIRCLE", dtype=Float64),
        Field(name="DAYS_LAST_PHONE_CHANGE", dtype=Float64),

        # Document flags
        Field(name="FLAG_DOCUMENT_2", dtype=Float64),
        Field(name="FLAG_DOCUMENT_3", dtype=Float64),
        Field(name="FLAG_DOCUMENT_4", dtype=Float64),
        Field(name="FLAG_DOCUMENT_5", dtype=Float64),
        Field(name="FLAG_DOCUMENT_6", dtype=Float64),
        Field(name="FLAG_DOCUMENT_7", dtype=Float64),
        Field(name="FLAG_DOCUMENT_8", dtype=Float64),
        Field(name="FLAG_DOCUMENT_9", dtype=Float64),
        Field(name="FLAG_DOCUMENT_10", dtype=Float64),
        Field(name="FLAG_DOCUMENT_11", dtype=Float64),
        Field(name="FLAG_DOCUMENT_12", dtype=Float64),
        Field(name="FLAG_DOCUMENT_13", dtype=Float64),
        Field(name="FLAG_DOCUMENT_14", dtype=Float64),
        Field(name="FLAG_DOCUMENT_15", dtype=Float64),
        Field(name="FLAG_DOCUMENT_16", dtype=Float64),
        Field(name="FLAG_DOCUMENT_17", dtype=Float64),
        Field(name="FLAG_DOCUMENT_18", dtype=Float64),
        Field(name="FLAG_DOCUMENT_19", dtype=Float64),
        Field(name="FLAG_DOCUMENT_20", dtype=Float64),
        Field(name="FLAG_DOCUMENT_21", dtype=Float64),

        # Credit bureau request counts
        Field(name="AMT_REQ_CREDIT_BUREAU_HOUR", dtype=Float64),
        Field(name="AMT_REQ_CREDIT_BUREAU_DAY", dtype=Float64),
        Field(name="AMT_REQ_CREDIT_BUREAU_WEEK", dtype=Float64),
        Field(name="AMT_REQ_CREDIT_BUREAU_MON", dtype=Float64),
        Field(name="AMT_REQ_CREDIT_BUREAU_QRT", dtype=Float64),
        Field(name="AMT_REQ_CREDIT_BUREAU_YEAR", dtype=Float64),

        # Anomaly detection features
        Field(name="outlier_label", dtype=Float64),
        Field(name="outlier_score", dtype=Float64),

        # Engineered ratio features
        Field(name="CREDIT_INCOME_RATIO", dtype=Float64),
        Field(name="ANNUITY_INCOME_RATIO", dtype=Float64),
        Field(name="CREDIT_GOODS_RATIO", dtype=Float64),
        Field(name="ANNUITY_CREDIT_RATIO", dtype=Float64),
        Field(name="INCOME_PER_FAMILY_MEMBER", dtype=Float64),
        Field(name="CREDIT_PER_PERSON", dtype=Float64),

        # External source combinations
        Field(name="EXT_SOURCE_MEAN", dtype=Float64),
        Field(name="EXT_SOURCE_MAX", dtype=Float64),
        Field(name="EXT_SOURCE_MIN", dtype=Float64),
        Field(name="EXT_SOURCE_STD", dtype=Float64),

        # Count features
        Field(name="DOCUMENT_COUNT", dtype=Float64),
        Field(name="CONTACT_COUNT", dtype=Float64),
        Field(name="REGION_MISMATCH_COUNT", dtype=Float64),

        # Bureau aggregations
        Field(name="BUREAU_DAYS_CREDIT_MEAN", dtype=Float64),
        Field(name="BUREAU_DAYS_CREDIT_MIN", dtype=Float64),
        Field(name="BUREAU_DAYS_CREDIT_MAX", dtype=Float64),
        Field(name="BUREAU_CREDIT_DAY_OVERDUE_MEAN", dtype=Float64),
        Field(name="BUREAU_CREDIT_DAY_OVERDUE_MAX", dtype=Float64),
        Field(name="BUREAU_DAYS_CREDIT_ENDDATE_MEAN", dtype=Float64),
        Field(name="BUREAU_DAYS_CREDIT_ENDDATE_MIN", dtype=Float64),
        Field(name="BUREAU_AMT_CREDIT_MAX_OVERDUE_MEAN", dtype=Float64),
        Field(name="BUREAU_AMT_CREDIT_SUM_MEAN", dtype=Float64),
        Field(name="BUREAU_AMT_CREDIT_SUM_SUM", dtype=Float64),
        Field(name="BUREAU_AMT_CREDIT_SUM_MAX", dtype=Float64),
        Field(name="BUREAU_AMT_CREDIT_SUM_DEBT_MEAN", dtype=Float64),
        Field(name="BUREAU_AMT_CREDIT_SUM_DEBT_SUM", dtype=Float64),
        Field(name="BUREAU_AMT_CREDIT_SUM_LIMIT_MEAN", dtype=Float64),
        Field(name="BUREAU_AMT_CREDIT_SUM_LIMIT_SUM", dtype=Float64),
        Field(name="BUREAU_AMT_CREDIT_SUM_OVERDUE_MEAN", dtype=Float64),
        Field(name="BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM", dtype=Float64),
        Field(name="BUREAU_SK_ID_BUREAU_COUNT", dtype=Float64),
        Field(name="BUREAU_ACTIVE_COUNT", dtype=Float64),
        Field(name="BUREAU_CREDIT_TYPE_COUNT", dtype=Float64),

        # Previous application aggregations
        Field(name="PREV_AMT_ANNUITY_MEAN", dtype=Float64),
        Field(name="PREV_AMT_ANNUITY_MAX", dtype=Float64),
        Field(name="PREV_AMT_ANNUITY_MIN", dtype=Float64),
        Field(name="PREV_AMT_APPLICATION_MEAN", dtype=Float64),
        Field(name="PREV_AMT_APPLICATION_MAX", dtype=Float64),
        Field(name="PREV_AMT_APPLICATION_SUM", dtype=Float64),
        Field(name="PREV_AMT_DOWN_PAYMENT_MEAN", dtype=Float64),
        Field(name="PREV_AMT_DOWN_PAYMENT_MAX", dtype=Float64),
        Field(name="PREV_AMT_GOODS_PRICE_MEAN", dtype=Float64),
        Field(name="PREV_HOUR_APPR_PROCESS_START_MEAN", dtype=Float64),
        Field(name="PREV_RATE_DOWN_PAYMENT_MEAN", dtype=Float64),
        Field(name="PREV_RATE_DOWN_PAYMENT_MAX", dtype=Float64),
        Field(name="PREV_DAYS_DECISION_MEAN", dtype=Float64),
        Field(name="PREV_DAYS_DECISION_MIN", dtype=Float64),
        Field(name="PREV_CNT_PAYMENT_MEAN", dtype=Float64),
        Field(name="PREV_CNT_PAYMENT_SUM", dtype=Float64),
        Field(name="PREV_SK_ID_PREV_COUNT", dtype=Float64),
        Field(name="PREV_APPROVED_COUNT", dtype=Float64),
        Field(name="PREV_CONTRACT_TYPE_COUNT", dtype=Float64),
        Field(name="PREV_APPROVAL_RATE", dtype=Float64),

        # POS cash balance aggregations
        Field(name="POS_MONTHS_BALANCE_MEAN", dtype=Float64),
        Field(name="POS_MONTHS_BALANCE_MAX", dtype=Float64),
        Field(name="POS_CNT_INSTALMENT_MEAN", dtype=Float64),
        Field(name="POS_CNT_INSTALMENT_SUM", dtype=Float64),
        Field(name="POS_SK_DPD_MEAN", dtype=Float64),
        Field(name="POS_SK_DPD_MAX", dtype=Float64),
        Field(name="POS_SK_DPD_DEF_MEAN", dtype=Float64),

        # Installment payment aggregations
        Field(name="INST_AMT_INSTALMENT_MEAN", dtype=Float64),
        Field(name="INST_AMT_INSTALMENT_SUM", dtype=Float64),
        Field(name="INST_AMT_INSTALMENT_MAX", dtype=Float64),
        Field(name="INST_PAYMENT_DIFF_MEAN", dtype=Float64),
        Field(name="INST_PAYMENT_DIFF_SUM", dtype=Float64),
        Field(name="INST_PAYMENT_RATIO_MEAN", dtype=Float64),
        Field(name="INST_PAYMENT_RATIO_MIN", dtype=Float64),
        Field(name="INST_DAYS_DIFF_MEAN", dtype=Float64),
        Field(name="INST_DAYS_DIFF_MAX", dtype=Float64),
        Field(name="INST_LATE_PAYMENT_SUM", dtype=Float64),
        Field(name="INST_LATE_PAYMENT_MEAN", dtype=Float64),

        # KNN-based target encoding
        Field(name="KNN_TARGET_MEAN_500", dtype=Float64),

        # Time-sliced previous application features
        Field(name="PREV_LAST3_AMT_CREDIT", dtype=Float64),
        Field(name="PREV_LAST3_AMT_ANNUITY", dtype=Float64),
        Field(name="PREV_LAST3_DAYS_DECISION", dtype=Float64),
        Field(name="PREV_LAST3_APPROVAL_RATE", dtype=Float64),
        Field(name="PREV_LAST5_AMT_ANNUITY", dtype=Float64),
        Field(name="PREV_FIRST2_AMT_CREDIT", dtype=Float64),
        Field(name="PREV_FIRST2_DAYS_DECISION", dtype=Float64),
        Field(name="PREV_FIRST4_AMT_CREDIT", dtype=Float64),

        # Past due analysis
        Field(name="PAST_DUE_DAYS_PAST_DUE_MEAN", dtype=Float64),
        Field(name="PAST_DUE_DAYS_PAST_DUE_SUM", dtype=Float64),
        Field(name="PAST_DUE_PAYMENT_RATIO_STD", dtype=Float64),
        Field(name="PAST_DUE_SEVERITY", dtype=Float64),

        # Advanced bureau features
        Field(name="BUREAU_DEBT_CREDIT_RATIO", dtype=Float64),
        Field(name="BUREAU_CREDIT_UTILIZATION", dtype=Float64),

        # EXT_SOURCE normalized features
        Field(name="AMT_CREDIT_div_EXT_SOURCE_3", dtype=Float64),
        Field(name="AMT_ANNUITY_div_EXT_SOURCE_3", dtype=Float64),
        Field(name="AMT_INCOME_TOTAL_div_EXT_SOURCE_3", dtype=Float64),

        # Interest rate estimates
        Field(name="ESTIMATED_TOTAL_PAYMENT", dtype=Float64),
        Field(name="ESTIMATED_INTEREST", dtype=Float64),
        Field(name="ESTIMATED_INTEREST_RATE", dtype=Float64),
        Field(name="ESTIMATED_YEARLY_RATE", dtype=Float64),

        # Down payment and credit duration
        Field(name="DOWN_PAYMENT_AMT", dtype=Float64),
        Field(name="CREDIT_ANNUITY_YEARS", dtype=Float64),
        Field(name="INCOME_PER_YEAR_EMPLOYED", dtype=Float64),
    ],
    source=features_source,
    ttl=timedelta(days=36500),  # ~100 years (features don't expire)
    online=True,  # Enable online serving
    description="Home Credit risk model features (170 engineered features)",
    tags={"team": "credit_risk", "model": "lgbm_tuned_v1"},
)
