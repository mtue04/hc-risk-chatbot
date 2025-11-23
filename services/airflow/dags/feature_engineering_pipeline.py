"""
Feature Engineering Pipeline DAG

This DAG orchestrates the complete feature engineering pipeline:
1. Extract raw data from PostgreSQL using Polars
2. Engineer features using Polars transformations
3. Load engineered features into feature_store.features table
4. Materialize features to Feast online store (Redis)

Schedule: Daily at 2 AM
"""

import os
import tempfile
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
import polars as pl
import pandas as pd
import logging
import numpy as np
from psycopg2 import sql
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


def read_table_via_pandas(engine, query: str) -> pl.DataFrame:
    """Read table using pandas as intermediary to avoid Polars schema inference issues."""
    df_pandas = pd.read_sql(query, engine)
    return pl.from_pandas(df_pandas)


def polars_dtype_to_sql(dtype: pl.datatypes.DataType) -> str:
    """Map Polars dtype to a Postgres column type."""
    if dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
        return "DOUBLE PRECISION"
    if dtype == pl.Boolean:
        return "BOOLEAN"
    return "TEXT"


def ensure_feature_table(conn, schema: list[tuple[str, pl.datatypes.DataType]]) -> bool:
    """
    Ensure feature_store.features matches the current dataframe schema.

    Returns True if the table was recreated, False if it already matched.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'feature_store'
              AND table_name = 'features'
            ORDER BY ordinal_position
            """
        )
        existing_columns = [row[0] for row in cur.fetchall()]

    desired_columns = [name for name, _ in schema]

    if existing_columns == desired_columns:
        return False

    logger.info("Recreating feature_store.features to match current schema")
    column_definitions = sql.SQL(", ").join(
        sql.SQL("{} {}").format(sql.Identifier(name), sql.SQL(polars_dtype_to_sql(dtype)))
        for name, dtype in schema
    )

    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS feature_store.features CASCADE")
        cur.execute(
            sql.SQL("CREATE TABLE feature_store.features ({cols})").format(cols=column_definitions)
        )
        cur.execute(
            "COMMENT ON TABLE feature_store.features IS "
            "'Engineered feature matrix for Home Credit risk model'"
        )

    return True


def apply_eda_cleaning(df: pl.DataFrame) -> pl.DataFrame:
    """Replicate critical EDA cleaning steps before feature engineering."""
    logger.info("Applying EDA cleaning transformations...")

    # Replace sentinel value in DAYS_EMPLOYED
    df = df.with_columns(
        pl.when(pl.col("DAYS_EMPLOYED") == 365243)
        .then(pl.lit(None))
        .otherwise(pl.col("DAYS_EMPLOYED"))
        .alias("DAYS_EMPLOYED")
    )

    # Derive AGE_YEARS from DAYS_BIRTH
    df = df.with_columns((-pl.col("DAYS_BIRTH") / 365).alias("AGE_YEARS"))

    # Cap financial features at 1st/99th percentiles
    financial_features = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]
    quantiles = df.select(
        [pl.col(col).quantile(0.01).alias(f"{col}_p01") for col in financial_features]
        + [pl.col(col).quantile(0.99).alias(f"{col}_p99") for col in financial_features]
    )
    lower_bounds = {col: quantiles[f"{col}_p01"][0] for col in financial_features}
    upper_bounds = {col: quantiles[f"{col}_p99"][0] for col in financial_features}

    df = df.with_columns(
        [
            pl.col(col).clip(lower_bounds[col], upper_bounds[col]).alias(col)
            for col in financial_features
        ]
    )

    # Isolation Forest-based outlier detection
    features_for_iso = [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AGE_YEARS",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
    ]
    iso_input = df.select(features_for_iso).to_pandas()
    iso_input = iso_input.fillna(iso_input.median())

    iso_forest = IsolationForest(
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )
    labels = iso_forest.fit_predict(iso_input)
    scores = iso_forest.score_samples(iso_input)

    df = df.with_columns(
        [
            pl.Series("outlier_label", labels),
            pl.Series("outlier_score", scores),
        ]
    )

    logger.info("EDA cleaning complete")
    return df

# DAG default arguments
default_args = {
    'owner': 'hc_risk',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'feature_engineering_pipeline',
    default_args=default_args,
    description='Extract, transform, and materialize credit risk features using Polars',
    schedule='0 2 * * *',  # Daily at 2 AM
    catchup=False,
    tags=['feature_engineering', 'polars', 'feast'],
)


def extract_raw_data(**context):
    """
    Extract only application_train table (base table).
    Other tables will be loaded on-demand during feature engineering.

    This memory-efficient approach processes one table at a time.
    """
    logger.info("Extracting base application_train table from PostgreSQL")

    # Get Postgres connection
    pg_hook = PostgresHook(postgres_conn_id='postgres_homecredit')
    engine = pg_hook.get_sqlalchemy_engine()

    # Only read the base application table
    logger.info("Reading application_train table...")
    df_app = read_table_via_pandas(engine, "SELECT * FROM home_credit.application_train")
    # Ensure SK_ID_CURR is Int64 for consistent joins
    df_app = df_app.with_columns(pl.col("SK_ID_CURR").cast(pl.Int64))

    logger.info(f"Extracted application_train: {df_app.shape}")

    # Store connection string for later use
    temp_dir = tempfile.mkdtemp()

    app_path = f"{temp_dir}/application.parquet"
    df_app.write_parquet(app_path)

    logger.info(f"Saved base table to: {app_path}")

    return {
        'app_path': app_path,
        'temp_dir': temp_dir
    }


def engineer_features(**context):
    """
    Engineer features using Polars transformations.

    Memory-efficient approach: loads and processes one auxiliary table at a time.
    """
    logger.info("Starting feature engineering with Polars")

    # Get paths from upstream task
    ti = context['ti']
    data = ti.xcom_pull(task_ids='extract_raw_data')

    # Load base application table
    df = pl.read_parquet(data['app_path'])
    logger.info(f"Loaded application data: {df.shape}")

    # Apply cleaning derived from EDA notebooks
    df = apply_eda_cleaning(df)

    # Get Postgres connection for on-demand table loading
    pg_hook = PostgresHook(postgres_conn_id='postgres_homecredit')
    engine = pg_hook.get_sqlalchemy_engine()

    # --- Feature Engineering ---

    # 1. Basic ratio features
    logger.info("Creating basic ratio features...")
    df = df.with_columns([
        (pl.col("AMT_CREDIT") / pl.col("AMT_INCOME_TOTAL")).alias("CREDIT_INCOME_RATIO"),
        (pl.col("AMT_ANNUITY") / pl.col("AMT_INCOME_TOTAL")).alias("ANNUITY_INCOME_RATIO"),
        (pl.col("AMT_CREDIT") / pl.col("AMT_GOODS_PRICE")).alias("CREDIT_GOODS_RATIO"),
        (pl.col("AMT_ANNUITY") / pl.col("AMT_CREDIT")).alias("ANNUITY_CREDIT_RATIO"),
        (pl.col("AMT_INCOME_TOTAL") / pl.col("CNT_FAM_MEMBERS")).alias("INCOME_PER_FAMILY_MEMBER"),
        (pl.col("AMT_CREDIT") / pl.col("CNT_FAM_MEMBERS")).alias("CREDIT_PER_PERSON"),
    ])

    # 2. External source aggregations
    logger.info("Creating external source features...")
    # Cast to Float64 and fill nulls first to avoid type issues
    df = df.with_columns([
        pl.col("EXT_SOURCE_1").cast(pl.Float64),
        pl.col("EXT_SOURCE_2").cast(pl.Float64),
        pl.col("EXT_SOURCE_3").cast(pl.Float64),
    ])

    # Compute mean, max, min manually
    df = df.with_columns([
        ((pl.col("EXT_SOURCE_1").fill_null(0) + pl.col("EXT_SOURCE_2").fill_null(0) + pl.col("EXT_SOURCE_3").fill_null(0)) / 3.0).alias("EXT_SOURCE_MEAN"),
        pl.max_horizontal(["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]).alias("EXT_SOURCE_MAX"),
        pl.min_horizontal(["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]).alias("EXT_SOURCE_MIN"),
    ])

    # Compute standard deviation manually
    df = df.with_columns([
        (
            (
                (pl.col("EXT_SOURCE_1").fill_null(0) - pl.col("EXT_SOURCE_MEAN")).pow(2) +
                (pl.col("EXT_SOURCE_2").fill_null(0) - pl.col("EXT_SOURCE_MEAN")).pow(2) +
                (pl.col("EXT_SOURCE_3").fill_null(0) - pl.col("EXT_SOURCE_MEAN")).pow(2)
            ) / 3.0
        ).sqrt().alias("EXT_SOURCE_STD")
    ])

    # 3. Document and contact counts
    logger.info("Creating count features...")
    doc_cols = [col for col in df.columns if "FLAG_DOCUMENT" in col]
    contact_cols = ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"]
    region_cols = [
        "REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION", "LIVE_REGION_NOT_WORK_REGION",
        "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY"
    ]

    df = df.with_columns([
        pl.sum_horizontal(doc_cols).alias("DOCUMENT_COUNT"),
        pl.sum_horizontal(contact_cols).alias("CONTACT_COUNT"),
        pl.sum_horizontal(region_cols).alias("REGION_MISMATCH_COUNT"),
    ])

    # 4. Bureau aggregations (load table on-demand)
    logger.info("Loading bureau table...")
    df_bureau = read_table_via_pandas(engine, "SELECT * FROM home_credit.bureau")
    df_bureau = df_bureau.with_columns(pl.col("SK_ID_CURR").cast(pl.Int64))
    logger.info(f"Loaded bureau: {df_bureau.shape}")

    logger.info("Creating bureau aggregations...")
    bureau_agg = df_bureau.group_by("SK_ID_CURR").agg([
        pl.col("DAYS_CREDIT").mean().alias("BUREAU_DAYS_CREDIT_MEAN"),
        pl.col("DAYS_CREDIT").min().alias("BUREAU_DAYS_CREDIT_MIN"),
        pl.col("DAYS_CREDIT").max().alias("BUREAU_DAYS_CREDIT_MAX"),
        pl.col("CREDIT_DAY_OVERDUE").mean().alias("BUREAU_CREDIT_DAY_OVERDUE_MEAN"),
        pl.col("CREDIT_DAY_OVERDUE").max().alias("BUREAU_CREDIT_DAY_OVERDUE_MAX"),
        pl.col("DAYS_CREDIT_ENDDATE").mean().alias("BUREAU_DAYS_CREDIT_ENDDATE_MEAN"),
        pl.col("DAYS_CREDIT_ENDDATE").min().alias("BUREAU_DAYS_CREDIT_ENDDATE_MIN"),
        pl.col("AMT_CREDIT_MAX_OVERDUE").mean().alias("BUREAU_AMT_CREDIT_MAX_OVERDUE_MEAN"),
        pl.col("AMT_CREDIT_SUM").mean().alias("BUREAU_AMT_CREDIT_SUM_MEAN"),
        pl.col("AMT_CREDIT_SUM").sum().alias("BUREAU_AMT_CREDIT_SUM_SUM"),
        pl.col("AMT_CREDIT_SUM").max().alias("BUREAU_AMT_CREDIT_SUM_MAX"),
        pl.col("AMT_CREDIT_SUM_DEBT").mean().alias("BUREAU_AMT_CREDIT_SUM_DEBT_MEAN"),
        pl.col("AMT_CREDIT_SUM_DEBT").sum().alias("BUREAU_AMT_CREDIT_SUM_DEBT_SUM"),
        pl.col("AMT_CREDIT_SUM_LIMIT").mean().alias("BUREAU_AMT_CREDIT_SUM_LIMIT_MEAN"),
        pl.col("AMT_CREDIT_SUM_LIMIT").sum().alias("BUREAU_AMT_CREDIT_SUM_LIMIT_SUM"),
        pl.col("AMT_CREDIT_SUM_OVERDUE").mean().alias("BUREAU_AMT_CREDIT_SUM_OVERDUE_MEAN"),
        pl.col("AMT_CREDIT_SUM_OVERDUE").sum().alias("BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM"),
        pl.col("SK_ID_BUREAU").count().alias("BUREAU_SK_ID_BUREAU_COUNT"),
        (pl.col("CREDIT_ACTIVE") == "Active").cast(pl.Int32).sum().cast(pl.Float64).alias("BUREAU_ACTIVE_COUNT"),
        pl.col("CREDIT_TYPE").n_unique().cast(pl.Float64).alias("BUREAU_CREDIT_TYPE_COUNT"),
    ])

    # Compute advanced bureau features after aggregation
    bureau_agg = bureau_agg.with_columns([
        (pl.col("BUREAU_AMT_CREDIT_SUM_DEBT_SUM") / (pl.col("BUREAU_AMT_CREDIT_SUM_SUM") + 1.0)).alias("BUREAU_DEBT_CREDIT_RATIO"),
        (pl.col("BUREAU_AMT_CREDIT_SUM_DEBT_SUM") / (pl.col("BUREAU_AMT_CREDIT_SUM_LIMIT_MEAN") + 1.0)).alias("BUREAU_CREDIT_UTILIZATION"),
    ])

    df = df.join(bureau_agg, on="SK_ID_CURR", how="left")
    del df_bureau, bureau_agg  # Free memory
    logger.info("Bureau features complete, memory freed")

    # 5. Previous application aggregations via SQL (avoid loading full table)
    logger.info("Aggregating previous_application table via SQL...")
    prev_agg_query = """
    SELECT
        "SK_ID_CURR",
        AVG("AMT_ANNUITY") AS "PREV_AMT_ANNUITY_MEAN",
        MAX("AMT_ANNUITY") AS "PREV_AMT_ANNUITY_MAX",
        MIN("AMT_ANNUITY") AS "PREV_AMT_ANNUITY_MIN",
        AVG("AMT_APPLICATION") AS "PREV_AMT_APPLICATION_MEAN",
        MAX("AMT_APPLICATION") AS "PREV_AMT_APPLICATION_MAX",
        SUM("AMT_APPLICATION") AS "PREV_AMT_APPLICATION_SUM",
        AVG("AMT_DOWN_PAYMENT") AS "PREV_AMT_DOWN_PAYMENT_MEAN",
        MAX("AMT_DOWN_PAYMENT") AS "PREV_AMT_DOWN_PAYMENT_MAX",
        AVG("AMT_GOODS_PRICE") AS "PREV_AMT_GOODS_PRICE_MEAN",
        AVG("HOUR_APPR_PROCESS_START") AS "PREV_HOUR_APPR_PROCESS_START_MEAN",
        AVG("RATE_DOWN_PAYMENT") AS "PREV_RATE_DOWN_PAYMENT_MEAN",
        MAX("RATE_DOWN_PAYMENT") AS "PREV_RATE_DOWN_PAYMENT_MAX",
        AVG("DAYS_DECISION") AS "PREV_DAYS_DECISION_MEAN",
        MIN("DAYS_DECISION") AS "PREV_DAYS_DECISION_MIN",
        AVG("CNT_PAYMENT") AS "PREV_CNT_PAYMENT_MEAN",
        SUM("CNT_PAYMENT") AS "PREV_CNT_PAYMENT_SUM",
        COUNT("SK_ID_PREV") AS "PREV_SK_ID_PREV_COUNT",
        SUM(CASE WHEN "NAME_CONTRACT_STATUS" = 'Approved' THEN 1 ELSE 0 END)::FLOAT AS "PREV_APPROVED_COUNT",
        COUNT(DISTINCT "NAME_CONTRACT_TYPE")::FLOAT AS "PREV_CONTRACT_TYPE_COUNT"
    FROM home_credit.previous_application
    GROUP BY "SK_ID_CURR"
    """
    prev_agg = read_table_via_pandas(engine, prev_agg_query)
    prev_agg = prev_agg.with_columns([
        pl.col("SK_ID_CURR").cast(pl.Int64),
        (pl.col("PREV_APPROVED_COUNT") / pl.col("PREV_SK_ID_PREV_COUNT")).alias("PREV_APPROVAL_RATE")
    ])
    logger.info(f"Loaded previous application aggregations: {prev_agg.shape}")

    df = df.join(prev_agg, on="SK_ID_CURR", how="left")
    del prev_agg  # Free memory
    logger.info("Previous application features complete, memory freed")

    # Additional previous application sequence-based features
    logger.info("Computing previous application sequence features...")
    prev_sequence_query = """
    WITH ranked AS (
        SELECT
            "SK_ID_CURR",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "DAYS_DECISION",
            CASE WHEN "NAME_CONTRACT_STATUS" = 'Approved' THEN 1 ELSE 0 END AS approval_flag,
            ROW_NUMBER() OVER (PARTITION BY "SK_ID_CURR" ORDER BY "DAYS_DECISION" DESC) AS rn_desc,
            ROW_NUMBER() OVER (PARTITION BY "SK_ID_CURR" ORDER BY "DAYS_DECISION" ASC) AS rn_asc
        FROM home_credit.previous_application
    )
    SELECT
        "SK_ID_CURR",
        AVG(CASE WHEN rn_desc <= 3 THEN "AMT_CREDIT" END) AS "PREV_LAST3_AMT_CREDIT",
        AVG(CASE WHEN rn_desc <= 3 THEN "AMT_ANNUITY" END) AS "PREV_LAST3_AMT_ANNUITY",
        AVG(CASE WHEN rn_desc <= 3 THEN "DAYS_DECISION" END) AS "PREV_LAST3_DAYS_DECISION",
        AVG(CASE WHEN rn_desc <= 3 THEN approval_flag END)::FLOAT AS "PREV_LAST3_APPROVAL_RATE",
        AVG(CASE WHEN rn_desc <= 5 THEN "AMT_ANNUITY" END) AS "PREV_LAST5_AMT_ANNUITY",
        AVG(CASE WHEN rn_asc <= 2 THEN "AMT_CREDIT" END) AS "PREV_FIRST2_AMT_CREDIT",
        AVG(CASE WHEN rn_asc <= 2 THEN "DAYS_DECISION" END) AS "PREV_FIRST2_DAYS_DECISION",
        AVG(CASE WHEN rn_asc <= 4 THEN "AMT_CREDIT" END) AS "PREV_FIRST4_AMT_CREDIT"
    FROM ranked
    GROUP BY "SK_ID_CURR"
    """
    prev_sequence = read_table_via_pandas(engine, prev_sequence_query)
    prev_sequence = prev_sequence.with_columns(pl.col("SK_ID_CURR").cast(pl.Int64))
    df = df.join(prev_sequence, on="SK_ID_CURR", how="left")
    del prev_sequence
    logger.info("Previous application sequence features complete, memory freed")

    # 6. POS cash balance aggregations via SQL
    logger.info("Aggregating pos_cash_balance table via SQL...")
    pos_agg_query = """
    SELECT
        "SK_ID_CURR",
        AVG("MONTHS_BALANCE") AS "POS_MONTHS_BALANCE_MEAN",
        MAX("MONTHS_BALANCE") AS "POS_MONTHS_BALANCE_MAX",
        AVG("CNT_INSTALMENT") AS "POS_CNT_INSTALMENT_MEAN",
        SUM("CNT_INSTALMENT") AS "POS_CNT_INSTALMENT_SUM",
        AVG("SK_DPD") AS "POS_SK_DPD_MEAN",
        MAX("SK_DPD") AS "POS_SK_DPD_MAX",
        AVG("SK_DPD_DEF") AS "POS_SK_DPD_DEF_MEAN"
    FROM home_credit.pos_cash_balance
    GROUP BY "SK_ID_CURR"
    """
    pos_agg = read_table_via_pandas(engine, pos_agg_query)
    pos_agg = pos_agg.with_columns(pl.col("SK_ID_CURR").cast(pl.Int64))
    logger.info(f"Loaded POS cash aggregations: {pos_agg.shape}")

    df = df.join(pos_agg, on="SK_ID_CURR", how="left")
    del pos_agg
    logger.info("POS cash features complete, memory freed")

    # 7. Installments aggregations (use SQL to reduce memory)
    logger.info("Creating installments aggregations using SQL...")
    inst_agg_query = """
    SELECT
        "SK_ID_CURR",
        AVG("AMT_INSTALMENT") as "INST_AMT_INSTALMENT_MEAN",
        SUM("AMT_INSTALMENT") as "INST_AMT_INSTALMENT_SUM",
        MAX("AMT_INSTALMENT") as "INST_AMT_INSTALMENT_MAX",
        AVG("AMT_PAYMENT" - "AMT_INSTALMENT") as "INST_PAYMENT_DIFF_MEAN",
        SUM("AMT_PAYMENT" - "AMT_INSTALMENT") as "INST_PAYMENT_DIFF_SUM",
        AVG("AMT_PAYMENT" / NULLIF("AMT_INSTALMENT", 0)) as "INST_PAYMENT_RATIO_MEAN",
        MIN("AMT_PAYMENT" / NULLIF("AMT_INSTALMENT", 0)) as "INST_PAYMENT_RATIO_MIN",
        AVG("DAYS_ENTRY_PAYMENT" - "DAYS_INSTALMENT") as "INST_DAYS_DIFF_MEAN",
        MAX("DAYS_ENTRY_PAYMENT" - "DAYS_INSTALMENT") as "INST_DAYS_DIFF_MAX",
        SUM(CASE WHEN "DAYS_ENTRY_PAYMENT" > "DAYS_INSTALMENT" THEN 1 ELSE 0 END) as "INST_LATE_PAYMENT_SUM",
        AVG(CASE WHEN "DAYS_ENTRY_PAYMENT" > "DAYS_INSTALMENT" THEN 1.0 ELSE 0.0 END) as "INST_LATE_PAYMENT_MEAN"
    FROM home_credit.installments_payments
    GROUP BY "SK_ID_CURR"
    """
    inst_agg = read_table_via_pandas(engine, inst_agg_query)
    inst_agg = inst_agg.with_columns(pl.col("SK_ID_CURR").cast(pl.Int64))
    logger.info(f"Loaded installments aggregations: {inst_agg.shape}")

    df = df.join(inst_agg, on="SK_ID_CURR", how="left")
    del inst_agg  # Free memory
    logger.info("Installments features complete, memory freed")

    # 8. Past due severity features (derived from installments_payments)
    logger.info("Computing past due severity features...")
    past_due_query = """
    SELECT
        "SK_ID_CURR",
        AVG(GREATEST("DAYS_ENTRY_PAYMENT" - "DAYS_INSTALMENT", 0)) AS "PAST_DUE_DAYS_PAST_DUE_MEAN",
        SUM(GREATEST("DAYS_ENTRY_PAYMENT" - "DAYS_INSTALMENT", 0)) AS "PAST_DUE_DAYS_PAST_DUE_SUM",
        STDDEV_POP(CASE WHEN "AMT_INSTALMENT" > 0 THEN "AMT_PAYMENT" / NULLIF("AMT_INSTALMENT", 0) END) AS "PAST_DUE_PAYMENT_RATIO_STD",
        AVG(CASE WHEN "DAYS_ENTRY_PAYMENT" > "DAYS_INSTALMENT" THEN ("DAYS_ENTRY_PAYMENT" - "DAYS_INSTALMENT") ELSE 0 END) AS "PAST_DUE_SEVERITY"
    FROM home_credit.installments_payments
    GROUP BY "SK_ID_CURR"
    """
    past_due_agg = read_table_via_pandas(engine, past_due_query)
    past_due_agg = past_due_agg.with_columns(pl.col("SK_ID_CURR").cast(pl.Int64))
    df = df.join(past_due_agg, on="SK_ID_CURR", how="left")
    del past_due_agg
    logger.info("Past due features complete, memory freed")

    # Close database connection
    engine.dispose()

    # 9. Advanced financial features
    logger.info("Creating advanced financial features...")
    df = df.with_columns([
        (pl.col("AMT_CREDIT") / (pl.col("EXT_SOURCE_3") + 0.00001)).alias("AMT_CREDIT_div_EXT_SOURCE_3"),
        (pl.col("AMT_ANNUITY") / (pl.col("EXT_SOURCE_3") + 0.00001)).alias("AMT_ANNUITY_div_EXT_SOURCE_3"),
        (pl.col("AMT_INCOME_TOTAL") / (pl.col("EXT_SOURCE_3") + 0.00001)).alias("AMT_INCOME_TOTAL_div_EXT_SOURCE_3"),
        (pl.col("AMT_ANNUITY") * pl.col("PREV_CNT_PAYMENT_MEAN")).alias("ESTIMATED_TOTAL_PAYMENT"),
        (pl.col("AMT_CREDIT") / (pl.col("AMT_ANNUITY") * 12 + 1)).alias("CREDIT_ANNUITY_YEARS"),
        (pl.col("AMT_INCOME_TOTAL") / ((-pl.col("DAYS_EMPLOYED") / 365) + 1)).alias("INCOME_PER_YEAR_EMPLOYED"),
    ])

    df = df.with_columns([
        (pl.col("ESTIMATED_TOTAL_PAYMENT") - pl.col("AMT_CREDIT")).alias("ESTIMATED_INTEREST"),
    ])

    df = df.with_columns([
        (pl.col("ESTIMATED_INTEREST") / (pl.col("AMT_CREDIT") + 1)).alias("ESTIMATED_INTEREST_RATE"),
        (pl.col("AMT_GOODS_PRICE") - pl.col("AMT_CREDIT")).alias("DOWN_PAYMENT_AMT"),
    ])

    df = df.with_columns([
        (pl.col("ESTIMATED_INTEREST_RATE") / ((pl.col("PREV_CNT_PAYMENT_MEAN") / 12) + 0.001)).alias("ESTIMATED_YEARLY_RATE"),
    ])

    # Placeholder for KNN-based feature (to be replaced with actual computation)
    if "KNN_TARGET_MEAN_500" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("KNN_TARGET_MEAN_500"))

    # Fill nulls with 0 for aggregated features (no history)
    logger.info("Filling null values...")
    agg_cols = [col for col in df.columns if any(x in col for x in ['BUREAU', 'PREV', 'POS', 'INST', 'PAST'])]
    df = df.with_columns([
        pl.col(col).fill_null(0) for col in agg_cols
    ])

    # Fill remaining nulls with median for base features
    base_numeric_cols = [
        col for col in df.select(pl.col(pl.NUMERIC_DTYPES)).columns
        if col not in agg_cols and col not in ["SK_ID_CURR", "TARGET"]
    ]

    for col in base_numeric_cols:
        median_val = df[col].median()
        if median_val is not None:
            df = df.with_columns([pl.col(col).fill_null(median_val)])

    logger.info(f"Final engineered features shape: {df.shape}")

    # Save to parquet
    event_ts = datetime.utcnow()
    df = df.with_columns(pl.lit(event_ts).alias("event_timestamp"))

    output_path = f"{tempfile.gettempdir()}/engineered_features.parquet"
    df.write_parquet(output_path)

    logger.info(f"Saved engineered features to: {output_path}")

    return output_path


def load_features_to_postgres(**context):
    """
    Load engineered features into feature_store.features table.
    """
    logger.info("Loading features to PostgreSQL")

    # Get path from upstream task
    ti = context['ti']
    features_path = ti.xcom_pull(task_ids='engineer_features')

    # Read engineered features
    df = pl.read_parquet(features_path)
    logger.info(f"Loading {df.shape[0]:,} rows with {df.shape[1]} columns")

    schema = list(zip(df.columns, df.dtypes))
    columns = df.columns

    # Export to temporary CSV for efficient COPY
    tmp_csv = tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False)
    tmp_csv_path = tmp_csv.name
    tmp_csv.close()

    logger.info(f"Writing temporary CSV to {tmp_csv_path}")
    if "event_timestamp" in df.columns:
        df_csv = df.with_columns(
            pl.col("event_timestamp")
            .dt.replace_time_zone("UTC")
            .dt.strftime("%Y-%m-%d %H:%M:%S%z")
            .alias("event_timestamp")
        )
    else:
        df_csv = df
    df_csv.write_csv(tmp_csv_path)

    pg_hook = PostgresHook(postgres_conn_id='postgres_homecredit')
    conn = pg_hook.get_conn()
    conn.autocommit = False

    try:
        recreated = ensure_feature_table(conn, schema)

        with conn.cursor() as cur:
            logger.info("Truncating feature_store.features")
            cur.execute("TRUNCATE TABLE feature_store.features")

            copy_sql = sql.SQL("COPY feature_store.features ({fields}) FROM STDIN WITH CSV HEADER").format(
                fields=sql.SQL(", ").join(sql.Identifier(col) for col in columns)
            )

            logger.info("Copying data into PostgreSQL via COPY ...")
            with open(tmp_csv_path, "r", encoding="utf-8") as csv_file:
                cur.copy_expert(copy_sql.as_string(cur), csv_file)

            # Ensure event_timestamp column uses timestamptz type
            cur.execute(
                """
                ALTER TABLE feature_store.features
                ALTER COLUMN event_timestamp
                TYPE timestamptz
                USING event_timestamp::timestamptz
                """
            )

        conn.commit()
        logger.info(f"✓ Loaded {df.shape[0]:,} rows to feature_store.features")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
        if os.path.exists(tmp_csv_path):
            os.remove(tmp_csv_path)
            logger.info("Temporary CSV removed")

    # Verify
    verify_query = "SELECT COUNT(*) as count FROM feature_store.features"
    result = pg_hook.get_first(verify_query)
    logger.info(f"✓ Verified: {result[0]:,} rows in features table")


# Define tasks
task_extract = PythonOperator(
    task_id='extract_raw_data',
    python_callable=extract_raw_data,
    dag=dag,
)

task_engineer = PythonOperator(
    task_id='engineer_features',
    python_callable=engineer_features,
    dag=dag,
)

task_load = PythonOperator(
    task_id='load_to_postgres',
    python_callable=load_features_to_postgres,
    dag=dag,
)

task_feast_apply = BashOperator(
    task_id='feast_apply',
    bash_command='cd /opt/feast/feature_repo && feast apply',
    dag=dag,
)

task_feast_materialize = BashOperator(
    task_id='feast_materialize',
    bash_command='cd /opt/feast/feature_repo && feast materialize-incremental $(date +%Y-%m-%d)',
    dag=dag,
)

# Define dependencies
task_extract >> task_engineer >> task_load >> task_feast_apply >> task_feast_materialize
