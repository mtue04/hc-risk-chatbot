"""
Feature Engineering Pipeline DAG

This DAG orchestrates the complete feature engineering pipeline:
1. Extract raw data from PostgreSQL using Polars
2. Engineer features using Polars transformations
3. Load engineered features into home_credit.features table
4. Materialize features to Feast online store (Redis)

Schedule: Daily at 2 AM
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
import polars as pl
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def read_table_via_pandas(engine, query: str) -> pl.DataFrame:
    """Read table using pandas as intermediary to avoid Polars schema inference issues."""
    df_pandas = pd.read_sql(query, engine)
    return pl.from_pandas(df_pandas)


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
    import tempfile
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

    # 5. Previous application aggregations (load table on-demand)
    logger.info("Loading previous_application table...")
    df_previous = read_table_via_pandas(engine, "SELECT * FROM home_credit.previous_application")
    df_previous = df_previous.with_columns(pl.col("SK_ID_CURR").cast(pl.Int64))
    logger.info(f"Loaded previous_application: {df_previous.shape}")

    logger.info("Creating previous application aggregations...")
    # Check which columns exist to avoid errors
    prev_cols = df_previous.columns
    logger.info(f"Previous application columns: {prev_cols}")

    prev_agg = df_previous.group_by("SK_ID_CURR").agg([
        pl.col("AMT_ANNUITY").mean().alias("PREV_AMT_ANNUITY_MEAN"),
        pl.col("AMT_ANNUITY").max().alias("PREV_AMT_ANNUITY_MAX"),
        pl.col("AMT_ANNUITY").min().alias("PREV_AMT_ANNUITY_MIN"),
        pl.col("AMT_APPLICATION").mean().alias("PREV_AMT_APPLICATION_MEAN"),
        pl.col("AMT_APPLICATION").max().alias("PREV_AMT_APPLICATION_MAX"),
        pl.col("AMT_APPLICATION").sum().alias("PREV_AMT_APPLICATION_SUM"),
        pl.col("AMT_GOODS_PRICE").mean().alias("PREV_AMT_GOODS_PRICE_MEAN"),
        pl.col("HOUR_APPR_PROCESS_START").mean().alias("PREV_HOUR_APPR_PROCESS_START_MEAN"),
        pl.col("DAYS_DECISION").mean().alias("PREV_DAYS_DECISION_MEAN"),
        pl.col("DAYS_DECISION").min().alias("PREV_DAYS_DECISION_MIN"),
        pl.col("CNT_PAYMENT").mean().alias("PREV_CNT_PAYMENT_MEAN"),
        pl.col("CNT_PAYMENT").sum().alias("PREV_CNT_PAYMENT_SUM"),
        pl.col("SK_ID_PREV").count().alias("PREV_SK_ID_PREV_COUNT"),
        (pl.col("NAME_CONTRACT_STATUS") == "Approved").cast(pl.Int32).sum().cast(pl.Float64).alias("PREV_APPROVED_COUNT"),
        pl.col("NAME_CONTRACT_TYPE").n_unique().cast(pl.Float64).alias("PREV_CONTRACT_TYPE_COUNT"),
    ])

    prev_agg = prev_agg.with_columns([
        (pl.col("PREV_APPROVED_COUNT") / pl.col("PREV_SK_ID_PREV_COUNT")).alias("PREV_APPROVAL_RATE")
    ])

    df = df.join(prev_agg, on="SK_ID_CURR", how="left")
    del df_previous, prev_agg  # Free memory
    logger.info("Previous application features complete, memory freed")

    # 6. POS cash balance aggregations (load table on-demand)
    logger.info("Loading pos_cash_balance table...")
    df_pos_cash = read_table_via_pandas(engine, "SELECT * FROM home_credit.pos_cash_balance")
    df_pos_cash = df_pos_cash.with_columns(pl.col("SK_ID_CURR").cast(pl.Int64))
    logger.info(f"Loaded pos_cash_balance: {df_pos_cash.shape}")

    logger.info("Creating POS cash aggregations...")
    pos_agg = df_pos_cash.group_by("SK_ID_CURR").agg([
        pl.col("MONTHS_BALANCE").mean().alias("POS_MONTHS_BALANCE_MEAN"),
        pl.col("MONTHS_BALANCE").max().alias("POS_MONTHS_BALANCE_MAX"),
        pl.col("CNT_INSTALMENT").mean().alias("POS_CNT_INSTALMENT_MEAN"),
        pl.col("CNT_INSTALMENT").sum().alias("POS_CNT_INSTALMENT_SUM"),
        pl.col("SK_DPD").mean().alias("POS_SK_DPD_MEAN"),
        pl.col("SK_DPD").max().alias("POS_SK_DPD_MAX"),
        pl.col("SK_DPD_DEF").mean().alias("POS_SK_DPD_DEF_MEAN"),
    ])

    df = df.join(pos_agg, on="SK_ID_CURR", how="left")
    del df_pos_cash, pos_agg  # Free memory
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

    # Close database connection
    engine.dispose()

    # 8. Advanced financial features
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

    # Fill nulls with 0 for aggregated features (no history)
    logger.info("Filling null values...")
    agg_cols = [col for col in df.columns if any(x in col for x in ['BUREAU', 'PREV', 'POS', 'INST'])]
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
    import tempfile
    output_path = f"{tempfile.gettempdir()}/engineered_features.parquet"
    df.write_parquet(output_path)

    logger.info(f"Saved engineered features to: {output_path}")

    return output_path


def load_features_to_postgres(**context):
    """
    Load engineered features into home_credit.features table.
    """
    logger.info("Loading features to PostgreSQL")

    # Get path from upstream task
    ti = context['ti']
    features_path = ti.xcom_pull(task_ids='engineer_features')

    # Read engineered features
    df = pl.read_parquet(features_path)

    logger.info(f"Loading {df.shape[0]:,} rows with {df.shape[1]} columns")

    # Get Postgres connection
    pg_hook = PostgresHook(postgres_conn_id='postgres_homecredit')
    engine = pg_hook.get_sqlalchemy_engine()

    # Create table if not exists and load data
    # Using Polars write_database (efficient bulk insert)
    df.write_database(
        table_name='features',
        connection=engine,
        if_table_exists='replace',
        engine='sqlalchemy'
    )

    logger.info(f"✓ Loaded {df.shape[0]:,} rows to home_credit.features")

    # Verify
    verify_query = "SELECT COUNT(*) as count FROM home_credit.features"
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
