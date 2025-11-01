# Airflow DAGs

Place ETL and feature engineering DAGs in this directory. Example scaffold:

- `ingest_raw.py` for loading source data into PostgreSQL.
- `feature_engineering.py` for Polars transformations and Feast materialization triggers.
