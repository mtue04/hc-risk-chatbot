# Airflow DAGs

Place ETL and feature engineering DAGs in this directory. Example scaffold:

- `ingest_raw.py` for loading source data into PostgreSQL.
- `feature_engineering.py` for Polars transformations and Feast materialization triggers.

## Test model serving

```bash
docker compose exec model_serving bash -lc '
python - <<"PY"
from feast import FeatureStore
store = FeatureStore(repo_path="/opt/feast/feature_repo")

feature_service = store.get_feature_service("credit_risk_model_v1")
features = store.get_online_features(
features=feature_service,
entity_rows=[{"SK_ID_CURR": 100175}],
).to_dict()
print(features)
PY'
```
