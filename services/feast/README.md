# Feast Feature Store Setup

This directory contains the Feast feature store configuration for the Home Credit risk model.

## Architecture

- **Offline Store**: PostgreSQL (same database as raw data)
- **Online Store**: Redis (for low-latency serving)
- **Registry**: Local file (`data/registry.db`)

## Feature Definitions

Located in `feature_repo/`:

- **Entities**: `applicant` (identified by `SK_ID_CURR`)
- **Data Sources**: `features_source` (points to `feature_store.features` table)
- **Feature Views**: `credit_risk_features` (all 170 model features)
- **Feature Services**: `credit_risk_model_v1` (groups features for model inference)

## Prerequisites

1. **Postgres running** with features table populated:
   ```bash
   docker compose up -d postgres redis
   ```

2. **Features loaded** into `feature_store.features` table:
   ```bash
   # Run feature engineering notebook first, then:
   python scripts/load_features_to_postgres.py
   ```

3. **Environment variables** set (in `config/.env`):
   ```bash
   POSTGRES_USER=hc_admin
   POSTGRES_PASSWORD=hc_password
   POSTGRES_DB=homecredit_db
   FEAST_PROJECT=hc_risk
   ```

## Setup Instructions

### 1. Install Feast (local development)

```bash
pip install feast[postgres,redis]
```

### 2. Apply Feature Definitions

```bash
cd services/feast/feature_repo
feast apply
```

This will:
- Register all feature definitions with Feast
- Create the feature registry
- Validate data sources

### 3. Materialize Features to Online Store

```bash
# Materialize all historical data
feast materialize-incremental $(date +%Y-%m-%d)

# Or materialize specific date range
feast materialize 2024-01-01T00:00:00 2024-12-31T23:59:59
```

This loads features from PostgreSQL (offline store) into Redis (online store) for fast retrieval.

### 4. Test Feature Retrieval

```python
from feast import FeatureStore
import pandas as pd

# Initialize feature store
store = FeatureStore(repo_path="services/feast/feature_repo")

# Get features for specific applicants
entity_df = pd.DataFrame({
    "SK_ID_CURR": [100001, 100002, 100003]
})

# Fetch from online store
features = store.get_online_features(
    features=[
        "credit_risk_features:EXT_SOURCE_2",
        "credit_risk_features:EXT_SOURCE_3",
        "credit_risk_features:CREDIT_INCOME_RATIO",
        "credit_risk_features:BUREAU_DAYS_CREDIT_MAX",
    ],
    entity_rows=entity_df.to_dict("records"),
).to_df()

print(features)
```

### 5. Get All Features for Model Inference

```python
# Use feature service to get all 170 features
features = store.get_online_features(
    feature_refs=["credit_risk_model_v1:*"],
    entity_rows=[{"SK_ID_CURR": 100001}],
).to_dict()

print(f"Retrieved {len(features)} features")
```

## Common Operations

### View Registered Features

```bash
feast feature-views list
feast feature-services list
```

### Validate Feature Store

```bash
feast validate
```

### Refresh Materialization

```bash
# Run periodically to sync offline -> online
feast materialize-incremental $(date +%Y-%m-%d)
```

### Inspect Registry

```bash
feast registry-dump
```

## Integration with Model Serving API

The model serving API (`services/model_serving/`) should be updated to:

1. Initialize Feast FeatureStore on startup
2. Accept `SK_ID_CURR` instead of feature dict in requests
3. Fetch features from Feast online store using the entity ID
4. Pass features to model for prediction

Example:
```python
from feast import FeatureStore

store = FeatureStore(repo_path="/feast/feature_repo")

@app.post("/predict")
def predict(sk_id_curr: int):
    # Fetch features from Feast
    features = store.get_online_features(
        feature_refs=["credit_risk_model_v1:*"],
        entity_rows=[{"SK_ID_CURR": sk_id_curr}],
    ).to_dict()

    # Convert to array in correct order
    feature_array = [features[col][0] for col in model.feature_names_]

    # Predict
    probability = model.predict_proba([feature_array])[0, 1]
    return {"probability": probability}
```

## Troubleshooting

### "Table not found" error

Make sure features are loaded into Postgres:
```bash
python scripts/load_features_to_postgres.py
```

### "Redis connection refused"

Start Redis container:
```bash
docker compose up -d redis
```

### "No features materialized"

Run materialization:
```bash
cd services/feast/feature_repo
feast materialize-incremental $(date +%Y-%m-%d)
```

### Check materialization status

```python
from feast import FeatureStore

store = FeatureStore(repo_path=".")
print(store.list_feature_views())
```
