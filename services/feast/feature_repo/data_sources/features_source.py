"""
Data source definition for engineered features table.

Points to the feature_store.features table in PostgreSQL which contains
all 170 pre-engineered features per applicant.
"""

from feast import Field
from feast.data_source import DataSource
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.types import Float64, Int64

# Data source pointing to materialized features table
features_source = PostgreSQLSource(
    name="hc_features_source",
    query="SELECT * FROM feature_store.features",
    timestamp_field="event_timestamp",
    description="Engineered features from Home Credit dataset (170 features)",
)
