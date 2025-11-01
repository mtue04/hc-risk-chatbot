# Feast Feature Repository

Define entities, feature views, and feature services here. Suggested structure:

- `entities/` – Feast entity declarations (e.g., `borrower.py`).
- `feature_views/` – Batch or streaming feature view definitions leveraging Polars transformations.
- `feature_services/` – Bundle feature views for specific model serving endpoints.
- `data_sources/` – Optional helper classes for PostgreSQL / file-backed sources.

Example command flow:

```bash
docker compose exec feast feast apply
docker compose exec feast feast materialize-incremental $(date +"%Y-%m-%dT%H:%M:%S")
```
