# HomeCredit Risk Chatbot

An end‑to‑end credit‑risk analysis and conversational assistant platform. The system ingests lending data, engineers features, trains predictive models, serves real‑time scores, and exposes the results through a guided chatbot experience.

The high‑level architecture (see `architecture.png`) is composed of:

- **Orchestration:** Docker Compose coordinates local services.
- **Exploratory analysis & training:** Jupyter notebooks for EDA and model development.
- **Data pipeline:** Apache Airflow orchestrating Polars data transforms.
- **Storage:** PostgreSQL as the system of record.
- **Feature store:** Feast managing offline/online features.
- **Model serving:** FastAPI endpoint enriched with SHAP explanations.
- **Chatbot server:** LangGraph workflow with Gemini, plus helper tools for queries, inference, and plotting.
- **Interface:** Streamlit front end that surfaces chatbot insights to stakeholders.

## Initial Setup Checklist

1. **Install prerequisites**
   - Docker Engine ≥ 24 and Docker Compose plugin.
   - Python 3.11 (for local notebooks and utilities).
   - `make`, `git`, and `openssl` (Airflow + Feast bootstrapping).
2. **Clone the repository**
   ```bash
   git clone <your-fork-url> hc-risk-chatbot
   cd hc-risk-chatbot
   ```
3. **Create `config/.env`** by copying the template and filling in credentials:
   ```bash
   cp config/.env.example config/.env
   ```
   Then edit it with values such as:
   ```
   POSTGRES_USER=hc_admin
   POSTGRES_PASSWORD=hc_password
   POSTGRES_DB=homecredit_db
   ```
4. **Download the dataset** from Kaggle into `data/raw/home-credit-default-risk/` using the original file names (already present in this repo for convenience).

## Docker Compose Stack

The current `docker-compose.yml` focuses on bootstrapping the data warehouse:

- `postgres` – PostgreSQL 16 instance seeded with the Home Credit datasets. Init scripts live in `db/init/`.

## Step-by-Step Bootstrap

### 1. Build the image

```bash
docker compose build
```

The build is quick because only `postgres:16-alpine` is used.

### 2. Start PostgreSQL

```bash
docker compose up -d postgres
```

Wait for the health check to pass (`docker compose ps`).

### 3. Dataset seeding

The `db/init/` scripts automatically create the `home_credit` schema and populate the following tables during the first container start:

- `application_train`
- `application_test`
- `bureau`
- `bureau_balance`
- `credit_card_balance`
- `pos_cash_balance`

To reload everything after the container already exists, run:

```bash
make reload
```

Under the hood this executes both `/docker-entrypoint-initdb.d/02_load_application.sql` and `/docker-entrypoint-initdb.d/04_load_bureau.sql`.

### 4. Inspect the data

Connect via `psql` or your preferred client:

```bash
docker compose exec postgres psql -U "${POSTGRES_USER:-hc_admin}" -d "${POSTGRES_DB:-homecredit_db}"
```

Example queries:

```sql
SELECT TARGET, COUNT(*) FROM home_credit.application_train GROUP BY TARGET ORDER BY TARGET;
SELECT COUNT(*) FROM home_credit.bureau;
SELECT COUNT(*) FROM home_credit.credit_card_balance;
```

Additional services (Airflow, Feast, model serving, chatbot UI) will return once the raw dataset ingestion is validated.
