# 🚀 Deployment Guide - HomeCredit Risk Chatbot

This guide provides step-by-step instructions to deploy the complete HomeCredit Risk Chatbot system.

---

## 📋 Prerequisites

Before starting, ensure you have:

- ✅ **Docker Engine** ≥ 24.0 with Docker Compose plugin
- ✅ **Python** 3.11+ (for local development/testing)
- ✅ **Git** for version control
- ✅ **Gemini API Key** (from Google AI Studio)
- ✅ **8GB RAM** minimum for all services
- ✅ **10GB disk space** for data and models

---

## ⚙️ Step 1: Environment Configuration

### 1.1 Configure Environment Variables

The `.env` file has already been created in `config/.env`. Update the following:

```bash
# Edit config/.env
nano config/.env
```

**Critical values to update:**

```bash
# Gemini API Key - REQUIRED for chatbot functionality
GEMINI_API_KEY=your_actual_gemini_api_key_here

# Database credentials (default is fine for development)
POSTGRES_USER=hc_admin
POSTGRES_PASSWORD=hc_password
POSTGRES_DB=homecredit_db

# Airflow credentials
AIRFLOW_USERNAME=admin
AIRFLOW_PASSWORD=admin
```

> [!IMPORTANT]
> **Getting a Gemini API Key:**
> 1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
> 2. Click "Create API Key"
> 3. Copy the key and paste into `config/.env`

### 1.2 Verify Configuration

```bash
# Check that .env file exists
ls -la config/.env

# Verify it's not the example file
cat config/.env | grep GEMINI_API_KEY
```

---

## 🐳 Step 2: Build Docker Images

### 2.1 Build All Services

```bash
# From project root
cd d:/university/S10/PTDLTM/hc-risk-chatbot

# Build all images
docker compose build

# This will build:
# - postgres:16-alpine (lightweight)
# - redis:7-alpine
# - Custom Airflow image with Polars
# - Model serving API with LightGBM
# - Chatbot with LangGraph + Gemini
```

**Expected build time:** 5-10 minutes

### 2.2 Verify Images

```bash
docker images | grep hc-risk-chatbot
```

You should see images for `airflow`, `model_serving`, and `chatbot`.

---

## 🗄️ Step 3: Start Core Infrastructure

### 3.1 Start Database and Cache

```bash
# Start PostgreSQL and Redis first
docker compose up -d postgres redis

# Wait for health checks to pass
docker compose ps

# Both should show "healthy" status
```

### 3.2 Verify Database Initialization

The PostgreSQL container automatically:
- Creates the `home_credit` schema
- Loads all 6 data tables from CSV files
- Creates the `feature_store` schema for Feast

**Verify:**

```bash
# Connect to PostgreSQL
docker compose exec postgres psql -U hc_admin -d homecredit_db

# Run verification queries
SELECT COUNT(*) FROM home_credit.application_train;
-- Expected: ~307,000 rows

SELECT COUNT(*) FROM home_credit.bureau;
-- Expected: ~1.7M rows

\q  # Exit psql
```

---

## 🔄 Step 4: Feature Engineering Pipeline

### 4.1 Start Airflow

```bash
# Start Airflow orchestrator
docker compose up -d airflow

# Wait ~60 seconds for initialization
sleep 60

# Check status
docker compose ps airflow
```

### 4.2 Access Airflow UI

1. Open browser: http://localhost:8080
2. Login with credentials from `.env`:
   - Username: `admin`
   - Password: `admin`

### 4.3 Run Feature Engineering DAG

**Via UI:**
1. Find DAG: `feature_engineering_pipeline`
2. Toggle it to "ON"
3. Click "Trigger DAG" (play button)
4. Monitor progress in Graph view

**Via CLI:**

```bash
# Trigger the DAG
docker compose exec airflow airflow dags trigger feature_engineering_pipeline

# Monitor logs
docker compose logs -f airflow
```

**Expected runtime:** 10-20 minutes

This DAG will:
- Extract data from PostgreSQL using Polars
- Engineer 170 features
- Load to `feature_store.features` table
- Register features with Feast
- Materialize to Redis online store

### 4.4 Verify Feature Store

```bash
# Check features table
docker compose exec postgres psql -U hc_admin -d homecredit_db \
  -c "SELECT COUNT(*) FROM feature_store.features;"

# Should show ~307,000 rows
```

---

## 🤖 Step 5: Model Serving API

### 5.1 Start Model Service

```bash
docker compose up -d model_serving

# Wait for startup
sleep 10

# Check health
curl http://localhost:8001/health
```

**Expected response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "feast_connected": true
}
```

### 5.2 Test Prediction Endpoint

```bash
# Get prediction for applicant 100001
curl http://localhost:8001/predict/applicant/100001

# Expected response includes:
# - probability: 0.0 to 1.0
# - prediction: "Low Risk" or "High Risk"
# - features_used: 170
```

### 5.3 Test SHAP Explanation

```bash
# Get SHAP explanation
curl http://localhost:8001/explain/applicant/100001

# Returns top feature contributions
```

---

## 💬 Step 6: Chatbot Service

### 6.1 Start Chatbot API

```bash
docker compose up -d chatbot

# Wait for LangGraph initialization
sleep 5

# Check health
curl http://localhost:8500/health
```

**Expected response:**
```json
{
  "status": "ok",
  "version": "0.2.0",
  "langgraph_enabled": true,
  "gemini_configured": true
}
```

> [!WARNING]
> If `gemini_configured: false`, the chatbot will work in limited mode without natural language understanding. Update `GEMINI_API_KEY` in `config/.env` and restart.

### 6.2 Test Chat Endpoint

```bash
# Send a test question
curl -X POST http://localhost:8500/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the risk score for applicant 100001?",
    "applicant_id": 100001
  }'
```

**Expected:** Natural language response with risk analysis

---

## 🎨 Step 7: Streamlit UI

### 7.1 Start Streamlit

```bash
docker compose up -d streamlit

# Check logs
docker compose logs streamlit
```

### 7.2 Access UI

Open browser: **http://localhost:8501**

You should see:
- 💬 Chat interface
- 👤 Applicant ID selector
- 💡 Example queries
- 🔧 Service health status

### 7.3 Test Conversation

Try these example queries:

1. **"What is the risk score for applicant 100001?"**
2. **"Show me the top risk factors"**
3. **"Compare this to the average applicant"**
4. **"What is their income and credit amount?"**

---

## 🔍 Step 8: Monitoring & Troubleshooting

### 8.1 Check All Services

```bash
# View all running services
docker compose ps

# All should show "Up" and "healthy"
```

### 8.2 View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f chatbot
docker compose logs -f model_serving
docker compose logs -f airflow
```

### 8.3 Redis Monitoring

Access RedisInsight: **http://localhost:5540**

1. Add connection:
   - Host: `redis`
   - Port: `6379`
2. Browse feature store keys

### 8.4 Common Issues

#### Database connection failed

```bash
# Restart PostgreSQL
docker compose restart postgres

# Wait for health check
docker compose ps postgres
```

#### Airflow DAG not running

```bash
# Check scheduler is running
docker compose exec airflow airflow dags list

# Restart Airflow
docker compose restart airflow
```

#### Chatbot not responding

```bash
# Check Gemini API key
docker compose exec chatbot env | grep GEMINI

# View chatbot logs
docker compose logs chatbot | grep ERROR
```

#### Model API errors

```bash
# Check model files exist
docker compose exec model_serving ls -la /app/models/

# Verify Feast connection
docker compose exec model_serving curl http://redis:6379/ping
```

---

## 📊 Step 9: Validation

### 9.1 End-to-End Test

Run this complete workflow:

1. ✅ **Ask chatbot:** "Analyze applicant 100001"
2. ✅ **Verify:** Chatbot calls `get_risk_prediction` tool
3. ✅ **Check:** Model API fetches features from Feast/Redis
4. ✅ **Confirm:** SHAP explanation is generated
5. ✅ **See:** Natural language response with insights

### 9.2 Performance Checks

```bash
# Prediction latency (should be < 500ms with Feast)
time curl http://localhost:8001/predict/applicant/100001

# Chat response time (should be < 5s with Gemini)
time curl -X POST http://localhost:8500/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Quick risk check for 100002"}'
```

---

## 🛑 Step 10: Shutdown

### 10.1 Stop All Services

```bash
# Graceful shutdown
docker compose down

# Force stop if needed
docker compose down -v  # WARNING: Deletes volumes!
```

### 10.2 Restart Clean

```bash
# Remove old data
docker compose down -v

# Start fresh
docker compose up -d postgres redis
# ... repeat deployment steps
```

---

## 🔐 Production Considerations

> [!CAUTION]
> This deployment is configured for **development/demo**. For production:

### Security

- [ ] Update all default passwords in `.env`
- [ ] Use secrets management (e.g., Docker Secrets, Vault)
- [ ] Enable HTTPS/TLS for all APIs
- [ ] Implement API authentication (JWT tokens)
- [ ] Set up firewall rules

### Scalability

- [ ] Use cloud-managed PostgreSQL (AWS RDS, Cloud SQL)
- [ ] Deploy Redis Cluster for HA
- [ ] Scale model serving with load balancer
- [ ] Use managed Airflow (Cloud Composer, MWAA)
- [ ] Implement rate limiting

### Monitoring

- [ ] Add Prometheus + Grafana for metrics
- [ ] Set up centralized logging (ELK, CloudWatch)
- [ ] Configure alerts for service failures
- [ ] Track model performance drift

### Data

- [ ] Implement data versioning
- [ ] Set up automated backups
- [ ] Add data validation checks
- [ ] Monitor feature drift

---

## 📚 Additional Resources

- **Feast Documentation:** https://docs.feast.dev/
- **LangGraph Guide:** https://langchain-ai.github.io/langgraph/
- **Gemini API:** https://ai.google.dev/docs
- **Project README:** [README.md](file:///d:/university/S10/PTDLTM/hc-risk-chatbot/README.md)

---

## 🆘 Getting Help

If you encounter issues:

1. Check logs: `docker compose logs [service]`
2. Verify health: `docker compose ps`
3. Review configuration: `cat config/.env`
4. Restart service: `docker compose restart [service]`

**Common log locations:**
- Airflow: `services/airflow/logs/`
- Application logs: `docker compose logs -f`

---

**Deployment Status Checklist:**

- [ ] PostgreSQL healthy with data loaded
- [ ] Redis running and accessible
- [ ] Airflow DAG completed successfully
- [ ] Features materialized to Redis
- [ ] Model API responding to predictions
- [ ] Chatbot API with Gemini configured
- [ ] Streamlit UI accessible
- [ ] End-to-end test passed

---

Last Updated: 2025-11-23
Version: 1.0
