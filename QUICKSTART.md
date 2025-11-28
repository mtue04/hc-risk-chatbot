# 🚀 Quick Start Guide

Get the HomeCredit Risk Chatbot running in 10 minutes!

---

## ⚡ Fast Track

```bash
# 1. Navigate to project
cd d:/university/S10/PTDLTM/hc-risk-chatbot

# 2. Configure Gemini API (REQUIRED)
# Edit config/.env and add your Gemini API key:
# GEMINI_API_KEY=your_key_here

# 3. Start all services
docker compose up -d

# 4. Wait for initialization (~2 minutes)
# Watch the logs:
docker compose logs -f

# 5. Access the UI
# Open browser: http://localhost:8501
```

---

## 🎯 What You Get

After deployment, you'll have:

| Service | URL | Purpose |
|---------|-----|---------|
| **Streamlit UI** | http://localhost:8501 | Chat interface |
| **Chatbot API** | http://localhost:8500 | LangGraph backend |
| **Model API** | http://localhost:8001 | Risk predictions |
| **Airflow** | http://localhost:8080 | Pipeline orchestration |
| **RedisInsight** | http://localhost:5540 | Feature store monitoring |

**Default Credentials:**
- Airflow: `admin` / `admin`

---

## 📝 First Steps

### 1. Check Service Health

```bash
# All services should be "Up (healthy)"
docker compose ps
```

### 2. Trigger Feature Pipeline

**Option A - Via UI:**
1. Open http://localhost:8080
2. Login: `admin` / `admin`
3. Find `feature_engineering_pipeline` DAG
4. Click "Trigger DAG" button

**Option B - Via CLI:**
```bash
docker compose exec airflow airflow dags trigger feature_engineering_pipeline
```

Wait ~10 minutes for completion.

### 3. Try the Chatbot

Open http://localhost:8501 and try:

```
"What is the risk score for applicant 100001?"
```

```
"Show me the top risk factors"
```

```
"Compare income and credit for applicant 100002"
```

---

## 🔍 Verify Everything Works

### Test 1: Database

```bash
docker compose exec postgres psql -U hc_admin -d homecredit_db \
  -c "SELECT COUNT(*) FROM home_credit.application_train;"
```

✅ **Expected:** ~307,000 rows

### Test 2: Model API

```bash
curl http://localhost:8001/health
```

✅ **Expected:** `{"status": "ok"}`

### Test 3: Chatbot API

```bash
curl http://localhost:8500/health
```

✅ **Expected:** `{"langgraph_enabled": true, "gemini_configured": true}`

### Test 4: Prediction

```bash
curl http://localhost:8001/predict/applicant/100001
```

✅ **Expected:** JSON with `probability` field

### Test 5: Chat

```bash
curl -X POST http://localhost:8500/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Risk for 100001?"}'
```

✅ **Expected:** Natural language response

---

## ❓ Troubleshooting

### Gemini Not Working?

Check your API key:
```bash
cat config/.env | grep GEMINI_API_KEY
```

If it says `changeme`, update it with a real key from https://makersuite.google.com/app/apikey

Then restart:
```bash
docker compose restart chatbot
```

### Services Not Starting?

```bash
# Check logs for errors
docker compose logs [service_name]

# Common fixes:
docker compose down
docker compose up -d
```

### Database Empty?

```bash
# Reload data
docker compose down -v
docker compose up -d postgres
# Wait 2 minutes for init scripts
```

---

## 🎓 Learn More

- **Full Deployment Guide:** [docs/DEPLOYMENT.md](file:///d:/university/S10/PTDLTM/hc-risk-chatbot/docs/DEPLOYMENT.md)
- **Project Architecture:** [README.md](file:///d:/university/S10/PTDLTM/hc-risk-chatbot/README.md)
- **Data Pipeline:** [docs/Datapipeline.md](file:///d:/university/S10/PTDLTM/hc-risk-chatbot/docs/Datapipeline.md)
- **Feature Store:** [services/feast/README.md](file:///d:/university/S10/PTDLTM/hc-risk-chatbot/services/feast/README.md)

---

## 🛑 Shutdown

```bash
# Stop all services
docker compose down

# Remove all data (fresh start)
docker compose down -v
```

---

**Need Help?** Check `docker compose logs -f` for error messages.

Enjoy! 🎉
