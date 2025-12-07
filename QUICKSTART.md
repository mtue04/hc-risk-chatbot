# Quick Start Guide

Get the HomeCredit Risk Chatbot running in 10 minutes.

## Prerequisites

- Docker Desktop installed and running
- Gemini API key from https://makersuite.google.com/app/apikey

## Setup

```bash
# 1. Navigate to project
cd d:/university/S10/PTDLTM/hc-risk-chatbot

# 2. Create config file
cp config/.env.example config/.env

# 3. Edit .env and add your Gemini API key
notepad config/.env
# Set: GEMINI_API_KEY=your_actual_key

# 4. Start all services
docker compose up -d --build

# 5. Wait for initialization (~2-3 minutes)
docker compose logs -f
```

## Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| Streamlit UI | http://localhost:8501 | Chat interface |
| Chatbot API | http://localhost:8500/docs | API documentation |
| Model API | http://localhost:8001/docs | Prediction endpoints |
| Airflow | http://localhost:8080 | Pipeline (admin/admin) |

## First Run

1. Open http://localhost:8080 (Airflow)
2. Login: `admin` / `admin`
3. Trigger DAG: `feature_engineering_pipeline`
4. Wait ~10 minutes for completion
5. Open http://localhost:8501 and start chatting

## Test Commands

```bash
# Check health
make status

# Test Model API
make test-api

# Test hypothetical prediction
make test-hypothetical
```

## Sample Queries

```
"What is the risk score for applicant 100002?"
"Show top factors affecting applicant 100001"
"What would be the risk for someone with income 300k, credit 1M, age 35?"
```

## Multimodal Setup (Optional)

For voice/image features:

1. Create Google Cloud project
2. Enable Cloud Speech-to-Text and Cloud Vision APIs
3. Create service account and download JSON key
4. Save as `config/google-credentials.json`
5. Restart: `docker compose restart chatbot`

## Troubleshooting

```bash
# View logs
make logs-chatbot
make logs-model

# Restart everything
make restart

# Clean and rebuild
make clean
docker compose up -d --build
```

## Shutdown

```bash
# Stop services
make down

# Remove all data
docker compose down -v
```
