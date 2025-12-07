# Chatbot Service - LangGraph + Gemini

Conversational AI chatbot for credit risk analysis using LangGraph state machine and Google Gemini.

**Version 0.3.0** - Enhanced with PostgreSQL checkpointing, Redis caching, retry logic, and 7 specialized tools.

---

## 🏗️ Architecture

### Enhanced (Default)
```
User Query
    ↓
FastAPI Endpoint (/chat)
    ↓
LangGraph State Machine (Enhanced)
    ├─→ Gemini LLM with retry logic
    ├─→ 7 Tools (cached in Redis):
    │   ├─→ get_risk_prediction
    │   ├─→ query_applicant_data
    │   ├─→ generate_feature_plot
    │   ├─→ compare_applicants ✨
    │   ├─→ explain_risk_factors ✨
    │   ├─→ query_bureau_history ✨
    │   └─→ get_portfolio_stats ✨
    ├─→ Extract insights node
    └─→ Summarization node (every 10 turns)
    ↓
Natural Language Response + Tool Outputs
    ↓
State persisted to PostgreSQL checkpoints
```

---

## 📁 File Structure

```
services/chatbot/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app
│   ├── graph.py             # LangGraph (all features, configurable)
│   ├── tools.py             # All 7 tools with caching & retry
│   └── cache.py             # Redis caching utilities
├── Dockerfile
└── requirements.txt
```

**Note**: All features are in `graph.py` and `tools.py`. Use environment variables to toggle:
- `USE_ENHANCED_GRAPH=true` → 7 tools + memory management (default)
- `USE_ENHANCED_GRAPH=false` → 3 tools only
- `ENABLE_CHECKPOINTING=true` → PostgreSQL persistence (default)
- `ENABLE_CHECKPOINTING=false` → In-memory storage

---

## 🔧 Components

### 1. FastAPI Application (`main.py`)

**Endpoints:**

- `GET /health` - Health check with Gemini status
- `POST /chat` - Main chat endpoint
  ```json
  {
    "question": "What is the risk for applicant 100001?",
    "session_id": "optional-uuid",
    "applicant_id": 100001
  }
  ```
- `DELETE /chat/{session_id}` - Clear conversation
- `GET /chat/{session_id}/history` - Get conversation history

**Features:**
- ✅ Session management with in-memory storage
- ✅ Conversation history tracking
- ✅ Automatic session ID generation
- ✅ Tool output extraction

### 2. LangGraph State Machine (`graph.py`)

**State Schema:**
```python
{
    "messages": List[BaseMessage],      # Conversation history
    "session_id": str,                   # Session identifier
    "applicant_id": int | None,          # Current applicant
    "risk_score": float | None,          # Latest risk prediction
    "last_tool_output": dict | None      # Latest tool result
}
```

**Workflow:**
1. User message → System prompt injection
2. Gemini generates response with potential tool calls
3. If tool calls needed → Execute tools
4. Tool results → Back to Gemini for synthesis
5. Final answer → Return to user

**System Prompt:**
- Expert credit risk analyst persona
- Business-friendly explanations
- Proactive tool usage
- Contextual responses

### 3. Tools (`tools.py`)

#### Tool 1: `get_risk_prediction`

**Purpose:** Get default probability and SHAP explanations

**Input:**
```python
applicant_id: int  # SK_ID_CURR
```

**Output:**
```json
{
  "applicant_id": 100001,
  "probability": 0.23,
  "prediction": "Low Risk",
  "top_factors": [
    {"feature": "EXT_SOURCE_2", "impact": -0.15},
    {"feature": "DAYS_BIRTH", "impact": 0.08},
    ...
  ]
}
```

#### Tool 2: `query_applicant_data`

**Purpose:** Fetch applicant demographics and financials

**Input:**
```python
applicant_id: int
fields: List[str] = None  # Default: income, credit, gender, age, etc.
```

**Output:**
```json
{
  "applicant_id": 100001,
  "AMT_INCOME_TOTAL": 202500.0,
  "AMT_CREDIT": 406597.5,
  "CODE_GENDER": 1.0,
  "age_years": 43,
  "employment_years": 3
}
```

#### Tool 3: `generate_feature_plot`

**Purpose:** Statistical comparison to population

**Input:**
```python
applicant_id: int
feature_names: List[str]  # Max 5 features
```

**Output:**
```json
{
  "applicant_id": 100001,
  "features": {
    "AMT_INCOME_TOTAL": {
      "applicant_value": 202500.0,
      "population_mean": 168797.9,
      "population_median": 147150.0,
      "percentile": 68.5
    }
  }
}
```

---

## 🚀 Usage

### Local Development

```bash
cd services/chatbot

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GEMINI_API_KEY=your_key
export POSTGRES_HOST=localhost
export POSTGRES_USER=hc_admin
export POSTGRES_PASSWORD=hc_password
export POSTGRES_DB=homecredit_db

# Run server
uvicorn app.main:app --reload --port 8500
```

### Docker

```bash
# Build image
docker build -t hc-chatbot services/chatbot/

# Run container
docker run -p 8500:8500 \
  -e GEMINI_API_KEY=your_key \
  --network hc-network \
  hc-chatbot
```

### Testing

```bash
# Health check
curl http://localhost:8500/health

# Chat request
curl -X POST http://localhost:8500/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Analyze applicant 100001",
    "applicant_id": 100001
  }'

# Get conversation history
curl http://localhost:8500/chat/{session_id}/history

# Clear conversation
curl -X DELETE http://localhost:8500/chat/{session_id}
```

---

## 🔑 Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | Yes | `changeme` | Google Gemini API key |
| `MODEL_API_URL` | No | `http://model_serving:8000` | Model service endpoint |
| `POSTGRES_HOST` | No | `postgres` | PostgreSQL host |
| `POSTGRES_PORT` | No | `5432` | PostgreSQL port |
| `POSTGRES_DB` | No | `homecredit_db` | Database name |
| `POSTGRES_USER` | No | `hc_admin` | DB username |
| `POSTGRES_PASSWORD` | No | `hc_password` | DB password |

### Fallback Mode

If `GEMINI_API_KEY` is not set or equals `changeme`:
- Chatbot returns static responses
- No tool calling or natural language understanding
- Health check shows `gemini_configured: false`

---

## 📊 Example Conversations

### Example 1: Risk Analysis

**User:** "What is the risk score for applicant 100001?"

**Chatbot:** 
1. Calls `get_risk_prediction(100001)`
2. Receives: `{"probability": 0.23, "prediction": "Low Risk"}`
3. Responds: "Applicant 100001 has a default probability of 23%, which is classified as Low Risk. This is below the average..."

### Example 2: Data Exploration

**User:** "Show me the income and employment details for applicant 100002"

**Chatbot:**
1. Calls `query_applicant_data(100002, ["AMT_INCOME_TOTAL", "DAYS_EMPLOYED"])`
2. Receives demographic data
3. Responds with formatted information

### Example 3: Comparison

**User:** "How does applicant 100003's income compare to average?"

**Chatbot:**
1. Calls `query_applicant_data(100003)`
2. Calls `generate_feature_plot(100003, ["AMT_INCOME_TOTAL"])`
3. Synthesizes comparison with percentiles

---

## 🔍 Monitoring

### Logs

```bash
# View chatbot logs
docker compose logs -f chatbot

# Key log events:
# - chat_request: New user question
# - llm_response: Gemini completion
# - risk_prediction_retrieved: Tool call success
# - chat_response_generated: Final response
```

### Metrics

Important fields logged:
- `session_id` - Conversation tracking
- `has_applicant_id` - Context availability
- `has_tool_calls` - Tool usage
- `num_messages` - Conversation length

---

## 🐛 Debugging

### Common Issues

**1. "Gemini API error"**
```bash
# Check API key
docker compose exec chatbot env | grep GEMINI_API_KEY

# Verify key is valid at https://makersuite.google.com
```

**2. "Database connection failed"**
```bash
# Test PostgreSQL connectivity
docker compose exec chatbot psql -h postgres -U hc_admin -d homecredit_db -c "SELECT 1"
```

**3. "Model API unavailable"**
```bash
# Check model service
docker compose ps model_serving

# Test endpoint
docker compose exec chatbot curl http://model_serving:8000/health
```

**4. "No response generated"**
```bash
# Check LangGraph errors
docker compose logs chatbot | grep ERROR

# Verify message history
curl http://localhost:8500/chat/{session_id}/history
```

---

## 🎨 Customization

### Adding New Tools

1. Define tool in `tools.py`:
```python
@tool
def my_new_tool(param: str) -> dict:
    """Tool description for Gemini."""
    # Implementation
    return {"result": "data"}
```

2. Register in `graph.py`:
```python
tools = [
    get_risk_prediction,
    query_applicant_data,
    generate_feature_plot,
    my_new_tool,  # Add here
]
```

3. Update system prompt to mention new capability

### Customizing System Prompt

Edit `SYSTEM_PROMPT` in `graph.py` to:
- Change persona/tone
- Add domain knowledge
- Modify response guidelines
- Include examples

### Conversation Storage

Current: In-memory dictionary (lost on restart)

**For production:**
```python
# Replace _conversations dict with Redis
import redis
r = redis.Redis(host='redis', port=6379)

def save_conversation(session_id, messages):
    r.set(f"conv:{session_id}", json.dumps(messages))

def load_conversation(session_id):
    data = r.get(f"conv:{session_id}")
    return json.loads(data) if data else []
```

---

## 📚 References

- **LangGraph Docs:** https://langchain-ai.github.io/langgraph/
- **Gemini API:** https://ai.google.dev/docs
- **LangChain Tools:** https://python.langchain.com/docs/modules/agents/tools/

---

**Version:** 0.3.0
**Last Updated:** 2025-12-07

---

## ✨ New in v0.3.0

- 🎯 **7 Tools** (vs 3): Added compare, explain, bureau history, portfolio stats
- 💾 **PostgreSQL Checkpointing**: Conversations persist across restarts
- ⚡ **Redis Caching**: 100x faster repeated queries
- 🔄 **Retry Logic**: Exponential backoff for LLM and tools
- 🧠 **Memory Management**: Auto-summarization every 10 turns
- 📊 **Enhanced State**: Tracks summaries, insights, mentioned applicants

See `../../IMPROVEMENTS.md` for full details.
