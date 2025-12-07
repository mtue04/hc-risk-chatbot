# Multi-Step Iterative Analysis Workflow

This document describes the new multi-step analysis workflow that enables systematic, iterative data analysis with human-in-the-loop review.

## Overview

The analysis workflow implements a sophisticated pattern for data exploration:

1. **Planner (Orchestrator)**: Reads PostgreSQL schema and user request, outputs a structured list of analysis steps
2. **Human Review Node**: Pauses for user approval or editing of the plan
3. **Executor Loop**: Iteratively processes each step:
   - **SQL Generator**: Creates SQL query for the analysis
   - **Data Analyzer**: Executes query, cleans data, generates visualization
   - **Vision Analyzer**: Extracts insights from the chart and data
4. **Synthesizer**: Combines all insights into a final executive summary

## Architecture

### State Management

The workflow uses `AnalysisState` to track:

```python
{
    "messages": [],              # Conversation history
    "user_request": str,         # Original analysis request
    "schema_info": dict,         # PostgreSQL schema (tables, columns)
    "plan": AnalysisPlan,        # Structured analysis plan
    "current_step": int,         # Current step being executed (1-indexed)
    "step_results": list,        # Results from completed steps
    "final_summary": str,        # Executive summary
    "workflow_status": str,      # planning, awaiting_approval, executing, synthesizing, completed
}
```

### Analysis Plan Structure

Each plan contains:

```python
{
    "steps": [
        {
            "step_number": 1,
            "description": "Analyze monthly revenue trends",
            "sql_needed": True,
            "chart_type": "line"  # line, bar, scatter, histogram, boxplot
        },
        # ... more steps
    ],
    "user_request": "Original request text",
    "approved": False,
    "user_edits": None
}
```

### Step Results

Each executed step produces:

```python
{
    "step_number": 1,
    "sql_query": "SELECT ...",
    "data_summary": {
        "row_count": 1000,
        "column_count": 3,
        "columns": ["date", "revenue", "count"],
        "sample_data": [...]
    },
    "chart_image_path": "/tmp/analysis_charts/step_1_line.png",
    "insights": "The monthly revenue shows an upward trend..."
}
```

## API Endpoints

### 1. Start Analysis

**POST** `/analysis/start`

Initiates a new analysis workflow.

**Request:**
```json
{
    "user_request": "Analyze default risk patterns by income level and age",
    "thread_id": "optional-custom-id"
}
```

**Response:**
```json
{
    "thread_id": "uuid-generated-id",
    "status": "awaiting_approval",
    "plan": {
        "steps": [
            {
                "step_number": 1,
                "description": "Analyze default rates by income quartile",
                "sql_needed": true,
                "chart_type": "bar"
            },
            {
                "step_number": 2,
                "description": "Examine age distribution for defaulters vs non-defaulters",
                "sql_needed": true,
                "chart_type": "histogram"
            },
            {
                "step_number": 3,
                "description": "Cross-tabulate income and age groups",
                "sql_needed": true,
                "chart_type": "scatter"
            }
        ],
        "user_request": "Analyze default risk patterns by income level and age",
        "approved": false,
        "user_edits": null
    },
    "message": "Analysis plan generated. Please review and approve to proceed."
}
```

### 2. Approve or Edit Plan

**POST** `/analysis/{thread_id}/approve`

Approve the plan to continue execution, or request modifications.

**Request (Approve):**
```json
{
    "approved": true
}
```

**Request (Request Edits):**
```json
{
    "approved": false,
    "edits": "Skip step 2, focus only on income quartiles and add a step for external credit scores"
}
```

**Response:**
```json
{
    "thread_id": "uuid",
    "status": "executing",
    "plan": {...},
    "current_step": 1,
    "message": "Analysis workflow resumed."
}
```

### 3. Check Status

**GET** `/analysis/{thread_id}/status`

Get the current status of an analysis workflow.

**Response (In Progress):**
```json
{
    "thread_id": "uuid",
    "status": "executing",
    "plan": {...},
    "current_step": 2,
    "step_results": [
        {
            "step_number": 1,
            "sql_query": "SELECT ...",
            "data_summary": {...},
            "chart_image_path": "/tmp/analysis_charts/step_1_bar.png",
            "insights": "Income quartile analysis shows higher default rates..."
        }
    ]
}
```

**Response (Completed):**
```json
{
    "thread_id": "uuid",
    "status": "completed",
    "plan": {...},
    "step_results": [...],
    "final_summary": "EXECUTIVE SUMMARY\n\nKey Findings:\n- Default rates increase significantly in lower income quartiles...\n- Age shows a non-linear relationship with default risk...\n\nRecommendations:\n- Implement stricter credit checks for low-income applicants...\n- Consider age-based risk adjustments..."
}
```

## Workflow Diagram

```
┌─────────────────┐
│   START         │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Schema Reader   │ ← Reads PostgreSQL schema
└────────┬────────┘
         │
         v
┌─────────────────┐
│    Planner      │ ← Generates analysis plan using LLM
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Human Review   │ ◄── INTERRUPT (pause for user approval)
└────────┬────────┘
         │
         v
    ┌────────────┐
    │ Approved?  │
    └─┬────────┬─┘
      │ No     │ Yes
      │(edits) │
      v        v
   Replan   Execute
             Steps
               │
        ┌──────v──────┐
        │ For each    │
        │ step in     │
        │ plan:       │
        └──────┬──────┘
               │
        ┌──────v──────┐
        │ SQL         │
        │ Generator   │
        └──────┬──────┘
               │
        ┌──────v──────┐
        │ Data        │
        │ Analyzer    │ ← Execute SQL, create chart
        └──────┬──────┘
               │
        ┌──────v──────┐
        │ Vision      │
        │ Analyzer    │ ← Extract insights
        └──────┬──────┘
               │
        ┌──────v──────┐
        │ Increment   │
        │ Step        │
        └──────┬──────┘
               │
         ┌─────v─────┐
         │ More      │
         │ steps?    │
         └─┬───────┬─┘
           │ Yes   │ No
           │       │
           └───┐   v
               │ ┌─────────────┐
               └─│ Synthesizer │ ← Create final summary
                 └──────┬──────┘
                        │
                        v
                   ┌────────┐
                   │  END   │
                   └────────┘
```

## Example Usage

### Python Client Example

```python
import httpx
import time

# Start analysis
response = httpx.post("http://localhost:8001/analysis/start", json={
    "user_request": "Analyze credit risk patterns by age and income"
})
data = response.json()
thread_id = data["thread_id"]
plan = data["plan"]

print(f"Thread ID: {thread_id}")
print(f"Generated plan with {len(plan['steps'])} steps:")
for step in plan["steps"]:
    print(f"  {step['step_number']}. {step['description']}")

# Review and approve the plan
approval = input("Approve plan? (y/n): ")
if approval.lower() == 'y':
    response = httpx.post(
        f"http://localhost:8001/analysis/{thread_id}/approve",
        json={"approved": True}
    )
else:
    edits = input("What changes would you like? ")
    response = httpx.post(
        f"http://localhost:8001/analysis/{thread_id}/approve",
        json={"approved": False, "edits": edits}
    )

# Poll for completion
while True:
    response = httpx.get(f"http://localhost:8001/analysis/{thread_id}/status")
    status_data = response.json()

    if status_data["status"] == "completed":
        print("\nAnalysis complete!")
        print("\nFINAL SUMMARY:")
        print(status_data["final_summary"])

        print("\nStep Results:")
        for result in status_data["step_results"]:
            print(f"\nStep {result['step_number']}:")
            print(f"  Insights: {result['insights']}")
            print(f"  Chart: {result['chart_image_path']}")
        break

    print(f"Status: {status_data['status']} - Step {status_data.get('current_step', 0)}/{len(plan['steps'])}")
    time.sleep(2)
```

### cURL Example

```bash
# 1. Start analysis
curl -X POST http://localhost:8001/analysis/start \
  -H "Content-Type: application/json" \
  -d '{
    "user_request": "Analyze default patterns by external credit scores"
  }'

# Save the thread_id from the response

# 2. Approve the plan
curl -X POST http://localhost:8001/analysis/{thread_id}/approve \
  -H "Content-Type: application/json" \
  -d '{
    "approved": true
  }'

# 3. Check status
curl http://localhost:8001/analysis/{thread_id}/status
```

## Chart Types

The workflow supports multiple visualization types:

- **line**: Time-series or sequential data trends
- **bar**: Categorical comparisons
- **scatter**: Relationship between two continuous variables
- **histogram**: Distribution of a single variable
- **boxplot**: Statistical summary with quartiles and outliers

Charts are automatically generated based on the plan's `chart_type` specification and saved to `/tmp/analysis_charts/`.

## Configuration

### Environment Variables

```bash
# Database connection
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=homecredit_db
POSTGRES_USER=hc_admin
POSTGRES_PASSWORD=hc_password

# LLM configuration
GEMINI_API_KEY=your-api-key
GEMINI_MODEL=gemini-2.0-flash

# Chart output directory
CHART_OUTPUT_DIR=/tmp/analysis_charts
```

### Dependencies

The workflow requires:

```
pandas>=2.0.0
matplotlib>=3.7.0
numpy>=1.24.0
langgraph>=0.1.0
psycopg2-binary>=2.9.11
```

## Human-in-the-Loop Review

The workflow uses LangGraph's checkpoint system with interrupts to enable human review:

1. The workflow automatically **pauses before the `human_review` node**
2. The plan is returned to the user via the API
3. The user can:
   - **Approve**: Set `plan.approved = True` and resume
   - **Request edits**: Set `plan.user_edits` with modifications and the workflow will regenerate the plan
4. State is persisted using `MemorySaver` checkpointing

This ensures users have full control over the analysis direction before execution begins.

## Error Handling

The workflow includes comprehensive error handling:

- **Schema reading failures**: Returns error in `schema_info`
- **SQL generation errors**: Falls back to basic queries
- **Query execution failures**: Captured in `data_summary.error`
- **Chart generation issues**: Set `chart_image_path = None`
- **LLM failures**: Graceful degradation with fallback responses

All errors are logged with `structlog` for debugging.

## Extending the Workflow

### Adding New Node Types

To add new analysis capabilities:

1. Define the node function in `analysis_nodes.py`:
```python
def custom_analyzer_node(state: AnalysisState) -> dict:
    # Your logic here
    return {"custom_field": value}
```

2. Add the node to the graph in `analysis_graph.py`:
```python
workflow.add_node("custom_analyzer", custom_analyzer_node)
workflow.add_edge("previous_node", "custom_analyzer")
```

### Custom Chart Types

Add new visualization types in `data_analyzer_node`:

```python
elif chart_type == "heatmap":
    # Create correlation heatmap
    corr = df.corr()
    plt.imshow(corr, cmap='coolwarm')
    plt.colorbar()
```

### Enhanced Vision Analysis

For actual image-based vision analysis (requires Gemini Pro Vision):

```python
# In vision_analyzer_node
from google.generativeai import GenerativeModel

model = GenerativeModel('gemini-pro-vision')
with open(chart_path, 'rb') as f:
    image_data = f.read()

response = model.generate_content([
    "Analyze this chart and provide insights:",
    {"mime_type": "image/png", "data": image_data}
])

insights = response.text
```

## Production Considerations

For production deployments:

1. **Persistent Checkpointing**: Replace `MemorySaver` with a database-backed checkpointer
2. **Chart Storage**: Use cloud storage (S3, GCS) instead of `/tmp/`
3. **Rate Limiting**: Add request throttling for LLM calls
4. **Caching**: Cache schema information to reduce database queries
5. **Async Execution**: Use background tasks for long-running analyses
6. **Monitoring**: Add metrics for workflow execution times and success rates

## Troubleshooting

### Common Issues

**Issue**: Workflow stuck in "awaiting_approval"
- **Solution**: Call the approve endpoint to continue execution

**Issue**: Charts not generating
- **Solution**: Ensure matplotlib backend is set to 'Agg' for headless environments

**Issue**: LLM not generating plan
- **Solution**: Check `GEMINI_API_KEY` is set and valid

**Issue**: SQL queries failing
- **Solution**: Review column name quoting (PostgreSQL requires double quotes for uppercase columns)

## Performance

Typical execution times:

- Schema reading: < 1s
- Plan generation: 2-5s (LLM call)
- Per-step execution: 3-8s (SQL + chart + vision analysis)
- Final synthesis: 2-4s (LLM call)

For a 3-step analysis: **~20-35 seconds total**

## Security

Best practices:

- **SQL Injection Prevention**: Use parameterized queries throughout
- **Resource Limits**: Limit SQL result set size with `LIMIT` clauses
- **Access Control**: Implement authentication on API endpoints
- **Input Validation**: Sanitize user requests and edits
- **Chart Storage**: Isolate chart directories per user/session
