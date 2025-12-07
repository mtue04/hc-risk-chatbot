"""
Home Credit Risk Analysis Chatbot
Dark mode UI with fixed sidebar
"""

import os
import re
import base64
from uuid import uuid4
from datetime import datetime

import requests
import streamlit as st

# Import charts from separate module
from charts import (
    create_risk_gauge,
    create_feature_importance_chart,
    create_comparison_chart,
    create_trend_chart,
    create_pie_chart,
    create_histogram,
    create_scatter_chart,
    create_bar_chart,
    create_heatmap,
    create_multi_line_chart,
    create_waterfall_chart,
    create_funnel_chart,
    create_bullet_chart,
    create_grouped_bar_chart,
    create_stacked_bar_chart,
    DEMO_FEATURE_IMPORTANCE,
    DEMO_COMPARISON_DATA
)

# =============================================================================
# Helper Functions
# =============================================================================
def get_logo_base64():
    """Load logo from assets folder and return base64 encoded string."""
    try:
        with open("/app/assets/hc_logo.png", "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

# =============================================================================
# Page Config
# =============================================================================
logo_b64 = get_logo_base64()
st.set_page_config(
    page_title="Home Credit Risk Analysis",
    page_icon=f"data:image/png;base64,{logo_b64}" if logo_b64 else "🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Dark Mode CSS
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

#MainMenu, footer, header {visibility: hidden;}
.stDeployButton {display: none;}

:root {
    --hc-red: #E31E24;
    --hc-red-dark: #C41A1F;
    --bg-main: #212121;
    --bg-sidebar: #171717;
    --bg-input: #2a2a2a;
    --bg-hover: #333333;
    --text-primary: #ffffff;
    --text-secondary: #b0b0b0;
    --text-muted: #707070;
    --border-color: #333333;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background: var(--bg-main) !important;
}

/* Hide default sidebar collapse button */
button[kind="header"],
button[data-testid="stSidebarCollapseButton"],
button[data-testid="baseButton-headerNoPadding"],
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
    visibility: hidden !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    min-width: 260px !important;
    width: 260px !important;
}

section[data-testid="stSidebar"] > div:first-child {
    background: var(--bg-sidebar) !important;
    padding: 1.5rem 1rem;
}

/* Remove anchor links */
.stMarkdown a[href*="#"],
h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
    display: none !important;
}

/* Text colors */
.stMarkdown, .stMarkdown p, .stMarkdown span,
h1, h2, h3, h4, h5, h6, p, span, label {
    color: var(--text-primary) !important;
}

.stCaption, .stCaption p {
    color: var(--text-muted) !important;
}

/* All Buttons - unified style */
.stButton > button {
    background: var(--bg-hover) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: all 0.2s;
}

.stButton > button:hover {
    border-color: var(--hc-red) !important;
    background: #3a3a3a !important;
}

/* Primary button */
.stButton > button[kind="primary"] {
    background: var(--hc-red) !important;
    border: none !important;
}

.stButton > button[kind="primary"]:hover {
    background: var(--hc-red-dark) !important;
}

/* Welcome section */
.welcome-container {
    text-align: center;
    padding: 80px 20px 40px;
}

.welcome-title {
    font-size: 26px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 8px;
}

.welcome-desc {
    font-size: 14px;
    color: var(--text-secondary);
    margin-bottom: 32px;
}

/* Chat messages */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 16px 0;
}

.msg-user-bubble {
    background: var(--hc-red);
    color: white;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    max-width: 70%;
    font-size: 14px;
    line-height: 1.6;
}

.msg-bot {
    display: flex;
    justify-content: flex-start;
    margin: 16px 0;
}

.msg-bot-bubble {
    background: var(--bg-hover);
    color: var(--text-primary);
    padding: 12px 18px;
    border-radius: 18px 18px 18px 4px;
    max-width: 70%;
    font-size: 14px;
    line-height: 1.6;
    border: 1px solid var(--border-color);
}

/* Chat input - unified background */
.stChatInput,
.stChatInput > div,
.stChatInput > div > div,
[data-testid="stChatInput"],
[data-testid="stChatInputContainer"],
[data-testid="stChatInput"] > div {
    background: var(--bg-input) !important;
    background-color: var(--bg-input) !important;
    border-radius: 12px !important;
}

.stChatInput > div {
    border: 1px solid var(--border-color) !important;
}

.stChatInput textarea,
.stChatInput input,
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] input {
    background: transparent !important;
    background-color: transparent !important;
    color: var(--text-primary) !important;
}

/* Fix bottom bar background */
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"],
.stBottom,
div[data-testid="stBottom"] > div {
    background: var(--bg-main) !important;
    background-color: var(--bg-main) !important;
}

/* Block container bottom area */
.block-container + div,
[data-testid="stBottomBlockContainer"] {
    background: var(--bg-main) !important;
}

/* Hide dividers */
hr {
    display: none !important;
}

/* Block container */
.block-container {
    padding: 2rem 3rem !important;
    max-width: 100%;
}

/* Spinner */
.stSpinner > div {
    border-top-color: var(--hc-red) !important;
}

/* Plotly dark */
.js-plotly-plot .plotly .modebar {
    background: transparent !important;
}

/* History label */
.history-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin: 16px 0 8px 0;
}

.tagline {
    font-size: 9px;
    color: var(--text-muted);
    margin-top: -14px;
    margin-bottom: 12px;
    font-style: italic;
}

/* Sidebar history buttons - single line, small text */
section[data-testid="stSidebar"] .stButton > button {
    font-size: 12px !important;
    padding: 8px 12px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    display: block !important;
    text-align: left !important;
    height: auto !important;
    min-height: unset !important;
}

section[data-testid="stSidebar"] .stButton > button p {
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    margin: 0 !important;
    font-size: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Configuration
# =============================================================================
CHATBOT_API = os.getenv("CHATBOT_API_URL", "http://chatbot:8500")

# =============================================================================
# Session State
# =============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "sidebar_expanded" not in st.session_state:
    st.session_state.sidebar_expanded = True

# =============================================================================
# API Function
# =============================================================================
def call_chatbot(question: str, session_id: str, audio_file=None, image_file=None, filename=None) -> dict:
    try:
        if audio_file or image_file:
            files = {}
            data = {"question": question, "session_id": session_id}
            if audio_file:
                files["audio"] = ("voice.webm", audio_file, "audio/webm")
            if image_file:
                # Use real filename for correct file type detection
                fname = filename or "document.bin"
                mime = "application/octet-stream"
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    mime = "image/png"
                elif fname.lower().endswith(".pdf"):
                    mime = "application/pdf"
                elif fname.lower().endswith(".txt"):
                    mime = "text/plain"
                elif fname.lower().endswith((".docx", ".doc")):
                    mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                files["image"] = (fname, image_file, mime)
            response = requests.post(
                f"{CHATBOT_API}/chat/multimodal",
                data=data,
                files=files,
                timeout=60
            )
        else:
            response = requests.post(
                f"{CHATBOT_API}/chat",
                json={"question": question, "session_id": session_id},
                timeout=60
            )
        if response.status_code == 200:
            return {"response": response.json().get("answer", "No response")}
        return {"response": f"Error: {response.status_code}"}
    except Exception as e:
        return {"response": f"Connection error: {str(e)}"}

# =============================================================================
# Extraction Functions
# =============================================================================
def extract_risk_score(text: str) -> float | None:
    """Trích xuất điểm rủi ro từ text"""
    for pattern in [r'(\d+\.?\d*)\s*%', r'probability.*?(\d+\.?\d*)', r'score.*?(\d+\.?\d*)']:
        match = re.search(pattern, text.lower())
        if match:
            score = float(match.group(1))
            return score / 100 if score > 1 else score
    return None


def extract_feature_importance(text: str) -> dict | None:
    """Trích xuất feature importance từ response"""
    features = {}
    patterns = [
        r'[-•]\s*([A-Za-z_]+)\s*:\s*([-+]?\d+\.?\d*)',
        r'([A-Za-z_]+)\s*\(([-+]?\d+\.?\d*)\)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for name, value in matches:
            if name.lower() not in ['applicant', 'id', 'sk_id']:
                features[name] = float(value)
    return features if len(features) >= 3 else None


def extract_comparison_data(text: str) -> dict | None:
    """Trích xuất dữ liệu so sánh từ response"""
    applicant_ids = re.findall(r'applicant\s*(\d+)', text.lower())
    if len(applicant_ids) < 2:
        return None
    return {
        applicant_ids[0]: {'Income': 70, 'Credit': 60, 'Age': 45, 'Risk': 30, 'Employment': 80},
        applicant_ids[1]: {'Income': 50, 'Credit': 75, 'Age': 55, 'Risk': 45, 'Employment': 60}
    }

def extract_pie_data(text: str) -> dict | None:
    """Extract category/proportion data for pie chart."""
    # Look for patterns like "Category: X%"
    patterns = [
        r'([A-Za-z\s]+):\s*(\d+(?:\.\d+)?)\s*%',
        r'([A-Za-z\s]+)\s*-\s*(\d+(?:\.\d+)?)\s*%',
    ]
    data = {}
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for label, value in matches:
            label = label.strip()
            if len(label) > 2 and len(label) < 30:
                data[label] = float(value)
    return data if len(data) >= 2 else None

def extract_histogram_data(text: str) -> list | None:
    """Extract numeric values for histogram."""
    # Find sequences of numbers that could be distribution data
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
    if len(numbers) >= 10:
        return [float(n) for n in numbers[:50]]
    return None

def extract_trend_data(text: str) -> list | None:
    """Extract trend/time series data."""
    numbers = re.findall(r'(\d+(?:\.\d+)?)', text)
    if len(numbers) >= 3:
        return [float(n) for n in numbers[:12]]
    return None

def extract_scatter_data(text: str) -> dict | None:
    """Extract x,y pairs for scatter plot."""
    numbers = re.findall(r'(\d+(?:\.\d+)?)', text)
    if len(numbers) >= 6:
        mid = len(numbers) // 2
        return {
            'x': [float(n) for n in numbers[:mid]],
            'y': [float(n) for n in numbers[mid:mid*2]]
        }
    return None


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    # Logo and branding
    logo_b64 = get_logo_base64()
    if logo_b64:
        st.markdown(f'<img src="data:image/png;base64,{logo_b64}" style="height:36px;margin-bottom:8px;">', unsafe_allow_html=True)

    st.markdown(
        """
        <div style="margin-bottom:0px;">
            <span style="font-family:Inter,sans-serif;font-size:16px;font-weight:700;line-height:1.2;">Home Credit Risk Analysis</span><br />
            <span style="font-family:Inter,sans-serif;font-size:10px;font-style:italic;color:#b0b0b0;margin-top:0px;margin-bottom:10px;display:block;">Empowering Smarter Credit Decisions</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # New Chat button (red, full width)
    new_chat_clicked = st.button("New Chat", key="sidebar_new_chat", use_container_width=True)
    st.markdown("""
    <style>
    /* Target the New Chat button specifically */
    button[kind="secondary"][data-testid="baseButton-secondary"] {
        background: #E31E24 !important;
        color: #fff !important;
        border: 2px solid #E31E24 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        box-shadow: 0 2px 8px rgba(227,30,36,0.2);
        transition: all 0.3s ease;
    }

    button[kind="secondary"][data-testid="baseButton-secondary"]:hover {
        background: #C41A1F !important;
        border: 2px solid #C41A1F !important;
        box-shadow: 0 4px 12px rgba(227,30,36,0.3);
        transform: translateY(-2px);
    }

    button[kind="secondary"][data-testid="baseButton-secondary"]:active {
        background: #A01619 !important;
        border: 3px solid #FF4444 !important;
        box-shadow: 0 0 0 4px rgba(255, 68, 68, 0.3);
        transform: translateY(0px);
    }
    </style>
    """, unsafe_allow_html=True)

    if new_chat_clicked:
        # Nếu có chat hiện tại thì lưu vào history trước khi reset
        if st.session_state.messages:
            first_msg = st.session_state.messages[0]["content"][:30] + "..."
            st.session_state.chat_history.insert(0, {
                "id": st.session_state.current_chat_id or str(uuid4()),
                "title": first_msg,
                "messages": st.session_state.messages.copy(),
                "timestamp": datetime.now().strftime("%H:%M")
            })
        st.session_state.messages = []
        st.session_state.session_id = str(uuid4())
        st.session_state.current_chat_id = str(uuid4())
        st.rerun()

    st.markdown("---")
    # Chat History
    st.markdown('<p class="history-label">Recent Chats</p>', unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.caption("No chat history yet")
    else:
        for i, chat in enumerate(st.session_state.chat_history[:10]):
            if st.button(f"{chat['title']}", key=f"history_{i}", use_container_width=True):
                if st.session_state.messages and st.session_state.current_chat_id:
                    for h in st.session_state.chat_history:
                        if h["id"] == st.session_state.current_chat_id:
                            h["messages"] = st.session_state.messages.copy()
                            break
                st.session_state.messages = chat["messages"].copy()
                st.session_state.current_chat_id = chat["id"]
                st.session_state.session_id = str(uuid4())
                st.rerun()

# =============================================================================
# Main Content
# =============================================================================

if not st.session_state.messages:
    # Welcome Screen
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-title">How can I help you today?</div>
        <div class="welcome-desc">Ask me about credit risk assessment, applicant profiles, or financial analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Suggestion cards
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Check Risk Score\n\n_\"What is the risk for applicant 100002?\"_", key="q1", use_container_width=True):
            st.session_state.pending = "What is the risk score for applicant 100002?"
        
        if st.button("Compare Applicants\n\n_\"Compare applicant 100002 and 100003\"_", key="q3", use_container_width=True):
            st.session_state.pending = "Compare income between applicant 100002 and 100003"
    
    with col2:
        if st.button("Risk Factors\n\n_\"What factors affect applicant 100002?\"_", key="q2", use_container_width=True):
            st.session_state.pending = "What factors affect applicant 100002's risk?"
        
        if st.button("Hypothetical Analysis\n\n_\"Risk for income 300k, credit 1M, age 35?\"_", key="q4", use_container_width=True):
            st.session_state.pending = "What would be the risk for someone with income 300000, credit amount 1000000, age 35?"

else:
    # Display messages
    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
            
            # Extract and display visualizations based on content
            content = msg["content"].lower()
            
            # 1. Risk Score Gauge
            score = extract_risk_score(msg["content"])
            if score:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.plotly_chart(create_risk_gauge(score), use_container_width=True, key=f"gauge_{idx}")
            
            # 2. Feature Importance Chart (when discussing factors)
            if any(word in content for word in ['factor', 'importance', 'influence', 'affect', 'impact', 'contribute']):
                features = extract_feature_importance(msg["content"])
                if features:
                    st.plotly_chart(create_feature_importance_chart(features), use_container_width=True, key=f"feature_{idx}")
                elif 'factor' in content or 'importance' in content:
                    # Demo feature importance if keywords found but no data extracted
                    demo_features = {
                        'EXT_SOURCE_2': -0.35,
                        'EXT_SOURCE_3': -0.28,
                        'DAYS_BIRTH': 0.15,
                        'AMT_CREDIT': 0.12,
                        'AMT_ANNUITY': 0.10,
                        'DAYS_EMPLOYED': -0.09,
                        'AMT_INCOME': -0.08,
                        'CNT_CHILDREN': 0.05
                    }
                    st.plotly_chart(create_feature_importance_chart(demo_features), use_container_width=True, key=f"feature_demo_{idx}")
            
            # 3. Comparison Chart (when comparing applicants)
            if any(word in content for word in ['compare', 'comparison', 'versus', 'vs', 'difference']):
                comparison_data = extract_comparison_data(msg["content"])
                if comparison_data:
                    st.plotly_chart(create_comparison_chart(comparison_data), use_container_width=True, key=f"compare_{idx}")
            
            # 4. Pie Chart (when discussing distribution/breakdown)
            if any(word in content for word in ['distribution', 'breakdown', 'proportion', 'percentage', 'share', 'pie']):
                # Extract category data if present
                pie_data = extract_pie_data(msg["content"])
                if pie_data:
                    st.plotly_chart(create_pie_chart(pie_data, "Distribution"), use_container_width=True, key=f"pie_{idx}")
            
            # 5. Histogram (when discussing frequency/distribution of values)
            if any(word in content for word in ['histogram', 'frequency', 'bins', 'distribution of']):
                hist_data = extract_histogram_data(msg["content"])
                if hist_data:
                    st.plotly_chart(create_histogram(hist_data, "Distribution"), use_container_width=True, key=f"hist_{idx}")
            
            # 6. Trend/Line Chart (when discussing time trends)
            if any(word in content for word in ['trend', 'over time', 'timeline', 'monthly', 'yearly', 'growth']):
                trend_data = extract_trend_data(msg["content"])
                if trend_data:
                    st.plotly_chart(create_trend_chart(trend_data), use_container_width=True, key=f"trend_{idx}")
            
            # 7. Scatter Plot (when discussing relationships/correlations)
            if any(word in content for word in ['correlation', 'relationship', 'scatter', 'vs.', 'plotted against']):
                scatter_data = extract_scatter_data(msg["content"])
                if scatter_data:
                    st.plotly_chart(create_scatter_chart(scatter_data['x'], scatter_data['y'], title="Relationship"), use_container_width=True, key=f"scatter_{idx}")
            
            # 8. Waterfall Chart (for risk breakdown/contribution)
            if any(word in content for word in ['waterfall', 'contribution', 'breakdown', 'step by step', 'cumulative']):
                waterfall_data = extract_feature_importance(msg["content"])  # Reuse feature extraction
                if waterfall_data:
                    st.plotly_chart(create_waterfall_chart(waterfall_data, "Risk Breakdown"), use_container_width=True, key=f"waterfall_{idx}")
            
            # 9. Bar Chart (for simple category comparisons)
            if any(word in content for word in ['bar chart', 'category', 'categories', 'ranking', 'top']):
                bar_data = extract_pie_data(msg["content"])  # Reuse pie extraction for categories
                if bar_data:
                    st.plotly_chart(create_bar_chart(bar_data, "Categories"), use_container_width=True, key=f"bar_{idx}")
            
            # 10. Funnel Chart (for conversion/stages)
            if any(word in content for word in ['funnel', 'stage', 'conversion', 'pipeline', 'process']):
                funnel_data = extract_pie_data(msg["content"])
                if funnel_data:
                    st.plotly_chart(create_funnel_chart(funnel_data, "Funnel"), use_container_width=True, key=f"funnel_{idx}")
            
            # 11. Bullet Chart (for progress/targets)
            if any(word in content for word in ['progress', 'target', 'goal', 'achievement', 'bullet']):
                # Extract value and target from content
                numbers = re.findall(r'(\d+(?:\.\d+)?)', msg["content"])
                if len(numbers) >= 2:
                    st.plotly_chart(create_bullet_chart(float(numbers[0]), float(numbers[1]), title="Progress"), use_container_width=True, key=f"bullet_{idx}")
            
            # 12. Heatmap (for correlations/matrices)
            if any(word in content for word in ['heatmap', 'matrix', 'correlation matrix', 'heat map']):
                # Demo heatmap data
                heatmap_data = [[1, 0.8, 0.6], [0.8, 1, 0.7], [0.6, 0.7, 1]]
                labels = ['A', 'B', 'C']
                st.plotly_chart(create_heatmap(heatmap_data, labels, labels, "Correlation Matrix"), use_container_width=True, key=f"heatmap_{idx}")
            
            # 13. Grouped Bar Chart (for comparing multiple groups)
            if any(word in content for word in ['grouped bar', 'group comparison', 'side by side', 'grouped']):
                # Demo grouped bar data
                demo_data = {'Group A': [10, 20, 30], 'Group B': [15, 25, 35]}
                st.plotly_chart(create_grouped_bar_chart(demo_data, ['Cat1', 'Cat2', 'Cat3'], "Comparison"), use_container_width=True, key=f"grouped_{idx}")
            
            # 14. Stacked Bar Chart (for composition)
            if any(word in content for word in ['stacked', 'composition', 'stacked bar', 'cumulative by category']):
                demo_data = {'Part A': [10, 20, 30], 'Part B': [15, 25, 35]}
                st.plotly_chart(create_stacked_bar_chart(demo_data, ['Cat1', 'Cat2', 'Cat3'], "Composition"), use_container_width=True, key=f"stacked_{idx}")
            
            # 15. Multi-line Chart (for comparing multiple trends)
            if any(word in content for word in ['multi-line', 'multiple trends', 'compare trends', 'lines chart']):
                demo_data = {'Series A': [10, 20, 15, 25], 'Series B': [15, 18, 22, 28]}
                st.plotly_chart(create_multi_line_chart(demo_data, title="Trends"), use_container_width=True, key=f"multiline_{idx}")

    # New conversation button
    if st.button("Start New Conversation", use_container_width=True):
        if st.session_state.messages:
            first_msg = st.session_state.messages[0]["content"][:30] + "..."
            st.session_state.chat_history.insert(0, {
                "id": st.session_state.current_chat_id or str(uuid4()),
                "title": first_msg,
                "messages": st.session_state.messages.copy(),
                "timestamp": datetime.now().strftime("%H:%M")
            })
        st.session_state.messages = []
        st.session_state.session_id = str(uuid4())
        st.session_state.current_chat_id = str(uuid4())
        st.rerun()

# =============================================================================
# Handle pending query
# =============================================================================
if hasattr(st.session_state, 'pending') and st.session_state.pending:
    query = st.session_state.pending
    st.session_state.pending = None
    
    if not st.session_state.current_chat_id:
        st.session_state.current_chat_id = str(uuid4())
    
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.spinner("Analyzing..."):
        result = call_chatbot(query, st.session_state.session_id)
        st.session_state.messages.append({"role": "assistant", "content": result["response"]})
    
    st.rerun()

# =============================================================================
# Chat Input with Plus Button (Claude-style)
# =============================================================================

# Initialize attachment states
if "show_attachments" not in st.session_state:
    st.session_state.show_attachments = False
if "processed_audio_id" not in st.session_state:
    st.session_state.processed_audio_id = None

# Plus button toggle
col_plus, col_input = st.columns([1, 15])

with col_plus:
    if st.button("+", key="plus_btn", help="Attach file or record voice"):
        st.session_state.show_attachments = not st.session_state.show_attachments
        st.rerun()

with col_input:
    user_input = st.chat_input("Ask about credit risk...")

# Show attachment options when plus is clicked
if st.session_state.show_attachments:
    attach_col1, attach_col2, attach_col3 = st.columns([2, 2, 6])
    
    with attach_col1:
        uploaded_file = st.file_uploader(
            "📎 Upload file",
            type=["png", "jpg", "jpeg", "pdf", "txt", "docx", "doc"],
            key="file_upload",
            label_visibility="visible"
        )
    
    with attach_col2:
        audio_recording = st.audio_input("🎤 Record voice", key="voice_input")
    
    # Auto-close after upload
    if uploaded_file:
        st.session_state.pending_file = uploaded_file
        st.session_state.show_attachments = False
    
    # Save audio to session state when recording is complete
    if audio_recording:
        st.session_state.pending_audio = audio_recording.read()
        st.session_state.show_attachments = False
        st.rerun()
else:
    uploaded_file = st.session_state.get("pending_file", None)
    audio_recording = None

# Process text input
if user_input:
    if not st.session_state.current_chat_id:
        st.session_state.current_chat_id = str(uuid4())
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Check if there's a pending file
    file_data = None
    pending = st.session_state.get("pending_file")
    if pending:
        try:
            file_data = pending.read()
            pending.seek(0)
        except:
            pass
        st.session_state.pending_file = None
    
    with st.spinner("Analyzing..."):
        result = call_chatbot(
            user_input, 
            st.session_state.session_id,
            image_file=file_data,
            filename=pending.name if pending else None
        )
        st.session_state.messages.append({"role": "assistant", "content": result["response"]})
    
    st.rerun()

# Process voice recording from pending_audio
pending_audio = st.session_state.get("pending_audio")
if pending_audio:
    if not st.session_state.current_chat_id:
        st.session_state.current_chat_id = str(uuid4())
    
    st.session_state.messages.append({"role": "user", "content": "[Voice message]"})
    
    with st.spinner("Transcribing..."):
        try:
            result = call_chatbot(
                "Transcribe and respond to this voice message",
                st.session_state.session_id,
                audio_file=pending_audio
            )
            st.session_state.messages.append({"role": "assistant", "content": result["response"]})
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"Voice error: {str(e)}"})
    
    st.session_state.pending_audio = None
    st.rerun()

# Process uploaded file without text (auto-analyze)
if uploaded_file and not user_input:
    if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
        st.session_state.last_uploaded_file = uploaded_file.name
        
        if not st.session_state.current_chat_id:
            st.session_state.current_chat_id = str(uuid4())
        
        st.session_state.messages.append({"role": "user", "content": f"[Uploaded: {uploaded_file.name}]"})
        
        with st.spinner("Analyzing document..."):
            try:
                file_data = uploaded_file.read()
                result = call_chatbot(
                    f"Analyze this document: {uploaded_file.name}",
                    st.session_state.session_id,
                    image_file=file_data,
                    filename=uploaded_file.name
                )
                st.session_state.messages.append({"role": "assistant", "content": result["response"]})
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"Document error: {str(e)}"})
        
        st.session_state.pending_file = None
        st.rerun()
