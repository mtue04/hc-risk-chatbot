"""
Home Credit Risk Analysis - Chart Components
Tổng hợp tất cả các biểu đồ visualization
"""

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from typing import Optional


# =============================================================================
# 1. RISK GAUGE - Biểu đồ đồng hồ đo rủi ro
# =============================================================================
def create_risk_gauge(score: float, chart_key: str = "risk_gauge") -> go.Figure:
    """
    Biểu đồ gauge hiển thị điểm rủi ro
    
    Args:
        score: Điểm rủi ro từ 0-1 (hoặc 0-100)
        chart_key: Unique key cho chart
    
    Returns:
        Plotly Figure object
    """
    # Normalize score to 0-1
    if score > 1:
        score = score / 100
    
    color = "#10B981" if score < 0.3 else "#F59E0B" if score < 0.6 else "#E31E24"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        number={'suffix': '%', 'font': {'size': 40, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#505050', 'tickfont': {'color': '#707070'}},
            'bar': {'color': color, 'thickness': 0.7},
            'bgcolor': "#333333",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30], 'color': 'rgba(16,185,129,0.2)'},
                {'range': [30, 60], 'color': 'rgba(245,158,11,0.2)'},
                {'range': [60, 100], 'color': 'rgba(227,30,36,0.2)'}
            ],
        }
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff'}
    )
    return fig


# =============================================================================
# 2. FEATURE IMPORTANCE - Biểu đồ bar các yếu tố ảnh hưởng
# =============================================================================
def create_feature_importance_chart(features: dict, chart_key: str = "feature_importance") -> go.Figure:
    """
    Biểu đồ bar ngang hiển thị feature importance
    
    Args:
        features: Dict {feature_name: importance_value}
        chart_key: Unique key cho chart
    
    Returns:
        Plotly Figure object
    """
    # Sort by absolute value, take top 10
    sorted_features = dict(sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:10])
    
    names = list(sorted_features.keys())
    values = list(sorted_features.values())
    colors = ['#E31E24' if v > 0 else '#10B981' for v in values]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation='h',
        marker_color=colors,
        text=[f'{v:+.2f}' for v in values],
        textposition='outside',
        textfont={'color': '#ffffff', 'size': 11}
    ))
    
    fig.update_layout(
        title={'text': '📊 Top Risk Factors', 'font': {'color': '#ffffff', 'size': 14}},
        height=300,
        margin=dict(l=10, r=50, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff'},
        xaxis={'gridcolor': '#333333', 'zerolinecolor': '#555555'},
        yaxis={'gridcolor': '#333333'},
        showlegend=False
    )
    return fig


# =============================================================================
# 3. COMPARISON RADAR - Biểu đồ radar so sánh applicants
# =============================================================================
def create_comparison_chart(data: dict, chart_key: str = "comparison") -> go.Figure:
    """
    Biểu đồ radar so sánh các applicant
    
    Args:
        data: Dict {applicant_id: {metric: value}}
        chart_key: Unique key cho chart
    
    Returns:
        Plotly Figure object
    """
    categories = list(data[list(data.keys())[0]].keys())
    
    fig = go.Figure()
    
    colors = ['#E31E24', '#10B981', '#F59E0B', '#3B82F6']
    for i, (applicant, values) in enumerate(data.items()):
        color = colors[i % len(colors)]
        # Convert hex to rgba for fill
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        
        fig.add_trace(go.Scatterpolar(
            r=list(values.values()),
            theta=categories,
            fill='toself',
            name=f'Applicant {applicant}',
            line_color=color,
            fillcolor=f'rgba({r},{g},{b},0.2)'
        ))
    
    fig.update_layout(
        title={'text': '📈 Applicant Comparison', 'font': {'color': '#ffffff', 'size': 14}},
        height=350,
        margin=dict(l=60, r=60, t=50, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff'},
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(gridcolor='#333333', tickfont={'color': '#707070'}),
            angularaxis=dict(gridcolor='#333333', tickfont={'color': '#b0b0b0'})
        ),
        legend=dict(font={'color': '#ffffff'})
    )
    return fig


# =============================================================================
# 4. TREND LINE - Biểu đồ xu hướng
# =============================================================================
def create_trend_chart(data: list, labels: list = None, chart_key: str = "trend") -> go.Figure:
    """
    Biểu đồ line cho trend analysis
    
    Args:
        data: List of values
        labels: List of x-axis labels
        chart_key: Unique key cho chart
    
    Returns:
        Plotly Figure object
    """
    x_values = labels if labels else list(range(len(data)))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_values,
        y=data,
        mode='lines+markers',
        line=dict(color='#E31E24', width=2),
        marker=dict(size=8, color='#E31E24'),
        fill='tozeroy',
        fillcolor='rgba(227,30,36,0.1)'
    ))
    
    fig.update_layout(
        title={'text': '📈 Risk Trend', 'font': {'color': '#ffffff', 'size': 14}},
        height=200,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff'},
        xaxis={'gridcolor': '#333333', 'showgrid': False},
        yaxis={'gridcolor': '#333333'},
        showlegend=False
    )
    return fig


# =============================================================================
# 5. BAR CHART - Biểu đồ cột đơn giản
# =============================================================================
def create_bar_chart(data: dict, title: str = "Chart", chart_key: str = "bar") -> go.Figure:
    """
    Biểu đồ bar cơ bản
    
    Args:
        data: Dict {label: value}
        title: Tiêu đề chart
        chart_key: Unique key cho chart
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure(go.Bar(
        x=list(data.keys()),
        y=list(data.values()),
        marker_color='#E31E24',
        text=[f'{v:,.0f}' for v in data.values()],
        textposition='outside',
        textfont={'color': '#ffffff'}
    ))
    
    fig.update_layout(
        title={'text': title, 'font': {'color': '#ffffff', 'size': 14}},
        height=250,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff'},
        xaxis={'gridcolor': '#333333'},
        yaxis={'gridcolor': '#333333'},
        showlegend=False
    )
    return fig


# =============================================================================
# 6. PIE CHART - Biểu đồ tròn
# =============================================================================
def create_pie_chart(data: dict, title: str = "Distribution", chart_key: str = "pie") -> go.Figure:
    """
    Biểu đồ pie/donut
    
    Args:
        data: Dict {label: value}
        title: Tiêu đề chart
        chart_key: Unique key cho chart
    
    Returns:
        Plotly Figure object
    """
    colors = ['#E31E24', '#10B981', '#F59E0B', '#3B82F6', '#8B5CF6', '#EC4899']
    
    fig = go.Figure(go.Pie(
        labels=list(data.keys()),
        values=list(data.values()),
        hole=0.4,
        marker_colors=colors[:len(data)],
        textfont={'color': '#ffffff'},
        textinfo='percent+label'
    ))
    
    fig.update_layout(
        title={'text': title, 'font': {'color': '#ffffff', 'size': 14}},
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff'},
        legend=dict(font={'color': '#ffffff'})
    )
    return fig


# =============================================================================
# 7. HISTOGRAM - Biểu đồ phân phối
# =============================================================================
def create_histogram(data: list, title: str = "Distribution", bins: int = 20, chart_key: str = "histogram") -> go.Figure:
    """
    Biểu đồ histogram
    
    Args:
        data: List of values
        title: Tiêu đề chart
        bins: Số lượng bins
        chart_key: Unique key cho chart
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure(go.Histogram(
        x=data,
        nbinsx=bins,
        marker_color='#E31E24',
        opacity=0.8
    ))
    
    fig.update_layout(
        title={'text': title, 'font': {'color': '#ffffff', 'size': 14}},
        height=250,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff'},
        xaxis={'gridcolor': '#333333'},
        yaxis={'gridcolor': '#333333'},
        showlegend=False
    )
    return fig


# =============================================================================
# 8. SCATTER PLOT - Biểu đồ phân tán
# =============================================================================
def create_scatter_chart(x: list, y: list, labels: list = None, title: str = "Scatter", chart_key: str = "scatter") -> go.Figure:
    """
    Biểu đồ scatter
    
    Args:
        x: List of x values
        y: List of y values
        labels: Optional labels for points
        title: Tiêu đề chart
        chart_key: Unique key cho chart
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(size=10, color='#E31E24', opacity=0.7),
        text=labels,
        hovertemplate='%{text}<br>X: %{x}<br>Y: %{y}<extra></extra>' if labels else None
    ))
    
    fig.update_layout(
        title={'text': title, 'font': {'color': '#ffffff', 'size': 14}},
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff'},
        xaxis={'gridcolor': '#333333'},
        yaxis={'gridcolor': '#333333'},
        showlegend=False
    )
    return fig


# =============================================================================
# 9. HEATMAP - Biểu đồ nhiệt
# =============================================================================
def create_heatmap(data: list, x_labels: list, y_labels: list, title: str = "Heatmap", chart_key: str = "heatmap") -> go.Figure:
    """
    Biểu đồ heatmap
    
    Args:
        data: 2D list of values
        x_labels: Labels for x-axis
        y_labels: Labels for y-axis
        title: Tiêu đề chart
        chart_key: Unique key cho chart
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure(go.Heatmap(
        z=data,
        x=x_labels,
        y=y_labels,
        colorscale=[[0, '#10B981'], [0.5, '#F59E0B'], [1, '#E31E24']],
        texttemplate='%{z:.2f}',
        textfont={'color': '#ffffff'}
    ))
    
    fig.update_layout(
        title={'text': title, 'font': {'color': '#ffffff', 'size': 14}},
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff'},
        xaxis={'gridcolor': '#333333'},
        yaxis={'gridcolor': '#333333'}
    )
    return fig


# =============================================================================
# 10. MULTI-LINE CHART - Biểu đồ nhiều đường
# =============================================================================
def create_multi_line_chart(data: dict, x_values: list = None, title: str = "Trend", chart_key: str = "multiline") -> go.Figure:
    """
    Biểu đồ nhiều đường
    
    Args:
        data: Dict {series_name: [values]}
        x_values: Optional x-axis values
        title: Tiêu đề chart
        chart_key: Unique key cho chart
    
    Returns:
        Plotly Figure object
    """
    colors = ['#E31E24', '#10B981', '#F59E0B', '#3B82F6', '#8B5CF6']
    
    fig = go.Figure()
    
    for i, (name, values) in enumerate(data.items()):
        x = x_values if x_values else list(range(len(values)))
        fig.add_trace(go.Scatter(
            x=x,
            y=values,
            mode='lines+markers',
            name=name,
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title={'text': title, 'font': {'color': '#ffffff', 'size': 14}},
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff'},
        xaxis={'gridcolor': '#333333'},
        yaxis={'gridcolor': '#333333'},
        legend=dict(font={'color': '#ffffff'})
    )
    return fig


# =============================================================================
# 11. WATERFALL CHART - Biểu đồ thác nước
# =============================================================================
def create_waterfall_chart(data: dict, title: str = "Waterfall", chart_key: str = "waterfall") -> go.Figure:
    """
    Biểu đồ waterfall cho phân tích đóng góp
    
    Args:
        data: Dict {label: value} (positive/negative values)
        title: Tiêu đề chart
        chart_key: Unique key cho chart
    
    Returns:
        Plotly Figure object
    """
    labels = list(data.keys())
    values = list(data.values())
    
    # Determine measure types
    measures = ['relative'] * len(values)
    measures[0] = 'absolute'  # First is base
    measures[-1] = 'total'    # Last is total
    
    fig = go.Figure(go.Waterfall(
        orientation='v',
        measure=measures,
        x=labels,
        y=values,
        connector={'line': {'color': '#555555'}},
        decreasing={'marker': {'color': '#10B981'}},
        increasing={'marker': {'color': '#E31E24'}},
        totals={'marker': {'color': '#3B82F6'}},
        textposition='outside',
        textfont={'color': '#ffffff'}
    ))
    
    fig.update_layout(
        title={'text': title, 'font': {'color': '#ffffff', 'size': 14}},
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff'},
        xaxis={'gridcolor': '#333333'},
        yaxis={'gridcolor': '#333333'},
        showlegend=False
    )
    return fig


# =============================================================================
# 12. FUNNEL CHART - Biểu đồ phễu
# =============================================================================
def create_funnel_chart(data: dict, title: str = "Funnel", chart_key: str = "funnel") -> go.Figure:
    """
    Biểu đồ funnel
    
    Args:
        data: Dict {stage: value}
        title: Tiêu đề chart
        chart_key: Unique key cho chart
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure(go.Funnel(
        y=list(data.keys()),
        x=list(data.values()),
        textinfo='value+percent initial',
        textfont={'color': '#ffffff'},
        marker_color=['#E31E24', '#F59E0B', '#10B981', '#3B82F6', '#8B5CF6'][:len(data)]
    ))
    
    fig.update_layout(
        title={'text': title, 'font': {'color': '#ffffff', 'size': 14}},
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff'}
    )
    return fig


# =============================================================================
# 13. BULLET CHART - Biểu đồ bullet
# =============================================================================
def create_bullet_chart(value: float, target: float, ranges: list = None, title: str = "Progress", chart_key: str = "bullet") -> go.Figure:
    """
    Biểu đồ bullet/progress
    
    Args:
        value: Current value
        target: Target value
        ranges: List of [poor, satisfactory, good] thresholds
        title: Tiêu đề chart
        chart_key: Unique key cho chart
    
    Returns:
        Plotly Figure object
    """
    if ranges is None:
        ranges = [30, 60, 100]
    
    fig = go.Figure(go.Indicator(
        mode="number+gauge+delta",
        value=value,
        delta={'reference': target, 'relative': True},
        gauge={
            'shape': 'bullet',
            'axis': {'range': [0, max(ranges)]},
            'threshold': {
                'line': {'color': '#ffffff', 'width': 2},
                'thickness': 0.75,
                'value': target
            },
            'steps': [
                {'range': [0, ranges[0]], 'color': 'rgba(227,30,36,0.3)'},
                {'range': [ranges[0], ranges[1]], 'color': 'rgba(245,158,11,0.3)'},
                {'range': [ranges[1], ranges[2]], 'color': 'rgba(16,185,129,0.3)'}
            ],
            'bar': {'color': '#E31E24'}
        },
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    
    fig.update_layout(
        title={'text': title, 'font': {'color': '#ffffff', 'size': 14}},
        height=150,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff'}
    )
    return fig


# =============================================================================
# 14. GROUPED BAR CHART - Biểu đồ bar nhóm
# =============================================================================
def create_grouped_bar_chart(data: dict, categories: list, title: str = "Comparison", chart_key: str = "grouped_bar") -> go.Figure:
    """
    Biểu đồ bar nhóm
    
    Args:
        data: Dict {group_name: [values]}
        categories: List of category names
        title: Tiêu đề chart
        chart_key: Unique key cho chart
    
    Returns:
        Plotly Figure object
    """
    colors = ['#E31E24', '#10B981', '#F59E0B', '#3B82F6']
    
    fig = go.Figure()
    
    for i, (name, values) in enumerate(data.items()):
        fig.add_trace(go.Bar(
            name=name,
            x=categories,
            y=values,
            marker_color=colors[i % len(colors)],
            text=[f'{v:,.0f}' for v in values],
            textposition='outside',
            textfont={'color': '#ffffff'}
        ))
    
    fig.update_layout(
        title={'text': title, 'font': {'color': '#ffffff', 'size': 14}},
        height=300,
        margin=dict(l=10, r=10, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff'},
        xaxis={'gridcolor': '#333333'},
        yaxis={'gridcolor': '#333333'},
        legend=dict(font={'color': '#ffffff'}),
        barmode='group'
    )
    return fig


# =============================================================================
# 15. STACKED BAR CHART - Biểu đồ bar xếp chồng
# =============================================================================
def create_stacked_bar_chart(data: dict, categories: list, title: str = "Stacked", chart_key: str = "stacked_bar") -> go.Figure:
    """
    Biểu đồ bar xếp chồng
    
    Args:
        data: Dict {group_name: [values]}
        categories: List of category names
        title: Tiêu đề chart
        chart_key: Unique key cho chart
    
    Returns:
        Plotly Figure object
    """
    colors = ['#E31E24', '#10B981', '#F59E0B', '#3B82F6']
    
    fig = go.Figure()
    
    for i, (name, values) in enumerate(data.items()):
        fig.add_trace(go.Bar(
            name=name,
            x=categories,
            y=values,
            marker_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        title={'text': title, 'font': {'color': '#ffffff', 'size': 14}},
        height=300,
        margin=dict(l=10, r=10, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff'},
        xaxis={'gridcolor': '#333333'},
        yaxis={'gridcolor': '#333333'},
        legend=dict(font={'color': '#ffffff'}),
        barmode='stack'
    )
    return fig


# =============================================================================
# PROFILE METRICS - Streamlit component
# =============================================================================
def display_profile_metrics(profile: dict):
    """
    Hiển thị metrics cards cho applicant profile
    
    Args:
        profile: Dict với keys: income, age, credit, risk
    """
    cols = st.columns(4)
    
    metrics = [
        ("💰 Income", f"${profile.get('income', 0):,.0f}", "Annual"),
        ("📅 Age", f"{profile.get('age', 0)}", "Years"),
        ("💳 Credit", f"${profile.get('credit', 0):,.0f}", "Amount"),
        ("📊 Risk", f"{profile.get('risk', 0):.1%}", "Score")
    ]
    
    for col, (label, value, subtitle) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div style="background:#333333;padding:12px;border-radius:8px;text-align:center;">
                <div style="font-size:12px;color:#707070;">{label}</div>
                <div style="font-size:20px;font-weight:600;color:#ffffff;">{value}</div>
                <div style="font-size:10px;color:#505050;">{subtitle}</div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS - Các hàm tiện ích
# =============================================================================
def get_risk_color(score: float) -> str:
    """Trả về màu dựa trên điểm rủi ro"""
    if score < 0.3:
        return "#10B981"  # Green - Low risk
    elif score < 0.6:
        return "#F59E0B"  # Yellow - Medium risk
    else:
        return "#E31E24"  # Red - High risk


def get_risk_label(score: float) -> str:
    """Trả về nhãn dựa trên điểm rủi ro"""
    if score < 0.3:
        return "Low Risk"
    elif score < 0.6:
        return "Medium Risk"
    else:
        return "High Risk"


# =============================================================================
# DEMO DATA - Dữ liệu mẫu cho testing
# =============================================================================
DEMO_FEATURE_IMPORTANCE = {
    'EXT_SOURCE_2': -0.35,
    'EXT_SOURCE_3': -0.28,
    'DAYS_BIRTH': 0.15,
    'AMT_CREDIT': 0.12,
    'AMT_ANNUITY': 0.10,
    'DAYS_EMPLOYED': -0.09,
    'AMT_INCOME': -0.08,
    'CNT_CHILDREN': 0.05
}

DEMO_COMPARISON_DATA = {
    '100001': {'Income': 70, 'Credit': 60, 'Age': 45, 'Risk': 30, 'Employment': 80},
    '100002': {'Income': 50, 'Credit': 75, 'Age': 55, 'Risk': 45, 'Employment': 60}
}

DEMO_PROFILE = {
    'income': 150000,
    'age': 35,
    'credit': 500000,
    'risk': 0.174
}
