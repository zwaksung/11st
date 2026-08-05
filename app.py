import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import google.generativeai as genai

# 1. Page Configuration & Custom CSS
st.set_page_config(
    page_title="AI Performance Hub | 11st Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium UI Theme using CSS Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Noto+Sans+KR:wght@300;400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', 'Noto Sans KR', sans-serif;
    }
    
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8F00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    /* Executive Summary Card styling */
    .summary-card {
        background: rgba(255, 75, 75, 0.04);
        border-left: 5px solid #FF4B4B;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.02);
    }
    .summary-title {
        font-weight: 700;
        font-size: 1.15rem;
        color: #FF4B4B;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .summary-text {
        font-size: 0.95rem;
        color: #2F3E46;
        line-height: 1.6;
    }
    
    /* Metric styling adjustments */
    div[data-testid="stMetric"] {
        background-color: #F8F9FA;
        border: 1px solid #E9ECEF;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.01);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-color: #FF4B4B;
    }
    
    /* Sidebar styling adjustments */
    .css-1d391kg {
        background-color: #FAFBFD;
    }
</style>
""", unsafe_allow_html=True)

# 2. Cached Data Loading & Preprocessing
@st.cache_data
def load_and_preprocess_data(filepath):
    # CSV file is encoded in cp949
    df = pd.read_csv(filepath, encoding='cp949')
    
    # Parse date column
    df['일자'] = pd.to_datetime(df['일자'])
    
    # Ensure numerical columns are correctly casted and fill NaNs
    num_cols = ['노출', '클릭', '집행 광고비', '결제거래액', '결제건수']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    # Calculate performance ratios at row-level (safely to prevent division by zero)
    df['ROAS'] = np.where(df['집행 광고비'] > 0, (df['결제거래액'] / df['집행 광고비']) * 100, 0.0)
    df['CTR'] = np.where(df['노출'] > 0, (df['클릭'] / df['노출']) * 100, 0.0)
    df['CPC'] = np.where(df['클릭'] > 0, df['집행 광고비'] / df['클릭'], 0.0)
    df['CVR'] = np.where(df['클릭'] > 0, (df['결제건수'] / df['클릭']) * 100, 0.0)
    
    return df

# Initialize session state for chatbot
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load data
csv_file_path = 'AI강의_11번가실습_260731.csv'
try:
    df_raw = load_and_preprocess_data(csv_file_path)
except Exception as e:
    st.error(f"데이터 파일을 불러오지 못했습니다. 경로와 인코딩을 확인하세요. 에러: {str(e)}")
    st.stop()

# 3. Sidebar Configuration (Filters & API Keys)
st.sidebar.image("https://img.icons8.com/color/96/000000/analytics.png", width=60)
st.sidebar.markdown("### **AI Performance Hub Settings**")

# Gemini API Key Input
gemini_key = st.sidebar.text_input(
    "🔑 **Gemini API Key**",
    type="password",
    placeholder="AI 챗봇용 API Key를 입력하세요.",
    help="Google AI Studio에서 발급받은 API 키를 입력하면 실시간 성과 브리핑 및 맞춤형 분석이 가능합니다."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### **대시보드 필터**")

# Media Filter Selection
available_media = sorted(df_raw['매체'].dropna().unique().tolist())
select_all_media = st.sidebar.checkbox("🌐 모든 매체 선택", value=True)

if select_all_media:
    selected_media = available_media
else:
    selected_media = st.sidebar.multiselect(
        "매체 필터",
        options=available_media,
        default=available_media[:3] if len(available_media) >= 3 else available_media
    )

# Date Filter Selection
min_date = df_raw['일자'].min().date()
max_date = df_raw['일자'].max().date()

selected_date_range = st.sidebar.date_input(
    "📅 조회 기간",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Parse Date Range safely (handling partial select when user is interacting)
if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
    start_date, end_date = selected_date_range
else:
    start_date = selected_date_range[0] if isinstance(selected_date_range, tuple) else selected_date_range
    end_date = max_date

# 4. Filter Data Based on User Input
filtered_df = df_raw[
    (df_raw['매체'].isin(selected_media)) &
    (df_raw['일자'].dt.date >= start_date) &
    (df_raw['일자'].dt.date <= end_date)
]

# Calculate previous period data for comparison delta
days_diff = (end_date - start_date).days + 1
prev_start_date = start_date - datetime.timedelta(days=days_diff)
prev_end_date = start_date - datetime.timedelta(days=1)

prev_df = df_raw[
    (df_raw['매체'].isin(selected_media)) &
    (df_raw['일자'].dt.date >= prev_start_date) &
    (df_raw['일자'].dt.date <= prev_end_date)
]

# 5. Core Metric Calculations
def calculate_metrics(df):
    spend = df['집행 광고비'].sum()
    revenue = df['결제거래액'].sum()
    roas = (revenue / spend * 100) if spend > 0 else 0.0
    clicks = df['클릭'].sum()
    impressions = df['노출'].sum()
    ctr = (clicks / impressions * 100) if impressions > 0 else 0.0
    cpc = (spend / clicks) if clicks > 0 else 0.0
    transactions = df['결제건수'].sum()
    cvr = (transactions / clicks * 100) if clicks > 0 else 0.0
    return spend, revenue, roas, clicks, impressions, ctr, cpc, transactions, cvr

curr_spend, curr_rev, curr_roas, curr_clicks, curr_imp, curr_ctr, curr_cpc, curr_trans, curr_cvr = calculate_metrics(filtered_df)
prev_spend, prev_rev, prev_roas, prev_clicks, prev_imp, prev_ctr, prev_cpc, prev_trans, prev_cvr = calculate_metrics(prev_df)

# Helpers for deltas
def get_delta_str(curr, prev, is_percentage=False, is_cpc=False):
    if prev == 0:
        return "N/A"
    change = ((curr - prev) / prev) * 100
    if is_cpc:
        # Lower CPC is better
        return f"{change:+.1f}% {'(개선)' if change < 0 else '(상승)'}"
    return f"{change:+.1f}%"

# 6. Main UI Layout (Full Width Dashboard + Floating AI Chatbot Widget)
title_col, chatbot_widget_col = st.columns([3, 1])

with title_col:
    st.markdown('<div class="main-title">AI 결합형 공동 퍼포먼스 대시보드</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">실시간 성과 모니터링 & AI 기여도 분석 허브</div>', unsafe_allow_html=True)

with chatbot_widget_col:
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    with st.popover("🤖 **AI Analyst 챗봇 열기**", use_container_width=True):
        draw_chatbot()
    
    # ----------------------------------------------------
    # Section 1: Executive Summary Comment
    # ----------------------------------------------------
    st.markdown("### 💡 **전일자 및 기간 핵심 성과 요약**")
    
    # Identify top media and top campaign
    top_media = "N/A"
    best_roas_media = "N/A"
    if not filtered_df.empty:
        media_agg = filtered_df.groupby('매체').agg({'집행 광고비': 'sum', '결제거래액': 'sum'}).reset_index()
        media_agg['ROAS'] = (media_agg['결제거래액'] / media_agg['집행 광고비'] * 100).fillna(0)
        
        top_media_row = media_agg.sort_values(by='결제거래액', ascending=False).iloc[0] if not media_agg.empty else None
        best_roas_row = media_agg.sort_values(by='ROAS', ascending=False).iloc[0] if not media_agg.empty else None
        
        if top_media_row is not None:
            top_media = f"{top_media_row['매체']} (매출 {top_media_row['결제거래액']:,.0f}원)"
        if best_roas_row is not None:
            best_roas_media = f"{best_roas_row['매체']} (ROAS {best_roas_row['ROAS']:.1f}%)"
            
    rule_summary = f"""현재 선택된 기간 동안 집행된 총 광고비는 **{curr_spend:,.0f}원**, 총 결제거래액은 **{curr_rev:,.0f}원**이며 평균 ROAS는 **{curr_roas:.1f}%**를 기록하고 있습니다. 
    가장 높은 매출을 창출한 매체는 **{top_media}**이며, 가장 효율이 우수한 ROAS 매체는 **{best_roas_media}**입니다. 
    이전 동기간 대비 광고비는 **{get_delta_str(curr_spend, prev_spend)}**, 매출액은 **{get_delta_str(curr_rev, prev_rev)}**, ROAS는 **{curr_roas - prev_roas:+.1f}%p** 변동하였습니다."""

    summary_placeholder = st.empty()
    summary_placeholder.markdown(f"""
    <div class="summary-card">
        <div class="summary-title">📈 실시간 성과 브리핑 (Rule-Based)</div>
        <div class="summary-text">{rule_summary}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Dynamic LLM summary generation if Gemini API key exists
    if gemini_key:
        try:
            # Let's perform a fast async/sync call to update the placeholder with LLM summary
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-3.5-flash")
            
            prompt_summary = f"""
            너는 10년 차 광고 대행사 AE이자 퍼포먼스 마케팅 디렉터야. 
            아래 제공된 데이터를 바탕으로 광고주에게 보낼 '전일자 및 선택 기간 핵심 성과 브리핑 요약'을 작성해 줘.
            데이터 수치에 완전히 근거해야 하며, 전문적인 톤앤매너로 작성하되 3문장 이내로 명확하게 브리핑해 줘.
            
            [조회 데이터 세부 요약]
            - 조회 기간: {start_date} ~ {end_date}
            - 집행 광고비: {curr_spend:,.0f}원 (이전 동기 대비 {get_delta_str(curr_spend, prev_spend)})
            - 결제거래액: {curr_rev:,.0f}원 (이전 동기 대비 {get_delta_str(curr_rev, prev_rev)})
            - 평균 ROAS: {curr_roas:.1f}% (이전 동기 대비 {curr_roas - prev_roas:+.1f}%p 변동)
            - 탑 매출 매체: {top_media}
            - 최고 효율 매체: {best_roas_media}
            """
            
            # Use brief timeout or fast generate
            response = model.generate_content(prompt_summary)
            llm_brief = response.text
            
            summary_placeholder.markdown(f"""
            <div class="summary-card">
                <div class="summary-title">✨ AI 실시간 성과 인사이트 브리핑</div>
                <div class="summary-text">{llm_brief}</div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            # Fallback quietly to rule-based summary
            pass

    # ----------------------------------------------------
    # Section 2: [전문가 추천] KPI 스코어카드 (3열 2줄)
    # ----------------------------------------------------
    st.markdown("### 📊 **핵심 성과 지표 (KPI Scorecard)**")
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    kpi_col4, kpi_col5, kpi_col6 = st.columns(3)
    
    with kpi_col1:
        st.metric(
            label="집행 광고비 (Cost)",
            value=f"₩ {curr_spend:,.0f}",
            delta=get_delta_str(curr_spend, prev_spend),
            delta_color="normal"
        )
    with kpi_col2:
        st.metric(
            label="결제거래액 (Revenue)",
            value=f"₩ {curr_rev:,.0f}",
            delta=get_delta_str(curr_rev, prev_rev),
            delta_color="normal"
        )
    with kpi_col3:
        st.metric(
            label="평균 ROAS",
            value=f"{curr_roas:.1f} %",
            delta=f"{curr_roas - prev_roas:+.1f}%p",
            delta_color="normal"
        )
    with kpi_col4:
        st.metric(
            label="클릭수 (Clicks)",
            value=f"{curr_clicks:,.0f} 회",
            delta=get_delta_str(curr_clicks, prev_clicks),
            delta_color="normal"
        )
    with kpi_col5:
        st.metric(
            label="클릭율 (CTR)",
            value=f"{curr_ctr:.2f} %",
            delta=f"{curr_ctr - prev_ctr:+.2f}%p",
            delta_color="normal"
        )
    with kpi_col6:
        st.metric(
            label="평균 클릭단가 (CPC)",
            value=f"₩ {curr_cpc:,.0f}",
            delta=get_delta_str(curr_cpc, prev_cpc, is_cpc=True),
            delta_color="inverse"
        )

    st.markdown("---")

    # ----------------------------------------------------
    # Section 3: 데일리 성과 추이 그래프 (시계열)
    # ----------------------------------------------------
    st.markdown("### 📈 **데일리 성과 추이 분석**")
    
    daily_grouped = filtered_df.groupby('일자').agg({
        '집행 광고비': 'sum',
        '결제거래액': 'sum',
        '노출': 'sum',
        '클릭': 'sum',
        '결제건수': 'sum'
    }).reset_index()
    
    daily_grouped['ROAS'] = np.where(daily_grouped['집행 광고비'] > 0, (daily_grouped['결제거래액'] / daily_grouped['집행 광고비']) * 100, 0.0)
    daily_grouped['CTR'] = np.where(daily_grouped['노출'] > 0, (daily_grouped['클릭'] / daily_grouped['노출']) * 100, 0.0)
    daily_grouped['CPC'] = np.where(daily_grouped['클릭'] > 0, daily_grouped['집행 광고비'] / daily_grouped['클릭'], 0.0)
    
    # Toggle metrics for lines
    selected_metrics = st.multiselect(
        "분석 대상 지표 선택 (다중 선택 가능)",
        options=['집행 광고비', '결제거래액', 'ROAS', '클릭', 'CTR', 'CPC'],
        default=['집행 광고비', '결제거래액']
    )
    
    if not daily_grouped.empty and selected_metrics:
        fig_trend = go.Figure()
        
        # Determine if we have multiple units (currency vs. percent)
        # For simplicity, if we mix ROAS/CTR with currency, we plot on secondary axes or standardize.
        for metric in selected_metrics:
            if metric in ['ROAS', 'CTR']:
                fig_trend.add_trace(go.Scatter(
                    x=daily_grouped['일자'],
                    y=daily_grouped[metric],
                    name=metric,
                    mode='lines+markers',
                    line=dict(width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>" + metric + ": %{y:.2f}%"
                ))
            elif metric in ['CPC']:
                fig_trend.add_trace(go.Scatter(
                    x=daily_grouped['일자'],
                    y=daily_grouped[metric],
                    name=metric,
                    mode='lines+markers',
                    line=dict(width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>" + metric + ": ₩%{y:,.0f}"
                ))
            else:
                fig_trend.add_trace(go.Scatter(
                    x=daily_grouped['일자'],
                    y=daily_grouped[metric],
                    name=metric,
                    mode='lines',
                    line=dict(width=2.5),
                    hovertemplate="%{x|%Y-%m-%d}<br>" + metric + ": ₩%{y:,.0f}"
                ))
                
        fig_trend.update_layout(
            template="plotly_white",
            margin=dict(l=40, r=40, t=20, b=40),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=True, gridcolor="#F1F3F5"),
            yaxis=dict(showgrid=True, gridcolor="#F1F3F5", title="지표 단위 (원 / % / 회)")
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("시계열 그래프를 그릴 지표를 선택해 주세요.")

    # ----------------------------------------------------
    # Section 4: 매체별 결제거래액 & ROAS 비교 그래프 (이중축 콤보 차트)
    # ----------------------------------------------------
    st.markdown("### 📊 **매체별 결제거래액 & ROAS 비교**")
    
    media_grouped = filtered_df.groupby('매체').agg({
        '집행 광고비': 'sum',
        '결제거래액': 'sum'
    }).reset_index()
    media_grouped['ROAS'] = np.where(media_grouped['집행 광고비'] > 0, (media_grouped['결제거래액'] / media_grouped['집행 광고비']) * 100, 0.0)
    
    if not media_grouped.empty:
        # Sort by Transaction Amount (결제거래액)
        media_grouped = media_grouped.sort_values(by='결제거래액', ascending=False)
        
        fig_combo = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Bar chart for Revenue
        fig_combo.add_trace(
            go.Bar(
                x=media_grouped['매체'],
                y=media_grouped['결제거래액'],
                name="결제거래액 (Revenue)",
                marker_color="#FF4B4B",
                hovertemplate="매체: %{x}<br>거래액: ₩%{y:,.0f}"
            ),
            secondary_y=False
        )
        
        # Line chart for ROAS
        fig_combo.add_trace(
            go.Scatter(
                x=media_grouped['매체'],
                y=media_grouped['ROAS'],
                name="ROAS (%)",
                mode="lines+markers",
                line=dict(color="#FF8F00", width=3),
                marker=dict(size=8),
                hovertemplate="매체: %{x}<br>ROAS: %{y:.1f}%"
            ),
            secondary_y=True
        )
        
        fig_combo.update_layout(
            template="plotly_white",
            margin=dict(l=40, r=40, t=20, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=False),
            yaxis=dict(title="결제거래액 (원)", showgrid=True, gridcolor="#F1F3F5"),
            yaxis2=dict(title="ROAS (%)", showgrid=False, overlaying="y", side="right")
        )
        st.plotly_chart(fig_combo, use_container_width=True)
    else:
        st.info("데이터가 없습니다.")

    # ----------------------------------------------------
    # Section 5: 효율 매트릭스 산점도 (Scatter Plot)
    # ----------------------------------------------------
    st.markdown("### 🎯 **제휴사별 효율 매트릭스 (광고비 vs ROAS)**")
    
    # We group by '제휴사명' (Campaign level details)
    campaign_grouped = filtered_df.groupby(['제휴사명', '매체']).agg({
        '집행 광고비': 'sum',
        '결제거래액': 'sum',
        '클릭': 'sum'
    }).reset_index()
    campaign_grouped['ROAS'] = np.where(campaign_grouped['집행 광고비'] > 0, (campaign_grouped['결제거래액'] / campaign_grouped['집행 광고비']) * 100, 0.0)
    campaign_grouped = campaign_grouped[campaign_grouped['집행 광고비'] > 0] # Filter out zero spends for visual clarity
    
    if not campaign_grouped.empty:
        fig_scatter = px.scatter(
            campaign_grouped,
            x="집행 광고비",
            y="ROAS",
            size="결제거래액",
            color="매체",
            hover_name="제휴사명",
            log_x=True, # Log scale on X as spend ranges can vary significantly
            title="제휴사별 광고비 대비 ROAS 산점도 (원 크기: 결제거래액)",
            labels={"집행 광고비": "집행 광고비 (원, 로그스케일)", "ROAS": "ROAS (%)"},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        # Add target horizontal reference line for target ROAS (e.g. 300% or average)
        fig_scatter.add_hline(y=curr_roas, line_dash="dash", line_color="gray", annotation_text=f"전체 평균 ROAS ({curr_roas:.1f}%)")
        
        fig_scatter.update_layout(
            template="plotly_white",
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("산점도를 구성할 집행 데이터가 충분하지 않습니다.")

# ==========================================
# RIGHT COLUMN: AI CHATBOT AGENT
# ==========================================
def draw_chatbot():
    st.markdown('<div class="sidebar-anchor"></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title" style="font-size:1.8rem; background: linear-gradient(135deg, #FF8F00 0%, #FFD600 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">AI Performance Analyst</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title" style="margin-bottom:1rem;">데이터 기반 퍼포먼스 마케팅 어시스턴트</div>', unsafe_allow_html=True)
    
    # 1. Structured Data Context preparation for Gemini
    # Media Summary (Markdown)
    llm_media_df = filtered_df.groupby('매체').agg({
        '집행 광고비': 'sum',
        '결제거래액': 'sum',
        '노출': 'sum',
        '클릭': 'sum',
        '결제건수': 'sum'
    }).reset_index()
    
    llm_media_df['ROAS(%)'] = np.where(llm_media_df['집행 광고비'] > 0, (llm_media_df['결제거래액'] / llm_media_df['집행 광고비'] * 100).round(1), 0.0)
    llm_media_df['CTR(%)'] = np.where(llm_media_df['노출'] > 0, (llm_media_df['클릭'] / llm_media_df['노출'] * 100).round(2), 0.0)
    llm_media_df['CPC(원)'] = np.where(llm_media_df['클릭'] > 0, (llm_media_df['집행 광고비'] / llm_media_df['클릭']).round(0), 0.0)
    llm_media_df['CVR(%)'] = np.where(llm_media_df['클릭'] > 0, (llm_media_df['결제건수'] / llm_media_df['클릭'] * 100).round(2), 0.0)
    
    # Drop raw values for compact tokens, keeping the KPIs and totals
    media_context_table = llm_media_df[['매체', '집행 광고비', '결제거래액', 'ROAS(%)', 'CTR(%)', 'CPC(원)', 'CVR(%)']].to_markdown(index=False)
    
    # Campaign Summary (Top 8 campaigns by spend)
    llm_camp_df = filtered_df.groupby(['제휴사명', '매체']).agg({
        '집행 광고비': 'sum',
        '결제거래액': 'sum',
        '클릭': 'sum',
        '결제건수': 'sum'
    }).reset_index()
    
    llm_camp_df['ROAS(%)'] = np.where(llm_camp_df['집행 광고비'] > 0, (llm_camp_df['결제거래액'] / llm_camp_df['집행 광고비'] * 100).round(1), 0.0)
    llm_camp_df['CVR(%)'] = np.where(llm_camp_df['클릭'] > 0, (llm_camp_df['결제건수'] / llm_camp_df['클릭'] * 100).round(2), 0.0)
    llm_camp_top = llm_camp_df.sort_values(by='집행 광고비', ascending=False).head(8)
    
    campaign_context_table = llm_camp_top[['제휴사명', '매체', '집행 광고비', '결제거래액', 'ROAS(%)', 'CVR(%)']].to_markdown(index=False)
    
    # System Instruction Definition (AE personality & context)
    system_instruction = f"""
    당신은 10년 차 이상의 최상급 디지털 광고 AE이자 퍼포먼스 마케팅 전문가입니다. 
    사용자가 질문하면 제공된 대시보드의 데이터 컨텍스트에 전적으로 근거하여 정확하고 분석적인 마케팅 피드백을 제공하세요.
    수치를 가공하거나 거짓 정보를 주지 마세요. 부족한 경우 컨텍스트에 주어진 데이터만 활용 가능하다고 명시하세요.
    마케팅 용어(ROAS, CPC, CTR, CVR 등)를 적재적소에 사용하고 예산 배분 제안이나 성과 개선 원인 분석 등의 AE적 인사이트를 추가해 주면 매우 좋습니다.
    친절하고 전문적인 비즈니스 톤앤매너(한국어)로 답변해 주세요.
    
    [실시간 대시보드 데이터 컨텍스트]
    - 조회 범위: {start_date} ~ {end_date} (총 {days_diff}일)
    - 매체 필터 상태: {", ".join(selected_media)}
    
    [종합 요약 KPI]
    - 총 광고비: {curr_spend:,.0f}원 (이전 기간 대비 {get_delta_str(curr_spend, prev_spend)})
    - 총 결제거래액: {curr_rev:,.0f}원 (이전 기간 대비 {get_delta_str(curr_rev, prev_rev)})
    - 평균 ROAS: {curr_roas:.1f}% (이전 기간 대비 {curr_roas - prev_roas:+.1f}%p)
    - 총 클릭수: {curr_clicks:,.0f}회
    - 평균 CTR: {curr_ctr:.2f}%
    - 평균 CPC: {curr_cpc:,.0f}원 (이전 기간 대비 {get_delta_str(curr_cpc, prev_cpc, is_cpc=True)})
    - 평균 CVR: {curr_cvr:.2f}%
    
    [매체별 KPI 집계 요약]
    {media_context_table}
    
    [상위 집행 제휴사(캠페인) 상세 요약 - Top 8]
    {campaign_context_table}
    """

    # Chat Container
    chat_container = st.container(height=500)
    
    # Render Chat History
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
    # Chat Input Process
    if user_query := st.chat_input("광고 성과에 대해 무엇이든 물어보세요 (예: 'ROAS 효율이 가장 높은 매체는?')"):
        # Display user message
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Exception check: API Key
        if not gemini_key:
            with chat_container:
                with st.chat_message("assistant"):
                    st.warning("⚠️ 좌측 사이드바에 Gemini API Key를 입력해주세요.")
            st.session_state.messages.append({"role": "assistant", "content": "⚠️ 좌측 사이드바에 Gemini API Key를 입력해주세요."})
        else:
            with chat_container:
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    try:
                        # Call Google Gemini API (streaming format)
                        genai.configure(api_key=gemini_key)
                        model = genai.GenerativeModel(
                            model_name="gemini-3.5-flash",
                            system_instruction=system_instruction
                        )
                        
                        # Prepare the dialog structure for model request
                        contents = []
                        for msg in st.session_state.messages:
                            role = "user" if msg["role"] == "user" else "model"
                            contents.append({
                                "role": role,
                                "parts": [msg["content"]]
                            })
                            
                        response_stream = model.generate_content(contents, stream=True)
                        for chunk in response_stream:
                            full_response += chunk.text
                            message_placeholder.markdown(full_response + "▌")
                        message_placeholder.markdown(full_response)
                        
                    except Exception as e:
                        full_response = f"죄송합니다. Gemini API 요청 중 오류가 발생했습니다: {str(e)}"
                        message_placeholder.error(full_response)
                        
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
    # Reset Chat button
    if st.button("🔄 대화 기록 초기화"):
        st.session_state.messages = []
        st.rerun()


