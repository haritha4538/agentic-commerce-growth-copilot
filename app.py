"""
Agentic Commerce Growth Copilot — Streamlit Dashboard (Phases 3 + 7)

This file is UI ONLY. Every deterministic number shown here comes from
src/analytics/analytics_engine.py — no calculation happens in this file.

The AI Strategy Center tab (Phase 7) triggers the existing four-agent
LangGraph pipeline (src/orchestration/graph.py — Analytics -> Retrieval ->
Strategy -> Validator, built in Phases 4-6) and displays its cited,
validated output. This file does not call Gemini, ChromaDB, or LangGraph
directly — it only invokes run_growth_copilot_graph() and renders the
result.

Run:
    streamlit run app.py
"""

import html
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make `src` importable regardless of where streamlit is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.loader import load_sample_data
from src.analytics.analytics_engine import run_full_analysis
# Phase 7: the LangGraph pipeline (Phase 6). Safe to import unconditionally —
# langgraph/chromadb/google-genai are all imported lazily deeper in this
# module chain, so this import succeeds even before those packages are
# installed; only actually running the pipeline requires them.
from src.orchestration.graph import run_growth_copilot_graph, GraphBuildError
# Phase 8: Commerce Action Center (Razorpay Test Mode, Payment Links only).
# Same lazy-import discipline as above — the 'razorpay' package is only
# imported deep inside razorpay_client.py when a Payment Link is actually
# created, so this import is always safe even before it's installed.
from src.commerce.razorpay_client import (
    create_payment_link,
    fetch_payment_link_status,
    is_test_mode_configured,
    RazorpayCredentialsError,
    RazorpayClientError,
    FIXED_TEST_AMOUNT_INR,
)
from src.commerce.action_mapper import build_commerce_action
from src.commerce.audit_log import append_audit_record, load_audit_log, update_audit_record_result

# ==========================================================================
# PAGE CONFIG + THEME
# ==========================================================================

st.set_page_config(
    page_title="Agentic Commerce Growth Copilot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    html, body, [class*="css"]  {
        font-family: -apple-system, "Segoe UI", "Inter", sans-serif;
    }
    #MainMenu, footer, header {visibility: hidden;}

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }

    /* ---- KPI cards ---- */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e6e8ec;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        height: 118px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        color: #667085;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #101828;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 0.80rem;
        margin-top: 6px;
    }
    .kpi-sub.positive { color: #12794f; font-weight: 600; }
    .kpi-sub.negative { color: #b42318; font-weight: 600; }
    .kpi-sub.neutral  { color: #98a2b3; }

    /* ---- Section headers ---- */
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #101828;
        margin: 4px 0 2px 0;
    }
    .section-caption {
        color: #667085;
        font-size: 0.85rem;
        margin-bottom: 14px;
    }

    /* ---- Signal cards ---- */
    .signal-card {
        background: #ffffff;
        border: 1px solid #e6e8ec;
        border-left: 4px solid #5850ec;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .signal-card .signal-title {
        font-weight: 700;
        color: #101828;
        font-size: 0.92rem;
        margin-bottom: 3px;
    }
    .signal-card .signal-desc {
        color: #475467;
        font-size: 0.86rem;
        line-height: 1.4;
    }
    .signal-card.churn { border-left-color: #b42318; }
    .signal-card.margin { border-left-color: #b54708; }
    .signal-card.seasonal { border-left-color: #175cd3; }
    .signal-card.channel { border-left-color: #12794f; }

    .badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 9px;
        border-radius: 999px;
        background: #f2f4f7;
        color: #344054;
        margin-right: 6px;
    }

    .empty-state {
        text-align: center;
        color: #98a2b3;
        padding: 40px 10px;
        font-size: 0.9rem;
    }

    /* ---- Phase 7: AI Strategy Center cards ---- */
    .opportunity-card {
        background: #ffffff;
        border: 1px solid #e6e8ec;
        border-left: 4px solid #5850ec;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .opportunity-card .opp-title {
        font-weight: 700;
        color: #101828;
        font-size: 0.92rem;
        margin-bottom: 3px;
    }
    .opportunity-card .opp-desc {
        color: #475467;
        font-size: 0.86rem;
        line-height: 1.4;
        margin-bottom: 6px;
    }
    .opportunity-card .opp-meta .tag {
        display: inline-block;
        font-size: 0.74rem;
        font-weight: 600;
        background: #f2f4f7;
        color: #344054;
        border-radius: 999px;
        padding: 2px 9px;
        margin-right: 6px;
        margin-top: 2px;
    }
    .opportunity-card.risk-card { border-left-color: #b42318; }
    .opportunity-card.action-card { border-left-color: #175cd3; }

    .priority-badge {
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 999px;
        margin-left: 8px;
        vertical-align: middle;
    }
    .priority-high { background: #fee4e2; color: #b42318; }
    .priority-medium { background: #fef0c7; color: #b54708; }
    .priority-low { background: #e6f4ea; color: #12794f; }

    .citation-chip {
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 600;
        background: #eef4ff;
        color: #175cd3;
        border: 1px solid #d1e0ff;
        border-radius: 999px;
        padding: 3px 12px;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    /* ---- Phase 8: Commerce Action Center ---- */
    .test-mode-banner {
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        background: #fef0c7;
        color: #b54708;
        border: 1px solid #fedf89;
        border-radius: 999px;
        padding: 4px 14px;
        margin-bottom: 10px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

CURRENCY_SYMBOL = "₹"


# ==========================================================================
# SESSION STATE
# ==========================================================================

def _init_state():
    defaults = {
        "loaded_data": None,     # LoadedData from src.data.loader
        "analysis": None,        # dict from run_full_analysis()
        "load_error": None,      # string or dict, shown in sidebar/main area
        "data_source": None,     # "sample" (uploads land here in a later phase)
        # --- Phase 7: AI Strategy Center ---
        "graph_state": None,     # final GraphState dict from run_growth_copilot_graph()
        "graph_error": None,     # friendly string shown instead of a raw traceback
        # --- Phase 8: Commerce Action Center ---
        "commerce_last_result": None,   # dict describing the most recent Payment Link creation attempt
        "commerce_last_error": None,    # friendly string shown instead of a raw traceback
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_sample_into_state():
    """Loads the bundled sample CSVs through the existing loader/validator,
    then runs the full analytics bundle. Never lets an exception escape to
    a raw Streamlit traceback — always leaves session_state in a coherent
    state either way."""
    try:
        loaded = load_sample_data()
        st.session_state.loaded_data = loaded
        st.session_state.data_source = "sample"

        if not loaded.is_valid:
            st.session_state.analysis = None
            st.session_state.load_error = loaded.summary()
            return

        st.session_state.load_error = None
        recompute_analysis()
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        st.session_state.analysis = None
        st.session_state.load_error = f"Unexpected error while loading data: {exc}"


def recompute_analysis():
    """Re-runs the analytics engine over whatever data is currently loaded.
    Kept separate from loading so the sidebar's 'Refresh Analytics' button
    can recompute without re-reading files from disk."""
    loaded = st.session_state.loaded_data
    if loaded is None or not loaded.is_valid:
        st.session_state.analysis = None
        return
    try:
        st.session_state.analysis = run_full_analysis(
            customers_df=loaded.dataframes["customers"],
            products_df=loaded.dataframes["products"],
            orders_df=loaded.dataframes["orders"],
        )
        st.session_state.load_error = None
    except Exception as exc:  # noqa: BLE001
        st.session_state.analysis = None
        st.session_state.load_error = f"Unexpected error while computing analytics: {exc}"


def execute_strategy_pipeline():
    """Runs the existing Phase 6 LangGraph pipeline (Analytics -> Retrieval
    -> Strategy -> Validator) over whatever data is currently loaded, and
    caches the resulting state in session_state. Called by both the
    'Generate Strategy' and 'Refresh Strategy' buttons — the only
    difference between them is when the UI chooses to call this function,
    not what it does. Never lets an exception escape to a raw traceback."""
    loaded = st.session_state.loaded_data
    if loaded is None or not loaded.is_valid:
        st.session_state.graph_state = None
        st.session_state.graph_error = (
            "Please load data first — click **Load Sample Data** in the sidebar, then try again."
        )
        return

    with st.spinner("Running multi-agent pipeline: Analytics → Retrieval → Strategy → Validator..."):
        try:
            final_state = run_growth_copilot_graph(
                customers_df=loaded.dataframes["customers"],
                products_df=loaded.dataframes["products"],
                orders_df=loaded.dataframes["orders"],
            )
            st.session_state.graph_state = final_state
            st.session_state.graph_error = None
        except GraphBuildError as exc:
            st.session_state.graph_state = None
            st.session_state.graph_error = (
                f"The AI strategy pipeline could not be started: {exc}"
            )
        except Exception as exc:  # noqa: BLE001
            st.session_state.graph_state = None
            st.session_state.graph_error = f"Unexpected error while running the AI strategy pipeline: {exc}"


_init_state()


# ==========================================================================
# SMALL FORMATTING HELPERS
# ==========================================================================

def fmt_currency(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{CURRENCY_SYMBOL}{value:,.0f}"


def fmt_number(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.0f}"


def fmt_pct(value, signed: bool = True) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    sign = "+" if (signed and value > 0) else ""
    return f"{sign}{value:.1f}%"


def is_empty(df) -> bool:
    return df is None or (isinstance(df, pd.DataFrame) and df.empty)


def empty_state(message: str):
    st.markdown(f'<div class="empty-state">{message}</div>', unsafe_allow_html=True)


# ==========================================================================
# SIDEBAR
# ==========================================================================

with st.sidebar:
    st.markdown("### 🧭 Growth Copilot")
    st.caption("AI Growth & Agentic Commerce")
    st.divider()

    st.markdown("**Data Controls**")
    if st.button("📥 Load Sample Data", use_container_width=True, type="primary"):
        load_sample_into_state()

    refresh_disabled = st.session_state.loaded_data is None
    if st.button("🔄 Refresh Analytics", use_container_width=True, disabled=refresh_disabled):
        recompute_analysis()
        st.toast("Analytics recomputed.", icon="✅")

    st.divider()
    st.markdown("**Dataset Summary**")

    loaded = st.session_state.loaded_data
    if loaded is None:
        st.info("No data loaded yet. Click **Load Sample Data** to begin.")
    else:
        for name in ["customers", "products", "orders"]:
            df = loaded.dataframes.get(name)
            result = loaded.validation_results.get(name)
            row_count = len(df) if df is not None else 0
            status_icon = "✅" if (result and result.is_valid) else "⚠️"
            st.markdown(f"{status_icon} **{name.capitalize()}** — {row_count:,} rows")

        if not loaded.is_valid:
            with st.expander("View validation issues"):
                st.json(loaded.summary())

        st.caption(f"Source: {st.session_state.data_source or 'unknown'}")

    

# ==========================================================================
# MAIN HEADER
# ==========================================================================

st.markdown("## 📈 Agentic Commerce Growth Copilot")
st.caption("AI-powered business decision-support system — analytics shown below are calculated directly from your data.")

if st.session_state.load_error:
    st.error("There was a problem with the loaded data. See details below.")
    with st.expander("Error details"):
        st.write(st.session_state.load_error)

if st.session_state.analysis is None:
    st.markdown("---")
    st.info("👈 Click **Load Sample Data** in the sidebar to load the demo dataset and populate the dashboard.")
    st.stop()

analysis = st.session_state.analysis
kpis = analysis["executive_kpis"]


# ==========================================================================
# KPI CARDS
# ==========================================================================

def render_kpi_card(col, label, value, sub_text=None, sub_class="neutral"):
    with col:
        sub_html = f'<div class="kpi-sub {sub_class}">{sub_text}</div>' if sub_text else ""
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                {sub_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


kpi_cols = st.columns(6)

render_kpi_card(kpi_cols[0], "Total Revenue", fmt_currency(kpis.get("total_revenue")))
render_kpi_card(kpi_cols[1], "Total Orders", fmt_number(kpis.get("total_orders")))

active = kpis.get("active_customers")
registered = kpis.get("total_registered_customers")
render_kpi_card(
    kpi_cols[2], "Total Customers", fmt_number(active),
    sub_text=f"of {fmt_number(registered)} registered", sub_class="neutral",
)

render_kpi_card(kpi_cols[3], "Avg Order Value", fmt_currency(kpis.get("average_order_value")))
render_kpi_card(kpi_cols[4], "Units Sold", fmt_number(kpis.get("total_units_sold")))

growth = kpis.get("revenue_growth_pct_30d")
if growth is None:
    render_kpi_card(kpi_cols[5], "Growth % (30d)", "N/A", sub_text="Needs 60+ days of history", sub_class="neutral")
else:
    growth_class = "positive" if growth >= 0 else "negative"
    arrow = "▲" if growth >= 0 else "▼"
    render_kpi_card(
        kpi_cols[5], "Growth % (30d)", fmt_pct(growth),
        sub_text=f"{arrow} vs prior 30 days", sub_class=growth_class,
    )

st.markdown("<div style='height: 6px'></div>", unsafe_allow_html=True)


# ==========================================================================
# TABS
# ==========================================================================

tab_overview, tab_category_channel, tab_products, tab_signals, tab_strategy, tab_commerce = st.tabs(
    ["📊 Revenue Trend", "🗂 Category & Channel", "📦 Top Products", "🚀 Growth Signals",
     "🤖 AI Strategy Center", "💳 Commerce Action Center"]
)

# --- Tab 1: Revenue Trend -------------------------------------------------
with tab_overview:
    st.markdown('<div class="section-title">Monthly Revenue Trend</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Completed-order revenue and order count, aggregated by month.</div>',
        unsafe_allow_html=True,
    )

    monthly_df = analysis.get("revenue_by_month")
    if is_empty(monthly_df):
        empty_state("No revenue data available to chart yet.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_df["period"], y=monthly_df["revenue"],
            mode="lines+markers", name="Revenue",
            line=dict(color="#5850ec", width=3),
            marker=dict(size=6),
            hovertemplate="%{x}<br>Revenue: " + CURRENCY_SYMBOL + "%{y:,.0f}<extra></extra>",
        ))
        fig.update_layout(
            template="plotly_white",
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title=None,
            yaxis_title="Revenue",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    growth_df = analysis.get("monthly_growth")
    if not is_empty(growth_df):
        with st.expander("View month-over-month growth table"):
            st.dataframe(
                growth_df.rename(columns={
                    "period": "Month", "revenue": "Revenue", "orders": "Orders",
                    "revenue_growth_pct": "Growth vs Prior Month (%)",
                }),
                use_container_width=True, hide_index=True,
            )

# --- Tab 2: Category & Channel --------------------------------------------
with tab_category_channel:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Revenue by Category</div>', unsafe_allow_html=True)
        cat_df = analysis.get("revenue_by_category")
        if is_empty(cat_df):
            empty_state("No category data available.")
        else:
            fig = px.bar(
                cat_df.sort_values("revenue", ascending=True),
                x="revenue", y="category", orientation="h",
                text=cat_df.sort_values("revenue", ascending=True)["revenue"].map(lambda v: f"{CURRENCY_SYMBOL}{v:,.0f}"),
                color="revenue", color_continuous_scale="Blues",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                template="plotly_white", height=380, showlegend=False,
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Revenue", yaxis_title=None,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Average Order Value by Channel</div>', unsafe_allow_html=True)
        channel_df = analysis.get("revenue_by_channel")
        if is_empty(channel_df):
            empty_state("No acquisition channel data available.")
        else:
            fig = px.bar(
                channel_df.sort_values("aov", ascending=True),
                x="aov", y="acquisition_channel", orientation="h",
                text=channel_df.sort_values("aov", ascending=True)["aov"].map(lambda v: f"{CURRENCY_SYMBOL}{v:,.0f}"),
                color="aov", color_continuous_scale="Purples",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                template="plotly_white", height=380, showlegend=False,
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Average Order Value", yaxis_title=None,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title" style="margin-top:18px;">Revenue &amp; Orders by Channel</div>', unsafe_allow_html=True)
    if is_empty(channel_df):
        empty_state("No channel data available.")
    else:
        st.dataframe(
            channel_df.rename(columns={
                "acquisition_channel": "Channel", "revenue": "Revenue",
                "orders": "Orders", "customers": "Customers", "aov": "AOV",
            }),
            use_container_width=True, hide_index=True,
        )

# --- Tab 3: Top Products ---------------------------------------------------
with tab_products:
    st.markdown('<div class="section-title">Top 10 Products by Revenue</div>', unsafe_allow_html=True)
    top_products = analysis.get("top_products")
    if is_empty(top_products):
        empty_state("No product sales data available yet.")
    else:
        display_df = top_products[[
            "product_id", "product_name", "category", "revenue", "units_sold", "margin_pct", "profit"
        ]].rename(columns={
            "product_id": "Product ID", "product_name": "Product", "category": "Category",
            "revenue": "Revenue", "units_sold": "Units Sold", "margin_pct": "Margin %", "profit": "Profit",
        })
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Revenue": st.column_config.ProgressColumn(
                    "Revenue", format=f"{CURRENCY_SYMBOL}%.0f",
                    min_value=0, max_value=float(display_df["Revenue"].max()),
                ),
                "Margin %": st.column_config.NumberColumn("Margin %", format="%.1f%%"),
                "Profit": st.column_config.NumberColumn("Profit", format=f"{CURRENCY_SYMBOL}%.0f"),
            },
        )

    with st.expander("View bottom 10 products by revenue"):
        bottom_products = analysis.get("bottom_products")
        if is_empty(bottom_products):
            empty_state("No product sales data available yet.")
        else:
            st.dataframe(
                bottom_products[[
                    "product_id", "product_name", "category", "revenue", "units_sold", "margin_pct"
                ]].rename(columns={
                    "product_id": "Product ID", "product_name": "Product", "category": "Category",
                    "revenue": "Revenue", "units_sold": "Units Sold", "margin_pct": "Margin %",
                }),
                use_container_width=True, hide_index=True,
            )

# --- Tab 4: Growth Signals --------------------------------------------------
with tab_signals:
    st.markdown('<div class="section-title">Growth Opportunity Signals</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Deterministic, rule-based signals computed directly from the data — '
        'no AI interpretation applied yet (that begins in Phase 4+).</div>',
        unsafe_allow_html=True,
    )

    signals = analysis.get("growth_signals", {})

    def render_signal_group(title, badge, key, css_class, render_line):
        st.markdown(f"#### {title}")
        items = signals.get(key, [])
        if not items:
            empty_state(f"No {title.lower()} detected with current thresholds.")
            return
        for item in items:
            st.markdown(
                f"""
                <div class="signal-card {css_class}">
                    <div class="signal-title"><span class="badge">{badge}</span>{render_line(item)}</div>
                    <div class="signal-desc">{item['description']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    render_signal_group(
        "Seasonal Category Spikes", "SEASONAL", "seasonal_category_spikes", "seasonal",
        lambda i: f"{i['category']} — {i['peak_month']} ({i['spike_ratio']}x average)",
    )
    render_signal_group(
        "High-AOV Acquisition Channels", "CHANNEL", "high_aov_channels", "channel",
        lambda i: f"{i['channel']} — {fmt_currency(i['channel_aov'])} AOV (+{i['lift_pct']:.0f}%)",
    )
    render_signal_group(
        "High-Margin / Low-Volume Products", "MARGIN", "high_margin_low_volume_products", "margin",
        lambda i: f"{i['product_name']} — {i['margin_pct']:.0f}% margin, {i['units_sold']} units sold",
    )
    render_signal_group(
        "Churn-Risk High-Value Customers", "CHURN", "churn_risk_customers", "churn",
        lambda i: f"{i['customer_id']} — {fmt_currency(i['monetary'])} lifetime, {i['recency_days']}d silent",
    )

# --- Tab 5: AI Strategy Center (Phase 7) -----------------------------------


def _esc(value) -> str:
    """Escapes any text before it's interpolated into unsafe_allow_html
    markup. The strategy report's text ultimately comes from an LLM, so —
    unlike the fully deterministic growth-signal text elsewhere in this
    file — it should never be trusted to be safe HTML as-is."""
    if value is None:
        return ""
    return html.escape(str(value))


def render_opportunity_card(item, extra_class: str = ""):
    if not isinstance(item, dict):
        st.markdown(f"- {_esc(item)}")
        return
    title = _esc(item.get("title") or "(untitled opportunity)")
    desc = _esc(item.get("description", ""))
    tags = []
    if item.get("supporting_metric"):
        tags.append(f'<span class="tag">📊 {_esc(item["supporting_metric"])}</span>')
    if item.get("policy_reference"):
        tags.append(f'<span class="tag">📄 {_esc(item["policy_reference"])}</span>')
    if item.get("estimated_impact"):
        tags.append(f'<span class="tag">📈 {_esc(item["estimated_impact"])}</span>')
    st.markdown(
        f"""
        <div class="opportunity-card {extra_class}">
            <div class="opp-title">{title}</div>
            <div class="opp-desc">{desc}</div>
            <div class="opp-meta">{''.join(tags)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_card(item):
    if not isinstance(item, dict):
        st.markdown(f"- {_esc(item)}")
        return
    risk = _esc(item.get("risk") or "(unnamed risk)")
    desc = _esc(item.get("description", ""))
    tags = []
    if item.get("supporting_metric"):
        tags.append(f'<span class="tag">📊 {_esc(item["supporting_metric"])}</span>')
    st.markdown(
        f"""
        <div class="opportunity-card risk-card">
            <div class="opp-title">⚠️ {risk}</div>
            <div class="opp-desc">{desc}</div>
            <div class="opp-meta">{''.join(tags)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_action_card(item):
    if not isinstance(item, dict):
        st.markdown(f"- {_esc(item)}")
        return
    action = _esc(item.get("action") or "(unspecified action)")
    rationale = _esc(item.get("rationale", ""))
    priority = str(item.get("priority") or "").strip()
    priority_class = {"High": "priority-high", "Medium": "priority-medium", "Low": "priority-low"}.get(priority, "")
    priority_html = f'<span class="priority-badge {priority_class}">{_esc(priority)}</span>' if priority else ""
    tags = []
    if item.get("supporting_metric"):
        tags.append(f'<span class="tag">📊 {_esc(item["supporting_metric"])}</span>')
    if item.get("policy_reference"):
        tags.append(f'<span class="tag">📄 {_esc(item["policy_reference"])}</span>')
    if item.get("estimated_impact"):
        tags.append(f'<span class="tag">📈 {_esc(item["estimated_impact"])}</span>')
    st.markdown(
        f"""
        <div class="opportunity-card action-card">
            <div class="opp-title">✅ {action}{priority_html}</div>
            <div class="opp-desc">{rationale}</div>
            <div class="opp-meta">{''.join(tags)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_report_section(title, items, empty_message, card_renderer):
    st.markdown(f"#### {title}")
    if not items:
        empty_state(empty_message)
        return
    for item in items:
        card_renderer(item)


def render_workflow_log(log_entries):
    status_icon = {"success": "✅", "warning": "⚠️", "error": "❌", "skipped": "⏭️"}
    if not log_entries:
        empty_state("No node execution log available.")
        return
    rows = [{
        "": status_icon.get(entry.get("status"), "•"),
        "Agent": entry.get("node", "unknown"),
        "Status": str(entry.get("status", "")).capitalize(),
        "Message": entry.get("message", ""),
        "Timestamp": entry.get("timestamp", ""),
    } for entry in log_entries]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


with tab_strategy:
    st.markdown('<div class="section-title">AI Strategy Center</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Runs the multi-agent LangGraph pipeline (Analytics → Retrieval → '
        'Strategy → Validator) to turn the data above into cited, policy-compliant recommendations. '
        'Requires the Phase 4 knowledge base to be ingested and <code>GEMINI_API_KEY</code> configured.</div>',
        unsafe_allow_html=True,
    )

    button_cols = st.columns([1, 1, 4])
    with button_cols[0]:
        generate_clicked = st.button(
            "✨ Generate Strategy", use_container_width=True, type="primary", key="btn_generate_strategy",
        )
    with button_cols[1]:
        refresh_clicked = st.button(
            "🔄 Refresh Strategy", use_container_width=True,
            disabled=st.session_state.graph_state is None, key="btn_refresh_strategy",
        )

    if generate_clicked or refresh_clicked:
        execute_strategy_pipeline()

    st.markdown("<div style='height: 6px'></div>", unsafe_allow_html=True)

    # ---- Error state ----
    if st.session_state.graph_error:
        st.error(f"⚠️ {st.session_state.graph_error}")
        with st.expander("What might cause this?"):
            st.markdown(
                "- The Phase 4 knowledge base hasn't been ingested yet — run `python scripts/test_rag.py`.\n"
                "- `GEMINI_API_KEY` isn't set in your `.env` file.\n"
                "- Required packages (`langgraph`, `chromadb`, `google-genai`) aren't installed — "
                "run `pip install -r requirements.txt`.\n"
                "- No data has been loaded yet — click **Load Sample Data** in the sidebar first."
            )

    graph_state = st.session_state.graph_state

    # ---- Empty state (nothing generated yet, no error either) ----
    if graph_state is None and not st.session_state.graph_error:
        st.info("👆 Click **Generate Strategy** to run the multi-agent pipeline and produce a strategy report.")

    # ---- Report display ----
    if graph_state is not None:
        report = graph_state.get("strategy_report") or {}
        validation = graph_state.get("validation_results") or {}
        is_valid = validation.get("is_valid", False)

        if not report:
            st.warning(
                "The pipeline ran but did not produce a strategy report. "
                "See **Agent Workflow Status** below for what happened at each step."
            )
        else:
            if not is_valid:
                st.warning(
                    "⚠️ This report has validation warnings — see **Agent Workflow Status** below for details. "
                    "Content is still shown below since it may be partially useful."
                )

            st.markdown("#### Executive Summary")
            st.markdown(
                f'<div class="opportunity-card">{_esc(report.get("executive_summary") or "(no summary produced)")}</div>',
                unsafe_allow_html=True,
            )

            render_report_section(
                "Revenue Opportunities", report.get("revenue_opportunities", []),
                "No revenue opportunities identified in this report.", render_opportunity_card,
            )
            render_report_section(
                "Customer Growth Opportunities", report.get("customer_growth_opportunities", []),
                "No customer growth opportunities identified in this report.", render_opportunity_card,
            )
            render_report_section(
                "Product Opportunities", report.get("product_opportunities", []),
                "No product opportunities identified in this report.", render_opportunity_card,
            )
            render_report_section(
                "Risks", report.get("risks", []),
                "No risks flagged in this report.", render_risk_card,
            )
            render_report_section(
                "Recommended Actions", report.get("recommended_actions", []),
                "No recommended actions in this report.", render_action_card,
            )

            # ---- Policy citations used ----
            st.markdown("#### Policy Citations Used")
            citation_summary = validation.get("citation_summary", {})
            cited_sources = citation_summary.get("cited_sources", [])
            total_citations = citation_summary.get("total_citations", 0)
            if cited_sources:
                chips = "".join(f'<span class="citation-chip">📄 {_esc(s)}</span>' for s in cited_sources)
                st.markdown(chips, unsafe_allow_html=True)
                st.caption(f"{total_citations} citation(s) across the recommendations above.")
            else:
                empty_state("No policy citations were found in this report.")

        # ---- Agent Workflow Status panel ----
        with st.expander("🔍 Agent Workflow Status", expanded=(not report or not is_valid)):
            st.markdown("**Node Execution Log**")
            render_workflow_log(graph_state.get("node_execution_log", []))

            st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
            st.markdown("**Validation Status**")
            val_cols = st.columns(3)
            with val_cols[0]:
                st.metric("All Sections Present", "Yes" if validation.get("all_sections_present") else "No")
            with val_cols[1]:
                st.metric("Report Valid", "Yes" if validation.get("is_valid") else "No")
            with val_cols[2]:
                st.metric("Total Citations", validation.get("citation_summary", {}).get("total_citations", 0))

            unrecognized = validation.get("unrecognized_citations", [])
            if unrecognized:
                st.warning(f"Unrecognized/possibly hallucinated citation(s): {', '.join(unrecognized)}")

            st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
            st.markdown("**Warnings**")
            all_warnings = graph_state.get("warnings", [])
            if not all_warnings:
                st.success("No warnings — clean run.")
            else:
                for w in all_warnings:
                    st.markdown(f"- {_esc(w)}")


# --- Tab 6: Commerce Action Center (Phase 8) -------------------------------

def _gather_commerce_candidates(report: dict) -> list:
    """
    Pulls actionable items out of the EXISTING strategy_report (already
    generated by the Strategy Agent and checked by the Validator) that make
    sense as a basis for a commerce action. No new AI call — this only
    reads what's already in graph_state.

    Returns a list of (section_key, item_dict) tuples.
    """
    candidates = []
    for section_key in [
        "revenue_opportunities", "product_opportunities",
        "customer_growth_opportunities", "recommended_actions",
    ]:
        for item in report.get(section_key, []) or []:
            if isinstance(item, dict) and (item.get("title") or item.get("action")):
                candidates.append((section_key, item))
    return candidates


with tab_commerce:
    st.markdown('<div class="section-title">Commerce Action Center</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Turns one AI-generated recommendation from the Strategy Center '
        'into a real Razorpay <strong>Test Mode</strong> action — only after you explicitly approve it.</div>',
        unsafe_allow_html=True,
    )

    graph_state_for_commerce = st.session_state.graph_state
    strategy_report_for_commerce = (graph_state_for_commerce or {}).get("strategy_report") or {}
    candidates = _gather_commerce_candidates(strategy_report_for_commerce)

    if not candidates:
        st.info(
            "👈 No AI recommendations available yet. Go to **🤖 AI Strategy Center** and click "
            "**Generate Strategy** first — this tab turns one of those recommendations into a real action."
        )
    else:
        # ---- Section 1: AI Revenue Opportunity ----
        st.markdown("#### 1. AI Revenue Opportunity")
        labels = [f"[{section}] {item.get('title') or item.get('action')}" for section, item in candidates]
        selected_idx = st.selectbox(
            "Choose a recommendation to act on:",
            options=list(range(len(candidates))),
            format_func=lambda i: labels[i],
            key="commerce_selected_idx",
        )
        selected_section, selected_item = candidates[selected_idx]
        commerce_action = build_commerce_action(selected_item, source_section=selected_section)

        tags = []
        if commerce_action["supporting_metric"]:
            tags.append(f'<span class="tag">📊 {_esc(commerce_action["supporting_metric"])}</span>')
        if commerce_action["policy_reference"]:
            tags.append(f'<span class="tag">📄 {_esc(commerce_action["policy_reference"])}</span>')
        if commerce_action["estimated_impact"]:
            tags.append(f'<span class="tag">📈 Estimated: {_esc(commerce_action["estimated_impact"])}</span>')

        st.markdown(
            f"""
            <div class="opportunity-card">
                <div class="opp-title">{_esc(commerce_action['title'])}</div>
                <div class="opp-desc"><strong>Reason:</strong> {_esc(commerce_action['reason']) or '(no reason provided)'}</div>
                <div class="opp-meta">{''.join(tags)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "Expected impact above is an AI **estimate** from the Strategy Center, not a guaranteed outcome. "
            f"The actual Test Mode transaction below always uses a fixed demo amount of ₹{FIXED_TEST_AMOUNT_INR}."
        )

        # ---- Section 2: Merchant Approval Gate ----
        st.markdown("#### 2. Merchant Approval Gate")
        approve_checked = st.checkbox(
            "I approve this action", key=f"commerce_approve_{selected_idx}",
        )
        launch_clicked = st.button(
            "🚀 Launch Commerce Action", type="primary", disabled=not approve_checked,
            key=f"commerce_launch_{selected_idx}",
        )
        if not approve_checked:
            st.caption("Tick the box above to enable the launch button — no action runs automatically.")

        # ---- Section 3: Razorpay Test Mode status ----
        st.markdown("#### 3. Razorpay Test Mode")
        if is_test_mode_configured():
            st.markdown('<span class="test-mode-banner">🧪 TEST MODE ACTIVE</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="test-mode-banner">🧪 TEST MODE — CREDENTIALS NOT CONFIGURED</span>', unsafe_allow_html=True)
            st.caption(
                "Set `RAZORPAY_KEY_ID` and `RAZORPAY_KEY_SECRET` (a Test Mode key pair starting with "
                "`rzp_test_`) in your `.env` file or Streamlit secrets. Live keys are refused outright."
            )

        # ---- Handle the launch click (Section 5: failure handling lives here) ----
        if launch_clicked:
            try:
                response = create_payment_link(
                    description=commerce_action["description"],
                )
                result_record = append_audit_record({
                    "reference_id": response.get("reference_id"),
                    "recommendation": commerce_action["title"],
                    "source_section": commerce_action["source_section"],
                    "reason": commerce_action["reason"],
                    "merchant_approved": True,
                    "amount_inr": commerce_action["amount_inr"],
                    "payment_link_id": response.get("id"),
                    "payment_link_url": response.get("short_url"),
                    "result": response.get("status", "created"),
                })
                st.session_state.commerce_last_result = result_record
                st.session_state.commerce_last_error = None
            except RazorpayCredentialsError as exc:
                st.session_state.commerce_last_result = None
                st.session_state.commerce_last_error = str(exc)
                append_audit_record({
                    "reference_id": None,
                    "recommendation": commerce_action["title"],
                    "source_section": commerce_action["source_section"],
                    "reason": commerce_action["reason"],
                    "merchant_approved": True,
                    "amount_inr": commerce_action["amount_inr"],
                    "payment_link_id": None,
                    "result": "failed_credentials",
                })
            except RazorpayClientError as exc:
                st.session_state.commerce_last_result = None
                st.session_state.commerce_last_error = str(exc)
                append_audit_record({
                    "reference_id": None,
                    "recommendation": commerce_action["title"],
                    "source_section": commerce_action["source_section"],
                    "reason": commerce_action["reason"],
                    "merchant_approved": True,
                    "amount_inr": commerce_action["amount_inr"],
                    "payment_link_id": None,
                    "result": "failed_api_error",
                })
            except Exception as exc:  # noqa: BLE001 - never let this tab crash the app
                st.session_state.commerce_last_result = None
                st.session_state.commerce_last_error = f"Unexpected error creating the Payment Link: {exc}"

        if st.session_state.commerce_last_error:
            st.warning(f"⚠️ Commerce action could not be completed: {st.session_state.commerce_last_error}")
        elif st.session_state.commerce_last_result:
            last = st.session_state.commerce_last_result
            st.success(
                f"✅ Test Mode Payment Link created for **{last.get('recommendation')}** "
                f"(₹{last.get('amount_inr')}). Reference: `{last.get('reference_id')}`"
            )
            if last.get("payment_link_url"):
                st.markdown(f"[Open Test Mode Payment Link]({last['payment_link_url']})")

        # ---- Section 4: Audit Trail ----
        st.markdown("#### 4. Audit Trail")
        audit_records = load_audit_log()
        if not audit_records:
            empty_state("No commerce actions recorded yet — launch one above to populate the audit trail.")
        else:
            audit_df = pd.DataFrame(audit_records)
            display_cols = [c for c in [
                "timestamp", "recommendation", "reason", "merchant_approved",
                "amount_inr", "payment_link_id", "result", "status_checked_at",
            ] if c in audit_df.columns]
            st.dataframe(audit_df[display_cols], use_container_width=True, hide_index=True)

            # Manual, single-shot status check (no polling loop) for the most recent
            # record that still has an open Payment Link.
            checkable = [r for r in reversed(audit_records) if r.get("payment_link_id") and r.get("reference_id")]
            if checkable:
                latest_checkable = checkable[0]
                if st.button(
                    f"🔄 Check status of most recent Payment Link ({latest_checkable['payment_link_id']})",
                    key="commerce_check_status",
                ):
                    try:
                        status_response = fetch_payment_link_status(latest_checkable["payment_link_id"])
                        update_audit_record_result(
                            latest_checkable["reference_id"],
                            {"result": status_response.get("status", "unknown")},
                        )
                        st.toast(f"Status updated: {status_response.get('status', 'unknown')}", icon="✅")
                        st.rerun()
                    except (RazorpayCredentialsError, RazorpayClientError) as exc:
                        st.warning(f"⚠️ Could not check status: {exc}")
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"⚠️ Unexpected error checking status: {exc}")

        # ---- Section 5: Failure handling notes (always visible, low-noise) ----
        with st.expander("ℹ️ About failure handling in this tab"):
            st.markdown(
                "- Missing or invalid Razorpay credentials → a friendly warning above, no crash.\n"
                "- A cancelled or expired payment → reflected the next time you click **Check status**.\n"
                "- Any Razorpay API error → caught, shown as a warning, and recorded in the audit trail "
                "with a `failed_*` result so the attempt is never silently lost."
            )

