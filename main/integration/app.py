"""
Streamlit interface for exploring the unified multimodal incident database.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard_utils import apply_filters, record_to_detail_df, severity_badge
from merge_data import MASTER_CSV, export_master_tables

st.set_page_config(
    page_title="Multimodal Incident Analyzer",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(120deg, #ff6b6b, #ffa94d, #ffd43b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .sub-header {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 0.75rem 1rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }

    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-family: 'Space Grotesk', sans-serif;
    }

    .severity-high { color: #ff6b6b; font-weight: 700; }
    .severity-medium { color: #ffa94d; font-weight: 700; }
    .severity-low { color: #51cf66; font-weight: 700; }

    .block-container {
        padding-top: 1.5rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
    }

    .stDataFrame {
        border: 1px solid #334155;
        border-radius: 12px;
        overflow: hidden;
    }
</style>
"""

SEVERITY_COLORS = {"High": "#ff6b6b", "Medium": "#ffa94d", "Low": "#51cf66"}
SOURCE_COLORS = {"Audio": "#74c0fc", "PDF": "#b197fc", "Image": "#63e6be", "Text": "#ffd43b"}


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not MASTER_CSV.exists():
        export_master_tables()
    return pd.read_csv(MASTER_CSV)


def render_metrics(df: pd.DataFrame) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Incidents", len(df))
    col2.metric("High Severity", int((df["Severity"] == "High").sum()))
    col3.metric("Data Sources", df["Source"].nunique())
    col4.metric("Unique Locations", df["Location"].replace("Unknown", pd.NA).nunique())


def render_charts(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No incidents match the current filters.")
        return

    left, right = st.columns(2)

    with left:
        severity_counts = df["Severity"].value_counts().reset_index()
        severity_counts.columns = ["Severity", "Count"]
        fig_sev = px.pie(
            severity_counts,
            names="Severity",
            values="Count",
            color="Severity",
            color_discrete_map=SEVERITY_COLORS,
            hole=0.45,
            title="Severity Distribution",
        )
        fig_sev.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig_sev, width="stretch")

    with right:
        source_counts = df["Source"].value_counts().reset_index()
        source_counts.columns = ["Source", "Count"]
        fig_src = px.bar(
            source_counts,
            x="Source",
            y="Count",
            color="Source",
            color_discrete_map=SOURCE_COLORS,
            title="Incidents by Data Source",
        )
        fig_src.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            showlegend=False,
            xaxis_title="",
            yaxis_title="Count",
        )
        st.plotly_chart(fig_src, width="stretch")

    top_events = df["Event"].value_counts().head(10).reset_index()
    top_events.columns = ["Event", "Count"]
    fig_events = px.bar(
        top_events,
        x="Count",
        y="Event",
        orientation="h",
        color="Count",
        color_continuous_scale="OrRd",
        title="Top 10 Incident Event Types",
    )
    fig_events.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        showlegend=False,
        yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(fig_events, width="stretch")


def render_incident_detail(master: pd.DataFrame) -> None:
    st.subheader("Incident Lookup")

    if master.empty:
        st.warning("No incidents available for lookup.")
        return

    incident_options = master["Incident_ID"].tolist()
    selected = st.selectbox("Select Incident ID", incident_options, index=0)

    record = master[master["Incident_ID"] == selected].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Source:** {record['Source']}")
    c2.markdown(f"**Event:** {record['Event']}")
    c3.markdown(f"**Severity:** {severity_badge(record['Severity'])}", unsafe_allow_html=True)

    c4, c5 = st.columns(2)
    c4.markdown(f"**Location:** {record['Location']}")
    c5.markdown(f"**Time:** {record['Time']}")

    st.markdown("**Summary Detail**")
    st.info(record.get("Detail", "No additional detail available."))

    with st.expander("Full incident record"):
        detail_df = record_to_detail_df(record)
        st.dataframe(detail_df, width="stretch", hide_index=True)


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown('<p class="main-header">Multimodal Incident Analyzer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Unified dashboard for audio, document, image, and text incident intelligence</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Filters & Controls")
    if st.button("Regenerate Master Dataset", width="stretch"):
        export_master_tables()
        st.cache_data.clear()
        st.success("Master dataset refreshed.")
        st.rerun()

    master_df = load_data()

    selected_sources = st.multiselect(
        "Source",
        sorted(master_df["Source"].unique()),
        default=sorted(master_df["Source"].unique()),
    )
    selected_severities = st.multiselect(
        "Severity",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
    )
    event_query = st.text_input("Event contains")
    location_query = st.text_input("Location contains")
    free_text = st.text_input("Search all fields")

    st.divider()
    st.caption("Pipeline outputs")
    st.code(str(MASTER_CSV.name), language=None)
    st.caption(f"{len(master_df)} records loaded")

filtered_df = apply_filters(
    master_df,
    selected_sources,
    selected_severities,
    event_query,
    location_query,
    free_text,
)

tab_overview, tab_table, tab_lookup = st.tabs(["Overview", "Incident Table", "Incident Lookup"])

with tab_overview:
    render_metrics(filtered_df)
    render_charts(filtered_df)

with tab_table:
    st.subheader("Filtered Incident Summaries")
    display_df = filtered_df[
        ["Incident_ID", "Source", "Event", "Location", "Time", "Severity", "Detail"]
    ].copy()
    st.dataframe(display_df, width="stretch", height=520, hide_index=True)

    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered results (CSV)",
        data=csv_bytes,
        file_name="filtered_incidents.csv",
        mime="text/csv",
        width="stretch",
    )

with tab_lookup:
    lookup_df = filtered_df if len(filtered_df) else master_df
    render_incident_detail(lookup_df)
