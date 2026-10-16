import subprocess
from html import escape
from io import StringIO
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

HDFS_BASE = "/data/raw/old-london-dataset-review1"

PROCESSED_PATH = (
    f"{HDFS_BASE}/processed/energy_daily/part-*.csv"
)

PREDICTION_PATH = (
    f"{HDFS_BASE}/processed/predictions/part-*.csv"
)

ANOMALY_PATH = (
    f"{HDFS_BASE}/processed/anomalies/part-*.csv"
)

HOUSE_DAILY_PATH = (
    f"{HDFS_BASE}/processed/house_daily/part-*.csv"
)

HOUSE_PREDICTION_PATH = (
    f"{HDFS_BASE}/processed/house_predictions/part-*.csv"
)

RAW_PATH = f"{HDFS_BASE}/daily_dataset.csv"


# ============================================================
# COLORS
# ============================================================

INK = "#16212B"
MUTED = "#5B6B7A"
GRID = "#E7ECF0"
PRIMARY = "#0E5A6B"
AMBER = "#E8A200"
GREEN = "#1F8A5B"
RED = "#C0392B"
PURPLE = "#6C5CE7"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="London Smart Meter Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

CSS = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root{
    --bg:#F4F6F8;
    --surface:#FFFFFF;
    --ink:#16212B;
    --muted:#5B6B7A;
    --line:#E1E6EB;
    --primary:#0E5A6B;
    --primary-dark:#0B3F4C;
    --amber:#E8A200;
}

html,body,.stApp,[class*="css"]{
    font-family:'IBM Plex Sans','Segoe UI',system-ui,sans-serif;
}

.stApp{
    background:var(--bg);
    color:var(--ink);
}

.block-container{
    padding-top:1.4rem;
    padding-bottom:3rem;
    max-width:1320px;
}

h1,h2,h3,h4{
    font-family:'Sora','IBM Plex Sans',sans-serif;
    color:var(--ink);
    letter-spacing:-0.01em;
}

h2{
    font-size:24px!important;
}

h3{
    font-size:17px!important;
    margin-top:1.2rem;
}

#MainMenu,footer{
    visibility:hidden;
}

[data-testid="stSidebar"]{
    background:var(--surface);
    border-right:1px solid var(--line);
}

[data-testid="stSidebar"] :is(h1,h2,h3,p,label,span,li){
    color:var(--ink)!important;
}

header[data-testid="stHeader"]{
    background:var(--bg);
}

.stApp :is(label,[data-testid="stWidgetLabel"] p){
    color:var(--ink)!important;
}

.stApp [data-testid="stCaptionContainer"]{
    color:var(--muted)!important;
}

.brand{
    display:flex;
    align-items:center;
    gap:10px;
    margin-bottom:8px;
}

.brand-mark{
    width:36px;
    height:36px;
    border-radius:9px;
    background:var(--primary);
    color:#fff;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:19px;
}

.brand-name{
    font-family:'Sora',sans-serif;
    font-weight:700;
    font-size:16px;
    line-height:1.1;
}

.brand-sub{
    color:var(--muted)!important;
    font-size:12px;
}

[data-testid="stSidebar"] [role="radiogroup"] label{
    padding:8px 12px;
    border-radius:8px;
    width:100%;
    transition:background .15s;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover{
    background:#EEF3F6;
}

[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){
    background:#E3F0F3;
    font-weight:600;
}

.hero{
    background:linear-gradient(120deg,#0B3F4C 0%,#0E5A6B 100%);
    border-radius:14px;
    padding:26px 32px;
    margin-bottom:20px;
    border-left:6px solid var(--amber);
}

.hero h1{
    color:#fff!important;
    font-size:29px;
    margin:0 0 6px;
    font-weight:700;
}

.hero p{
    color:#C9DCE2!important;
    font-size:15px;
    margin:0 0 14px;
    max-width:760px;
}

.chip{
    display:inline-block;
    padding:4px 11px;
    margin:0 6px 6px 0;
    border-radius:999px;
    font-size:12.5px;
    font-weight:500;
    background:rgba(255,255,255,.1);
    color:#E8F1F4!important;
    border:1px solid rgba(255,255,255,.18);
}

[data-testid="stMetric"]{
    background:var(--surface);
    border:1px solid var(--line);
    border-top:3px solid var(--primary);
    border-radius:12px;
    padding:14px 18px;
}

[data-testid="stMetricLabel"] p{
    color:var(--muted)!important;
    font-weight:500;
    font-size:13px;
}

[data-testid="stMetricValue"]{
    font-family:'Sora',sans-serif;
    font-weight:700;
    color:var(--ink);
}

.flow{
    display:flex;
    flex-wrap:wrap;
    align-items:center;
    gap:8px;
    margin:6px 0 12px;
}

.step{
    background:var(--surface);
    border:1px solid var(--line);
    border-radius:8px;
    padding:9px 13px;
    font-size:13.5px;
    font-weight:500;
}

.step.first{
    background:#E3F0F3;
    border-color:#BCD8DF;
}

.step.last{
    background:var(--primary);
    border-color:var(--primary);
    color:#fff;
}

.arrow{
    color:var(--muted);
    font-size:18px;
}

.ops{
    display:grid;
    grid-template-columns:repeat(auto-fill,minmax(330px,1fr));
    gap:8px;
    margin:6px 0 14px;
}

.op{
    display:flex;
    gap:12px;
    align-items:center;
    background:var(--surface);
    border:1px solid var(--line);
    border-radius:10px;
    padding:10px 14px;
    font-size:14px;
}

.op-n{
    min-width:26px;
    height:26px;
    border-radius:50%;
    background:var(--primary);
    color:#fff!important;
    font-size:12.5px;
    font-weight:600;
    display:flex;
    align-items:center;
    justify-content:center;
}

.insight{
    background:#EAF4F6;
    border:1px solid #C5DEE4;
    border-left:4px solid var(--primary);
    border-radius:10px;
    padding:12px 16px;
    font-size:14px;
    margin:10px 0;
}

.badge{
    display:inline-block;
    padding:3px 10px;
    border-radius:999px;
    font-size:12px;
    font-weight:600;
}

.badge.ok{
    background:#DDF3E8;
    color:#14653F;
}

.badge.bad{
    background:#FBE0DC;
    color:#8E2418;
}

.stTabs [data-baseweb="tab-list"]{
    gap:6px;
    border-bottom:1px solid var(--line);
}

.stTabs [data-baseweb="tab"]{
    padding:8px 14px;
    font-weight:500;
}

.stTabs [aria-selected="true"]{
    color:var(--primary)!important;
    font-weight:600;
}

.stTabs [data-baseweb="tab-highlight"]{
    background-color:var(--amber);
}

[data-testid="stPlotlyChart"]{
    background:#fff;
    border:1px solid var(--line);
    border-radius:12px;
    padding:6px;
}

[data-testid="stDataFrame"]{
    border:1px solid var(--line);
    border-radius:10px;
    overflow:hidden;
}

.stButton>button,
.stDownloadButton>button{
    border-radius:8px;
    border:1px solid var(--primary);
    background:var(--surface);
    color:var(--primary);
    font-weight:500;
}

.stButton>button:hover,
.stDownloadButton>button:hover{
    background:var(--primary);
    color:#fff;
}

</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================

def render(markup, sidebar=False):
    cleaned = "\n".join(
        line.strip()
        for line in markup.splitlines()
        if line.strip()
    )

    target = st.sidebar if sidebar else st

    target.markdown(
        cleaned,
        unsafe_allow_html=True
    )


def flow(steps):
    parts = []

    for i, name in enumerate(steps):
        css = "step"

        if i == 0:
            css += " first"

        if i == len(steps) - 1:
            css += " last"

        parts.append(
            f'<div class="{css}">{escape(name)}</div>'
        )

    render(
        '<div class="flow">' +
        '<span class="arrow">›</span>'.join(parts) +
        "</div>"
    )


def style(
    fig,
    title,
    height=440,
    xtitle=None,
    ytitle=None,
    legend=True
):
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.01,
            font=dict(
                size=16,
                color=INK
            )
        ),
        height=height,
        margin=dict(
            l=10,
            r=10,
            t=60,
            b=10
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(
            family="IBM Plex Sans, sans-serif",
            color=MUTED
        ),
        hovermode="x unified",
        showlegend=legend,
        legend=dict(
            orientation="h",
            y=1.08,
            x=1,
            xanchor="right"
        )
    )

    fig.update_xaxes(
        title=xtitle,
        gridcolor=GRID,
        showline=True,
        linecolor="#CBD3DA"
    )

    fig.update_yaxes(
        title=ytitle,
        gridcolor=GRID,
        zeroline=False
    )

    return fig


def plot(
    fig,
    key,
    filename="chart"
):
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
        config={
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": filename,
                "scale": 3
            }
        }
    )


def table(
    df,
    filename,
    key,
    **kw
):
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        **kw
    )

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode(),
        f"{filename}.csv",
        "text/csv",
        key=key
    )


def insight(text):
    render(
        f'<div class="insight">💡 {text}</div>'
    )


def time_controls(fig):
    fig.update_xaxes(
        rangeslider=dict(
            visible=True,
            thickness=0.06
        ),
        rangeselector=dict(
            buttons=[
                dict(
                    count=1,
                    label="1M",
                    step="month",
                    stepmode="backward"
                ),
                dict(
                    count=3,
                    label="3M",
                    step="month",
                    stepmode="backward"
                ),
                dict(
                    count=6,
                    label="6M",
                    step="month",
                    stepmode="backward"
                ),
                dict(
                    count=1,
                    label="1Y",
                    step="year",
                    stepmode="backward"
                ),
                dict(
                    step="all",
                    label="All"
                )
            ]
        )
    )

    return fig


# ============================================================
# HDFS HELPERS
# ============================================================

def _hdfs(command):
    return subprocess.run(
        [
            "docker",
            "exec",
            "namenode",
            "bash",
            "-c",
            command
        ],
        capture_output=True,
        text=True
    )


def read_hdfs_csv(path):
    result = _hdfs(
        f"hdfs dfs -cat '{path}'"
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr
        )

    return pd.read_csv(
        StringIO(result.stdout)
    )


def hdfs_exists(path):
    return (
        _hdfs(
            f"hdfs dfs -test -e '{path}'"
        ).returncode == 0
    )


# ============================================================
# LOAD AGGREGATE DATA
# ============================================================

@st.cache_data(
    show_spinner="Reading aggregate data from HDFS..."
)
def load_hdfs():

    energy = read_hdfs_csv(
        PROCESSED_PATH
    )

    predictions = read_hdfs_csv(
        PREDICTION_PATH
    )

    anomalies = read_hdfs_csv(
        ANOMALY_PATH
    )

    energy["day"] = pd.to_datetime(
        energy["day"]
    )

    predictions["day"] = pd.to_datetime(
        predictions["day"]
    )

    anomalies["day"] = pd.to_datetime(
        anomalies["day"]
    )

    return (
        energy,
        predictions,
        anomalies
    )


# ============================================================
# HOUSE LIST
# ============================================================

@st.cache_data(
    show_spinner="Reading household list from HDFS..."
)
def load_house_list():

    result = _hdfs(
        f"hdfs dfs -cat '{HOUSE_DAILY_PATH}' "
        f"| cut -d',' -f1 "
        f"| tail -n +2 "
        f"| grep -v '^LCLid$' "
        f"| sort -u"
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr
        )

    houses = [
        x.strip()
        for x in result.stdout.splitlines()
        if x.strip()
    ]

    return houses


# ============================================================
# LOAD ONE HOUSE ONLY
# ============================================================

@st.cache_data(
    show_spinner="Loading selected house..."
)
def load_house(house_id):

    safe_house = house_id.replace(
        "'",
        "'\\''"
    )

    command = (
        f"hdfs dfs -cat '{HOUSE_DAILY_PATH}' "
        f"| awk -F',' -v h='{safe_house}' "
        f"'NR==1 || $1==h'"
    )

    result = _hdfs(
        command
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr
        )

    df = pd.read_csv(
        StringIO(result.stdout)
    )

    df["day"] = pd.to_datetime(
        df["day"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["day"]
    )

    return df


# ============================================================
# HOUSE PREDICTIONS
# ============================================================

@st.cache_data(show_spinner="Reading house predictions from HDFS...")
def load_house_predictions():
    df = read_hdfs_csv(HOUSE_PREDICTION_PATH)

    # Convert date column
    df["day"] = pd.to_datetime(df["day"], errors="coerce")

    # Convert prediction values to numeric.
    # Repeated CSV header rows become NaN and are removed below.
    df["actual_energy"] = pd.to_numeric(
        df["actual_energy"],
        errors="coerce"
    )

    df["prediction"] = pd.to_numeric(
        df["prediction"],
        errors="coerce"
    )

    # Remove repeated headers / invalid rows
    df = df.dropna(
        subset=["day", "actual_energy", "prediction"]
    )

    return df

# ============================================================
# SIDEBAR
# ============================================================

render(
    """
    <div class="brand">
        <div class="brand-mark">⚡</div>
        <div>
            <div class="brand-name">Smart Meter</div>
            <div class="brand-sub">
                London household analytics
            </div>
        </div>
    </div>
    """,
    sidebar=True
)


page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Preprocessing",
        "Exploratory Analysis",
        "Correlation",
        "Regression and ML",
        "House-Level Analysis",
        "House-Level Predictions",
        "Anomaly Detection",
        "Daily Prediction",
        "Big Data and HDFS"
    ]
)


st.sidebar.divider()


if st.sidebar.button(
    "Reload from HDFS",
    use_container_width=True
):
    st.cache_data.clear()
    st.rerun()


# ============================================================
# LOAD MAIN DATA
# ============================================================

try:

    energy, predictions, anomalies = load_hdfs()

except Exception as e:

    st.error(
        "Unable to read the HDFS data. "
        "Make sure the Docker Hadoop cluster is running."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# GLOBAL DATE FILTER
# ============================================================

dmin = energy["day"].min().date()
dmax = energy["day"].max().date()


st.sidebar.subheader(
    "Global filters"
)


rng_sel = st.sidebar.date_input(
    "Date range",
    (dmin, dmax),
    min_value=dmin,
    max_value=dmax
)


if (
    isinstance(rng_sel, tuple)
    and len(rng_sel) == 2
):

    lo = pd.Timestamp(
        rng_sel[0]
    )

    hi = pd.Timestamp(
        rng_sel[1]
    )

else:

    lo = pd.Timestamp(dmin)
    hi = pd.Timestamp(dmax)


energy_all = energy.copy()
predictions_all = predictions.copy()
anomalies_all = anomalies.copy()


energy = energy[
    energy["day"].between(
        lo,
        hi
    )
]

predictions = predictions[
    predictions["day"].between(
        lo,
        hi
    )
]

anomalies = anomalies[
    anomalies["day"].between(
        lo,
        hi
    )
]


if energy.empty:

    st.warning(
        "No data exists in the selected date range."
    )

    st.stop()


st.sidebar.caption(
    f"{len(energy):,} days selected"
)


# ============================================================
# HERO
# ============================================================

render(
    """
    <div class="hero">
        <h1>⚡ London Smart Meter Analytics</h1>

        <p>
            Big Data Analytics using Hadoop HDFS,
            Apache Spark, Scala and Spark MLlib
        </p>

        <span class="chip">Hadoop HDFS</span>
        <span class="chip">Apache Spark</span>
        <span class="chip">Scala</span>
        <span class="chip">Spark SQL</span>
        <span class="chip">Spark MLlib</span>
        <span class="chip">Random Forest</span>
        <span class="chip">K-Means</span>
        <span class="chip">Streamlit</span>
    </div>
    """
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.header(
        "Project Overview"
    )

    c = st.columns(4)

    c[0].metric(
        "Processed Days",
        f"{len(energy):,}"
    )

    c[1].metric(
        "Households",
        "5,561"
    )

    c[2].metric(
        "House-Day Records",
        "3,510,403"
    )

    c[3].metric(
        "Anomalies",
        "50"
    )

    st.divider()

    st.subheader(
        "Machine Learning Results"
    )

    c = st.columns(4)

    c[0].metric(
        "Aggregate RMSE",
        "1.5435"
    )

    c[1].metric(
        "Aggregate R²",
        "-0.0048"
    )

    c[2].metric(
        "House RMSE",
        "4.4994"
    )

    c[3].metric(
        "House R²",
        "0.8037"
    )

    st.divider()

    st.subheader(
        "Project Pipeline"
    )

    flow(
        [
            "London Smart Meter Dataset",
            "Hadoop HDFS",
            "Apache Spark + Scala",
            "Cleaning + Deduplication",
            "Aggregation + Feature Engineering",
            "Processed Dataset in HDFS",
            "Spark SQL + MLlib",
            "Aggregate + House Models",
            "Predictions + Anomalies",
            "Streamlit Dashboard"
        ]
    )

    st.subheader(
        "Daily Energy Trend"
    )

    metric = st.radio(
        "Metric",
        [
            "total_energy",
            "avg_energy",
            "household_count"
        ],
        horizontal=True,
        format_func=lambda s:
            s.replace(
                "_",
                " "
            ).title()
    )

    smooth = st.slider(
        "Moving average window (days)",
        1,
        60,
        7
    )

    fig = go.Figure()

    fig.add_scatter(
        x=energy["day"],
        y=energy[metric],
        name="Daily",
        mode="lines",
        line=dict(
            color=PRIMARY,
            width=1.2
        ),
        fill="tozeroy",
        fillcolor="rgba(14,90,107,.12)"
    )

    if smooth > 1:

        fig.add_scatter(
            x=energy["day"],
            y=energy[metric].rolling(
                smooth,
                min_periods=1
            ).mean(),
            name=f"{smooth}-day average",
            line=dict(
                color=AMBER,
                width=2.4
            )
        )

    peak_row = energy.iloc[
        energy[metric].to_numpy().argmax()
    ]

    fig.add_scatter(
        x=[peak_row["day"]],
        y=[peak_row[metric]],
        mode="markers+text",
        name="Peak day",
        marker=dict(
            color=RED,
            size=11
        ),
        text=["Peak"],
        textposition="top center"
    )

    style(
        fig,
        f"Daily {metric.replace('_', ' ').title()}",
        480
    )

    plot(
        time_controls(fig),
        "overview_trend",
        "daily_energy_trend"
    )

    insight(
        f"Peak {metric.replace('_', ' ')} occurred on "
        f"<b>{peak_row['day']:%d %b %Y}</b> "
        f"({peak_row[metric]:,.2f})."
    )


# ============================================================
# PREPROCESSING
# ============================================================

elif page == "Preprocessing":

    st.header(
        "Data Preprocessing"
    )

    st.write(
        "Raw dataset: London Smart Meter Dataset"
    )

    st.write(
        f"Raw HDFS path: `{RAW_PATH}`"
    )

    ops = [
        "Read raw CSV data from HDFS.",
        "Select required columns.",
        "Convert date and energy fields.",
        "Remove invalid and null records.",
        "Remove duplicate records.",
        "Aggregate energy consumption by day.",
        "Calculate household count.",
        "Calculate average household energy.",
        "Generate temporal features.",
        "Store processed dataset back in HDFS."
    ]

    items = ""

    for i, operation in enumerate(
        ops,
        1
    ):

        items += (
            '<div class="op">'
            f'<div class="op-n">{i}</div>'
            f'<div>{escape(operation)}</div>'
            '</div>'
        )

    st.subheader(
        "Preprocessing Operations"
    )

    render(
        f'<div class="ops">{items}</div>'
    )

    raw_n = 3510433
    clean_n = 3510403

    st.subheader(
        "Data Reduction"
    )

    c = st.columns(4)

    c[0].metric(
        "Raw Records",
        f"{raw_n:,}"
    )

    c[1].metric(
        "After Cleaning",
        f"{clean_n:,}",
        f"-{raw_n - clean_n:,}",
        delta_color="off"
    )

    c[2].metric(
        "Final Daily Rows",
        "829"
    )

    c[3].metric(
        "Records Retained",
        f"{clean_n / raw_n:.4%}"
    )

    st.subheader(
        "Processed Schema"
    )

    schema = pd.DataFrame(
        {
            "Column": [
                "day",
                "total_energy",
                "household_count",
                "avg_energy",
                "year",
                "month",
                "day_of_month",
                "day_of_week",
                "is_weekend"
            ],
            "Type": [
                "date",
                "double",
                "long",
                "double",
                "integer",
                "integer",
                "integer",
                "integer",
                "integer"
            ]
        }
    )

    st.dataframe(
        schema,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Processed Data Explorer"
    )

    table(
        energy.head(500),
        "processed_energy_daily",
        "download_processed"
    )


# ============================================================
# EXPLORATORY ANALYSIS
# ============================================================

elif page == "Exploratory Analysis":

    st.header(
        "Exploratory Data Analysis"
    )

    t1, t2, t3, t4, t5 = st.tabs(
        [
            "Trends",
            "Seasonality",
            "Weekday vs Weekend",
            "Day-of-Week Heatmap",
            "Top Days"
        ]
    )

    # --------------------------------------------------------
    # TRENDS
    # --------------------------------------------------------

    with t1:

        fig = go.Figure(
            go.Scatter(
                x=energy["day"],
                y=energy["total_energy"],
                line=dict(
                    color=PRIMARY,
                    width=1.6
                ),
                fill="tozeroy",
                fillcolor="rgba(14,90,107,.12)",
                name="Total energy"
            )
        )

        plot(
            time_controls(
                style(
                    fig,
                    "Total Energy Consumption",
                    460,
                    "Date",
                    "Total energy",
                    False
                )
            ),
            "eda_trend",
            "total_energy_trend"
        )

    # --------------------------------------------------------
    # SEASONALITY
    # --------------------------------------------------------

    with t2:

        m = (
            energy
            .groupby(
                ["year", "month"]
            )["avg_energy"]
            .mean()
            .reset_index()
        )

        m["period"] = pd.to_datetime(
            m["year"].astype(str)
            + "-"
            + m["month"].astype(str).str.zfill(2)
            + "-01"
        )

        mode = st.radio(
            "View",
            [
                "Timeline",
                "Year-over-year"
            ],
            horizontal=True
        )

        if mode == "Timeline":

            fig = go.Figure(
                go.Scatter(
                    x=m["period"],
                    y=m["avg_energy"],
                    mode="lines+markers",
                    line=dict(
                        color=AMBER,
                        width=2.6
                    ),
                    marker=dict(
                        size=8,
                        color="white",
                        line=dict(
                            color=AMBER,
                            width=2
                        )
                    ),
                    fill="tozeroy",
                    fillcolor="rgba(232,162,0,.14)"
                )
            )

            style(
                fig,
                "Monthly Average Household Energy",
                440,
                "Month",
                "Average household energy",
                False
            )

        else:

            fig = px.line(
                m,
                x="month",
                y="avg_energy",
                color=m["year"].astype(str),
                markers=True
            )

            style(
                fig,
                "Seasonality by Year",
                440,
                "Month",
                "Average household energy"
            )

            fig.update_xaxes(
                dtick=1
            )

        plot(
            fig,
            "eda_season",
            "monthly_average_energy"
        )

    # --------------------------------------------------------
    # WEEKDAY VS WEEKEND
    # --------------------------------------------------------

    with t3:

        comp = (
            energy
            .groupby("is_weekend")
            .agg(
                average_total_energy=(
                    "total_energy",
                    "mean"
                ),
                average_household_energy=(
                    "avg_energy",
                    "mean"
                ),
                days=(
                    "day",
                    "count"
                )
            )
            .reset_index()
        )

        comp["day_type"] = comp[
            "is_weekend"
        ].map(
            {
                0: "Weekday",
                1: "Weekend"
            }
        )

        st.dataframe(
            comp[
                [
                    "day_type",
                    "average_total_energy",
                    "average_household_energy",
                    "days"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        fig = go.Figure(
            go.Bar(
                x=comp["day_type"],
                y=comp["average_total_energy"],
                marker_color=[
                    PRIMARY,
                    AMBER
                ][:len(comp)],
                text=comp[
                    "average_total_energy"
                ].round(2),
                textposition="outside"
            )
        )

        plot(
            style(
                fig,
                "Average Total Energy",
                380,
                legend=False
            ),
            "weekday_total",
            "weekday_total"
        )

        fig = go.Figure(
            go.Bar(
                x=comp["day_type"],
                y=comp["average_household_energy"],
                marker_color=[
                    PRIMARY,
                    AMBER
                ][:len(comp)],
                text=comp[
                    "average_household_energy"
                ].round(2),
                textposition="outside"
            )
        )

        plot(
            style(
                fig,
                "Average Household Energy",
                380,
                legend=False
            ),
            "weekday_household",
            "weekday_household"
        )

    # --------------------------------------------------------
    # DAY-OF-WEEK HEATMAP
    # --------------------------------------------------------

    with t4:

        h = energy.copy()

        h["week"] = (
            h["day"]
            .dt.to_period("W")
            .dt.start_time
        )

        pv = h.pivot_table(
            index="day_of_week",
            columns="week",
            values="avg_energy",
            aggfunc="mean"
        )

        fig = go.Figure(
            go.Heatmap(
                z=pv.values,
                x=pv.columns,
                y=[
                    "Mon",
                    "Tue",
                    "Wed",
                    "Thu",
                    "Fri",
                    "Sat",
                    "Sun"
                ][:len(pv)],
                colorscale="YlOrRd",
                colorbar=dict(
                    title="Avg Energy"
                )
            )
        )

        plot(
            style(
                fig,
                "Average Household Energy — Week × Day-of-Week",
                380
            ),
            "eda_heat",
            "dow_heatmap"
        )

    # --------------------------------------------------------
    # TOP DAYS
    # --------------------------------------------------------

    with t5:

        n = st.slider(
            "Number of days",
            5,
            30,
            10
        )

        top = (
            energy
            .sort_values(
                "total_energy",
                ascending=False
            )
            .head(n)
        )

        table(
            top[
                [
                    "day",
                    "total_energy",
                    "household_count",
                    "avg_energy"
                ]
            ],
            "top_days",
            "download_top_days"
        )


# ============================================================
# CORRELATION
# ============================================================

elif page == "Correlation":

    st.header(
        "Correlation Analysis"
    )

    cols = [
        "total_energy",
        "household_count",
        "avg_energy",
        "month",
        "day_of_month",
        "day_of_week",
        "is_weekend"
    ]

    selected = st.multiselect(
        "Variables",
        cols,
        default=cols
    )

    if len(selected) < 2:

        st.info(
            "Select at least two variables."
        )

        st.stop()

    corr = energy[
        selected
    ].corr()

    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=selected,
            y=selected,
            zmin=-1,
            zmax=1,
            colorscale="RdBu_r",
            text=corr.round(2).values,
            texttemplate="%{text}",
            colorbar=dict(
                title="Pearson"
            )
        )
    )

    fig.update_yaxes(
        autorange="reversed"
    )

    plot(
        style(
            fig,
            "Correlation Heatmap",
            520
        ),
        "corr_heat",
        "correlation_heatmap"
    )

    st.subheader(
        "Key Correlation"
    )

    correlation_value = energy[
        "total_energy"
    ].corr(
        energy[
            "household_count"
        ]
    )

    st.metric(
        "Total Energy ↔ Household Count",
        f"{correlation_value:.4f}"
    )

    insight(
        "The total energy and household count have a "
        f"<b>{correlation_value:.4f}</b> Pearson correlation."
    )


# ============================================================
# AGGREGATE REGRESSION
# ============================================================

elif page == "Regression and ML":

    st.header(
        "Spark MLlib — Aggregate Random Forest Regression"
    )

    st.markdown(
        """
        **Target:** `avg_energy`  
        **Algorithm:** `RandomForestRegressor`  
        **Training:** 2011–2013  
        **Testing:** 2014
        """
    )

    if predictions.empty:

        st.warning(
            "No aggregate predictions in the selected range."
        )

        st.stop()

    actual = predictions[
        "avg_energy"
    ]

    predicted = predictions[
        "prediction"
    ]

    rmse = (
        (
            actual - predicted
        ) ** 2
    ).mean() ** 0.5

    mae = (
        actual - predicted
    ).abs().mean()

    denominator = (
        (actual - actual.mean()) ** 2
    ).sum()

    if denominator == 0:

        r2 = np.nan

    else:

        r2 = (
            1
            -
            (
                (actual - predicted) ** 2
            ).sum()
            /
            denominator
        )

    c = st.columns(3)

    c[0].metric(
        "RMSE",
        f"{rmse:.4f}"
    )

    c[1].metric(
        "MAE",
        f"{mae:.4f}"
    )

    c[2].metric(
        "R²",
        f"{r2:.4f}"
    )

    st.subheader(
        "Actual vs Predicted"
    )

    fig = go.Figure()

    fig.add_scatter(
        x=predictions["day"],
        y=actual,
        name="Actual",
        line=dict(
            color=PRIMARY,
            width=2.2
        )
    )

    fig.add_scatter(
        x=predictions["day"],
        y=predicted,
        name="Predicted",
        line=dict(
            color=AMBER,
            width=2.2
        )
    )

    plot(
        time_controls(
            style(
                fig,
                "Aggregate Actual vs Predicted",
                480,
                "Date",
                "Average household energy"
            )
        ),
        "aggregate_prediction",
        "aggregate_prediction"
    )

    st.subheader(
        "Aggregate Feature Importance"
    )

    fi = pd.DataFrame(
        {
            "Feature": [
                "household_count",
                "month",
                "day_of_month",
                "day_of_week",
                "is_weekend"
            ],
            "Importance": [
                0.3565332221,
                0.5935017142,
                0.0210707986,
                0.0184758557,
                0.0104184094
            ]
        }
    )

    fi = fi.sort_values(
        "Importance"
    )

    fig = go.Figure(
        go.Bar(
            y=fi["Feature"],
            x=fi["Importance"],
            orientation="h",
            marker_color=PRIMARY,
            text=fi["Importance"].round(3),
            textposition="outside"
        )
    )

    plot(
        style(
            fig,
            "Aggregate Random Forest Feature Importance",
            380,
            "Importance",
            legend=False
        ),
        "aggregate_feature_importance",
        "aggregate_feature_importance"
    )

    st.info(
        "Aggregate model result: RMSE = 1.543546 and "
        "R² = -0.004833."
    )


# ============================================================
# HOUSE-LEVEL ANALYSIS
# ============================================================

elif page == "House-Level Analysis":

    st.header(
        "House-Level Energy Analysis"
    )

    st.write(
        "Select an individual household to inspect its "
        "complete available timeline and a specific day's consumption."
    )

    try:

        houses = load_house_list()

    except Exception as e:

        st.error(
            "Unable to read the household list from HDFS."
        )

        st.code(
            str(e)
        )

        st.stop()

    if not houses:

        st.warning(
            "No households were found."
        )

        st.stop()

    st.caption(
        f"{len(houses):,} unique houses available"
    )

    selected_house = st.selectbox(
        "Select Household (LCLid)",
        houses
    )

    house_data = load_house(
        selected_house
    )

    if house_data.empty:

        st.warning(
            "No records found for this household."
        )

        st.stop()

    # --------------------------------------------------------
    # HOUSE KPIs
    # --------------------------------------------------------

    c = st.columns(5)

    c[0].metric(
        "House",
        selected_house
    )

    c[1].metric(
        "Days Available",
        f"{len(house_data):,}"
    )

    c[2].metric(
        "Average Energy",
        f"{house_data['energy'].mean():.3f}"
    )

    c[3].metric(
        "Minimum Energy",
        f"{house_data['energy'].min():.3f}"
    )

    c[4].metric(
        "Maximum Energy",
        f"{house_data['energy'].max():.3f}"
    )

    st.subheader(
        "House Timeline Statistics"
    )

    stats = house_data[
        "energy"
    ].describe()

    stats_df = pd.DataFrame(
        {
            "Statistic": stats.index,
            "Value": stats.values
        }
    )

    st.dataframe(
        stats_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Complete House Consumption Timeline"
    )

    timeline = house_data.sort_values(
        "day"
    )

    fig = go.Figure()

    fig.add_scatter(
        x=timeline["day"],
        y=timeline["energy"],
        mode="lines",
        name="Energy",
        line=dict(
            color=PRIMARY,
            width=1.6
        )
    )

    plot(
        time_controls(
            style(
                fig,
                f"Energy Consumption — {selected_house}",
                460,
                "Date",
                "Energy"
            )
        ),
        "house_timeline",
        f"{selected_house}_timeline"
    )

    # --------------------------------------------------------
    # SPECIFIC DAY
    # --------------------------------------------------------

    st.subheader(
        "Specific House + Specific Day"
    )

    min_date = house_data[
        "day"
    ].min().date()

    max_date = house_data[
        "day"
    ].max().date()

    selected_date = st.date_input(
        "Select Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date,
        key="house_selected_date"
    )

    selected = house_data[
        house_data["day"].dt.date
        == selected_date
    ]

    if not selected.empty:

        value = float(
            selected.iloc[0]["energy"]
        )

        st.metric(
            f"{selected_house} consumption on {selected_date}",
            f"{value:.4f}"
        )

    else:

        st.warning(
            "No consumption record is available "
            "for this house on the selected date."
        )

    # --------------------------------------------------------
    # HOUSE DATA
    # --------------------------------------------------------

    st.subheader(
        "House Data"
    )

    table(
        house_data,
        f"{selected_house}_daily",
        "download_house_data"
    )


# ============================================================
# HOUSE-LEVEL PREDICTIONS
# ============================================================

elif page == "House-Level Predictions":

    st.header(
        "House-Level Random Forest Predictions"
    )

    st.markdown(
        """
        **Target:** `energy`  
        **Training:** 2011–2013  
        **Testing:** 2014  
        **Model:** Random Forest Regressor  
        **Features:** month, day_of_month, day_of_week,
        is_weekend, lag_1_energy, rolling_7_energy
        """
    )

    try:

        house_predictions = (
            load_house_predictions()
        )

    except Exception as e:

        st.error(
            "Unable to read house predictions from HDFS."
        )

        st.code(
            str(e)
        )

        st.stop()

    houses = sorted(
        house_predictions[
            "LCLid"
        ].dropna().unique()
    )

    if not houses:

        st.warning(
            "No house predictions were found."
        )

        st.stop()

    selected_house = st.selectbox(
        "Select Household (LCLid)",
        houses,
        key="prediction_house"
    )

    hp = house_predictions[
        house_predictions["LCLid"]
        == selected_house
    ].copy()

    hp = hp.sort_values(
        "day"
    )

    if hp.empty:

        st.warning(
            "No predictions available for this house."
        )

        st.stop()

    # --------------------------------------------------------
    # HOUSE MODEL METRICS
    # --------------------------------------------------------

    error = (
        hp["actual_energy"]
        -
        hp["prediction"]
    )

    mae = error.abs().mean()

    rmse = (
        error ** 2
    ).mean() ** 0.5

    actual_mean = hp[
        "actual_energy"
    ].mean()

    denominator = (
        (
            hp["actual_energy"]
            -
            actual_mean
        ) ** 2
    ).sum()

    if denominator == 0:

        r2 = np.nan

    else:

        r2 = (
            1
            -
            (
                error ** 2
            ).sum()
            /
            denominator
        )

    c = st.columns(5)

    c[0].metric(
        "Prediction Days",
        f"{len(hp):,}"
    )

    c[1].metric(
        "Actual Mean",
        f"{hp['actual_energy'].mean():.3f}"
    )

    c[2].metric(
        "Predicted Mean",
        f"{hp['prediction'].mean():.3f}"
    )

    c[3].metric(
        "House RMSE",
        f"{rmse:.4f}"
    )

    c[4].metric(
        "House R²",
        f"{r2:.4f}"
    )

    st.subheader(
        f"Actual vs Predicted — {selected_house}"
    )

    fig = go.Figure()

    fig.add_scatter(
        x=hp["day"],
        y=hp["actual_energy"],
        name="Actual",
        line=dict(
            color=PRIMARY,
            width=2
        )
    )

    fig.add_scatter(
        x=hp["day"],
        y=hp["prediction"],
        name="Predicted",
        line=dict(
            color=AMBER,
            width=2
        )
    )

    plot(
        time_controls(
            style(
                fig,
                f"2014 House Prediction — {selected_house}",
                480,
                "Date",
                "Energy"
            )
        ),
        "house_prediction_chart",
        f"{selected_house}_prediction"
    )

    st.subheader(
        "Prediction Error"
    )

    hp["error"] = (
        hp["actual_energy"]
        -
        hp["prediction"]
    )

    hp["absolute_error"] = (
        hp["error"].abs()
    )

    c = st.columns(3)

    c[0].metric(
        "MAE",
        f"{mae:.4f}"
    )

    c[1].metric(
        "RMSE",
        f"{rmse:.4f}"
    )

    c[2].metric(
        "Maximum Absolute Error",
        f"{hp['absolute_error'].max():.4f}"
    )

    st.subheader(
        "House Prediction Data"
    )

    table(
        hp,
        f"{selected_house}_predictions",
        "download_house_predictions"
    )

    st.subheader(
        "House-Level Model Feature Importance"
    )

    feature_importance = pd.DataFrame(
        {
            "Feature": [
                "month",
                "day_of_month",
                "day_of_week",
                "is_weekend",
                "lag_1_energy",
                "rolling_7_energy"
            ],
            "Importance": [
                0.009619,
                0.000179,
                0.000540,
                0.000340,
                0.418692,
                0.570630
            ]
        }
    )

    feature_importance = (
        feature_importance
        .sort_values("Importance")
    )

    fig = go.Figure(
        go.Bar(
            y=feature_importance["Feature"],
            x=feature_importance["Importance"],
            orientation="h",
            marker_color=PRIMARY,
            text=feature_importance[
                "Importance"
            ].round(3),
            textposition="outside"
        )
    )

    plot(
        style(
            fig,
            "House-Level Random Forest Feature Importance",
            420,
            "Importance",
            legend=False
        ),
        "house_feature_importance",
        "house_feature_importance"
    )

    insight(
        "<b>rolling_7_energy</b> and "
        "<b>lag_1_energy</b> dominate the house-level model, "
        "with importances of 0.570630 and 0.418692 respectively."
    )


# ============================================================
# ANOMALY DETECTION
# ============================================================

elif page == "Anomaly Detection":

    st.header(
        "Spark MLlib — K-Means Anomaly Detection"
    )

    st.write(
        "An unsupervised K-Means model flags days "
        "that are unusually far from their cluster center."
    )

    flow(
        [
            "Processed Energy Data",
            "Feature Assembly",
            "StandardScaler",
            "K-Means (k = 3)",
            "Distance from Cluster Center",
            "Threshold",
            "Anomaly Classification"
        ]
    )

    pct = st.slider(
        "Threshold percentile",
        80.0,
        99.9,
        95.0,
        0.5
    )

    an = anomalies.copy()

    threshold = an[
        "distance_from_center"
    ].quantile(
        pct / 100
    )

    an["is_anomaly"] = (
        an["distance_from_center"]
        >= threshold
    ).astype(int)

    n_an = int(
        an["is_anomaly"].sum()
    )

    c = st.columns(4)

    c[0].metric(
        "Detected Anomalies",
        f"{n_an:,}"
    )

    c[1].metric(
        "Normal Days",
        f"{len(an) - n_an:,}"
    )

    c[2].metric(
        "Distance Threshold",
        f"{threshold:.4f}"
    )

    c[3].metric(
        "Anomaly Rate",
        f"{n_an / max(len(an), 1):.1%}"
    )

    st.subheader(
        "Anomalous Days"
    )

    anomalous = (
        an[
            an["is_anomaly"] == 1
        ]
        .sort_values(
            "distance_from_center",
            ascending=False
        )
    )

    table(
        anomalous[
            [
                "day",
                "total_energy",
                "household_count",
                "avg_energy",
                "cluster",
                "distance_from_center"
            ]
        ],
        "anomalies",
        "download_anomalies"
    )


# ============================================================
# DAILY PREDICTION
# ============================================================

elif page == "Daily Prediction":

    st.header(
        "Daily Energy Prediction"
    )

    st.write(
        "Select a day from the 2014 aggregate test period "
        "to compare the Random Forest prediction with actual usage."
    )

    dates = sorted(
        predictions[
            "day"
        ].dt.date.tolist()
    )

    if not dates:

        st.warning(
            "No prediction days in the selected range."
        )

        st.stop()

    selected_date = st.selectbox(
        "Select Date",
        dates
    )

    row = predictions[
        predictions["day"].dt.date
        == selected_date
    ].iloc[0]

    actual_value = row[
        "avg_energy"
    ]

    predicted_value = row[
        "prediction"
    ]

    error = (
        actual_value
        -
        predicted_value
    )

    c = st.columns(4)

    c[0].metric(
        "Predicted Usage",
        f"{predicted_value:.3f}"
    )

    c[1].metric(
        "Actual Usage",
        f"{actual_value:.3f}"
    )

    c[2].metric(
        "Absolute Error",
        f"{abs(error):.3f}"
    )

    c[3].metric(
        "% Error",
        f"{abs(error) / actual_value:.2%}"
        if actual_value
        else "n/a"
    )

    fig = go.Figure(
        go.Bar(
            x=[
                "Actual",
                "Predicted"
            ],
            y=[
                actual_value,
                predicted_value
            ],
            marker_color=[
                PRIMARY,
                AMBER
            ],
            text=[
                f"{actual_value:.3f}",
                f"{predicted_value:.3f}"
            ],
            textposition="outside"
        )
    )

    plot(
        style(
            fig,
            f"Actual vs Predicted — {selected_date}",
            400,
            ytitle="Average household energy",
            legend=False
        ),
        "daily_prediction_bar",
        f"prediction_{selected_date}"
    )

    st.info(
        "This prediction comes from the aggregate "
        "Spark MLlib Random Forest model."
    )


# ============================================================
# BIG DATA AND HDFS
# ============================================================

elif page == "Big Data and HDFS":

    st.header(
        "Big Data Infrastructure"
    )

    st.subheader(
        "Technology Stack"
    )

    stack = pd.DataFrame(
        {
            "Technology": [
                "Hadoop HDFS",
                "Apache Spark",
                "Scala",
                "Spark SQL",
                "Spark MLlib",
                "Random Forest",
                "K-Means",
                "Streamlit"
            ],
            "Purpose": [
                "Distributed storage",
                "Distributed processing",
                "Preprocessing and ML implementation",
                "Analytics",
                "Machine learning framework",
                "Energy prediction",
                "Anomaly detection",
                "Interactive web dashboard"
            ]
        }
    )

    st.dataframe(
        stack,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "HDFS Storage Paths"
    )

    st.code(
        f"""
Raw Dataset:

{RAW_PATH}

Processed Aggregate Dataset:

{HDFS_BASE}/processed/energy_daily/

Aggregate Predictions:

{HDFS_BASE}/processed/predictions/

Anomalies:

{HDFS_BASE}/processed/anomalies/

House-Level Daily Dataset:

{HDFS_BASE}/processed/house_daily/

House-Level Predictions:

{HDFS_BASE}/processed/house_predictions/
""",
        language="text"
    )

    st.subheader(
        "Storage and Model Results"
    )

    c = st.columns(6)

    c[0].metric(
        "Raw Dataset",
        "335.5 MB"
    )

    c[1].metric(
        "Raw Replication",
        "3"
    )

    c[2].metric(
        "Processed Days",
        "829"
    )

    c[3].metric(
        "Unique Houses",
        "5,561"
    )

    c[4].metric(
        "Houses Predicted",
        "5,108"
    )

    c[5].metric(
        "House Predictions",
        "298,099"
    )

    st.subheader(
        "Machine Learning Results"
    )

    results = pd.DataFrame(
        {
            "Model": [
                "Aggregate Random Forest",
                "House-Level Random Forest"
            ],
            "Target": [
                "avg_energy",
                "energy"
            ],
            "Train Period": [
                "2011–2013",
                "2011–2013"
            ],
            "Test Period": [
                "2014",
                "2014"
            ],
            "RMSE": [
                1.543546,
                4.499445
            ],
            "R²": [
                -0.004833,
                0.803691
            ]
        }
    )

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "HDFS Health Check"
    )

    if st.button(
        "Run HDFS Health Check"
    ):

        paths = [
            (
                "Processed aggregate dataset",
                f"{HDFS_BASE}/processed/energy_daily"
            ),
            (
                "Aggregate predictions",
                f"{HDFS_BASE}/processed/predictions"
            ),
            (
                "Anomalies",
                f"{HDFS_BASE}/processed/anomalies"
            ),
            (
                "House daily dataset",
                f"{HDFS_BASE}/processed/house_daily"
            ),
            (
                "House predictions",
                f"{HDFS_BASE}/processed/house_predictions"
            )
        ]

        for label, path in paths:

            available = hdfs_exists(
                path
            )

            status = (
                "AVAILABLE"
                if available
                else "MISSING"
            )

            css = (
                "ok"
                if available
                else "bad"
            )

            render(
                f"""
                <p>
                    <span class="badge {css}">
                        {status}
                    </span>
                    &nbsp;
                    {escape(label)}
                </p>
                """
            )

    st.subheader(
        "End-to-End Architecture"
    )

    flow(
        [
            "Raw London Smart Meter Dataset",
            "HDFS Raw Storage",
            "Scala Preprocessing",
            "Processed HDFS Data",
            "Spark SQL Analytics",
            "Aggregate Random Forest",
            "House-Level Random Forest",
            "K-Means Anomaly Detection",
            "HDFS Predictions",
            "Streamlit Dashboard"
        ]
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "London Smart Meter Analytics | "
    "Hadoop + Spark + Scala + Spark MLlib + Streamlit"
)
