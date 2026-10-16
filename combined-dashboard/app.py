from __future__ import annotations

import os
import subprocess
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =============================================================================
# Combined London + UK-REFIT Analytics
# Save as a NEW file; do not overwrite the existing London app.
#
# Suggested path:
# D:\BigData Docker\combined-dashboard\app.py
#
# Launch:
# python -m streamlit run "D:\BigData Docker\combined-dashboard\app.py"
# =============================================================================

st.set_page_config(
    page_title="Smart Meter Analytics | London + UK-REFIT",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------- Configuration -------------------------------

LONDON_HDFS_BASE = os.getenv(
    "LONDON_HDFS_BASE", "/data/raw/old-london-dataset-review1"
)
LONDON_PROCESSED_PATH = f"{LONDON_HDFS_BASE}/processed/energy_daily/part-*.csv"
LONDON_PREDICTION_PATH = f"{LONDON_HDFS_BASE}/processed/predictions/part-*.csv"
LONDON_ANOMALY_PATH = f"{LONDON_HDFS_BASE}/processed/anomalies/part-*.csv"
LONDON_HOUSE_DAILY_PATH = f"{LONDON_HDFS_BASE}/processed/house_daily/part-*.csv"
LONDON_HOUSE_PREDICTION_PATH = f"{LONDON_HDFS_BASE}/processed/house_predictions/part-*.csv"
LONDON_RAW_PATH = f"{LONDON_HDFS_BASE}/daily_dataset.csv"

UKREFIT_PROJECT_ROOT = Path(
    os.getenv("UKREFIT_PROJECT_ROOT", r"D:\BigData Docker\UK-REFIT")
)
UKREFIT_OUTPUT_DIR = Path(
    os.getenv("UKREFIT_OUTPUT_DIR", str(UKREFIT_PROJECT_ROOT / "output"))
)
UKREFIT_PREDICTIONS_PATH = Path(
    os.getenv(
        "UKREFIT_PREDICTIONS_CSV",
        str(UKREFIT_OUTPUT_DIR / "clipped_predictions.csv"),
    )
)
UKREFIT_APPLIANCE_METRICS_PATH = Path(
    os.getenv(
        "UKREFIT_APPLIANCE_METRICS_CSV",
        str(UKREFIT_OUTPUT_DIR / "clipped_appliance_metrics.csv"),
    )
)
UKREFIT_HOURLY_PATH = Path(
    os.getenv(
        "UKREFIT_HOURLY_SUMMARY_CSV",
        str(UKREFIT_OUTPUT_DIR / "appliance_hourly_summary.csv"),
    )
)
UKREFIT_DAILY_PATH = Path(
    os.getenv(
        "UKREFIT_DAILY_SUMMARY_CSV",
        str(UKREFIT_OUTPUT_DIR / "appliance_daily_summary.csv"),
    )
)
UKREFIT_HDFS_BASE = "/data/raw/UKrefit-new/csv/"
UKREFIT_HDFS_PROCESSED = "/data/processed/UKrefit"
UKREFIT_HDFS_RESULTS = "/data/results/UKrefit/forecast_improved"

# London HDFS commands run inside the namenode container.
HDFS_CONTAINER = os.getenv("HDFS_CONTAINER", "namenode")

# ------------------------------- Theme ---------------------------------------

INK = "#16212B"
MUTED = "#5B6B7A"
GRID = "#E7ECF0"
PRIMARY = "#0E5A6B"
AMBER = "#E8A200"
GREEN = "#1F8A5B"
RED = "#C0392B"
PURPLE = "#6C5CE7"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root{--bg:#F4F6F8;--surface:#FFFFFF;--ink:#16212B;--muted:#5B6B7A;--line:#E1E6EB;--primary:#0E5A6B;--amber:#E8A200;}
html,body,.stApp,[class*="css"]{font-family:'IBM Plex Sans','Segoe UI',system-ui,sans-serif;}
.stApp{background:var(--bg);color:var(--ink);}
.block-container{padding-top:1.4rem;padding-bottom:3rem;max-width:1380px;}
h1,h2,h3,h4{font-family:'Sora','IBM Plex Sans',sans-serif;color:var(--ink);letter-spacing:-.01em;}
h2{font-size:24px!important;} h3{font-size:17px!important;margin-top:1.2rem;}
#MainMenu,footer{visibility:hidden;}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--line);}
[data-testid="stSidebar"] :is(h1,h2,h3,p,label,span,li){color:var(--ink)!important;}
header[data-testid="stHeader"]{background:var(--bg);}
.stApp :is(label,[data-testid="stWidgetLabel"] p){color:var(--ink)!important;}
.stApp [data-testid="stCaptionContainer"]{color:var(--muted)!important;}
.brand{display:flex;align-items:center;gap:10px;margin-bottom:8px;}
.brand-mark{width:36px;height:36px;border-radius:9px;background:var(--primary);color:#fff;display:flex;align-items:center;justify-content:center;font-size:19px;}
.brand-name{font-family:'Sora',sans-serif;font-weight:700;font-size:16px;line-height:1.1;}
.brand-sub{color:var(--muted)!important;font-size:12px;}
.hero{background:linear-gradient(120deg,#0B3F4C 0%,#0E5A6B 100%);border-radius:14px;padding:26px 32px;margin-bottom:20px;border-left:6px solid var(--amber);}
.hero h1{color:#fff!important;font-size:29px;margin:0 0 6px;font-weight:700;}
.hero p{color:#C9DCE2!important;font-size:15px;margin:0 0 14px;max-width:850px;}
.chip{display:inline-block;padding:4px 11px;margin:0 6px 6px 0;border-radius:999px;font-size:12.5px;font-weight:500;background:rgba(255,255,255,.1);color:#E8F1F4!important;border:1px solid rgba(255,255,255,.18);}
[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);border-top:3px solid var(--primary);border-radius:12px;padding:14px 18px;}
[data-testid="stMetricLabel"] p{color:var(--muted)!important;font-weight:500;font-size:13px;}
[data-testid="stMetricValue"]{font-family:'Sora',sans-serif;font-weight:700;color:var(--ink);}
.flow{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:6px 0 12px;}
.step{background:#fff;border:1px solid var(--line);border-radius:8px;padding:9px 13px;font-size:13.5px;font-weight:500;}
.step.first{background:#E3F0F3;border-color:#BCD8DF;}.step.last{background:var(--primary);border-color:var(--primary);color:#fff;}
.insight{background:#EAF4F6;border:1px solid #C5DEE4;border-left:4px solid var(--primary);border-radius:10px;padding:12px 16px;font-size:14px;margin:10px 0;}
.stTabs [data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{padding:8px 14px;font-weight:500;}
.stTabs [aria-selected="true"]{color:var(--primary)!important;font-weight:600;}
[data-testid="stPlotlyChart"]{background:#fff;border:1px solid var(--line);border-radius:12px;padding:6px;}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:10px;overflow:hidden;}\n[data-testid="stAlert"]{color:#16212B!important;} [data-testid="stAlert"] p{color:#16212B!important;} [data-testid="stMarkdownContainer"] p{color:#16212B;} [data-testid="stWidgetLabel"] p{color:#16212B!important;} [role="radiogroup"] label p{color:#16212B!important;}
.stButton>button,.stDownloadButton>button{border-radius:8px;border:1px solid var(--primary);background:#fff;color:var(--primary);font-weight:500;}
.stButton>button:hover,.stDownloadButton>button:hover{background:var(--primary);color:#fff;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# -------------------------------- Helpers -------------------------------------

def render(markup: str, sidebar: bool = False) -> None:
    (st.sidebar if sidebar else st).markdown(markup, unsafe_allow_html=True)


def flow(steps: list[str]) -> None:
    import html
    parts = []
    for i, name in enumerate(steps):
        cls = "step"
        if i == 0:
            cls += " first"
        if i == len(steps) - 1:
            cls += " last"
        parts.append(f'<div class="{cls}">{html.escape(name)}</div>')
    render('<div class="flow">' + '<span style="color:#5B6B7A">›</span>'.join(parts) + "</div>")


def style(fig, title, height=440, xtitle=None, ytitle=None, legend=True):
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.01, font=dict(size=16, color=INK)),
        height=height, margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="IBM Plex Sans, sans-serif", color=MUTED),
        hovermode="x unified", showlegend=legend,
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
    )
    fig.update_xaxes(title=xtitle, gridcolor=GRID, showline=True, linecolor="#CBD3DA")
    fig.update_yaxes(title=ytitle, gridcolor=GRID, zeroline=False)
    return fig


def plot(fig, key, filename="chart"):
    st.plotly_chart(
        fig, use_container_width=True, key=key,
        config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "filename": filename, "scale": 3}},
    )


def table(df, filename, key):
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV", df.to_csv(index=False).encode("utf-8-sig"),
        f"{filename}.csv", "text/csv", key=key,
    )


def insight(text):
    render(f'<div class="insight">💡 {text}</div>')


def time_controls(fig):
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.06),
        rangeselector=dict(buttons=[
            dict(count=1, label="1M", step="month", stepmode="backward"),
            dict(count=3, label="3M", step="month", stepmode="backward"),
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(step="all", label="All"),
        ]),
    )
    return fig


def run_hdfs(command: str, timeout=30):
    return subprocess.run(
        ["docker", "exec", HDFS_CONTAINER, "bash", "-c", command],
        capture_output=True, text=True, timeout=timeout, check=False,
    )


def read_hdfs_csv(path: str) -> pd.DataFrame:
    # cat all part files; intended for aggregated output, not raw high-volume files.
    result = run_hdfs(f"hdfs dfs -cat '{path}'")
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "HDFS read failed")
    return pd.read_csv(StringIO(result.stdout))


def load_london_house_list() -> list[str]:
    result = run_hdfs(
        f"hdfs dfs -cat '{LONDON_HOUSE_DAILY_PATH}' "
        "| awk -F',' 'NR>1 {print $1}' | sort -u",
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Could not list London households")
    return [v.strip() for v in result.stdout.splitlines() if v.strip() and v.strip() != "LCLid"]


def load_london_house(house_id: str) -> pd.DataFrame:
    # Escape apostrophes in the selected identifier before embedding it in awk.
    safe = house_id.replace("\\", "\\\\").replace("'", "\\'")
    result = run_hdfs(
        f"hdfs dfs -cat '{LONDON_HOUSE_DAILY_PATH}' "
        f"| awk -F',' -v h='{safe}' 'NR==1 || $1==h'",
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Could not load house data")
    df = pd.read_csv(StringIO(result.stdout))
    if "day" in df:
        df["day"] = pd.to_datetime(df["day"], errors="coerce")
    return df.dropna(subset=["day"]) if "day" in df else df


def load_london_house_predictions() -> pd.DataFrame:
    df = read_hdfs_csv(LONDON_HOUSE_PREDICTION_PATH)
    if "day" in df:
        df["day"] = pd.to_datetime(df["day"], errors="coerce")
    for col in ("actual_energy", "prediction"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=[c for c in ["day", "actual_energy", "prediction"] if c in df])


def load_london():
    energy = read_hdfs_csv(LONDON_PROCESSED_PATH)
    predictions = read_hdfs_csv(LONDON_PREDICTION_PATH)
    anomalies = read_hdfs_csv(LONDON_ANOMALY_PATH)
    for df in (energy, predictions, anomalies):
        if "day" in df:
            df["day"] = pd.to_datetime(df["day"], errors="coerce")
    return energy, predictions, anomalies


def read_local_csv(path: Path) -> pd.DataFrame:
    """Read a local CSV safely; return empty when it is absent."""
    if not path.exists() or not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def normalize_uk_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    # Spark part files may be concatenated with repeated headers.
    if not df.empty and len(df.columns):
        first = df.columns[0]
        df = df[df[first].astype(str).str.strip().str.lower() != first.lower()]
    return df.reset_index(drop=True)


def read_ukrefit_hdfs(path: str) -> pd.DataFrame:
    """Read a processed HDFS CSV directory or glob via the NameNode container."""
    # `hdfs` works via `docker exec namenode hdfs ...`; do not require an interactive shell.
    command = f"docker exec {HDFS_CONTAINER} hdfs dfs -cat '{path}'"
    result = subprocess.run(command, capture_output=True, text=True, timeout=180, shell=True)
    if result.returncode != 0 or not result.stdout.strip():
        return pd.DataFrame()
    try:
        return normalize_uk_columns(pd.read_csv(StringIO(result.stdout), low_memory=False))
    except Exception:
        return pd.DataFrame()


def first_available(local_path: Path, hdfs_paths: list[str]) -> pd.DataFrame:
    df = normalize_uk_columns(read_local_csv(local_path))
    if not df.empty:
        return df
    for hdfs_path in hdfs_paths:
        df = read_ukrefit_hdfs(hdfs_path)
        if not df.empty:
            return df
    return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=60)
def load_ukrefit():
    """Prefer local exported CSVs, then fall back to the processed HDFS outputs."""
    return {
        "predictions": first_available(
            UKREFIT_PREDICTIONS_PATH,
            [
                f"{UKREFIT_HDFS_RESULTS}/forecasts/test_predictions_clipped/*",
                f"{UKREFIT_HDFS_RESULTS}/forecasts/test_predictions/*",
            ],
        ),
        "metrics": first_available(
            UKREFIT_APPLIANCE_METRICS_PATH,
            [f"{UKREFIT_HDFS_RESULTS}/metrics/clipped_appliance_metrics/*"],
        ),
        "hourly": first_available(
            UKREFIT_HOURLY_PATH,
            [f"{UKREFIT_HDFS_PROCESSED}/appliance_hourly_summary/*"],
        ),
        "daily": first_available(
            UKREFIT_DAILY_PATH,
            [f"{UKREFIT_HDFS_PROCESSED}/appliance_daily_summary/*"],
        ),
    }


def uk_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Timestamp" in out:
        out["_ts"] = pd.to_datetime(out["Timestamp"], errors="coerce")
    elif "DateTime" in out:
        out["_ts"] = pd.to_datetime(out["DateTime"], errors="coerce")
    elif "Date" in out and "Hour" in out:
        out["_ts"] = pd.to_datetime(
            out["Date"].astype(str) + " " + out["Hour"].astype(str) + ":00:00",
            errors="coerce",
        )
    elif "Date" in out:
        out["_ts"] = pd.to_datetime(out["Date"], errors="coerce")
    elif "day" in out:
        out["_ts"] = pd.to_datetime(out["day"], errors="coerce")
    elif "Unix" in out:
        out["_ts"] = pd.to_datetime(out["Unix"], unit="s", errors="coerce")
    else:
        out["_ts"] = pd.NaT
    return out


def apply_uk_filters(df: pd.DataFrame, houses: list[str], apps: list[str], date_range, hours: list[int]):
    if df.empty:
        return df
    out = df.copy()
    hcol = next((c for c in ["House", "HouseId", "House_ID"] if c in out), None)
    acol = next((c for c in ["Appliance", "Channel", "ApplianceName"] if c in out), None)
    if hcol and houses:
        def _house_key(value):
            text_value = str(value).strip()
            try:
                number = float(text_value)
                if number.is_integer():
                    return str(int(number))
            except (TypeError, ValueError):
                pass
            return text_value
        allowed_houses = {_house_key(v) for v in houses}
        out = out[out[hcol].map(_house_key).isin(allowed_houses)]
    if acol and apps:
        out = out[out[acol].astype(str).isin(apps)]
    out = uk_timestamp(out)
    if out["_ts"].notna().any():
        if date_range and len(date_range) == 2:
            lo, hi = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
            out = out[(out["_ts"] >= lo) & (out["_ts"] < hi)]
        if hours and "Hour" in out:
            out = out[pd.to_numeric(out["Hour"], errors="coerce").isin(hours)]
        elif hours:
            out = out[out["_ts"].dt.hour.isin(hours)]
    elif hours and "Hour" in out:
        out = out[pd.to_numeric(out["Hour"], errors="coerce").isin(hours)]
    return out


def calculate_anomalies(df: pd.DataFrame, value_col: str, groups: list[str], resolution: str):
    if df.empty or value_col not in df:
        return pd.DataFrame()
    out = df.copy()
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=[value_col])
    if out.empty:
        return out
    if groups:
        grouped = out.groupby(groups, dropna=False)[value_col]
        out["_baseline"] = grouped.transform("median")
        out["_q1"] = grouped.transform(lambda s: s.quantile(.25))
        out["_q3"] = grouped.transform(lambda s: s.quantile(.75))
        out["_mad"] = grouped.transform(lambda s: (s - s.median()).abs().median())
    else:
        out["_baseline"] = out[value_col].median()
        out["_q1"], out["_q3"] = out[value_col].quantile(.25), out[value_col].quantile(.75)
        out["_mad"] = (out[value_col] - out[value_col].median()).abs().median()
    out["_iqr"] = out["_q3"] - out["_q1"]
    robust_z = (out[value_col] - out["_baseline"]) / (1.4826 * out["_mad"].replace(0, np.nan))
    out["_robust_z"] = robust_z
    out["_lower"] = out["_q1"] - 1.5 * out["_iqr"]
    out["_upper"] = out["_q3"] + 1.5 * out["_iqr"]
    out["_anomaly"] = robust_z.abs().ge(3.5) | (out[value_col] < out["_lower"]) | (out[value_col] > out["_upper"])
    out["_deviation"] = out[value_col] - out["_baseline"]
    score = robust_z.abs().fillna(0)
    iqr_score = ((out[value_col] - out["_baseline"]).abs() / out["_iqr"].replace(0, np.nan)).fillna(0)
    score = score.where(score > 0, iqr_score)
    out["_severity"] = np.select(
        [score >= 5, score >= 3.5, score >= 2],
        ["Critical", "High", "Medium"],
        default="Low",
    )
    out["_reason"] = "Outside historical robust range (median/MAD and IQR)"
    out["_resolution"] = resolution
    return out[out["_anomaly"]].copy()


def metric_col(df: pd.DataFrame, candidates: list[str]):
    return next((c for c in candidates if c in df.columns), None)


# ------------------------------ Sidebar ---------------------------------------

render(
    """
    <div class="brand">
      <div class="brand-mark">⚡</div>
      <div><div class="brand-name">Smart Meter Analytics</div>
      <div class="brand-sub">London + UK-REFIT</div></div>
    </div>
    """,
    sidebar=True,
)

dataset = st.sidebar.radio(
    "Dataset workspace",
    ["London Smart Meter", "UK-REFIT"],
    help="Switch between datasets. Each workspace has independent source paths and pages.",
)
st.sidebar.divider()

# -------------------------------- London app ----------------------------------

def render_london_workspace():
    page = st.sidebar.radio(
        "London navigation",
        [
            "Overview", "Preprocessing", "Exploratory Analysis", "Correlation",
            "Regression and ML", "House-Level Analysis", "House-Level Predictions",
            "Anomaly Detection", "Daily Prediction", "Big Data and HDFS",
        ],
        key="london_page",
    )
    if st.sidebar.button("Reload London data from HDFS", key="reload_london", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    try:
        energy, predictions, anomalies = load_london()
    except Exception as exc:
        render('<div class="hero"><h1>London Smart Meter Analytics</h1><p>HDFS-backed London workspace</p></div>')
        st.error("Unable to load London aggregate data. Verify Docker/HDFS and the London output paths.")
        st.code(str(exc))
        return

    if energy.empty:
        st.error("London energy summary is empty.")
        return
    energy["day"] = pd.to_datetime(energy["day"], errors="coerce")
    dmin, dmax = energy["day"].min().date(), energy["day"].max().date()
    selected = st.sidebar.date_input(
        "London date range", (dmin, dmax), min_value=dmin, max_value=dmax, key="london_dates"
    )
    lo, hi = (pd.Timestamp(selected[0]), pd.Timestamp(selected[1])) if isinstance(selected, (list, tuple)) and len(selected) == 2 else (pd.Timestamp(dmin), pd.Timestamp(dmax))
    energy = energy[energy["day"].between(lo, hi)]
    predictions = predictions[predictions["day"].between(lo, hi)] if "day" in predictions else predictions
    anomalies = anomalies[anomalies["day"].between(lo, hi)] if "day" in anomalies else anomalies

    render("""
    <div class="hero"><h1>⚡ London Smart Meter Analytics</h1>
    <p>Big Data Analytics using Hadoop HDFS, Apache Spark, Scala and Spark MLlib</p>
    <span class="chip">Hadoop HDFS</span><span class="chip">Apache Spark</span>
    <span class="chip">Spark SQL</span><span class="chip">Spark MLlib</span>
    <span class="chip">Random Forest</span><span class="chip">K-Means</span></div>
    """)

    if page == "Overview":
        st.header("Project Overview")
        c = st.columns(4)
        c[0].metric("Processed Days", f"{len(energy):,}")
        c[1].metric("Households", "5,561")
        c[2].metric("House-Day Records", "3,510,403")
        c[3].metric("Anomalies in selected range", f"{len(anomalies):,}")
        c = st.columns(4)
        c[0].metric("Aggregate RMSE", "1.5435")
        c[1].metric("Aggregate R²", "-0.0048")
        c[2].metric("House RMSE", "4.4994")
        c[3].metric("House R²", "0.8037")
        st.subheader("Project pipeline")
        flow(["London Smart Meter Dataset", "HDFS", "Spark + Scala", "Cleaning + Aggregation", "Spark SQL + MLlib", "Predictions + Anomalies", "Dashboard"])
        if "total_energy" in energy:
            metric = st.radio("Trend metric", ["total_energy", "avg_energy", "household_count"], horizontal=True, key="lon_trend_metric")
            smooth = st.slider("Moving average window (days)", 1, 60, 7, key="lon_smooth")
            fig = go.Figure()
            fig.add_scatter(x=energy["day"], y=energy[metric], name="Daily", mode="lines", line=dict(color=PRIMARY, width=1.3), fill="tozeroy", fillcolor="rgba(14,90,107,.12)")
            if smooth > 1:
                fig.add_scatter(x=energy["day"], y=energy[metric].rolling(smooth, min_periods=1).mean(), name=f"{smooth}-day average", line=dict(color=AMBER, width=2.4))
            if len(energy):
                peak = energy.loc[pd.to_numeric(energy[metric], errors="coerce").idxmax()]
                fig.add_scatter(x=[peak["day"]], y=[peak[metric]], mode="markers+text", name="Peak day", marker=dict(color=RED, size=10), text=["Peak"], textposition="top center")
            plot(time_controls(style(fig, f"Daily {metric.replace('_',' ').title()}", 470, "Date", metric)), "lon_overview_trend", "london_daily_trend")

    elif page == "Preprocessing":
        st.header("Data Preprocessing")
        st.write(f"Raw path: `{LONDON_RAW_PATH}`")
        flow(["Raw CSV in HDFS", "Spark cleaning", "Deduplication", "Daily aggregation", "Feature engineering", "Processed HDFS summaries"])
        st.info("Counts and schema below are project metadata from the original London dashboard; verify them against your current pipeline before using them in a formal report.")
        c = st.columns(4)
        c[0].metric("Raw Records", "3,510,433")
        c[1].metric("After Cleaning", "3,510,403")
        c[2].metric("Final Daily Rows", f"{len(energy):,}")
        c[3].metric("Records Retained", f"{3510403/3510433:.4%}")
        table(energy.head(500), "london_processed_energy_daily", "lon_preprocess_download")

    elif page == "Exploratory Analysis":
        st.header("Exploratory Data Analysis")
        tabs = st.tabs(["Trends", "Seasonality", "Weekday vs Weekend", "Top Days"])
        with tabs[0]:
            if "total_energy" in energy:
                fig = go.Figure(go.Scatter(x=energy["day"], y=energy["total_energy"], line=dict(color=PRIMARY, width=1.6), fill="tozeroy", fillcolor="rgba(14,90,107,.12)", name="Total energy"))
                plot(time_controls(style(fig, "Total Energy Consumption", 460, "Date", "Total energy", False)), "lon_eda_trends", "london_energy_trend")
        with tabs[1]:
            if {"year", "month", "avg_energy"}.issubset(energy.columns):
                m = energy.groupby(["year", "month"], as_index=False)["avg_energy"].mean()
                m["period"] = pd.to_datetime(m["year"].astype(str) + "-" + m["month"].astype(str).str.zfill(2) + "-01", errors="coerce")
                fig = px.line(m, x="period", y="avg_energy", markers=True)
                plot(style(fig, "Monthly Average Household Energy", 430, "Month", "Average energy", False), "lon_eda_season", "london_monthly_energy")
        with tabs[2]:
            if {"is_weekend", "total_energy", "avg_energy"}.issubset(energy.columns):
                comp = energy.groupby("is_weekend", as_index=False).agg(avg_total=("total_energy", "mean"), avg_house=("avg_energy", "mean"), days=("day", "count"))
                comp["day_type"] = comp["is_weekend"].map({0: "Weekday", 1: "Weekend"})
                st.dataframe(comp, use_container_width=True, hide_index=True)
                plot(style(px.bar(comp, x="day_type", y="avg_total", color="day_type"), "Average Total Energy by Day Type", 370, ytitle="Average total energy"), "lon_weekend_total", "london_weekend_total")
        with tabs[3]:
            if "total_energy" in energy:
                n = st.slider("Top days", 5, 30, 10, key="lon_topn")
                table(energy.nlargest(n, "total_energy"), "london_top_days", "lon_top_days_download")

    elif page == "Correlation":
        st.header("Correlation Analysis")
        cols = [c for c in ["total_energy", "household_count", "avg_energy", "month", "day_of_month", "day_of_week", "is_weekend"] if c in energy]
        selected_cols = st.multiselect("Variables", cols, default=cols, key="lon_corr_cols")
        if len(selected_cols) >= 2:
            corr = energy[selected_cols].corr(numeric_only=True)
            fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1, colorscale="RdBu_r", text=corr.round(2).values, texttemplate="%{text}", colorbar=dict(title="Pearson")))
            fig.update_yaxes(autorange="reversed")
            plot(style(fig, "Correlation Heatmap", 520), "lon_corr_heat", "london_correlation")

    elif page == "Regression and ML":
        st.header("Spark MLlib — Aggregate Random Forest")
        if predictions.empty or not {"avg_energy", "prediction"}.issubset(predictions.columns):
            st.warning("No compatible London aggregate predictions are available in the selected date range.")
        else:
            actual = pd.to_numeric(predictions["avg_energy"], errors="coerce")
            predicted = pd.to_numeric(predictions["prediction"], errors="coerce")
            valid = actual.notna() & predicted.notna()
            actual, predicted = actual[valid], predicted[valid]
            rmse = float(np.sqrt(((actual-predicted)**2).mean())) if len(actual) else np.nan
            mae = float((actual-predicted).abs().mean()) if len(actual) else np.nan
            denom = float(((actual-actual.mean())**2).sum()) if len(actual) else 0
            r2 = 1 - float(((actual-predicted)**2).sum())/denom if denom else np.nan
            c = st.columns(3)
            c[0].metric("RMSE", f"{rmse:.4f}" if pd.notna(rmse) else "—")
            c[1].metric("MAE", f"{mae:.4f}" if pd.notna(mae) else "—")
            c[2].metric("R²", f"{r2:.4f}" if pd.notna(r2) else "—")
            fig = go.Figure()
            fig.add_scatter(x=predictions.loc[valid, "day"], y=actual, name="Actual", line=dict(color=PRIMARY, width=2))
            fig.add_scatter(x=predictions.loc[valid, "day"], y=predicted, name="Predicted", line=dict(color=AMBER, width=2))
            plot(time_controls(style(fig, "Aggregate Actual vs Predicted", 470, "Date", "Average household energy")), "lon_rf_actual_pred", "london_rf_predictions")

    elif page == "House-Level Analysis":
        st.header("House-Level Energy Analysis")
        try:
            houses = load_london_house_list()
        except Exception as exc:
            st.error("Could not load London house list.")
            st.code(str(exc))
            return
        if not houses:
            st.warning("No houses found.")
            return
        house = st.selectbox("Select household (LCLid)", houses, key="lon_house")
        try:
            df = load_london_house(house)
        except Exception as exc:
            st.error(str(exc))
            return
        if df.empty or "energy" not in df:
            st.warning("No compatible house data was found.")
            return
        df = df.sort_values("day")
        c = st.columns(4)
        c[0].metric("House", house)
        c[1].metric("Days available", f"{len(df):,}")
        c[2].metric("Average energy", f"{df['energy'].mean():.3f}")
        c[3].metric("Maximum energy", f"{df['energy'].max():.3f}")
        plot(time_controls(style(px.line(df, x="day", y="energy"), f"Energy Consumption — {house}", 460, "Date", "Energy", False)), "lon_house_timeline", f"{house}_timeline")
        table(df, f"{house}_daily", "lon_house_download")

    elif page == "House-Level Predictions":
        st.header("House-Level Random Forest Predictions")
        try:
            hp_all = load_london_house_predictions()
        except Exception as exc:
            st.error("Could not load house-level predictions.")
            st.code(str(exc))
            return
        if hp_all.empty or "LCLid" not in hp_all or not {"actual_energy", "prediction"}.issubset(hp_all.columns):
            st.warning("House-level prediction file is empty or its schema differs from expected LCLid, actual_energy, prediction columns.")
            return
        houses = sorted(hp_all["LCLid"].dropna().astype(str).unique())
        house = st.selectbox("Select household (LCLid)", houses, key="lon_pred_house")
        hp = hp_all[hp_all["LCLid"].astype(str) == house].sort_values("day").copy()
        err = hp["actual_energy"] - hp["prediction"]
        c = st.columns(4)
        c[0].metric("Prediction days", f"{len(hp):,}")
        c[1].metric("MAE", f"{err.abs().mean():.4f}")
        c[2].metric("RMSE", f"{np.sqrt((err**2).mean()):.4f}")
        c[3].metric("Max absolute error", f"{err.abs().max():.4f}")
        fig = go.Figure()
        fig.add_scatter(x=hp["day"], y=hp["actual_energy"], name="Actual", line=dict(color=PRIMARY, width=2))
        fig.add_scatter(x=hp["day"], y=hp["prediction"], name="Predicted", line=dict(color=AMBER, width=2))
        plot(time_controls(style(fig, f"Actual vs Predicted — {house}", 460, "Date", "Energy")), "lon_house_pred_chart", f"{house}_prediction")
        table(hp, f"{house}_predictions", "lon_house_pred_download")

    elif page == "Anomaly Detection":
        st.header("K-Means Anomaly Detection")
        if anomalies.empty or "distance_from_center" not in anomalies:
            st.warning("London anomaly output is missing or has an unexpected schema.")
        else:
            pct = st.slider("Threshold percentile", 80.0, 99.9, 95.0, .5, key="lon_anom_pct")
            an = anomalies.copy()
            an["distance_from_center"] = pd.to_numeric(an["distance_from_center"], errors="coerce")
            threshold = an["distance_from_center"].quantile(pct/100)
            an["is_anomaly"] = (an["distance_from_center"] >= threshold).astype(int)
            selected_an = an[an["is_anomaly"] == 1].sort_values("distance_from_center", ascending=False)
            c = st.columns(4)
            c[0].metric("Flagged rows", f"{len(selected_an):,}")
            c[1].metric("Threshold", f"{threshold:.4f}")
            c[2].metric("Anomaly rate", f"{len(selected_an)/max(len(an),1):.1%}")
            c[3].metric("Percentile", f"{pct:.1f}")
            table(selected_an, "london_anomalies", "lon_anom_download")

    elif page == "Daily Prediction":
        st.header("Daily Aggregate Prediction")
        if predictions.empty or not {"day", "avg_energy", "prediction"}.issubset(predictions.columns):
            st.warning("No compatible London prediction rows are available.")
        else:
            dates = sorted(predictions["day"].dropna().dt.date.unique().tolist())
            selected_date = st.selectbox("Select date", dates, key="lon_daily_pred_date")
            row = predictions[predictions["day"].dt.date == selected_date].iloc[0]
            actual = float(row["avg_energy"])
            forecast = float(row["prediction"])
            c = st.columns(4)
            c[0].metric("Actual", f"{actual:.3f}")
            c[1].metric("Predicted", f"{forecast:.3f}")
            c[2].metric("Absolute error", f"{abs(actual-forecast):.3f}")
            c[3].metric("% error", f"{abs(actual-forecast)/abs(actual):.2%}" if actual else "—")
            plot(style(px.bar(pd.DataFrame({"Type": ["Actual", "Predicted"], "Energy": [actual, forecast]}), x="Type", y="Energy", color="Type"), f"Prediction for {selected_date}", 380, ytitle="Energy"), "lon_daily_pred_bar", "london_daily_prediction")

    elif page == "Big Data and HDFS":
        st.header("Big Data Infrastructure — London")
        st.dataframe(pd.DataFrame([
            {"Dataset": "Raw data", "HDFS path": LONDON_RAW_PATH},
            {"Dataset": "Processed daily", "HDFS path": LONDON_PROCESSED_PATH},
            {"Dataset": "Aggregate predictions", "HDFS path": LONDON_PREDICTION_PATH},
            {"Dataset": "Anomalies", "HDFS path": LONDON_ANOMALY_PATH},
            {"Dataset": "House daily", "HDFS path": LONDON_HOUSE_DAILY_PATH},
            {"Dataset": "House predictions", "HDFS path": LONDON_HOUSE_PREDICTION_PATH},
        ]), use_container_width=True, hide_index=True)
        if st.button("Check London HDFS paths", key="lon_hdfs_check"):
            checks = []
            for label, path in [
                ("Processed daily", f"{LONDON_HDFS_BASE}/processed/energy_daily"),
                ("Predictions", f"{LONDON_HDFS_BASE}/processed/predictions"),
                ("Anomalies", f"{LONDON_HDFS_BASE}/processed/anomalies"),
                ("House daily", f"{LONDON_HDFS_BASE}/processed/house_daily"),
                ("House predictions", f"{LONDON_HDFS_BASE}/processed/house_predictions"),
            ]:
                result = run_hdfs(f"hdfs dfs -test -e '{path}'")
                checks.append({"Dataset": label, "Path": path, "Available": result.returncode == 0})
            st.dataframe(pd.DataFrame(checks), use_container_width=True, hide_index=True)


# ------------------------------ UK-REFIT app ----------------------------------
def prepare_uk_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Clean historical UK-REFIT actual-versus-predicted rows."""
    if df.empty:
        return pd.DataFrame()

    out = normalize_uk_columns(df)

    required = {"House", "Date", "Hour", "Appliance", "label", "prediction"}
    if not required.issubset(out.columns):
        return pd.DataFrame()

    out["House"] = out["House"].map(
        lambda v: str(int(float(v)))
        if pd.notna(v) and str(v).replace(".", "", 1).isdigit()
        and float(v).is_integer()
        else str(v).strip()
    )
    out["Appliance"] = out["Appliance"].astype(str).str.strip()
    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    out["prediction"] = pd.to_numeric(out["prediction"], errors="coerce")
    out["Hour"] = pd.to_numeric(out["Hour"], errors="coerce")
    out["_ts"] = pd.to_datetime(
        out["Date"].astype(str) + " " +
        out["Hour"].fillna(0).astype(int).astype(str) + ":00:00",
        errors="coerce",
    )
    out["ErrorW"] = out["prediction"] - out["label"]
    out["AbsoluteErrorW"] = out["ErrorW"].abs()

    return out.dropna(
        subset=["_ts", "label", "prediction"]
    ).sort_values("_ts").reset_index(drop=True)

def render_ukrefit_workspace():
    page = st.sidebar.radio(
        "UK-REFIT navigation",
        [
    "Overview",
    "Consumption Analysis",
    "Prediction vs Actual",
    "Anomaly Detection",
    "Forecast Evaluation",
    "Data Explorer",
    "Project & Data Quality",
],
        key="uk_page",
    )
    if st.sidebar.button("Reload UK-REFIT local files", key="reload_uk", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    data = load_ukrefit()
    pred = data["predictions"]
    metrics = data["metrics"]
    hourly = data["hourly"]
    daily = data["daily"]

    with st.expander("Data source status", expanded=any(df.empty for df in [pred, metrics, hourly, daily])):
        source_status = pd.DataFrame([
            {"Dataset": "Historical predictions", "Local file": str(UKREFIT_PREDICTIONS_PATH), "Rows loaded": len(pred), "Status": "Loaded" if not pred.empty else "Missing"},
            {"Dataset": "Appliance metrics", "Local file": str(UKREFIT_APPLIANCE_METRICS_PATH), "Rows loaded": len(metrics), "Status": "Loaded" if not metrics.empty else "Missing"},
            {"Dataset": "Hourly summary", "Local file": str(UKREFIT_HOURLY_PATH), "Rows loaded": len(hourly), "Status": "Loaded" if not hourly.empty else "Missing"},
            {"Dataset": "Daily summary", "Local file": str(UKREFIT_DAILY_PATH), "Rows loaded": len(daily), "Status": "Loaded" if not daily.empty else "Missing"},
        ])
        st.dataframe(source_status, use_container_width=True, hide_index=True)
        st.caption("The complete local hourly CSV is the primary UK-REFIT graphing source. The processed HDFS summary directories shown in your setup contain Parquet files; they cannot be parsed as CSV. If the daily CSV is absent, the app derives a daily mean-power summary from the hourly file.")
    if all(df.empty for df in [pred, metrics, hourly, daily]):
        st.error("No UK-REFIT saved outputs were found. Check the source status above and verify the HDFS paths under /data/processed/UKrefit and /data/results/UKrefit/forecast_improved.")

    render("""
    <div class="hero"><h1>⚡ UK-REFIT Energy Analytics</h1>
    <p>Household energy analysis and historical forecast evaluation using Spark-processed outputs</p>
    <span class="chip">Spark</span><span class="chip">HDFS</span><span class="chip">Random Forest / GBT</span>
    <span class="chip">Anomaly Analysis</span><span class="chip">Streamlit</span></div>
    """)
    st.info("UK-REFIT appliance channels are generic Appliance1–Appliance9. Saved predictions are historical test-period predictions, not automatically future forecasts.")

    # Filters must be based on the full consumption summary, NOT the smaller
    # historical test-prediction file (which covers only a limited date range).
    filter_df = hourly if not hourly.empty else (daily if not daily.empty else pred)
    house_col = metric_col(filter_df, ["House", "HouseId", "House_ID"])
    app_col = metric_col(filter_df, ["Appliance", "Channel", "ApplianceName"])

    def normalize_house(value):
        text_value = str(value).strip()
        # Prediction CSV may parse House as 5.0 while summary CSV stores "5".
        try:
            number = float(text_value)
            if number.is_integer():
                return str(int(number))
        except (TypeError, ValueError):
            pass
        return text_value

    house_opts = sorted({normalize_house(v) for v in filter_df[house_col].dropna()}) if house_col else []
    app_opts = sorted(filter_df[app_col].dropna().astype(str).unique()) if app_col else []
    with st.sidebar:
        st.subheader("UK-REFIT filters")
        chosen_houses = st.multiselect("House", house_opts, default=house_opts, key="uk_houses")
        chosen_apps = st.multiselect("Appliance channel", app_opts, default=app_opts, key="uk_apps")
        ts_df = uk_timestamp(filter_df) if not filter_df.empty else pd.DataFrame()
        valid_ts = ts_df["_ts"].dropna() if "_ts" in ts_df else pd.Series(dtype="datetime64[ns]")
        if not valid_ts.empty:
            date_bounds = st.date_input(
                "UK-REFIT date range",
                (valid_ts.min().date(), valid_ts.max().date()),
                min_value=valid_ts.min().date(),
                max_value=valid_ts.max().date(),
                key="uk_date_range",
            )
        else:
            date_bounds = ()
            st.caption("No parseable dates in current local files.")
        chosen_hours = st.multiselect("Hour of day", list(range(24)), default=list(range(24)), key="uk_hours")
        uk_limit = st.number_input("UK-REFIT table row limit", 100, 100000, 5000, 500, key="uk_limit")

    # If a local daily CSV is absent and the HDFS daily output is Parquet,
    # derive a clearly-labelled daily mean-power summary from the available
    # complete hourly CSV. This avoids treating Parquet bytes as CSV.
    if daily.empty and not hourly.empty and {"House", "Date", "Appliance", "MeanW"}.issubset(hourly.columns):
        _hourly_copy = hourly.copy()
        _hourly_copy["MeanW"] = pd.to_numeric(_hourly_copy["MeanW"], errors="coerce")
        daily = (
            _hourly_copy.dropna(subset=["MeanW"])
            .groupby(["House", "Date", "Appliance"], as_index=False)
            .agg(MeanW=("MeanW", "mean"), MinHourlyMeanW=("MeanW", "min"), MaxHourlyMeanW=("MeanW", "max"), HoursObserved=("MeanW", "count"))
        )
        daily.attrs["derived_from_hourly"] = True

    fp = apply_uk_filters(pred, chosen_houses, chosen_apps, date_bounds, chosen_hours)
    fm = apply_uk_filters(metrics, chosen_houses, chosen_apps, date_bounds, chosen_hours)
    fh = apply_uk_filters(hourly, chosen_houses, chosen_apps, date_bounds, chosen_hours)
    # Hour filters are not meaningful for daily summaries, so apply date/house/appliance only.
    fd = apply_uk_filters(daily, chosen_houses, chosen_apps, date_bounds, list(range(24)))

    if page == "Overview":
        st.header("UK-REFIT Overview")
        c = st.columns(4)
        c[0].metric("Filtered test predictions", f"{len(fp):,}")
        c[1].metric("Hourly summary rows", f"{len(fh):,}")
        c[2].metric("Daily summary rows", f"{len(fd):,}")
        c[3].metric("Appliance channels", f"{fm[app_col].nunique():,}" if not fm.empty and app_col and app_col in fm else "9 (expected)")
        if not fp.empty and {"label", "prediction"}.issubset(fp.columns):
            actual = pd.to_numeric(fp["label"], errors="coerce")
            forecast = pd.to_numeric(fp["prediction"], errors="coerce")
            valid = actual.notna() & forecast.notna()
            mae = (actual[valid] - forecast[valid]).abs().mean()
            rmse = np.sqrt(((actual[valid] - forecast[valid])**2).mean())
            c = st.columns(3)
            c[0].metric("Filtered MAE", f"{mae:.3f} W")
            c[1].metric("Filtered RMSE", f"{rmse:.3f} W")
            c[2].metric("Valid pairs", f"{int(valid.sum()):,}")
        st.subheader("Appliance model comparison")
        if not fm.empty and {"Appliance", "MAE_W", "RMSE_W"}.issubset(fm.columns):
            melted = fm.melt(id_vars="Appliance", value_vars=["MAE_W", "RMSE_W"], var_name="Metric", value_name="Error_W")
            plot(style(px.bar(melted, x="Appliance", y="Error_W", color="Metric", barmode="group"), "Saved clipped-prediction metrics by channel", 420, "Appliance channel", "Error (W)"), "uk_overview_metrics", "ukrefit_metrics")
        st.subheader("Known evaluation context")
        st.markdown("- Lagged GBT test RMSE: **68.642 W**; persistence RMSE: **79.066 W**.\n- Lagged GBT test MAE: **18.330 W**; persistence MAE: **16.967 W**.\n- The GBT reduces RMSE but does not beat persistence on MAE.")
        flow(["UK-REFIT raw HDFS", "Spark preprocessing", "Hourly / daily summaries", "Lagged features", "GBT model", "Historical test predictions", "Dashboard"])

    elif page == "Consumption Analysis":
        st.header("Consumption Analysis")
        dataset_choice = st.radio("Summary", ["Hourly summary", "Daily summary", "Historical test predictions"], horizontal=True, key="uk_consumption_source")
        df = {"Hourly summary": fh, "Daily summary": fd, "Historical test predictions": fp}[dataset_choice]
        if df.empty:
            st.warning("No rows are available for this dataset and filter combination.")
        elif page == "Prediction vs Actual":
                st.header("Actual vs Predicted Power")
                st.caption(
                    "Explore every available saved test prediction in chronological "
                    "order. Actual and predicted power are measured in watts."
                )    
        else:
            nums = [c for c in df.select_dtypes(include="number").columns if not c.startswith("_")]
            if nums:
                measure = st.selectbox("Numeric measure", nums, key="uk_measure")
                values = pd.to_numeric(df[measure], errors="coerce").dropna()
                c = st.columns(4)
                c[0].metric("Rows", f"{len(df):,}")
                c[1].metric("Mean", f"{values.mean():,.3f}")
                c[2].metric("Median", f"{values.median():,.3f}")
                c[3].metric("Maximum", f"{values.max():,.3f}")
                ts = uk_timestamp(df)
                if ts["_ts"].notna().any():
                    ts[measure] = pd.to_numeric(ts[measure], errors="coerce")
                    trend = ts.dropna(subset=["_ts", measure]).groupby(pd.Grouper(key="_ts", freq="D"))[measure].mean().reset_index()
                    plot(time_controls(style(px.line(trend, x="_ts", y=measure), f"{measure} over time", 450, "Date", measure, False)), "uk_consumption_trend", "ukrefit_consumption")
                if "House" in df and df["House"].nunique() > 1:
                    by_house = df.assign(_v=pd.to_numeric(df[measure], errors="coerce")).groupby("House", as_index=False)["_v"].mean()
                    plot(style(px.bar(by_house, x="House", y="_v"), f"Average {measure} by house", 400, "House", measure, False), "uk_consumption_house", "ukrefit_house_consumption")
                if "Appliance" in df and df["Appliance"].nunique() > 1:
                    by_app = df.assign(_v=pd.to_numeric(df[measure], errors="coerce")).groupby("Appliance", as_index=False)["_v"].mean()
                    plot(style(px.bar(by_app, x="Appliance", y="_v"), f"Average {measure} by appliance", 400, "Appliance", measure, False), "uk_consumption_app", "ukrefit_app_consumption")
            st.subheader("Filtered records")
            st.dataframe(df.drop(columns=["_ts"], errors="ignore").head(int(uk_limit)), use_container_width=True, hide_index=True)
        
        p = prepare_uk_predictions(pred)

        if p.empty:
            st.error(
                "The prediction file is empty or does not contain "
                "House, Date, Hour, Appliance, label and prediction."
            )
        else:
            # Use the prediction file's own available dates and IDs.
            # This avoids excluding valid test rows using summary-file dates.
            house_options = sorted(p["House"].unique())
            appliance_options = sorted(p["Appliance"].unique())

            view = st.radio(
                "Choose comparison",
                [
                    "One household — all appliances",
                    "One household — one appliance",
                    "One appliance — selected households",
                ],
                horizontal=False,
                key="uk_prediction_view",
            )

            if view in [
                "One household — all appliances",
                "One household — one appliance",
            ]:
                selected_house = st.selectbox(
                    "Household",
                    house_options,
                    key="uk_prediction_house",
                )
                p = p[p["House"] == selected_house]

            if view == "One household — one appliance":
                selected_appliance = st.selectbox(
                    "Appliance",
                    sorted(p["Appliance"].unique()),
                    key="uk_prediction_single_appliance",
                )
                p = p[p["Appliance"] == selected_appliance]

            elif view == "One appliance — selected households":
                selected_appliance = st.selectbox(
                    "Appliance",
                    appliance_options,
                    key="uk_prediction_multi_appliance",
                )
                selected_houses = st.multiselect(
                    "Households to compare",
                    house_options,
                    default=house_options[:min(5, len(house_options))],
                    key="uk_prediction_compare_houses",
                )
                p = p[
                    (p["Appliance"] == selected_appliance)
                    & p["House"].isin(selected_houses)
                ]

            if p.empty:
                st.warning("No prediction rows match these selections.")
            else:
                min_date = p["_ts"].min().date()
                max_date = p["_ts"].max().date()

                date_selection = st.date_input(
                    "Prediction date range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="uk_prediction_dates",
                )

                if isinstance(date_selection, (tuple, list)) and len(date_selection) == 2:
                    start_date, end_date = date_selection
                    p = p[
                        (p["_ts"] >= pd.Timestamp(start_date))
                        & (p["_ts"] < pd.Timestamp(end_date) + pd.Timedelta(days=1))
                    ]

                if p.empty:
                    st.warning("No prediction rows are available for this date range.")
                else:
                    # Metrics use ALL filtered rows, not the plotted subset.
                    error = p["prediction"] - p["label"]
                    mae = error.abs().mean()
                    rmse = np.sqrt((error ** 2).mean())
                    bias = error.mean()

                    c = st.columns(4)
                    c[0].metric("Prediction rows", f"{len(p):,}")
                    c[1].metric("MAE", f"{mae:.2f} W")
                    c[2].metric("RMSE", f"{rmse:.2f} W")
                    c[3].metric("Mean error", f"{bias:.2f} W")

                    # Bound browser rendering without silently dropping
                    # early or late dates: choose a uniform chronological sample.
                    max_points = st.select_slider(
                        "Maximum plotted timestamps per line",
                        options=[500, 1000, 2500, 5000, 10000, 25000],
                        value=5000,
                        key="uk_prediction_plot_limit",
                    )

                    fig = go.Figure()

                    if view == "One appliance — selected households":
                        groups = list(p.groupby("House", sort=True))
                    elif view == "One household — all appliances":
                        groups = list(p.groupby("Appliance", sort=True))
                    else:
                        groups = [(selected_appliance, p)]

                    for group_name, group in groups:
                        group = group.sort_values("_ts")

                        if len(group) > max_points:
                            indexes = np.linspace(
                                0, len(group) - 1, max_points
                            ).astype(int)
                            plotted = group.iloc[indexes]
                        else:
                            plotted = group

                        label = str(group_name)

                        fig.add_trace(go.Scatter(
                            x=plotted["_ts"],
                            y=plotted["label"],
                            mode="lines",
                            name=f"{label} — Actual",
                            legendgroup=label,
                            line=dict(width=1.5),
                        ))
                        fig.add_trace(go.Scatter(
                            x=plotted["_ts"],
                            y=plotted["prediction"],
                            mode="lines",
                            name=f"{label} — Predicted",
                            legendgroup=label,
                            line=dict(width=1.5, dash="dash"),
                        ))

                    plot(
                        time_controls(style(
                            fig,
                            "Actual vs Predicted Power — Chronological Test Period",
                            560,
                            "Date and time",
                            "Power (W)",
                        )),
                        "uk_full_prediction_comparison",
                        "ukrefit_full_prediction_comparison",
                    )

                    st.caption(
                        f"Metrics use all {len(p):,} filtered prediction rows. "
                        "The plotted lines may use a uniformly spaced subset "
                        "to keep the chart responsive."
                    )

                    st.subheader("Prediction error over time")
                    error_fig = go.Figure()
                    error_fig.add_trace(go.Scatter(
                        x=p["_ts"],
                        y=p["ErrorW"],
                        mode="lines",
                        name="Prediction − Actual",
                    ))
                    error_fig.add_hline(
                        y=0,
                        line_dash="dash",
                        line_color=RED,
                    )
                    plot(
                        style(
                            error_fig,
                            "Prediction Error (W)",
                            380,
                            "Date and time",
                            "Error (W)",
                            False,
                        ),
                        "uk_prediction_error_timeline",
                        "ukrefit_prediction_error",
                    )

                    st.subheader("Filtered prediction records")
                    display_cols = [
                        "House", "Date", "Hour", "Appliance",
                        "label", "prediction", "ErrorW", "AbsoluteErrorW",
                    ]
                    table(
                        p[display_cols].head(int(uk_limit)),
                        "ukrefit_prediction_comparison",
                        "uk_prediction_comparison_download",
                    )        

    
    elif page == "Anomaly Detection":
        st.header("UK-REFIT Anomaly Detection")

        anomaly_type = st.radio(
            "Anomaly type",
            ["Consumption anomalies", "Prediction-error anomalies"],
            horizontal=True,
            key="uk_anomaly_type"
        )

        resolution = st.radio(
            "Resolution",
            ["Hourly", "Daily"],
            horizontal=True,
            key="uk_anom_res"
        )

        scope = st.radio(
            "Analysis scope",
            [
                "One household — all appliances",
                "One household — one appliance",
                "One appliance — multiple households"
            ],
            key="uk_anomaly_scope"
        )

        if anomaly_type == "Consumption anomalies":
            source = fh if resolution == "Hourly" else fd

            if source.empty:
                st.warning("No consumption summary data is available.")
            else:
                df = source.copy()

                value_options = [
                    c for c in [
                        "MeanPowerW", "AveragePowerW", "AvgPowerW",
                        "MeanW", "PowerW", "MeanPower",
                        "EnergyWh", "Energy_kWh", "Energy"
                    ]
                    if c in df.columns
                ]

                if not value_options:
                    value_options = [
                        c for c in df.select_dtypes(include="number").columns
                        if c.lower() not in {
                            "house", "hour", "day", "month", "year"
                        }
                    ]

                if not value_options:
                    st.error("No numeric consumption measure was found.")
                else:
                    value_col = st.selectbox(
                        "Consumption measure",
                        value_options,
                        key="uk_anomaly_measure"
                    )

                    if "House" not in df.columns or "Appliance" not in df.columns:
                        st.error("The summary must contain House and Appliance columns.")
                    else:
                        houses = sorted(
                            df["House"].dropna().astype(str).unique().tolist()
                        )
                        appliances = sorted(
                            df["Appliance"].dropna().astype(str).unique().tolist()
                        )

                        if scope in [
                            "One household — all appliances",
                            "One household — one appliance"
                        ]:
                            selected_house = st.selectbox(
                                "Household",
                                houses,
                                key="uk_anomaly_house"
                            )
                            df = df[
                                df["House"].astype(str) == selected_house
                            ]

                        if scope in [
                            "One household — one appliance",
                            "One appliance — multiple households"
                        ]:
                            available_appliances = sorted(
                                df["Appliance"].dropna().astype(str).unique().tolist()
                            )
                            if not available_appliances:
                                st.warning("No appliances are available for this selection.")
                                st.stop()

                            selected_appliance = st.selectbox(
                                "Appliance",
                                available_appliances,
                                key="uk_anomaly_appliance"
                            )
                            df = df[
                                df["Appliance"].astype(str) == selected_appliance
                            ]

                        if df.empty:
                            st.warning("No rows match the selected household and appliance.")
                        else:
                            # Compare each appliance separately when examining
                            # all appliances in one household.
                            if scope == "One household — all appliances":
                                groups = ["Appliance"]
                            elif scope == "One appliance — multiple households":
                                groups = ["House"]
                            else:
                                groups = []

                            anomalies = calculate_anomalies(
                                df, value_col, groups, resolution
                            )

                            severity_options = [
                                "Low", "Medium", "High", "Critical"
                            ]
                            selected_severity = st.multiselect(
                                "Severity",
                                severity_options,
                                default=severity_options,
                                key="uk_anomaly_severity"
                            )

                            if "_severity" in anomalies.columns:
                                anomalies = anomalies[
                                    anomalies["_severity"].isin(selected_severity)
                                ]

                            c1, c2, c3 = st.columns(3)
                            c1.metric("Flagged rows", f"{len(anomalies):,}")
                            c2.metric(
                                "High / critical",
                                f"{int(anomalies['_severity'].isin(['High', 'Critical']).sum()):,}"
                                if not anomalies.empty and "_severity" in anomalies
                                else "0"
                            )
                            c3.metric("Measure", value_col)

                            if not anomalies.empty:
                                anomalies["_ts"] = uk_timestamp(anomalies)["_ts"]

                                timeline_data = anomalies.dropna(subset=["_ts"])

                                if not timeline_data.empty:
                                    timeline = (
                                        timeline_data
                                        .assign(Date=timeline_data["_ts"].dt.floor("D"))
                                        .groupby("Date")
                                        .size()
                                        .reset_index(name="Flagged rows")
                                    )

                                    plot(
                                        style(
                                            px.line(
                                                timeline,
                                                x="Date",
                                                y="Flagged rows",
                                                markers=True
                                            ),
                                            "Consumption anomalies over time",
                                            400,
                                            "Date",
                                            "Flagged rows",
                                            False
                                        ),
                                        "uk_consumption_anomaly_timeline",
                                        "ukrefit_consumption_anomalies"
                                    )

                                st.dataframe(
                                    anomalies.head(int(uk_limit)),
                                    use_container_width=True,
                                    hide_index=True
                                )

                                st.download_button(
                                    "Download consumption anomaly report",
                                    anomalies.to_csv(index=False).encode("utf-8-sig"),
                                    f"ukrefit_{resolution.lower()}_consumption_anomalies.csv",
                                    "text/csv"
                                )
                            else:
                                st.success("No anomalies were flagged for these selections.")

        else:
            if fp.empty or not {"label", "prediction"}.issubset(fp.columns):
                st.warning("The saved predictions file is missing or incompatible.")
            else:
                df = fp.copy()

                df["label"] = pd.to_numeric(df["label"], errors="coerce")
                df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")

                df = df.dropna(subset=["label", "prediction"])
                df["ErrorW"] = df["label"] - df["prediction"]
                df["AbsoluteErrorW"] = df["ErrorW"].abs()

                if "House" not in df.columns or "Appliance" not in df.columns:
                    st.error("The predictions file must contain House and Appliance columns.")
                else:
                    houses = sorted(
                        df["House"].dropna().astype(str).unique().tolist()
                    )
                    appliances = sorted(
                        df["Appliance"].dropna().astype(str).unique().tolist()
                    )

                    if scope in [
                        "One household — all appliances",
                        "One household — one appliance"
                    ]:
                        selected_house = st.selectbox(
                            "Household",
                            houses,
                            key="uk_error_house"
                        )
                        df = df[df["House"].astype(str) == selected_house]

                    if scope in [
                        "One household — one appliance",
                        "One appliance — multiple households"
                    ]:
                        available_appliances = sorted(
                            df["Appliance"].dropna().astype(str).unique().tolist()
                        )
                        if not available_appliances:
                            st.warning("No appliances are available for this selection.")
                            st.stop()

                        selected_appliance = st.selectbox(
                            "Appliance",
                            available_appliances,
                            key="uk_error_appliance"
                        )
                        df = df[
                            df["Appliance"].astype(str) == selected_appliance
                        ]

                    if df.empty:
                        st.warning("No prediction rows match the selected scope.")
                    else:
                        # Find unusually large absolute prediction errors.
                        if scope == "One household — all appliances":
                            groups = ["Appliance"]
                        elif scope == "One appliance — multiple households":
                            groups = ["House"]
                        else:
                            groups = []

                        anomalies = calculate_anomalies(
                            df,
                            "AbsoluteErrorW",
                            groups,
                            "Hourly"
                        )

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Flagged prediction errors", f"{len(anomalies):,}")
                        c2.metric(
                            "Mean absolute error",
                            f"{df['AbsoluteErrorW'].mean():.3f} W"
                        )
                        c3.metric(
                            "Maximum absolute error",
                            f"{df['AbsoluteErrorW'].max():.3f} W"
                        )

                        if not anomalies.empty:
                            anomalies["_ts"] = uk_timestamp(anomalies)["_ts"]

                            timeline_data = (
                                anomalies.dropna(subset=["_ts"])
                                .sort_values("_ts")
                            )

                            if not timeline_data.empty:
                                plot_df = timeline_data

                                # Keep plots responsive for large files.
                                if len(plot_df) > int(uk_limit):
                                    indices = np.linspace(
                                        0,
                                        len(plot_df) - 1,
                                        int(uk_limit),
                                        dtype=int
                                    )
                                    plot_df = plot_df.iloc[indices]

                                fig = go.Figure()
                                fig.add_scatter(
                                    x=plot_df["_ts"],
                                    y=plot_df["AbsoluteErrorW"],
                                    mode="markers",
                                    name="Absolute prediction error"
                                )

                                plot(
                                    time_controls(
                                        style(
                                            fig,
                                            "Unusual prediction errors over time",
                                            420,
                                            "Timestamp",
                                            "Absolute error (W)"
                                        )
                                    ),
                                    "uk_prediction_error_anomaly_timeline",
                                    "ukrefit_prediction_error_anomalies"
                                )

                            st.dataframe(
                                anomalies.head(int(uk_limit)),
                                use_container_width=True,
                                hide_index=True
                            )

                            st.download_button(
                                "Download prediction-error anomaly report",
                                anomalies.to_csv(index=False).encode("utf-8-sig"),
                                "ukrefit_prediction_error_anomalies.csv",
                                "text/csv"
                            )
                        else:
                            st.success("No unusual prediction errors were flagged.")


    elif page == "Forecast Evaluation":
        st.header("Forecast Evaluation")
        st.caption("This evaluates saved historical test predictions. It does not create a future forecast.")
        if fp.empty or not {"label", "prediction"}.issubset(fp.columns):
            st.warning("Prediction file is missing or incompatible.")
        else:
            actual = pd.to_numeric(fp["label"], errors="coerce")
            forecast = pd.to_numeric(fp["prediction"], errors="coerce")
            valid = actual.notna() & forecast.notna()
            eval_df = fp.loc[valid].copy()
            error = actual[valid] - forecast[valid]
            mae = error.abs().mean()
            rmse = np.sqrt((error**2).mean())
            bias = (forecast[valid] - actual[valid]).mean()
            c = st.columns(3)
            c[0].metric("MAE", f"{mae:.3f} W")
            c[1].metric("RMSE", f"{rmse:.3f} W")
            c[2].metric("Bias (predicted − actual)", f"{bias:.3f} W")
            plot(style(px.scatter(eval_df.head(int(uk_limit)), x="label", y="prediction", opacity=.45), "Actual vs predicted power", 450, "Actual (W)", "Predicted (W)", False), "uk_actual_pred_scatter", "ukrefit_actual_vs_predicted")
            if "Date" in eval_df and "Hour" in eval_df:
                eval_df["_ts"] = pd.to_datetime(eval_df["Date"].astype(str) + " " + eval_df["Hour"].astype(str) + ":00:00", errors="coerce")
                sample = eval_df.dropna(subset=["_ts"]).sort_values("_ts").head(int(uk_limit))
                if not sample.empty:
                    fig = go.Figure()
                    fig.add_scatter(x=sample["_ts"], y=sample["label"], name="Actual", line=dict(color=PRIMARY))
                    fig.add_scatter(x=sample["_ts"], y=sample["prediction"], name="Predicted", line=dict(color=AMBER))
                    plot(time_controls(style(fig, "Actual and predicted power (display-limited sample)", 460, "Timestamp", "Power (W)")), "uk_actual_pred_timeline", "ukrefit_actual_pred_timeline")
            st.download_button("Export filtered test predictions", eval_df.to_csv(index=False).encode("utf-8-sig"), "ukrefit_filtered_test_predictions.csv", "text/csv")
        st.subheader("Per-appliance metrics")
        if not fm.empty:
            st.dataframe(fm, use_container_width=True, hide_index=True)
        st.info("Saved GBT test metrics: MAE 18.330 W, RMSE 68.642 W. Persistence baseline: MAE 16.967 W, RMSE 79.066 W. Compare both metrics: GBT improves RMSE but not MAE.")

    elif page == "Data Explorer":
        st.header("UK-REFIT Data Explorer")
        options = {"Clipped predictions": fp, "Appliance metrics": fm, "Hourly summary": fh, "Daily summary": fd}
        name = st.selectbox("Dataset", list(options), key="uk_explorer_dataset")
        df = options[name]
        st.write(f"Rows available: **{len(df):,}**")
        if not df.empty:
            st.dataframe(df.drop(columns=["_ts"], errors="ignore").head(int(uk_limit)), use_container_width=True, hide_index=True)
            st.download_button("Download displayed rows", df.head(int(uk_limit)).to_csv(index=False).encode("utf-8-sig"), f"{name.lower().replace(' ', '_')}.csv", "text/csv")

    elif page == "Project & Data Quality":
        st.header("UK-REFIT Project & Data Quality")
        st.markdown(f"""
        - Raw HDFS location: `{UKREFIT_HDFS_BASE}`
        - Processed summaries: `{UKREFIT_HDFS_PROCESSED}`
        - Improved forecast outputs: `{UKREFIT_HDFS_RESULTS}`
        - Raw dataset: approximately 114 million readings across 19 house files (House 14 absent in the supplied inventory).
        - Channels are named generically `Appliance1`–`Appliance9`; physical appliance identities are not inferred.
        - Existing preprocessing capped intervals at 300 seconds, so long gaps can undercount energy.
        """)
        st.subheader("Local file availability")
        st.dataframe(pd.DataFrame([
            {"Dataset": "Clipped predictions", "Path": str(UKREFIT_PREDICTIONS_PATH), "Exists": UKREFIT_PREDICTIONS_PATH.exists(), "Rows loaded": len(pred)},
            {"Dataset": "Appliance metrics", "Path": str(UKREFIT_APPLIANCE_METRICS_PATH), "Exists": UKREFIT_APPLIANCE_METRICS_PATH.exists(), "Rows loaded": len(metrics)},
            {"Dataset": "Hourly summary", "Path": str(UKREFIT_HOURLY_PATH), "Exists": UKREFIT_HOURLY_PATH.exists(), "Rows loaded": len(hourly)},
            {"Dataset": "Daily summary", "Path": str(UKREFIT_DAILY_PATH), "Exists": UKREFIT_DAILY_PATH.exists(), "Rows loaded": len(daily)},
        ]), use_container_width=True, hide_index=True)
        st.markdown("""
        **Interpretation:** The saved prediction file represents a historical test period. The current anomaly page applies robust statistics to the loaded summary and does not prove a hardware fault. For large-scale reprocessing and genuine future forecasts, use Spark jobs; this dashboard does not load the 114-million-row raw source into memory.
        """)


# -------------------------------- Dispatch ------------------------------------

if dataset == "London Smart Meter":
    render_london_workspace()
else:
    render_ukrefit_workspace()

st.divider()
st.caption("Smart Meter Analytics | London Smart Meter + UK-REFIT | Independent data sources and navigation")
