from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

# Separate UK-REFIT dashboard. This file does not import or modify the London app.
PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "output"
PREDICTIONS_FILE = OUTPUT_DIR / "clipped_predictions.csv"
METRICS_FILE = OUTPUT_DIR / "clipped_appliance_metrics.csv"

st.set_page_config(
    page_title="UK-REFIT Forecast Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ UK-REFIT Appliance Forecast Dashboard")
st.caption(
    "Saved clipped predictions from the improved Spark model. "
    "Power values are in watts (W); appliance channels are shown by their dataset labels."
)

@st.cache_data(show_spinner="Loading saved UK-REFIT results…")
def load_data(predictions_path: str, metrics_path: str):
    predictions = pd.read_csv(predictions_path)
    metrics = pd.read_csv(metrics_path)

    required_prediction_columns = {
        "House", "Date", "Hour", "Appliance", "label", "prediction"
    }
    required_metric_columns = {
        "Appliance", "Rows", "MeanActualW", "MeanPredictedW", "MAE_W", "RMSE_W"
    }

    missing_predictions = required_prediction_columns - set(predictions.columns)
    missing_metrics = required_metric_columns - set(metrics.columns)
    if missing_predictions:
        raise ValueError(f"Prediction CSV is missing columns: {sorted(missing_predictions)}")
    if missing_metrics:
        raise ValueError(f"Metrics CSV is missing columns: {sorted(missing_metrics)}")

    predictions["Date"] = pd.to_datetime(predictions["Date"], errors="coerce")
    for column in ["House", "Hour", "label", "prediction"]:
        predictions[column] = pd.to_numeric(predictions[column], errors="coerce")
    predictions = predictions.dropna(
        subset=["Date", "House", "Hour", "Appliance", "label", "prediction"]
    )
    predictions["House"] = predictions["House"].astype(int)
    predictions["Hour"] = predictions["Hour"].astype(int)
    predictions["AbsoluteError"] = (predictions["prediction"] - predictions["label"]).abs()
    predictions["SquaredError"] = (predictions["prediction"] - predictions["label"]) ** 2

    metrics["Rows"] = pd.to_numeric(metrics["Rows"], errors="coerce")
    for column in [
        "MeanActualW", "MeanPredictedW", "MAE_W", "RMSE_W",
        "MinPredictedW", "MaxPredictedW"
    ]:
        if column in metrics.columns:
            metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    return predictions, metrics


if not PREDICTIONS_FILE.exists() or not METRICS_FILE.exists():
    st.error("One or both input CSV files are missing.")
    st.write("Expected files:")
    st.code(f"{PREDICTIONS_FILE}\n{METRICS_FILE}")
    st.info("Copy the two exported CSV files into the UK-REFIT output folder, then refresh this page.")
    st.stop()

try:
    predictions, appliance_metrics = load_data(str(PREDICTIONS_FILE), str(METRICS_FILE))
except Exception as exc:
    st.error(f"Could not load dashboard data: {exc}")
    st.stop()

if predictions.empty:
    st.error("The prediction CSV contains no usable records.")
    st.stop()

min_date = predictions["Date"].min().date()
max_date = predictions["Date"].max().date()

with st.sidebar:
    st.header("Filters")
    available_houses = sorted(predictions["House"].unique().tolist())
    selected_houses = st.multiselect(
        "House", available_houses, default=available_houses
    )
    available_appliances = sorted(predictions["Appliance"].unique().tolist())
    selected_appliances = st.multiselect(
        "Appliance channel", available_appliances, default=available_appliances
    )
    selected_dates = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    st.caption("Filters affect the prediction charts and filtered metrics.")

filtered = predictions[
    predictions["House"].isin(selected_houses)
    & predictions["Appliance"].isin(selected_appliances)
].copy()

if isinstance(selected_dates, (tuple, list)):
    if len(selected_dates) == 2:
        start_date, end_date = selected_dates
    elif len(selected_dates) == 1:
        start_date = end_date = selected_dates[0]
    else:
        start_date, end_date = min_date, max_date
else:
    start_date = end_date = selected_dates

filtered = filtered[
    filtered["Date"].dt.date.between(start_date, end_date)
].copy()

if filtered.empty:
    st.warning("No records match the selected filters. Choose a wider date range or more houses/appliances.")
    st.stop()

filtered_mae = filtered["AbsoluteError"].mean()
filtered_rmse = (filtered["SquaredError"].mean()) ** 0.5
mean_actual = filtered["label"].mean()
mean_predicted = filtered["prediction"].mean()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Filtered records", f"{len(filtered):,}")
m2.metric("MAE", f"{filtered_mae:.2f} W")
m3.metric("RMSE", f"{filtered_rmse:.2f} W")
m4.metric("Mean actual / predicted", f"{mean_actual:.2f} / {mean_predicted:.2f} W")

st.caption(
    "Metrics above are recalculated over the selected rows. They are not a claim that the model "
    "outperforms every baseline: in the recorded overall test results, clipped GBT RMSE is "
    "68.63 W versus 79.07 W for persistence, while clipped GBT MAE is 18.23 W versus "
    "16.97 W for persistence."
)

tab_overview, tab_appliances, tab_predictions, tab_data = st.tabs(
    ["Overview", "Appliance metrics", "Actual vs predicted", "Data table"]
)

with tab_overview:
    st.subheader("Actual and predicted power distribution")
    hist_data = filtered[["label", "prediction"]].melt(
        var_name="Series", value_name="Power (W)"
    )
    hist_data["Series"] = hist_data["Series"].map(
        {"label": "Actual", "prediction": "Clipped prediction"}
    )
    fig = px.histogram(
        hist_data,
        x="Power (W)",
        color="Series",
        barmode="overlay",
        opacity=0.65,
        nbins=60,
        title="Distribution of actual and predicted power",
    )
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Daily mean power")
    daily = (
        filtered.groupby("Date", as_index=False)
        .agg(ActualW=("label", "mean"), PredictedW=("prediction", "mean"))
        .sort_values("Date")
    )
    daily_long = daily.melt("Date", var_name="Series", value_name="Mean power (W)")
    daily_long["Series"] = daily_long["Series"].map(
        {"ActualW": "Actual", "PredictedW": "Clipped prediction"}
    )
    fig_daily = px.line(
        daily_long,
        x="Date",
        y="Mean power (W)",
        color="Series",
        markers=True,
        title="Daily mean actual versus predicted power",
    )
    st.plotly_chart(fig_daily, use_container_width=True)

with tab_appliances:
    st.subheader("Saved appliance-wise test metrics")
    metrics_view = appliance_metrics.sort_values("MAE_W")
    fig_mae = px.bar(
        metrics_view,
        x="Appliance",
        y="MAE_W",
        title="MAE by appliance channel (lower is better)",
        labels={"MAE_W": "MAE (W)", "Appliance": "Appliance channel"},
        text_auto=".2f",
    )
    st.plotly_chart(fig_mae, use_container_width=True)

    fig_rmse = px.bar(
        metrics_view.sort_values("RMSE_W"),
        x="Appliance",
        y="RMSE_W",
        title="RMSE by appliance channel (lower is better)",
        labels={"RMSE_W": "RMSE (W)", "Appliance": "Appliance channel"},
        text_auto=".2f",
    )
    st.plotly_chart(fig_rmse, use_container_width=True)

    st.dataframe(
        appliance_metrics.sort_values("Appliance").round(3),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Appliance1–Appliance9 are generic channel labels from the dataset. "
        "No real-world appliance names are inferred."
    )

with tab_predictions:
    st.subheader("Actual versus clipped predicted power")
    plot_df = filtered.copy()
    if len(plot_df) > 15000:
        plot_df = plot_df.sample(n=15000, random_state=42)
        st.info("Showing a reproducible 15,000-row sample for chart responsiveness; metrics use all filtered rows.")

    fig_scatter = px.scatter(
        plot_df,
        x="label",
        y="prediction",
        color="Appliance",
        hover_data=["House", "Date", "Hour"],
        opacity=0.45,
        title="Actual power versus clipped prediction",
        labels={
            "label": "Actual power (W)",
            "prediction": "Clipped predicted power (W)",
            "Appliance": "Appliance channel",
        },
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("#### Time series for a selected house and appliance")
    series_house = st.selectbox("House for time series", sorted(filtered["House"].unique()))
    series_appliance = st.selectbox(
        "Appliance channel for time series",
        sorted(filtered.loc[filtered["House"] == series_house, "Appliance"].unique()),
    )
    series_df = filtered[
        (filtered["House"] == series_house)
        & (filtered["Appliance"] == series_appliance)
    ].copy()
    series_df["Timestamp"] = (
        series_df["Date"] + pd.to_timedelta(series_df["Hour"], unit="h")
    )
    series_df = series_df.sort_values("Timestamp")
    if len(series_df) > 5000:
        series_df = series_df.tail(5000)
        st.info("Showing the latest 5,000 matching observations in the time-series chart.")
    series_long = series_df.melt(
        id_vars=["Timestamp"],
        value_vars=["label", "prediction"],
        var_name="Series",
        value_name="Power (W)",
    )
    series_long["Series"] = series_long["Series"].map(
        {"label": "Actual", "prediction": "Clipped prediction"}
    )
    fig_series = px.line(
        series_long,
        x="Timestamp",
        y="Power (W)",
        color="Series",
        title=f"{series_appliance} — House {series_house}",
    )
    st.plotly_chart(fig_series, use_container_width=True)

with tab_data:
    st.subheader("Filtered prediction records")
    st.dataframe(
        filtered[
            ["House", "Date", "Hour", "Appliance", "label", "prediction", "AbsoluteError"]
        ].sort_values(["Date", "House", "Appliance", "Hour"]),
        use_container_width=True,
        hide_index=True,
    )
    csv_bytes = filtered[
        ["House", "Date", "Hour", "Appliance", "label", "prediction"]
    ].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered predictions as CSV",
        data=csv_bytes,
        file_name="ukrefit_filtered_predictions.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    f"Source: saved clipped prediction CSV ({len(predictions):,} usable rows) and "
    f"appliance metrics CSV. Test period: {min_date} to {max_date}."
)
