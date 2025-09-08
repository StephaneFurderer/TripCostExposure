import os
import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from exposure_data import (
    load_policies,
    normalize_policies_df,
    load_folder_policies,
    aggregate_daily_exposure_by_departure_year,
    aggregate_exposure_by_departure_period,
    aggregate_traveling_by_period,
    aggregate_departures_by_period,
)


st.set_page_config(page_title="Trip Cost Exposure", layout="wide")

st.title("Daily Trip Cost Exposure by Departure Year")

@st.cache_data(show_spinner=False)
def get_data(use_dummy: bool, folder_path: str | None, erase_cache: bool) -> pd.DataFrame:
    if not use_dummy and folder_path:
        return load_folder_policies(folder_path, force_rebuild=False, erase_cache=erase_cache)
    # Fallback to dummy
    default_path = os.path.join(os.path.dirname(__file__), "_data", "policies.csv")
    return load_policies(default_path)


# Sidebar controls for data source
with st.sidebar:
    st.header("Data Source")
    use_dummy_data = st.checkbox("Use dummy data", value=True, help="Toggle to use packaged sample dataset")
    selected_folder = None
    erase_cache = False
    if not use_dummy_data:
        # Discover extract dates under _data
        data_root = os.path.join(os.path.dirname(__file__), "_data")
        try:
            subdirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        except FileNotFoundError:
            subdirs = []
        # Keep folders matching YYYY-MM-DD and that contain csv or parquet
        import re
        candidates = []
        for d in subdirs:
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
                full = os.path.join(data_root, d)
                has_files = any(name.endswith(".csv") or name.endswith(".parquet") for name in os.listdir(full))
                if has_files:
                    candidates.append(d)
        candidates = sorted(candidates, reverse=True)
        extract_date = st.selectbox("Extract date", options=candidates, index=0 if candidates else None)
        if extract_date:
            selected_folder = os.path.join(data_root, extract_date)
        # One-click erase parquet and reload
        if extract_date:
            cache_path = os.path.join(selected_folder, "combined.parquet")
            if st.button("Erase parquet and reload"):
                try:
                    if os.path.exists(cache_path):
                        os.remove(cache_path)
                finally:
                    st.rerun()

df = get_data(use_dummy_data, selected_folder, erase_cache)

group_by_segment = st.checkbox("Group by Segment", value=False)
period = st.selectbox("Time grain", options=["day", "week", "month"], index=0)
metric_mode = st.radio("Metric base", options=["Departures", "Traveling"], index=0, horizontal=True)
year_order_choice = st.selectbox("Departure Year order", options=["Ascending", "Descending"], index=0)
# Unified metric selector based on mode
if metric_mode == "Traveling":
    metric_options = ["volume", "maxTripCostExposure", "tripCostPerNightExposure"]
    default_idx = 1
else:
    metric_options = ["volume", "maxTripCostExposure", "avgTripCostPerNight"]
    default_idx = 1
selected_metric = st.selectbox("Metric to plot", options=metric_options, index=default_idx)

# Main display panel
## plot the trip cost exposure by departure year per day, week or month
if group_by_segment:
    segments = sorted(df["segment"].dropna().astype(str).unique().tolist())
    selected_segments = st.multiselect("Select segments", segments, default=segments)
    df_use = df[df["segment"].astype(str).isin(selected_segments)]
    if metric_mode == "Traveling":
        data = aggregate_traveling_by_period(
            df_use, period=period, additional_group_by="segment"
        )
        years = sorted(data["year"].unique().tolist())
        if year_order_choice == "Descending":
            years = list(reversed(years))
        data["year"] = pd.Categorical(data["year"], categories=years, ordered=True)
        ycol = selected_metric
        fig = px.line(
            data,
            x="x",
            y=ycol,
            color="year",
            facet_row="segment",
            color_discrete_sequence=px.colors.qualitative.Safe,
            category_orders={"year": years},
            labels={"x": f"{period.title()} (normalized)", ycol: ycol, "year": "Year", "segment": "Segment"},
        )
    else:
        data = aggregate_departures_by_period(df_use, period=period, additional_group_by="segment")
        years = sorted(data["year"].unique().tolist())
        if year_order_choice == "Descending":
            years = list(reversed(years))
        data["year"] = pd.Categorical(data["year"], categories=years, ordered=True)
        ycol = selected_metric
        fig = px.line(
            data,
            x="x",
            y=ycol,
            color="year",
            facet_row="segment",
            color_discrete_sequence=px.colors.qualitative.Safe,
            category_orders={"year": years},
            labels={"x": f"Departure {period.title()}", ycol: ycol, "year": "Year", "segment": "Segment"},
        )
else:
    if metric_mode == "Traveling":
        data = aggregate_traveling_by_period(df, period=period)
        years = sorted(data["year"].unique().tolist())
        if year_order_choice == "Descending":
            years = list(reversed(years))
        data["year"] = pd.Categorical(data["year"], categories=years, ordered=True)
        ycol = selected_metric
        fig = px.line(
            data,
            x="x",
            y=ycol,
            color="year",
            color_discrete_sequence=px.colors.qualitative.Safe,
            category_orders={"year": years},
            labels={"x": f"{period.title()} (normalized)", ycol: ycol, "year": "Year"},
        )
    else:
        data = aggregate_departures_by_period(df, period=period)
        years = sorted(data["year"].unique().tolist())
        if year_order_choice == "Descending":
            years = list(reversed(years))
        data["year"] = pd.Categorical(data["year"], categories=years, ordered=True)
        ycol = selected_metric
        fig = px.line(
            data,
            x="x",
            y=ycol,
            color="year",
            color_discrete_sequence=px.colors.qualitative.Safe,
            category_orders={"year": years},
            labels={"x": f"Departure {period.title()}", ycol: ycol, "year": "Year"},
        )

# Configure x-axis ticks per selected period
if period == "day":
    # Show month ticks on the normalized 2000 calendar
    tickvals = pd.to_datetime([f"2000-{m:02d}-01" for m in range(1, 13)])
    ticktext = [pd.Timestamp(2000, m, 1).strftime("%b") for m in range(1, 13)]
    fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)
elif period == "week":
    # Tick each 4 weeks across the year
    week_starts = pd.to_datetime("2000-01-03") + pd.to_timedelta(range(0, 52, 4), unit="W")
    tickvals = week_starts
    ticktext = [f"W{int(((d - pd.Timestamp('2000-01-03')).days)/7)+1}" for d in tickvals]
    fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)
else:  # month
    tickvals = pd.to_datetime([f"2000-{m:02d}-01" for m in range(1, 13)])
    ticktext = [pd.Timestamp(2000, m, 1).strftime("%b") for m in range(1, 13)]
    fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)

# Add holiday markers on the normalized axis (applies to day and week especially)
holidays = [
    (pd.Timestamp(2000, 9, 4), "Labor Day (approx)"),   # first Monday of Sep 2000 is 4th
    (pd.Timestamp(2000, 10, 9), "Columbus Day (approx)"), # second Monday of Oct 2000 is 9th
    (pd.Timestamp(2000, 12, 25), "Christmas"),
    (pd.Timestamp(2000, 1, 1), "New Year"),
]
for x_val, name in holidays:
    fig.add_vline(x=x_val, line_width=1, line_dash="dot", line_color="#888")
    fig.add_annotation(x=x_val, yref="paper", y=1.02, showarrow=False, text=name, xanchor="left", font=dict(size=10, color="#666"))

fig.update_layout(legend_title_text="Departure Year", hovermode="x unified")

st.plotly_chart(fig, use_container_width=True)


