import os
import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from exposure_data import (
    load_policies,
    normalize_policies_df,
    load_folder_policies,
    precompute_all_with_timing,
    aggregate_daily_exposure_by_departure_year,
    aggregate_exposure_by_departure_period,
    aggregate_traveling_by_period,
    aggregate_traveling_unique_by_period,
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
    
    # Country filter
    st.header("Country Filter")
    country_filter_options = ["US", "ROW", "null"]
    selected_countries = st.multiselect(
        "Select countries to include",
        options=country_filter_options,
        default=country_filter_options,
        help="US: United States, ROW: Rest of World, null: Missing country data"
    )
    
    # Region filter
    st.header("Region Filter")
    region_filter_options = ["Atlantic", "Florida", "Gulf", "Pacific", "Other"]
    selected_regions = st.multiselect(
        "Select regions to include",
        options=region_filter_options,
        default=region_filter_options,
        help="Coastal regions: Atlantic (Maine to Georgia), Florida, Gulf (Texas to Alabama), Pacific (Hawaii + California), Other (non-coastal)"
    )
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
        # Clean and rebuild all parquet artifacts
        if extract_date and selected_folder:
            if st.button("Rebuild all parquet (clean + precompute)"):
                with st.spinner("Cleaning parquet files..."):
                    for name in os.listdir(selected_folder):
                        if name.endswith(".parquet"):
                            try:
                                os.remove(os.path.join(selected_folder, name))
                            except Exception:
                                pass
                with st.spinner("Rebuilding combined.parquet from CSVs..."):
                    # Force rebuild combined and trigger precompute from loader
                    _ = load_folder_policies(selected_folder, force_rebuild=True, erase_cache=False)
                with st.spinner("Precomputing all aggregates (timed)..."):
                    report = precompute_all_with_timing(selected_folder)
                st.success("All parquet files rebuilt.")
                st.dataframe(report)

# UI Controls (needed before data loading)
group_by_segment = st.checkbox("Group by Segment", value=False)
# Default to week/all for fast initial load
period = st.selectbox("Time grain", options=["day", "week", "month"], index=1)
metric_mode = st.radio("Metric base", options=["Departures", "Traveling"], index=0, horizontal=True)
year_order_choice = st.selectbox("Departure Year order", options=["Ascending", "Descending"], index=0)

# Try to load precomputed aggregates first (for both dummy and real data)
precomputed_data = None
folder_path = None

# Determine folder path for precomputed data
if use_dummy_data:
    # For dummy data, look in _data folder
    folder_path = Path("_data")
else:
    # For real data, use selected folder
    if selected_folder:
        folder_path = Path(selected_folder)

if folder_path:
    # Map to aggregate file names (we only create by_segment files now)
    if metric_mode == "Traveling":
        if period == "day":
            agg_file = folder_path / "agg_travel_day_by_segment.parquet"
        else:
            agg_file = folder_path / f"agg_travel_{period}_by_segment.parquet"
    else:
        agg_file = folder_path / f"agg_depart_{period}_by_segment.parquet"
    
    # Load precomputed data if available
    if agg_file.exists():
        precomputed_data = pd.read_parquet(agg_file)
        st.sidebar.success(f"✅ Loaded precomputed {period} {metric_mode.lower()} data")
        
        # If user doesn't want segment grouping, we need to aggregate the segment data
        if not group_by_segment and "segment" in precomputed_data.columns:
            # Aggregate across segments to get "all" data
            group_cols = ["year", "x", "country_class", "region_class"]
            agg_dict = {}
            if "volume" in precomputed_data.columns:
                agg_dict["volume"] = "sum"
            if "maxTripCostExposure" in precomputed_data.columns:
                agg_dict["maxTripCostExposure"] = "sum"
            if "tripCostPerNightExposure" in precomputed_data.columns:
                agg_dict["tripCostPerNightExposure"] = "sum"
            if "avgTripCostPerNight" in precomputed_data.columns:
                agg_dict["avgTripCostPerNight"] = "mean"
            
            precomputed_data = precomputed_data.groupby(group_cols, as_index=False).agg(agg_dict)
            st.sidebar.info("📊 Aggregated segment data to show 'all' view")
    else:
        st.sidebar.warning(f"⚠️ Precomputed {period} {metric_mode.lower()} data not found, will compute from raw data")

# Fallback to raw data if no precomputed data available
df = None
if precomputed_data is None:
    if use_dummy_data:
        # Load dummy data
        df = get_data(use_dummy_data, selected_folder, erase_cache)
    else:
        # Load real data from combined.parquet
        if selected_folder:
            combined_path = os.path.join(selected_folder, "combined.parquet")
            if os.path.exists(combined_path):
                df = pd.read_parquet(combined_path)
                # Ensure datetime types
                for c in ["dateDepart", "dateReturn", "dateApp"]:
                    if c in df.columns:
                        df[c] = pd.to_datetime(df[c], errors="coerce")
    
    # Apply country filter to raw data only if we're using raw data
    if df is not None and not df.empty:
        # Create country filter mask
        country_mask = pd.Series(False, index=df.index)
        
        if "US" in selected_countries:
            country_mask |= (df['Country'] == 'US')
        if "ROW" in selected_countries:
            # ROW includes all countries that are not US and not null
            country_mask |= ((df['Country'] != 'US') & (df['Country'].notna()))
        if "null" in selected_countries:
            country_mask |= df['Country'].isna()
        
        # Apply the filter
        df = df[country_mask].copy()
        
        # Show filter summary
        st.sidebar.caption(f"📊 Showing {len(df):,} policies after country filtering")

def filter_aggregate_data(data, selected_countries, selected_regions):
    """Filter aggregate data based on country and region selections"""
    if data is None or data.empty:
        return data
    
    # Apply country filter
    country_mask = pd.Series(False, index=data.index)
    if "US" in selected_countries:
        country_mask |= (data['country_class'] == 'US')
    if "ROW" in selected_countries:
        country_mask |= (data['country_class'] == 'ROW')
    if "null" in selected_countries:
        country_mask |= (data['country_class'] == 'null')
    
    # Apply region filter
    region_mask = pd.Series(False, index=data.index)
    for region in selected_regions:
        region_mask |= (data['region_class'] == region)
    
    # Combine filters
    combined_mask = country_mask & region_mask
    filtered_data = data[combined_mask].copy()
    
    return filtered_data

# Unified metric selector based on mode
if metric_mode == "Traveling":
    metric_options = ["volume", "maxTripCostExposure", "tripCostPerNightExposure", "avgTripCostPerNight"]
    default_idx = 1
else:
    metric_options = ["volume", "maxTripCostExposure", "tripCostPerNightExposure", "avgTripCostPerNight"]
    default_idx = 1
selected_metric = st.selectbox("Metric to plot", options=metric_options, index=default_idx)

# Main display panel
## plot the trip cost exposure by departure year per day, week or month

# Use precomputed data if available, otherwise compute from raw data
if precomputed_data is not None:
    # Use precomputed data and apply filters
    filtered_data = filter_aggregate_data(precomputed_data, selected_countries, selected_regions)
    
    # Show filter summary
    if not filtered_data.empty:
        st.sidebar.caption(f"📈 Showing {len(filtered_data):,} records after filtering")
    
    # Handle segment filtering if needed
    if group_by_segment and "segment" in filtered_data.columns:
        segments = sorted(filtered_data["segment"].dropna().astype(str).unique().tolist())
        selected_segments = st.multiselect("Select segments", segments, default=segments)
        filtered_data = filtered_data[filtered_data["segment"].astype(str).isin(selected_segments)]
    
    # Aggregate the filtered data by year and x (time period)
    # Group by year, x, and segment (if present)
    # Note: We don't include country_class and region_class in groupby because
    # we want to aggregate across all selected countries/regions
    group_cols = ["year", "x"]
    if group_by_segment and "segment" in filtered_data.columns:
        group_cols.append("segment")
    
    # Aggregate metrics
    agg_dict = {}
    if "volume" in filtered_data.columns:
        agg_dict["volume"] = "sum"
    if "maxTripCostExposure" in filtered_data.columns:
        agg_dict["maxTripCostExposure"] = "sum"
    if "tripCostPerNightExposure" in filtered_data.columns:
        agg_dict["tripCostPerNightExposure"] = "sum"
    if "avgTripCostPerNight" in filtered_data.columns:
        agg_dict["avgTripCostPerNight"] = "mean"
    
    data = filtered_data.groupby(group_cols, as_index=False).agg(agg_dict)
    
    years = sorted(data["year"].unique().tolist())
    if year_order_choice == "Descending":
        years = list(reversed(years))
    data["year"] = pd.Categorical(data["year"], categories=years, ordered=True)
    
else:
    # No precomputed data available - show error message
    st.error("❌ No precomputed data available. Please run the precomputation first:")
    st.code("python3 test_precompute.py _data --limit-rows 1000", language="bash")
    st.stop()

# Create the plot
ycol = selected_metric
if group_by_segment and "segment" in data.columns:
    fig = px.line(
        data,
        x="x",
        y=ycol,
        color="year",
        facet_row="segment",
        color_discrete_sequence=px.colors.qualitative.Safe,
        category_orders={"year": years},
        labels={"x": f"{period.title()} (normalized)" if metric_mode == "Traveling" else f"Departure {period.title()}", ycol: ycol, "year": "Year", "segment": "Segment"},
    )
else:
    fig = px.line(
        data,
        x="x",
        y=ycol,
        color="year",
        color_discrete_sequence=px.colors.qualitative.Safe,
        category_orders={"year": years},
        labels={"x": f"{period.title()} (normalized)" if metric_mode == "Traveling" else f"Departure {period.title()}", ycol: ycol, "year": "Year"},
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

# Add data table below the plot for debugging
st.markdown("---")
st.subheader("Data Table: Metric by ISO Week and Year")

if 'data' in locals() and not data.empty:
    # Create a pivot table for better visualization
    if period == "week":
        # For weeks, create a table with ISO week numbers
        data_table = data.copy()
        
        # Convert x to ISO week number
        if 'x' in data_table.columns:
            # x is normalized to 2000, so we need to extract the week number
            data_table['iso_week'] = data_table['x'].dt.isocalendar().week
            # Use the actual year column, not the normalized x year
            data_table['iso_year'] = data_table['year']
        
        # Create pivot table
        pivot_cols = ['iso_week', 'iso_year', ycol]
        if group_by_segment and 'segment' in data_table.columns:
            pivot_cols.append('segment')
        
        # Group by week and year, then pivot
        if group_by_segment and 'segment' in data_table.columns:
            # For segmented data, show each segment as separate columns
            pivot_data = data_table.groupby(['iso_week', 'iso_year', 'segment'])[ycol].sum().reset_index()
            pivot_table = pivot_data.pivot_table(
                index='iso_week', 
                columns=['iso_year', 'segment'], 
                values=ycol, 
                fill_value=0
            )
        else:
            # For non-segmented data
            pivot_data = data_table.groupby(['iso_week', 'iso_year'])[ycol].sum().reset_index()
            pivot_table = pivot_data.pivot_table(
                index='iso_week', 
                columns='iso_year', 
                values=ycol, 
                fill_value=0
            )
        
        # Format the table for better readability
        if not pivot_table.empty:
            # Round numeric values
            pivot_table = pivot_table.round(2)
            
            # Add total row
            pivot_table.loc['TOTAL'] = pivot_table.sum()
            
            st.dataframe(pivot_table, use_container_width=True)
            
            # Add week date reference table
            st.subheader("Week Date Reference")
            week_ref_data = []
            for year in sorted(data_table['iso_year'].unique()):
                for week in sorted(data_table[data_table['iso_year'] == year]['iso_week'].unique()):
                    # Calculate ISO week start and end dates
                    try:
                        # Get the first day of the ISO week
                        week_start = pd.to_datetime(f"{year}-W{week:02d}-1", format='%Y-W%W-%w')
                        week_end = week_start + pd.Timedelta(days=6)
                        week_ref_data.append({
                            'Year': year,
                            'Week': week,
                            'Start Date': week_start.strftime('%Y-%m-%d'),
                            'End Date': week_end.strftime('%Y-%m-%d'),
                            'Start Day': week_start.strftime('%A'),
                            'End Day': week_end.strftime('%A')
                        })
                    except:
                        # Fallback for edge cases
                        week_ref_data.append({
                            'Year': year,
                            'Week': week,
                            'Start Date': 'N/A',
                            'End Date': 'N/A',
                            'Start Day': 'N/A',
                            'End Day': 'N/A'
                        })
            
            if week_ref_data:
                week_ref_df = pd.DataFrame(week_ref_data)
                st.dataframe(week_ref_df, use_container_width=True)
            
            # Show summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(data_table))
            with col2:
                st.metric("Total Value", f"{pivot_table.loc['TOTAL'].sum():,.2f}")
            with col3:
                st.metric("Weeks Covered", len(pivot_table.index) - 1)  # -1 for TOTAL row
        else:
            st.info("No data available for the selected filters.")
    
    elif period == "month":
        # For months, show month names
        data_table = data.copy()
        
        if 'x' in data_table.columns:
            data_table['month'] = data_table['x'].dt.month
            data_table['month_name'] = data_table['x'].dt.strftime('%b')
            # Use the actual year column, not the normalized x year
            # data_table['year'] is already the real year from the data
        
        # Create pivot table
        if group_by_segment and 'segment' in data_table.columns:
            pivot_data = data_table.groupby(['month', 'month_name', 'year', 'segment'])[ycol].sum().reset_index()
            pivot_table = pivot_data.pivot_table(
                index=['month', 'month_name'], 
                columns=['year', 'segment'], 
                values=ycol, 
                fill_value=0
            )
        else:
            pivot_data = data_table.groupby(['month', 'month_name', 'year'])[ycol].sum().reset_index()
            pivot_table = pivot_data.pivot_table(
                index=['month', 'month_name'], 
                columns='year', 
                values=ycol, 
                fill_value=0
            )
        
        if not pivot_table.empty:
            pivot_table = pivot_table.round(2)
            pivot_table.loc['TOTAL'] = pivot_table.sum()
            st.dataframe(pivot_table, use_container_width=True)
        else:
            st.info("No data available for the selected filters.")
    
    else:  # day
        # For days, show a sample of the data
        st.dataframe(data.head(20), use_container_width=True)
        if len(data) > 20:
            st.caption(f"Showing first 20 rows of {len(data)} total records")

else:
    st.info("No data available to display in table.")

# Search policies traveling during a period
st.markdown("---")
st.subheader("Search policies traveling during a period")

def load_search_df() -> pd.DataFrame:
    # Use already loaded df for dummy mode; otherwise read combined.parquet from selected folder
    if df is not None:
        base = df.copy()
    else:
        if not selected_folder:
            st.info("Select an extract folder in the sidebar to search real data.")
            return pd.DataFrame()
        combined_path = os.path.join(selected_folder, "combined.parquet")
        if not os.path.exists(combined_path):
            st.warning("combined.parquet not found. Click 'Build all aggregates (timed)' or reload the extract.")
            return pd.DataFrame()
        base = pd.read_parquet(combined_path)
    # Ensure datetime types
    for c in ["dateDepart", "dateReturn", "dateApp"]:
        if c in base.columns:
            base[c] = pd.to_datetime(base[c], errors="coerce")
    return base

search_df = load_search_df()

if not search_df.empty:
    min_date = pd.to_datetime(search_df["dateDepart"].min()).date()
    max_date = pd.to_datetime(search_df["dateReturn"].max()).date()
    # Defaults: today to today + 7 days, clamped to available data range
    today = pd.Timestamp.today().normalize().date()
    default_start = max(min_date, min(today, max_date))
    default_end = min(max_date, max(min_date, today + pd.Timedelta(days=7).to_pytimedelta()))
    
    # Ensure defaults are within valid range - clamp to min/max dates
    default_start = max(min_date, min(default_start, max_date))
    default_end = max(min_date, min(default_end, max_date))
    
    # Ensure start is not after end
    if default_start > default_end:
        default_start = default_end
    col1, col2 = st.columns(2)
    with col1:
        start_q = st.date_input("Start date", value=default_start, min_value=min_date, max_value=max_date)
    with col2:
        end_q = st.date_input("End date", value=default_end, min_value=min_date, max_value=max_date)

    if start_q > end_q:
        st.error("Start date must be before or equal to end date.")
    else:
        # Overlap condition: dateDepart <= end AND dateReturn >= start covers the three cases listed
        start_ts = pd.Timestamp(start_q)
        end_ts = pd.Timestamp(end_q)
        mask = (search_df["dateDepart"] <= end_ts) & (search_df["dateReturn"] >= start_ts)
        results = search_df.loc[mask].copy()
        st.caption(f"Matching policies: {len(results):,}")
        # Show a compact set of useful columns when present
        preferred_cols = [
            "idpol", "segment", "dateApp", "dateDepart", "dateReturn",
            "tripCost", "nightsCount", "travelersCount", "ZipCode", "Country", "State",
        ]
        cols_to_show = [c for c in preferred_cols if c in results.columns]
        if not cols_to_show:
            cols_to_show = list(results.columns)
        st.dataframe(results[cols_to_show].sort_values("dateDepart"), use_container_width=True)
        
st.markdown("---")
st.subheader("Inspect policies for a specific ISO week")

def iso_week_start(year: int, week: int) -> pd.Timestamp:
    # Robust ISO week start using Python's stdlib
    import datetime as _dt
    return pd.Timestamp(_dt.date.fromisocalendar(int(year), int(week), 1))

def load_week_search_df() -> pd.DataFrame:
    if df is not None:
        base = df.copy()
    else:
        if not selected_folder:
            st.info("Select an extract folder in the sidebar to inspect a week.")
            return pd.DataFrame()
        combined_path = os.path.join(selected_folder, "combined.parquet")
        if not os.path.exists(combined_path):
            st.warning("combined.parquet not found. Rebuild parquets first.")
            return pd.DataFrame()
        base = pd.read_parquet(combined_path)
    for c in ["dateDepart", "dateReturn", "dateApp"]:
        if c in base.columns:
            base[c] = pd.to_datetime(base[c], errors="coerce")
    return base

week_df = load_week_search_df()
if not week_df.empty:
    years = sorted(week_df["dateDepart"].dt.year.dropna().astype(int).unique().tolist())
    default_year = years[-1] if years else pd.Timestamp.today().year
    c1, c2 = st.columns(2)
    with c1:
        sel_year = st.number_input("Year", min_value=1900, max_value=2100, value=int(default_year), step=1)
    with c2:
        sel_week = st.number_input("ISO Week", min_value=1, max_value=53, value=int(pd.Timestamp.today().isocalendar().week), step=1)

    wk_start = iso_week_start(int(sel_year), int(sel_week))
    wk_end = wk_start + pd.to_timedelta(6, unit="D")
    st.caption(f"Week {int(sel_week)} of {int(sel_year)}: start={wk_start.date()} end={wk_end.date()}")

    # Overlap if any travel day intersects the week: depart <= wk_end and return > wk_start (night-start convention)
    mask_overlap = (week_df["dateDepart"] <= wk_end) & (week_df["dateReturn"] > wk_start)
    sel = week_df.loc[mask_overlap].copy()

    # Compute per-night price
    sel["perNight"] = (sel["tripCost"] / sel["nightsCount"].replace(0, pd.NA)).fillna(0.0)
    # Nights counted by start date: from dateDepart to dateReturn-1
    night_range_start = sel["dateDepart"]
    night_range_end = sel["dateReturn"] - pd.to_timedelta(1, unit="D")
    overlap_start = night_range_start.where(night_range_start > wk_start, wk_start)
    overlap_end = night_range_end.where(night_range_end < wk_end, wk_end)
    delta = (overlap_end - overlap_start).dt.days + 1
    sel["nightsInWeek"] = delta.clip(lower=0).fillna(0).astype(int)
    sel["remainingTripCost"] = (sel["nightsInWeek"] * sel["perNight"]).round(2)

    show_cols = [
        c for c in [
            "idpol", "segment", "dateApp", "dateDepart", "dateReturn",
            "nightsCount", "nightsInWeek", "tripCost", "perNight", "remainingTripCost",
            "ZipCode", "Country", "State",
        ] if c in sel.columns
    ]
    st.dataframe(sel[show_cols].sort_values(["nightsInWeek", "dateDepart"], ascending=[False, True]), use_container_width=True)
