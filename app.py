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

def calculate_traveling_metrics(df, period, group_by_segment):
    """Calculate traveling metrics using same logic as week search"""
    from exposure_data import classify_country, classify_region
    
    # Apply classification
    df["country_class"] = df["Country"].apply(classify_country)
    df["region_class"] = df["ZipCode"].apply(classify_region)
    
    # Calculate per-night cost
    df["perNight"] = (df["tripCost"] / df["nightsCount"].replace(0, pd.NA)).fillna(0.0)
    
    records = []
    
    if period == "week":
        # Process all weeks in the data
        all_dates = pd.concat([df["dateDepart"], df["dateReturn"]]).dropna()
        min_date = all_dates.min()
        max_date = all_dates.max()
        
        current_week = min_date - pd.to_timedelta(min_date.weekday(), unit="D")
        
        while current_week <= max_date:
            week_end = current_week + pd.to_timedelta(6, unit="D")
            
            # Find policies traveling during this week (same logic as week search)
            traveling_mask = (df["dateDepart"] <= week_end) & (df["dateReturn"] > current_week)
            traveling_policies = df[traveling_mask]
            
            if len(traveling_policies) > 0:
                # Calculate nights in week (same logic as week search)
                night_range_start = traveling_policies["dateDepart"]
                night_range_end = traveling_policies["dateReturn"] - pd.to_timedelta(1, unit="D")
                overlap_start = night_range_start.where(night_range_start > current_week, current_week)
                overlap_end = night_range_end.where(night_range_end < week_end, week_end)
                delta = (overlap_end - overlap_start).dt.days + 1
                traveling_policies["nightsInWeek"] = delta.clip(lower=0).fillna(0).astype(int)
                traveling_policies["remainingTripCost"] = (traveling_policies["nightsInWeek"] * traveling_policies["perNight"]).round(2)
                
                # Calculate normalized x-axis value
                x_norm = pd.Timestamp(2000, 1, 3) + pd.to_timedelta((current_week.isocalendar().week - 1) * 7, unit="D")
                year = current_week.year
                
                # Group by country_class and region_class
                groupby_cols = ["country_class", "region_class"]
                if group_by_segment and "segment" in traveling_policies.columns:
                    groupby_cols.append("segment")
                
                for group_vals, group_df in traveling_policies.groupby(groupby_cols, dropna=False):
                    if not isinstance(group_vals, tuple):
                        group_vals = (group_vals,)
                    
                    # Aggregate metrics (same as week search)
                    volume = len(group_df)
                    maxTripCostExposure = group_df["tripCost"].sum()
                    tripCostPerNightExposure = group_df["remainingTripCost"].sum()
                    avgTripCostPerNight = group_df["perNight"].mean()
                    
                    # Extract values
                    country_class = group_vals[0] if len(group_vals) > 0 else "null"
                    region_class = group_vals[1] if len(group_vals) > 1 else "Other"
                    extra_vals = list(group_vals[2:]) if len(group_vals) > 2 else []
                    
                    record = [year, x_norm, country_class, region_class, volume, maxTripCostExposure, tripCostPerNightExposure, avgTripCostPerNight] + extra_vals
                    records.append(record)
            
            current_week += pd.to_timedelta(7, unit="D")
    
    # Convert to DataFrame
    cols = ["year", "x", "country_class", "region_class", "volume", "maxTripCostExposure", "tripCostPerNightExposure", "avgTripCostPerNight"]
    if group_by_segment and "segment" in df.columns:
        cols.append("segment")
    
    result_df = pd.DataFrame(records, columns=cols)
    return result_df

def calculate_departures_metrics(df, period, group_by_segment):
    """Calculate departures metrics using same logic as week search"""
    from exposure_data import classify_country, classify_region
    
    # Apply classification
    df["country_class"] = df["Country"].apply(classify_country)
    df["region_class"] = df["ZipCode"].apply(classify_region)
    
    # Calculate per-night cost
    df["perNight"] = (df["tripCost"] / df["nightsCount"].replace(0, pd.NA)).fillna(0.0)
    
    if period == "week":
        # Group by departure week
        df["depart_week"] = df["dateDepart"].dt.to_period('W')
        df["x"] = df["dateDepart"].dt.to_period('W').dt.start_time
        df["year"] = df["dateDepart"].dt.year
        
        # Normalize x to 2000
        df["x"] = pd.Timestamp(2000, 1, 3) + pd.to_timedelta((df["x"].dt.isocalendar().week - 1) * 7, unit="D")
        
        group_cols = ["year", "x", "country_class", "region_class"]
        if group_by_segment and "segment" in df.columns:
            group_cols.append("segment")
        
        result_df = df.groupby(group_cols, as_index=False).agg(
            volume=("tripCost", "count"),
            maxTripCostExposure=("tripCost", "sum"),
            avgTripCostPerNight=("perNight", "mean"),
            tripCostPerNightExposure=("perNight", "sum"),
        )
        
        return result_df
    
    return pd.DataFrame()

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

# Use precomputed data for performance, but verify with raw data calculation
if precomputed_data is not None:
    # Use precomputed data and apply filters (FAST)
    filtered_data = filter_aggregate_data(precomputed_data, selected_countries, selected_regions)
    
    # Show filter summary
    if not filtered_data.empty:
        st.sidebar.caption(f"📈 Showing {len(filtered_data):,} records after filtering")
    
    # Handle segment filtering if needed
    if group_by_segment and "segment" in filtered_data.columns:
        segments = sorted(filtered_data["segment"].dropna().astype(str).unique().tolist())
        selected_segments = st.multiselect("Select segments", segments, default=segments)
        filtered_data = filtered_data[filtered_data["segment"].astype(str).isin(selected_segments)]
else:
    # Fallback to raw data calculation if no precomputed data
    st.warning("⚠️ No precomputed data found. Using raw data calculation (slower).")
    
    # Load raw data
    if use_dummy_data:
        df = get_data(use_dummy_data, selected_folder, erase_cache)
    else:
        if selected_folder:
            combined_path = os.path.join(selected_folder, "combined.parquet")
            if os.path.exists(combined_path):
                df = pd.read_parquet(combined_path)
                # Ensure datetime types
                for c in ["dateDepart", "dateReturn", "dateApp"]:
                    if c in df.columns:
                        df[c] = pd.to_datetime(df[c], errors="coerce")
    
    # Apply country filter to raw data
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
    
    # Calculate plot data using raw data calculation
    if df is not None and not df.empty:
        # Import classification functions
        from exposure_data import classify_country, classify_region
        
        # Apply country and region classification
        df_classified = df.copy()
        df_classified["country_class"] = df_classified["Country"].apply(classify_country)
        df_classified["region_class"] = df_classified["ZipCode"].apply(classify_region)
        
        # Apply region filter
        region_mask = pd.Series(False, index=df_classified.index)
        for region in selected_regions:
            region_mask |= (df_classified['region_class'] == region)
        df_classified = df_classified[region_mask].copy()
        
        # Calculate metrics using week search logic
        if metric_mode == "Traveling":
            # Calculate traveling metrics using same logic as week search
            filtered_data = calculate_traveling_metrics(df_classified, period, group_by_segment)
        else:
            # Calculate departures metrics
            filtered_data = calculate_departures_metrics(df_classified, period, group_by_segment)
        
        # Show filter summary
        if not filtered_data.empty:
            st.sidebar.caption(f"📈 Showing {len(filtered_data):,} records after filtering")
        
        # Handle segment filtering if needed
        if group_by_segment and "segment" in filtered_data.columns:
            segments = sorted(filtered_data["segment"].dropna().astype(str).unique().tolist())
            selected_segments = st.multiselect("Select segments", segments, default=segments)
            filtered_data = filtered_data[filtered_data["segment"].astype(str).isin(selected_segments)]
    else:
        st.error("No data available for plotting")
        filtered_data = pd.DataFrame()

# Get W1 2024 from plot data (aggregated)
filtered_data_w1_2024 = filtered_data[(filtered_data["year"] == 2024) & (filtered_data["x"].dt.isocalendar().week == 1)]
st.write("**Plot Data (Aggregated) for W1 2024:**")
st.dataframe(filtered_data_w1_2024)
    
# Get W1 2024 from precomputed data (direct from parquet)
if precomputed_data is not None:
    st.write("**Precomputed Data (Direct from Parquet) for W1 2024:**")
    precomputed_w1_2024 = precomputed_data[(precomputed_data["year"] == 2024) & (precomputed_data["x"].dt.isocalendar().week == 1)]
    st.dataframe(precomputed_w1_2024)
    
    # Apply same filters as plot data to precomputed data
    precomputed_filtered_w1_2024 = filter_aggregate_data(precomputed_w1_2024, selected_countries, selected_regions)
    st.write("**Precomputed Data (After Filtering) for W1 2024:**")
    st.dataframe(precomputed_filtered_w1_2024)
    
    # Aggregate precomputed data by plotting dimensions (same as plot data)
    if not precomputed_filtered_w1_2024.empty:
        plot_group_cols = ["year", "x"]
        if group_by_segment and "segment" in precomputed_filtered_w1_2024.columns:
            plot_group_cols.append("segment")
        
        plot_agg_dict = {}
        for col in ["volume", "maxTripCostExposure", "tripCostPerNightExposure", "avgTripCostPerNight"]:
            if col in precomputed_filtered_w1_2024.columns:
                if col == "avgTripCostPerNight":
                    plot_agg_dict[col] = "mean"
                else:
                    plot_agg_dict[col] = "sum"
        
        precomputed_plot_w1_2024 = precomputed_filtered_w1_2024.groupby(plot_group_cols, as_index=False).agg(plot_agg_dict)
        st.write("**Precomputed Data (Aggregated for Plotting) for W1 2024:**")
        st.dataframe(precomputed_plot_w1_2024)
        
        # Compare totals
        st.write("**Comparison of W1 2024 Totals:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Plot Data Volume", filtered_data_w1_2024["volume"].sum() if not filtered_data_w1_2024.empty else 0)
            st.metric("Plot Data Max Trip Cost", f"${filtered_data_w1_2024['maxTripCostExposure'].sum():,.2f}" if not filtered_data_w1_2024.empty else "$0.00")
        with col2:
            st.metric("Precomputed Volume", precomputed_plot_w1_2024["volume"].sum() if not precomputed_plot_w1_2024.empty else 0)
            st.metric("Precomputed Max Trip Cost", f"${precomputed_plot_w1_2024['maxTripCostExposure'].sum():,.2f}" if not precomputed_plot_w1_2024.empty else "$0.00")
        with col3:
            plot_vol = filtered_data_w1_2024["volume"].sum() if not filtered_data_w1_2024.empty else 0
            precomp_vol = precomputed_plot_w1_2024["volume"].sum() if not precomputed_plot_w1_2024.empty else 0
            diff_vol = plot_vol - precomp_vol
            st.metric("Difference (Plot - Precomp)", diff_vol)
            
            plot_cost = filtered_data_w1_2024["maxTripCostExposure"].sum() if not filtered_data_w1_2024.empty else 0
            precomp_cost = precomputed_plot_w1_2024["maxTripCostExposure"].sum() if not precomputed_plot_w1_2024.empty else 0
            diff_cost = plot_cost - precomp_cost
            st.metric("Cost Difference", f"${diff_cost:,.2f}")

# Get raw policies for W1 2024 from combined data
combined_data = pd.read_parquet(os.path.join(selected_folder, "combined.parquet"))
wk_start = pd.Timestamp("2024-01-01")  # W1 2024 starts Jan 1
wk_end = wk_start + pd.to_timedelta(6, unit="D")  # Jan 7

# Filter raw policies that overlap with W1 2024
mask_overlap = (combined_data["dateDepart"] <= wk_end) & (combined_data["dateReturn"] > wk_start)
raw_w1_2024 = combined_data.loc[mask_overlap].copy()

st.write(f"**Raw Policies Overlapping W1 2024 ({wk_start.date()} to {wk_end.date()}):**")
st.write(f"Found {len(raw_w1_2024)} policies")
st.dataframe(raw_w1_2024[["idpol", "segment", "dateDepart", "dateReturn", "tripCost", "nightsCount", "ZipCode", "Country"]].head(20))
    
# Calculate metrics the same way as week search
if not raw_w1_2024.empty:
    raw_w1_2024["perNight"] = (raw_w1_2024["tripCost"] / raw_w1_2024["nightsCount"].replace(0, pd.NA)).fillna(0.0)
    
    # Calculate nights in week (same logic as week search)
    night_range_start = raw_w1_2024["dateDepart"]
    night_range_end = raw_w1_2024["dateReturn"] - pd.to_timedelta(1, unit="D")
    overlap_start = night_range_start.where(night_range_start > wk_start, wk_start)
    overlap_end = night_range_end.where(night_range_end < wk_end, wk_end)
    delta = (overlap_end - overlap_start).dt.days + 1
    raw_w1_2024["nightsInWeek"] = delta.clip(lower=0).fillna(0).astype(int)
    raw_w1_2024["remainingTripCost"] = (raw_w1_2024["nightsInWeek"] * raw_w1_2024["perNight"]).round(2)
        
        # Calculate totals
    total_volume = len(raw_w1_2024)
    total_max_trip_cost = raw_w1_2024["tripCost"].sum()
    total_remaining_trip_cost = raw_w1_2024["remainingTripCost"].sum()
    avg_trip_cost_per_night = raw_w1_2024["perNight"].mean()
    
    st.write("**Week Search Method Calculation:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Volume (policies)", total_volume)
    with col2:
        st.metric("Max Trip Cost", f"${total_max_trip_cost:,.2f}")
    with col3:
        st.metric("Remaining Trip Cost", f"${total_remaining_trip_cost:,.2f}")
    with col4:
        st.metric("Avg Trip Cost/Night", f"${avg_trip_cost_per_night:,.2f}")
    
    # NEW: Apply country and region classification to raw data and aggregate like plot data
    st.write("**Raw Data with Country/Region Classification (like plot data):**")
    
    # Import classification functions
    from exposure_data import classify_country, classify_region
    
    # Apply country and region classification
    raw_w1_2024_classified = raw_w1_2024.copy()
    raw_w1_2024_classified["country_class"] = raw_w1_2024_classified["Country"].apply(classify_country)
    raw_w1_2024_classified["region_class"] = raw_w1_2024_classified["ZipCode"].apply(classify_region)
    
    # Show the classified data
    st.write("Classified raw data:")
    st.dataframe(raw_w1_2024_classified[["idpol", "segment", "dateDepart", "dateReturn", "tripCost", "nightsCount", "Country", "country_class", "ZipCode", "region_class"]].head(10))
    
    # Apply the same filters as the plot data (if any country/region filters are selected)
    filtered_raw_w1_2024 = raw_w1_2024_classified.copy()
    
    # Apply country filter if selected
    if selected_countries and "All" not in selected_countries:
        filtered_raw_w1_2024 = filtered_raw_w1_2024[filtered_raw_w1_2024["country_class"].isin(selected_countries)]
    
    # Apply region filter if selected  
    if selected_regions and "All" not in selected_regions:
        filtered_raw_w1_2024 = filtered_raw_w1_2024[filtered_raw_w1_2024["region_class"].isin(selected_regions)]
    
    st.write(f"After applying filters: {len(filtered_raw_w1_2024)} policies (from {len(raw_w1_2024_classified)} original)")
    
    # Now aggregate the filtered raw data the same way as plot data
    if not filtered_raw_w1_2024.empty:
        # Calculate the same metrics as the plot data
        raw_agg_data = filtered_raw_w1_2024.groupby(["country_class", "region_class"], as_index=False).agg(
            volume=("idpol", "count"),
            maxTripCostExposure=("tripCost", "sum"),
            avgTripCostPerNight=("perNight", "mean"),
            tripCostPerNightExposure=("remainingTripCost", "sum")
        )
        
        # Add year and x columns to match plot data structure
        raw_agg_data["year"] = 2024
        raw_agg_data["x"] = pd.Timestamp("2000-01-03")  # W1 normalized to 2000
        
        st.write("**Raw Data Aggregated (like plot data):**")
        st.dataframe(raw_agg_data)
        
        # Calculate totals from aggregated data
        raw_total_volume = raw_agg_data["volume"].sum()
        raw_total_max_trip_cost = raw_agg_data["maxTripCostExposure"].sum()
        raw_total_trip_cost_per_night = raw_agg_data["tripCostPerNightExposure"].sum()
        raw_avg_trip_cost_per_night = raw_agg_data["avgTripCostPerNight"].mean()
        
        st.write("**Raw Data Aggregated Totals:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Volume", raw_total_volume)
        with col2:
            st.metric("Max Trip Cost", f"${raw_total_max_trip_cost:,.2f}")
        with col3:
            st.metric("Trip Cost/Night Exposure", f"${raw_total_trip_cost_per_night:,.2f}")
        with col4:
            st.metric("Avg Trip Cost/Night", f"${raw_avg_trip_cost_per_night:,.2f}")
    else:
        st.write("**No data remaining after applying filters**")
    
    # W2 2024 Comparison
    st.markdown("---")
    st.subheader("🔍 W2 2024 Comparison")
    
    # Get W2 2024 from plot data (aggregated)
    filtered_data_w2_2024 = filtered_data[(filtered_data["year"] == 2024) & (filtered_data["x"].dt.isocalendar().week == 2)]
    st.write("**Plot Data (Aggregated) for W2 2024:**")
    st.dataframe(filtered_data_w2_2024)
    
    # Get raw policies for W2 2024 from combined data
    wk2_start = pd.Timestamp("2024-01-08")  # W2 2024 starts Jan 8
    wk2_end = wk2_start + pd.to_timedelta(6, unit="D")  # Jan 14
    
    # Filter raw policies that overlap with W2 2024
    mask_overlap_w2 = (combined_data["dateDepart"] <= wk2_end) & (combined_data["dateReturn"] > wk2_start)
    raw_w2_2024 = combined_data.loc[mask_overlap_w2].copy()
    
    st.write(f"**Raw Policies Overlapping W2 2024 ({wk2_start.date()} to {wk2_end.date()}):**")
    st.write(f"Found {len(raw_w2_2024)} policies")
    st.dataframe(raw_w2_2024[["idpol", "segment", "dateDepart", "dateReturn", "tripCost", "nightsCount", "ZipCode", "Country"]].head(20))
    
# Calculate metrics the same way as week search for W2
if not raw_w2_2024.empty:
    raw_w2_2024["perNight"] = (raw_w2_2024["tripCost"] / raw_w2_2024["nightsCount"].replace(0, pd.NA)).fillna(0.0)
    
    # Calculate nights in week (same logic as week search)
    night_range_start = raw_w2_2024["dateDepart"]
    night_range_end = raw_w2_2024["dateReturn"] - pd.to_timedelta(1, unit="D")
    overlap_start = night_range_start.where(night_range_start > wk2_start, wk2_start)
    overlap_end = night_range_end.where(night_range_end < wk2_end, wk2_end)
    delta = (overlap_end - overlap_start).dt.days + 1
    raw_w2_2024["nightsInWeek"] = delta.clip(lower=0).fillna(0).astype(int)
    raw_w2_2024["remainingTripCost"] = (raw_w2_2024["nightsInWeek"] * raw_w2_2024["perNight"]).round(2)
    
    # Calculate totals
    total_volume_w2 = len(raw_w2_2024)
    total_max_trip_cost_w2 = raw_w2_2024["tripCost"].sum()
    total_remaining_trip_cost_w2 = raw_w2_2024["remainingTripCost"].sum()
    avg_trip_cost_per_night_w2 = raw_w2_2024["perNight"].mean()
    
    st.write("**Week Search Method Calculation for W2 2024:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Volume (policies)", total_volume_w2)
    with col2:
        st.metric("Max Trip Cost", f"${total_max_trip_cost_w2:,.2f}")
    with col3:
        st.metric("Remaining Trip Cost", f"${total_remaining_trip_cost_w2:,.2f}")
    with col4:
        st.metric("Avg Trip Cost/Night", f"${avg_trip_cost_per_night_w2:,.2f}")
    
    # Apply country and region classification to raw W2 data
    st.write("**Raw W2 Data with Country/Region Classification (like plot data):**")
    
    # Apply country and region classification
    raw_w2_2024_classified = raw_w2_2024.copy()
    raw_w2_2024_classified["country_class"] = raw_w2_2024_classified["Country"].apply(classify_country)
    raw_w2_2024_classified["region_class"] = raw_w2_2024_classified["ZipCode"].apply(classify_region)
    
    # Show the classified data
    st.write("Classified raw W2 data:")
    st.dataframe(raw_w2_2024_classified[["idpol", "segment", "dateDepart", "dateReturn", "tripCost", "nightsCount", "Country", "country_class", "ZipCode", "region_class"]].head(10))
    
    # Apply the same filters as the plot data
    filtered_raw_w2_2024 = raw_w2_2024_classified.copy()
    
    # Apply country filter if selected
    if selected_countries and "All" not in selected_countries:
        filtered_raw_w2_2024 = filtered_raw_w2_2024[filtered_raw_w2_2024["country_class"].isin(selected_countries)]
    
    # Apply region filter if selected  
    if selected_regions and "All" not in selected_regions:
        filtered_raw_w2_2024 = filtered_raw_w2_2024[filtered_raw_w2_2024["region_class"].isin(selected_regions)]
    
    st.write(f"After applying filters: {len(filtered_raw_w2_2024)} policies (from {len(raw_w2_2024_classified)} original)")
    
    # Now aggregate the filtered raw W2 data the same way as plot data
    if not filtered_raw_w2_2024.empty:
        # Calculate the same metrics as the plot data
        raw_agg_data_w2 = filtered_raw_w2_2024.groupby(["country_class", "region_class"], as_index=False).agg(
            volume=("idpol", "count"),
            maxTripCostExposure=("tripCost", "sum"),
            avgTripCostPerNight=("perNight", "mean"),
            tripCostPerNightExposure=("remainingTripCost", "sum")
        )
        
        # Add year and x columns to match plot data structure
        raw_agg_data_w2["year"] = 2024
        raw_agg_data_w2["x"] = pd.Timestamp("2000-01-10")  # W2 normalized to 2000
        
        st.write("**Raw W2 Data Aggregated (like plot data):**")
        st.dataframe(raw_agg_data_w2)
        
        # Calculate totals from aggregated W2 data
        raw_total_volume_w2 = raw_agg_data_w2["volume"].sum()
        raw_total_max_trip_cost_w2 = raw_agg_data_w2["maxTripCostExposure"].sum()
        raw_total_trip_cost_per_night_w2 = raw_agg_data_w2["tripCostPerNightExposure"].sum()
        raw_avg_trip_cost_per_night_w2 = raw_agg_data_w2["avgTripCostPerNight"].mean()
        
        st.write("**Raw W2 Data Aggregated Totals:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Volume", raw_total_volume_w2)
        with col2:
            st.metric("Max Trip Cost", f"${raw_total_max_trip_cost_w2:,.2f}")
        with col3:
            st.metric("Trip Cost/Night Exposure", f"${raw_total_trip_cost_per_night_w2:,.2f}")
        with col4:
            st.metric("Avg Trip Cost/Night", f"${raw_avg_trip_cost_per_night_w2:,.2f}")
    else:
        st.write("**No W2 data remaining after applying filters**")

# Show what the plot data contains for W2 - RECALCULATE using same logic as raw data
st.write("**Plot Data Method Calculation for W2 2024 (RECALCULATED):**")

# Recalculate W2 2024 using the EXACT SAME data as raw data aggregation
if not raw_w2_2024.empty:
    # Use the EXACT SAME filtered data as the raw data aggregation above
    plot_raw_w2_2024 = filtered_raw_w2_2024.copy()  # Use the same data source!
    
    if not plot_raw_w2_2024.empty:
        # Calculate the same metrics as the raw data aggregation
        plot_agg_data_w2 = plot_raw_w2_2024.groupby(["country_class", "region_class"], as_index=False).agg(
            volume=("idpol", "count"),
            maxTripCostExposure=("tripCost", "sum"),
            avgTripCostPerNight=("perNight", "mean"),
            tripCostPerNightExposure=("remainingTripCost", "sum")
        )
        
        # Calculate totals from recalculated data
        plot_total_volume_w2 = plot_agg_data_w2["volume"].sum()
        plot_total_max_trip_cost_w2 = plot_agg_data_w2["maxTripCostExposure"].sum()
        plot_total_trip_cost_per_night_w2 = plot_agg_data_w2["tripCostPerNightExposure"].sum()
        plot_avg_trip_cost_per_night_w2 = plot_agg_data_w2["avgTripCostPerNight"].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Volume", f"{plot_total_volume_w2:,.0f}")
        with col2:
            st.metric("Max Trip Cost", f"${plot_total_max_trip_cost_w2:,.2f}")
        with col3:
            st.metric("Trip Cost/Night Exposure", f"${plot_total_trip_cost_per_night_w2:,.2f}")
        with col4:
            st.metric("Avg Trip Cost/Night", f"${plot_avg_trip_cost_per_night_w2:,.2f}")
        
        st.write("**Comparison - Should now match Raw Data Aggregated Totals above!**")
        
        # Additional debugging to identify the remaining gap
        st.write("**🔍 DEBUG: Detailed Comparison**")
        st.write(f"Raw Data policies: {len(filtered_raw_w2_2024)}")
        st.write(f"Plot Data policies: {len(plot_raw_w2_2024)}")
        st.write(f"Policy count difference: {len(filtered_raw_w2_2024) - len(plot_raw_w2_2024)}")
        
        # Show sample of policies from each method
        if len(filtered_raw_w2_2024) > 0:
            st.write("**Raw Data sample policies:**")
            st.dataframe(filtered_raw_w2_2024[["idpol", "segment", "dateDepart", "dateReturn", "tripCost", "nightsCount", "country_class", "region_class"]].head(5))
        
        if len(plot_raw_w2_2024) > 0:
            st.write("**Plot Data sample policies:**")
            st.dataframe(plot_raw_w2_2024[["idpol", "segment", "dateDepart", "dateReturn", "tripCost", "nightsCount", "country_class", "region_class"]].head(5))
        
        # Check if policies are identical
        if len(filtered_raw_w2_2024) == len(plot_raw_w2_2024):
            policies_match = filtered_raw_w2_2024["idpol"].sort_values().reset_index(drop=True).equals(plot_raw_w2_2024["idpol"].sort_values().reset_index(drop=True))
            st.write(f"**Same policies in both methods: {policies_match}**")
        else:
            st.write("**Different number of policies - this explains the gap!**")
    else:
        st.write("**No data remaining after applying filters**")
else:
    st.write("**No raw W2 2024 data found**")

# Show what the plot data contains
if not filtered_data_w1_2024.empty:
    st.write("**Plot Data Method Calculation:**")
    plot_metrics = {}
    for col in ["volume", "maxTripCostExposure", "tripCostPerNightExposure", "avgTripCostPerNight"]:
        if col in filtered_data_w1_2024.columns:
            plot_metrics[col] = filtered_data_w1_2024[col].sum() if col != "avgTripCostPerNight" else filtered_data_w1_2024[col].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Volume", f"{plot_metrics.get('volume', 0):,.0f}")
    with col2:
        st.metric("Max Trip Cost", f"${plot_metrics.get('maxTripCostExposure', 0):,.2f}")
    with col3:
        st.metric("Trip Cost/Night Exposure", f"${plot_metrics.get('tripCostPerNightExposure', 0):,.2f}")
    with col4:
        st.metric("Avg Trip Cost/Night", f"${plot_metrics.get('avgTripCostPerNight', 0):,.2f}")
else:
    st.write("**No plot data found for W1 2024**")

st.markdown("---")

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
    
    #plot the data
#else:
#    # No precomputed data available - show error message
 #   st.error("❌ No precomputed data available. Please run the precomputation first:")
  #  st.code("python3 test_precompute.py _data --limit-rows 1000", language="bash")
   # st.stop()

# Define years for plot ordering (must be outside conditional blocks)
years = sorted(filtered_data["year"].unique().tolist())
if year_order_choice == "Descending":
    years = list(reversed(years))

# Aggregate filtered_data by plotting dimensions (x, year, segment)
plot_group_cols = ["year", "x"]
if group_by_segment and "segment" in filtered_data.columns:
    plot_group_cols.append("segment")

# Create aggregation dictionary for all available metrics
plot_agg_dict = {}
for col in ["volume", "maxTripCostExposure", "tripCostPerNightExposure", "avgTripCostPerNight"]:
    if col in filtered_data.columns:
        if col == "avgTripCostPerNight":
            plot_agg_dict[col] = "mean"  # Average of averages
        else:
            plot_agg_dict[col] = "sum"   # Sum across country/region groups

# Aggregate data for plotting
plot_data = filtered_data.groupby(plot_group_cols, as_index=False).agg(plot_agg_dict)

# Apply year ordering to plot data
plot_data["year"] = pd.Categorical(plot_data["year"], categories=years, ordered=True)

# Create the plot
ycol = selected_metric
if group_by_segment and "segment" in plot_data.columns:
    fig = px.line(
        plot_data,
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
        plot_data,
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

# DEBUG: Compare W1 and W2 2024 calculations between plot data and week search
# This section runs regardless of whether we use precomputed or raw data
st.markdown("---")
st.subheader("🔍 DEBUG: W1 & W2 2024 Comparison")

# Get W1 2024 from plot data (aggregated)
filtered_data_w1_2024 = filtered_data[(filtered_data["year"] == 2024) & (filtered_data["x"].dt.isocalendar().week == 1)]
st.write("**Plot Data (Aggregated) for W1 2024:**")
st.dataframe(filtered_data_w1_2024)

# Add raw data table below the plot for debugging
st.markdown("---")
st.subheader("Raw Data Underlying the Plot")

if 'plot_data' in locals() and not plot_data.empty:
    st.write("This is the exact dataframe used to create the plot above:")
    st.dataframe(plot_data, use_container_width=True)
    
    # Show some basic info about the data
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", len(plot_data))
    with col2:
        st.metric("Unique Years", plot_data['year'].nunique() if 'year' in plot_data.columns else 0)
    with col3:
        st.metric("Unique Segments", plot_data['segment'].nunique() if 'segment' in plot_data.columns else 0)
    with col4:
        st.metric("Metric Column", ycol)
    
    # Show data types and sample values
    st.write("**Data Types:**")
    st.write(plot_data.dtypes)
    
    # Show sample values for debugging
    st.write("**Sample Values (first 10 rows):**")
    st.write(plot_data.head(10))
    
    # Show specific W1 2024 data if it exists
    if 'year' in plot_data.columns and 'x' in plot_data.columns:
        w1_2024_data = plot_data[(plot_data['year'] == 2024) & (plot_data['x'].dt.isocalendar().week == 1)]
        if not w1_2024_data.empty:
            st.write("**W1 2024 Data (the problematic week):**")
            st.write(w1_2024_data)
        else:
            st.write("**No W1 2024 data found in the dataset**")
else:
    st.info("No data available to display.")

# Add data table below the plot for debugging
st.markdown("---")
st.subheader("Data Table: Metric by ISO Week and Year")

if 'plot_data' in locals() and not plot_data.empty:
    # Create a pivot table for better visualization
    if period == "week":
        # For weeks, create a table with ISO week numbers
        data_table = plot_data.copy()
        
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
