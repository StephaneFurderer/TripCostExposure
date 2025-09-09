import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
from exposure_data import load_folder_policies, classify_country, classify_region

def calculate_traveling_metrics(df, period, group_by_segment):
    """Calculate traveling metrics using raw data"""
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
            
            # Find policies traveling during this week
            traveling_mask = (df["dateDepart"] <= week_end) & (df["dateReturn"] > current_week)
            traveling_policies = df[traveling_mask]
            
            if len(traveling_policies) > 0:
                # Calculate nights in week
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
                    
                    # Aggregate metrics
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
    """Calculate departures metrics using raw data"""
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

# Page config
st.set_page_config(page_title="Trip Cost Exposure Analysis", layout="wide")

# Sidebar controls for data source
with st.sidebar:
    st.header("Data Source")
    use_dummy_data = st.checkbox("Use dummy data", value=True, help="Toggle to use packaged sample dataset")
    
    # Data folder selection (moved above filters)
    if not use_dummy_data:
        # Look for folders under _data
        data_folders = []
        if Path("_data").exists():
            data_folders = [f.name for f in Path("_data").iterdir() if f.is_dir() and not f.name.startswith(".")]
        
        # Also look in current directory for other data folders
        current_folders = [f.name for f in Path(".").iterdir() if f.is_dir() and not f.name.startswith(".") and f.name != "_data"]
        
        all_folders = data_folders + current_folders
        selected_folder = st.selectbox("Select data folder", [""] + all_folders)
    else:
        selected_folder = None
    
    # Country filter
    country_filter_options = ["US", "ROW", "null"]
    selected_countries = st.multiselect(
        "Country Filter",
        options=country_filter_options,
        default=country_filter_options,
        help="US: United States, ROW: Rest of World, null: Missing country data"
    )
    
    # Region filter
    region_filter_options = ["Atlantic", "Florida", "Gulf", "Pacific", "Other"]
    selected_regions = st.multiselect(
        "Region Filter",
        options=region_filter_options,
        default=region_filter_options,
        help="Filter by US coastal regions"
    )

# UI Controls
group_by_segment = st.checkbox("Group by Segment", value=False)
period = st.selectbox("Time Period", ["day", "week", "month"], index=1)
metric_mode = st.radio("Mode", ["Traveling", "Departures"], index=0)
year_order_choice = st.selectbox("Departure Year order", options=["Ascending", "Descending"], index=0)

# Try to load precomputed aggregates first
precomputed_data = None
folder_path = None

# Determine folder path for precomputed data
if use_dummy_data:
    folder_path = Path("_data")
else:
    if selected_folder:
        # Check if selected folder is under _data or in current directory
        if Path(f"_data/{selected_folder}").exists():
            base_folder = Path(f"_data/{selected_folder}")
        else:
            base_folder = Path(selected_folder)
        
        # Look for date subfolder in the selected folder
        folder_contents = [f for f in base_folder.iterdir() if f.is_dir() and f.name[0].isdigit()]
        if folder_contents:
            # Use the most recent date folder
            folder_path = sorted(folder_contents)[-1]
        else:
            folder_path = base_folder
    else:
        folder_path = None

# Load precomputed data if available
if folder_path:
    if metric_mode == "Traveling":
        if period == "day":
            agg_file = folder_path / "agg_travel_day_by_segment.parquet"
        else:
            agg_file = folder_path / f"agg_travel_{period}_by_segment.parquet"
    else:
        agg_file = folder_path / f"agg_depart_{period}_by_segment.parquet"
    
    if agg_file.exists():
        precomputed_data = pd.read_parquet(agg_file)
        st.sidebar.success(f"✅ Loaded precomputed {period} {metric_mode.lower()} data")
        
        # If user doesn't want segment grouping, aggregate the segment data
        if not group_by_segment and "segment" in precomputed_data.columns:
            group_cols = ["year", "x", "country_class", "region_class"]
            agg_dict = {}
            for col in ["volume", "maxTripCostExposure", "tripCostPerNightExposure", "avgTripCostPerNight"]:
                if col in precomputed_data.columns:
                    if col == "avgTripCostPerNight":
                        agg_dict[col] = "mean"
                    else:
                        agg_dict[col] = "sum"
            
            precomputed_data = precomputed_data.groupby(group_cols, as_index=False).agg(agg_dict)
            st.sidebar.info("📊 Aggregated segment data to show 'all' view")
    else:
        st.sidebar.warning(f"⚠️ Precomputed {period} {metric_mode.lower()} data not found, will compute from raw data")

# Fallback to raw data if no precomputed data available
df = None
if precomputed_data is None:
    if use_dummy_data:
        df = load_folder_policies("_data", force_rebuild=False, erase_cache=False)
    else:
        if selected_folder:
            combined_path = os.path.join(selected_folder, "combined.parquet")
            if os.path.exists(combined_path):
                df = pd.read_parquet(combined_path)
                for c in ["dateDepart", "dateReturn", "dateApp"]:
                    if c in df.columns:
                        df[c] = pd.to_datetime(df[c], errors="coerce")

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

# Unified metric selector
if metric_mode == "Traveling":
    metric_options = ["volume", "maxTripCostExposure", "tripCostPerNightExposure", "avgTripCostPerNight"]
    default_idx = 1
else:
    metric_options = ["volume", "maxTripCostExposure", "tripCostPerNightExposure", "avgTripCostPerNight"]
    default_idx = 1

selected_metric = st.selectbox("Metric to plot", options=metric_options, index=default_idx)

# Main display panel
st.header("Trip Cost Exposure Analysis")

# Use precomputed data for performance, with fallback to raw data calculation
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
        df = load_folder_policies("_data", force_rebuild=False, erase_cache=False)
    else:
        if selected_folder:
            combined_path = os.path.join(selected_folder, "combined.parquet")
            if os.path.exists(combined_path):
                df = pd.read_parquet(combined_path)
                for c in ["dateDepart", "dateReturn", "dateApp"]:
                    if c in df.columns:
                        df[c] = pd.to_datetime(df[c], errors="coerce")
    
    # Apply country filter to raw data
    if df is not None and not df.empty:
        country_mask = pd.Series(False, index=df.index)
        
        if "US" in selected_countries:
            country_mask |= (df['Country'] == 'US')
        if "ROW" in selected_countries:
            country_mask |= ((df['Country'] != 'US') & (df['Country'].notna()))
        if "null" in selected_countries:
            country_mask |= df['Country'].isna()
        
        df = df[country_mask].copy()
        st.sidebar.caption(f"📊 Showing {len(df):,} policies after country filtering")
    
    # Calculate plot data using raw data calculation
    if df is not None and not df.empty:
        # Apply country and region classification
        df["country_class"] = df["Country"].apply(classify_country)
        df["region_class"] = df["ZipCode"].apply(classify_region)
        
        # Apply region filter
        region_mask = pd.Series(False, index=df.index)
        for region in selected_regions:
            region_mask |= (df['region_class'] == region)
        df = df[region_mask].copy()
        
        # Calculate metrics using raw data calculation
        if metric_mode == "Traveling":
            # Calculate traveling metrics using raw data
            filtered_data = calculate_traveling_metrics(df, period, group_by_segment)
        else:
            # Calculate departures metrics using raw data
            filtered_data = calculate_departures_metrics(df, period, group_by_segment)
        
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

# Define years for plot ordering
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
    tickvals = pd.to_datetime([f"2000-{m:02d}-01" for m in range(1, 13)])
    ticktext = [pd.Timestamp(2000, m, 1).strftime("%b") for m in range(1, 13)]
    fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)
elif period == "week":
    week_starts = pd.to_datetime("2000-01-03") + pd.to_timedelta(range(0, 52, 4), unit="W")
    tickvals = week_starts
    ticktext = [f"W{int(((d - pd.Timestamp('2000-01-03')).days)/7)+1}" for d in tickvals]
    fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)
else:  # month
    tickvals = pd.to_datetime([f"2000-{m:02d}-01" for m in range(1, 13)])
    ticktext = [pd.Timestamp(2000, m, 1).strftime("%b") for m in range(1, 13)]
    fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)

fig.update_layout(legend_title_text="Departure Year", hovermode="x unified")

st.plotly_chart(fig, use_container_width=True)

# Second plot: Chronological time series
st.markdown("---")
st.subheader("Chronological Time Series (No Normalization)")

# Create chronological plot data
if not plot_data.empty:
    # Create a copy for chronological plotting
    chrono_data = plot_data.copy()
    
    # For traveling data, we need to convert back to actual dates
    if metric_mode == "Traveling":
        # Convert normalized x back to actual dates using proper ISO week calculation
        # The normalized x is set to 2000, so we need to extract the week and apply to actual year
        def convert_to_actual_date(row):
            year = row['year']
            week_num = row['x'].isocalendar().week
            
            # Use pandas to find the actual date for this ISO week
            # Start with January 1st of the year
            jan_1 = pd.Timestamp(f"{year}-01-01")
            
            # Find the first Monday of the ISO year
            # ISO week 1 is the first week with at least 4 days in the new year
            first_monday = jan_1 + pd.to_timedelta((7 - jan_1.weekday()) % 7, unit="D")
            
            # If the first Monday is in the previous year, it means week 1 starts later
            if first_monday.year < year:
                # Week 1 starts on the first Monday of the new year
                first_monday = first_monday + pd.to_timedelta(7, unit="D")
            
            # Calculate the actual date for this week
            actual_date = first_monday + pd.to_timedelta((week_num - 1) * 7, unit="D")
            return actual_date
        
        chrono_data["actual_date"] = chrono_data.apply(convert_to_actual_date, axis=1)
    else:
        # For departures, x is already the departure date
        chrono_data["actual_date"] = chrono_data["x"]
    
    # Create chronological plot
    if group_by_segment and "segment" in chrono_data.columns:
        fig_chrono = px.line(
            chrono_data,
            x="actual_date",
            y=ycol,
            color="year",
            facet_row="segment",
            color_discrete_sequence=px.colors.qualitative.Safe,
            category_orders={"year": years},
            labels={"actual_date": "Date", ycol: ycol, "year": "Year", "segment": "Segment"},
        )
    else:
        fig_chrono = px.line(
            chrono_data,
            x="actual_date",
            y=ycol,
            color="year",
            color_discrete_sequence=px.colors.qualitative.Safe,
            category_orders={"year": years},
            labels={"actual_date": "Date", ycol: ycol, "year": "Year"},
        )
    a
    # Configure x-axis for chronological display
    fig_chrono.update_xaxes(
        tickformat="%Y-%m-%d",
        tickangle=45
    )
    
    fig_chrono.update_layout(
        legend_title_text="Departure Year", 
        hovermode="x unified",
        xaxis_title="Date"
    )
    
    st.plotly_chart(fig_chrono, use_container_width=True)
    
    # Show some info about the chronological data
    st.caption(f"Showing {len(chrono_data)} data points from {chrono_data['actual_date'].min().date()} to {chrono_data['actual_date'].max().date()}")
else:
    st.info("No data available for chronological plot")

# Data table below the plot
st.markdown("---")
st.subheader("Data Table: Metric by ISO Week and Year")

if 'plot_data' in locals() and not plot_data.empty:
    # Create a pivot table for better visualization
    if period == "week":
        data_table = plot_data.copy()
        
        # Convert x to ISO week number
        if 'x' in data_table.columns:
            data_table['iso_week'] = data_table['x'].dt.isocalendar().week
            data_table['iso_year'] = data_table['year']
        
        # Create pivot table
        pivot_cols = ['iso_week', 'iso_year', ycol]
        if group_by_segment and 'segment' in data_table.columns:
            pivot_cols.append('segment')
        
        # Group by week and year, then pivot
        if group_by_segment and 'segment' in data_table.columns:
            pivot_data = data_table.groupby(['iso_week', 'iso_year', 'segment'])[ycol].sum().reset_index()
            pivot_table = pivot_data.pivot_table(
                index='iso_week', 
                columns=['iso_year', 'segment'], 
                values=ycol, 
                fill_value=0
            )
        else:
            pivot_data = data_table.groupby(['iso_week', 'iso_year'])[ycol].sum().reset_index()
            pivot_table = pivot_data.pivot_table(
                index='iso_week', 
                columns='iso_year', 
                values=ycol, 
                fill_value=0
            )
        
        # Format the table for better readability
        if not pivot_table.empty:
            pivot_table = pivot_table.round(2)
            pivot_table.loc['TOTAL'] = pivot_table.sum()
            st.dataframe(pivot_table, use_container_width=True)
        else:
            st.info("No data available for the selected filters.")
    
    else:  # day
        st.dataframe(plot_data.head(20), use_container_width=True)
        if len(plot_data) > 20:
            st.caption(f"Showing first 20 rows of {len(plot_data)} total records")

else:
    st.info("No data available to display in table.")

# Week search functionality
st.markdown("---")
st.subheader("Week Search")

def load_week_search_df():
    """Load data for week search functionality"""
    if use_dummy_data:
        return load_folder_policies("_data", force_rebuild=False, erase_cache=False)
    else:
        if selected_folder:
            combined_path = os.path.join(selected_folder, "combined.parquet")
            if os.path.exists(combined_path):
                df = pd.read_parquet(combined_path)
                for c in ["dateDepart", "dateReturn", "dateApp"]:
                    if c in df.columns:
                        df[c] = pd.to_datetime(df[c], errors="coerce")
                return df
    return pd.DataFrame()

week_df = load_week_search_df()
if not week_df.empty:
    years = sorted(week_df["dateDepart"].dt.year.dropna().astype(int).unique().tolist())
    default_year = years[-1] if years else pd.Timestamp.today().year
    c1, c2 = st.columns(2)
    with c1:
        sel_year = st.number_input("Year", min_value=1900, max_value=2100, value=int(default_year), step=1)
    with c2:
        sel_week = st.number_input("ISO Week", min_value=1, max_value=53, value=1, step=1)
    
    # Calculate week dates using ISO week logic
    # Find the first Monday of the year
    jan_1 = pd.Timestamp(f"{sel_year}-01-01")
    first_monday = jan_1 + pd.to_timedelta((7 - jan_1.weekday()) % 7, unit="D")
    
    # Calculate the start of the requested ISO week
    week_start = first_monday + pd.to_timedelta((sel_week - 1) * 7, unit="D")
    week_end = week_start + pd.to_timedelta(6, unit="D")
    
    st.write(f"**Week {sel_week}, {sel_year}:** {week_start.date()} to {week_end.date()}")
    
    # Filter policies that overlap with the selected week
    mask_overlap = (week_df["dateDepart"] <= week_end) & (week_df["dateReturn"] > week_start)
    week_policies = week_df.loc[mask_overlap].copy()
    
    if not week_policies.empty:
        # Calculate metrics
        week_policies["perNight"] = (week_policies["tripCost"] / week_policies["nightsCount"].replace(0, pd.NA)).fillna(0.0)
        
        # Calculate nights in week
        night_range_start = week_policies["dateDepart"]
        night_range_end = week_policies["dateReturn"] - pd.to_timedelta(1, unit="D")
        overlap_start = night_range_start.where(night_range_start > week_start, week_start)
        overlap_end = night_range_end.where(night_range_end < week_end, week_end)
        delta = (overlap_end - overlap_start).dt.days + 1
        week_policies["nightsInWeek"] = delta.clip(lower=0).fillna(0).astype(int)
        week_policies["remainingTripCost"] = (week_policies["nightsInWeek"] * week_policies["perNight"]).round(2)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Volume (policies)", len(week_policies))
        with col2:
            st.metric("Max Trip Cost", f"${week_policies['tripCost'].sum():,.2f}")
        with col3:
            st.metric("Trip Cost/Night Exposure", f"${week_policies['remainingTripCost'].sum():,.2f}")
        with col4:
            st.metric("Avg Trip Cost/Night", f"${week_policies['perNight'].mean():,.2f}")
        
        # Show policy details
        st.write("**Policy Details:**")
        display_cols = ["idpol", "segment", "dateDepart", "dateReturn", "tripCost", "nightsCount", "nightsInWeek", "remainingTripCost", "perNight"]
        available_cols = [col for col in display_cols if col in week_policies.columns]
        st.dataframe(week_policies[available_cols], use_container_width=True)
    else:
        st.info("No policies found for the selected week.")
else:
    if not selected_folder:
        st.info("Select an extract folder in the sidebar to search real data.")
    else:
        st.info("No data available for week search.")
