import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
from datetime import datetime

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

def get_extraction_date(folder_path):
    """Extract the extraction date from folder structure"""
    if folder_path is None:
        return None
    
    folder_name = folder_path.name
    # Try to parse folder name as date (format: YYYY-MM-DD or similar)
    try:
        # Handle various date formats in folder names
        if len(folder_name) >= 8 and folder_name[:8].replace('-', '').replace('_', '').isdigit():
            date_part = folder_name[:10]  # Take first 10 chars for YYYY-MM-DD
            return pd.to_datetime(date_part, errors='coerce')
    except:
        pass
    
    # Fallback: use the most recent file modification time
    try:
        parquet_files = list(folder_path.glob("*.parquet"))
        if parquet_files:
            most_recent = max(parquet_files, key=lambda f: f.stat().st_mtime)
            return pd.to_datetime(datetime.fromtimestamp(most_recent.stat().st_mtime))
    except:
        pass
    
    return None

def get_us_holidays(years):
    """Get major US holidays for given years"""
    holidays = []
    
    for year in years:
        # Fixed date holidays
        holidays.extend([
            (pd.Timestamp(f"{year}-01-01"), "New Year's Day"),
            (pd.Timestamp(f"{year}-07-04"), "Independence Day"),
            (pd.Timestamp(f"{year}-12-25"), "Christmas Day"),
        ])
        
        # Memorial Day (last Monday in May)
        may_last = pd.Timestamp(f"{year}-05-31")
        memorial_day = may_last - pd.Timedelta(days=may_last.weekday())
        holidays.append((memorial_day, "Memorial Day"))
        
        # Labor Day (first Monday in September)  
        sep_first = pd.Timestamp(f"{year}-09-01")
        labor_day = sep_first + pd.Timedelta(days=((7 - sep_first.weekday()) % 7))
        holidays.append((labor_day, "Labor Day"))
        
        # Thanksgiving (fourth Thursday in November)
        nov_first = pd.Timestamp(f"{year}-11-01")
        first_thursday = nov_first + pd.Timedelta(days=((3 - nov_first.weekday()) % 7))
        thanksgiving = first_thursday + pd.Timedelta(days=21)  # Add 3 weeks
        holidays.append((thanksgiving, "Thanksgiving"))
    
    return holidays

def load_week_search_df(selected_folder):
    """Load raw data for week search functionality"""
    if selected_folder:
        # Use same folder path logic as main app
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
        
        combined_path = folder_path / "combined.parquet"
        if combined_path.exists():
            df = pd.read_parquet(combined_path)
            for c in ["dateDepart", "dateReturn", "dateApp"]:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
            return df
    return pd.DataFrame()

# Page config
st.set_page_config(page_title="Trip Cost Exposure Analysis v2.0", layout="wide")

# Sidebar controls
with st.sidebar:
    st.header("Data Source")
    
    # Data folder selection
    data_folders = []
    if Path("_data").exists():
        data_folders = [f.name for f in Path("_data").iterdir() if f.is_dir() and not f.name.startswith(".")]
    
    # Also look in current directory for other data folders
    current_folders = [f.name for f in Path(".").iterdir() if f.is_dir() and not f.name.startswith(".") and f.name != "_data"]
    
    all_folders = data_folders + current_folders
    selected_folder = st.selectbox("Select data folder", [""] + all_folders)
    
    if not selected_folder:
        st.warning("⚠️ Please select a data folder to continue")
        st.stop()
    
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

# Main UI Controls
st.header("Trip Cost Exposure Analysis v2.0")
period = st.selectbox("Time Period", ["day", "week", "month"], index=1)
metric_mode = st.radio("Mode", ["Traveling", "Departures"], index=0)
year_order_choice = st.selectbox("Departure Year order", options=["Ascending", "Descending"], index=0)

# Determine folder path for precomputed data
folder_path = None
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

# Load precomputed data
precomputed_data = None
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
        
        # Aggregate segment data (simplified - always aggregate segments)
        if "segment" in precomputed_data.columns:
            group_cols = ["year", "x", "country_class", "region_class"]
            agg_dict = {}
            for col in ["volume", "maxTripCostExposure", "tripCostPerNightExposure", "avgTripCostPerNight"]:
                if col in precomputed_data.columns:
                    if col == "avgTripCostPerNight":
                        agg_dict[col] = "mean"
                    else:
                        agg_dict[col] = "sum"
            
            precomputed_data = precomputed_data.groupby(group_cols, as_index=False).agg(agg_dict)
            st.sidebar.info("📊 Aggregated all segments")
    else:
        st.error(f"❌ Precomputed {period} {metric_mode.lower()} data not found at: {agg_file}")
        st.error("Please run the following command in your terminal:")
        st.code(f"python test_precomputed.py _data/{selected_folder}")
        st.stop()

# Metric selector
metric_options = ["volume", "maxTripCostExposure", "tripCostPerNightExposure", "avgTripCostPerNight"]
selected_metric = st.selectbox("Metric to plot", options=metric_options, index=1)

# Filter and process data
if precomputed_data is not None:
    filtered_data = filter_aggregate_data(precomputed_data, selected_countries, selected_regions)
    
    if filtered_data.empty:
        st.warning("⚠️ No data matches the selected filters")
        st.stop()
    
    st.sidebar.caption(f"📈 Showing {len(filtered_data):,} records after filtering")
    
    # Define years for plot ordering
    years = sorted(filtered_data["year"].unique().tolist())
    if year_order_choice == "Descending":
        years = list(reversed(years))
    
    # Aggregate filtered_data by plotting dimensions (x, year)
    plot_group_cols = ["year", "x"]
    
    # Create aggregation dictionary for all available metrics
    plot_agg_dict = {}
    for col in ["volume", "maxTripCostExposure", "tripCostPerNightExposure", "avgTripCostPerNight"]:
        if col in filtered_data.columns:
            if col == "avgTripCostPerNight":
                plot_agg_dict[col] = "mean"
            else:
                plot_agg_dict[col] = "sum"
    
    # Aggregate data for plotting
    plot_data = filtered_data.groupby(plot_group_cols, as_index=False).agg(plot_agg_dict)
    
    # Apply year ordering to plot data
    plot_data["year"] = pd.Categorical(plot_data["year"], categories=years, ordered=True)
    
    # Create the plot
    ycol = selected_metric
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
    
    # Add vertical lines for extraction date and holidays
    if period == "week":
        # Add extraction date line
        extraction_date = get_extraction_date(folder_path)
        if extraction_date:
            # Use ISO week number directly (since x is now ISO week numbers)
            extraction_week = extraction_date.isocalendar().week
            extraction_year = extraction_date.isocalendar()[0]
            
            # Only add line if the extraction year is in our data
            if extraction_year in years:
                fig.add_vline(
                    x=extraction_week,
                    line_dash="solid",
                    line_color="red",
                    annotation_text=f"Extraction W{extraction_week}",
                    annotation_position="top"
                )
        
        # Add holiday lines (grouped by week to avoid duplicates) - only for current year
        current_year = pd.Timestamp.now().year
        holidays = get_us_holidays([current_year])
        holiday_weeks = {}
        
        for holiday_date, holiday_name in holidays:
            holiday_week = holiday_date.isocalendar().week
            holiday_year = holiday_date.isocalendar()[0]
            
            # Only consider holidays from current year
            if holiday_year == current_year:
                if holiday_week not in holiday_weeks:
                    holiday_weeks[holiday_week] = []
                holiday_weeks[holiday_week].append(holiday_name)
        
        # Add one vertical line per week with all holiday names
        for week_num, holiday_names in holiday_weeks.items():
            # Remove duplicates and join holiday names
            unique_holidays = list(set(holiday_names))
            annotation_text = ", ".join(unique_holidays)
            
            fig.add_vline(
                x=week_num,
                line_dash="dash",
                line_color="gray",
                opacity=0.6,
                annotation_text=annotation_text,
                annotation_position="top",
                annotation_font_size=8
            )
    
    st.plotly_chart(fig, use_container_width=True)
    

    
    # Data table below the plot
    st.markdown("---")
    st.subheader("Data Table: Metric by ISO Week and Year")
    
    if not plot_data.empty and period == "week":
        data_table = plot_data.copy()
        
        # Convert x to ISO week number
        if 'x' in data_table.columns:
            data_table['iso_week'] = data_table['x']
            data_table['iso_year'] = data_table['year']
        
        # Group by week and year, then pivot
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
    
    elif period == "day":
        st.dataframe(plot_data.head(20), use_container_width=True)
        if len(plot_data) > 20:
            st.caption(f"Showing first 20 rows of {len(plot_data)} total records")

# Week search functionality
st.markdown("---")
with st.expander("Week Search", expanded=False):
    if st.button("Load Data for Week Search"):
        with st.spinner("Loading raw data for week search..."):
            week_df = load_week_search_df(selected_folder)
            if not week_df.empty:
                st.session_state.week_search_df = week_df
                st.success(f"✅ Loaded {len(week_df):,} policies for week search")
            else:
                st.error("❌ No raw data found. Check that combined.parquet exists in the selected folder.")
    
    if "week_search_df" in st.session_state and not st.session_state.week_search_df.empty:
        week_df = st.session_state.week_search_df
        
        # Add country and region classification (same as main app)
        from exposure_data import classify_country, classify_region
        
        if "Country" in week_df.columns:
            week_df["country_class"] = week_df["Country"].apply(classify_country)
        else:
            week_df["country_class"] = "null"
        
        if "ZipCode" in week_df.columns:
            week_df["region_class"] = week_df["ZipCode"].apply(classify_region)
        else:
            week_df["region_class"] = "Other"
        
        # Apply country and region filters (same as main app)
        country_mask = pd.Series(False, index=week_df.index)
        if "US" in selected_countries:
            country_mask |= (week_df['country_class'] == 'US')
        if "ROW" in selected_countries:
            country_mask |= (week_df['country_class'] == 'ROW')
        if "null" in selected_countries:
            country_mask |= (week_df['country_class'] == 'null')
        
        region_mask = pd.Series(False, index=week_df.index)
        for region in selected_regions:
            region_mask |= (week_df['region_class'] == region)
        
        # Apply filters
        week_df = week_df[country_mask & region_mask].copy()
        
        years = sorted(week_df["dateDepart"].dt.year.dropna().astype(int).unique().tolist())
        default_year = years[-1] if years else pd.Timestamp.today().year
        
        c1, c2 = st.columns(2)
        with c1:
            sel_year = st.number_input("Year", min_value=1900, max_value=2100, value=int(default_year), step=1)
        with c2:
            sel_week = st.number_input("ISO Week", min_value=1, max_value=53, value=1, step=1)
        
        # Calculate week dates using ISO week logic
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
        st.info("Click 'Load Data for Week Search' to enable week search functionality")