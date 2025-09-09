import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
from datetime import datetime, timedelta
import numpy as np
from exposure_data import load_folder_policies, classify_country, classify_region

def load_historical_purchases(folder_path):
    """Load historical purchase data for forecasting with caching"""
    if not folder_path:
        return pd.DataFrame()
    
    # Check for cached processed data
    cache_path = folder_path / "processed_forecast_data.parquet"
    combined_path = folder_path / "combined.parquet"
    
    if not combined_path.exists():
        return pd.DataFrame()
    
    # Load from cache if it exists and is newer than combined.parquet
    if cache_path.exists():
        cache_time = cache_path.stat().st_mtime
        source_time = combined_path.stat().st_mtime
        
        if cache_time > source_time:
            try:
                df = pd.read_parquet(cache_path)
                st.sidebar.success("✅ Loaded from cache")
                return df
            except Exception as e:
                st.sidebar.warning(f"⚠️ Cache corrupted, reprocessing... ({e})")
    
    # Process data if no cache or cache is outdated
    st.sidebar.info("🔄 Processing data (first time or cache outdated)...")
    
    df = pd.read_parquet(combined_path)
    
    # Ensure date columns are datetime
    for c in ["dateApp", "dateDepart", "dateReturn"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    
    # Add country and region classification
    if "Country" in df.columns:
        df["country_class"] = df["Country"].apply(classify_country)
    else:
        df["country_class"] = "null"
    
    if "ZipCode" in df.columns:
        df["region_class"] = df["ZipCode"].apply(classify_region)
    else:
        df["region_class"] = "Other"
    
    # Cache the processed data
    try:
        df.to_parquet(cache_path, index=False)
        st.sidebar.success("✅ Data processed and cached")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Could not cache data: {e}")
    
    return df

def aggregate_weekly_purchases(df, selected_countries, selected_regions):
    """Aggregate historical purchases by week for forecasting"""
    if df.empty:
        return pd.DataFrame()
    
    # Check required columns exist
    required_cols = ['dateApp', 'country_class', 'region_class']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
        return pd.DataFrame()
    
    # Check for idpol or tripCost columns
    if 'idpol' not in df.columns and 'tripCost' not in df.columns:
        st.error("❌ No policy identifier or cost data found")
        return pd.DataFrame()
    
    # Apply country and region filters
    country_mask = pd.Series(False, index=df.index)
    if "US" in selected_countries:
        country_mask |= (df['country_class'] == 'US')
    if "ROW" in selected_countries:
        country_mask |= (df['country_class'] == 'ROW')
    if "null" in selected_countries:
        country_mask |= (df['country_class'] == 'null')
    
    region_mask = pd.Series(False, index=df.index)
    for region in selected_regions:
        region_mask |= (df['region_class'] == region)
    
    # Apply filters
    filtered_df = df[country_mask & region_mask].copy()
    
    if filtered_df.empty:
        return pd.DataFrame()
    
    # Aggregate by purchase week (dateApp)
    filtered_df['purchase_week'] = filtered_df['dateApp'].dt.to_period('W')
    filtered_df['week_start'] = filtered_df['purchase_week'].dt.start_time
    
    # Build aggregation dictionary based on available columns
    agg_dict = {}
    if 'idpol' in filtered_df.columns:
        agg_dict['idpol'] = 'count'
    else:
        # Use index count if no idpol column
        agg_dict['index'] = 'count'
    
    if 'tripCost' in filtered_df.columns:
        agg_dict['tripCost'] = ['sum', 'mean']
    
    weekly_purchases = filtered_df.groupby('week_start').agg(agg_dict).round(2)
    
    # Flatten column names based on what we aggregated
    if 'idpol' in agg_dict and 'tripCost' in agg_dict:
        weekly_purchases.columns = ['policy_volume', 'total_trip_cost', 'avg_trip_cost']
    elif 'idpol' in agg_dict:
        weekly_purchases.columns = ['policy_volume']
    elif 'tripCost' in agg_dict:
        weekly_purchases.columns = ['total_trip_cost', 'avg_trip_cost']
    else:
        weekly_purchases.columns = ['policy_volume']
    
    weekly_purchases = weekly_purchases.reset_index()
    
    # Add missing columns with defaults if needed
    if 'policy_volume' not in weekly_purchases.columns:
        weekly_purchases['policy_volume'] = 0
    if 'total_trip_cost' not in weekly_purchases.columns:
        weekly_purchases['total_trip_cost'] = 0
    if 'avg_trip_cost' not in weekly_purchases.columns:
        weekly_purchases['avg_trip_cost'] = 0
    
    # Add ISO week and year
    weekly_purchases['iso_week'] = weekly_purchases['week_start'].dt.isocalendar().week
    weekly_purchases['iso_year'] = weekly_purchases['week_start'].dt.isocalendar().year
    weekly_purchases['year'] = weekly_purchases['week_start'].dt.year
    
    return weekly_purchases

def load_external_forecast(folder_path):
    """Load external monthly policy count forecast from CSV"""
    if not folder_path:
        return pd.DataFrame()
    
    forecast_path = folder_path / "pol_count_finance.csv"
    if not forecast_path.exists():
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(forecast_path)
        st.sidebar.success(f"✅ Loaded external forecast: {len(df)} records")
        return df
    except Exception as e:
        st.sidebar.error(f"❌ Error loading forecast CSV: {e}")
        return pd.DataFrame()

def convert_monthly_to_weekly_forecast(monthly_forecast, historical_data, weeks_ahead=26, selected_segment=None):
    """Convert monthly policy count forecast to weekly forecast"""
    if monthly_forecast.empty:
        return simple_forecast_fallback(historical_data, weeks_ahead, selected_segment)
    
    # Get the last week of historical data
    last_week = historical_data['week_start'].max()
    avg_cost = historical_data['avg_trip_cost'].mean() if not historical_data.empty else 0
    
    # Process monthly forecast data - convert YYYYMM to datetime
    monthly_forecast['month'] = pd.to_datetime(monthly_forecast['month'], format='%Y%m')
    monthly_forecast = monthly_forecast.sort_values('month')
    
    # Get segment columns (all columns except 'month')
    all_segment_columns = [col for col in monthly_forecast.columns if col != 'month']
    
    # Filter to selected segment if provided
    if selected_segment and selected_segment in all_segment_columns:
        segment_columns = [selected_segment]
    elif selected_segment:
        st.warning(f"⚠️ Selected segment '{selected_segment}' not found in CSV. Available: {all_segment_columns}")
        return pd.DataFrame()
    else:
        segment_columns = all_segment_columns
    
    # Generate weekly forecast
    forecast_data = []
    model_point_id = 1
    
    for _, row in monthly_forecast.iterrows():
        month_start = row['month'].replace(day=1)
        
        # Get all weeks in this month
        if month_start.month == 12:
            next_month = month_start.replace(year=month_start.year + 1, month=1, day=1)
        else:
            next_month = month_start.replace(month=month_start.month + 1, day=1)
        
        # Generate ALL weeks for this month (including historical ones)
        current_week = month_start - pd.to_timedelta(month_start.weekday(), unit='D')  # Monday of first week
        month_end = next_month - pd.to_timedelta(1, unit='D')
        
        all_weeks_in_month = []
        while current_week <= month_end:
            all_weeks_in_month.append(current_week)
            current_week += timedelta(weeks=1)
        
        # Filter to only future weeks for actual forecast
        future_weeks_in_month = [week for week in all_weeks_in_month if week >= last_week + timedelta(weeks=1)]
        
        # Process each segment for this month
        for segment in segment_columns:
            policy_count = row[segment]
            if pd.isna(policy_count) or policy_count == 0:
                continue
            
            # Distribute monthly policy count across ALL weeks in month, but only forecast future weeks
            if all_weeks_in_month and future_weeks_in_month:
                weekly_volume = int(policy_count / len(all_weeks_in_month))
                remaining_volume = policy_count % len(all_weeks_in_month)
                
                for i, week in enumerate(future_weeks_in_month):
                    # Find the position of this week in the full month
                    week_position = all_weeks_in_month.index(week)
                    # Add remaining volume to first few weeks of the month
                    volume = weekly_volume + (1 if week_position < remaining_volume else 0)
                    
                    forecast_data.append({
                        'model_point_id': model_point_id,
                        'week_purchased': week,
                        'iso_week': week.isocalendar().week,
                        'iso_year': week.isocalendar().year,
                        'year': week.year,
                        'policy_volume': volume,
                        'avg_trip_cost': avg_cost,
                        'total_trip_cost': int(volume * avg_cost),
                        'segment': segment,
                        'forecast_type': 'external_csv'
                    })
                    model_point_id += 1
    
    return pd.DataFrame(forecast_data)

def simple_forecast_fallback(historical_data, weeks_ahead=26, selected_segment=None):
    """Fallback simple forecasting using historical averages and trends"""
    if historical_data.empty:
        return pd.DataFrame()
    
    # Get the last week of historical data
    last_week = historical_data['week_start'].max()
    
    # Calculate simple statistics
    avg_volume = historical_data['policy_volume'].mean()
    recent_avg = historical_data.tail(12)['policy_volume'].mean()  # Last 12 weeks
    avg_cost = historical_data['avg_trip_cost'].mean()
    
    # Simple trend calculation (last 12 weeks vs previous 12 weeks)
    if len(historical_data) >= 24:
        recent_trend = recent_avg - historical_data.tail(24).head(12)['policy_volume'].mean()
    else:
        recent_trend = 0
    
    # Generate forecast
    forecast_data = []
    for i in range(1, weeks_ahead + 1):
        forecast_week = last_week + timedelta(weeks=i)
        
        # Simple linear trend projection
        projected_volume = max(0, recent_avg + (recent_trend * i / 12))
        
        forecast_data.append({
            'model_point_id': i,
            'week_purchased': forecast_week,
            'iso_week': forecast_week.isocalendar().week,
            'iso_year': forecast_week.isocalendar().year,
            'year': forecast_week.year,
            'policy_volume': int(projected_volume),
            'avg_trip_cost': avg_cost,
            'total_trip_cost': int(projected_volume * avg_cost),
            'segment': selected_segment or 'all',
            'forecast_type': 'simple_trend'
        })
    
    return pd.DataFrame(forecast_data)

def analyze_departure_patterns(historical_df, selected_segment=None, max_depart_weeks=52, folder_path=None):
    """
    Analyze historical departure patterns by looking at same week ±1 week from previous year
    Includes trip length and per-night cost analysis for accurate forecasting
    Caches results to speed up subsequent runs
    
    Parameters:
    - historical_df: DataFrame with historical purchase and departure data
    - selected_segment: Segment to analyze (None for all segments)
    - max_depart_weeks: Maximum weeks ahead to analyze departure patterns
    - folder_path: Path to data folder for caching results
    
    Returns:
    - DataFrame with departure distribution patterns including trip metrics
    """
    if historical_df.empty or 'dateDepart' not in historical_df.columns:
        return pd.DataFrame()
    
    # Check for cached departure patterns
    if folder_path:
        cache_file = folder_path / f"departure_patterns_{selected_segment or 'all'}.parquet"
        if cache_file.exists():
            try:
                cached_patterns = pd.read_parquet(cache_file)
                st.sidebar.success(f"✅ Loaded cached departure patterns for {selected_segment or 'all'}")
                return cached_patterns
            except Exception as e:
                st.sidebar.warning(f"⚠️ Could not load cached patterns: {e}")
    
    # Filter by segment if specified
    if selected_segment and 'segment' in historical_df.columns:
        df = historical_df[historical_df['segment'] == selected_segment].copy()
    else:
        df = historical_df.copy()
    
    if df.empty:
        return pd.DataFrame()
    
    # Ensure date columns are datetime
    df['dateApp'] = pd.to_datetime(df['dateApp'], errors='coerce')
    df['dateDepart'] = pd.to_datetime(df['dateDepart'], errors='coerce')
    df['dateReturn'] = pd.to_datetime(df['dateReturn'], errors='coerce')
    
    # Calculate trip metrics
    df['weeks_to_depart'] = ((df['dateDepart'] - df['dateApp']).dt.days / 7).round().astype(int)
    df['trip_length_days'] = (df['dateReturn'] - df['dateDepart']).dt.days
    
    # Calculate per-night cost if tripCost is available
    if 'tripCost' in df.columns:
        df['trip_cost_per_night'] = df['tripCost'] / df['trip_length_days'].replace(0, 1)  # Avoid division by zero
    else:
        df['trip_cost_per_night'] = 0
    
    # Filter out invalid data
    df = df[
        (df['weeks_to_depart'] >= 0) & 
        (df['weeks_to_depart'] <= max_depart_weeks) &
        (df['trip_length_days'] > 0) &
        (df['trip_length_days'] <= 365)  # Reasonable trip length limit
    ]
    
    if df.empty:
        return pd.DataFrame()
    
    # Group by purchase week and analyze departure patterns
    departure_patterns = []
    
    for purchase_week in df['dateApp'].dt.to_period('W-MON').unique():
        week_data = df[df['dateApp'].dt.to_period('W-MON') == purchase_week]
        
        # Get the same week from previous year ±1 week
        purchase_date = purchase_week.start_time
        prev_year_week = purchase_date.replace(year=purchase_date.year - 1)
        
        # Look at 3 weeks around the same time last year
        # Convert dates to periods first, then get start_time
        df_periods = df['dateApp'].dt.to_period('W-MON')
        period_start_times = df_periods.dt.start_time
        
        prev_year_data = df[
            (period_start_times >= prev_year_week - pd.Timedelta(weeks=1)) &
            (period_start_times <= prev_year_week + pd.Timedelta(weeks=1))
        ]
        
        if prev_year_data.empty:
            continue
        
        # Calculate departure distribution for this purchase week
        total_policies = len(week_data)
        if total_policies == 0:
            continue
        
        # Get departure distribution from previous year's similar period
        prev_departure_dist = prev_year_data['weeks_to_depart'].value_counts().sort_index()
        
        # Calculate trip metrics by weeks_to_depart for previous year data
        trip_metrics_by_weeks = {}
        for weeks_ahead in prev_departure_dist.index:
            weeks_data = prev_year_data[prev_year_data['weeks_to_depart'] == weeks_ahead]
            if len(weeks_data) > 0:
                avg_trip_length = weeks_data['trip_length_days'].mean()
                avg_cost_per_night = weeks_data['trip_cost_per_night'].mean()
                trip_metrics_by_weeks[weeks_ahead] = {
                    'avg_trip_length_days': avg_trip_length,
                    'avg_cost_per_night': avg_cost_per_night
                }
        
        # Normalize to get proportions
        departure_proportions = {}
        for weeks_ahead, count in prev_departure_dist.items():
            proportion = count / len(prev_year_data)
            departure_proportions[weeks_ahead] = {
                'proportion': proportion,
                'trip_metrics': trip_metrics_by_weeks.get(weeks_ahead, {
                    'avg_trip_length_days': 7,  # Default fallback
                    'avg_cost_per_night': 100   # Default fallback
                })
            }
        
        departure_patterns.append({
            'purchase_week': purchase_week.start_time,
            'total_policies': total_policies,
            'departure_distribution': departure_proportions
        })
    
    patterns_df = pd.DataFrame(departure_patterns)
    
    # Cache the results if folder_path is provided
    if folder_path and not patterns_df.empty:
        try:
            cache_file = folder_path / f"departure_patterns_{selected_segment or 'all'}.parquet"
            patterns_df.to_parquet(cache_file, index=False)
            st.sidebar.success(f"✅ Cached departure patterns for {selected_segment or 'all'}")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Could not cache patterns: {e}")
    
    return patterns_df

def create_departure_forecast(purchase_forecast, departure_patterns, selected_segment=None, folder_path=None):
    """
    Create departure forecast based on purchase forecast and historical patterns
    Includes trip length and per-night cost calculations
    
    Parameters:
    - purchase_forecast: DataFrame with weekly purchase forecasts
    - departure_patterns: DataFrame with historical departure patterns
    - selected_segment: Segment being forecasted
    - folder_path: Path to data folder for caching results
    
    Returns:
    - DataFrame with departure forecasts by week including trip metrics
    """
    if purchase_forecast.empty or departure_patterns.empty:
        return pd.DataFrame()
    
    # Check for cached departure forecast
    if folder_path:
        cache_file = folder_path / f"departure_forecast_{selected_segment or 'all'}.parquet"
        if cache_file.exists():
            try:
                cached_forecast = pd.read_parquet(cache_file)
                st.sidebar.success(f"✅ Loaded cached departure forecast for {selected_segment or 'all'}")
                return cached_forecast
            except Exception as e:
                st.sidebar.warning(f"⚠️ Could not load cached forecast: {e}")
    
    departure_forecast = []
    
    for _, purchase_row in purchase_forecast.iterrows():
        purchase_week = purchase_row['week_purchased']
        policy_volume = purchase_row['policy_volume']
        
        # Find matching historical pattern (same week of year)
        purchase_period = purchase_week.to_period('W-MON')
        matching_patterns = departure_patterns[
            departure_patterns['purchase_week'].dt.to_period('W-MON') == purchase_period
        ]
        
        if matching_patterns.empty:
            # Use average pattern if no exact match
            avg_distribution = {}
            for _, pattern in departure_patterns.iterrows():
                for weeks_ahead, data in pattern['departure_distribution'].items():
                    if weeks_ahead not in avg_distribution:
                        avg_distribution[weeks_ahead] = {
                            'proportions': [],
                            'trip_lengths': [],
                            'costs_per_night': []
                        }
                    avg_distribution[weeks_ahead]['proportions'].append(data['proportion'])
                    avg_distribution[weeks_ahead]['trip_lengths'].append(data['trip_metrics']['avg_trip_length_days'])
                    avg_distribution[weeks_ahead]['costs_per_night'].append(data['trip_metrics']['avg_cost_per_night'])
            
            # Calculate average proportions and metrics
            for weeks_ahead in avg_distribution:
                avg_distribution[weeks_ahead] = {
                    'proportion': sum(avg_distribution[weeks_ahead]['proportions']) / len(avg_distribution[weeks_ahead]['proportions']),
                    'trip_metrics': {
                        'avg_trip_length_days': sum(avg_distribution[weeks_ahead]['trip_lengths']) / len(avg_distribution[weeks_ahead]['trip_lengths']),
                        'avg_cost_per_night': sum(avg_distribution[weeks_ahead]['costs_per_night']) / len(avg_distribution[weeks_ahead]['costs_per_night'])
                    }
                }
        else:
            # Use the first matching pattern
            avg_distribution = matching_patterns.iloc[0]['departure_distribution']
        
        # Apply distribution to forecasted policy volume
        for weeks_ahead, data in avg_distribution.items():
            depart_week = purchase_week + pd.Timedelta(weeks=weeks_ahead)
            depart_volume = int(policy_volume * data['proportion'])
            
            if depart_volume > 0:
                # Calculate trip metrics
                avg_trip_length = data['trip_metrics']['avg_trip_length_days']
                avg_cost_per_night = data['trip_metrics']['avg_cost_per_night']
                total_trip_cost = depart_volume * avg_trip_length * avg_cost_per_night
                
                # Calculate return date (handles decimal trip lengths)
                # Ensure trip length is valid and convert to float
                trip_length_days = float(avg_trip_length) if pd.notna(avg_trip_length) and avg_trip_length > 0 else 7.0
                return_week = depart_week + pd.Timedelta(days=trip_length_days)
                
                departure_forecast.append({
                    'purchase_week': purchase_week,
                    'depart_week': depart_week,
                    'return_week': return_week,
                    'weeks_to_depart': weeks_ahead,
                    'policy_volume': depart_volume,
                    'avg_trip_length_days': round(avg_trip_length, 1),
                    'avg_cost_per_night': round(avg_cost_per_night, 2),
                    'total_trip_cost': round(total_trip_cost, 2),
                    'segment': selected_segment or 'all',
                    'forecast_type': 'departure_forecast'
                })
    
    forecast_df = pd.DataFrame(departure_forecast)
    
    # Cache the results if folder_path is provided
    if folder_path and not forecast_df.empty:
        try:
            cache_file = folder_path / f"departure_forecast_{selected_segment or 'all'}.parquet"
            forecast_df.to_parquet(cache_file, index=False)
            st.sidebar.success(f"✅ Cached departure forecast for {selected_segment or 'all'}")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Could not cache forecast: {e}")
    
    return forecast_df

def calculate_traveling_policies_by_week(departure_forecast_df, start_date, end_date, folder_path=None, selected_segment=None):
    """
    Calculate traveling policies for each week in the forecast period
    
    Parameters:
    - departure_forecast_df: DataFrame with departure forecasts including return_week
    - start_date: Start date for the analysis period
    - end_date: End date for the analysis period
    - folder_path: Path to data folder for caching results
    - selected_segment: Segment being analyzed
    
    Returns:
    - DataFrame with weekly traveling policy counts
    """
    if departure_forecast_df.empty:
        return pd.DataFrame()
    
    # Check for cached traveling policies
    if folder_path:
        cache_file = folder_path / f"traveling_policies_{selected_segment or 'all'}.parquet"
        if cache_file.exists():
            try:
                cached_traveling = pd.read_parquet(cache_file)
                st.sidebar.success(f"✅ Loaded cached traveling policies for {selected_segment or 'all'}")
                return cached_traveling
            except Exception as e:
                st.sidebar.warning(f"⚠️ Could not load cached traveling policies: {e}")
    
    # Generate all weeks in the period
    weeks = pd.date_range(start=start_date, end=end_date, freq='W-MON')
    
    traveling_by_week = []
    
    for week in weeks:
        # Policies that departed before or during this week and return after this week
        traveling_policies = departure_forecast_df[
            (departure_forecast_df['depart_week'] <= week) & 
            (departure_forecast_df['return_week'] > week)
        ]
        
        total_traveling = traveling_policies['policy_volume'].sum()
        total_trip_cost = traveling_policies['total_trip_cost'].sum()
        
        traveling_by_week.append({
            'week': week,
            'traveling_policies': total_traveling,
            'total_trip_cost': total_trip_cost,
            'avg_trip_length': traveling_policies['avg_trip_length_days'].mean() if len(traveling_policies) > 0 else 0,
            'avg_cost_per_night': traveling_policies['avg_cost_per_night'].mean() if len(traveling_policies) > 0 else 0
        })
    
    traveling_df = pd.DataFrame(traveling_by_week)
    
    # Cache the results if folder_path is provided
    if folder_path and not traveling_df.empty:
        try:
            cache_file = folder_path / f"traveling_policies_{selected_segment or 'all'}.parquet"
            traveling_df.to_parquet(cache_file, index=False)
            st.sidebar.success(f"✅ Cached traveling policies for {selected_segment or 'all'}")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Could not cache traveling policies: {e}")
    
    return traveling_df

def simple_forecast(historical_data, weeks_ahead=26, folder_path=None, selected_segment=None):
    """Main forecasting function - tries external CSV first, falls back to simple trend"""
    if historical_data.empty:
        return pd.DataFrame()
    
    # Try to load external forecast first
    if folder_path:
        external_forecast = load_external_forecast(folder_path)
        if not external_forecast.empty:
            return convert_monthly_to_weekly_forecast(external_forecast, historical_data, weeks_ahead, selected_segment)
    
    # Fallback to simple trend forecast
    st.sidebar.info("📊 Using simple trend forecast (no external CSV found)")
    return simple_forecast_fallback(historical_data, weeks_ahead, selected_segment)

# Page config
st.set_page_config(page_title="Forecast Configuration", layout="wide")

st.header("📈 Trip Cost Exposure Forecasting")
st.markdown("**Step 1: Policy Purchase Volume Forecasting**")

# Sidebar controls
with st.sidebar:
    st.header("Data Source")
    
    # Data folder selection
    data_folders = []
    if Path("_data").exists():
        data_folders = [f.name for f in Path("_data").iterdir() if f.is_dir() and not f.name.startswith(".")]
    
    #current_folders = [f.name for f in Path(".").iterdir() if f.is_dir() and not f.name.startswith(".") and f.name != "_data"]
    all_folders = data_folders #+ current_folders
    
    selected_folder = st.selectbox("Select data folder", all_folders,index = 0)
    
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
    
    # Segment filter (will be populated after loading data)
    # st.header("Forecast Segment")
    # selected_segment = st.selectbox(
    #     "Select Segment to Forecast",
    #     options=[],  # Will be populated after loading historical data
    #     index=0,
    #     help="Select which segment to focus on for forecasting"
    # )
    
    # Forecast parameters
    st.header("Forecast Parameters")
    weeks_ahead = st.slider("Weeks to forecast", min_value=4, max_value=52, value=26, step=2)
    st.caption(f"Forecasting {weeks_ahead} weeks ahead")

# Determine folder path for data loading
folder_path = None
if selected_folder:
    if Path(f"_data/{selected_folder}").exists():
        base_folder = Path(f"_data/{selected_folder}")
    else:
        base_folder = Path(selected_folder)
    
    # Look for date subfolder
    folder_contents = [f for f in base_folder.iterdir() if f.is_dir() and f.name[0].isdigit()]
    if folder_contents:
        folder_path = sorted(folder_contents)[-1]
    else:
        folder_path = base_folder

# Load and process data
if folder_path:
    with st.spinner("Loading historical data..."):
        historical_df = load_historical_purchases(folder_path)
    
    if historical_df.empty:
        st.error("❌ No historical data found. Check that combined.parquet exists.")
        st.stop()
    
    st.success(f"✅ Loaded {len(historical_df):,} historical policies")
    
    # Aggregate weekly purchases
    with st.spinner("Processing weekly purchase data..."):
        weekly_purchases = aggregate_weekly_purchases(historical_df, selected_countries, selected_regions)
    
    if weekly_purchases.empty:
        st.warning("⚠️ No data matches the selected filters")
        st.stop()
    
    st.success(f"✅ Processed {len(weekly_purchases)} weeks of purchase data")
    
    # Show historical data summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Weeks", len(weekly_purchases))
    with col2:
        st.metric("Avg Weekly Volume", f"{weekly_purchases['policy_volume'].mean():.0f}")
    with col3:
        st.metric("Avg Trip Cost", f"${weekly_purchases['avg_trip_cost'].mean():,.0f}")
    with col4:
        st.metric("Total Volume", f"{weekly_purchases['policy_volume'].sum():,}")
    
    # Get available segments from historical data
    available_segments = []
    if 'segment' in historical_df.columns:
        available_segments = sorted(historical_df['segment'].dropna().unique().tolist())
        st.session_state.available_segments = available_segments
    
    # Update segment selection if we have segments
    if 'available_segments' in st.session_state and st.session_state.available_segments:
        # Update the segment selectbox with available segments
        st.sidebar.selectbox(
            "Select Segment to Forecast",
            options=st.session_state.available_segments,
            index=0,
            key="segment_selector"
        )
        selected_segment = st.session_state.get('segment_selector', st.session_state.available_segments[0] if st.session_state.available_segments else None)
    else:
        selected_segment = None
        st.warning("⚠️ No segment data found in historical data")
    
    # Generate forecast
    with st.spinner("Generating forecast..."):
        forecast_df = simple_forecast(weekly_purchases, weeks_ahead, folder_path, selected_segment)
    
    if forecast_df.empty:
        st.error("❌ Could not generate forecast")
        st.stop()
    
    # Analyze departure patterns
    with st.spinner("Analyzing departure patterns..."):
        departure_patterns = analyze_departure_patterns(historical_df, selected_segment, folder_path=folder_path)
    
    # Create departure forecast
    departure_forecast_df = pd.DataFrame()
    if not departure_patterns.empty:
        departure_forecast_df = create_departure_forecast(forecast_df, departure_patterns, selected_segment, folder_path)
    
    # Calculate traveling policies by week
    traveling_by_week_df = pd.DataFrame()
    if not departure_forecast_df.empty:
        # Get the forecast period
        forecast_start = forecast_df['week_purchased'].min()
        forecast_end = departure_forecast_df['return_week'].max()
        traveling_by_week_df = calculate_traveling_policies_by_week(departure_forecast_df, forecast_start, forecast_end, folder_path, selected_segment)
    
    st.success(f"✅ Generated forecast for {len(forecast_df)} weeks")
    
    # Display forecast results
    st.markdown("---")
    st.subheader("📊 Forecast Results")
    
    # Show forecast summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Forecast Weeks", len(forecast_df))
    with col2:
        st.metric("Avg Forecast Volume", f"{forecast_df['policy_volume'].mean():.0f}")
    with col3:
        st.metric("Total Forecast Volume", f"{forecast_df['policy_volume'].sum():,}")
    with col4:
        st.metric("Total Forecast Cost", f"${forecast_df['total_trip_cost'].sum():,.0f}")
    
    # Plot historical vs forecast
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=weekly_purchases['week_start'],
        y=weekly_purchases['policy_volume'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='blue', width=2)
    ))
    
    # Forecast data
    fig.add_trace(go.Scatter(
        x=forecast_df['week_purchased'],
        y=forecast_df['policy_volume'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="Weekly Policy Purchase Volume: Historical vs Forecast",
        xaxis_title="Week",
        yaxis_title="Policy Volume",
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show forecast table
    st.markdown("---")
    st.subheader("📋 Forecast Details")
    
    display_cols = ['model_point_id', 'week_purchased', 'iso_week', 'year', 'policy_volume', 'avg_trip_cost', 'total_trip_cost']
    st.dataframe(
        forecast_df[display_cols].round(2),
        use_container_width=True,
        hide_index=True
    )
    
    # Download forecast data
    csv = forecast_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Purchase Forecast Data",
        data=csv,
        file_name=f"policy_purchase_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
    # Departure forecast section
    if not departure_forecast_df.empty:
        st.markdown("---")
        st.subheader("🚀 Departure Forecast")
        
        # Aggregate departure forecast by depart week
        departure_weekly = departure_forecast_df.groupby('depart_week').agg({
            'policy_volume': 'sum'
        }).reset_index()
        departure_weekly = departure_weekly.sort_values('depart_week')
        
        # Departure forecast summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Departure Volume", f"{departure_forecast_df['policy_volume'].sum():,}")
        with col2:
            st.metric("Avg Weekly Departures", f"{departure_weekly['policy_volume'].mean():.0f}")
        with col3:
            st.metric("Peak Departure Week", f"{departure_weekly['policy_volume'].max():,}")
        with col4:
            st.metric("Total Trip Cost", f"${departure_forecast_df['total_trip_cost'].sum():,.0f}")
        
        # Additional trip metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Trip Length", f"{departure_forecast_df['avg_trip_length_days'].mean():.1f} days")
        with col2:
            st.metric("Avg Cost/Night", f"${departure_forecast_df['avg_cost_per_night'].mean():.0f}")
        with col3:
            st.metric("Forecast Period", f"{departure_weekly['depart_week'].min().strftime('%Y-%m-%d')} to {departure_weekly['depart_week'].max().strftime('%Y-%m-%d')}")
        with col4:
            st.metric("Total Nights", f"{departure_forecast_df['policy_volume'].sum() * departure_forecast_df['avg_trip_length_days'].mean():,.0f}")
        
        # Departure forecast chart
        fig_departure = go.Figure()
        fig_departure.add_trace(go.Scatter(
            x=departure_weekly['depart_week'],
            y=departure_weekly['policy_volume'],
            mode='lines+markers',
            name='Departure Forecast',
            line=dict(color='green', width=2)
        ))
        
        fig_departure.update_layout(
            title=f"Policy Departure Forecast - {selected_segment or 'All Segments'}",
            xaxis_title="Departure Week",
            yaxis_title="Departure Volume",
            hovermode='x unified',
            legend=dict(x=0.02, y=0.98)
        )
        
        st.plotly_chart(fig_departure, use_container_width=True)
        
        # Trip cost forecast chart
        departure_cost_weekly = departure_forecast_df.groupby('depart_week').agg({
            'total_trip_cost': 'sum'
        }).reset_index()
        departure_cost_weekly = departure_cost_weekly.sort_values('depart_week')
        
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Scatter(
            x=departure_cost_weekly['depart_week'],
            y=departure_cost_weekly['total_trip_cost'],
            mode='lines+markers',
            name='Trip Cost Forecast',
            line=dict(color='orange', width=2)
        ))
        
        fig_cost.update_layout(
            title=f"Trip Cost Forecast - {selected_segment or 'All Segments'}",
            xaxis_title="Departure Week",
            yaxis_title="Total Trip Cost ($)",
            hovermode='x unified',
            legend=dict(x=0.02, y=0.98)
        )
        
        st.plotly_chart(fig_cost, use_container_width=True)
        
        # Traveling policies analysis
        if not traveling_by_week_df.empty:
            st.subheader("✈️ Traveling Policies by Week")
            
            # Traveling policies summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Peak Traveling Week", f"{traveling_by_week_df['traveling_policies'].max():,}")
            with col2:
                st.metric("Avg Traveling Policies", f"{traveling_by_week_df['traveling_policies'].mean():.0f}")
            with col3:
                st.metric("Total Traveling Cost", f"${traveling_by_week_df['total_trip_cost'].sum():,.0f}")
            with col4:
                st.metric("Peak Traveling Cost", f"${traveling_by_week_df['total_trip_cost'].max():,.0f}")
            
            # Traveling policies chart
            fig_traveling = go.Figure()
            fig_traveling.add_trace(go.Scatter(
                x=traveling_by_week_df['week'],
                y=traveling_by_week_df['traveling_policies'],
                mode='lines+markers',
                name='Traveling Policies',
                line=dict(color='purple', width=2)
            ))
            
            fig_traveling.update_layout(
                title=f"Traveling Policies by Week - {selected_segment or 'All Segments'}",
                xaxis_title="Week",
                yaxis_title="Number of Traveling Policies",
                hovermode='x unified',
                legend=dict(x=0.02, y=0.98)
            )
            
            st.plotly_chart(fig_traveling, use_container_width=True)
            
            # Traveling policies cost chart
            fig_traveling_cost = go.Figure()
            fig_traveling_cost.add_trace(go.Scatter(
                x=traveling_by_week_df['week'],
                y=traveling_by_week_df['total_trip_cost'],
                mode='lines+markers',
                name='Traveling Trip Cost',
                line=dict(color='brown', width=2)
            ))
            
            fig_traveling_cost.update_layout(
                title=f"Traveling Trip Cost by Week - {selected_segment or 'All Segments'}",
                xaxis_title="Week",
                yaxis_title="Total Trip Cost ($)",
                hovermode='x unified',
                legend=dict(x=0.02, y=0.98)
            )
            
            st.plotly_chart(fig_traveling_cost, use_container_width=True)
            
            # Traveling policies table
            st.subheader("📋 Traveling Policies by Week")
            st.dataframe(traveling_by_week_df, use_container_width=True)
        
        # Departure distribution analysis
        st.subheader("📊 Departure Distribution Analysis")
        
        # Show distribution by weeks to depart with trip metrics
        depart_dist = departure_forecast_df.groupby('weeks_to_depart').agg({
            'policy_volume': 'sum',
            'avg_trip_length_days': 'mean',
            'avg_cost_per_night': 'mean',
            'total_trip_cost': 'sum'
        }).reset_index()
        depart_dist['proportion'] = depart_dist['policy_volume'] / depart_dist['policy_volume'].sum()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution chart
            fig_dist = px.bar(
                depart_dist, 
                x='weeks_to_depart', 
                y='policy_volume',
                title="Departure Distribution by Weeks Ahead",
                labels={'policy_volume': 'Policy Volume', 'weeks_to_depart': 'Weeks to Depart'}
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Trip length distribution
            fig_length = px.bar(
                depart_dist, 
                x='weeks_to_depart', 
                y='avg_trip_length_days',
                title="Average Trip Length by Weeks to Depart",
                labels={'avg_trip_length_days': 'Avg Trip Length (days)', 'weeks_to_depart': 'Weeks to Depart'}
            )
            st.plotly_chart(fig_length, use_container_width=True)
        
        # Distribution table with trip metrics
        st.subheader("Distribution Summary with Trip Metrics")
        display_cols = ['weeks_to_depart', 'policy_volume', 'proportion', 'avg_trip_length_days', 'avg_cost_per_night', 'total_trip_cost']
        st.dataframe(
            depart_dist[display_cols].round(3),
            use_container_width=True
        )
        
        # Detailed departure forecast table
        st.subheader("📋 Detailed Departure Forecast")
        st.dataframe(departure_forecast_df, use_container_width=True)
        
        # Download departure forecast
        departure_csv = departure_forecast_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Departure Forecast CSV",
            data=departure_csv,
            file_name=f"departure_forecast_{selected_segment or 'all'}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ No departure patterns found in historical data. Cannot generate departure forecast.")

else:
    st.info("Please select a data folder in the sidebar to begin forecasting.")
