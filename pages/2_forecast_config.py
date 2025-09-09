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
        
        # Generate weeks for this month
        current_week = month_start - pd.to_timedelta(month_start.weekday(), unit='D')  # Monday of first week
        month_end = next_month - pd.to_timedelta(1, unit='D')
        
        weeks_in_month = []
        while current_week <= month_end:
            if current_week >= last_week + timedelta(weeks=1):  # Only future weeks
                weeks_in_month.append(current_week)
            current_week += timedelta(weeks=1)
        
        # Process each segment for this month
        for segment in segment_columns:
            policy_count = row[segment]
            if pd.isna(policy_count) or policy_count == 0:
                continue
            
            # Distribute monthly policy count across weeks for this segment
            if weeks_in_month:
                weekly_volume = int(policy_count / len(weeks_in_month))
                remaining_volume = policy_count % len(weeks_in_month)
                
                for i, week in enumerate(weeks_in_month):
                    # Add remaining volume to first few weeks
                    volume = weekly_volume + (1 if i < remaining_volume else 0)
                    
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
    st.header("Forecast Segment")
    selected_segment = st.selectbox(
        "Select Segment to Forecast",
        options=[],  # Will be populated after loading historical data
        index=0,
        help="Select which segment to focus on for forecasting"
    )
    
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
        label="📥 Download Forecast Data",
        data=csv,
        file_name=f"policy_purchase_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

else:
    st.info("Please select a data folder in the sidebar to begin forecasting.")
