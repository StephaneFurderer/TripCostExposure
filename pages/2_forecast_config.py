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
    """Load historical purchase data for forecasting"""
    if not folder_path:
        return pd.DataFrame()
    
    combined_path = folder_path / "combined.parquet"
    if not combined_path.exists():
        return pd.DataFrame()
    
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
    
    return df

def aggregate_weekly_purchases(df, selected_countries, selected_regions):
    """Aggregate historical purchases by week for forecasting"""
    if df.empty:
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
    
    weekly_purchases = filtered_df.groupby('week_start').agg({
        'idpol': 'count',  # Volume of policies purchased
        'tripCost': ['sum', 'mean'],  # Total and average trip cost
    }).round(2)
    
    # Flatten column names
    weekly_purchases.columns = ['policy_volume', 'total_trip_cost', 'avg_trip_cost']
    weekly_purchases = weekly_purchases.reset_index()
    
    # Add ISO week and year
    weekly_purchases['iso_week'] = weekly_purchases['week_start'].dt.isocalendar().week
    weekly_purchases['iso_year'] = weekly_purchases['week_start'].dt.isocalendar()[0]
    weekly_purchases['year'] = weekly_purchases['week_start'].dt.year
    
    return weekly_purchases

def simple_forecast(historical_data, weeks_ahead=26):
    """Simple forecasting using historical averages and trends"""
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
            'iso_year': forecast_week.isocalendar()[0],
            'year': forecast_week.year,
            'policy_volume': int(projected_volume),
            'avg_trip_cost': avg_cost,
            'total_trip_cost': int(projected_volume * avg_cost),
            'forecast_type': 'simple_trend'
        })
    
    return pd.DataFrame(forecast_data)

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
    
    # Generate forecast
    with st.spinner("Generating forecast..."):
        forecast_df = simple_forecast(weekly_purchases, weeks_ahead)
    
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
