import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import json

st.set_page_config(page_title="Forecast Configuration", page_icon="📊", layout="wide")

def load_historical_purchases(folder_path):
    """Load and process historical purchase data with caching"""
    cache_file = folder_path / "processed_forecast_data.parquet"
    
    if cache_file.exists():
        try:
            cached_data = pd.read_parquet(cache_file)
            st.sidebar.success("✅ Loaded cached processed data")
            return cached_data
        except Exception as e:
            st.sidebar.warning(f"⚠️ Could not load cached data: {e}")
    
    # Load combined data
    combined_file = folder_path / "combined.parquet"
    if not combined_file.exists():
        st.error(f"❌ Combined data file not found: {combined_file}")
        return pd.DataFrame()
    
    df = pd.read_parquet(combined_file)
    
    # Process data
    df['dateApp'] = pd.to_datetime(df['dateApp'], errors='coerce')
    df['dateDepart'] = pd.to_datetime(df['dateDepart'], errors='coerce')
    df['dateReturn'] = pd.to_datetime(df['dateReturn'], errors='coerce')
    
    # Filter valid data
    df = df.dropna(subset=['dateApp', 'dateDepart', 'dateReturn'])
    df = df[df['dateDepart'] >= df['dateApp']]
    df = df[df['dateReturn'] >= df['dateDepart']]
    
    # Cache processed data
    try:
        df.to_parquet(cache_file, index=False)
        st.sidebar.success("✅ Cached processed data")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Could not cache data: {e}")
    
    return df

def aggregate_weekly_purchases(df, selected_countries, selected_regions):
    """Aggregate weekly purchase data"""
    if df.empty:
        return pd.DataFrame()
    
    # Apply filters
    if selected_countries:
        df = df[df['country'].isin(selected_countries)]
    if selected_regions:
        df = df[df['region'].isin(selected_regions)]
    
    # Group by week
    df['week_start'] = df['dateApp'].dt.to_period('W-MON').dt.start_time
    weekly_purchases = df.groupby('week_start').agg({
        'idpol': 'count',
        'tripCost': 'sum'
    }).reset_index()
    
    weekly_purchases.columns = ['week_start', 'policy_volume', 'total_trip_cost']
    weekly_purchases['avg_trip_cost'] = weekly_purchases['total_trip_cost'] / weekly_purchases['policy_volume']
    weekly_purchases['iso_week'] = weekly_purchases['week_start'].dt.isocalendar().week
    weekly_purchases['year'] = weekly_purchases['week_start'].dt.year
    
    return weekly_purchases

def load_external_forecast(folder_path):
    """Load external monthly forecast data"""
    forecast_file = folder_path / "pol_count_finance.csv"
    if not forecast_file.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(forecast_file)
    return df

def convert_monthly_to_weekly_forecast(monthly_forecast, historical_data, weeks_ahead=104, selected_segment=None):
    """Convert monthly forecast to weekly using historical distribution"""
    if monthly_forecast.empty or historical_data.empty:
        return pd.DataFrame()
    
    # Get last historical week
    last_week = historical_data['week_start'].max()
    
    forecast_data = []
    model_point_id = 1
    weeks_generated = 0
    
    for _, row in monthly_forecast.iterrows():
        # Stop if we've reached the weeks_ahead limit
        if weeks_generated >= weeks_ahead:
            break
            
        month_str = str(row['month'])
        year = int(month_str[:4])
        month = int(month_str[4:])
        
        # Get policy count for selected segment
        if selected_segment and selected_segment in row:
            policy_count = row[selected_segment]
        else:
            # Use first available segment column
            segment_cols = [col for col in monthly_forecast.columns if col != 'month']
            if segment_cols:
                policy_count = row[segment_cols[0]]
            else:
                continue
        
        if pd.isna(policy_count) or policy_count <= 0:
            continue
        
        # Create month start and end dates
        month_start = pd.Timestamp(year=year, month=month, day=1)
        next_month = month_start + pd.DateOffset(months=1)
        
        # Generate ALL weeks for this month (including historical ones)
        current_week = month_start - pd.to_timedelta(month_start.weekday(), unit='D')
        month_end = next_month - pd.to_timedelta(1, unit='D')
        all_weeks_in_month = []
        while current_week <= month_end:
            all_weeks_in_month.append(current_week)
            current_week += timedelta(weeks=1)
        
        # Filter to only future weeks for actual forecast
        future_weeks_in_month = [week for week in all_weeks_in_month if week >= last_week + timedelta(weeks=1)]
        
        # Distribute monthly policy count across ALL weeks in month, but only forecast future weeks
        if all_weeks_in_month and future_weeks_in_month:
            weekly_volume = int(policy_count / len(all_weeks_in_month))
            remaining_volume = policy_count % len(all_weeks_in_month)
            
            for i, week in enumerate(future_weeks_in_month):
                # Stop if we've reached the weeks_ahead limit
                if weeks_generated >= weeks_ahead:
                    break
                    
                week_position = all_weeks_in_month.index(week)
                volume = weekly_volume + (1 if week_position < remaining_volume else 0)
                
                if volume > 0:
                    forecast_data.append({
                        'model_point_id': model_point_id,
                        'week_purchased': week,
                        'policy_volume': volume,
                        'iso_week': week.isocalendar().week,
                        'year': week.year,
                        'avg_trip_cost': 0,  # Placeholder
                        'total_trip_cost': 0  # Placeholder
                    })
                    model_point_id += 1  # Increment for each week
                    weeks_generated += 1  # Track weeks generated
    
    return pd.DataFrame(forecast_data)


def analyze_historical_trip_costs_by_week(historical_df, selected_segment=None, folder_path=None):
    """
    Analyze historical trip costs by purchase week for trend and seasonality analysis
    
    Parameters:
    - historical_df: DataFrame with historical purchase data
    - selected_segment: Segment to analyze
    - folder_path: Path for caching
    
    Returns:
    - DataFrame with trip cost analysis by purchase week
    """
    if historical_df.empty or 'tripCost' not in historical_df.columns:
        return pd.DataFrame()
    
    # Check for cached trip cost analysis
    if folder_path:
        cache_file = folder_path / f"trip_cost_analysis_{selected_segment or 'all'}.parquet"
        if cache_file.exists():
            try:
                cached_data = pd.read_parquet(cache_file)
                st.sidebar.success(f"✅ Loaded cached trip cost analysis for {selected_segment or 'all'}")
                return cached_data
            except Exception as e:
                st.sidebar.warning(f"⚠️ Could not load cached trip cost analysis: {e}")
    
    # Filter by segment if specified
    if selected_segment and 'segment' in historical_df.columns:
        df = historical_df[historical_df['segment'] == selected_segment].copy()
    else:
        df = historical_df.copy()
    
    if df.empty:
        return pd.DataFrame()
    
    # Ensure date columns are datetime
    df['dateApp'] = pd.to_datetime(df['dateApp'], errors='coerce')
    
    # Filter out invalid data
    df = df.dropna(subset=['dateApp', 'tripCost'])
    df = df[df['tripCost'] > 0]  # Only positive trip costs
    
    if df.empty:
        return pd.DataFrame()
    
    # Group by purchase week (Monday start)
    df['purchase_week'] = df['dateApp'].dt.to_period('W-MON').dt.start_time
    
    # Aggregate by purchase week
    weekly_trip_costs = df.groupby('purchase_week').agg({
        'tripCost': ['mean', 'median', 'std', 'count'],
        'dateApp': 'min'  # Keep the first date of the week for reference
    }).reset_index()
    
    # Flatten column names
    weekly_trip_costs.columns = ['purchase_week', 'avg_trip_cost', 'median_trip_cost', 'std_trip_cost', 'policy_count', 'week_start_date']
    
    # Add ISO week and year for seasonality analysis
    weekly_trip_costs['iso_week'] = weekly_trip_costs['purchase_week'].dt.isocalendar().week
    weekly_trip_costs['iso_year'] = weekly_trip_costs['purchase_week'].dt.isocalendar().year
    
    # Cache the results
    if folder_path and not weekly_trip_costs.empty:
        try:
            cache_file = folder_path / f"trip_cost_analysis_{selected_segment or 'all'}.parquet"
            st.sidebar.info(f"💾 Caching trip cost analysis to: {cache_file}")
            weekly_trip_costs.to_parquet(cache_file, index=False)
            st.sidebar.success(f"✅ Cached trip cost analysis for {selected_segment or 'all'}")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Could not cache trip cost analysis: {e}")
    
    return weekly_trip_costs


def analyze_traveling_patterns_by_cohort(historical_df, selected_segment=None, folder_path=None):
    """
    Analyze traveling patterns by purchase week cohort using historical data
    For each purchase week, find how many policies are traveling in subsequent weeks
    
    Parameters:
    - historical_df: DataFrame with historical purchase and departure data
    - selected_segment: Segment to analyze
    - folder_path: Path for caching
    
    Returns:
    - DataFrame with traveling patterns by cohort
    """
    if historical_df.empty or 'dateDepart' not in historical_df.columns or 'dateReturn' not in historical_df.columns:
        return pd.DataFrame()
    
    # Check for cached traveling patterns
    if folder_path:
        cache_file = folder_path / f"traveling_patterns_{selected_segment or 'all'}.parquet"
        if cache_file.exists():
            try:
                cached_patterns = pd.read_parquet(cache_file)
                st.sidebar.success(f"✅ Loaded cached traveling patterns for {selected_segment or 'all'}")
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
    
    # Filter out invalid data
    df = df.dropna(subset=['dateApp', 'dateDepart', 'dateReturn'])
    df = df[df['dateDepart'] >= df['dateApp']]  # Departure after purchase
    df = df[df['dateReturn'] >= df['dateDepart']]  # Return after departure
    
    if df.empty:
        return pd.DataFrame()
    
    # Group by purchase week (Monday start)
    df['purchase_week'] = df['dateApp'].dt.to_period('W-MON').dt.start_time
    
    traveling_patterns = []
    
    for purchase_week in df['purchase_week'].unique():
        cohort_policies = df[df['purchase_week'] == purchase_week]
        total_purchased = len(cohort_policies)
        
        if total_purchased == 0:
            continue
        
        # Calculate average trip cost for the entire cohort
        cohort_avg_trip_cost = cohort_policies['tripCost'].mean() if 'tripCost' in cohort_policies.columns else 0
        
        # For each week from 0 to 104 (2 years max)
        for weeks_after in range(0, 105):  # 0 to 104 weeks (2 years)
            target_week_start = purchase_week + pd.Timedelta(weeks=weeks_after)
            target_week_end = target_week_start + pd.Timedelta(days=6)
            
            # Count policies traveling during this week
            traveling_policies = cohort_policies[
                (cohort_policies['dateDepart'] <= target_week_end) &
                (cohort_policies['dateReturn'] > target_week_start)
            ]
            
            traveling_count = len(traveling_policies)
            proportion = traveling_count / total_purchased if total_purchased > 0 else 0
            
            # Calculate average trip cost for traveling policies this week
            traveling_avg_trip_cost = traveling_policies['tripCost'].mean() if 'tripCost' in traveling_policies.columns and len(traveling_policies) > 0 else 0
            
            # Calculate trip cost proportion
            trip_cost_proportion = traveling_avg_trip_cost / cohort_avg_trip_cost if cohort_avg_trip_cost > 0 else 0
            
            traveling_patterns.append({
                'purchase_week': purchase_week,
                'weeks_after_purchase': weeks_after,
                'total_purchased': total_purchased,
                'traveling_policies': traveling_count,
                'proportion': proportion,
                'cohort_avg_trip_cost': cohort_avg_trip_cost,
                'traveling_avg_trip_cost': traveling_avg_trip_cost,
                'trip_cost_proportion': trip_cost_proportion
            })
            
            # Stop if proportion is small (5%) or after 2 years
            if proportion < 0.05 or weeks_after >= 104:
                break
    
    patterns_df = pd.DataFrame(traveling_patterns)
    
    # Cache the results
    if folder_path and not patterns_df.empty:
        try:
            cache_file = folder_path / f"traveling_patterns_{selected_segment or 'all'}.parquet"
            st.sidebar.info(f"💾 Caching traveling patterns to: {cache_file}")
            patterns_df.to_parquet(cache_file, index=False)
            st.sidebar.success(f"✅ Cached traveling patterns for {selected_segment or 'all'}")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Could not cache traveling patterns: {e}")
    
    return patterns_df

def create_traveling_forecast_by_cohort(purchase_forecast, traveling_patterns, selected_segment=None):
    """
    Create traveling forecast using cohort-based patterns
    
    Parameters:
    - purchase_forecast: DataFrame with weekly purchase forecasts
    - traveling_patterns: DataFrame with historical traveling patterns by cohort
    - selected_segment: Segment being forecasted
    
    Returns:
    - DataFrame with traveling forecasts by week
    """
    if purchase_forecast.empty or traveling_patterns.empty:
        return pd.DataFrame()
    
    traveling_forecast = []
    
    for _, purchase_row in purchase_forecast.iterrows():
        purchase_week = purchase_row['week_purchased']
        policy_volume = purchase_row['policy_volume']
        
        # Find matching historical pattern (same week from previous year)
        forecast_week = purchase_week.isocalendar().week
        forecast_year = purchase_week.isocalendar().year
        
        matching_patterns = traveling_patterns[
            (traveling_patterns['purchase_week'].dt.isocalendar().week == forecast_week) &
            (traveling_patterns['purchase_week'].dt.isocalendar().year == forecast_year - 1)
        ]
        
        if matching_patterns.empty:
            # Use average pattern if no exact match
            avg_patterns = traveling_patterns.groupby('weeks_after_purchase').agg({
                'proportion': 'mean'
            }).reset_index()
        else:
            # Average across all years for the same week of year
            avg_patterns = matching_patterns.groupby('weeks_after_purchase').agg({
                'proportion': 'mean'
            }).reset_index()
        
        # Apply pattern to forecasted policy volume
        for _, pattern in avg_patterns.iterrows():
            weeks_after = pattern['weeks_after_purchase']
            proportion = pattern['proportion']
            
            if proportion > 0.001:  # Only include meaningful proportions
                target_week = purchase_week + pd.Timedelta(weeks=weeks_after)
                traveling_volume = int(policy_volume * proportion)
                
                if traveling_volume > 0:
                    traveling_forecast.append({
                        'model_point_id': purchase_row['model_point_id'],
                        'purchase_week': purchase_week,
                        'target_week': target_week,
                        'weeks_after_purchase': weeks_after,
                        'traveling_policies': traveling_volume,
                        'proportion': proportion,
                        'segment': selected_segment or 'all',
                        'forecast_type': 'traveling_cohort_forecast'
                    })
    
    return pd.DataFrame(traveling_forecast)


def simple_forecast(historical_data, weeks_ahead=26, folder_path=None, selected_segment=None):
    """Generate forecast using external CSV data"""
    if folder_path:
        external_forecast = load_external_forecast(folder_path)
        if not external_forecast.empty:
            return convert_monthly_to_weekly_forecast(external_forecast, historical_data, weeks_ahead, selected_segment)
    
    return pd.DataFrame()

# Main app
st.title("📊 Forecast Configuration")

# Sidebar controls
st.sidebar.header("Configuration")

# Data folder selection
data_folders = []
if Path("_data").exists():
    data_folders = [f.name for f in Path("_data").iterdir() if f.is_dir()]

if data_folders:
    selected_folder = st.sidebar.selectbox("Select data folder", data_folders)
    folder_path = Path("_data") / selected_folder
else:
    st.sidebar.warning("No data folders found in _data directory")
    folder_path = None

if folder_path and folder_path.exists():
    # Load historical data
    with st.spinner("Loading historical data..."):
        historical_df = load_historical_purchases(folder_path)
    
    if not historical_df.empty:
        # Segment selection
        if 'segment' in historical_df.columns:
            unique_segments = historical_df['segment'].unique()
            selected_segment = st.sidebar.selectbox("Select segment", unique_segments)
        else:
            selected_segment = None
            st.sidebar.warning("No segment column found in data")
        
        # Country and region filters
        if 'country' in historical_df.columns:
            unique_countries = sorted(historical_df['country'].unique())
            selected_countries = st.sidebar.multiselect("Filter by countries", unique_countries, default=unique_countries)
        else:
            selected_countries = []
        
        if 'region' in historical_df.columns:
            unique_regions = sorted(historical_df['region'].unique())
            selected_regions = st.sidebar.multiselect("Filter by regions", unique_regions, default=unique_regions)
        else:
            selected_regions = []
        
        # Forecast parameters
        st.sidebar.subheader("Forecast Parameters")
        weeks_ahead = 104  # Fixed at 104 weeks (2 years)
        
        # Generate forecast
        if st.sidebar.button("Generate Forecast", type="primary"):
            # Aggregate historical data
            weekly_purchases = aggregate_weekly_purchases(historical_df, selected_countries, selected_regions)
            
            # Generate purchase forecast
            with st.spinner("Generating purchase forecast..."):
                forecast_df = simple_forecast(weekly_purchases, weeks_ahead, folder_path, selected_segment)
            
            if forecast_df.empty:
                st.error("❌ Could not generate forecast")
                st.stop()
            
            # Analyze historical trip costs by week
            with st.spinner("Analyzing historical trip costs by week..."):
                trip_cost_analysis = analyze_historical_trip_costs_by_week(historical_df, selected_segment, folder_path)
            
            # Analyze traveling patterns by cohort (new approach)
            with st.spinner("Analyzing traveling patterns by cohort..."):
                traveling_patterns = analyze_traveling_patterns_by_cohort(historical_df, selected_segment, folder_path)
            
            # Create traveling forecast using cohort patterns
            traveling_forecast_df = pd.DataFrame()
            if not traveling_patterns.empty:
                traveling_forecast_df = create_traveling_forecast_by_cohort(forecast_df, traveling_patterns, selected_segment)
            
            # Aggregate traveling policies by week
            traveling_by_week_df = pd.DataFrame()
            if not traveling_forecast_df.empty:
                # Group by target_week to get weekly traveling totals
                traveling_by_week_df = traveling_forecast_df.groupby('target_week').agg({
                    'traveling_policies': 'sum'
                }).reset_index()
                traveling_by_week_df = traveling_by_week_df.rename(columns={'target_week': 'week'})
                
                # Add additional metrics
                traveling_by_week_df['total_trip_cost'] = 0  # Placeholder - would need trip cost data
                traveling_by_week_df['avg_trip_length'] = 0  # Placeholder
                traveling_by_week_df['avg_cost_per_night'] = 0  # Placeholder
            
            st.success(f"✅ Generated forecast for {len(forecast_df)} weeks")
            
            # Display forecast results
            st.markdown("---")
            
            # Historical Trip Cost Analysis - First Plot
            if not trip_cost_analysis.empty:
                st.subheader("💰 Historical Trip Cost Analysis")
                
                # Create the plot
                fig = px.line(
                    trip_cost_analysis,
                    x='iso_week',
                    y='avg_trip_cost',
                    color='iso_year',
                    color_discrete_sequence=px.colors.qualitative.Safe,
                    labels={'iso_week': 'ISO Week', 'avg_trip_cost': 'Average Trip Cost', 'iso_year': 'ISO Year'},
                )
                
                # Configure x-axis ticks for ISO weeks
                tickvals = list(range(1, 53, 4))  # W1, W5, W9, W13, etc.
                ticktext = [f"W{w}" for w in tickvals]
                fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)
                
                fig.update_layout(
                    title="Historical Average Trip Cost by Purchase Week",
                    legend_title_text="ISO Year",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Data table
                st.dataframe(trip_cost_analysis, use_container_width=True)
            else:
                st.warning("No trip cost data available for analysis")
            
            # Purchase forecast section
            st.subheader("📈 Purchase Forecast")
            st.dataframe(forecast_df, use_container_width=True)
            
            # Show cohort patterns analysis
            if not traveling_patterns.empty:
                st.subheader("📊 Cohort Patterns from historical data used for forecasting")
                st.dataframe(traveling_patterns, use_container_width=True)
            
            # Show detailed model points with matching patterns
            if not traveling_forecast_df.empty:
                st.subheader("📋 Model Points with Matching Patterns")
                
                # Display the traveling forecast with model point details
                st.dataframe(traveling_forecast_df, use_container_width=True)
                
                
            # Traveling policies analysis (cohort-based)
            if not traveling_by_week_df.empty:
                st.subheader("✈️ Traveling Policies by Week")
                
                # Create normalized week visualization similar to historical page
                plot_data = traveling_by_week_df.copy()
                
                # Sort by week to ensure proper ordering
                plot_data = plot_data.sort_values('week')
                
                # Use ISO week numbers for x-axis
                plot_data['x'] = plot_data['week'].dt.isocalendar().week
                
                # Add ISO year for coloring
                plot_data['iso_year'] = plot_data['week'].dt.isocalendar().year
                
                # Create the plot
                fig = px.line(
                    plot_data,
                    x="x",
                    y="traveling_policies",
                    color="iso_year",
                    color_discrete_sequence=px.colors.qualitative.Safe,
                    labels={"x": "ISO Week", "traveling_policies": "Traveling Policies", "iso_year": "ISO Year"},
                )
                
                # Configure x-axis ticks for ISO weeks
                tickvals = list(range(1, 53, 4))  # W1, W5, W9, W13, etc.
                ticktext = [f"W{w}" for w in tickvals]
                fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)
                
                fig.update_layout(
                    title="Traveling Policies by Week (Normalized)",
                    legend_title_text="ISO Year",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Data table
                st.dataframe(traveling_by_week_df, use_container_width=True)
            
            
            
            # Download forecast data
            st.markdown("---")
            st.subheader("📥 Download Forecast Data")
            
            # Download buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if not forecast_df.empty:
                    csv_purchase = forecast_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Purchase Forecast",
                        data=csv_purchase,
                        file_name=f"purchase_forecast_{selected_segment or 'all'}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if not traveling_by_week_df.empty:
                    csv_traveling = traveling_by_week_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Traveling Policies",
                        data=csv_traveling,
                        file_name=f"traveling_policies_{selected_segment or 'all'}.csv",
                        mime="text/csv"
                    )
            
            with col3:
                if not traveling_forecast_df.empty:
                    csv_cohort = traveling_forecast_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Cohort Forecast",
                        data=csv_cohort,
                        file_name=f"cohort_forecast_{selected_segment or 'all'}.csv",
                        mime="text/csv"
                    )
    else:
        st.error("❌ No historical data found")
else:
    st.info("Please select a data folder in the sidebar to begin forecasting.")