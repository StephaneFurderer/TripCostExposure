#!/usr/bin/env python3
"""
Debug script to validate weekly aggregation logic step by step.
Loads combined.parquet and recreates aggregate_traveling_unique_by_period 
to validate correct amounts per week.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from exposure_data import classify_country, classify_region


def load_combined_data(folder_path: str) -> pd.DataFrame:
    """Load combined.parquet from the specified folder."""
    combined_path = Path(folder_path) / "combined.parquet"
    if not combined_path.exists():
        raise FileNotFoundError(f"combined.parquet not found in {folder_path}")
    
    df = pd.read_parquet(combined_path)
    print(f"Loaded {len(df):,} policies from {combined_path}")
    print(f"Date range: {df['dateDepart'].min().date()} to {df['dateReturn'].max().date()}")
    print(f"Columns: {list(df.columns)}")
    return df


def debug_aggregate_traveling_unique_by_period(
    df: pd.DataFrame,
    period: str = "week",
    start_date: pd.Timestamp = None,
    end_date: pd.Timestamp = None,
) -> pd.DataFrame:
    """
    Debug version of aggregate_traveling_unique_by_period.
    Aggregates for ALL segments, countries, and zip codes to validate weekly totals.
    """
    print(f"\n=== DEBUG: Weekly Aggregation ===")
    print(f"Period: {period}")
    print(f"Input data: {len(df):,} policies")
    
    # Ensure date columns are datetime
    df = df.copy()
    df["dateDepart"] = pd.to_datetime(df["dateDepart"]).dt.normalize()
    df["dateReturn"] = pd.to_datetime(df["dateReturn"]).dt.normalize()
    
    # Add country and region classification
    if "Country" in df.columns:
        df["country_class"] = df["Country"].apply(classify_country)
    else:
        df["country_class"] = "null"
    
    if "ZipCode" in df.columns:
        df["region_class"] = df["ZipCode"].apply(classify_region)
    else:
        df["region_class"] = "Other"
    
    # Calculate per-night cost
    df["perNight"] = (df["tripCost"] / df["nightsCount"].replace(0, pd.NA)).fillna(0.0)
    
    # Get date range
    all_dates = pd.concat([df["dateDepart"], df["dateReturn"]]).dropna()
    min_date = all_dates.min()
    max_date = all_dates.max()
    
    if start_date is None:
        start_date = min_date
    if end_date is None:
        end_date = max_date
    
    print(f"Processing from {start_date.date()} to {end_date.date()}")
    
    records = []
    
    if period == "week":
        # Process all weeks from min_date to max_date
        current_week = min_date - pd.to_timedelta(min_date.weekday(), unit="D")  # Monday of first week
        week_count = 0
        total_weeks = int((max_date - current_week).days / 7) + 1
        
        print(f"Processing {total_weeks} weeks from {current_week.date()}")
        
        while current_week <= max_date:
            week_count += 1
            week_end = current_week + pd.to_timedelta(6, unit="D")
            
            # Find policies traveling during this week (same logic as week search)
            traveling_mask = (df["dateDepart"] <= week_end) & (df["dateReturn"] > current_week)
            traveling_policies = df[traveling_mask]
            
            if len(traveling_policies) > 0:
                # Calculate normalized x-axis value (using ISO week number)
                x_norm = current_week.isocalendar()[1]  # ISO week number
                year = current_week.year
                
                # Calculate nights in week (same logic as week search)
                night_range_start = traveling_policies["dateDepart"]
                night_range_end = traveling_policies["dateReturn"] - pd.to_timedelta(1, unit="D")
                overlap_start = night_range_start.where(night_range_start > current_week, current_week)
                overlap_end = night_range_end.where(night_range_end < week_end, week_end)
                delta = (overlap_end - overlap_start).dt.days + 1
                traveling_policies["nightsInWeek"] = delta.clip(lower=0).fillna(0).astype(int)
                traveling_policies["remainingTripCost"] = (traveling_policies["nightsInWeek"] * traveling_policies["perNight"]).round(2)
                
                # Aggregate ALL data (no grouping by segment/country/region)
                volume = len(traveling_policies)
                maxTripCostExposure = traveling_policies["tripCost"].sum()
                tripCostPerNightExposure = traveling_policies["remainingTripCost"].sum()
                avgTripCostPerNight = traveling_policies["perNight"].mean()
                
                record = {
                    'year': year,
                    'x': x_norm,  # ISO week number
                    'week_start': current_week,
                    'week_end': week_end,
                    'volume': volume,
                    'maxTripCostExposure': maxTripCostExposure,
                    'tripCostPerNightExposure': tripCostPerNightExposure,
                    'avgTripCostPerNight': avgTripCostPerNight,
                    'country_class': 'ALL',
                    'region_class': 'ALL'
                }
                records.append(record)
                
                # Debug output for first few weeks
                if week_count <= 5 or week_count % 10 == 0:
                    print(f"  Week {week_count:2d}: {current_week.date()} to {week_end.date()} | "
                          f"Policies: {volume:4d} | TripCost: ${maxTripCostExposure:10,.0f} | "
                          f"PerNight: ${tripCostPerNightExposure:10,.0f}")
            
            # Move to next week
            current_week += pd.to_timedelta(7, unit="D")
    
    result_df = pd.DataFrame(records)
    print(f"\n=== RESULTS ===")
    print(f"Generated {len(result_df)} weekly records")
    if not result_df.empty:
        print(f"Date range: {result_df['week_start'].min().date()} to {result_df['week_end'].max().date()}")
        print(f"Total policies across all weeks: {result_df['volume'].sum():,}")
        print(f"Total trip cost exposure: ${result_df['maxTripCostExposure'].sum():,.0f}")
        print(f"Total per-night exposure: ${result_df['tripCostPerNightExposure'].sum():,.0f}")
        
        # Show first few weeks
        print(f"\nFirst 10 weeks:")
        display_cols = ['year', 'x', 'week_start', 'week_end', 'volume', 'maxTripCostExposure', 'tripCostPerNightExposure']
        print(result_df[display_cols].head(10).to_string(index=False))
    
    return result_df


def main():
    """Main debug function."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python debug_weekly_aggregation.py <folder_path>")
        print("Example: python debug_weekly_aggregation.py _data")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    
    try:
        # Step 1: Load combined.parquet
        print("=== STEP 1: Loading combined.parquet ===")
        df = load_combined_data(folder_path)
        
        # Step 2: Debug weekly aggregation
        print("\n=== STEP 2: Weekly Aggregation Debug ===")
        result = debug_aggregate_traveling_unique_by_period(df, period="week")
        
        # Step 3: Show summary by year
        if not result.empty:
            print(f"\n=== STEP 3: Summary by Year ===")
            yearly_summary = result.groupby('year').agg({
                'volume': 'sum',
                'maxTripCostExposure': 'sum',
                'tripCostPerNightExposure': 'sum',
                'avgTripCostPerNight': 'mean'
            }).round(2)
            print(yearly_summary)
            
            # Step 4: Show specific weeks (e.g., W1 2024)
            print(f"\n=== STEP 4: Week 1, 2024 Details ===")
            w1_2024 = result[(result['year'] == 2024) & (result['x'] == 1)]
            if not w1_2024.empty:
                print("W1 2024 found:")
                print(w1_2024[['week_start', 'week_end', 'volume', 'maxTripCostExposure', 'tripCostPerNightExposure']].to_string(index=False))
            else:
                print("W1 2024 not found in results")
                print("Available weeks in 2024:")
                weeks_2024 = result[result['year'] == 2024]['x'].unique()
                print(f"Week numbers: {sorted(weeks_2024)}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
