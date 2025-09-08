from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, List, Union

import pandas as pd
import glob
from pathlib import Path
import time


DEFAULT_CSV_PATH = os.path.join(os.path.dirname(__file__), "_data", "policies.csv")


@dataclass
class ExposureDataConfig:
    csv_path: str = DEFAULT_CSV_PATH


def normalize_policies_df(df: pd.DataFrame) -> pd.DataFrame:
    MISSING_TOKENS = {None, "", " ", "NA", "N/A", "<N/A>", "NULL", "null", "NaN", "nan"}

    def _string_clean(col: pd.Series) -> pd.Series:
        s = col.astype("string")
        s = s.replace(list(MISSING_TOKENS), pd.NA)
        return s

    """
    Apply the same sanity checks/normalization for both dummy and real data.
    - Types for key fields
    - Compute nightsCount if missing; filter invalid date ranges
    - Default travelersCount to 1 if missing/null
    - Coerce tripCost numeric
    """
    # Ensure date parsing
    for col in ["dateApp", "dateDepart", "dateReturn"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Coerce cost numeric
    if "tripCost" in df.columns:
        df["tripCost"] = pd.to_numeric(df["tripCost"], errors="coerce").fillna(0.0)
    else:
        df["tripCost"] = 0.0

    # Travelers: if present, coerce numeric but keep missing (no default)
    if "travelersCount" in df.columns:
        df["travelersCount"] = pd.to_numeric(df["travelersCount"], errors="coerce")

    # Nights count
    if "nightsCount" not in df.columns:
        df = df[df["dateReturn"].notna() & df["dateDepart"].notna()]
        df = df[df["dateReturn"] >= df["dateDepart"]]
        df["nightsCount"] = (df["dateReturn"] - df["dateDepart"]).dt.days
    else:
        df["nightsCount"] = pd.to_numeric(df["nightsCount"], errors="coerce")
    # Drop zero or negative nights (same-day or invalid)
    df = df[df["nightsCount"] > 0]
    df["nightsCount"] = df["nightsCount"].astype(int)

    # Enforce positive tripCost per requirements
    df = df[df["tripCost"] > 0]

    # Handle ZIP code variants and preserve leading zeros
    if "ZipCode" not in df.columns:
        # search common variants case-insensitively
        lower_cols = {c.lower(): c for c in df.columns}
        candidate = None
        for key in ["zipcode", "zip", "zip_code", "postalcode", "postal_code"]:
            if key in lower_cols:
                candidate = lower_cols[key]
                break
        if candidate is not None:
            df["ZipCode"] = df[candidate]
    if "ZipCode" in df.columns:
        # Preserve non-US alphanumeric; standardize only if numeric up to 5 digits
        zorig = _string_clean(df["ZipCode"])  # normalize missing tokens first
        is_numeric_zip = zorig.str.fullmatch(r"\d{1,5}")
        standardized = zorig.where(~is_numeric_zip, zorig.str.zfill(5))
        df["ZipCode"] = standardized.astype("string")

    # Optional categoricals/strings
    # Keep segment as plain string (per requirements)
    if "segment" in df.columns:
        df["segment"] = _string_clean(df["segment"]).astype("string")
    if "Country" in df.columns:
        df["Country"] = _string_clean(df["Country"]).astype("string")

    # Ensure idpol as string if present
    if "idpol" in df.columns:
        df["idpol"] = _string_clean(df["idpol"]).astype("string")

    # Optional state normalization: keep as string and allow missing
    lower_cols = {c.lower(): c for c in df.columns}
    if "state" in lower_cols:
        col = lower_cols["state"]
        df[col] = _string_clean(df[col]).astype("string")

    return df


def load_policies(csv_path: Optional[str] = None) -> pd.DataFrame:
    path = csv_path or DEFAULT_CSV_PATH
    df = pd.read_csv(
        path,
        dtype={
            "segment": "category",
            "ZipCode": "string",
            "Country": "category",
        },
        parse_dates=["dateApp", "dateDepart", "dateReturn"],
        infer_datetime_format=True,
    )
    return normalize_policies_df(df)


def _read_policy_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={
            "segment": "category",
            "ZipCode": "string",
            "Country": "category",
        },
        parse_dates=["dateApp", "dateDepart", "dateReturn"],
        infer_datetime_format=True,
    )
    # Normalize ZIP column name at import: rename any alias to 'ZipCode'
    lower_cols = {c.lower(): c for c in df.columns}
    for alias in ["zipcode", "zip", "zip_code", "postalcode", "postal_code"]:
        if alias in lower_cols and lower_cols[alias] != "ZipCode":
            df = df.rename(columns={lower_cols[alias]: "ZipCode"})
            break
    # If segment missing, derive from file name
    if "segment" not in df.columns:
        seg = Path(path).stem
        df["segment"] = seg
    return normalize_policies_df(df)


def load_folder_policies(folder_path: str, force_rebuild: bool = False, erase_cache: bool = False) -> pd.DataFrame:
    """
    Load and append all CSVs under a folder like _data/2025-08-01/ and cache to parquet for speed.
    Cache file location: <folder_path>/combined.parquet
    - erase_cache=True deletes the parquet before proceeding
    - force_rebuild=True rebuilds parquet from CSVs even if cache exists
    """
    folder = Path(folder_path)
    folder = folder.expanduser().resolve()
    cache_path = folder / "combined.parquet"

    if erase_cache and cache_path.exists():
        cache_path.unlink(missing_ok=True)

    if cache_path.exists() and not force_rebuild:
        return pd.read_parquet(cache_path)

    csv_files = sorted(glob.glob(str(folder / "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in folder: {folder}")

    frames: List[pd.DataFrame] = []
    for csv_file in csv_files:
        try:
            frames.append(_read_policy_csv(csv_file))
        except Exception as exc:
            # Skip unreadable files but continue
            continue

    if not frames:
        raise ValueError(f"No readable CSV files in folder: {folder}")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    # Ensure Arrow-friendly dtypes before parquet write
    for col in ["segment", "ZipCode", "Country", "idpol"]:
        if col in combined.columns:
            combined[col] = combined[col].astype("string")
    # Datetime columns
    for col in ["dateApp", "dateDepart", "dateReturn"]:
        if col in combined.columns:
            combined[col] = pd.to_datetime(combined[col], errors="coerce")
    # Integers as regular ints (nightsCount) and floats (tripCost)
    if "nightsCount" in combined.columns:
        combined["nightsCount"] = pd.to_numeric(combined["nightsCount"], errors="coerce").astype("Int64")
    if "travelersCount" in combined.columns:
        combined["travelersCount"] = pd.to_numeric(combined["travelersCount"], errors="coerce").astype("Int64")
    if "tripCost" in combined.columns:
        combined["tripCost"] = pd.to_numeric(combined["tripCost"], errors="coerce").astype(float)
    # Persist parquet cache
    combined.to_parquet(cache_path, index=False)
    # Build precomputed aggregates for fast UI
    try:
        precompute_aggregates(folder)
    except Exception:
        # Do not fail the load if precompute has issues
        pass
    return combined


def aggregate_daily_exposure_by_departure_year(
    df: pd.DataFrame, additional_group_by: Optional[Union[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Returns a tidy DataFrame with columns: dateDepart, md_date, year, exposure,
    and optionally the additional group-by columns.
    """
    if additional_group_by is None:
        extra_cols: List[str] = []
    elif isinstance(additional_group_by, str):
        extra_cols = [additional_group_by]
    else:
        extra_cols = list(additional_group_by)

    select_cols: List[str] = ["dateDepart", "tripCost"] + extra_cols
    temp = df[select_cols].copy()
    temp["dateDepart"] = temp["dateDepart"].dt.date

    group_cols: List[str] = ["dateDepart"] + extra_cols
    grouped = (
        temp.groupby(group_cols, as_index=False)["tripCost"].sum().rename(columns={"tripCost": "exposure"})
    )

    dt = pd.to_datetime(grouped["dateDepart"])  # Timestamp series for convenience
    grouped["year"] = dt.dt.year.astype("int64")
    grouped["md_date"] = dt.apply(lambda x: pd.Timestamp(year=2000, month=x.month, day=x.day))

    sort_cols: List[str] = ["md_date", "year"] + [c for c in extra_cols]
    grouped = grouped.sort_values(sort_cols)

    base_cols = ["dateDepart", "md_date", "year", "exposure"]
    return grouped[base_cols + extra_cols]


def aggregate_exposure_by_departure_period(
    df: pd.DataFrame, period: str = "day", additional_group_by: Optional[Union[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Aggregate exposure by departure period with overlay by year.

    period: one of {"day", "week", "month"}
    additional_group_by: optional extra grouping column(s) (str or list)

    Returns columns: ["year", "x", "exposure", ...extras]
      - For day:   x = md_date (Timestamp in year 2000 for same month-day)
      - For week:  x = week_start_2000 (Timestamp Monday of ISO week number in a common year)
      - For month: x = month_start_2000 (Timestamp first day of month in a common year)
    """
    period = period.lower()
    if period not in {"day", "week", "month"}:
        raise ValueError("period must be one of 'day', 'week', 'month'")

    if additional_group_by is None:
        extra_cols: List[str] = []
    elif isinstance(additional_group_by, str):
        extra_cols = [additional_group_by]
    else:
        extra_cols = list(additional_group_by)

    select_cols: List[str] = ["dateDepart", "tripCost"] + extra_cols
    tmp = df[select_cols].copy()
    dt = pd.to_datetime(tmp["dateDepart"])
    tmp["year"] = dt.dt.year.astype("int64")

    if period == "day":
        # Month-day normalized to year 2000
        tmp["x"] = dt.apply(lambda x: pd.Timestamp(year=2000, month=x.month, day=x.day))
        group_cols = ["year", "x"] + extra_cols
    elif period == "week":
        # ISO week number
        week = dt.dt.isocalendar().week.astype(int)
        # Build a pseudo date in a common year for plotting: Monday of that ISO week in 2000
        # 2000-01-03 is Monday (ISO week 2000-01)
        week_start_2000 = pd.Timestamp(2000, 1, 3) + pd.to_timedelta((week - 1) * 7, unit="D")
        tmp["x"] = week_start_2000
        group_cols = ["year", "x"] + extra_cols
    else:  # month
        month = dt.dt.month
        tmp["x"] = pd.to_datetime({"year": 2000, "month": month, "day": 1})
        group_cols = ["year", "x"] + extra_cols

    grouped = (
        tmp.groupby(group_cols, as_index=False)["tripCost"].sum().rename(columns={"tripCost": "exposure"})
    )
    sort_cols = ["x", "year"] + [c for c in extra_cols]
    grouped = grouped.sort_values(sort_cols)
    return grouped[["year", "x", "exposure"] + extra_cols]


def _normalize_period_x(dt: pd.Series, period: str) -> pd.Series:
    if period == "day":
        return dt.apply(lambda x: pd.Timestamp(year=2000, month=x.month, day=x.day))
    if period == "week":
        week = dt.dt.isocalendar().week.astype(int)
        return pd.Timestamp(2000, 1, 3) + pd.to_timedelta((week - 1) * 7, unit="D")
    if period == "month":
        month = dt.dt.month
        return pd.to_datetime({"year": 2000, "month": month, "day": 1})
    raise ValueError("period must be one of 'day', 'week', 'month'")


def compute_traveling_daily(
    df: pd.DataFrame,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    additional_group_by: Optional[Union[str, List[str]]] = None,
) -> pd.DataFrame:
    """
    Compute per-day traveling metrics WITHOUT exploding per-day rows using a sweep-line approach:
      - volume: number of active policies per day
      - maxTripCostExposure: sum of full tripCost across active policies
      - tripCostPerNightExposure: sum of perNight across active policies

    Returns columns: ["day", "year", metrics..., ...extras]
    """
    if additional_group_by is None:
        extra_cols: List[str] = []
    elif isinstance(additional_group_by, str):
        extra_cols = [additional_group_by]
    else:
        extra_cols = list(additional_group_by)

    tmp = df[["dateDepart", "dateReturn", "tripCost", "nightsCount"] + extra_cols].copy()
    tmp["dateDepart"] = pd.to_datetime(tmp["dateDepart"]).dt.date
    tmp["dateReturn"] = pd.to_datetime(tmp["dateReturn"]).dt.date

    # Window bounds
    overall_start = tmp["dateDepart"].min()
    overall_end = tmp["dateReturn"].max()
    requested_start = pd.to_datetime(start_date).date() if start_date is not None else overall_start
    requested_end = pd.to_datetime(end_date).date() if end_date is not None else overall_end

    # Clip to requested window
    start = tmp["dateDepart"].where(tmp["dateDepart"] >= requested_start, requested_start)
    end = tmp["dateReturn"].where(tmp["dateReturn"] <= requested_end, requested_end)

    # Remove invalid ranges after clipping
    valid_mask = start <= end
    tmp = tmp.loc[valid_mask].copy()
    start = start.loc[valid_mask]
    end = end.loc[valid_mask]

    # Per-night
    per_night = (tmp["tripCost"] / tmp["nightsCount"].replace(0, pd.NA)).fillna(0.0)

    # Build event deltas: + at start, - at end + 1 day
    # Create DataFrame of events with minimal rows (2 per policy)
    start_ts = pd.to_datetime(start)
    end_plus1_ts = pd.to_datetime(end) + pd.to_timedelta(1, unit="D")

    def build_events(sign: int, days: pd.Series) -> pd.DataFrame:
        data = {
            "day": days.values,
            "volume": sign * 1,
            "maxTripCostExposure": sign * tmp["tripCost"].values,
            "tripCostPerNightExposure": sign * per_night.values,
        }
        events = pd.DataFrame(data)
        for c in extra_cols:
            events[c] = tmp[c].values
        return events

    ev_start = build_events(+1, start_ts)
    ev_end = build_events(-1, end_plus1_ts)
    events = pd.concat([ev_start, ev_end], ignore_index=True)

    # Aggregate deltas per day (and segment if needed)
    group_cols = ["day"] + extra_cols
    deltas = events.groupby(group_cols, as_index=False).sum()
    deltas = deltas.sort_values(group_cols)

    # Cumulative sum over days per group
    # We need a continuous date index over [requested_start, requested_end]
    full_days = pd.date_range(requested_start, requested_end, freq="D")

    if extra_cols:
        # Build grid per group for reindexing
        grids = []
        for keys, grp in deltas.groupby(extra_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            idx = pd.Index(full_days, name="day")
            # Only reindex numeric metric columns to avoid string fill errors
            metric_cols = ["volume", "maxTripCostExposure", "tripCostPerNightExposure"]
            grp_metrics = grp[["day"] + metric_cols]
            g = grp_metrics.set_index("day").reindex(idx, fill_value=0)
            # Add back extra cols as columns
            for i, c in enumerate(extra_cols):
                g[c] = keys[i]
            g = g.reset_index().rename(columns={"index": "day"})
            grids.append(g)
        deltas_full = pd.concat(grids, ignore_index=True)
        sort_cols = extra_cols + ["day"]
        deltas_full = deltas_full.sort_values(sort_cols)
        # Perform cumulative sums via group-wise transform
        deltas_full = deltas_full.sort_values(sort_cols)
        for col in ["volume", "maxTripCostExposure", "tripCostPerNightExposure"]:
            deltas_full[col] = deltas_full.groupby(extra_cols, dropna=False)[col].cumsum()
        daily = deltas_full
    else:
        idx = pd.Index(full_days, name="day")
        deltas_full = deltas.set_index("day").reindex(idx, fill_value=0).reset_index()
        for col in ["volume", "maxTripCostExposure", "tripCostPerNightExposure"]:
            deltas_full[col] = deltas_full[col].cumsum()
        daily = deltas_full

    # Finalize
    daily["day"] = pd.to_datetime(daily["day"]).dt.date
    dt = pd.to_datetime(daily["day"])  # Timestamp
    daily["year"] = dt.dt.year.astype("int64")
    return daily[["day", "year", "volume", "maxTripCostExposure", "tripCostPerNightExposure"] + extra_cols]


def aggregate_traveling_by_period(
    df: pd.DataFrame, period: str = "day", additional_group_by: Optional[Union[str, List[str]]] = None,
    start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Aggregate traveling exposure metrics by period with overlay by year.
    Returns columns: ["year", "x", metrics..., ...extras]
    """
    daily = compute_traveling_daily(
        df, start_date=start_date, end_date=end_date, additional_group_by=additional_group_by
    )
    period = period.lower()
    if period not in {"day", "week", "month"}:
        raise ValueError("period must be one of 'day', 'week', 'month'")

    extra_cols: List[str]
    if additional_group_by is None:
        extra_cols = []
    elif isinstance(additional_group_by, str):
        extra_cols = [additional_group_by]
    else:
        extra_cols = list(additional_group_by)

    dt = pd.to_datetime(daily["day"])
    daily["x"] = _normalize_period_x(dt, period)

    group_cols = ["year", "x"] + extra_cols
    agg = daily.groupby(group_cols, as_index=False).agg(
        volume=("volume", "sum"),
        maxTripCostExposure=("maxTripCostExposure", "sum"),
        tripCostPerNightExposure=("tripCostPerNightExposure", "sum"),
    )
    sort_cols = ["x", "year"] + extra_cols
    agg = agg.sort_values(sort_cols)
    return agg[["year", "x", "volume", "maxTripCostExposure", "tripCostPerNightExposure"] + extra_cols]


def compute_departures_daily(
    df: pd.DataFrame, additional_group_by: Optional[Union[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Compute departures-based metrics by exact departure day:
      - volume: policies departing that day
      - maxTripCostExposure: sum of tripCost of those departing
      - avgTripCostPerNight: mean of per-night cost among those departing
    Returns: ["day", "year", metrics..., ...extras]
    """
    if additional_group_by is None:
        extra_cols: List[str] = []
    elif isinstance(additional_group_by, str):
        extra_cols = [additional_group_by]
    else:
        extra_cols = list(additional_group_by)

    tmp = df[["dateDepart", "tripCost", "nightsCount"] + extra_cols].copy()
    tmp["day"] = pd.to_datetime(tmp["dateDepart"]).dt.date
    tmp["perNight"] = tmp["tripCost"] / tmp["nightsCount"].replace(0, pd.NA)
    tmp["perNight"] = tmp["perNight"].fillna(0.0)

    group_cols = ["day"] + extra_cols
    daily = tmp.groupby(group_cols, as_index=False).agg(
        volume=("tripCost", "count"),
        maxTripCostExposure=("tripCost", "sum"),
        avgTripCostPerNight=("perNight", "mean"),
    )
    dt = pd.to_datetime(daily["day"])  # Timestamp
    daily["year"] = dt.dt.year.astype("int64")
    return daily[["day", "year", "volume", "maxTripCostExposure", "avgTripCostPerNight"] + extra_cols]


def aggregate_departures_by_period(
    df: pd.DataFrame, period: str = "day", additional_group_by: Optional[Union[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Aggregate departures-based metrics by period with overlay by year.
    Returns: ["year", "x", metrics..., ...extras]
    """
    daily = compute_departures_daily(df, additional_group_by=additional_group_by)
    period = period.lower()
    if period not in {"day", "week", "month"}:
        raise ValueError("period must be one of 'day', 'week', 'month'")

    if additional_group_by is None:
        extra_cols: List[str] = []
    elif isinstance(additional_group_by, str):
        extra_cols = [additional_group_by]
    else:
        extra_cols = list(additional_group_by)

    dt = pd.to_datetime(daily["day"])
    daily["x"] = _normalize_period_x(dt, period)

    group_cols = ["year", "x"] + extra_cols
    agg = daily.groupby(group_cols, as_index=False).agg(
        volume=("volume", "sum"),
        maxTripCostExposure=("maxTripCostExposure", "sum"),
        avgTripCostPerNight=("avgTripCostPerNight", "mean"),
    )
    sort_cols = ["x", "year"] + extra_cols
    agg = agg.sort_values(sort_cols)
    return agg[["year", "x", "volume", "maxTripCostExposure", "avgTripCostPerNight"] + extra_cols]


# ---------- Unique-per-period traveling aggregation ----------

def _iter_nights_chunks_by_week(start_date: pd.Timestamp, nights: int):
    current = start_date
    remaining = int(nights)
    while remaining > 0:
        # ISO week starts on Monday (weekday=0)
        week_start = current - pd.to_timedelta(current.weekday(), unit="D")
        week_end = week_start + pd.to_timedelta(6, unit="D")
        take = int(min(remaining, (week_end - current).days + 1))
        yield week_start.normalize(), take
        current = current + pd.to_timedelta(take, unit="D"); remaining -= take


def _iter_nights_chunks_by_month(start_date: pd.Timestamp, nights: int):
    current = start_date
    remaining = int(nights)
    while remaining > 0:
        month_start = current.replace(day=1)
        # next month start
        if month_start.month == 12:
            next_month = month_start.replace(year=month_start.year + 1, month=1, day=1)
        else:
            next_month = month_start.replace(month=month_start.month + 1, day=1)
        month_end = next_month - pd.to_timedelta(1, unit="D")
        take = int(min(remaining, (month_end - current).days + 1))
        yield month_start.normalize(), take
        current = current + pd.to_timedelta(take, unit="D"); remaining -= take


def aggregate_traveling_unique_by_period(
    df: pd.DataFrame,
    period: str,
    additional_group_by: Optional[Union[str, List[str]]] = None,
    max_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Unique-per-period traveling metrics (no daily summation):
      - volume_unique: count policy once per period if active any day in that period
      - maxTripCostExposure_unique: add tripCost once per period if active
      - tripCostPerNightExposure_period: sum perNight for nights whose start falls in the period

    period: 'week' or 'month'
    max_date: Optional cutoff date (defaults to 3 weeks after today if None)
    """
    period = period.lower()
    if period not in {"week", "month"}:
        raise ValueError("unique traveling aggregation supported only for 'week' or 'month'")

    if additional_group_by is None:
        extra_cols: List[str] = []
    elif isinstance(additional_group_by, str):
        extra_cols = [additional_group_by]
    else:
        extra_cols = list(additional_group_by)

    tmp = df[["dateDepart", "dateReturn", "tripCost", "nightsCount"] + extra_cols].copy()
    tmp["dateDepart"] = pd.to_datetime(tmp["dateDepart"]).dt.normalize()
    tmp["dateReturn"] = pd.to_datetime(tmp["dateReturn"]).dt.normalize()

    per_night = (tmp["tripCost"] / tmp["nightsCount"].replace(0, pd.NA)).fillna(0.0)

    # Most efficient approach: iterate through periods and find traveling policies
    records = []
    
    # Set default max_date to 3 weeks after today if not provided
    if max_date is None:
        max_date = pd.Timestamp.now() + pd.to_timedelta(21, unit="D")
    
    # Get date range from all policies, but limit to max_date
    all_dates = pd.concat([tmp["dateDepart"], tmp["dateReturn"]]).dropna()
    min_date = all_dates.min()
    # Don't limit max_date here - we want to use the user-provided max_date as the cutoff
    
    print(f"Processing {period} aggregation from {min_date.date()} to {max_date.date()}")
    
    if period == "week":
        # Generate all ISO weeks in the range, but limit to max_date
        current_week = min_date - pd.to_timedelta(min_date.weekday(), unit="D")  # Monday of first week
        week_count = 0
        total_weeks = int((max_date - current_week).days / 7) + 1
        
        while current_week <= max_date:
            week_count += 1
            week_end = current_week + pd.to_timedelta(6, unit="D")
            
            # Find policies traveling during this week
            traveling_mask = (tmp["dateDepart"] <= week_end) & (tmp["dateReturn"] >= current_week)
            traveling_policies = tmp[traveling_mask]
            
            if len(traveling_policies) > 0:
                # Calculate normalized x-axis value
                x_norm = pd.Timestamp(2000, 1, 3) + pd.to_timedelta((current_week.isocalendar().week - 1) * 7, unit="D")
                year = current_week.year
                
                # For each group (if segment grouping), calculate metrics
                if extra_cols:
                    for group_vals, group_df in traveling_policies.groupby(extra_cols, dropna=False):
                        if not isinstance(group_vals, tuple):
                            group_vals = (group_vals,)
                        
                        # Calculate nights in this week for each policy
                        nights_in_week = []
                        for _, policy in group_df.iterrows():
                            start_in_week = max(policy["dateDepart"], current_week)
                            end_in_week = min(policy["dateReturn"], week_end)
                            nights = (end_in_week - start_in_week).days + 1
                            nights_in_week.append(nights)
                        
                        # Aggregate metrics for this group
                        volume_unique = len(group_df)
                        maxTripCostExposure_unique = group_df["tripCost"].sum()
                        tripCostPerNightExposure_period = (group_df["tripCost"] / group_df["nightsCount"] * nights_in_week).sum()
                        
                        record = [year, x_norm, volume_unique, maxTripCostExposure_unique, tripCostPerNightExposure_period] + list(group_vals)
                        records.append(record)
                else:
                    # No grouping - aggregate all traveling policies
                    nights_in_week = []
                    for _, policy in traveling_policies.iterrows():
                        start_in_week = max(policy["dateDepart"], current_week)
                        end_in_week = min(policy["dateReturn"], week_end)
                        nights = (end_in_week - start_in_week).days + 1
                        nights_in_week.append(nights)
                    
                    volume_unique = len(traveling_policies)
                    maxTripCostExposure_unique = traveling_policies["tripCost"].sum()
                    tripCostPerNightExposure_period = (traveling_policies["tripCost"] / traveling_policies["nightsCount"] * nights_in_week).sum()
                    
                    record = [year, x_norm, volume_unique, maxTripCostExposure_unique, tripCostPerNightExposure_period]
                    records.append(record)
            
            # Debug message every 10 weeks or at the end
            if week_count % 10 == 0 or current_week + pd.to_timedelta(7, unit="D") > max_date:
                print(f"  Processed week {week_count}/{total_weeks}: {current_week.date()} ({len(traveling_policies)} policies)")
            
            # Move to next week
            current_week += pd.to_timedelta(7, unit="D")
    
    else:  # month
        # Generate all months in the range
        current_month = min_date.replace(day=1)
        month_count = 0
        total_months = (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month) + 1
        
        while current_month <= max_date:
            month_count += 1
            # Calculate month end
            if current_month.month == 12:
                next_month = current_month.replace(year=current_month.year + 1, month=1, day=1)
            else:
                next_month = current_month.replace(month=current_month.month + 1, day=1)
            month_end = next_month - pd.to_timedelta(1, unit="D")
            
            # Find policies traveling during this month
            traveling_mask = (tmp["dateDepart"] <= month_end) & (tmp["dateReturn"] >= current_month)
            traveling_policies = tmp[traveling_mask]
            
            if len(traveling_policies) > 0:
                # Calculate normalized x-axis value
                x_norm = pd.Timestamp(year=2000, month=current_month.month, day=1)
                year = current_month.year
                
                # For each group (if segment grouping), calculate metrics
                if extra_cols:
                    for group_vals, group_df in traveling_policies.groupby(extra_cols, dropna=False):
                        if not isinstance(group_vals, tuple):
                            group_vals = (group_vals,)
                        
                        # Calculate nights in this month for each policy
                        nights_in_month = []
                        for _, policy in group_df.iterrows():
                            start_in_month = max(policy["dateDepart"], current_month)
                            end_in_month = min(policy["dateReturn"], month_end)
                            nights = (end_in_month - start_in_month).days + 1
                            nights_in_month.append(nights)
                        
                        # Aggregate metrics for this group
                        volume_unique = len(group_df)
                        maxTripCostExposure_unique = group_df["tripCost"].sum()
                        tripCostPerNightExposure_period = (group_df["tripCost"] / group_df["nightsCount"] * nights_in_month).sum()
                        
                        record = [year, x_norm, volume_unique, maxTripCostExposure_unique, tripCostPerNightExposure_period] + list(group_vals)
                        records.append(record)
                else:
                    # No grouping - aggregate all traveling policies
                    nights_in_month = []
                    for _, policy in traveling_policies.iterrows():
                        start_in_month = max(policy["dateDepart"], current_month)
                        end_in_month = min(policy["dateReturn"], month_end)
                        nights = (end_in_month - start_in_month).days + 1
                        nights_in_month.append(nights)
                    
                    volume_unique = len(traveling_policies)
                    maxTripCostExposure_unique = traveling_policies["tripCost"].sum()
                    tripCostPerNightExposure_period = (traveling_policies["tripCost"] / traveling_policies["nightsCount"] * nights_in_month).sum()
                    
                    record = [year, x_norm, volume_unique, maxTripCostExposure_unique, tripCostPerNightExposure_period]
                    records.append(record)
            
            # Debug message every 5 months or at the end
            if month_count % 5 == 0 or next_month > max_date:
                print(f"  Processed month {month_count}/{total_months}: {current_month.strftime('%Y-%m')} ({len(traveling_policies)} policies)")
            
            # Move to next month
            current_month = next_month

    if not records:
        cols = ["year", "x", "volume_unique", "maxTripCostExposure_unique", "tripCostPerNightExposure_period"] + extra_cols
        return pd.DataFrame(columns=cols)

    cols = ["year", "x", "volume_unique", "maxTripCostExposure_unique", "tripCostPerNightExposure_period"] + extra_cols
    df_rec = pd.DataFrame.from_records(records, columns=cols)

    sort_cols = ["x", "year"] + extra_cols
    df_rec = df_rec.sort_values(sort_cols)
    
    print(f"Completed {period} aggregation: {len(df_rec)} records")
    return df_rec


# ---------- Precomputation helpers ----------

def _agg_filename(folder: Path, kind: str, period: str, by_segment: bool) -> Path:
    suffix = "by_segment" if by_segment else "all"
    return folder / f"agg_{kind}_{period}_{suffix}.parquet"


def precompute_aggregates(folder: Path) -> None:
    """
    Precompute key aggregates from combined.parquet into small parquet files for fast dashboarding.
    Creates:
      - agg_travel_{day,week,month}_{all,by_segment}.parquet
      - agg_depart_{day,week,month}_{all,by_segment}.parquet
    """
    combined_path = folder / "combined.parquet"
    if not combined_path.exists():
        return
    df = pd.read_parquet(combined_path)

    for period in ["day", "week", "month"]:
        # Traveling
        if period == "day":
            agg_travel_all = aggregate_traveling_by_period(df, period=period)
        else:
            agg_travel_all = aggregate_traveling_unique_by_period(df, period=period)
        agg_travel_all.to_parquet(_agg_filename(folder, "travel", period, False), index=False)
        if "segment" in df.columns:
            if period == "day":
                agg_travel_seg = aggregate_traveling_by_period(df, period=period, additional_group_by="segment")
            else:
                agg_travel_seg = aggregate_traveling_unique_by_period(df, period=period, additional_group_by="segment")
            agg_travel_seg.to_parquet(_agg_filename(folder, "travel", period, True), index=False)

        # Departures
        agg_dep_all = aggregate_departures_by_period(df, period=period)
        agg_dep_all.to_parquet(_agg_filename(folder, "depart", period, False), index=False)
        if "segment" in df.columns:
            agg_dep_seg = aggregate_departures_by_period(df, period=period, additional_group_by="segment")
            agg_dep_seg.to_parquet(_agg_filename(folder, "depart", period, True), index=False)


def load_precomputed_aggregate(folder_path: str, kind: str, period: str, by_segment: bool) -> Optional[pd.DataFrame]:
    folder = Path(folder_path)
    path = _agg_filename(folder, kind, period, by_segment)
    if path.exists():
        return pd.read_parquet(path)
    # Ensure week/all travel aggregate exists at minimum
    if kind == "travel" and period == "week" and by_segment is False:
        combined_path = folder / "combined.parquet"
        if combined_path.exists():
            df = pd.read_parquet(combined_path)
            agg = aggregate_traveling_by_period(df, period="week")
            agg.to_parquet(path, index=False)
            return agg
    return None


def precompute_all_with_timing(folder_path: str) -> pd.DataFrame:
    """
    Compute all aggregates (travel/depart x day/week/month x all/segment) and time each.
    Returns a DataFrame report with columns: kind, period, scope, rows, seconds, created
    """
    folder = Path(folder_path)
    combined_path = folder / "combined.parquet"
    if not combined_path.exists():
        raise FileNotFoundError(f"combined.parquet not found in {folder}")
    df = pd.read_parquet(combined_path)

    tasks = []
    for kind in ["travel", "depart"]:
        for period in ["day", "week", "month"]:
            for by_segment in [False, True if "segment" in df.columns else False]:
                if by_segment is False:
                    tasks.append((kind, period, False))
                elif by_segment is True:
                    tasks.append((kind, period, True))

    records = []
    for kind, period, by_segment in tasks:
        path = _agg_filename(folder, kind, period, by_segment)
        t0 = time.perf_counter()
        created = False
        if kind == "travel":
            if period == "day":
                agg = aggregate_traveling_by_period(df, period=period, additional_group_by=("segment" if by_segment else None))
            else:
                agg = aggregate_traveling_unique_by_period(df, period=period, additional_group_by=("segment" if by_segment else None))
        else:
            agg = aggregate_departures_by_period(df, period=period, additional_group_by=("segment" if by_segment else None))
        agg.to_parquet(path, index=False)
        created = True
        t1 = time.perf_counter()
        records.append({
            "kind": kind,
            "period": period,
            "scope": "by_segment" if by_segment else "all",
            "rows": len(agg),
            "seconds": round(t1 - t0, 3),
            "created": created,
            "path": str(path),
        })

    return pd.DataFrame.from_records(records)


