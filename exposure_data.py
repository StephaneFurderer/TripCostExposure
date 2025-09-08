from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, List, Union

import pandas as pd
import glob
from pathlib import Path


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
    df: pd.DataFrame, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None,
    additional_group_by: Optional[Union[str, List[str]]] = None,
) -> pd.DataFrame:
    """
    Expand policies across traveling days and compute per-day metrics:
      - volume: number of policies traveling
      - maxTripCostExposure: sum of total tripCost of the traveling policies
      - tripCostPerNightExposure: sum of (tripCost / nightsCount) across traveling policies
    Returns columns: ["day", "year", "volume", "maxTripCostExposure", "tripCostPerNightExposure", ...extras]
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

    # Default window bounds
    overall_start = tmp["dateDepart"].min()
    requested_start = pd.to_datetime(start_date).date() if start_date is not None else overall_start
    requested_end = pd.to_datetime(end_date).date() if end_date is not None else tmp["dateReturn"].max()

    # Clip to requested window
    tmp["start"] = tmp["dateDepart"].mask(tmp["dateDepart"] < requested_start, requested_start)
    tmp["end"] = tmp["dateReturn"].mask(tmp["dateReturn"] > requested_end, requested_end)

    # Remove any with start > end after clipping
    tmp = tmp[tmp["start"] <= tmp["end"]].copy()

    # Precompute per-night cost
    tmp["perNight"] = tmp["tripCost"] / tmp["nightsCount"].replace(0, pd.NA)
    tmp["perNight"] = tmp["perNight"].fillna(0.0)

    # Build rows per day via explode
    tmp["day"] = tmp.apply(lambda r: pd.date_range(r["start"], r["end"], freq="D"), axis=1)
    exploded = tmp.explode("day")
    exploded["day"] = exploded["day"].dt.date

    group_cols = ["day"] + extra_cols
    daily = exploded.groupby(group_cols, as_index=False).agg(
        volume=("tripCost", "count"),
        maxTripCostExposure=("tripCost", "sum"),
        tripCostPerNightExposure=("perNight", "sum"),
    )
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


