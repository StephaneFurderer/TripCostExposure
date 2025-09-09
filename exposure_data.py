from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, List, Union

import pandas as pd
import glob
from pathlib import Path
import time


DEFAULT_CSV_PATH = os.path.join(os.path.dirname(__file__), "_data", "policies.csv")


def classify_country(country_value) -> str:
    """
    Classify country values into US, ROW, or null categories.
    """
    if pd.isna(country_value) or country_value is None:
        return "null"
    elif str(country_value).upper() == "US":
        return "US"
    else:
        return "ROW"


def classify_region(zip_code: str) -> str:
    """
    Classify ZIP codes into coastal regions based on US Census Bureau data.
    Returns: 'Atlantic', 'Florida', 'Gulf', 'Pacific', or 'Other'
    """
    if pd.isna(zip_code) or zip_code is None:
        return "Other"
    
    zip_str = str(zip_code).strip()
    if len(zip_str) < 5:
        return "Other"
    
    # Extract first 3 digits for region classification
    zip_prefix = zip_str[:3]
    
    # Atlantic Coast (without Florida) - Maine to Georgia
    atlantic_prefixes = [
        # Maine (040-049)
        "040", "041", "042", "043", "044", "045", "046", "047", "048", "049",
        # New Hampshire (030-039)
        "030", "031", "032", "033", "034", "035", "036", "037", "038", "039",
        # Massachusetts (010-027)
        "010", "011", "012", "013", "014", "015", "016", "017", "018", "019",
        "020", "021", "022", "023", "024", "025", "026", "027",
        # Rhode Island (028-029)
        "028", "029",
        # Connecticut (060-069)
        "060", "061", "062", "063", "064", "065", "066", "067", "068", "069",
        # New York (100-149)
        "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
        "110", "111", "112", "113", "114", "115", "116", "117", "118", "119",
        "120", "121", "122", "123", "124", "125", "126", "127", "128", "129",
        "130", "131", "132", "133", "134", "135", "136", "137", "138", "139",
        "140", "141", "142", "143", "144", "145", "146", "147", "148", "149",
        # New Jersey (070-089)
        "070", "071", "072", "073", "074", "075", "076", "077", "078", "079",
        "080", "081", "082", "083", "084", "085", "086", "087", "088", "089",
        # Pennsylvania (150-196) - only coastal areas
        "150", "151", "152", "153", "154", "155", "156", "157", "158", "159",
        "160", "161", "162", "163", "164", "165", "166", "167", "168", "169",
        "170", "171", "172", "173", "174", "175", "176", "177", "178", "179",
        "180", "181", "182", "183", "184", "185", "186", "187", "188", "189",
        "190", "191", "192", "193", "194", "195", "196",
        # Delaware (197-199)
        "197", "198", "199",
        # Maryland (206-219)
        "206", "207", "208", "209", "210", "211", "212", "213", "214", "215",
        "216", "217", "218", "219",
        # Virginia (220-246) - coastal areas
        "220", "221", "222", "223", "224", "225", "226", "227", "228", "229",
        "230", "231", "232", "233", "234", "235", "236", "237", "238", "239",
        "240", "241", "242", "243", "244", "245", "246",
        # North Carolina (270-289) - coastal areas
        "270", "271", "272", "273", "274", "275", "276", "277", "278", "279",
        "280", "281", "282", "283", "284", "285", "286", "287", "288", "289",
        # South Carolina (290-299) - coastal areas
        "290", "291", "292", "293", "294", "295", "296", "297", "298", "299",
        # Georgia (300-319) - coastal areas
        "300", "301", "302", "303", "304", "305", "306", "307", "308", "309",
        "310", "311", "312", "313", "314", "315", "316", "317", "318", "319"
    ]
    
    # Florida - all Florida ZIP codes
    florida_prefixes = [
        "320", "321", "322", "323", "324", "325", "326", "327", "328", "329",
        "330", "331", "332", "333", "334", "335", "336", "337", "338", "339",
        "340", "341", "342", "343", "344", "345", "346", "347", "348", "349"
    ]
    
    # Gulf of Mexico (without Florida) - Texas, Louisiana, Mississippi, Alabama
    gulf_prefixes = [
        # Texas (770-799) - Gulf Coast areas
        "770", "771", "772", "773", "774", "775", "776", "777", "778", "779",
        "780", "781", "782", "783", "784", "785", "786", "787", "788", "789",
        "790", "791", "792", "793", "794", "795", "796", "797", "798", "799",
        # Louisiana (700-714)
        "700", "701", "702", "703", "704", "705", "706", "707", "708", "709",
        "710", "711", "712", "713", "714",
        # Mississippi (390-397)
        "390", "391", "392", "393", "394", "395", "396", "397",
        # Alabama (350-369) - Gulf Coast areas
        "350", "351", "352", "353", "354", "355", "356", "357", "358", "359",
        "360", "361", "362", "363", "364", "365", "366", "367", "368", "369"
    ]
    
    # Pacific Coast - Hawaii + California south of San Diego
    pacific_prefixes = [
        # Hawaii (967-968)
        "967", "968",
        # California (900-949) - Southern California coastal areas
        "900", "901", "902", "903", "904", "905", "906", "907", "908", "909",
        "910", "911", "912", "913", "914", "915", "916", "917", "918", "919",
        "920", "921", "922", "923", "924", "925", "926", "927", "928", "929",
        "930", "931", "932", "933", "934", "935", "936", "937", "938", "939",
        "940", "941", "942", "943", "944", "945", "946", "947", "948", "949"
    ]
    
    if zip_prefix in atlantic_prefixes:
        return "Atlantic"
    elif zip_prefix in florida_prefixes:
        return "Florida"
    elif zip_prefix in gulf_prefixes:
        return "Gulf"
    elif zip_prefix in pacific_prefixes:
        return "Pacific"
    else:
        return "Other"


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
    
    # Handle Country column with case-insensitive matching
    if "Country" not in df.columns:
        # search common variants case-insensitively
        lower_cols = {c.lower(): c for c in df.columns}
        candidate = None
        for key in ["country", "country_code", "nation"]:
            if key in lower_cols:
                candidate = lower_cols[key]
                break
        if candidate is not None:
            df["Country"] = df[candidate]
    
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

    # Include country and region classification
    base_cols = ["dateDepart", "dateReturn", "tripCost", "nightsCount"]
    if "Country" in df.columns:
        base_cols.append("Country")
    if "ZipCode" in df.columns:
        base_cols.append("ZipCode")
    
    tmp = df[base_cols + extra_cols].copy()
    tmp["dateDepart"] = pd.to_datetime(tmp["dateDepart"]).dt.date
    tmp["dateReturn"] = pd.to_datetime(tmp["dateReturn"]).dt.date
    
    # Add country and region classification
    if "Country" in tmp.columns:
        tmp["country_class"] = tmp["Country"].apply(classify_country)
    else:
        tmp["country_class"] = "null"
    
    if "ZipCode" in tmp.columns:
        tmp["region_class"] = tmp["ZipCode"].apply(classify_region)
    else:
        tmp["region_class"] = "Other"

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
            "avgTripCostPerNight": sign * per_night.values,  # Same as tripCostPerNightExposure for daily
            "country_class": tmp["country_class"].values,
            "region_class": tmp["region_class"].values,
        }
        events = pd.DataFrame(data)
        for c in extra_cols:
            events[c] = tmp[c].values
        return events

    ev_start = build_events(+1, start_ts)
    ev_end = build_events(-1, end_plus1_ts)
    events = pd.concat([ev_start, ev_end], ignore_index=True)

    # Aggregate deltas per day (and segment if needed)
    # Always include country and region classifications in grouping
    group_cols = ["day", "country_class", "region_class"] + extra_cols
    deltas = events.groupby(group_cols, as_index=False).sum()
    deltas = deltas.sort_values(group_cols)

    # Cumulative sum over days per group
    # We need a continuous date index over [requested_start, requested_end]
    full_days = pd.date_range(requested_start, requested_end, freq="D")

    if extra_cols or True:  # Always use grouping since we have country_class and region_class
        # Build grid per group for reindexing
        grids = []
        # Group by country_class, region_class, and extra_cols
        groupby_cols = ["country_class", "region_class"] + extra_cols
        # Handle single column case to avoid pandas warning
        groupby_cols_single = groupby_cols[0] if len(groupby_cols) == 1 else groupby_cols
        for keys, grp in deltas.groupby(groupby_cols_single, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            idx = pd.Index(full_days, name="day")
            # Only reindex numeric metric columns to avoid string fill errors
            metric_cols = ["volume", "maxTripCostExposure", "tripCostPerNightExposure", "avgTripCostPerNight"]
            grp_metrics = grp[["day"] + metric_cols]
            g = grp_metrics.set_index("day").reindex(idx, fill_value=0)
            # Add back grouping columns as columns
            for i, c in enumerate(groupby_cols):
                g[c] = keys[i]
            g = g.reset_index().rename(columns={"index": "day"})
            grids.append(g)
        deltas_full = pd.concat(grids, ignore_index=True)
        sort_cols = groupby_cols + ["day"]
        deltas_full = deltas_full.sort_values(sort_cols)
        # Perform cumulative sums via group-wise transform
        deltas_full = deltas_full.sort_values(sort_cols)
        for col in ["volume", "maxTripCostExposure", "tripCostPerNightExposure", "avgTripCostPerNight"]:
                # Handle single column case to avoid pandas warning
                groupby_cols_cumsum = groupby_cols[0] if len(groupby_cols) == 1 else groupby_cols
                deltas_full[col] = deltas_full.groupby(groupby_cols_cumsum, dropna=False)[col].cumsum()
        daily = deltas_full
    else:
        idx = pd.Index(full_days, name="day")
        deltas_full = deltas.set_index("day").reindex(idx, fill_value=0).reset_index()
        for col in ["volume", "maxTripCostExposure", "tripCostPerNightExposure", "avgTripCostPerNight"]:
            deltas_full[col] = deltas_full[col].cumsum()
        daily = deltas_full

    # Finalize
    daily["day"] = pd.to_datetime(daily["day"]).dt.date
    dt = pd.to_datetime(daily["day"])  # Timestamp
    daily["year"] = dt.dt.year.astype("int64")
    return daily[["day", "year", "country_class", "region_class", "volume", "maxTripCostExposure", "tripCostPerNightExposure", "avgTripCostPerNight"] + extra_cols]


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

    # Always include country and region classifications in grouping
    group_cols = ["year", "x", "country_class", "region_class"] + extra_cols
    agg = daily.groupby(group_cols, as_index=False).agg(
        volume=("volume", "sum"),
        maxTripCostExposure=("maxTripCostExposure", "sum"),
        tripCostPerNightExposure=("tripCostPerNightExposure", "sum"),
        avgTripCostPerNight=("avgTripCostPerNight", "mean"),
    )
    sort_cols = ["x", "year", "country_class", "region_class"] + extra_cols
    agg = agg.sort_values(sort_cols)
    return agg[["year", "x", "country_class", "region_class", "volume", "maxTripCostExposure", "tripCostPerNightExposure", "avgTripCostPerNight"] + extra_cols]


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

    # Include country and region classification
    base_cols = ["dateDepart", "tripCost", "nightsCount"]
    if "Country" in df.columns:
        base_cols.append("Country")
    if "ZipCode" in df.columns:
        base_cols.append("ZipCode")
    
    tmp = df[base_cols + extra_cols].copy()
    tmp["day"] = pd.to_datetime(tmp["dateDepart"]).dt.date
    tmp["perNight"] = tmp["tripCost"] / tmp["nightsCount"].replace(0, pd.NA)
    tmp["perNight"] = tmp["perNight"].fillna(0.0)
    
    # Add country and region classification
    if "Country" in tmp.columns:
        tmp["country_class"] = tmp["Country"].apply(classify_country)
    else:
        tmp["country_class"] = "null"
    
    if "ZipCode" in tmp.columns:
        tmp["region_class"] = tmp["ZipCode"].apply(classify_region)
    else:
        tmp["region_class"] = "Other"

    group_cols = ["day", "country_class", "region_class"] + extra_cols
    daily = tmp.groupby(group_cols, as_index=False).agg(
        volume=("tripCost", "count"),
        maxTripCostExposure=("tripCost", "sum"),
        avgTripCostPerNight=("perNight", "mean"),
        tripCostPerNightExposure=("perNight", "sum"),
    )
    dt = pd.to_datetime(daily["day"])  # Timestamp
    daily["year"] = dt.dt.year.astype("int64")
    return daily[["day", "year", "country_class", "region_class", "volume", "maxTripCostExposure", "avgTripCostPerNight", "tripCostPerNightExposure"] + extra_cols]


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

    # Always include country and region classifications in grouping
    group_cols = ["year", "x", "country_class", "region_class"] + extra_cols
    agg = daily.groupby(group_cols, as_index=False).agg(
        volume=("volume", "sum"),
        maxTripCostExposure=("maxTripCostExposure", "sum"),
        avgTripCostPerNight=("avgTripCostPerNight", "mean"),
        tripCostPerNightExposure=("tripCostPerNightExposure", "sum"),
    )
    sort_cols = ["x", "year", "country_class", "region_class"] + extra_cols
    agg = agg.sort_values(sort_cols)
    return agg[["year", "x", "country_class", "region_class", "volume", "maxTripCostExposure", "avgTripCostPerNight", "tripCostPerNightExposure"] + extra_cols]


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
    folder_path: Optional[str] = None,
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

    # Include country and region classification
    base_cols = ["dateDepart", "dateReturn", "tripCost", "nightsCount"]
    if "Country" in df.columns:
        base_cols.append("Country")
    if "ZipCode" in df.columns:
        base_cols.append("ZipCode")
    
    tmp = df[base_cols + extra_cols].copy()
    tmp["dateDepart"] = pd.to_datetime(tmp["dateDepart"]).dt.normalize()
    tmp["dateReturn"] = pd.to_datetime(tmp["dateReturn"]).dt.normalize()
    
    # Add country and region classification
    if "Country" in tmp.columns:
        tmp["country_class"] = tmp["Country"].apply(classify_country)
    else:
        tmp["country_class"] = "null"
    
    if "ZipCode" in tmp.columns:
        tmp["region_class"] = tmp["ZipCode"].apply(classify_region)
    else:
        tmp["region_class"] = "Other"

    per_night = (tmp["tripCost"] / tmp["nightsCount"].replace(0, pd.NA)).fillna(0.0)

    # Most efficient approach: iterate through periods and find traveling policies
    records = []
    
    # Get date range from all policies first
    all_dates = pd.concat([tmp["dateDepart"], tmp["dateReturn"]]).dropna()
    min_date = all_dates.min()
    data_max_date = all_dates.max()
    
    # Set default max_date to 1 year after extraction date if not provided
    if max_date is None:
        if folder_path:
            # Extract date from folder path (assumes folder name is YYYY-MM-DD format)
            import os
            folder_name = os.path.basename(folder_path)
            try:
                # Try to parse YYYY-MM-DD format
                extraction_date = pd.to_datetime(folder_name)
                max_date = extraction_date + pd.to_timedelta(365, unit="D")
                print(f"Using extraction date {extraction_date.date()}, max_date set to {max_date.date()}")
            except:
                # Fallback to 1 year after today if folder name doesn't match expected format
                max_date = pd.Timestamp.now() + pd.to_timedelta(365, unit="D")
                print(f"Could not parse folder date from '{folder_name}', using 1 year from today: {max_date.date()}")
        else:
            # Fallback to 1 year after today if no folder path provided
            max_date = pd.Timestamp.now() + pd.to_timedelta(365, unit="D")
            print(f"No folder path provided, using 1 year from today: {max_date.date()}")
    
    # If max_date is in the future compared to data, use the data's max date instead
    if max_date > data_max_date:
        max_date = data_max_date
        print(f"Adjusted max_date to data range: {max_date.date()}")
    
    # Determine scenario for debug output
    scenario = f"{period}"
    if additional_group_by:
        if isinstance(additional_group_by, str):
            scenario += f" by {additional_group_by}"
        else:
            scenario += f" by {', '.join(additional_group_by)}"
    else:
        scenario += " (all)"
    
    print(f"Processing {scenario} aggregation from {min_date.date()} to {max_date.date()}")
    
    if period == "week":
        # Process all weeks from min_date to max_date
        current_week = min_date - pd.to_timedelta(min_date.weekday(), unit="D")  # Monday of first week
        week_count = 0
        total_weeks = int((max_date - current_week).days / 7) + 1
        
        print(f"Processing all weeks from {current_week.date()} to {max_date.date()} ({total_weeks} weeks)")
        
        while current_week <= max_date:
            week_count += 1
            week_end = current_week + pd.to_timedelta(6, unit="D")
            
            # Find policies traveling during this week (same logic as week search)
            traveling_mask = (tmp["dateDepart"] <= week_end) & (tmp["dateReturn"] > current_week)
            traveling_policies = tmp[traveling_mask]
            
            if len(traveling_policies) > 0:
                # Calculate normalized x-axis value
                x_norm = pd.Timestamp(2000, 1, 3) + pd.to_timedelta((current_week.isocalendar().week - 1) * 7, unit="D")
                year = current_week.year
                
                # Calculate per-night cost (same as week search)
                traveling_policies = traveling_policies.copy()
                traveling_policies["perNight"] = (traveling_policies["tripCost"] / traveling_policies["nightsCount"].replace(0, pd.NA)).fillna(0.0)
                
                # Calculate nights in week (same logic as week search)
                night_range_start = traveling_policies["dateDepart"]
                night_range_end = traveling_policies["dateReturn"] - pd.to_timedelta(1, unit="D")
                overlap_start = night_range_start.where(night_range_start > current_week, current_week)
                overlap_end = night_range_end.where(night_range_end < week_end, week_end)
                delta = (overlap_end - overlap_start).dt.days + 1
                traveling_policies["nightsInWeek"] = delta.clip(lower=0).fillna(0).astype(int)
                traveling_policies["remainingTripCost"] = (traveling_policies["nightsInWeek"] * traveling_policies["perNight"]).round(2)
                
                # Always group by country_class and region_class, plus any extra_cols
                groupby_cols = ["country_class", "region_class"] + extra_cols
                # Handle single column case to avoid pandas warning
                groupby_cols_single = groupby_cols[0] if len(groupby_cols) == 1 else groupby_cols
                for group_vals, group_df in traveling_policies.groupby(groupby_cols_single, dropna=False):
                    if not isinstance(group_vals, tuple):
                        group_vals = (group_vals,)
                    
                    # Aggregate metrics for this group (using week search logic)
                    volume = len(group_df)
                    maxTripCostExposure = group_df["tripCost"].sum()
                    tripCostPerNightExposure = group_df["remainingTripCost"].sum()  # Use calculated remainingTripCost
                    avgTripCostPerNight = group_df["perNight"].mean()  # Use calculated perNight
                    
                    # Extract country_class and region_class from group_vals
                    country_class = group_vals[0] if len(group_vals) > 0 else "null"
                    region_class = group_vals[1] if len(group_vals) > 1 else "Other"
                    extra_vals = list(group_vals[2:]) if len(group_vals) > 2 else []
                    record = [year, x_norm, country_class, region_class, volume, maxTripCostExposure, tripCostPerNightExposure, avgTripCostPerNight] + extra_vals
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
            
            # Find policies traveling during this month (same logic as week search)
            traveling_mask = (tmp["dateDepart"] <= month_end) & (tmp["dateReturn"] > current_month)
            traveling_policies = tmp[traveling_mask]
            
            if len(traveling_policies) > 0:
                # Calculate normalized x-axis value
                x_norm = pd.Timestamp(year=2000, month=current_month.month, day=1)
                year = current_month.year
                
                # Calculate per-night cost (same as week search)
                traveling_policies = traveling_policies.copy()
                traveling_policies["perNight"] = (traveling_policies["tripCost"] / traveling_policies["nightsCount"].replace(0, pd.NA)).fillna(0.0)
                
                # Calculate nights in month (same logic as week search)
                night_range_start = traveling_policies["dateDepart"]
                night_range_end = traveling_policies["dateReturn"] - pd.to_timedelta(1, unit="D")
                overlap_start = night_range_start.where(night_range_start > current_month, current_month)
                overlap_end = night_range_end.where(night_range_end < month_end, month_end)
                delta = (overlap_end - overlap_start).dt.days + 1
                traveling_policies["nightsInMonth"] = delta.clip(lower=0).fillna(0).astype(int)
                traveling_policies["remainingTripCost"] = (traveling_policies["nightsInMonth"] * traveling_policies["perNight"]).round(2)
                
                # Always group by country_class and region_class, plus any extra_cols
                groupby_cols = ["country_class", "region_class"] + extra_cols
                # Handle single column case to avoid pandas warning
                groupby_cols_single = groupby_cols[0] if len(groupby_cols) == 1 else groupby_cols
                for group_vals, group_df in traveling_policies.groupby(groupby_cols_single, dropna=False):
                    if not isinstance(group_vals, tuple):
                        group_vals = (group_vals,)
                    
                    # Aggregate metrics for this group (using week search logic)
                    volume = len(group_df)
                    maxTripCostExposure = group_df["tripCost"].sum()
                    tripCostPerNightExposure = group_df["remainingTripCost"].sum()  # Use calculated remainingTripCost
                    avgTripCostPerNight = group_df["perNight"].mean()  # Use calculated perNight
                    
                    # Extract country_class and region_class from group_vals
                    country_class = group_vals[0] if len(group_vals) > 0 else "null"
                    region_class = group_vals[1] if len(group_vals) > 1 else "Other"
                    extra_vals = list(group_vals[2:]) if len(group_vals) > 2 else []
                    record = [year, x_norm, country_class, region_class, volume, maxTripCostExposure, tripCostPerNightExposure, avgTripCostPerNight] + extra_vals
                    records.append(record)
            
            # Debug message every 5 months or at the end
            if month_count % 5 == 0 or next_month > max_date:
                print(f"  Processed month {month_count}/{total_months}: {current_month.strftime('%Y-%m')} ({len(traveling_policies)} policies)")
            
            # Move to next month
            current_month = next_month

    if not records:
        cols = ["year", "x", "country_class", "region_class", "volume", "maxTripCostExposure", "tripCostPerNightExposure", "avgTripCostPerNight"] + extra_cols
        return pd.DataFrame(columns=cols)

    cols = ["year", "x", "country_class", "region_class", "volume", "maxTripCostExposure", "tripCostPerNightExposure", "avgTripCostPerNight"] + extra_cols
    df_rec = pd.DataFrame.from_records(records, columns=cols)

    sort_cols = ["x", "year", "country_class", "region_class"] + extra_cols
    df_rec = df_rec.sort_values(sort_cols)
    
    print(f"Completed {scenario} aggregation: {len(df_rec)} records")
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
            agg_travel_all = aggregate_traveling_unique_by_period(df, period=period, folder_path=str(folder))
        agg_travel_all.to_parquet(_agg_filename(folder, "travel", period, False), index=False)
        if "segment" in df.columns:
            if period == "day":
                agg_travel_seg = aggregate_traveling_by_period(df, period=period, additional_group_by="segment")
            else:
                agg_travel_seg = aggregate_traveling_unique_by_period(df, period=period, additional_group_by="segment", folder_path=str(folder))
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
        for period in ["day", "week"]:
            for by_segment in [True if "segment" in df.columns else False]:
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
                agg = aggregate_traveling_unique_by_period(df, period=period, additional_group_by=("segment" if by_segment else None), folder_path=str(folder))
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


