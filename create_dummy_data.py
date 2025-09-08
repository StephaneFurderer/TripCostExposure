import csv
import math
import os
import random
from datetime import date, datetime, timedelta
from typing import List, Tuple


RANDOM_SEED = 42
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "_data", "policies.csv")
NUM_POLICIES = 10_000

# Date constraints
DATE_APP_START = date(2022, 1, 1)
DATE_APP_END = date(2024, 12, 31)

# Departure window relative to application date
MIN_DEPART_OFFSET_DAYS = 1
MAX_DEPART_OFFSET_DAYS = 180  # ~6 months
TARGET_MEAN_DEPART_OFFSET_DAYS = 14

# Nights constraints
MIN_NIGHTS = 1
MAX_NIGHTS = 21
TARGET_MEAN_NIGHTS = 4

# Per-night cost distribution (truncated normal > 0)
PER_NIGHT_MEAN = 100.0
PER_NIGHT_STD = 25.0

# Segments and probabilities
SEGMENTS = ["Economy", "Standard", "Premium"]
SEGMENT_PROBS = [0.50, 0.35, 0.15]


def first_monday_of_september(year: int) -> date:
    d = date(year, 9, 1)
    # Monday is 0
    delta = (0 - d.weekday()) % 7
    return d + timedelta(days=delta)


def second_monday_of_october(year: int) -> date:
    d = date(year, 10, 1)
    delta = (0 - d.weekday()) % 7
    first_monday = d + timedelta(days=delta)
    return first_monday + timedelta(days=7)


def daterange(start_date: date, end_date: date) -> List[date]:
    days = (end_date - start_date).days
    return [start_date + timedelta(days=i) for i in range(days + 1)]


def build_seasonal_departure_calendar(start_date: date, end_date: date) -> Tuple[List[date], List[float]]:
    """
    Build a list of dates and associated positive weights to induce seasonal peaks:
      - Summer peak (roughly mid-June through end of August)
      - Labor Day (first Monday in September) bump (+/- ~7 days)
      - Columbus Day (second Monday in October) bump (+/- ~7 days)
      - Christmas weeks (Dec 15 to Dec 31 and Jan 1 to Jan 7) bump
    """
    all_dates = daterange(start_date, end_date)
    weights = [1.0 for _ in all_dates]  # base uniform

    for idx, d in enumerate(all_dates):
        year = d.year

        # Summer multiplier: June 15 - Aug 31
        summer_start = date(year, 6, 15)
        summer_end = date(year, 8, 31)
        if summer_start <= d <= summer_end:
            weights[idx] *= 2.0

        # Labor Day bump: Gaussian around the holiday within +/- 10 days
        labor_day = first_monday_of_september(year)
        ld_days = abs((d - labor_day).days)
        if ld_days <= 10:
            # Gaussian bump centered at 0, sigma ~= 4
            bump = math.exp(-0.5 * (ld_days / 4.0) ** 2) * 1.5
            weights[idx] += bump

        # Columbus Day bump: Gaussian around the holiday within +/- 10 days
        columbus_day = second_monday_of_october(year)
        cd_days = abs((d - columbus_day).days)
        if cd_days <= 10:
            bump = math.exp(-0.5 * (cd_days / 4.0) ** 2) * 1.2
            weights[idx] += bump

        # Christmas weeks bump: Dec 15 - Dec 31 and Jan 1 - Jan 7
        xmas_start = date(year, 12, 15)
        xmas_end = date(year, 12, 31)
        if xmas_start <= d <= xmas_end:
            weights[idx] *= 2.2

        # New year week of next year (handle Jan 1 - Jan 7)
        if d.month == 1 and 1 <= d.day <= 7:
            weights[idx] *= 1.8

    # Normalize weights to sum to 1 for numerical stability in sampling
    total = sum(weights)
    if total <= 0:
        # Fallback to uniform if somehow zero
        weights = [1.0 for _ in weights]
        total = sum(weights)
    norm_weights = [w / total for w in weights]
    return all_dates, norm_weights


def sample_departure_date(calendar_dates: List[date], calendar_weights: List[float]) -> date:
    return random.choices(calendar_dates, weights=calendar_weights, k=1)[0]


def sample_truncated_discrete_offset(min_days: int, max_days: int, target_mean: float) -> int:
    """
    Sample a truncated discrete offset (in days) between [min_days, max_days],
    with an average around target_mean. We'll use a shifted Gamma-like discrete distribution
    via a Poisson with mean=lambda_, then clamp to range and retry if needed.
    """
    # Choose lambda to be close to target_mean, apply small randomness to avoid degenerate mode
    lam = max(0.1, target_mean)
    # Draw until in range
    while True:
        # Poisson-like via exponential sum approximation: use random.gammavariate(k, theta) ~ Gamma
        # But Python standard lib doesn't have Poisson, so approximate by rounding Gamma(k=lam, theta=1)
        # For small lam, this is rough but acceptable for synthetic data.
        val = int(round(random.gammavariate(lam, 1.0)))
        if val < min_days:
            val = min_days
        if val > max_days:
            # Retry to respect the hard max
            continue
        if min_days <= val <= max_days:
            return val


def sample_truncated_poisson_nights(min_nights: int, max_nights: int, target_mean: float) -> int:
    lam = max(0.1, target_mean)
    while True:
        val = int(round(random.gammavariate(lam, 1.0)))
        if val < min_nights:
            val = min_nights
        if val > max_nights:
            continue
        if min_nights <= val <= max_nights:
            return val


def sample_truncated_normal_positive(mean: float, std: float) -> float:
    # Box-Muller via random.gauss and truncate at > 0
    while True:
        x = random.gauss(mean, std)
        if x > 0:
            return x


def random_us_zip_code() -> str:
    # 5-digit ZIP with leading zeros allowed
    return f"{random.randint(0, 99999):05d}"


def random_travelers_count() -> int:
    # Skew towards smaller groups
    # Probabilities for counts 1..5
    counts = [1, 2, 3, 4, 5]
    probs = [0.55, 0.25, 0.12, 0.05, 0.03]
    return random.choices(counts, weights=probs, k=1)[0]


def generate_policies(num_policies: int) -> List[dict]:
    random.seed(RANDOM_SEED)

    # Build a wide departure calendar to accommodate max offset past last app date
    depart_calendar_start = DATE_APP_START + timedelta(days=MIN_DEPART_OFFSET_DAYS)
    # Allow departures up to MAX_DEPART_OFFSET_DAYS after last app date
    depart_calendar_end = DATE_APP_END + timedelta(days=MAX_DEPART_OFFSET_DAYS)
    cal_dates, cal_weights = build_seasonal_departure_calendar(depart_calendar_start, depart_calendar_end)

    policies = []
    attempts = 0
    while len(policies) < num_policies:
        attempts += 1
        if attempts > num_policies * 50:
            # Safety to avoid infinite loop in pathological cases
            break

        # Sample a seasonally weighted departure date
        date_depart = sample_departure_date(cal_dates, cal_weights)

        # Sample offset with target mean ~14 days, [1, 180]
        offset_days = sample_truncated_discrete_offset(
            MIN_DEPART_OFFSET_DAYS, MAX_DEPART_OFFSET_DAYS, TARGET_MEAN_DEPART_OFFSET_DAYS
        )

        date_app = date_depart - timedelta(days=offset_days)
        if not (DATE_APP_START <= date_app <= DATE_APP_END):
            continue  # keep only records with dateApp in the specified window

        # Nights
        nights = sample_truncated_poisson_nights(MIN_NIGHTS, MAX_NIGHTS, TARGET_MEAN_NIGHTS)
        date_return = date_depart + timedelta(days=nights)

        # Per-night cost and total trip cost
        per_night_cost = sample_truncated_normal_positive(PER_NIGHT_MEAN, PER_NIGHT_STD)
        total_cost = round(per_night_cost * nights, 2)

        # Segment
        segment = random.choices(SEGMENTS, weights=SEGMENT_PROBS, k=1)[0]

        # Travelers
        travelers = random_travelers_count()

        # ZIP and country
        zip_code = random_us_zip_code()
        country = "US"

        policies.append(
            {
                "segment": segment,
                "dateApp": date_app.isoformat(),
                "dateDepart": date_depart.isoformat(),
                "dateReturn": date_return.isoformat(),
                "ZipCode": zip_code,
                "Country": country,
                "tripCost": f"{total_cost:.2f}",
                "travelersCount": str(travelers),
                "nightsCount": str(nights),
            }
        )

    return policies


def write_csv(policies: List[dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "segment",
        "dateApp",
        "dateDepart",
        "dateReturn",
        "ZipCode",
        "Country",
        "tripCost",
        "travelersCount",
        "nightsCount",
    ]
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(policies)


def main() -> None:
    policies = generate_policies(NUM_POLICIES)
    write_csv(policies, OUTPUT_PATH)
    print(f"Wrote {len(policies)} policies to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


