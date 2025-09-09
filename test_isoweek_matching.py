import pandas as pd

print("=== ISO Week Matching Example ===\n")

# Sample traveling patterns data
traveling_patterns = pd.DataFrame({
    'purchase_week': [
        pd.Timestamp('2021-09-13'),  # Week 37 of 2021
        pd.Timestamp('2022-09-12'),  # Week 37 of 2022  
        pd.Timestamp('2023-09-11'),  # Week 37 of 2023
        pd.Timestamp('2021-06-14'),  # Week 24 of 2021
        pd.Timestamp('2022-06-13'),  # Week 24 of 2022
    ],
    'weeks_after_purchase': [0, 0, 0, 0, 0],
    'proportion': [0.8, 0.7, 0.9, 0.5, 0.6]
})

print("Sample traveling patterns:")
print(traveling_patterns)
print()

# Show week numbers and years
traveling_patterns['week'] = traveling_patterns['purchase_week'].dt.isocalendar().week
traveling_patterns['year'] = traveling_patterns['purchase_week'].dt.isocalendar().year
print("With ISO week and year:")
print(traveling_patterns[['purchase_week', 'week', 'year', 'proportion']])
print()

# Forecast purchase week (Week 37 of 2024)
purchase_week = pd.Timestamp('2024-09-16')  # Week 38 of 2024
forecast_week = purchase_week.isocalendar().week
forecast_year = purchase_week.isocalendar().year

print(f"Forecast purchase week: {purchase_week}")
print(f"Forecast week: {forecast_week}, year: {forecast_year}")
print(f"Looking for patterns from week {forecast_week} of year {forecast_year - 1}")
print()

# Apply the matching logic
matching_patterns = traveling_patterns[
    (traveling_patterns['purchase_week'].dt.isocalendar().week == forecast_week) &
    (traveling_patterns['purchase_week'].dt.isocalendar().year == forecast_year - 1)
]

print("Matching patterns (same week from previous year):")
print(matching_patterns)
print()

if not matching_patterns.empty:
    avg_patterns = matching_patterns.groupby('weeks_after_purchase').agg({
        'proportion': 'mean'
    }).reset_index()
    print("Averaged proportions:")
    print(avg_patterns)
else:
    print("No matching patterns found")

print("\n=== Summary ===")
print(f"Looking for week {forecast_week} from year {forecast_year - 1}")
print(f"Found {len(matching_patterns)} matching patterns")
