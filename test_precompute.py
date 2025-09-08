import argparse
import os
from pathlib import Path

import pandas as pd

from exposure_data import (
    load_folder_policies,
    precompute_all_with_timing,
)


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug precompute aggregates and show outputs")
    parser.add_argument("folder", help="Path to extract folder (e.g., _data/2025-08-01)")
    parser.add_argument("--rebuild-combined", action="store_true", help="Rebuild combined.parquet from CSVs")
    parser.add_argument("--erase-parquet", action="store_true", help="Erase combined.parquet before starting")
    parser.add_argument("--head", type=int, default=5, help="Number of rows to show from each aggregate")
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    combined_path = folder / "combined.parquet"

    print_section("Step 1: Prepare combined.parquet")
    if args.erase_parquet and combined_path.exists():
        print(f"- Erasing {combined_path}")
        combined_path.unlink(missing_ok=True)

    if args.rebuild_combined or not combined_path.exists():
        print(f"- Building combined.parquet from CSVs in {folder}")
        # Load without precomputing aggregates to avoid redundancy
        from exposure_data import _read_policy_csv, normalize_policies_df
        import glob
        
        csv_files = list(folder.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {folder}")
        
        dfs = []
        for csv_file in csv_files:
            df = _read_policy_csv(str(csv_file))
            dfs.append(df)
        
        df_combined = pd.concat(dfs, ignore_index=True)
        df_combined = normalize_policies_df(df_combined)
        df_combined.to_parquet(combined_path, index=False)
        print(f"Saved combined.parquet with {len(df_combined)} policies")
    else:
        print(f"- Loading existing {combined_path}")
        df_combined = pd.read_parquet(combined_path)

    print(f"combined.parquet shape: {df_combined.shape}")
    print(df_combined.head(min(args.head, len(df_combined))))

    print_section("Step 2: Precompute all aggregates with timing")
    report = precompute_all_with_timing(str(folder))
    print(report)

    print_section("Step 3: Inspect each parquet aggregate")
    for _, row in report.iterrows():
        path = Path(row["path"])
        exists = path.exists()
        print(f"- {row['kind']}/{row['period']}/{row['scope']} -> {path} | exists={exists} | rows={row['rows']} | {row['seconds']}s")
        if exists:
            df = pd.read_parquet(path)
            print(f"  DataFrame: shape={df.shape} cols={list(df.columns)}")
            print(df.head(min(args.head, len(df))))


if __name__ == "__main__":
    main()


