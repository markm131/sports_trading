# src/data/cleaner.py
"""Clean and filter raw match data"""

import pandas as pd

from src.config import (
    COLUMN_MAPPING,
    ESSENTIAL_COLUMNS,
    LEAGUES,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)


def filter_and_rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to essential columns and rename them to be descriptive"""

    # Select essential columns - will raise KeyError if missing
    df_filtered = df[ESSENTIAL_COLUMNS].copy()

    # Rename to descriptive names
    df_renamed = df_filtered.rename(columns=COLUMN_MAPPING)

    return df_renamed


def clean_combined_file(league: str) -> pd.DataFrame:
    """Load combined raw file, filter columns, rename, and save cleaned version

    Args:
        league: League name (e.g., 'premier_league')

    Returns:
        Cleaned DataFrame
    """
    league_info = LEAGUES[league]
    country = league_info["country"]

    # Load raw combined file from country/league subdirectory
    raw_file = RAW_DATA_DIR / country / league / f"{league}_combined.csv"
    print(f"\nProcessing {league}...")
    print(f"  Loading: {raw_file}")

    df = pd.read_csv(raw_file)
    print(f"  Raw data: {len(df)} matches, {len(df.columns)} columns")

    # Filter and rename columns
    df_cleaned = filter_and_rename_columns(df)
    print(f"  Cleaned: {len(df_cleaned)} matches, {len(df_cleaned.columns)} columns")

    # Show sample of new column names
    print(f"  Columns: {', '.join(df_cleaned.columns[:5])}...")

    # Save cleaned version to country subdirectory
    processed_dir = PROCESSED_DATA_DIR / country
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_file = processed_dir / f"{league}_cleaned.csv"
    df_cleaned.to_csv(output_file, index=False)

    print(f"  Saved: {output_file}")

    return df_cleaned


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Clean raw match data")
    parser.add_argument(
        "--league",
        default="premier_league",
        choices=list(LEAGUES.keys()),
        help="League to clean",
    )
    parser.add_argument("--all-leagues", action="store_true", help="Clean all leagues")

    args = parser.parse_args()

    # Determine which leagues to process
    leagues = ["premier_league", "championship"] if args.all_leagues else [args.league]

    # Process each league
    for league in leagues:
        clean_combined_file(league)

    print("\nCleaning complete!")


if __name__ == "__main__":
    main()
