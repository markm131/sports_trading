# src/data/fetcher.py
"""Download historical match data from football-data.co.uk"""

import sys
from datetime import datetime
from typing import Optional

import pandas as pd

from src.config import (
    DEFAULT_END_YEAR,
    DEFAULT_START_YEAR,
    FOOTBALL_DATA_BASE_URL,
    LEAGUES,
    RAW_DATA_DIR,
    SEASON_START_MONTH,
)


def get_current_season() -> str:
    """Get current season code (e.g., '2526' for 2025-26 season)"""
    now = datetime.now()
    year = now.year if now.month >= SEASON_START_MONTH else now.year - 1
    return f"{str(year)[-2:]}{str(year + 1)[-2:]}"


def generate_season_code(start_year: int) -> str:
    """Convert year to season code (e.g., 2024 -> '2425')"""
    return f"{str(start_year)[-2:]}{str(start_year + 1)[-2:]}"


def format_season_readable(start_year: int) -> str:
    """Convert year to readable season (e.g., 2024 -> '2024-25')"""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def download_season(
    league: str,
    season_year: int,
    force: bool = False,
) -> Optional[pd.DataFrame]:
    """Download a single season"""
    league_info = LEAGUES[league]
    league_code = league_info["code"]
    league_name = league_info["name"]
    country = league_info["country"]

    league_dir = RAW_DATA_DIR / country / league
    league_dir.mkdir(parents=True, exist_ok=True)

    season_code = generate_season_code(season_year)
    season_readable = format_season_readable(season_year)

    filepath = league_dir / f"{league}_{season_readable}.csv"

    # Use cached file if exists and not forcing
    if filepath.exists() and not force:
        print(f"Using cached {league_name} {season_readable}")
        return pd.read_csv(filepath)

    # Download from web
    url = FOOTBALL_DATA_BASE_URL.format(season_code, league_code)
    try:
        df = pd.read_csv(url, encoding="latin-1")
        df["season"] = season_readable
        df["league"] = league_name

        # Save individual season file (overwrite)
        df.to_csv(filepath, index=False)

        print(f"Downloaded {league_name} {season_readable}: {len(df)} matches")
        return df
    except Exception as e:
        print(f"Failed {league_name} {season_readable}: {e}")
        return None


def download_multiple_seasons(
    league: str, start_year: int, end_year: int, force: bool = False
) -> Optional[pd.DataFrame]:
    """Download multiple seasons and combine them"""
    all_data = []

    for year in range(start_year, end_year):
        df = download_season(league, year, force)

        if df is not None:
            all_data.append(df)

    if not all_data:
        return None

    combined = pd.concat(all_data, ignore_index=True)
    
    # Deduplicate - keep last occurrence
    before = len(combined)
    combined = combined.drop_duplicates(
        subset=["Date", "HomeTeam", "AwayTeam", "season"],
        keep="last"
    )
    after = len(combined)
    if before != after:
        print(f"  Removed {before - after} duplicates")
    
    print(f"\nTotal: {len(combined)} matches from {len(all_data)} seasons")

    return combined


def save_combined_file(df: pd.DataFrame, league: str) -> None:
    """Save combined CSV file - always overwrites"""
    league_info = LEAGUES[league]
    country = league_info["country"]

    league_dir = RAW_DATA_DIR / country / league
    combined_path = league_dir / f"{league}_combined.csv"
    
    # Always overwrite, never append
    df.to_csv(combined_path, index=False)
    print(f"Saved: {combined_path}")


def weekly_update(
    league: str, start_year: int, end_year: int
) -> Optional[pd.DataFrame]:
    """Smart weekly update: force current season, use cached historical"""
    now = datetime.now()
    current_year = now.year if now.month >= SEASON_START_MONTH else now.year - 1

    league_name = LEAGUES[league]["name"]

    print(f"\nUpdating {league_name}...")
    print(f"  Current season: {format_season_readable(current_year)}")

    # Force re-download current season only
    current_df = download_season(league, current_year, force=True)

    if current_df is None:
        print("  Failed to update current season")
        return None

    # Rebuild combined file (uses cached historical + fresh current)
    combined = download_multiple_seasons(league, start_year, end_year, force=False)

    if combined is not None:
        save_combined_file(combined, league)
        print(f"  Update complete! {len(current_df)} matches in current season.")

    return combined


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download football data")
    parser.add_argument(
        "--league",
        default="premier_league",
        choices=list(LEAGUES.keys()),
        help="League name",
    )
    parser.add_argument(
        "--all-leagues", action="store_true", help="Download all leagues"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=DEFAULT_START_YEAR,
        help="First season start year",
    )
    parser.add_argument(
        "--end-year", type=int, default=DEFAULT_END_YEAR, help="Last season start year"
    )
    parser.add_argument("--update", action="store_true", help="Weekly update mode")
    parser.add_argument("--force", action="store_true", help="Force re-download all")

    args = parser.parse_args()

    leagues_to_process = list(LEAGUES.keys()) if args.all_leagues else [args.league]

    any_updates = False

    for league in leagues_to_process:
        if len(leagues_to_process) > 1 and not args.update:
            print(f"\n{'=' * 60}")
            print(f"Processing: {LEAGUES[league]['name']}")
            print(f"{'=' * 60}")

        if args.update:
            result = weekly_update(league, args.start_year, args.end_year)
            if result is not None:
                any_updates = True
        else:
            df = download_multiple_seasons(
                league, args.start_year, args.end_year, args.force
            )
            if df is not None:
                save_combined_file(df, league)
                any_updates = True

    if args.update and not any_updates:
        print("\nNo updates found for any league")
        sys.exit(2)
    elif args.update:
        print("\nUpdates processed successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()