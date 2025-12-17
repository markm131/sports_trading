# src/data/db_writer.py
"""Load cleaned match data into SQLite database with proper date handling"""

import sqlite3
from typing import Any, Dict, Tuple

import pandas as pd

from src.config import LEAGUES, PROCESSED_DATA_DIR, PROJECT_ROOT


def get_db_connection():
    """Get SQLite database connection"""
    db_path = PROJECT_ROOT / "db" / "betfair.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def prepare_match_data(row: pd.Series, league_info: dict) -> Dict[str, Any]:
    """Prepare match data for database insertion with proper date conversion"""
    match_date_raw = row["match_date"]
    match_date_iso = match_date_raw

    try:
        dt = pd.to_datetime(match_date_raw, format="%d/%m/%Y")
        match_date_iso = dt.strftime("%Y-%m-%d")
    except Exception:
        try:
            dt = pd.to_datetime(match_date_raw, format="%d/%m/%y")
            match_date_iso = dt.strftime("%Y-%m-%d")
        except Exception:
            try:
                dt = pd.to_datetime(match_date_raw, dayfirst=True)
                match_date_iso = dt.strftime("%Y-%m-%d")
            except Exception:
                print(f"  WARNING: Could not parse date: {match_date_raw}")
                match_date_iso = match_date_raw

    data = {
        "country": row.get("country", league_info["country"]),
        "league": row["league"],
        "season": row["season"],
        "match_date": match_date_iso,
        "match_time": row.get("match_time"),
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "referee": row.get("referee"),
        "full_time_home_goals": row.get("full_time_home_goals"),
        "full_time_away_goals": row.get("full_time_away_goals"),
        "full_time_result": row.get("full_time_result"),
        "half_time_home_goals": row.get("half_time_home_goals"),
        "half_time_away_goals": row.get("half_time_away_goals"),
        "half_time_result": row.get("half_time_result"),
        "home_shots": row.get("home_shots"),
        "away_shots": row.get("away_shots"),
        "home_shots_on_target": row.get("home_shots_on_target"),
        "away_shots_on_target": row.get("away_shots_on_target"),
        "home_fouls": row.get("home_fouls"),
        "away_fouls": row.get("away_fouls"),
        "home_yellow_cards": row.get("home_yellow_cards"),
        "away_yellow_cards": row.get("away_yellow_cards"),
        "home_red_cards": row.get("home_red_cards"),
        "away_red_cards": row.get("away_red_cards"),
        "home_corners": row.get("home_corners"),
        "away_corners": row.get("away_corners"),
        "pinnacle_home_odds": row.get("pinnacle_home_odds"),
        "pinnacle_draw_odds": row.get("pinnacle_draw_odds"),
        "pinnacle_away_odds": row.get("pinnacle_away_odds"),
        "pinnacle_over_2_5_odds": row.get("pinnacle_over_2_5_odds"),
        "pinnacle_under_2_5_odds": row.get("pinnacle_under_2_5_odds"),
    }

    for key, value in data.items():
        if pd.isna(value):
            data[key] = None

    return data


def load_league_to_db(league: str) -> Tuple[int, int, int]:
    """Load cleaned league data into database.

    Uses INSERT OR IGNORE to prevent duplicates.
    """
    league_info = LEAGUES[league]
    country = league_info["country"]
    league_name = league_info["name"]

    csv_path = PROCESSED_DATA_DIR / country / f"{league}_cleaned.csv"

    if not csv_path.exists():
        print(f"\nWARNING: Skipping {league} - file not found: {csv_path}")
        return 0, 0, 0

    print(f"\nLoading {league_name}...")

    df = pd.read_csv(csv_path)
    print(f"  Found {len(df)} matches in CSV")

    conn = get_db_connection()
    cursor = conn.cursor()

    inserted = 0
    skipped = 0
    errors = 0

    columns = [
        "country",
        "league",
        "season",
        "match_date",
        "match_time",
        "home_team",
        "away_team",
        "referee",
        "full_time_home_goals",
        "full_time_away_goals",
        "full_time_result",
        "half_time_home_goals",
        "half_time_away_goals",
        "half_time_result",
        "home_shots",
        "away_shots",
        "home_shots_on_target",
        "away_shots_on_target",
        "home_fouls",
        "away_fouls",
        "home_yellow_cards",
        "away_yellow_cards",
        "home_red_cards",
        "away_red_cards",
        "home_corners",
        "away_corners",
        "pinnacle_home_odds",
        "pinnacle_draw_odds",
        "pinnacle_away_odds",
        "pinnacle_over_2_5_odds",
        "pinnacle_under_2_5_odds",
    ]

    placeholders = ",".join(["?" for _ in columns])
    insert_query = (
        f"INSERT OR IGNORE INTO matches ({','.join(columns)}) VALUES ({placeholders})"
    )

    for idx, row in df.iterrows():
        try:
            match_data = prepare_match_data(row, league_info)
            values = [match_data.get(col) for col in columns]

            cursor.execute(insert_query, values)

            if cursor.rowcount > 0:
                inserted += 1
            else:
                skipped += 1

            if (inserted + skipped) % 500 == 0 and (inserted + skipped) > 0:
                print(f"    Processed {inserted + skipped} matches...")

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(
                    f"  ERROR: {row.get('match_date')} {row.get('home_team')} vs {row.get('away_team')}: {str(e)[:50]}"
                )
            continue

    conn.commit()
    conn.close()

    print(
        f"  Complete: {inserted} inserted, {skipped} skipped (duplicates), {errors} errors"
    )

    return inserted, skipped, errors


def create_database_schema():
    """Create the matches table with UNIQUE constraint"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Drop existing table to ensure clean schema
    cursor.execute("DROP TABLE IF EXISTS matches")

    cursor.execute("""
        CREATE TABLE matches (
            match_id INTEGER PRIMARY KEY AUTOINCREMENT,

            country TEXT NOT NULL,
            league TEXT NOT NULL,
            season TEXT NOT NULL,

            match_date TEXT NOT NULL,
            match_time TEXT,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            referee TEXT,

            full_time_home_goals INTEGER,
            full_time_away_goals INTEGER,
            full_time_result TEXT CHECK (full_time_result IN ('H', 'D', 'A', NULL)),

            half_time_home_goals INTEGER,
            half_time_away_goals INTEGER,
            half_time_result TEXT CHECK (half_time_result IN ('H', 'D', 'A', NULL)),

            home_shots INTEGER,
            away_shots INTEGER,
            home_shots_on_target INTEGER,
            away_shots_on_target INTEGER,

            home_fouls INTEGER,
            away_fouls INTEGER,
            home_yellow_cards INTEGER,
            away_yellow_cards INTEGER,
            home_red_cards INTEGER,
            away_red_cards INTEGER,

            home_corners INTEGER,
            away_corners INTEGER,

            pinnacle_home_odds REAL,
            pinnacle_draw_odds REAL,
            pinnacle_away_odds REAL,
            pinnacle_over_2_5_odds REAL,
            pinnacle_under_2_5_odds REAL,

            total_goals INTEGER GENERATED ALWAYS AS (full_time_home_goals + full_time_away_goals) STORED,

            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),

            UNIQUE(league, match_date, home_team, away_team)
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_match_date ON matches(match_date)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_league_date ON matches(league, match_date)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_teams ON matches(home_team, away_team)"
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_season ON matches(season, league)")

    conn.commit()
    conn.close()
    print("Database schema created successfully")


def load_all_leagues() -> None:
    """Load all leagues into database"""
    total_inserted = 0
    total_skipped = 0
    total_errors = 0

    for league in LEAGUES.keys():
        inserted, skipped, errors = load_league_to_db(league)
        total_inserted += inserted
        total_skipped += skipped
        total_errors += errors

    print(
        f"\nTotal: {total_inserted} inserted, {total_skipped} skipped, {total_errors} errors"
    )


def verify_load() -> None:
    """Show database stats"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM matches")
    total = cursor.fetchone()[0]
    print("\nDatabase Statistics")
    print(f"   Total matches: {total}")

    print("\n   By League:")
    cursor.execute(
        "SELECT league, COUNT(*) FROM matches GROUP BY league ORDER BY league"
    )
    for league, count in cursor.fetchall():
        print(f"     - {league}: {count}")

    print("\n   Recent Seasons:")
    cursor.execute("""
        SELECT season, COUNT(*)
        FROM matches
        GROUP BY season
        ORDER BY season DESC
        LIMIT 5
    """)
    for season, count in cursor.fetchall():
        print(f"     - {season}: {count}")

    conn.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Load match data into database")
    parser.add_argument("--all", action="store_true", help="Load all leagues")
    parser.add_argument(
        "--league", choices=list(LEAGUES.keys()), help="Load specific league"
    )
    parser.add_argument("--verify", action="store_true", help="Show DB stats")
    parser.add_argument(
        "--init", action="store_true", help="Initialize database schema"
    )

    args = parser.parse_args()

    if args.init:
        create_database_schema()
    elif args.verify:
        verify_load()
    elif args.all:
        load_all_leagues()
        verify_load()
    elif args.league:
        load_league_to_db(args.league)
        verify_load()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
