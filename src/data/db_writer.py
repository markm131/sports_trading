# src/data/db_loader.py
"""Load cleaned match data into SQLite database with proper date handling"""

import sqlite3
from typing import Any, Dict, Tuple

import pandas as pd

from src.config import LEAGUES, PROCESSED_DATA_DIR, PROJECT_ROOT


def get_db_connection():
    """Get SQLite database connection"""
    db_path = PROJECT_ROOT / "db" / "betfair.db"
    return sqlite3.connect(db_path)


def prepare_match_data(row: pd.Series, league_info: dict) -> Dict[str, Any]:
    """Prepare match data for database insertion with proper date conversion"""
    # Convert DD/MM/YYYY or DD/MM/YY to YYYY-MM-DD for proper date handling
    match_date_raw = row["match_date"]
    match_date_iso = match_date_raw

    try:
        # Try DD/MM/YYYY format first (4-digit year)
        dt = pd.to_datetime(match_date_raw, format="%d/%m/%Y")
        match_date_iso = dt.strftime("%Y-%m-%d")
    except Exception:
        try:
            # Try DD/MM/YY format (2-digit year) - common in older data
            dt = pd.to_datetime(match_date_raw, format="%d/%m/%y")
            match_date_iso = dt.strftime("%Y-%m-%d")
        except Exception:
            # Last resort: let pandas infer the format
            try:
                dt = pd.to_datetime(match_date_raw, dayfirst=True)
                match_date_iso = dt.strftime("%Y-%m-%d")
            except Exception:
                print(f"  ⚠️  Could not parse date: {match_date_raw}")
                match_date_iso = match_date_raw

    # Build data dictionary
    data = {
        # Location
        "country": row.get("country", league_info["country"]),
        "league": row["league"],
        "season": row["season"],
        # Match identification
        "match_date": match_date_iso,
        "match_time": row.get("match_time"),
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "referee": row.get("referee"),
        # Results
        "full_time_home_goals": row.get("full_time_home_goals"),
        "full_time_away_goals": row.get("full_time_away_goals"),
        "full_time_result": row.get("full_time_result"),
        "half_time_home_goals": row.get("half_time_home_goals"),
        "half_time_away_goals": row.get("half_time_away_goals"),
        "half_time_result": row.get("half_time_result"),
        # Match statistics
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
        # Pinnacle odds
        "pinnacle_home_odds": row.get("pinnacle_home_odds"),
        "pinnacle_draw_odds": row.get("pinnacle_draw_odds"),
        "pinnacle_away_odds": row.get("pinnacle_away_odds"),
        "pinnacle_over_2_5_odds": row.get("pinnacle_over_2_5_odds"),
        "pinnacle_under_2_5_odds": row.get("pinnacle_under_2_5_odds"),
    }

    # Convert NaN to None for SQLite
    for key, value in data.items():
        if pd.isna(value):
            data[key] = None

    return data


def get_latest_match_date(conn, league: str) -> str:
    """Get the most recent match date in database for a league"""
    cursor = conn.cursor()

    query = """
        SELECT MAX(match_date) 
        FROM matches 
        WHERE league = ?
    """

    cursor.execute(query, (LEAGUES[league]["name"],))
    result = cursor.fetchone()

    # Return the date or a very old date if no matches exist
    return result[0] if result[0] else "2000-01-01"


def load_league_to_db(league: str, incremental: bool = False) -> Tuple[int, int, int]:
    """Load cleaned league data into database with upsert logic

    Args:
        league: League name (e.g., 'premier_league')
        incremental: If True, only load matches newer than latest in DB

    Returns:
        Tuple of (matches_inserted, matches_updated, matches_with_errors)
    """
    league_info = LEAGUES[league]
    country = league_info["country"]
    league_name = league_info["name"]

    # Read cleaned CSV
    csv_path = PROCESSED_DATA_DIR / country / f"{league}_cleaned.csv"

    # Check if file exists
    if not csv_path.exists():
        print(f"\n⚠️  Skipping {league} - file not found: {csv_path}")
        return 0, 0, 0

    print(f"\nLoading {league_name}...")

    df = pd.read_csv(csv_path)
    print(f"  Found {len(df)} matches in CSV")

    # Get database connection
    conn = get_db_connection()
    cursor = conn.cursor()

    # If incremental, filter to only new matches
    if incremental:
        latest_date = get_latest_match_date(conn, league)
        print(f"  Latest match in DB: {latest_date}")

        # Convert match_date to ISO format for comparison
        df["match_date_iso"] = pd.to_datetime(
            df["match_date"], format="%d/%m/%Y", errors="coerce"
        ).dt.strftime("%Y-%m-%d")

        # Filter to only matches after latest date
        df_new = df[df["match_date_iso"] > latest_date].copy()

        if len(df_new) == 0:
            print("  ✓ No new matches to process")
            conn.close()
            return 0, 0, 0

        print(f"  Found {len(df_new)} new matches to process")
        df = df_new  # Replace df with only new matches
    else:
        print(f"  Full reload mode - processing all {len(df)} matches")

    # Prepare for database operations
    inserted = 0
    updated = 0
    errors = 0

    # Column list for insert statement (excluding auto-generated fields)
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

    # Process each match
    for idx, row in df.iterrows():
        try:
            # Prepare data with proper conversions
            match_data = prepare_match_data(row, league_info)

            # Get values in column order
            values = [match_data.get(col) for col in columns]

            # Try to insert new row
            placeholders = ",".join(["?" for _ in columns])
            insert_query = (
                f"INSERT INTO matches ({','.join(columns)}) VALUES ({placeholders})"
            )

            try:
                cursor.execute(insert_query, values)
                inserted += 1
            except sqlite3.IntegrityError:
                # Match already exists - check if data has actually changed
                # Define update columns (exclude key fields)
                update_cols = [
                    col
                    for col in columns
                    if col not in ["match_date", "home_team", "away_team"]
                ]

                # First, fetch the existing row to compare
                check_query = f"""
                    SELECT {", ".join(update_cols)}
                    FROM matches 
                    WHERE match_date = ? AND home_team = ? AND away_team = ?
                """
                cursor.execute(
                    check_query,
                    [
                        match_data["match_date"],
                        match_data["home_team"],
                        match_data["away_team"],
                    ],
                )
                existing_row = cursor.fetchone()

                if existing_row:
                    # Compare new values with existing values
                    existing_values = list(existing_row)
                    new_values = [match_data.get(col) for col in update_cols]

                    # Check if any value has changed (handle None/NULL comparisons)
                    has_changed = False
                    for old_val, new_val in zip(existing_values, new_values):
                        # Handle None comparisons and convert to comparable types
                        old_comparable = None if old_val is None else old_val
                        new_comparable = (
                            None
                            if new_val is None
                            or (isinstance(new_val, float) and pd.isna(new_val))
                            else new_val
                        )

                        if old_comparable != new_comparable:
                            has_changed = True
                            break

                    # Only update if data has actually changed
                    if has_changed:
                        set_clause = ", ".join([f"{col} = ?" for col in update_cols])

                        update_query = f"""
                            UPDATE matches 
                            SET {set_clause}, updated_at = datetime('now')
                            WHERE match_date = ? AND home_team = ? AND away_team = ?
                        """

                        # Build update values: update columns + where clause values
                        update_values = [match_data.get(col) for col in update_cols]
                        update_values += [
                            match_data["match_date"],
                            match_data["home_team"],
                            match_data["away_team"],
                        ]

                        cursor.execute(update_query, update_values)
                        updated += 1

            # Progress indicator every 100 matches (only if something was processed)
            total_processed = inserted + updated
            if total_processed > 0 and total_processed % 100 == 0:
                print(f"    Processed {total_processed} matches...")

        except Exception as e:
            errors += 1
            print(
                f"  ⚠️  Error processing match {row.get('match_date', 'unknown')} "
                f"{row.get('home_team', 'unknown')} vs {row.get('away_team', 'unknown')}: {str(e)[:100]}"
            )

            # Continue processing other matches
            continue

    # Commit all changes
    conn.commit()
    conn.close()

    # Report results
    print("\n  ✓ Complete:")
    print(f"    - {inserted} new matches inserted")
    if updated > 0:
        print(f"    - {updated} matches updated")
    if errors > 0:
        print(f"    - {errors} errors encountered")

    return inserted, updated, errors


def create_database_schema():
    """Create the matches table with improved schema"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Drop existing table if doing a schema migration
    # cursor.execute("DROP TABLE IF EXISTS matches")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            match_id INTEGER PRIMARY KEY AUTOINCREMENT,
            
            -- Location
            country TEXT NOT NULL,
            league TEXT NOT NULL,
            season TEXT NOT NULL,
            
            -- Match identification  
            match_date TEXT NOT NULL,  -- YYYY-MM-DD format
            match_time TEXT,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            referee TEXT,
            
            -- Full time results
            full_time_home_goals INTEGER,
            full_time_away_goals INTEGER,
            full_time_result TEXT CHECK (full_time_result IN ('H', 'D', 'A', NULL)),
            
            -- Half time results
            half_time_home_goals INTEGER,
            half_time_away_goals INTEGER,
            half_time_result TEXT CHECK (half_time_result IN ('H', 'D', 'A', NULL)),
            
            -- Match statistics - shots
            home_shots INTEGER,
            away_shots INTEGER,
            home_shots_on_target INTEGER,
            away_shots_on_target INTEGER,
            
            -- Match statistics - fouls and cards
            home_fouls INTEGER,
            away_fouls INTEGER,
            home_yellow_cards INTEGER,
            away_yellow_cards INTEGER,
            home_red_cards INTEGER,
            away_red_cards INTEGER,
            
            -- Match statistics - corners
            home_corners INTEGER,
            away_corners INTEGER,
            
            -- Pinnacle odds
            pinnacle_home_odds REAL CHECK (pinnacle_home_odds > 1.0 OR pinnacle_home_odds IS NULL),
            pinnacle_draw_odds REAL CHECK (pinnacle_draw_odds > 1.0 OR pinnacle_draw_odds IS NULL),
            pinnacle_away_odds REAL CHECK (pinnacle_away_odds > 1.0 OR pinnacle_away_odds IS NULL),
            pinnacle_over_2_5_odds REAL CHECK (pinnacle_over_2_5_odds > 1.0 OR pinnacle_over_2_5_odds IS NULL),
            pinnacle_under_2_5_odds REAL CHECK (pinnacle_under_2_5_odds > 1.0 OR pinnacle_under_2_5_odds IS NULL),
            
            -- Calculated fields (SQLite 3.31.0+)
            total_goals INTEGER GENERATED ALWAYS AS (full_time_home_goals + full_time_away_goals) STORED,
            goal_difference INTEGER GENERATED ALWAYS AS (full_time_home_goals - full_time_away_goals) STORED,
            
            -- Data quality tracking
            data_version INTEGER DEFAULT 1,
            
            -- Metadata
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            
            -- Prevent duplicates
            UNIQUE(match_date, home_team, away_team)
        )
    """)

    # Create indexes for common queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_match_date ON matches(match_date)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_league_date ON matches(league, match_date)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_teams ON matches(home_team, away_team)"
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_season ON matches(season, league)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON matches(created_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_updated_at ON matches(updated_at)")

    conn.commit()
    conn.close()
    print("✓ Database schema created successfully")


def load_all_leagues() -> None:
    """Load all leagues into database"""
    total_inserted = 0
    total_updated = 0
    total_errors = 0

    for league in LEAGUES.keys():
        inserted, updated, errors = load_league_to_db(league)
        total_inserted += inserted
        total_updated += updated
        total_errors += errors

    print(
        f"\n✅ Total: {total_inserted} inserted, {total_updated} updated, {total_errors} errors"
    )


def verify_load() -> None:
    """Show database stats"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Total matches
    cursor.execute("SELECT COUNT(*) FROM matches")
    total = cursor.fetchone()[0]
    print("\n📊 Database Statistics")
    print(f"   Total matches: {total}")

    # By league
    print("\n   By League:")
    cursor.execute(
        "SELECT league, COUNT(*) FROM matches GROUP BY league ORDER BY league"
    )
    for league, count in cursor.fetchall():
        print(f"     • {league}: {count}")

    # By season
    print("\n   Recent Seasons:")
    cursor.execute("""
        SELECT season, COUNT(*) 
        FROM matches 
        GROUP BY season 
        ORDER BY season DESC 
        LIMIT 5
    """)
    for season, count in cursor.fetchall():
        print(f"     • {season}: {count}")

    # Recent updates
    cursor.execute("""
        SELECT COUNT(*) 
        FROM matches 
        WHERE updated_at > created_at
    """)
    updated_count = cursor.fetchone()[0]
    print(f"\n   Updated records: {updated_count}")

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
        print("\n💡 Examples:")
        print("   python -m src.data.db_loader --init           # Initialize database")
        print("   python -m src.data.db_loader --all            # Load all leagues")
        print("   python -m src.data.db_loader --league premier_league")
        print("   python -m src.data.db_loader --verify         # Check database stats")


if __name__ == "__main__":
    main()
