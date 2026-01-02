# src/data/db_writer.py
"""Load cleaned match data into SQLite database with proper date handling"""

import sqlite3
from datetime import datetime
from typing import Any, Dict, Tuple

import pandas as pd

from src.config import LEAGUES, PROCESSED_DATA_DIR, PROJECT_ROOT


def get_db_connection():
    """Get SQLite database connection"""
    db_path = PROJECT_ROOT / "db" / "betfair.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


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


def create_betting_tables():
    """Create tables for paper trading / live betting."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Bankroll tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bankroll (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL DEFAULT (datetime('now')),
            event_type TEXT NOT NULL,  -- 'initial', 'bet_placed', 'bet_settled', 'adjustment'
            amount REAL NOT NULL,
            balance_after REAL NOT NULL,
            bet_id INTEGER,
            notes TEXT,
            FOREIGN KEY (bet_id) REFERENCES bets(bet_id)
        )
    """)

    # Bets table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bets (
            bet_id INTEGER PRIMARY KEY AUTOINCREMENT,
            
            -- When the bet was placed
            placed_at TEXT NOT NULL DEFAULT (datetime('now')),
            
            -- Match details
            league TEXT NOT NULL,
            match_date TEXT NOT NULL,
            match_time TEXT,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            market_id TEXT,
            
            -- Bet details
            selection TEXT NOT NULL CHECK (selection IN ('home', 'draw', 'away')),
            odds REAL NOT NULL,
            stake REAL NOT NULL,
            
            -- Model info at time of bet
            model_prob REAL NOT NULL,
            market_prob REAL NOT NULL,
            edge REAL NOT NULL,
            
            -- Closing line value tracking
            closing_odds REAL,  -- Odds at kickoff (filled by settlement)
            clv REAL,  -- Closing Line Value = (our_odds - closing_odds) / closing_odds
            
            -- Status
            status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'won', 'lost', 'void')),
            
            -- Settlement
            settled_at TEXT,
            actual_result TEXT CHECK (actual_result IN ('H', 'D', 'A', NULL)),
            profit REAL,
            
            -- Tracking
            is_paper BOOLEAN NOT NULL DEFAULT 1,  -- 1 = paper trade, 0 = real money
            
            created_at TEXT DEFAULT (datetime('now')),
            
            UNIQUE(match_date, home_team, away_team, selection)
        )
    """)

    # Opportunities tracking (value found but not yet bet)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS opportunities (
            opp_id INTEGER PRIMARY KEY AUTOINCREMENT,
            
            market_id TEXT NOT NULL,
            selection TEXT NOT NULL,
            
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            scans_with_value INTEGER DEFAULT 1,
            
            kickoff TEXT NOT NULL,
            league TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            
            current_edge REAL,
            current_odds REAL,
            best_edge REAL,
            best_odds REAL,
            
            bet_placed BOOLEAN DEFAULT 0,
            
            UNIQUE(market_id, selection)
        )
    """)

    # Odds history tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS odds_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scanned_at TEXT NOT NULL DEFAULT (datetime('now')),
            market_id TEXT,
            kickoff TEXT,
            mins_to_kickoff INTEGER,
            league TEXT,
            home_team TEXT,
            away_team TEXT,
            home_odds REAL,
            draw_odds REAL,
            away_odds REAL,
            home_edge REAL,
            draw_edge REAL,
            away_edge REAL
        )
    """)

    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bets_status ON bets(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bets_date ON bets(match_date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bets_league ON bets(league)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_opp_kickoff ON opportunities(kickoff)"
    )

    conn.commit()
    conn.close()
    print("Betting tables created successfully")


def init_bankroll(initial_amount: float = 1000.0):
    """Initialize bankroll with starting amount."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Check if already initialized
    cursor.execute("SELECT COUNT(*) FROM bankroll")
    if cursor.fetchone()[0] > 0:
        print("Bankroll already initialized")
        conn.close()
        return

    cursor.execute(
        """
        INSERT INTO bankroll (event_type, amount, balance_after, notes)
        VALUES ('initial', ?, ?, 'Paper trading started')
    """,
        (initial_amount, initial_amount),
    )

    conn.commit()
    conn.close()
    print(f"Bankroll initialized: £{initial_amount:.2f}")


def get_current_bankroll() -> float:
    """Get current bankroll balance."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT balance_after FROM bankroll ORDER BY id DESC LIMIT 1")
    result = cursor.fetchone()
    conn.close()

    return result[0] if result else 0.0


def get_bets_summary() -> Dict[str, Any]:
    """Get summary statistics for all bets."""
    conn = get_db_connection()

    # Overall stats
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM bets")
    total_bets = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM bets WHERE status = 'pending'")
    pending = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM bets WHERE status = 'won'")
    wins = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM bets WHERE status = 'lost'")
    losses = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COALESCE(SUM(profit), 0) FROM bets WHERE status IN ('won', 'lost')"
    )
    total_profit = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COALESCE(SUM(stake), 0) FROM bets WHERE status IN ('won', 'lost')"
    )
    total_staked = cursor.fetchone()[0]

    # By league
    leagues_df = pd.read_sql(
        """
        SELECT league,
               COUNT(*) as bets,
               SUM(CASE WHEN status = 'won' THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN status = 'lost' THEN 1 ELSE 0 END) as losses,
               COALESCE(SUM(profit), 0) as profit
        FROM bets
        WHERE status IN ('won', 'lost')
        GROUP BY league
        ORDER BY profit DESC
        """,
        conn,
    )

    # By selection type
    selection_df = pd.read_sql(
        """
        SELECT selection,
               COUNT(*) as bets,
               SUM(CASE WHEN status = 'won' THEN 1 ELSE 0 END) as wins,
               COALESCE(SUM(profit), 0) as profit,
               AVG(edge) as avg_edge
        FROM bets
        WHERE status IN ('won', 'lost')
        GROUP BY selection
        """,
        conn,
    )

    conn.close()

    settled = wins + losses
    win_rate = (wins / settled * 100) if settled > 0 else 0
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0

    return {
        "total_bets": total_bets,
        "pending": pending,
        "settled": settled,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_profit": total_profit,
        "total_staked": total_staked,
        "roi": roi,
        "by_league": leagues_df.to_dict("records"),
        "by_selection": selection_df.to_dict("records"),
    }


def get_recent_bets(limit: int = 20) -> pd.DataFrame:
    """Get most recent bets."""
    conn = get_db_connection()
    df = pd.read_sql(
        f"""
        SELECT bet_id, placed_at, league, match_date, home_team, away_team,
               selection, odds, stake, edge, status, profit
        FROM bets
        ORDER BY bet_id DESC
        LIMIT {limit}
        """,
        conn,
    )
    conn.close()
    return df


def get_performance_analytics() -> Dict[str, Any]:
    """
    Get advanced performance analytics including:
    - Max drawdown
    - Win/loss streaks
    - Model calibration (predicted vs actual win rates by edge bucket)
    - Expected value analysis
    """
    conn = get_db_connection()

    # Get all settled bets in chronological order
    bets = pd.read_sql(
        """
        SELECT bet_id, placed_at, match_date, odds, stake, edge, 
               model_prob, status, profit
        FROM bets
        WHERE status IN ('won', 'lost')
        ORDER BY settled_at, bet_id
        """,
        conn,
    )

    if len(bets) == 0:
        conn.close()
        return {"error": "No settled bets yet"}

    # Calculate running P/L and drawdown
    bets["cumulative_profit"] = bets["profit"].cumsum()
    bets["peak"] = bets["cumulative_profit"].cummax()
    bets["drawdown"] = bets["peak"] - bets["cumulative_profit"]
    max_drawdown = bets["drawdown"].max()
    max_drawdown_pct = (
        (max_drawdown / (1000 + bets["peak"].max())) * 100
        if bets["peak"].max() > 0
        else 0
    )

    # Streak analysis
    bets["is_win"] = (bets["status"] == "won").astype(int)
    streaks = []
    current_streak = 0
    current_type = None

    for won in bets["is_win"]:
        if current_type is None:
            current_type = won
            current_streak = 1
        elif won == current_type:
            current_streak += 1
        else:
            streaks.append((current_type, current_streak))
            current_type = won
            current_streak = 1
    streaks.append((current_type, current_streak))

    win_streaks = [s[1] for s in streaks if s[0] == 1]
    loss_streaks = [s[1] for s in streaks if s[0] == 0]

    max_win_streak = max(win_streaks) if win_streaks else 0
    max_loss_streak = max(loss_streaks) if loss_streaks else 0

    # Current streak
    current_streak_type = "win" if bets["is_win"].iloc[-1] == 1 else "loss"
    current_streak_len = streaks[-1][1] if streaks else 0

    # Model calibration - group by edge buckets
    bets["edge_bucket"] = pd.cut(
        bets["edge"],
        bins=[0, 0.04, 0.06, 0.08, 0.10, 0.15],
        labels=["3-4%", "4-6%", "6-8%", "8-10%", "10%+"],
    )
    calibration = (
        bets.groupby("edge_bucket", observed=True)
        .agg(
            bets=("bet_id", "count"),
            wins=("is_win", "sum"),
            avg_model_prob=("model_prob", "mean"),
            avg_odds=("odds", "mean"),
            profit=("profit", "sum"),
        )
        .reset_index()
    )
    calibration["actual_win_rate"] = calibration["wins"] / calibration["bets"]
    calibration["expected_win_rate"] = calibration["avg_model_prob"]

    # Expected value analysis
    total_ev = (
        bets["model_prob"] * (bets["odds"] - 1) - (1 - bets["model_prob"])
    ).sum()
    avg_ev_per_bet = total_ev / len(bets) if len(bets) > 0 else 0

    # Yield (profit per unit staked)
    total_staked = bets["stake"].sum()
    total_profit = bets["profit"].sum()
    yield_pct = (total_profit / total_staked * 100) if total_staked > 0 else 0

    # Average odds and edge
    avg_odds = bets["odds"].mean()
    avg_edge = bets["edge"].mean()

    conn.close()

    return {
        "total_bets": len(bets),
        "total_profit": total_profit,
        "total_staked": total_staked,
        "yield_pct": yield_pct,
        "avg_odds": avg_odds,
        "avg_edge": avg_edge * 100,  # as percentage
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "current_streak": f"{current_streak_len} {current_streak_type}s",
        "avg_ev_per_bet": avg_ev_per_bet,
        "calibration": calibration.to_dict("records"),
    }


def get_bankroll_history() -> pd.DataFrame:
    """Get bankroll balance over time for charting."""
    conn = get_db_connection()
    df = pd.read_sql(
        """
        SELECT timestamp, event_type, amount, balance_after, notes
        FROM bankroll
        ORDER BY id
        """,
        conn,
    )
    conn.close()
    return df


def get_daily_summary(date: str = None) -> Dict[str, Any]:
    """Get summary for a specific date."""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    conn = get_db_connection()

    # Bets placed on this date
    placed = pd.read_sql(
        """
        SELECT * FROM bets
        WHERE DATE(placed_at) = ?
        """,
        conn,
        params=(date,),
    )

    # Bets settled on this date
    settled = pd.read_sql(
        """
        SELECT * FROM bets
        WHERE DATE(settled_at) = ?
        """,
        conn,
        params=(date,),
    )

    conn.close()

    settled_profit = settled["profit"].sum() if len(settled) > 0 else 0
    settled_wins = (settled["status"] == "won").sum() if len(settled) > 0 else 0
    settled_total = len(settled)

    return {
        "date": date,
        "bets_placed": len(placed),
        "stake_placed": placed["stake"].sum() if len(placed) > 0 else 0,
        "bets_settled": settled_total,
        "settled_wins": settled_wins,
        "settled_profit": settled_profit,
        "win_rate": (settled_wins / settled_total * 100) if settled_total > 0 else 0,
    }


def export_bets_to_csv(filepath: str = None) -> str:
    """Export all bets to CSV file."""
    if filepath is None:
        filepath = str(
            PROJECT_ROOT
            / "predictions"
            / f"bets_export_{datetime.now().strftime('%Y%m%d')}.csv"
        )

    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM bets ORDER BY bet_id", conn)
    conn.close()

    df.to_csv(filepath, index=False)
    return filepath


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
    parser.add_argument(
        "--init-betting", action="store_true", help="Initialize betting tables"
    )
    parser.add_argument(
        "--init-bankroll", type=float, help="Initialize bankroll with amount"
    )

    args = parser.parse_args()

    if args.init:
        create_database_schema()
        create_betting_tables()
    elif args.init_betting:
        create_betting_tables()
    elif args.init_bankroll:
        create_betting_tables()
        init_bankroll(args.init_bankroll)
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
