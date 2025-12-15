# src/config.py
"""Central configuration for the sports trading project"""

from pathlib import Path

# ============================================================================
# Project Paths
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


# ============================================================================
# Leagues
# ============================================================================

LEAGUES = {
    "premier_league": {
        "code": "E0",
        "name": "Premier League",
        "country": "england",
    },
    "championship": {
        "code": "E1",
        "name": "Championship",
        "country": "england",
    },
}


# ============================================================================
# Data Source
# ============================================================================

FOOTBALL_DATA_BASE_URL = "https://www.football-data.co.uk/mmz4281/{}/{}.csv"

DEFAULT_START_YEAR = 2017
DEFAULT_END_YEAR = 2030

SEASON_START_MONTH = 8


# ============================================================================
# Column Mappings
# ============================================================================

COLUMN_MAPPING = {
    # Match identification
    "Date": "match_date",
    "Time": "match_time",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "Referee": "referee",
    # Results
    "FTHG": "full_time_home_goals",
    "FTAG": "full_time_away_goals",
    "FTR": "full_time_result",
    "HTHG": "half_time_home_goals",
    "HTAG": "half_time_away_goals",
    "HTR": "half_time_result",
    # Match statistics
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HY": "home_yellow_cards",
    "AY": "away_yellow_cards",
    "HR": "home_red_cards",
    "AR": "away_red_cards",
    "HC": "home_corners",
    "AC": "away_corners",
    # Pinnacle odds
    "PSH": "pinnacle_home_odds",
    "PSD": "pinnacle_draw_odds",
    "PSA": "pinnacle_away_odds",
    "P>2.5": "pinnacle_over_2_5_odds",
    "P<2.5": "pinnacle_under_2_5_odds",
    # Metadata
    "season": "season",
    "league": "league",
}

ESSENTIAL_COLUMNS = list(COLUMN_MAPPING.keys())
