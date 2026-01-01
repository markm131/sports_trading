# src/trading/scanner.py
"""
Multi-league value bet scanner with paper trading.

All bet tracking and bankroll management uses SQLite database.
Supports continuous monitoring with configurable bet placement timing.
Designed to run locally, on AWS Lambda, or as a daemon on EC2.

Usage:
    python -m src.trading.scanner                    # Single scan
    python -m src.trading.scanner --daemon           # Run continuously
    python -m src.trading.scanner --daemon --interval 15  # Every 15 mins
    python -m src.trading.scanner --results          # Settle yesterday's bets
    python -m src.trading.scanner --status           # Show bankroll status
    python -m src.trading.scanner --reset            # Reset paper trading
"""

import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

from src.config import PROJECT_ROOT
from src.data.db_writer import (
    create_betting_tables,
    export_bets_to_csv,
    get_current_bankroll,
    get_db_connection,
    get_performance_analytics,
    get_recent_bets,
    init_bankroll,
)
from src.models.poisson import PoissonModel
from src.trading.edge import remove_vig
from src.trading.kelly import kelly_stake

# Betfair imports
try:
    from betfairlightweight import APIClient, filters

    BETFAIR_AVAILABLE = True
except ImportError:
    BETFAIR_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

INITIAL_BANKROLL = 1000.0
KELLY_FRACTION = 0.10  # 10% Kelly - conservative
MIN_STAKE = 2.0
MAX_STAKE_PCT = 0.02  # 2% max per bet

# Edge thresholds
DEFAULT_MIN_EDGE = 0.03
DEFAULT_MAX_EDGE = 0.12

# Bet timing configuration
BET_TIMING = {
    # When to place bets (minutes before kickoff)
    # We track opportunities but only "place" when within this window
    "place_bet_mins_before": 60,  # Place bet 60 mins before kickoff
    "min_mins_before": 5,  # Don't bet if less than 5 mins to kickoff
    # How long value must persist before betting (reduces noise)
    "min_consecutive_scans": 1,  # Bet immediately if value found (set to 2+ to wait)
}

# Scanning configuration
SCAN_CONFIG = {
    "interval_minutes": 15,  # How often to scan in daemon mode
    "days_ahead": 2,  # How far ahead to look for fixtures
    "quiet_hours_start": 1,  # Don't scan between 1am and 6am (UTC)
    "quiet_hours_end": 6,
}

# Logging
LOG_FILE = PROJECT_ROOT / "logs" / "scanner.log"

# Betfair competition IDs for all leagues
BETFAIR_COMPETITIONS = {
    # England
    "Premier League": 10932509,
    "Championship": 7129730,
    # Germany
    "Bundesliga": 59,
    "Bundesliga 2": 59197,
    # Spain
    "La Liga": 117,
    "La Liga 2": 12204313,
    # Italy
    "Serie A": 81,
    "Serie B": 12199689,
    # France
    "Ligue 1": 55,
    "Ligue 2": 60,
    # Netherlands
    "Eredivisie": 9404054,
    # Belgium
    "Jupiler League": 89979,
    # Portugal
    "Primeira Liga": 99,
    # Scotland
    "Scottish Premiership": 105,
}

# Team name mappings (Betfair -> Our DB format)
BETFAIR_TEAM_MAP = {
    # England - Premier League
    "Man Utd": "Man United",
    "Manchester United": "Man United",
    "Man City": "Man City",
    "Manchester City": "Man City",
    "Newcastle": "Newcastle",
    "Newcastle United": "Newcastle",
    "Spurs": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "Nottm Forest": "Nott'm Forest",
    "Nottingham Forest": "Nott'm Forest",
    "Wolves": "Wolves",
    "Wolverhampton": "Wolves",
    "Wolverhampton Wanderers": "Wolves",
    "Sheffield Utd": "Sheffield United",
    "Brighton and Hove Albion": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "West Ham United": "West Ham",
    "AFC Bournemouth": "Bournemouth",
    "Ipswich Town": "Ipswich",
    "Leicester City": "Leicester",
    "Luton Town": "Luton",
    # England - Championship
    "Leeds United": "Leeds",
    "Sheffield Wed": "Sheffield Weds",
    "Sheffield Wednesday": "Sheffield Weds",
    "West Brom": "West Brom",
    "West Bromwich Albion": "West Brom",
    "Queens Park Rangers": "QPR",
    "Stoke City": "Stoke",
    "Norwich City": "Norwich",
    "Hull City": "Hull",
    "Bristol City": "Bristol City",
    "Cardiff City": "Cardiff",
    "Swansea City": "Swansea",
    "Preston North End": "Preston",
    "Coventry City": "Coventry",
    "Plymouth Argyle": "Plymouth",
    # Germany - Bundesliga
    "Bayern Munich": "Bayern Munich",
    "Bayer Leverkusen": "Leverkusen",
    "Borussia Dortmund": "Dortmund",
    "RB Leipzig": "RB Leipzig",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "VfB Stuttgart": "Stuttgart",
    "Borussia M'gladbach": "M'gladbach",
    "Borussia Monchengladbach": "M'gladbach",
    "Werder Bremen": "Werder Bremen",
    "SC Freiburg": "Freiburg",
    "VfL Wolfsburg": "Wolfsburg",
    "TSG Hoffenheim": "Hoffenheim",
    "1. FC Union Berlin": "Union Berlin",
    "FC Augsburg": "Augsburg",
    "1. FC Heidenheim": "Heidenheim",
    "FSV Mainz 05": "Mainz",
    "Mainz 05": "Mainz",
    "VfL Bochum": "Bochum",
    "FC St Pauli": "St Pauli",
    "Holstein Kiel": "Holstein Kiel",
    # Germany - Bundesliga 2
    "Hamburger SV": "Hamburg",
    "Fortuna Dusseldorf": "Fortuna Dusseldorf",
    "FC Koln": "FC Koln",
    "1. FC Koln": "FC Koln",
    "Greuther Furth": "Greuther Furth",
    "FC Nurnberg": "Nurnberg",
    "1. FC Nurnberg": "Nurnberg",
    "Hertha BSC": "Hertha Berlin",
    "Schalke 04": "Schalke 04",
    "Hannover 96": "Hannover",
    "1. FC Kaiserslautern": "Kaiserslautern",
    "SC Paderborn": "Paderborn",
    "Karlsruher SC": "Karlsruhe",
    "1. FC Magdeburg": "Magdeburg",
    "Eintracht Braunschweig": "Braunschweig",
    "SV Darmstadt 98": "Darmstadt",
    "SSV Ulm 1846": "Ulm",
    "Preussen Munster": "Munster",
    "SV Elversberg": "Elversberg",
    "Jahn Regensburg": "Regensburg",
    # Spain - La Liga
    "Real Madrid": "Real Madrid",
    "FC Barcelona": "Barcelona",
    "Atletico Madrid": "Ath Madrid",
    "Athletic Bilbao": "Ath Bilbao",
    "Real Sociedad": "Sociedad",
    "Real Betis": "Betis",
    "Villarreal": "Villarreal",
    "Sevilla FC": "Sevilla",
    "Valencia CF": "Valencia",
    "Celta Vigo": "Celta",
    "RCD Mallorca": "Mallorca",
    "Girona FC": "Girona",
    "Rayo Vallecano": "Vallecano",
    "Osasuna": "Osasuna",
    "Getafe CF": "Getafe",
    "UD Las Palmas": "Las Palmas",
    "Deportivo Alaves": "Alaves",
    "RCD Espanyol": "Espanol",
    "Real Valladolid": "Valladolid",
    "CD Leganes": "Leganes",
    # Italy - Serie A
    "Inter Milan": "Inter",
    "AC Milan": "Milan",
    "Juventus": "Juventus",
    "SSC Napoli": "Napoli",
    "AS Roma": "Roma",
    "SS Lazio": "Lazio",
    "Atalanta": "Atalanta",
    "ACF Fiorentina": "Fiorentina",
    "Bologna FC": "Bologna",
    "Torino FC": "Torino",
    "Udinese": "Udinese",
    "Genoa CFC": "Genoa",
    "Cagliari": "Cagliari",
    "Hellas Verona": "Verona",
    "Empoli FC": "Empoli",
    "Parma Calcio": "Parma",
    "Como 1907": "Como",
    "Venezia FC": "Venezia",
    "US Lecce": "Lecce",
    "Monza": "Monza",
    # France - Ligue 1
    "Paris Saint-Germain": "Paris SG",
    "Paris St Germain": "Paris SG",
    "AS Monaco": "Monaco",
    "Olympique Marseille": "Marseille",
    "Olympique Lyon": "Lyon",
    "Lille OSC": "Lille",
    "OGC Nice": "Nice",
    "RC Lens": "Lens",
    "Stade Rennais": "Rennes",
    "Stade Brestois": "Brest",
    "RC Strasbourg": "Strasbourg",
    "Toulouse FC": "Toulouse",
    "Montpellier HSC": "Montpellier",
    "FC Nantes": "Nantes",
    "Stade Reims": "Reims",
    "AJ Auxerre": "Auxerre",
    "Angers SCO": "Angers",
    "Le Havre AC": "Le Havre",
    "AS Saint-Etienne": "St Etienne",
    # Netherlands - Eredivisie
    "Ajax": "Ajax",
    "PSV Eindhoven": "PSV",
    "Feyenoord": "Feyenoord",
    "AZ Alkmaar": "AZ",
    "FC Twente": "Twente",
    "FC Utrecht": "Utrecht",
    "Sparta Rotterdam": "Sparta Rotterdam",
    "Go Ahead Eagles": "Go Ahead Eagles",
    "SC Heerenveen": "Heerenveen",
    "NEC Nijmegen": "NEC Nijmegen",
    "Fortuna Sittard": "For Sittard",
    "PEC Zwolle": "Zwolle",
    "Willem II": "Willem II",
    "Heracles Almelo": "Heracles",
    "RKC Waalwijk": "Waalwijk",
    "FC Groningen": "Groningen",
    "NAC Breda": "NAC Breda",
    "Almere City FC": "Almere City",
    # Belgium - Jupiler League
    "Club Brugge": "Club Brugge",
    "RSC Anderlecht": "Anderlecht",
    "KRC Genk": "Genk",
    "Royal Antwerp": "Antwerp",
    "Union Saint-Gilloise": "St. Gilloise",
    "KAA Gent": "Gent",
    "Cercle Brugge": "Cercle Brugge",
    "Standard Liege": "Standard",
    "OH Leuven": "Oud-Heverlee Leuven",
    "KV Mechelen": "Mechelen",
    "Charleroi": "Charleroi",
    "Westerlo": "Westerlo",
    "STVV": "St Truiden",
    "Sint-Truidense VV": "St Truiden",
    "KV Kortrijk": "Kortrijk",
    "FCV Dender EH": "Dender",
    "Beerschot VA": "Beerschot",
    # Portugal - Primeira Liga
    "SL Benfica": "Benfica",
    "FC Porto": "Porto",
    "Sporting CP": "Sporting CP",
    "Sporting Lisbon": "Sporting CP",
    "SC Braga": "Sp Braga",
    "Vitoria SC": "Guimaraes",
    "Vitoria Guimaraes": "Guimaraes",
    "Rio Ave FC": "Rio Ave",
    "Famalicao": "Famalicao",
    "Santa Clara": "Santa Clara",
    "Moreirense FC": "Moreirense",
    "Casa Pia AC": "Casa Pia",
    "Gil Vicente": "Gil Vicente",
    "Boavista FC": "Boavista",
    "Arouca": "Arouca",
    "Estoril Praia": "Estoril",
    "Estrela Amadora": "Estrela",
    "AVS": "AVS",
    "Farense": "Farense",
    "Nacional": "Nacional",
    # Scotland - Premiership
    "Celtic": "Celtic",
    "Rangers": "Rangers",
    "Aberdeen": "Aberdeen",
    "Hearts": "Hearts",
    "Heart of Midlothian": "Hearts",
    "Hibernian": "Hibernian",
    "Dundee": "Dundee",
    "Dundee United": "Dundee Utd",
    "Motherwell": "Motherwell",
    "Kilmarnock": "Kilmarnock",
    "St Mirren": "St Mirren",
    "St Johnstone": "St Johnstone",
    "Ross County": "Ross County",
}


def normalize_team(name: str) -> str:
    """Normalize Betfair team name to our database format."""
    return BETFAIR_TEAM_MAP.get(name, name)


# =============================================================================
# Logging Setup
# =============================================================================


def setup_logging(verbose: bool = False):
    """Configure logging for scanner."""
    LOG_FILE.parent.mkdir(exist_ok=True)

    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout),
        ],
    )

    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)


# =============================================================================
# Database-backed Bankroll Management
# =============================================================================


class DatabaseBankroll:
    """Manages paper trading bankroll using SQLite database."""

    def __init__(self):
        # Ensure tables exist
        create_betting_tables()
        self._ensure_initialized()

    def _ensure_initialized(self):
        """Ensure bankroll is initialized."""
        if self.bankroll == 0:
            init_bankroll(INITIAL_BANKROLL)

    @property
    def bankroll(self) -> float:
        """Get current bankroll from database."""
        return get_current_bankroll()

    @property
    def initial(self) -> float:
        """Get initial bankroll amount."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT amount FROM bankroll WHERE event_type = 'initial' ORDER BY id LIMIT 1"
        )
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else INITIAL_BANKROLL

    @property
    def started(self) -> str:
        """Get start date of paper trading."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp FROM bankroll ORDER BY id LIMIT 1")
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else datetime.now().isoformat()

    def place_bet(self, stake: float, bet_id: int = None):
        """Deduct stake from bankroll."""
        conn = get_db_connection()
        cursor = conn.cursor()
        new_balance = self.bankroll - stake
        cursor.execute(
            """
            INSERT INTO bankroll (event_type, amount, balance_after, bet_id, notes)
            VALUES ('bet_placed', ?, ?, ?, 'Stake deducted')
            """,
            (-stake, new_balance, bet_id),
        )
        conn.commit()
        conn.close()

    def settle_bet(self, stake: float, odds: float, won: bool, bet_id: int = None):
        """Settle a bet - add returns if won."""
        conn = get_db_connection()
        cursor = conn.cursor()
        if won:
            winnings = stake * odds
            new_balance = self.bankroll + winnings
            notes = f"Bet won @ {odds:.2f}"
        else:
            new_balance = self.bankroll  # Already deducted at placement
            winnings = 0
            notes = "Bet lost"

        cursor.execute(
            """
            INSERT INTO bankroll (event_type, amount, balance_after, bet_id, notes)
            VALUES ('bet_settled', ?, ?, ?, ?)
            """,
            (winnings, new_balance, bet_id, notes),
        )
        conn.commit()
        conn.close()

    def status(self) -> Dict:
        """Return current status."""
        current = self.bankroll
        initial = self.initial
        return {
            "bankroll": current,
            "initial": initial,
            "profit": current - initial,
            "roi_pct": ((current - initial) / initial * 100) if initial > 0 else 0,
            "started": self.started,
        }


# =============================================================================
# Database-backed Opportunity Tracker
# =============================================================================


class DatabaseOpportunityTracker:
    """
    Tracks betting opportunities in SQLite database.

    Handles:
    - Avoiding duplicate bets on same match/selection
    - Tracking how long value has persisted
    - Deciding when to place bets based on timing rules
    """

    def _make_key(self, market_id: str, selection: str) -> str:
        """Create unique key for match/selection combo."""
        return f"{market_id}_{selection}"

    def _is_bet_placed(self, market_id: str, selection: str) -> bool:
        """Check if bet was already placed for this opportunity."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT bet_placed FROM opportunities WHERE market_id = ? AND selection = ?",
            (market_id, selection),
        )
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else False

    def update_opportunity(
        self,
        market_id: str,
        selection: str,
        edge: float,
        odds: float,
        kickoff: datetime,
        bet_data: Dict,
    ) -> Optional[Dict]:
        """
        Update an opportunity and decide if we should bet now.

        Returns bet_data if we should place the bet, None otherwise.
        """
        # Already placed this bet
        if self._is_bet_placed(market_id, selection):
            return None

        now = datetime.utcnow()
        mins_to_kickoff = (kickoff - now).total_seconds() / 60

        # Too close to kickoff
        if mins_to_kickoff < BET_TIMING["min_mins_before"]:
            return None

        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if opportunity exists
        cursor.execute(
            "SELECT opp_id, scans_with_value, best_edge FROM opportunities WHERE market_id = ? AND selection = ?",
            (market_id, selection),
        )
        existing = cursor.fetchone()

        if existing:
            opp_id, scans, best_edge = existing
            new_scans = scans + 1
            new_best_edge = max(edge, best_edge)
            new_best_odds = odds if edge >= best_edge else None

            if new_best_odds:
                cursor.execute(
                    """
                    UPDATE opportunities
                    SET last_seen = ?, scans_with_value = ?, current_edge = ?, current_odds = ?,
                        best_edge = ?, best_odds = ?
                    WHERE opp_id = ?
                    """,
                    (
                        now.isoformat(),
                        new_scans,
                        edge,
                        odds,
                        new_best_edge,
                        new_best_odds,
                        opp_id,
                    ),
                )
            else:
                cursor.execute(
                    """
                    UPDATE opportunities
                    SET last_seen = ?, scans_with_value = ?, current_edge = ?, current_odds = ?
                    WHERE opp_id = ?
                    """,
                    (now.isoformat(), new_scans, edge, odds, opp_id),
                )
            scans_with_value = new_scans
        else:
            cursor.execute(
                """
                INSERT INTO opportunities
                (market_id, selection, first_seen, last_seen, scans_with_value, kickoff,
                 league, home_team, away_team, current_edge, current_odds, best_edge, best_odds)
                VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    market_id,
                    selection,
                    now.isoformat(),
                    now.isoformat(),
                    kickoff.isoformat(),
                    bet_data.get("league", ""),
                    bet_data.get("home_team", ""),
                    bet_data.get("away_team", ""),
                    edge,
                    odds,
                    edge,
                    odds,
                ),
            )
            scans_with_value = 1

        conn.commit()

        # Decide if we should bet now
        should_bet = False

        if scans_with_value >= BET_TIMING["min_consecutive_scans"]:
            if mins_to_kickoff <= BET_TIMING["place_bet_mins_before"]:
                should_bet = True
                logger.info(
                    f"Placing bet: {market_id}_{selection} | {mins_to_kickoff:.0f} mins to KO | "
                    f"Edge {edge:.1%} | Seen {scans_with_value} times"
                )
                # Mark as bet placed
                cursor.execute(
                    "UPDATE opportunities SET bet_placed = 1 WHERE market_id = ? AND selection = ?",
                    (market_id, selection),
                )
                conn.commit()

        conn.close()

        if should_bet:
            return bet_data

        return None

    def cleanup_old(self):
        """Remove opportunities for matches that have started."""
        now = datetime.utcnow()
        conn = get_db_connection()
        cursor = conn.cursor()

        # Delete opportunities where kickoff has passed
        cursor.execute(
            "DELETE FROM opportunities WHERE kickoff < ?",
            (now.isoformat(),),
        )
        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        if deleted:
            logger.debug(f"Cleaned up {deleted} old opportunities")

    def get_pending_opportunities(self) -> List[Dict]:
        """Get opportunities we're tracking but haven't bet on."""
        conn = get_db_connection()
        df = pd.read_sql(
            "SELECT * FROM opportunities WHERE bet_placed = 0",
            conn,
        )
        conn.close()
        return df.to_dict("records")


# =============================================================================
# Betfair Client
# =============================================================================


class BetfairClient:
    """Simple Betfair API wrapper."""

    def __init__(self):
        if not BETFAIR_AVAILABLE:
            raise ImportError("betfairlightweight not installed")

        load_dotenv()

        self.username = os.getenv("BETFAIR_USERNAME")
        self.password = os.getenv("BETFAIR_PASSWORD")
        self.app_key = os.getenv("BETFAIR_APP_KEY")

        certs_dir = PROJECT_ROOT / "certs"

        self.client = APIClient(
            self.username,
            self.password,
            app_key=self.app_key,
            certs=str(certs_dir),
        )

        self._logged_in = False

    def login(self):
        if not self._logged_in:
            self.client.login()
            self._logged_in = True

    def logout(self):
        if self._logged_in:
            self.client.logout()
            self._logged_in = False

    def get_football_matches(
        self, competition_id: int, days_ahead: int = 2
    ) -> List[Dict]:
        """Get matches for a competition within the next N days."""
        self.login()

        now = datetime.utcnow()
        end_time = now + timedelta(days=days_ahead)

        time_range = filters.time_range(
            from_=now.isoformat(),
            to=end_time.isoformat(),
        )

        market_filter = filters.market_filter(
            event_type_ids=["1"],  # Football
            competition_ids=[str(competition_id)],
            market_type_codes=["MATCH_ODDS"],
            market_start_time=time_range,
        )

        try:
            catalogues = self.client.betting.list_market_catalogue(
                filter=market_filter,
                market_projection=["EVENT", "RUNNER_METADATA", "MARKET_START_TIME"],
                max_results=100,
            )
        except Exception as e:
            logger.error(f"Error fetching markets for comp {competition_id}: {e}")
            return []

        matches = []

        for cat in catalogues:
            try:
                books = self.client.betting.list_market_book(
                    market_ids=[cat.market_id],
                    price_projection=filters.price_projection(
                        price_data=["EX_BEST_OFFERS"]
                    ),
                )

                if not books:
                    continue

                book = books[0]

                # Parse runners
                odds = {}
                for runner in book.runners:
                    name = None
                    for r in cat.runners:
                        if r.selection_id == runner.selection_id:
                            name = r.runner_name
                            break

                    if runner.ex.available_to_back:
                        back_price = runner.ex.available_to_back[0].price
                    else:
                        back_price = None

                    if name == "The Draw":
                        odds["draw"] = back_price
                    elif name:
                        if "home" not in odds:
                            odds["home"] = back_price
                            odds["home_team"] = normalize_team(name)
                        else:
                            odds["away"] = back_price
                            odds["away_team"] = normalize_team(name)

                if all(
                    k in odds
                    for k in ["home", "draw", "away", "home_team", "away_team"]
                ):
                    matches.append(
                        {
                            "market_id": cat.market_id,
                            "event_name": cat.event.name,
                            "start_time": cat.market_start_time,
                            "home_team": odds["home_team"],
                            "away_team": odds["away_team"],
                            "home_odds": odds["home"],
                            "draw_odds": odds["draw"],
                            "away_odds": odds["away"],
                        }
                    )
            except Exception:
                continue

        return matches


# =============================================================================
# Scanner
# =============================================================================


class Scanner:
    """Multi-league value bet scanner with database-backed paper trading."""

    def __init__(self):
        self.models: Dict[str, PoissonModel] = {}
        self.bankroll = DatabaseBankroll()
        self.tracker = DatabaseOpportunityTracker()

    def load_model(self, league: str) -> Optional[PoissonModel]:
        """Load or fit model for a league."""
        if league in self.models:
            return self.models[league]

        conn = get_db_connection()
        df = pd.read_sql(
            """
            SELECT home_team, away_team,
                   full_time_home_goals, full_time_away_goals,
                   home_shots_on_target, away_shots_on_target,
                   match_date
            FROM matches
            WHERE league = ?
            ORDER BY match_date
        """,
            conn,
            params=(league,),
        )
        conn.close()

        if len(df) < 300:
            logger.debug(f"Skipping {league}: only {len(df)} matches (need 300+)")
            return None

        try:
            model = PoissonModel(
                half_life_days=180,
                form_weight=0.3,
            ).fit(df)
            self.models[league] = model
            return model
        except Exception as e:
            logger.error(f"Error fitting model for {league}: {e}")
            return None

    def calculate_stake(self, model_prob: float, odds: float) -> float:
        """Calculate Kelly stake for a bet."""
        stake = kelly_stake(
            prob=model_prob,
            odds=odds,
            bankroll=self.bankroll.bankroll,
            fraction=KELLY_FRACTION,
            max_stake_pct=MAX_STAKE_PCT,
        )
        return round(stake, 2) if stake >= MIN_STAKE else 0.0

    def find_value_bets(self, matches: List[Dict], league: str) -> List[Dict]:
        """
        Find value bets and decide which to place based on timing.

        Returns list of bets that should be placed NOW.
        """
        model = self.load_model(league)
        if model is None:
            return []

        bets_to_place = []

        for match in matches:
            home = match["home_team"]
            away = match["away_team"]

            # Check teams are known
            if home not in model.teams or away not in model.teams:
                continue

            # Get predictions
            try:
                pred = model.predict_fixture(home, away)
            except Exception:
                continue

            # Get market odds
            odds = {
                "home": match["home_odds"],
                "draw": match["draw_odds"],
                "away": match["away_odds"],
            }

            if not all(o and o > 1.0 for o in odds.values()):
                continue

            # Remove vig for fair comparison
            fair = remove_vig(odds)

            # Check each selection
            for sel, model_key in [
                ("home", "home_win"),
                ("draw", "draw"),
                ("away", "away_win"),
            ]:
                model_prob = pred[model_key]
                market_prob = fair[sel]
                edge = model_prob - market_prob

                if DEFAULT_MIN_EDGE <= edge <= DEFAULT_MAX_EDGE:
                    stake = self.calculate_stake(model_prob, odds[sel])
                    if stake > 0:
                        bet_data = {
                            "scan_time": datetime.now().isoformat(),
                            "league": league,
                            "match_time": match["start_time"].isoformat(),
                            "match_date": match["start_time"].strftime("%Y-%m-%d"),
                            "home_team": home,
                            "away_team": away,
                            "selection": sel,
                            "model_prob": round(model_prob, 4),
                            "market_prob": round(market_prob, 4),
                            "edge": round(edge, 4),
                            "odds": odds[sel],
                            "stake": stake,
                            "potential_profit": round(stake * (odds[sel] - 1), 2),
                            "market_id": match["market_id"],
                            "status": "pending",
                            "result": None,
                            "profit": None,
                        }

                        # Check if we should place this bet now
                        result = self.tracker.update_opportunity(
                            market_id=match["market_id"],
                            selection=sel,
                            edge=edge,
                            odds=odds[sel],
                            kickoff=match["start_time"],
                            bet_data=bet_data,
                        )

                        if result:
                            bets_to_place.append(result)

        return bets_to_place

    def place_paper_bets(self, bets: List[Dict]):
        """Record paper bets and deduct stakes from bankroll."""
        if not bets:
            return

        for bet in bets:
            # Insert bet into database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO bets
                (league, match_date, match_time, home_team, away_team, market_id,
                 selection, odds, stake, model_prob, market_prob, edge, is_paper)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    bet["league"],
                    bet["match_date"],
                    bet.get("match_time"),
                    bet["home_team"],
                    bet["away_team"],
                    bet.get("market_id"),
                    bet["selection"],
                    bet["odds"],
                    bet["stake"],
                    bet["model_prob"],
                    bet["market_prob"],
                    bet["edge"],
                ),
            )
            bet_id = cursor.lastrowid
            conn.commit()
            conn.close()

            # Deduct stake from bankroll
            self.bankroll.place_bet(bet["stake"], bet_id=bet_id)

        logger.info(
            f"Placed {len(bets)} paper bets | Bankroll: £{self.bankroll.bankroll:.2f}"
        )

    def settle_bets(self, date: str = None):
        """Settle bets for a given date using actual results."""
        if date is None:
            date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        conn = get_db_connection()

        # Get pending bets from database
        pending_df = pd.read_sql(
            """
            SELECT bet_id, league, match_date, home_team, away_team, selection, odds, stake
            FROM bets
            WHERE match_date = ? AND status = 'pending'
            """,
            conn,
            params=(date,),
        )

        if len(pending_df) == 0:
            print(f"No pending bets for {date}")
            conn.close()
            return

        print(f"\nSettling {len(pending_df)} bets from {date}")
        print("=" * 70)

        cursor = conn.cursor()

        for _, bet in pending_df.iterrows():
            # Find result AND closing odds (Pinnacle)
            result_df = pd.read_sql(
                """
                SELECT full_time_result,
                       pinnacle_home_odds, pinnacle_draw_odds, pinnacle_away_odds
                FROM matches
                WHERE home_team = ?
                  AND away_team = ?
                  AND match_date = ?
            """,
                conn,
                params=(bet["home_team"], bet["away_team"], date),
            )

            if len(result_df) == 0:
                print(f"  {bet['home_team']} vs {bet['away_team']} - Result not found")
                continue

            actual = result_df["full_time_result"].iloc[0]
            sel_map = {"home": "H", "draw": "D", "away": "A"}

            # Get closing odds for CLV calculation
            closing_odds_map = {
                "home": result_df["pinnacle_home_odds"].iloc[0],
                "draw": result_df["pinnacle_draw_odds"].iloc[0],
                "away": result_df["pinnacle_away_odds"].iloc[0],
            }
            closing_odds = closing_odds_map.get(bet["selection"])

            # Calculate CLV (Closing Line Value)
            clv = None
            clv_str = ""
            if closing_odds and closing_odds > 1:
                # CLV = how much better our odds were vs closing
                # Positive = we got better odds (good!)
                clv = (bet["odds"] - closing_odds) / closing_odds
                clv_str = f" | CLV: {clv * 100:+.1f}%"

            won = actual == sel_map[bet["selection"]]
            profit = (bet["stake"] * (bet["odds"] - 1)) if won else -bet["stake"]
            status = "won" if won else "lost"

            # Update bet record in database with CLV
            cursor.execute(
                """
                UPDATE bets
                SET status = ?, settled_at = ?, actual_result = ?, profit = ?,
                    closing_odds = ?, clv = ?
                WHERE bet_id = ?
                """,
                (
                    status,
                    datetime.now().isoformat(),
                    actual,
                    profit,
                    closing_odds,
                    clv,
                    bet["bet_id"],
                ),
            )

            # Update bankroll
            self.bankroll.settle_bet(
                bet["stake"], bet["odds"], won, bet_id=bet["bet_id"]
            )

            status_str = "✓ WIN" if won else "✗ LOSS"
            print(
                f"  {bet['league']}: {bet['home_team']} vs {bet['away_team']} | "
                f"{bet['selection'].upper()} @ {bet['odds']:.2f} | "
                f"£{bet['stake']:.2f} | {status_str} (£{profit:+.2f}){clv_str}"
            )

        conn.commit()
        conn.close()

        # Show summary with CLV stats
        self._print_summary()

    def _print_summary(self):
        """Print betting summary from database including CLV."""
        conn = get_db_connection()
        settled = pd.read_sql(
            "SELECT * FROM bets WHERE status IN ('won', 'lost')",
            conn,
        )
        conn.close()

        if len(settled) > 0:
            total_profit = settled["profit"].sum()
            wins = (settled["status"] == "won").sum()
            total = len(settled)

            print("\n" + "=" * 70)
            print("PAPER TRADING SUMMARY")
            print("=" * 70)
            print(f"Total bets: {total}")
            print(f"Win rate: {wins}/{total} ({wins / total * 100:.1f}%)")
            print(f"Total profit: £{total_profit:+.2f}")
            print(f"Current bankroll: £{self.bankroll.bankroll:.2f}")
            roi = (
                (self.bankroll.bankroll - self.bankroll.initial)
                / self.bankroll.initial
                * 100
            )
            print(f"ROI: {roi:+.1f}%")

            # CLV stats (key indicator of edge)
            if "clv" in settled.columns and settled["clv"].notna().any():
                avg_clv = settled["clv"].mean() * 100
                positive_clv = (settled["clv"] > 0).sum()
                print("\nClosing Line Value (vs Pinnacle):")
                print(f"  Average CLV: {avg_clv:+.2f}%")
                print(
                    f"  Positive CLV: {positive_clv}/{total} ({positive_clv / total * 100:.0f}%)"
                )
                if avg_clv > 0:
                    print("  ✓ Getting better prices than closing - good sign!")

    def show_status(self):
        """Show current paper trading status from database."""
        status = self.bankroll.status()

        print("\n" + "=" * 70)
        print("PAPER TRADING STATUS")
        print("=" * 70)
        print(f"Started: {status['started'][:10]}")
        print(f"Initial bankroll: £{status['initial']:.2f}")
        print(f"Current bankroll: £{status['bankroll']:.2f}")
        print(f"Profit/Loss: £{status['profit']:+.2f}")
        print(f"ROI: {status['roi_pct']:+.1f}%")

        # Pending opportunities
        pending_opps = self.tracker.get_pending_opportunities()
        if pending_opps:
            print(f"\nTracking {len(pending_opps)} opportunities (not yet bet)")

        # Get bets from database
        conn = get_db_connection()
        bets_df = pd.read_sql("SELECT * FROM bets", conn)
        conn.close()

        if len(bets_df) > 0:
            pending = bets_df[bets_df["status"] == "pending"]
            settled = bets_df[bets_df["status"].isin(["won", "lost"])]

            print(f"\nTotal bets placed: {len(bets_df)}")
            print(f"Pending settlement: {len(pending)}")
            print(f"Settled: {len(settled)}")

            if len(settled) > 0:
                wins = (settled["status"] == "won").sum()
                print(
                    f"Win rate: {wins}/{len(settled)} ({wins / len(settled) * 100:.1f}%)"
                )
                print(f"Total profit: £{settled['profit'].sum():+.2f}")

                # By league
                print("\nBy league:")
                for league in sorted(settled["league"].unique()):
                    lg_bets = settled[settled["league"] == league]
                    lg_profit = lg_bets["profit"].sum()
                    lg_wins = (lg_bets["status"] == "won").sum()
                    print(
                        f"  {league}: {len(lg_bets)} bets, "
                        f"{lg_wins} wins, £{lg_profit:+.2f}"
                    )


# =============================================================================
# Main Functions
# =============================================================================


def is_quiet_hours() -> bool:
    """Check if we're in quiet hours (no scanning)."""
    hour = datetime.utcnow().hour
    return SCAN_CONFIG["quiet_hours_start"] <= hour < SCAN_CONFIG["quiet_hours_end"]


def scan_all_leagues(days_ahead: int = None) -> List[Dict]:
    """Scan all configured leagues for value bets."""
    if days_ahead is None:
        days_ahead = SCAN_CONFIG["days_ahead"]

    logger.info("=" * 70)
    logger.info(f"SCAN STARTED - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"Looking {days_ahead} day(s) ahead")
    logger.info("=" * 70)

    scanner = Scanner()
    scanner.tracker.cleanup_old()

    client = BetfairClient()

    all_bets = []

    try:
        for league_name, comp_id in BETFAIR_COMPETITIONS.items():
            matches = client.get_football_matches(comp_id, days_ahead)

            if not matches:
                continue

            logger.debug(f"{league_name}: {len(matches)} fixtures")

            bets_to_place = scanner.find_value_bets(matches, league_name)

            if bets_to_place:
                for bet in bets_to_place:
                    logger.info(
                        f"  BET: {bet['home_team']} vs {bet['away_team']} | "
                        f"{bet['selection'].upper()} @ {bet['odds']:.2f} | "
                        f"Edge {bet['edge']:.1%} | £{bet['stake']:.2f}"
                    )
                all_bets.extend(bets_to_place)

    finally:
        client.logout()

    # Place paper bets
    if all_bets:
        scanner.place_paper_bets(all_bets)

    # Summary
    pending_opps = len(scanner.tracker.get_pending_opportunities())
    logger.info(
        f"Scan complete: {len(all_bets)} bets placed, {pending_opps} opportunities tracking"
    )

    return all_bets


def run_daemon(interval_minutes: int = None):
    """Run scanner continuously at intervals."""
    if interval_minutes is None:
        interval_minutes = SCAN_CONFIG["interval_minutes"]

    logger.info("=" * 70)
    logger.info("SCANNER DAEMON STARTED")
    logger.info(f"Scanning every {interval_minutes} minutes")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 70)

    running = True

    def signal_handler(sig, frame):
        nonlocal running
        logger.info("Shutdown signal received...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    while running:
        try:
            if is_quiet_hours():
                logger.debug("Quiet hours - skipping scan")
            else:
                scan_all_leagues()

        except Exception as e:
            logger.error(f"Scan error: {e}")

        # Sleep in small increments so we can respond to signals
        for _ in range(interval_minutes * 60):
            if not running:
                break
            time.sleep(1)

    logger.info("Scanner daemon stopped")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-league value bet scanner")
    parser.add_argument("--daemon", action="store_true", help="Run continuously")
    parser.add_argument(
        "--interval", type=int, help="Scan interval in minutes (default: 15)"
    )
    parser.add_argument(
        "--results", action="store_true", help="Settle yesterday's bets"
    )
    parser.add_argument("--date", help="Date to settle (YYYY-MM-DD)")
    parser.add_argument("--status", action="store_true", help="Show bankroll status")
    parser.add_argument("--bets", action="store_true", help="Show recent bets")
    parser.add_argument(
        "--analytics", action="store_true", help="Show performance analytics"
    )
    parser.add_argument("--export", action="store_true", help="Export bets to CSV")
    parser.add_argument("--days", type=int, help="Days ahead to scan (default: 2)")
    parser.add_argument("--reset", action="store_true", help="Reset paper trading")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.reset:
        # Reset by clearing database tables
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM bets")
        cursor.execute("DELETE FROM bankroll")
        cursor.execute("DELETE FROM opportunities")
        conn.commit()
        conn.close()
        # Re-initialize bankroll
        init_bankroll(INITIAL_BANKROLL)
        print(f"Paper trading reset. Starting bankroll: £{INITIAL_BANKROLL:.2f}")

    elif args.bets:
        bets = get_recent_bets(20)
        if len(bets) == 0:
            print("No bets recorded yet")
        else:
            print("\n" + "=" * 90)
            print("RECENT BETS")
            print("=" * 90)
            for _, bet in bets.iterrows():
                status_sym = {"pending": "⏳", "won": "✓", "lost": "✗"}.get(
                    bet["status"], "?"
                )
                profit_str = f"£{bet['profit']:+.2f}" if bet["profit"] else "pending"
                print(
                    f"{status_sym} {bet['match_date']} | {bet['league'][:15]:15} | "
                    f"{bet['home_team'][:12]:12} vs {bet['away_team'][:12]:12} | "
                    f"{bet['selection'].upper():4} @ {bet['odds']:.2f} | "
                    f"£{bet['stake']:.2f} | {profit_str}"
                )

    elif args.analytics:
        stats = get_performance_analytics()
        if "error" in stats:
            print(stats["error"])
        else:
            print("\n" + "=" * 70)
            print("PERFORMANCE ANALYTICS")
            print("=" * 70)
            print(f"Total bets settled: {stats['total_bets']}")
            print(f"Total profit: £{stats['total_profit']:+.2f}")
            print(f"Total staked: £{stats['total_staked']:.2f}")
            print(f"Yield: {stats['yield_pct']:+.2f}%")
            print(f"\nAverage odds: {stats['avg_odds']:.2f}")
            print(f"Average edge: {stats['avg_edge']:.1f}%")
            print(
                f"\nMax drawdown: £{stats['max_drawdown']:.2f} ({stats['max_drawdown_pct']:.1f}%)"
            )
            print(f"Max win streak: {stats['max_win_streak']}")
            print(f"Max loss streak: {stats['max_loss_streak']}")
            print(f"Current streak: {stats['current_streak']}")

            if stats["calibration"]:
                print("\nMODEL CALIBRATION (predicted vs actual win rates):")
                print(
                    f"{'Edge Bucket':<12} {'Bets':<6} {'Wins':<6} {'Expected':<10} {'Actual':<10} {'Profit':<10}"
                )
                print("-" * 60)
                for row in stats["calibration"]:
                    print(
                        f"{row['edge_bucket']:<12} {row['bets']:<6} {row['wins']:<6} "
                        f"{row['expected_win_rate'] * 100:.1f}%{'':5} {row['actual_win_rate'] * 100:.1f}%{'':5} "
                        f"£{row['profit']:+.2f}"
                    )

    elif args.export:
        filepath = export_bets_to_csv()
        print(f"Bets exported to: {filepath}")

    elif args.status:
        scanner = Scanner()
        scanner.show_status()

    elif args.results or args.date:
        scanner = Scanner()
        scanner.settle_bets(args.date)

    elif args.daemon:
        if not BETFAIR_AVAILABLE:
            print("ERROR: betfairlightweight not installed")
            return
        run_daemon(interval_minutes=args.interval)

    else:
        if not BETFAIR_AVAILABLE:
            print("ERROR: betfairlightweight not installed")
            print("Run: pip install betfairlightweight")
            return
        scan_all_leagues(days_ahead=args.days)


if __name__ == "__main__":
    main()
