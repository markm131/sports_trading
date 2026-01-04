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
    # England
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
    # Germany
    "bundesliga": {
        "code": "D1",
        "name": "Bundesliga",
        "country": "germany",
    },
    "bundesliga_2": {
        "code": "D2",
        "name": "Bundesliga 2",
        "country": "germany",
    },
    # Spain
    "la_liga": {
        "code": "SP1",
        "name": "La Liga",
        "country": "spain",
    },
    "la_liga_2": {
        "code": "SP2",
        "name": "La Liga 2",
        "country": "spain",
    },
    # Italy
    "serie_a": {
        "code": "I1",
        "name": "Serie A",
        "country": "italy",
    },
    "serie_b": {
        "code": "I2",
        "name": "Serie B",
        "country": "italy",
    },
    # France
    "ligue_1": {
        "code": "F1",
        "name": "Ligue 1",
        "country": "france",
    },
    "ligue_2": {
        "code": "F2",
        "name": "Ligue 2",
        "country": "france",
    },
    # Netherlands
    "eredivisie": {
        "code": "N1",
        "name": "Eredivisie",
        "country": "netherlands",
    },
    # Belgium
    "jupiler_league": {
        "code": "B1",
        "name": "Jupiler League",
        "country": "belgium",
    },
    # Portugal
    "primeira_liga": {
        "code": "P1",
        "name": "Primeira Liga",
        "country": "portugal",
    },
    # Scotland
    "scottish_premiership": {
        "code": "SC0",
        "name": "Scottish Premiership",
        "country": "scotland",
    },
}


# ============================================================================
# Trading Configuration
# ============================================================================

# Bankroll settings
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


# ============================================================================
# Data Source
# ============================================================================

FOOTBALL_DATA_BASE_URL = "https://www.football-data.co.uk/mmz4281/{}/{}.csv"

DEFAULT_START_YEAR = 2017
DEFAULT_END_YEAR = 2026

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


# ============================================================================
# Betfair Configuration
# ============================================================================

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
