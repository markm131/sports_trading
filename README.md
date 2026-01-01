# Sports Trading Bot

Automated value betting system for European football using Dixon-Coles Poisson model and Betfair Exchange.

## Features

- **14 European Leagues** - Premier League, Bundesliga, La Liga, Serie A, Ligue 1, and more
- **Dixon-Coles Model** - Time-weighted Poisson with form factors and home advantage
- **Betfair Integration** - Live odds scanning via Betfair API
- **Paper Trading** - Database-backed bankroll tracking with CLV measurement
- **Backtesting** - Walk-forward validation using historical Pinnacle odds
- **Notifications** - Discord webhooks and email alerts
- **Deployment Ready** - Scripts for Windows Task Scheduler and AWS EC2

## Quick Start

```bash
# Clone and setup
git clone https://github.com/markm131/sports_trading.git
cd sports_trading
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Configure credentials
cp .env.example .env
# Edit .env with your Betfair credentials

# Run a scan
python -m src.trading.scanner

# Check status
python -m src.trading.scanner --status
```

## Commands

| Command | Description |
|---------|-------------|
| `python -m src.trading.scanner` | Single scan of all leagues |
| `python -m src.trading.scanner --daemon` | Run continuously (every 15 mins) |
| `python -m src.trading.scanner --status` | Show bankroll and pending bets |
| `python -m src.trading.scanner --results` | Settle yesterday's bets |
| `python -m src.trading.scanner --analytics` | Performance statistics |
| `python -m src.trading.scanner --bets` | Show recent bets |
| `python -m src.trading.scanner --export` | Export bets to CSV |
| `python -m src.trading.backtest` | Run historical backtest |

## Project Structure

```
sports_trading/
├── src/
│   ├── data/           # Data pipeline (fetch, clean, store)
│   │   ├── fetcher.py      # Download from football-data.co.uk
│   │   ├── cleaner.py      # Standardize team names
│   │   └── db_writer.py    # SQLite database operations
│   ├── models/         # Prediction models
│   │   └── poisson.py      # Dixon-Coles implementation
│   ├── trading/        # Trading logic
│   │   ├── scanner.py      # Main Betfair scanner
│   │   ├── edge.py         # Edge calculation
│   │   ├── kelly.py        # Kelly criterion staking
│   │   ├── backtest.py     # Historical backtesting
│   │   └── notifications.py
│   └── api/            # Betfair API wrapper
├── scripts/            # Automation scripts
├── docs/               # Deployment guides
├── db/                 # SQLite database
├── data/               # Raw and processed CSV data
└── logs/               # Scanner logs
```

## Configuration

Create a `.env` file:

```env
BETFAIR_USERNAME=your_username
BETFAIR_PASSWORD=your_password
BETFAIR_APP_KEY=your_app_key
BETFAIR_CERT_PATH=certs/client-2048.crt
BETFAIR_KEY_PATH=certs/client-2048.key

# Optional - for notifications
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

## Strategy Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `KELLY_FRACTION` | 0.10 | 10% Kelly for conservative staking |
| `MIN_EDGE` | 0.03 | Minimum 3% edge to bet |
| `MAX_EDGE` | 0.12 | Maximum 12% edge (avoid model errors) |
| `MIN_STAKE` | £2 | Minimum stake size |
| `MAX_STAKE_PCT` | 0.02 | Max 2% of bankroll per bet |

## Deployment

### Local (Windows Task Scheduler)
See [docs/scheduler_setup.md](docs/scheduler_setup.md)

### Cloud (AWS EC2)
See [docs/aws_deployment.md](docs/aws_deployment.md)

## Backtest Results

Using actual Pinnacle closing odds (the sharpest market):

```
python -m src.trading.backtest --start 2024-01-01 --end 2025-12-31
```

The backtest measures your edge against sharp closing lines. Live Betfair performance may exceed backtest due to:
- Earlier prices (before line movement)
- Zero margin on Betfair Exchange
- CLV (Closing Line Value) tracking measures this

## Adding New Models

Create a new model in `src/models/`:

```python
# src/models/your_model.py
class YourModel:
    def fit(self, df: pd.DataFrame) -> "YourModel":
        # Train on historical data
        return self
    
    def predict_fixture(self, home: str, away: str) -> dict:
        # Return probabilities
        return {
            "home_win": 0.45,
            "draw": 0.28,
            "away_win": 0.27
        }
```

Then swap it into `scanner.py` in the `load_model()` method.

## License

MIT
