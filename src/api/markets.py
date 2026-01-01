# src/api/markets.py
"""
Betfair market listing utilities.

The main scanner uses BetfairClient in scanner.py directly.
This module provides standalone market queries for testing.
"""

from datetime import datetime, timedelta

from betfairlightweight import filters

from src.api.auth import get_betfair_client


def list_football_markets(days_ahead: int = 1, max_results: int = 20):
    """List upcoming football Match Odds markets."""
    client = get_betfair_client()
    client.login()

    try:
        now = datetime.utcnow()
        end_time = now + timedelta(days=days_ahead)

        time_range = filters.time_range(
            from_=now.isoformat(),
            to=end_time.isoformat(),
        )

        market_filter = filters.market_filter(
            event_type_ids=["1"],  # Soccer
            market_type_codes=["MATCH_ODDS"],
            market_start_time=time_range,
        )

        catalogues = client.betting.list_market_catalogue(
            filter=market_filter,
            market_projection=["EVENT", "MARKET_START_TIME", "RUNNER_METADATA"],
            sort="FIRST_TO_START",
            max_results=max_results,
        )

        print(f"Found {len(catalogues)} markets:\n")

        for cat in catalogues:
            print(f"{cat.market_start_time} | {cat.event.name}")
            for runner in cat.runners:
                print(f"  - {runner.runner_name}")
            print()

        return catalogues

    finally:
        client.logout()


if __name__ == "__main__":
    list_football_markets()
