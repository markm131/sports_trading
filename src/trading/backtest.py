# src/trading/backtest.py
"""
Historical backtesting for the betting strategy.

Uses actual Pinnacle closing odds from our database for realistic simulation.
Pinnacle odds are the sharpest in the market and closely track Betfair exchange prices.
"""

from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd

from src.data.db_writer import get_db_connection
from src.models.poisson import PoissonModel
from src.trading.edge import remove_vig
from src.trading.kelly import kelly_stake


def backtest_strategy(
    start_date: str = None,
    end_date: str = None,
    initial_bankroll: float = 1000.0,
    min_edge: float = 0.03,
    max_edge: float = 0.12,
    kelly_fraction: float = 0.10,
    max_stake_pct: float = 0.02,
    min_stake: float = 2.0,
    leagues: List[str] = None,
    training_days: int = 365,
) -> Dict:
    """
    Backtest the betting strategy on historical data using ACTUAL Pinnacle odds.

    Args:
        start_date: Start of backtest period (YYYY-MM-DD)
        end_date: End of backtest period (YYYY-MM-DD)
        initial_bankroll: Starting bankroll
        min_edge: Minimum edge to bet
        max_edge: Maximum edge to bet
        kelly_fraction: Kelly criterion fraction
        max_stake_pct: Max stake as % of bankroll
        min_stake: Minimum stake size
        leagues: List of leagues to include (None = all)
        training_days: Days of data to train model on

    Returns:
        Dict with backtest results
    """
    conn = get_db_connection()

    # Default date range: last 6 months
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

    # Get matches in backtest period WITH Pinnacle odds
    league_filter = ""
    if leagues:
        league_list = ", ".join(f"'{lg}'" for lg in leagues)
        league_filter = f"AND league IN ({league_list})"

    matches = pd.read_sql(
        f"""
        SELECT match_date, league, home_team, away_team,
               full_time_home_goals, full_time_away_goals, full_time_result,
               home_shots_on_target, away_shots_on_target,
               pinnacle_home_odds, pinnacle_draw_odds, pinnacle_away_odds
        FROM matches
        WHERE match_date >= ? AND match_date <= ?
        AND pinnacle_home_odds IS NOT NULL
        {league_filter}
        ORDER BY match_date
        """,
        conn,
        params=(start_date, end_date),
    )

    if len(matches) == 0:
        conn.close()
        return {"error": "No matches found in date range with Pinnacle odds"}

    print(f"Backtesting {len(matches)} matches from {start_date} to {end_date}")
    print("Using actual Pinnacle closing odds")
    print("=" * 70)

    # Track results
    bankroll = initial_bankroll
    bets = []
    models_cache = {}

    # Process each match date
    for match_date in sorted(matches["match_date"].unique()):
        day_matches = matches[matches["match_date"] == match_date]

        for _, match in day_matches.iterrows():
            league = match["league"]
            home = match["home_team"]
            away = match["away_team"]
            actual_result = match["full_time_result"]

            # Use ACTUAL Pinnacle odds
            odds = {
                "home": match["pinnacle_home_odds"],
                "draw": match["pinnacle_draw_odds"],
                "away": match["pinnacle_away_odds"],
            }

            # Skip if any odds are invalid
            if not all(o and o > 1.0 for o in odds.values()):
                continue

            # Get or fit model using data BEFORE this match
            cache_key = f"{league}_{match_date}"
            if cache_key not in models_cache:
                # Get training data (matches before this date)
                train_start = (
                    datetime.strptime(match_date, "%Y-%m-%d")
                    - timedelta(days=training_days)
                ).strftime("%Y-%m-%d")

                train_data = pd.read_sql(
                    """
                    SELECT home_team, away_team,
                           full_time_home_goals, full_time_away_goals,
                           home_shots_on_target, away_shots_on_target,
                           match_date
                    FROM matches
                    WHERE league = ? AND match_date >= ? AND match_date < ?
                    ORDER BY match_date
                    """,
                    conn,
                    params=(league, train_start, match_date),
                )

                if len(train_data) < 200:
                    continue

                try:
                    model = PoissonModel(half_life_days=180, form_weight=0.3).fit(
                        train_data
                    )
                    models_cache[cache_key] = model
                except Exception:
                    continue
            else:
                model = models_cache[cache_key]

            # Check if teams are known
            if home not in model.teams or away not in model.teams:
                continue

            # Get prediction
            try:
                pred = model.predict_fixture(home, away)
            except Exception:
                continue

            # Remove vig from actual Pinnacle odds to get fair market probabilities
            fair = remove_vig(odds)

            # Check each selection for value
            for sel, model_key, result_code in [
                ("home", "home_win", "H"),
                ("draw", "draw", "D"),
                ("away", "away_win", "A"),
            ]:
                model_prob = pred[model_key]
                market_prob = fair[sel]
                edge = model_prob - market_prob

                if min_edge <= edge <= max_edge:
                    # Calculate stake
                    stake = kelly_stake(
                        prob=model_prob,
                        odds=odds[sel],
                        bankroll=bankroll,
                        fraction=kelly_fraction,
                        max_stake_pct=max_stake_pct,
                    )

                    if stake >= min_stake:
                        # Determine outcome
                        won = actual_result == result_code
                        profit = stake * (odds[sel] - 1) if won else -stake
                        bankroll += profit

                        bets.append(
                            {
                                "date": match_date,
                                "league": league,
                                "home_team": home,
                                "away_team": away,
                                "selection": sel,
                                "odds": odds[sel],
                                "stake": stake,
                                "model_prob": model_prob,
                                "market_prob": market_prob,
                                "edge": edge,
                                "won": won,
                                "profit": profit,
                                "bankroll": bankroll,
                            }
                        )

    conn.close()

    if len(bets) == 0:
        return {"error": "No bets would have been placed with these parameters"}

    # Calculate results
    bets_df = pd.DataFrame(bets)

    total_bets = len(bets_df)
    wins = bets_df["won"].sum()
    losses = total_bets - wins
    win_rate = wins / total_bets * 100

    total_staked = bets_df["stake"].sum()
    total_profit = bets_df["profit"].sum()
    final_bankroll = bets_df["bankroll"].iloc[-1]
    roi = (final_bankroll - initial_bankroll) / initial_bankroll * 100
    yield_pct = total_profit / total_staked * 100

    # Drawdown
    bets_df["peak"] = bets_df["bankroll"].cummax()
    bets_df["drawdown"] = bets_df["peak"] - bets_df["bankroll"]
    max_drawdown = bets_df["drawdown"].max()
    max_drawdown_pct = max_drawdown / bets_df["peak"].max() * 100

    # By league
    league_results = (
        bets_df.groupby("league")
        .agg(
            bets=("date", "count"),
            wins=("won", "sum"),
            profit=("profit", "sum"),
            avg_edge=("edge", "mean"),
        )
        .reset_index()
    )
    league_results["win_rate"] = league_results["wins"] / league_results["bets"] * 100

    results = {
        "start_date": start_date,
        "end_date": end_date,
        "total_bets": total_bets,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "initial_bankroll": initial_bankroll,
        "final_bankroll": final_bankroll,
        "total_profit": total_profit,
        "total_staked": total_staked,
        "roi": roi,
        "yield_pct": yield_pct,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "avg_odds": bets_df["odds"].mean(),
        "avg_edge": bets_df["edge"].mean() * 100,
        "by_league": league_results.to_dict("records"),
        "bets": bets_df,
    }

    return results


def print_backtest_results(results: Dict):
    """Print formatted backtest results."""
    if "error" in results:
        print(f"Error: {results['error']}")
        return

    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    print(f"Period: {results['start_date']} to {results['end_date']}")
    print(f"\nTotal bets: {results['total_bets']}")
    print(f"Wins: {results['wins']} | Losses: {results['losses']}")
    print(f"Win rate: {results['win_rate']:.1f}%")
    print(f"\nInitial bankroll: £{results['initial_bankroll']:.2f}")
    print(f"Final bankroll: £{results['final_bankroll']:.2f}")
    print(f"Total profit: £{results['total_profit']:+.2f}")
    print(f"ROI: {results['roi']:+.1f}%")
    print(f"Yield: {results['yield_pct']:+.2f}%")
    print(
        f"\nMax drawdown: £{results['max_drawdown']:.2f} ({results['max_drawdown_pct']:.1f}%)"
    )
    print(f"Average odds: {results['avg_odds']:.2f}")
    print(f"Average edge: {results['avg_edge']:.1f}%")

    print("\nBy League:")
    print(
        f"{'League':<20} {'Bets':<6} {'Wins':<6} {'Win%':<8} {'Profit':<12} {'Avg Edge':<10}"
    )
    print("-" * 70)
    for lg in sorted(results["by_league"], key=lambda x: x["profit"], reverse=True):
        print(
            f"{lg['league']:<20} {lg['bets']:<6} {lg['wins']:<6} "
            f"{lg['win_rate']:.1f}%{'':3} £{lg['profit']:+.2f}{'':4} "
            f"{lg['avg_edge'] * 100:.1f}%"
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Backtest betting strategy")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--bankroll", type=float, default=1000, help="Initial bankroll")
    parser.add_argument(
        "--min-edge", type=float, default=0.03, help="Min edge (0.03 = 3%)"
    )
    parser.add_argument("--max-edge", type=float, default=0.12, help="Max edge")
    parser.add_argument("--kelly", type=float, default=0.10, help="Kelly fraction")
    parser.add_argument("--export", help="Export results to CSV file")

    args = parser.parse_args()

    results = backtest_strategy(
        start_date=args.start,
        end_date=args.end,
        initial_bankroll=args.bankroll,
        min_edge=args.min_edge,
        max_edge=args.max_edge,
        kelly_fraction=args.kelly,
    )

    print_backtest_results(results)

    if args.export and "bets" in results:
        results["bets"].to_csv(args.export, index=False)
        print(f"\nResults exported to: {args.export}")


if __name__ == "__main__":
    main()
