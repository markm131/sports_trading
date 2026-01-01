# src/trading/kelly.py
"""
Kelly Criterion bet sizing.

Calculates optimal stake based on edge and odds.
Uses fractional Kelly to reduce variance.

Example
-------
>>> from src.trading.kelly import KellyCalculator
>>> kelly = KellyCalculator(bankroll=1000, fraction=0.25)
>>> stake = kelly.calculate_stake(model_prob=0.40, odds=3.0)
>>> print(f"Stake: £{stake:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

# ---------------------------------------------------------------------
# Kelly Criterion
# ---------------------------------------------------------------------


def kelly_fraction(prob: float, odds: float) -> float:
    """
    Calculate full Kelly fraction.

    f* = (bp - q) / b

    Where:
        b = odds - 1 (profit per unit)
        p = probability of winning
        q = 1 - p (probability of losing)

    Returns
    -------
    float
        Optimal fraction of bankroll to bet.
        Returns 0 if no edge (negative Kelly).
    """
    if prob <= 0 or prob >= 1 or odds <= 1:
        return 0.0

    b = odds - 1  # Profit per unit staked
    p = prob
    q = 1 - prob

    f = (b * p - q) / b

    # No edge = no bet
    return max(0.0, f)


def kelly_stake(
    prob: float,
    odds: float,
    bankroll: float,
    fraction: float = 0.25,
    max_stake_pct: float = 0.05,
) -> float:
    """
    Calculate stake using fractional Kelly.

    Parameters
    ----------
    prob : float
        Model's estimated probability of winning.
    odds : float
        Decimal odds offered.
    bankroll : float
        Current bankroll size.
    fraction : float
        Kelly fraction to use (0.25 = quarter Kelly). Default 0.25.
    max_stake_pct : float
        Maximum stake as percentage of bankroll. Default 0.05 (5%).

    Returns
    -------
    float
        Recommended stake amount.
    """
    f = kelly_fraction(prob, odds)

    if f <= 0:
        return 0.0

    # Apply fractional Kelly
    stake_pct = f * fraction

    # Cap at maximum
    stake_pct = min(stake_pct, max_stake_pct)

    return bankroll * stake_pct


# ---------------------------------------------------------------------
# Kelly Calculator Class
# ---------------------------------------------------------------------


@dataclass
class BetRecommendation:
    """A sized bet recommendation."""

    fixture: str
    market: str
    selection: str
    model_prob: float
    odds: float
    edge: float
    kelly_fraction: float
    stake: float
    bankroll_pct: float
    expected_value: float

    def __repr__(self) -> str:
        return (
            f"{self.fixture} | {self.selection} @ {self.odds:.2f} | "
            f"Edge {self.edge:+.1%} | Stake £{self.stake:.2f} ({self.bankroll_pct:.1%})"
        )


class KellyCalculator:
    """
    Kelly Criterion bet sizing calculator.

    Parameters
    ----------
    bankroll : float
        Current bankroll in your currency.
    fraction : float
        Kelly fraction to use. Default 0.25 (quarter Kelly).
        - 0.25 = Conservative, lower variance
        - 0.50 = Moderate
        - 1.00 = Full Kelly (aggressive, high variance)
    min_edge : float
        Minimum edge to consider betting. Default 0.03 (3%).
    max_edge : float
        Maximum edge to consider. Default 0.12 (12%).
    max_stake_pct : float
        Maximum stake as % of bankroll. Default 0.05 (5%).
    min_stake : float
        Minimum stake amount. Default 1.0.
    """

    def __init__(
        self,
        bankroll: float,
        fraction: float = 0.10,
        min_edge: float = 0.03,
        max_edge: float = 0.12,
        max_stake_pct: float = 0.03,
        min_stake: float = 1.0,
    ):
        self.bankroll = bankroll
        self.fraction = fraction
        self.min_edge = min_edge
        self.max_edge = max_edge
        self.max_stake_pct = max_stake_pct
        self.min_stake = min_stake

    def update_bankroll(self, new_bankroll: float) -> None:
        """Update bankroll after wins/losses."""
        self.bankroll = new_bankroll

    def calculate_stake(
        self,
        model_prob: float,
        odds: float,
    ) -> float:
        """
        Calculate recommended stake for a single bet.

        Returns 0 if edge is outside acceptable range.
        """
        # Calculate edge
        implied_prob = 1 / odds
        edge = model_prob - implied_prob

        # Check edge bounds
        if edge < self.min_edge or edge > self.max_edge:
            return 0.0

        stake = kelly_stake(
            prob=model_prob,
            odds=odds,
            bankroll=self.bankroll,
            fraction=self.fraction,
            max_stake_pct=self.max_stake_pct,
        )

        # Apply minimum
        if stake < self.min_stake:
            return 0.0

        return round(stake, 2)

    def size_bet(
        self,
        fixture: str,
        market: str,
        selection: str,
        model_prob: float,
        odds: float,
    ) -> Optional[BetRecommendation]:
        """
        Size a bet and return full recommendation.

        Returns None if bet doesn't meet criteria.
        """
        stake = self.calculate_stake(model_prob, odds)

        if stake == 0:
            return None

        implied_prob = 1 / odds
        edge = model_prob - implied_prob
        f = kelly_fraction(model_prob, odds)
        ev = (model_prob * (odds - 1)) - ((1 - model_prob) * 1)

        return BetRecommendation(
            fixture=fixture,
            market=market,
            selection=selection,
            model_prob=model_prob,
            odds=odds,
            edge=edge,
            kelly_fraction=f,
            stake=stake,
            bankroll_pct=stake / self.bankroll,
            expected_value=ev,
        )

    def size_multiple_bets(
        self,
        bets: List[dict],
    ) -> List[BetRecommendation]:
        """
        Size multiple bets.

        Parameters
        ----------
        bets : list of dict
            Each dict should have: fixture, market, selection, model_prob, odds

        Returns
        -------
        list
            List of BetRecommendation objects, sorted by stake descending.
        """
        recommendations = []

        for bet in bets:
            rec = self.size_bet(
                fixture=bet["fixture"],
                market=bet["market"],
                selection=bet["selection"],
                model_prob=bet["model_prob"],
                odds=bet["odds"],
            )
            if rec is not None:
                recommendations.append(rec)

        # Sort by stake (highest first)
        recommendations.sort(key=lambda x: x.stake, reverse=True)

        return recommendations

    def simulate_returns(
        self,
        bt_df: pd.DataFrame,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Simulate Kelly-sized returns on backtest data.

        Parameters
        ----------
        bt_df : pd.DataFrame
            Backtest results with columns:
            model_prob, odds, edge, won, profit
        verbose : bool
            Print progress updates.

        Returns
        -------
        pd.DataFrame
            Original dataframe with added columns:
            kelly_stake, kelly_profit, running_bankroll
        """
        df = bt_df.copy()

        # Filter to value bets
        df = df[(df["edge"] >= self.min_edge) & (df["edge"] <= self.max_edge)].copy()
        df = df.sort_values("match_date").reset_index(drop=True)

        if len(df) == 0:
            return df

        # Track bankroll
        running_bankroll = self.bankroll
        stakes = []
        profits = []
        bankrolls = []

        for i, row in df.iterrows():
            stake = kelly_stake(
                prob=row["model_prob"],
                odds=row["odds"],
                bankroll=running_bankroll,
                fraction=self.fraction,
                max_stake_pct=self.max_stake_pct,
            )

            # Apply minimum
            if stake < self.min_stake:
                stake = 0

            # Calculate profit
            if stake > 0:
                if row["won"]:
                    profit = stake * (row["odds"] - 1)
                else:
                    profit = -stake
            else:
                profit = 0

            # Update bankroll
            running_bankroll += profit

            stakes.append(stake)
            profits.append(profit)
            bankrolls.append(running_bankroll)

            if verbose and i % 500 == 0:
                print(f"  Bet {i}: Bankroll £{running_bankroll:.2f}")

        df["kelly_stake"] = stakes
        df["kelly_profit"] = profits
        df["running_bankroll"] = bankrolls

        return df

    def summary(self) -> str:
        """Return calculator settings summary."""
        return (
            f"Kelly Calculator\n"
            f"  Bankroll:    £{self.bankroll:,.2f}\n"
            f"  Fraction:    {self.fraction:.0%} Kelly\n"
            f"  Edge range:  {self.min_edge:.0%} - {self.max_edge:.0%}\n"
            f"  Max stake:   {self.max_stake_pct:.0%} of bankroll\n"
            f"  Min stake:   £{self.min_stake:.2f}"
        )


# ---------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------

if __name__ == "__main__":
    from src.data.db_writer import get_db_connection
    from src.trading.edge import rolling_backtest

    # Load data
    conn = get_db_connection()
    df = pd.read_sql(
        """
        SELECT home_team, away_team,
               full_time_home_goals, full_time_away_goals,
               home_shots_on_target, away_shots_on_target,
               full_time_result, match_date,
               pinnacle_home_odds, pinnacle_draw_odds, pinnacle_away_odds
        FROM matches
        WHERE league = 'Championship'
          AND season >= '2020-21'
        ORDER BY match_date
        """,
        conn,
    )
    print(f"Loaded {len(df)} matches\n")

    # Run backtest
    print("Running backtest...")
    bt = rolling_backtest(
        df,
        min_train_matches=500,
        half_life_days=180,
        form_weight=0.3,
        min_edge=0.03,
        max_edge=0.12,
        refit_frequency="monthly",
        markets=["1x2"],
        verbose=False,
    )
    print(f"Generated {len(bt)} predictions\n")

    # Flat staking baseline
    value_bets = bt[(bt["edge"] >= 0.03) & (bt["edge"] <= 0.12)]
    flat_profit = value_bets["profit"].sum()
    flat_roi = flat_profit / len(value_bets) * 100

    print("=" * 60)
    print("Flat Staking (£10/bet)")
    print("=" * 60)
    print(f"  Bets:        {len(value_bets)}")
    print(f"  Profit:      {flat_profit * 10:+.2f} units (£{flat_profit * 10:+.2f})")
    print(f"  ROI:         {flat_roi:+.1f}%")

    # Kelly staking
    print("\n" + "=" * 60)
    print("Kelly Staking (£1000 bankroll, 10% Kelly)")
    print("=" * 60)

    kelly = KellyCalculator(
        bankroll=1000,
        fraction=0.10,
        min_edge=0.03,
        max_edge=0.12,
        max_stake_pct=0.03,
        min_stake=5.0,
    )
    print(kelly.summary())
    print()

    # Simulate
    results = kelly.simulate_returns(bt, verbose=True)

    if len(results) > 0:
        total_staked = results["kelly_stake"].sum()
        total_profit = results["kelly_profit"].sum()
        final_bankroll = results["running_bankroll"].iloc[-1]
        peak_bankroll = results["running_bankroll"].max()
        min_bankroll = results["running_bankroll"].min()

        print("\nResults:")
        print(f"  Bets placed: {(results['kelly_stake'] > 0).sum()}")
        print(f"  Total staked: £{total_staked:,.2f}")
        print(f"  Total profit: £{total_profit:+,.2f}")
        print(f"  ROI:          {total_profit / total_staked * 100:+.1f}%")
        print(f"  Final bank:   £{final_bankroll:,.2f}")
        print(f"  Peak bank:    £{peak_bankroll:,.2f}")
        print(f"  Min bank:     £{min_bankroll:,.2f}")
        print(f"  Return:       {(final_bankroll - 1000) / 1000 * 100:+.1f}%")

    # Example single bet sizing
    print("\n" + "=" * 60)
    print("Example: Sizing a single bet")
    print("=" * 60)

    rec = kelly.size_bet(
        fixture="Arsenal vs Chelsea",
        market="1x2",
        selection="home",
        model_prob=0.45,
        odds=2.20,
    )

    if rec:
        print(rec)
        print(f"  Full Kelly: {rec.kelly_fraction:.1%}")
        print(f"  EV per £1:  £{rec.expected_value:.3f}")
    else:
        print("  No bet (edge outside range)")

    print("\nDone")
