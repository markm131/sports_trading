# src/trading/edge.py
"""
Edge calculation for sports trading.

Compares model probabilities to market odds to identify value bets.
Filters to 3-12% edge range (the profitable sweet spot).

Example
-------
>>> from src.models.poisson import PoissonModel
>>> from src.trading.edge import EdgeCalculator
>>>
>>> model = PoissonModel().fit(df)
>>> calc = EdgeCalculator(model)
>>> value_bets = calc.find_value_bets("Arsenal", "Chelsea", odds_1x2={...})
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.models.poisson import PoissonModel

# ---------------------------------------------------------------------
# Edge thresholds (based on backtest results)
# ---------------------------------------------------------------------

MIN_EDGE = 0.03  # 3% minimum edge
MAX_EDGE = 0.12  # 12% maximum (above this, model is overconfident)


# ---------------------------------------------------------------------
# Probability / Odds Conversion
# ---------------------------------------------------------------------


def odds_to_prob(odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if odds <= 1.0:
        raise ValueError(f"Invalid odds: {odds}")
    return 1.0 / odds


def prob_to_odds(prob: float) -> float:
    """Convert probability to decimal odds."""
    if prob <= 0 or prob >= 1:
        raise ValueError(f"Invalid probability: {prob}")
    return 1.0 / prob


def remove_vig(odds: Dict[str, float]) -> Dict[str, float]:
    """Remove bookmaker margin from odds."""
    implied = {k: odds_to_prob(v) for k, v in odds.items()}
    total = sum(implied.values())
    return {k: v / total for k, v in implied.items()}


def calculate_overround(odds: Dict[str, float]) -> float:
    """Calculate bookmaker margin as percentage."""
    implied = sum(odds_to_prob(o) for o in odds.values())
    return (implied - 1.0) * 100


# ---------------------------------------------------------------------
# Edge Calculation
# ---------------------------------------------------------------------


@dataclass
class BetOpportunity:
    """Represents a betting opportunity."""

    fixture: str
    market: str
    selection: str
    model_prob: float
    market_prob: float
    edge: float
    odds: float
    expected_value: float

    def __repr__(self) -> str:
        return (
            f"{self.fixture} | {self.market} {self.selection}: "
            f"Edge {self.edge:+.1%} @ {self.odds:.2f}"
        )


class EdgeCalculator:
    """
    Calculate edges between model probabilities and market odds.

    Parameters
    ----------
    model : PoissonModel
        Fitted probability model.
    min_edge : float
        Minimum edge threshold (default 0.03 = 3%).
    max_edge : float
        Maximum edge threshold (default 0.12 = 12%).
        Edges above this are often model overconfidence.
    """

    def __init__(
        self,
        model: PoissonModel,
        min_edge: float = MIN_EDGE,
        max_edge: float = MAX_EDGE,
    ):
        self.model = model
        self.min_edge = min_edge
        self.max_edge = max_edge
        model._check_fitted()

    def _calc_edge(self, model_prob: float, market_prob: float) -> Tuple[float, float]:
        """Calculate edge and expected value."""
        edge = model_prob - market_prob
        odds = prob_to_odds(market_prob)
        ev = (model_prob * (odds - 1)) - ((1 - model_prob) * 1)
        return edge, ev

    def _is_value(self, edge: float) -> bool:
        """Check if edge is in the profitable range."""
        return self.min_edge <= edge <= self.max_edge

    def check_fixture(
        self,
        home: str,
        away: str,
        odds_1x2: Optional[Dict[str, float]] = None,
        odds_ou25: Optional[Dict[str, float]] = None,
        odds_btts: Optional[Dict[str, float]] = None,
        match_date: Optional[pd.Timestamp] = None,
    ) -> Dict[str, Dict]:
        """
        Check a fixture for edges across multiple markets.

        Parameters
        ----------
        home, away : str
            Team names.
        odds_1x2 : dict
            {"home": x, "draw": y, "away": z}
        odds_ou25 : dict
            {"over": x, "under": y}
        odds_btts : dict
            {"yes": x, "no": y}
        match_date : Timestamp
            For form calculation.

        Returns
        -------
        dict
            Nested dict of market -> selection -> edge info
        """
        pred = self.model.predict_fixture(home, away, match_date=match_date)
        results = {}

        # 1X2 Market
        if odds_1x2:
            fair_probs = remove_vig(odds_1x2)
            results["1x2"] = {}

            for selection, model_key in [
                ("home", "home_win"),
                ("draw", "draw"),
                ("away", "away_win"),
            ]:
                model_prob = pred[model_key]
                market_prob = fair_probs[selection]
                edge, ev = self._calc_edge(model_prob, market_prob)

                results["1x2"][selection] = {
                    "model_prob": model_prob,
                    "market_prob": market_prob,
                    "edge": edge,
                    "ev": ev,
                    "odds": odds_1x2[selection],
                    "value": self._is_value(edge),
                }

        # O/U 2.5 Market
        if odds_ou25:
            fair_probs = remove_vig(odds_ou25)
            results["ou25"] = {}

            for selection, model_key in [
                ("over", "over_2_5"),
                ("under", "under_2_5"),
            ]:
                model_prob = pred[model_key]
                market_prob = fair_probs[selection]
                edge, ev = self._calc_edge(model_prob, market_prob)

                results["ou25"][selection] = {
                    "model_prob": model_prob,
                    "market_prob": market_prob,
                    "edge": edge,
                    "ev": ev,
                    "odds": odds_ou25[selection],
                    "value": self._is_value(edge),
                }

        # BTTS Market
        if odds_btts:
            fair_probs = remove_vig(odds_btts)
            results["btts"] = {}

            for selection, model_key in [
                ("yes", "btts_yes"),
                ("no", "btts_no"),
            ]:
                model_prob = pred[model_key]
                market_prob = fair_probs[selection]
                edge, ev = self._calc_edge(model_prob, market_prob)

                results["btts"][selection] = {
                    "model_prob": model_prob,
                    "market_prob": market_prob,
                    "edge": edge,
                    "ev": ev,
                    "odds": odds_btts[selection],
                    "value": self._is_value(edge),
                }

        return results

    def find_value_bets(
        self,
        home: str,
        away: str,
        odds_1x2: Optional[Dict[str, float]] = None,
        odds_ou25: Optional[Dict[str, float]] = None,
        odds_btts: Optional[Dict[str, float]] = None,
        match_date: Optional[pd.Timestamp] = None,
    ) -> List[BetOpportunity]:
        """
        Find all value bets (3-12% edge) for a fixture.

        Returns list of BetOpportunity objects sorted by edge.
        """
        edges = self.check_fixture(
            home, away, odds_1x2, odds_ou25, odds_btts, match_date
        )
        fixture = f"{home} vs {away}"

        opportunities = []

        for market, selections in edges.items():
            for selection, data in selections.items():
                if data["value"]:
                    opportunities.append(
                        BetOpportunity(
                            fixture=fixture,
                            market=market,
                            selection=selection,
                            model_prob=data["model_prob"],
                            market_prob=data["market_prob"],
                            edge=data["edge"],
                            odds=data["odds"],
                            expected_value=data["ev"],
                        )
                    )

        opportunities.sort(key=lambda x: x.edge, reverse=True)
        return opportunities

    @staticmethod
    def _analyze_df(bt_df: pd.DataFrame, min_edge: float = MIN_EDGE) -> Dict:
        """Analyze backtest results."""
        all_bets = bt_df
        value_bets = bt_df[(bt_df["edge"] >= min_edge) & (bt_df["edge"] <= MAX_EDGE)]

        def calc_stats(df: pd.DataFrame, name: str) -> Dict:
            if len(df) == 0:
                return {f"{name}_count": 0}
            return {
                f"{name}_count": len(df),
                f"{name}_win_rate": df["won"].mean(),
                f"{name}_avg_odds": df["odds"].mean(),
                f"{name}_total_profit": df["profit"].sum(),
                f"{name}_roi": df["profit"].sum() / len(df) * 100,
                f"{name}_avg_edge": df["edge"].mean(),
            }

        stats = {}
        stats.update(calc_stats(all_bets, "all"))
        stats.update(calc_stats(value_bets, "value"))
        return stats


# ---------------------------------------------------------------------
# Rolling Backtest
# ---------------------------------------------------------------------


def rolling_backtest(
    df: pd.DataFrame,
    min_train_matches: int = 500,
    half_life_days: float = 180.0,
    form_weight: float = 0.3,
    min_edge: float = MIN_EDGE,
    max_edge: float = MAX_EDGE,
    refit_frequency: str = "monthly",
    markets: list = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Walk-forward backtest with no lookahead bias.

    Parameters
    ----------
    df : pd.DataFrame
        Match data with odds columns.
    min_train_matches : int
        Minimum training set size.
    half_life_days : float
        Time decay half-life.
    form_weight : float
        Weight for form adjustment.
    min_edge, max_edge : float
        Edge thresholds for value bets.
    refit_frequency : str
        "daily", "weekly", or "monthly".
    markets : list
        ["1x2"], ["ou25"], or both.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Backtest results.
    """
    if markets is None:
        markets = ["1x2"]

    df = df.copy()
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values("match_date").reset_index(drop=True)
    df = df.dropna(subset=["full_time_home_goals", "full_time_away_goals"])
    df["total_goals"] = df["full_time_home_goals"] + df["full_time_away_goals"]

    if len(df) < min_train_matches + 50:
        raise ValueError(f"Need at least {min_train_matches + 50} matches")

    match_counts = df.groupby(df["match_date"].dt.date).size().cumsum()
    start_idx = match_counts[match_counts >= min_train_matches].index[0]
    start_date = pd.Timestamp(start_idx)

    if verbose:
        print(f"Starting from {start_date.date()}")
        print(f"Total matches: {len(df)}")
        print(f"Markets: {', '.join(markets)}")
        print(f"Edge range: {min_edge:.0%} - {max_edge:.0%}")

    freq_days = {"daily": 1, "weekly": 7, "monthly": 30}[refit_frequency]

    results = []
    current_model = None
    last_fit_date = None
    n_refits = 0

    test_dates = df[df["match_date"] >= start_date]["match_date"].dt.date.unique()

    for i, test_date in enumerate(test_dates):
        test_date = pd.Timestamp(test_date)

        needs_refit = (
            current_model is None
            or last_fit_date is None
            or (test_date - last_fit_date).days >= freq_days
        )

        if needs_refit:
            train_df = df[df["match_date"] < test_date].copy()

            if len(train_df) < min_train_matches:
                continue

            try:
                current_model = PoissonModel(
                    half_life_days=half_life_days,
                    form_weight=form_weight,
                ).fit(train_df)
                last_fit_date = test_date
                n_refits += 1

                if verbose:
                    pct = (i / len(test_dates)) * 100
                    print(
                        f"  [{pct:5.1f}%] Refit #{n_refits} at {test_date.date()} ({len(train_df)} matches)"
                    )
            except Exception as e:
                if verbose:
                    print(f"  Warning: Fit failed at {test_date.date()}: {e}")
                continue

        day_matches = df[df["match_date"].dt.date == test_date.date()]

        for _, row in day_matches.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            result = row["full_time_result"]
            total_goals = row["total_goals"]

            if home not in current_model.teams or away not in current_model.teams:
                continue

            try:
                pred = current_model.predict_fixture(home, away, match_date=test_date)
            except Exception:
                continue

            # 1X2 Market
            if "1x2" in markets:
                odds = {
                    "home": row.get("pinnacle_home_odds"),
                    "draw": row.get("pinnacle_draw_odds"),
                    "away": row.get("pinnacle_away_odds"),
                }

                if all(pd.notna(o) and o > 1.0 for o in odds.values()):
                    fair = remove_vig(odds)

                    for sel, model_key, result_val in [
                        ("home", "home_win", "H"),
                        ("draw", "draw", "D"),
                        ("away", "away_win", "A"),
                    ]:
                        model_prob = pred[model_key]
                        market_prob = fair[sel]
                        edge = model_prob - market_prob
                        won = 1 if result == result_val else 0
                        profit = (odds[sel] - 1) if won else -1
                        is_value = min_edge <= edge <= max_edge

                        results.append(
                            {
                                "match_date": row["match_date"],
                                "home_team": home,
                                "away_team": away,
                                "market": "1x2",
                                "selection": sel,
                                "model_prob": model_prob,
                                "market_prob": market_prob,
                                "edge": edge,
                                "odds": odds[sel],
                                "won": won,
                                "profit": profit,
                                "is_value": is_value,
                            }
                        )

            # O/U 2.5 Market
            if "ou25" in markets:
                odds = {
                    "over": row.get("pinnacle_over_2_5_odds"),
                    "under": row.get("pinnacle_under_2_5_odds"),
                }

                if all(pd.notna(o) and o > 1.0 for o in odds.values()):
                    fair = remove_vig(odds)

                    for sel, model_key in [
                        ("over", "over_2_5"),
                        ("under", "under_2_5"),
                    ]:
                        model_prob = pred[model_key]
                        market_prob = fair[sel]
                        edge = model_prob - market_prob

                        if sel == "over":
                            won = 1 if total_goals > 2.5 else 0
                        else:
                            won = 1 if total_goals < 2.5 else 0

                        profit = (odds[sel] - 1) if won else -1
                        is_value = min_edge <= edge <= max_edge

                        results.append(
                            {
                                "match_date": row["match_date"],
                                "home_team": home,
                                "away_team": away,
                                "market": "ou25",
                                "selection": sel,
                                "model_prob": model_prob,
                                "market_prob": market_prob,
                                "edge": edge,
                                "odds": odds[sel],
                                "won": won,
                                "profit": profit,
                                "is_value": is_value,
                            }
                        )

    if verbose:
        print(f"Completed: {len(results)} predictions")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------

if __name__ == "__main__":
    from src.data.db_writer import get_db_connection

    conn = get_db_connection()
    df = pd.read_sql(
        """
        SELECT home_team, away_team,
               full_time_home_goals, full_time_away_goals,
               home_shots_on_target, away_shots_on_target,
               full_time_result, match_date,
               pinnacle_home_odds, pinnacle_draw_odds, pinnacle_away_odds,
               pinnacle_over_2_5_odds, pinnacle_under_2_5_odds
        FROM matches
        WHERE league = 'Championship'
          AND season >= '2020-21'
        ORDER BY match_date
        """,
        conn,
    )

    print(f"Loaded {len(df)} matches\n")

    # Run backtest
    print("=" * 60)
    print("Rolling Backtest (3-12% Edge Filter)")
    print("=" * 60)

    bt = rolling_backtest(
        df,
        min_train_matches=500,
        half_life_days=180,
        form_weight=0.3,
        min_edge=0.03,
        max_edge=0.12,
        refit_frequency="monthly",
        markets=["1x2"],
        verbose=True,
    )

    # Results
    stats = EdgeCalculator._analyze_df(bt)

    print("\nAll 1X2 bets:")
    print(f"  Count:  {stats['all_count']}")
    print(f"  Win:    {stats['all_win_rate']:.1%}")
    print(f"  ROI:    {stats['all_roi']:+.1f}%")

    print("\nValue bets (3-12% edge):")
    print(f"  Count:  {stats['value_count']}")
    if stats["value_count"] > 0:
        print(f"  Win:    {stats['value_win_rate']:.1%}")
        print(f"  ROI:    {stats['value_roi']:+.1f}%")
        print(f"  Profit: {stats['value_total_profit']:+.1f} units")

    # Edge breakdown
    print("\nROI by Edge Bucket:")
    bt["bucket"] = pd.cut(
        bt["edge"],
        bins=[-1, 0, 0.03, 0.05, 0.08, 0.12, 1],
        labels=["<0%", "0-3%", "3-5%", "5-8%", "8-12%", ">12%"],
    )

    for bucket in ["<0%", "0-3%", "3-5%", "5-8%", "8-12%", ">12%"]:
        bdf = bt[bt["bucket"] == bucket]
        if len(bdf) > 0:
            roi = bdf["profit"].sum() / len(bdf) * 100
            flag = " <-- VALUE" if bucket in ["3-5%", "5-8%", "8-12%"] else ""
            print(f"  {bucket:6s}: {len(bdf):4d} bets | ROI {roi:+.1f}%{flag}")

    print("\nDone")
