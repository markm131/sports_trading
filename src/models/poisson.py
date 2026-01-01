# src/models/poisson.py
"""
Dixon-Coles Poisson model with shots-based xG proxy and form adjustment.

Key features:
    - xG proxy from shots on target (reduces goal noise)
    - Form adjustment (recent matches weighted)
    - Time decay (180-day half-life)
    - Dixon-Coles correlation for low-scoring games

Example
-------
>>> from src.models.poisson import PoissonModel
>>> model = PoissonModel().fit(df)
>>> model.predict_fixture("Arsenal", "Chelsea")
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import factorial


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

SOT_CONVERSION_RATE = 0.30  # ~30% of shots on target become goals
XG_WEIGHT = 0.6  # Weight for xG proxy vs actual goals
GOALS_WEIGHT = 0.4


# ---------------------------------------------------------------------
# Dixon-Coles adjustment
# ---------------------------------------------------------------------


def _tau(x: int, y: int, lambda_h: float, lambda_a: float, rho: float) -> float:
    """Dixon-Coles correlation adjustment for low-scoring games."""
    if x == 0 and y == 0:
        return 1 - lambda_h * lambda_a * rho
    elif x == 0 and y == 1:
        return 1 + lambda_h * rho
    elif x == 1 and y == 0:
        return 1 + lambda_a * rho
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1.0


def _goal_matrix(
    lambda_home: float,
    lambda_away: float,
    rho: float = 0.0,
    max_goals: int = 10
) -> np.ndarray:
    """Goal probability matrix with Dixon-Coles adjustment."""
    goals_range = np.arange(0, max_goals + 1)
    fact = factorial(goals_range)

    p_home = np.exp(-lambda_home) * (lambda_home ** goals_range) / fact
    p_away = np.exp(-lambda_away) * (lambda_away ** goals_range) / fact

    P = np.outer(p_home, p_away)

    if rho != 0:
        for i in range(min(2, max_goals + 1)):
            for j in range(min(2, max_goals + 1)):
                P[i, j] *= _tau(i, j, lambda_home, lambda_away, rho)

    P /= P.sum()
    return P


# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------


def calculate_xg_proxy(goals: float, shots_on_target: float) -> float:
    """
    Calculate xG proxy from goals and shots on target.
    
    Blends actual goals with expected goals from shots to reduce noise.
    """
    if pd.isna(shots_on_target):
        return goals
    
    xg_from_shots = shots_on_target * SOT_CONVERSION_RATE
    return (GOALS_WEIGHT * goals) + (XG_WEIGHT * xg_from_shots)


def calculate_form(
    team: str,
    df: pd.DataFrame,
    before_date: pd.Timestamp,
    n_matches: int = 6,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate recent form for a team.
    
    Returns (avg_scored, avg_conceded) over last n matches.
    """
    team_home = df[(df["home_team"] == team) & (df["match_date"] < before_date)]
    team_away = df[(df["away_team"] == team) & (df["match_date"] < before_date)]
    
    home_matches = team_home[["match_date", "home_xg_proxy", "away_xg_proxy"]].copy()
    home_matches.columns = ["match_date", "scored", "conceded"]
    
    away_matches = team_away[["match_date", "away_xg_proxy", "home_xg_proxy"]].copy()
    away_matches.columns = ["match_date", "scored", "conceded"]
    
    all_matches = pd.concat([home_matches, away_matches])
    all_matches = all_matches.sort_values("match_date", ascending=False).head(n_matches)
    
    if len(all_matches) == 0:
        return None, None
    
    return all_matches["scored"].mean(), all_matches["conceded"].mean()


# ---------------------------------------------------------------------
# Main Model Class
# ---------------------------------------------------------------------


class PoissonModel:
    """
    Dixon-Coles model with shots-based xG and form adjustment.
    
    Parameters
    ----------
    half_life_days : float
        Time decay half-life. Default 180 days.
    form_matches : int
        Number of recent matches for form calculation.
    form_weight : float
        Weight for form adjustment (0-1).
    estimate_rho : bool
        Whether to estimate Dixon-Coles correlation.
    """

    def __init__(
        self,
        half_life_days: float = 180.0,
        form_matches: int = 6,
        form_weight: float = 0.3,
        estimate_rho: bool = True,
    ):
        self.half_life_days = half_life_days
        self.form_matches = form_matches
        self.form_weight = form_weight
        self.estimate_rho = estimate_rho

        self.fitted: bool = False
        self.teams: List[str] = []
        
        self.mu_: float = 0.0  # Home advantage
        self.rho_: float = 0.0  # Dixon-Coles correlation
        
        self.attack_: Dict[str, float] = {}
        self.defence_: Dict[str, float] = {}
        
        self._league_avg_attack: float = 0.0
        self._league_avg_defence: float = 0.0
        self._league_avg_goals: float = 1.4
        
        self._match_data: Optional[pd.DataFrame] = None
        
        self.n_matches_: int = 0
        self.fit_date_: Optional[str] = None

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add xG proxy columns."""
        df = df.copy()
        
        for col in ["full_time_home_goals", "full_time_away_goals",
                    "home_shots_on_target", "away_shots_on_target"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df["home_xg_proxy"] = df.apply(
            lambda r: calculate_xg_proxy(
                r["full_time_home_goals"],
                r.get("home_shots_on_target", np.nan)
            ),
            axis=1
        )
        df["away_xg_proxy"] = df.apply(
            lambda r: calculate_xg_proxy(
                r["full_time_away_goals"],
                r.get("away_shots_on_target", np.nan)
            ),
            axis=1
        )
        
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
        
        return df

    def _compute_weights(self, df: pd.DataFrame) -> np.ndarray:
        """Exponential time decay weights."""
        if self.half_life_days is None:
            return np.ones(len(df))

        dates = df["match_date"]
        if dates.isna().all():
            return np.ones(len(df))

        max_date = dates.max()
        days_ago = (max_date - dates).dt.days.fillna(0).values

        decay_rate = np.log(2) / self.half_life_days
        return np.exp(-decay_rate * days_ago)

    def fit(self, df: pd.DataFrame) -> "PoissonModel":
        """
        Fit the model.
        
        Required columns:
            - home_team, away_team
            - full_time_home_goals, full_time_away_goals
            - home_shots_on_target, away_shots_on_target (optional)
            - match_date
        """
        required = {"home_team", "away_team", "full_time_home_goals", "full_time_away_goals"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df = self._preprocess_data(df)
        df = df.dropna(subset=["full_time_home_goals", "full_time_away_goals"])
        
        if len(df) < 100:
            raise ValueError(f"Need at least 100 matches, got {len(df)}")

        self._match_data = df.copy()

        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self.teams = teams
        n_teams = len(teams)
        team_to_idx = {t: i for i, t in enumerate(teams)}

        home_idx = np.array([team_to_idx[t] for t in df["home_team"]])
        away_idx = np.array([team_to_idx[t] for t in df["away_team"]])
        
        home_xg = df["home_xg_proxy"].values
        away_xg = df["away_xg_proxy"].values
        home_goals = df["full_time_home_goals"].astype(int).values
        away_goals = df["full_time_away_goals"].astype(int).values

        weights = self._compute_weights(df)
        self._league_avg_goals = (home_xg.mean() + away_xg.mean()) / 2

        # Optimization
        n_params = 1 + (n_teams - 1) + n_teams
        if self.estimate_rho:
            n_params += 1

        def neg_log_likelihood(params):
            mu = params[0]
            
            attack = np.zeros(n_teams)
            attack[1:] = params[1:n_teams]
            defence = params[n_teams:2*n_teams]
            rho = params[2*n_teams] if self.estimate_rho else 0.0
            
            lambda_h = np.exp(mu + attack[home_idx] + defence[away_idx])
            lambda_a = np.exp(attack[away_idx] + defence[home_idx])
            
            lambda_h = np.clip(lambda_h, 0.1, 10.0)
            lambda_a = np.clip(lambda_a, 0.1, 10.0)
            
            ll = (
                home_xg * np.log(lambda_h) - lambda_h
                + away_xg * np.log(lambda_a) - lambda_a
            )
            
            if self.estimate_rho and abs(rho) > 0.001:
                for i in range(len(home_goals)):
                    hg, ag = int(home_goals[i]), int(away_goals[i])
                    if hg <= 1 and ag <= 1:
                        tau = _tau(hg, ag, lambda_h[i], lambda_a[i], rho)
                        if tau > 0:
                            ll[i] += np.log(tau)

            return -np.sum(weights * ll)

        x0 = np.zeros(n_params)
        x0[0] = 0.25

        bounds = [(0.05, 0.5)]
        bounds += [(None, None)] * (n_teams - 1)
        bounds += [(None, None)] * n_teams
        if self.estimate_rho:
            bounds.append((-0.2, 0.05))

        result = minimize(
            neg_log_likelihood,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 300, "ftol": 1e-6}
        )

        if not result.success:
            warnings.warn(f"Optimization warning: {result.message}")

        params = result.x
        self.mu_ = params[0]
        
        attack = np.zeros(n_teams)
        attack[1:] = params[1:n_teams]
        defence = params[n_teams:2*n_teams]
        
        self.attack_ = {t: attack[i] for i, t in enumerate(teams)}
        self.defence_ = {t: defence[i] for i, t in enumerate(teams)}
        
        if self.estimate_rho:
            self.rho_ = params[2*n_teams]

        self._league_avg_attack = np.mean(attack)
        self._league_avg_defence = np.mean(defence)

        self.n_matches_ = len(df)
        self.fit_date_ = str(df["match_date"].max().date())
        self.fitted = True

        return self

    def _get_form_adjustment(
        self,
        team: str,
        match_date: Optional[pd.Timestamp] = None
    ) -> Tuple[float, float]:
        """Get form-based multipliers for attack and defence."""
        if self._match_data is None or self.form_weight == 0:
            return 1.0, 1.0
        
        if match_date is None:
            match_date = pd.Timestamp.now()
        
        scored, conceded = calculate_form(
            team, 
            self._match_data, 
            match_date, 
            self.form_matches,
        )
        
        if scored is None:
            return 1.0, 1.0
        
        avg = self._league_avg_goals
        
        attack_form = np.clip(scored / avg, 0.7, 1.4) if avg > 0 else 1.0
        defence_form = np.clip(conceded / avg, 0.7, 1.4) if avg > 0 else 1.0
        
        attack_mult = (1 - self.form_weight) + self.form_weight * attack_form
        defence_mult = (1 - self.form_weight) + self.form_weight * defence_form
        
        return attack_mult, defence_mult

    def expected_goals(
        self,
        home: str,
        away: str,
        match_date: Optional[pd.Timestamp] = None,
        use_form: bool = True
    ) -> Tuple[float, float]:
        """Calculate expected goals for a fixture."""
        self._check_fitted()
        
        if home in self.attack_:
            att_h, def_h = self.attack_[home], self.defence_[home]
        else:
            warnings.warn(f"Unknown team '{home}', using league average")
            att_h, def_h = self._league_avg_attack, self._league_avg_defence
        
        if away in self.attack_:
            att_a, def_a = self.attack_[away], self.defence_[away]
        else:
            warnings.warn(f"Unknown team '{away}', using league average")
            att_a, def_a = self._league_avg_attack, self._league_avg_defence
        
        lambda_home = np.exp(self.mu_ + att_h + def_a)
        lambda_away = np.exp(att_a + def_h)
        
        if use_form:
            h_att_form, h_def_form = self._get_form_adjustment(home, match_date)
            a_att_form, a_def_form = self._get_form_adjustment(away, match_date)
            
            lambda_home *= h_att_form * a_def_form
            lambda_away *= a_att_form * h_def_form
        
        return lambda_home, lambda_away

    def score_prob_matrix(
        self,
        home: str,
        away: str,
        max_goals: int = 10,
        **kwargs
    ) -> np.ndarray:
        """Goal probability matrix."""
        lam_h, lam_a = self.expected_goals(home, away, **kwargs)
        return _goal_matrix(lam_h, lam_a, rho=self.rho_, max_goals=max_goals)

    def match_outcome_probs(self, home: str, away: str, **kwargs) -> Dict[str, float]:
        """Home win / draw / away win probabilities."""
        P = self.score_prob_matrix(home, away, **kwargs)
        return {
            "home_win": np.tril(P, -1).sum(),
            "draw": np.trace(P),
            "away_win": np.triu(P, 1).sum(),
        }

    def over_under_prob(
        self,
        home: str,
        away: str,
        line: float = 2.5,
        **kwargs
    ) -> Dict[str, float]:
        """Over/under probabilities."""
        P = self.score_prob_matrix(home, away, **kwargs)
        goals = np.add.outer(np.arange(P.shape[0]), np.arange(P.shape[1]))
        return {
            "over": P[goals > line].sum(),
            "under": P[goals <= line].sum(),
        }

    def bts_prob(self, home: str, away: str, **kwargs) -> float:
        """Both teams to score probability."""
        P = self.score_prob_matrix(home, away, **kwargs)
        return P[1:, 1:].sum()

    def predict_fixture(self, home: str, away: str, **kwargs) -> Dict[str, float]:
        """Full prediction for a fixture."""
        lam_h, lam_a = self.expected_goals(home, away, **kwargs)
        outcome = self.match_outcome_probs(home, away, **kwargs)
        ou = self.over_under_prob(home, away, **kwargs)
        bts = self.bts_prob(home, away, **kwargs)

        return {
            "exp_home_goals": lam_h,
            "exp_away_goals": lam_a,
            **outcome,
            "over_2_5": ou["over"],
            "under_2_5": ou["under"],
            "btts_yes": bts,
            "btts_no": 1 - bts,
        }

    def team_ratings(self) -> pd.DataFrame:
        """DataFrame of team ratings sorted by overall strength."""
        self._check_fitted()

        data = []
        for team in self.teams:
            att = self.attack_.get(team, 0)
            def_ = self.defence_.get(team, 0)
            data.append({
                "team": team,
                "attack": att,
                "defence": def_,
                "overall": att - def_,
            })

        return pd.DataFrame(data).sort_values("overall", ascending=False).reset_index(drop=True)

    def summary(self) -> str:
        """Model summary."""
        self._check_fitted()
        lines = [
            "Dixon-Coles Poisson Model (Shots + Form)",
            "-" * 45,
            f"Matches:          {self.n_matches_}",
            f"Teams:            {len(self.teams)}",
            f"Home advantage:   {self.mu_:.3f} ({np.exp(self.mu_):.2f}x)",
            f"DC rho:           {self.rho_:.4f}",
            f"Half-life:        {self.half_life_days} days",
            f"Form weight:      {self.form_weight:.0%}",
            f"Form matches:     {self.form_matches}",
            f"Fit date:         {self.fit_date_}",
        ]
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Save model to file."""
        self._check_fitted()
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "PoissonModel":
        """Load model from file."""
        with open(path, "rb") as f:
            model = pickle.load(f)
        if not isinstance(model, PoissonModel):
            raise TypeError("File does not contain a PoissonModel")
        return model

    def _check_fitted(self):
        if not self.fitted:
            raise RuntimeError("Model not fitted - call .fit() first")


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
               match_date
        FROM matches
        WHERE league = 'Premier League'
          AND season >= '2022-23'
        ORDER BY match_date
        """,
        conn,
    )
    print(f"Loaded {len(df)} matches\n")

    model = PoissonModel().fit(df)
    print(model.summary())
    print()

    print("Top 5 teams:")
    print(model.team_ratings().head())
    print()

    for h, a in [("Arsenal", "Chelsea"), ("Man City", "Liverpool")]:
        pred = model.predict_fixture(h, a)
        print(f"{h} vs {a}:")
        print(f"  xG: {pred['exp_home_goals']:.2f} - {pred['exp_away_goals']:.2f}")
        print(f"  1X2: H {pred['home_win']:.1%} | D {pred['draw']:.1%} | A {pred['away_win']:.1%}")
        print()

    print("Done")