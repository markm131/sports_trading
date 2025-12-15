# src/models/poisson.py
"""
Poisson goal-model for football (soccer)

The model assumes
    log(λ_home) =   μ              + A_home  + D_away
    log(λ_away) =             A_away + D_home

where
    λ_home / λ_away  : expected goals
    μ               : global home-advantage
    A_team          : attacking strength of `team`
    D_team          : defensive strength of `team`

Fitting is done with two independent Poisson GLMs.  Identifiability is
guaranteed by mean-centring the attack parameters (∑A = 0) which in turn
forces the defence parameters to centre around 0 as well.

Example
-------
>>> from src.models.poisson import PoissonModel
>>> from src.data.cleaner import clean_combined_file
>>> df = clean_combined_file("premier_league")
>>> m  = PoissonModel().fit(df)
>>> m.predict_fixture("Arsenal", "Chelsea")
{'exp_home_goals': 1.72, 'exp_away_goals': 1.09,
 'home_win': 0.46, 'draw': 0.26, 'away_win': 0.28}
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------


def _goal_matrix(
    lambda_home: float, lambda_away: float, max_goals: int = 10
) -> np.ndarray:
    """
    Return a (max_goals+1) x (max_goals+1) matrix with probabilities for every
    exact score from 0-0 up to max_goals-max_goals assuming *independent*
    Poisson goal counts.
    """
    hg = np.arange(0, max_goals + 1)[:, None]  # shape (g+1,1)
    ag = np.arange(0, max_goals + 1)[None, :]  # shape (1,g+1)

    # P(X=x) = e^{-λ} λ^x / x!
    p_home = np.exp(-lambda_home) * lambda_home**hg / np.vectorize(np.math.factorial)(hg)
    p_away = np.exp(-lambda_away) * lambda_away**ag / np.vectorize(np.math.factorial)(ag)

    return p_home * p_away  # outer product


# ---------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------


class PoissonModel:
    """Attack-defence Poisson model with convenient helper methods."""

    def __init__(self):
        self.fitted: bool = False
        self.teams: list[str] = []
        self.mu_: float | None = None  # home advantage
        self.attack_: Dict[str, float] = {}
        self.defence_: Dict[str, float] = {}

        # Keep underlying GLMs if you want to inspect them
        self._glm_home = None
        self._glm_away = None

    # -----------------------------------------------------------------
    # Fitting / persistence
    # -----------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "PoissonModel":
        """
        Fit the model on a DataFrame containing at minimum
            ['home_team', 'away_team', 'full_time_home_goals', 'full_time_away_goals']
        """
        required = {
            "home_team",
            "away_team",
            "full_time_home_goals",
            "full_time_away_goals",
        }
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        # Build design matrices ------------------------------------------------
        teams = pd.concat([df.home_team, df.away_team]).unique().tolist()
        teams.sort()
        self.teams = teams

        # One-hot attack/defence for home-goals
        X_home = pd.get_dummies(df.home_team)[teams]  # attack terms
        X_home += pd.get_dummies(df.away_team, prefix="", prefix_sep="")[teams] * 0  # ensure same cols
        X_home.rename(columns=lambda t: f"att_{t}", inplace=True)

        X_def_away = pd.get_dummies(df.away_team)[teams]
        X_def_away.rename(columns=lambda t: f"def_{t}", inplace=True)

        Xh = pd.concat([X_home, X_def_away], axis=1)
        Xh["intercept"] = 1  # home advantage

        # Equivalent for away goals (note: no intercept)
        X_attack_away = pd.get_dummies(df.away_team)[teams]
        X_attack_away.rename(columns=lambda t: f"att_{t}", inplace=True)

        X_def_home = pd.get_dummies(df.home_team)[teams]
        X_def_home.rename(columns=lambda t: f"def_{t}", inplace=True)

        Xa = pd.concat([X_attack_away, X_def_home], axis=1)
        Xa["intercept"] = 0  # no home advantage

        # Responses
        y_home = df.full_time_home_goals
        y_away = df.full_time_away_goals

        # Fit GLMs -------------------------------------------------------------
        poisson = sm.families.Poisson()
        self._glm_home = sm.GLM(y_home, Xh, family=poisson).fit()
        self._glm_away = sm.GLM(y_away, Xa, family=poisson).fit()

        # Extract parameters ---------------------------------------------------
        coef_home = self._glm_home.params
        coef_away = self._glm_away.params

        # Home advantage
        self.mu_ = coef_home["intercept"]

        # Attack strengths: take value from *either* model (they are estimated twice)
        self.attack_ = {
            t: coef_home.get(f"att_{t}", 0.0) for t in self.teams
        }  # could also average the two

        # Defence strengths: take avg of both models for stability
        self.defence_ = {
            t: (coef_home.get(f"def_{t}", 0.0) + coef_away.get(f"def_{t}", 0.0)) / 2
            for t in self.teams
        }

        # Centre attack parameters (identifiability) ---------------------------
        mean_att = np.mean(list(self.attack_.values()))
        for t in self.teams:
            self.attack_[t] -= mean_att
            # corresponding shift in defence to keep expected goals unchanged
            self.defence_[t] += mean_att

        self.fitted = True
        return self

    # -----------------------------------------------------------------
    # Prediction helpers
    # -----------------------------------------------------------------

    def expected_goals(self, home: str, away: str) -> Tuple[float, float]:
        """
        Return λ_home, λ_away for a fixture.
        """
        self._check_fitted()
        if home not in self.attack_ or away not in self.attack_:
            raise ValueError("Unknown team(s)")

        lamb_home = np.exp(self.mu_ + self.attack_[home] + self.defence_[away])
        lamb_away = np.exp(self.attack_[away] + self.defence_[home])
        return lamb_home, lamb_away

    def score_prob_matrix(
        self, home: str, away: str, max_goals: int = 10
    ) -> np.ndarray:
        """
        Matrix P(i,j) = P(home=i, away=j), 0<=i,j<=max_goals
        """
        lam_h, lam_a = self.expected_goals(home, away)
        return _goal_matrix(lam_h, lam_a, max_goals=max_goals)

    # -----------------------------------------------------------------
    # Aggregate probabilities
    # -----------------------------------------------------------------

    def match_outcome_probs(self, home: str, away: str) -> Dict[str, float]:
        """
        Returns probabilities for home win / draw / away win.
        """
        P = self.score_prob_matrix(home, away)
        home_win = np.tril(P, -1).sum()
        draw = np.trace(P)
        away_win = np.triu(P, 1).sum()
        return {"home_win": home_win, "draw": draw, "away_win": away_win}

    def over_under_prob(self, home: str, away: str, line: float = 2.5) -> Dict[str, float]:
        """
        Probability of total goals > line and <= line.
        """
        P = self.score_prob_matrix(home, away)
        goals = np.add.outer(np.arange(P.shape[0]), np.arange(P.shape[1]))
        over = P[goals > line].sum()
        under = P[goals <= line].sum()
        return {"over": over, "under": under}

    def bts_prob(self, home: str, away: str) -> float:
        """
        Both teams to score probability.
        """
        P = self.score_prob_matrix(home, away)
        bts = P[1:, 1:].sum()
        return bts

    def predict_fixture(self, home: str, away: str) -> Dict[str, float]:
        """
        Convenience wrapper returning exp goals + 1X2 probs.
        """
        lam_h, lam_a = self.expected_goals(home, away)
        outcome = self.match_outcome_probs(home, away)
        return {
            "exp_home_goals": lam_h,
            "exp_away_goals": lam_a,
            **outcome,
        }

    # -----------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Pickle the fitted model."""
        self._check_fitted()
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "PoissonModel":
        with open(path, "rb") as f:
            model = pickle.load(f)
        if not isinstance(model, PoissonModel):
            raise TypeError("File does not contain a PoissonModel")
        return model

    # -----------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------

    def _check_fitted(self):
        if not self.fitted:
            raise RuntimeError("Model not yet fitted – call .fit(df) first.")


# ---------------------------------------------------------------------
# Quick demo / manual test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Minimal self-test with whatever data is available in the DB
    from src.data.db_writer import get_db_connection

    conn = get_db_connection()
    df = pd.read_sql(
        """
        SELECT home_team, away_team, full_time_home_goals, full_time_away_goals
        FROM matches
        WHERE season >= '2022-23'
        """,
        conn,
    )
    print(f"Loaded {len(df)} matches from DB")

    model = PoissonModel().fit(df)
    print("Fitted model ✓\n")

    for fixture in [
        ("Manchester United", "Liverpool"),
        ("Arsenal", "Chelsea"),
        ("Leeds", "Norwich"),
    ]:
        h, a = fixture
        print(f"{h} vs {a}:")
        res = model.predict_fixture(h, a)
        print(
            "  λ_home={exp_home_goals:.2f}, λ_away={exp_away_goals:.2f} "
            "| H:{home_win:.2%} D:{draw:.2%} A:{away_win:.2%}".format(**res)
        )
    print("\nAll good 👍")