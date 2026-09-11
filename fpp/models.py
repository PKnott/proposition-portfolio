"""Fitting and scoring one target family.

Every target family is modelled **independently**: its own features, its own
hyperparameters, its own dispersion. There is no joint output distribution --
shots on target is not modelled as a share of shots, and corners has no
containment relationship to anything. What they do share is the feature pipeline,
so each family may use the others' rolling priors as candidate inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from .build import FeatureTable, scorable_mask
from .config import FULL, PROBE, RANDOM_SEED, FitConfig
from .metrics import convolve_pmf, fit_nb_dispersion, logloss, pmf_for
from .spec import target_spec


def base_params(target: str, fit: FitConfig = PROBE, **overrides) -> dict:
    """Starting hyperparameters for a target family."""
    s = target_spec(target)
    p = {
        "objective": s.objective,
        "n_estimators": fit.n_estimators,
        "learning_rate": fit.learning_rate,
        "max_depth": 3,
        "max_bin": fit.max_bin,
        "enable_categorical": True,
        "random_state": RANDOM_SEED,
        "verbosity": 0,
        "nthread": fit.nthread,
    }
    if s.objective == "reg:tweedie":
        p["tweedie_variance_power"] = 1.5
    p.update(overrides)
    return p


def fit_model(
    X: pd.DataFrame,
    y: np.ndarray,
    params: dict,
    *,
    eval_set: tuple[pd.DataFrame, np.ndarray] | None = None,
    early_stopping_rounds: int | None = None,
) -> xgb.XGBRegressor:
    """Fit, optionally with early stopping against a validation fold."""
    kwargs = dict(params)
    if early_stopping_rounds and eval_set is not None:
        kwargs["early_stopping_rounds"] = early_stopping_rounds
    model = xgb.XGBRegressor(**kwargs)
    if eval_set is not None:
        model.fit(X, y, eval_set=[(eval_set[0], eval_set[1])], verbose=False)
    else:
        model.fit(X, y, verbose=False)
    return model


def predict_mu(model: xgb.XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    """Predicted conditional mean, floored away from zero."""
    return np.clip(model.predict(X), 1e-8, None)


# --- Per-fixture scoring --------------------------------------------------


def pair_predictions(
    ft: FeatureTable, mu: np.ndarray, target: str, *, require_actual: bool = True
) -> pd.DataFrame:
    """Re-pair per-team predictions into one row per fixture.

    ``require_actual`` drops fixtures whose observed value is missing on either
    side -- shots/SOT/corners are absent for a handful of fixtures where the ESPN
    join failed, and a fixture with no outcome cannot be scored. Set it False when
    pairing pure predictions (no actuals exist yet).
    """
    m = ft.meta.copy()
    m["mu"] = mu
    m["actual"] = ft.y[target]

    home = m[m["is_home"] == 1].set_index("fixture_id")
    away = m[m["is_home"] == 0].set_index("fixture_id")
    common = home.index.intersection(away.index)

    if require_actual:
        ok = home.loc[common, "actual"].notna().to_numpy() & away.loc[common, "actual"].notna().to_numpy()
        common = common[ok]

    return pd.DataFrame({
        "fixture_id": common,
        "league_key": home.loc[common, "league_key"].to_numpy(),
        "season": home.loc[common, "season"].to_numpy(),
        "date": home.loc[common, "date"].to_numpy(),
        "home_team": home.loc[common, "team"].to_numpy(),
        "away_team": away.loc[common, "team"].to_numpy(),
        "mu_home": home.loc[common, "mu"].to_numpy(),
        "mu_away": away.loc[common, "mu"].to_numpy(),
        "actual_home": home.loc[common, "actual"].to_numpy(),
        "actual_away": away.loc[common, "actual"].to_numpy(),
    }).reset_index(drop=True)


def total_pmfs(
    paired: pd.DataFrame, target: str, dispersion: dict[str, float] | float | None = None
) -> list[np.ndarray]:
    """Match-total pmf per fixture: convolution of the two teams' distributions.

    For the overdispersed families a dispersion is required. If none is supplied
    we fit one from ``paired`` itself, so callers that only want a quick score
    (a baseline, a diagnostic) do not have to thread one through.
    """
    s = target_spec(target)
    cap = s.total_cap
    if s.dist == "nbinom" and dispersion is None:
        dispersion = fit_dispersion_by_league(paired)

    out = []
    for r in paired.itertuples():
        disp = dispersion.get(r.league_key) if isinstance(dispersion, dict) else dispersion
        ph = pmf_for(s.dist, r.mu_home, cap, disp)
        pa = pmf_for(s.dist, r.mu_away, cap, disp)
        out.append(convolve_pmf(ph, pa, cap))
    return out


def score_total_logloss(
    paired: pd.DataFrame, target: str, dispersion: dict[str, float] | float | None = None
) -> np.ndarray:
    """Per-fixture log loss of the bucketed match total.

    One value per fixture. This is what the *priced product* delivers, so it is
    the number to judge the workbook by -- but it is no longer what training and
    selection optimise. See ``score_team_logloss``.
    """
    cap = target_spec(target).total_cap
    pmfs = total_pmfs(paired, target, dispersion)
    totals = paired["actual_home"].to_numpy() + paired["actual_away"].to_numpy()
    return np.array([logloss(p, t, cap) for p, t in zip(pmfs, totals)])


def score_team_logloss(
    paired: pd.DataFrame, target: str, dispersion: dict[str, float] | float | None = None
) -> np.ndarray:
    """Per-team log loss: each side scored against its own actual.

    The training/selection objective. The model already *fits* per team --
    ``y = ft.y[target]`` in ``cv.evaluate`` is one team's own goal count -- so
    this stops the scoring step convolving the two sides back into a match total
    before comparing to reality. Nothing about model fitting changes.

    Returns **two values per fixture**, home then away, so a fold contributes
    roughly twice the scoring rows it did under the match-total metric. That is
    genuinely more data per fold, not the same data counted twice: the two sides
    are separate observations with separate predictions.

    Uses ``team_cap``, not ``total_cap`` -- one team's range, not the match's.
    """
    s = target_spec(target)
    cap = s.team_cap
    if s.dist == "nbinom" and dispersion is None:
        dispersion = fit_dispersion_by_league(paired)

    out = []
    for r in paired.itertuples():
        disp = dispersion.get(r.league_key) if isinstance(dispersion, dict) else dispersion
        out.append(logloss(pmf_for(s.dist, r.mu_home, cap, disp), r.actual_home, cap))
        out.append(logloss(pmf_for(s.dist, r.mu_away, cap, disp), r.actual_away, cap))
    return np.array(out)


def fit_dispersion_by_league(paired: pd.DataFrame) -> dict[str, float]:
    """Per-league Negative Binomial dispersion, from per-team residuals."""
    out: dict[str, float] = {}
    for lk, g in paired.groupby("league_key", observed=True):
        y = np.concatenate([g["actual_home"].to_numpy(), g["actual_away"].to_numpy()])
        mu = np.concatenate([g["mu_home"].to_numpy(), g["mu_away"].to_numpy()])
        out[lk] = fit_nb_dispersion(y, mu)
    return out


# --- Baseline -------------------------------------------------------------


def league_mean_baseline(train_ft: FeatureTable, test_ft: FeatureTable, target: str) -> pd.DataFrame:
    """Per-league mean rate -- naive by design, and a genuinely hard bar for goals.

    Deliberately fitted *per league* even though the model is pooled: "above
    average" has to mean above that league's own average.
    """
    tr = train_ft.meta.copy()
    tr["actual"] = train_ft.y[target]
    rates = tr.groupby("league_key", observed=True)["actual"].mean().to_dict()

    mu = test_ft.meta["league_key"].map(rates).to_numpy(dtype=float)
    return pair_predictions(test_ft, mu, target)


__all__ = [
    "base_params", "fit_model", "predict_mu", "pair_predictions",
    "total_pmfs", "score_total_logloss", "score_team_logloss", "fit_dispersion_by_league",
    "league_mean_baseline", "PROBE", "FULL",
]
