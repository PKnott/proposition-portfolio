"""Expanding-window walk-forward cross-validation.

Never a random split. Fold ``k`` trains on every season strictly before season
``k`` and validates on season ``k``, so the model is only ever asked to predict
forward in time -- the same discipline the rest of the pipeline depends on.

The held-out test seasons (``>= TEST_SEASON``) are excluded here entirely; they
are scored exactly once, at the end, by the Evaluation notebook.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .build import FeatureTable, scorable_mask
from .config import (
    PROBE,
    FitConfig,
    first_val_season,
    holdout_season,
    season_sort_key,
    test_season,
)
from .models import (
    fit_dispersion_by_league,
    fit_model,
    pair_predictions,
    predict_mu,
    score_team_logloss,
)


@dataclass(frozen=True)
class Fold:
    val_season: str
    train: np.ndarray  # boolean row mask
    val: np.ndarray


def _years(ft: FeatureTable) -> np.ndarray:
    """Each row's season as the year it starts in.

    Every boundary in this module is a comparison between seasons, and they used
    to be string comparisons. The clean table mixes three label shapes --
    ``'2014/2015'``, ``'1920'`` (Ligue 1's curtailed 2019/20) and ``'2526'`` --
    and lexically ``'1920'`` sorts *before* ``'2014/2015'``. So 2019/20 counted as
    the earliest season in the data: it could land in the training set for a
    validation season five years its junior, and it could never be a validation
    season itself.

    That never fired, because `first_val_season()` is 2020/21 and the folds that
    would have exposed it are not built. It was one config change away from
    firing, which is not a property worth relying on.
    """
    return ft.meta["season"].map(season_sort_key).to_numpy()


def dev_mask(ft: FeatureTable) -> np.ndarray:
    """Rows before the sacred test boundary."""
    return _years(ft) < season_sort_key(test_season())


def test_mask(ft: FeatureTable) -> np.ndarray:
    return _years(ft) >= season_sort_key(test_season())


def season_folds(
    ft: FeatureTable,
    first_val: str | None = None,
    last_val_season: str | None = None,
) -> list[Fold]:
    """One fold per validation season, training on everything earlier.

    ``first_val`` is resolved from the config at call time rather than in the
    signature -- a default argument is evaluated once at import and would pin the
    boundary for the life of the process.
    """
    first_val = first_val or first_val_season()
    season = ft.meta["season"].to_numpy()
    year = _years(ft)
    dev = dev_mask(ft)
    seasons = sorted(set(season[dev]), key=season_sort_key)
    first_year = season_sort_key(first_val)
    last_year = (season_sort_key(last_val_season) if last_val_season
                 else (season_sort_key(seasons[-1]) if seasons else None))

    folds: list[Fold] = []
    for s in seasons:
        y = season_sort_key(s)
        if y < first_year or (last_year is not None and y > last_year):
            continue
        # Strictly earlier *seasons*, by the year they start in. Two labels for
        # one real season (Ligue 1's '1920' and everyone else's '2019/2020')
        # therefore share a year and neither trains on the other.
        train = dev & (year < y)
        val = dev & (year == y)
        if train.sum() == 0 or val.sum() == 0:
            continue
        folds.append(Fold(val_season=s, train=train, val=val))
    return folds


def tuning_folds(ft: FeatureTable) -> list[Fold]:
    """Folds `03_Tuning` may search against -- everything up to the holdout.

    The most recent development season is withheld so that there is something to
    measure a tuning run *against*. Without it the only honest read on whether
    extra dials helped or overfitted is the test season, and that can be spent
    once.
    """
    last = season_sort_key(holdout_season())
    seasons = sorted(set(ft.meta["season"].to_numpy()[dev_mask(ft)]), key=season_sort_key)
    before = [s for s in seasons if season_sort_key(s) < last]
    if not before:
        raise ValueError(
            f"holdout season {holdout_season()!r} leaves no tuning folds; "
            f"development seasons are {seasons}"
        )
    return season_folds(ft, last_val_season=before[-1])


def holdout_fold(ft: FeatureTable) -> Fold | None:
    """The single withheld fold, scored once with the settled configuration.

    Returns ``None`` when the table has no such season, so a caller working on a
    slice of the data degrades to "no holdout read" rather than raising.
    """
    for fold in season_folds(ft):
        if season_sort_key(fold.val_season) == season_sort_key(holdout_season()):
            return fold
    return None


@dataclass
class CVResult:
    target: str
    mean: float
    se: float
    per_fold: dict[str, float]
    # Scoring rows, not fixtures: per-team scoring contributes two rows per
    # fixture. Named for what it counts so printed diagnostics do not look
    # inconsistent with what was actually scored.
    n_scored: int
    best_iterations: list[int]
    dispersion: dict[str, float]


def evaluate(
    ft: FeatureTable,
    target: str,
    params: dict,
    folds: list[Fold],
    *,
    features: list[str] | None = None,
    fit_cfg: FitConfig = PROBE,
) -> CVResult:
    """Walk-forward CV for one target family. Score is bucketed per-team log loss.

    The single choke point for the metric: every search stage, the window grid,
    the hyperparameter descent, the safety-net check and the anchor all reach the
    score through here rather than computing it themselves, so the scoring call
    below is the only place the objective is chosen.
    """
    X = ft.X[list(features)] if features else ft.X
    y = ft.y[target]
    usable = scorable_mask(ft, target)

    per_fold: dict[str, float] = {}
    best_iters: list[int] = []
    n_total = 0
    all_paired = []

    for fold in folds:
        tr = fold.train & usable
        va = fold.val & usable
        if tr.sum() == 0 or va.sum() == 0:
            continue

        model = fit_model(
            X.loc[tr], y[tr], params,
            eval_set=(X.loc[va], y[va]),
            early_stopping_rounds=fit_cfg.early_stopping_rounds,
        )
        best = getattr(model, "best_iteration", None)
        if best is not None:
            best_iters.append(int(best))

        sub = ft.mask(va)
        paired = pair_predictions(sub, predict_mu(model, X.loc[va]), target)
        if paired.empty:
            continue
        all_paired.append(paired)

    if not all_paired:
        return CVResult(target, float("nan"), float("nan"), {}, 0, [], {})

    # Dispersion is fitted on the validation predictions -- never on training data,
    # and never on the test seasons.
    import pandas as pd

    pooled = pd.concat(all_paired, ignore_index=True)
    dispersion = fit_dispersion_by_league(pooled)

    # Counted off the scoring array rather than off `len(paired)`, so the number
    # reported is whatever the metric in force actually scored -- two rows per
    # fixture today, and still correct if that ever changes.
    for paired in all_paired:
        ll = score_team_logloss(paired, target, dispersion)
        n_total += len(ll)
        per_fold[str(paired["season"].iloc[0])] = float(np.mean(ll))

    vals = np.array(list(per_fold.values()), dtype=float)
    mean = float(vals.mean())
    se = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else float("nan")

    return CVResult(
        target=target, mean=mean, se=se, per_fold=per_fold,
        n_scored=n_total, best_iterations=best_iters, dispersion=dispersion,
    )


def production_n_estimators(best_iterations: list[int], default: int = 400) -> int:
    """Tree count for a final fit with no validation fold to stop against.

    Standard refit-on-all-data adjustment: the median stopping point, scaled up
    a little because the production fit sees more data than any CV fold did.
    """
    if not best_iterations:
        return default
    return int(1.1 * float(np.median(best_iterations))) + 1


__all__ = [
    "Fold", "CVResult", "dev_mask", "test_mask", "season_folds",
    "tuning_folds", "holdout_fold",
    "evaluate", "production_n_estimators",
]
