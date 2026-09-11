"""Held-out test scoring, significance, and per-league segments.

The test seasons are touched exactly once, here, after everything else is frozen.
Nothing in this module may be used to make a modelling decision -- that is the
whole point of having held them back.

Calibration lives in `fpp.calibration`, not here. It used to be a single function
in this module covering match-total lines on ten equal-width bins; it now needs
per-team and per-segment streams, quantile bands, interval tests and a verdict,
which is a module's worth of work rather than a function's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .build import FeatureTable, scorable_mask
from .config import FULL, FitConfig
from .cv import dev_mask, test_mask
from .metrics import bootstrap_ci
from .models import (
    fit_dispersion_by_league,
    fit_model,
    pair_predictions,
    predict_mu,
    score_team_logloss,
    score_total_logloss,
)
from .spec import target_spec


def fit_predict_test(
    ft: FeatureTable,
    target: str,
    params: dict,
    features: list[str],
    *,
    fit_cfg: FitConfig = FULL,
    dispersion: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Train once on dev seasons, score every fixture in the test seasons."""
    usable = scorable_mask(ft, target)
    tr = dev_mask(ft) & usable
    te = test_mask(ft) & usable
    if tr.sum() == 0 or te.sum() == 0:
        raise ValueError(f"{target}: empty train or test split")

    X = ft.X[features]
    p = dict(params)
    p.pop("early_stopping_rounds", None)
    model = fit_model(X.loc[tr], ft.y[target][tr], p)

    paired = pair_predictions(ft.mask(te), predict_mu(model, X.loc[te]), target)
    disp = dispersion if dispersion is not None else fit_dispersion_by_league(paired)
    return paired, disp


def baseline_test(ft: FeatureTable, target: str) -> pd.DataFrame:
    """Per-league mean rate learned on dev, applied to test."""
    tr = dev_mask(ft) & scorable_mask(ft, target)
    te = test_mask(ft) & scorable_mask(ft, target)

    m = ft.meta.copy()
    m["actual"] = ft.y[target]
    rates = m.loc[tr].groupby("league_key", observed=True)["actual"].mean().to_dict()

    sub = ft.mask(te)
    mu = sub.meta["league_key"].map(rates).to_numpy(dtype=float)
    return pair_predictions(sub, mu, target)


# --- The two metrics ------------------------------------------------------
#
# These answer genuinely different questions and are never interchangeable:
#
#   per-team    -- how good is the model at predicting each side? The training
#                  and selection objective (`cv.evaluate`).
#   match-total -- how good is the priced match total the product delivers?
#
# They are not comparable to each other: different bucket grids (`team_cap` vs
# `total_cap`) and a different number of scoring rows. Every function below comes
# in a `_team` and a `_total` form so a number can never be printed, stored or
# compared without its metric being named. The same discipline as
# `team_attack` vs `team_attack_merged` -- two different things never share one
# name.

_SCORERS = {"team": score_team_logloss, "total": score_total_logloss}


def _errors(paired: pd.DataFrame, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Predicted and actual, on the same footing as the log loss for ``mode``."""
    if mode == "team":
        pred = np.concatenate([paired["mu_home"].to_numpy(), paired["mu_away"].to_numpy()])
        actual = np.concatenate([paired["actual_home"].to_numpy(), paired["actual_away"].to_numpy()])
    else:
        pred = paired["mu_home"].to_numpy() + paired["mu_away"].to_numpy()
        actual = paired["actual_home"].to_numpy() + paired["actual_away"].to_numpy()
    return pred, actual


def _headline(paired: pd.DataFrame, target: str, dispersion, label: str, mode: str) -> dict:
    ll = _SCORERS[mode](paired, target, dispersion)
    pred, actual = _errors(paired, mode)
    return {
        "model": label, "target": target, "metric": mode,
        "n_fixtures": len(paired), "n_scored": len(ll),
        "logloss": float(np.mean(ll)),
        "logloss_se": float(np.std(ll, ddof=1) / np.sqrt(len(ll))) if len(ll) > 1 else float("nan"),
        "mae": float(np.mean(np.abs(pred - actual))),
        "rmse": float(np.sqrt(np.mean((pred - actual) ** 2))),
    }


def headline_total(paired: pd.DataFrame, target: str, dispersion, label: str) -> dict:
    """Match-total log loss, MAE and RMSE. What the priced product delivers."""
    return _headline(paired, target, dispersion, label, "total")


def headline_team(paired: pd.DataFrame, target: str, dispersion, label: str) -> dict:
    """Per-team log loss, MAE and RMSE -- each side against its own actual.

    ``mae``/``rmse`` stack home and away rather than comparing match totals, so
    every number in the row refers to the same thing the log loss does.
    """
    return _headline(paired, target, dispersion, label, "team")


def _significance(model_paired: pd.DataFrame, base_paired: pd.DataFrame, target: str,
                  dispersion, base_dispersion, alpha: float, mode: str) -> dict:
    """Paired bootstrap CI + Wilcoxon on the log-loss edge.

    The two tests answer different questions and can legitimately disagree. The
    bootstrap CI is about the **mean** of the paired differences; Wilcoxon is
    about the **median**. Log-loss differences are right-skewed -- near zero on
    most fixtures, large on the minority where the model calls an unusual
    scoreline -- so the mean can sit clearly above zero while the median does not.

    A single ``significant`` flag driven by one of them would misreport that. So
    both are surfaced, ``significant`` requires them to agree, and ``median_gain``
    / ``win_rate`` are reported because they are what make a mean-versus-median
    split legible: a positive mean with a ~50% win rate means the edge lives in
    the tail rather than being a consistent shift.

    Pairing is on ``fixture_id`` for both modes. Under per-team scoring each
    fixture yields two aligned scoring rows (home then away, same order on both
    sides), so the differences stay paired row-for-row.
    """
    m = model_paired.set_index("fixture_id")
    b = base_paired.set_index("fixture_id")
    common = m.index.intersection(b.index)
    m, b = m.loc[common].reset_index(), b.loc[common].reset_index()

    score = _SCORERS[mode]
    ll_m = score(m, target, dispersion)
    ll_b = score(b, target, base_dispersion if base_dispersion is not None else dispersion)
    diff = ll_b - ll_m  # positive => model beats baseline on that scoring row

    lo, hi = bootstrap_ci(diff)
    try:
        _, p = wilcoxon(ll_b, ll_m)
    except ValueError:
        p = float("nan")

    ci_excludes_zero = bool(lo > 0)
    wilcoxon_significant = bool(p < alpha) if np.isfinite(p) else False

    return {
        "target": target, "metric": mode, "n": len(diff),
        "mean_gain": float(diff.mean()),
        "median_gain": float(np.median(diff)),
        "win_rate": float((diff > 0).mean()),
        "ci_lo": lo, "ci_hi": hi,
        "wilcoxon_p": float(p),
        "ci_excludes_zero": ci_excludes_zero,
        "wilcoxon_significant": wilcoxon_significant,
        # Conservative: an edge only counts when mean and median agree.
        "significant": bool(ci_excludes_zero and wilcoxon_significant),
        "tests_disagree": bool(ci_excludes_zero != wilcoxon_significant),
    }


def significance_total(model_paired: pd.DataFrame, base_paired: pd.DataFrame, target: str,
                       dispersion, base_dispersion=None, alpha: float = 0.05) -> dict:
    """Is the match-total edge over the baseline real?"""
    return _significance(model_paired, base_paired, target, dispersion,
                         base_dispersion, alpha, "total")


def significance_team(model_paired: pd.DataFrame, base_paired: pd.DataFrame, target: str,
                      dispersion, base_dispersion=None, alpha: float = 0.05) -> dict:
    """Is the per-team edge over the baseline real?"""
    return _significance(model_paired, base_paired, target, dispersion,
                         base_dispersion, alpha, "team")


def describe_significance(row: dict | pd.Series) -> str:
    """One-line plain reading of a significance row, including disagreements.

    The metric is named in the label, never left to be inferred: the per-team and
    match-total rows are the same shape and would otherwise be indistinguishable
    once printed.
    """
    t = f"{row['target']} [{row.get('metric', '?')}]"
    if row["significant"]:
        return (f"{t}: edge of {row['mean_gain']:+.5f} confirmed by both tests "
                f"(CI [{row['ci_lo']:+.5f}, {row['ci_hi']:+.5f}], Wilcoxon p={row['wilcoxon_p']:.3g})")
    if row["tests_disagree"] and row["ci_excludes_zero"]:
        return (f"{t}: mean gain {row['mean_gain']:+.5f} excludes zero, but the median gain is "
                f"{row['median_gain']:+.5f} and the model wins only {100 * row['win_rate']:.1f}% of "
                f"fixtures (Wilcoxon p={row['wilcoxon_p']:.3g}). The edge sits in a minority of "
                f"fixtures rather than being a consistent shift -- not counted as significant.")
    if row["tests_disagree"]:
        return (f"{t}: Wilcoxon significant (p={row['wilcoxon_p']:.3g}) but the mean-gain CI "
                f"[{row['ci_lo']:+.5f}, {row['ci_hi']:+.5f}] includes zero -- a consistent but tiny "
                f"edge, offset by occasional large losses. Not counted as significant.")
    return (f"{t}: no edge distinguishable from zero "
            f"(CI [{row['ci_lo']:+.5f}, {row['ci_hi']:+.5f}], Wilcoxon p={row['wilcoxon_p']:.3g})")


def _by_league(model_paired: pd.DataFrame, base_paired: pd.DataFrame, target: str,
               dispersion, base_dispersion, mode: str) -> pd.DataFrame:
    """Per-league breakdown -- the pooling sanity check.

    A league that is materially worse here than a dedicated model would be is a
    signal to investigate, not just a line in a report.
    """
    rows = []
    for lk in sorted(model_paired["league_key"].unique()):
        m = model_paired[model_paired["league_key"] == lk]
        b = base_paired[base_paired["league_key"] == lk]
        if m.empty:
            continue
        hm = _headline(m, target, dispersion, "model", mode)
        # An empty baseline slice would otherwise reach `np.mean([])` and come
        # back as NaN plus a RuntimeWarning, with `edge` quietly NaN too. Say the
        # baseline is missing instead of computing a number that is not one.
        hb = (_headline(b, target, base_dispersion if base_dispersion is not None else dispersion,
                        "baseline", mode)
              if not b.empty else None)
        rows.append({
            "league": lk, "metric": mode, "n": len(m),
            "model_logloss": hm["logloss"],
            "baseline_logloss": hb["logloss"] if hb else float("nan"),
            "edge": (hb["logloss"] - hm["logloss"]) if hb else float("nan"),
            "model_mae": hm["mae"],
        })
    return pd.DataFrame(rows)


def by_league_total(model_paired: pd.DataFrame, base_paired: pd.DataFrame, target: str,
                    dispersion, base_dispersion=None) -> pd.DataFrame:
    """Per-league match-total breakdown."""
    return _by_league(model_paired, base_paired, target, dispersion, base_dispersion, "total")


def by_league_team(model_paired: pd.DataFrame, base_paired: pd.DataFrame, target: str,
                   dispersion, base_dispersion=None) -> pd.DataFrame:
    """Per-league per-team breakdown."""
    return _by_league(model_paired, base_paired, target, dispersion, base_dispersion, "team")


__all__ = [
    "fit_predict_test", "baseline_test",
    "headline_team", "headline_total",
    "significance_team", "significance_total",
    "by_league_team", "by_league_total",
    "describe_significance",
]
