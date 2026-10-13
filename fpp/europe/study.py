"""Does the European layer earn its place? Walk-forward first, holdout once.

Steps 2 and 3 of the plan, as one driver:

* **offset only** -- the league model with the opponent at par. If this does not
  beat the European venue average, the league priors carry nothing into Europe
  and the experiment stops.
* **layer** -- league exchange rates, competition and stage on top. It has to beat
  the offset alone, not just the venue average.
* **layer + euroform** -- recent European results. Kept only if it beats the layer.

Walk-forward over European seasons, each scored by a fit on the seasons before it,
is the primary evidence: the 2025/26 holdout is a couple of hundred team-rows per
target and cannot carry a verdict alone. `holdout` scores it, once, and records
that it has done so.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

from ..artifacts import RunContext
from ..clean import load_clean_table
from ..config import test_season, season_sort_key
from ..evaluate import describe_significance
from ..spec import TARGETS
from . import ingest, layer, model
from .config import EURO_ARTIFACTS

WALK_FORWARD = list(range(2018, 2025))   # 2018/19 .. 2024/25
ALPHAS = (0.001, 0.01, 0.1)


def load_offsets(targets=TARGETS) -> dict[str, pd.DataFrame]:
    return model.offsets(ingest.load_match_stats(), load_clean_table(), targets=targets)


def _row(pred, t, label, a, b) -> dict:
    s = layer.significance(pred, t, a, b)
    return {"target": t, "comparison": label, "n": s["n"], "mean_gain": s["mean_gain"],
            "ci": (round(s["ci_lo"], 4), round(s["ci_hi"], 4)), "win_rate": s["win_rate"],
            "wilcoxon_p": s["wilcoxon_p"], "significant": s["significant"],
            "reading": describe_significance(s)}


def walk_forward(offs: dict[str, pd.DataFrame], seasons=WALK_FORWARD, alphas=ALPHAS) -> dict:
    """Log loss per variant and the significance of each step, per target."""
    scores, tests, thetas = [], [], {}
    for t, rows in offs.items():
        for a in alphas:
            for form in (False, True):
                pred = layer.walk_forward(rows, t, seasons, alpha=a, form=form)["rows"]
                sc = layer.score(pred, t).assign(target=t, alpha=a, form=form)
                scores.append(sc)
                if form:
                    pred_nf = layer.walk_forward(rows, t, seasons, alpha=a)["rows"]
                    joined = pred.assign(mu_nf=pred_nf["mu_layer"].to_numpy(),
                                         disp_nf=pred_nf["disp_layer"].to_numpy())
                    tests.append(_row(joined, t, f"form over layer (alpha={a})", "layer", "nf"))
                else:
                    tests.append(_row(pred, t, "offset over venue baseline", "off", "base"))
                    tests.append(_row(pred, t, f"layer over offset (alpha={a})", "layer", "off"))
        fitted = layer.fit(rows[rows["year"] < max(seasons) + 1], t, alpha=0.01)
        thetas[t] = fitted.coefficients()
    scores = pd.concat(scores, ignore_index=True)
    tests = pd.DataFrame(tests).drop_duplicates(["target", "comparison"])
    return {"scores": scores, "tests": tests, "coefficients": pd.DataFrame(thetas)}


# --- Tuning -------------------------------------------------------------------

ALPHA_GRID = (0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0)


def _wf_logloss(rows: pd.DataFrame, t: str, seasons, alpha: float, min_train: int) -> float:
    pred = layer.walk_forward(rows, t, seasons, alpha=alpha, min_train_seasons=min_train)["rows"]
    if pred.empty:
        return float("nan")
    return float(layer.score(pred, t).set_index("variant").at["layer", "logloss"])


def choose_alpha(rows: pd.DataFrame, t: str, seasons, grid=ALPHA_GRID, min_train: int = 1) -> float:
    """The grid alpha with the lowest walk-forward log loss over ``seasons``."""
    scores = {a: _wf_logloss(rows, t, seasons, a, min_train) for a in grid}
    scores = {a: v for a, v in scores.items() if np.isfinite(v)}
    return min(scores, key=scores.get)


def nested(offs: dict[str, pd.DataFrame], seasons=WALK_FORWARD, grid=ALPHA_GRID) -> dict:
    """Walk-forward with alpha chosen inside each fold, from earlier seasons only.

    The outer score is what the *procedure* -- tune on the past, predict the next
    season -- is worth. A score at a single alpha picked after seeing every
    season is not that, and is flattering by however much the pick was luck.
    """
    preds, picks = {}, []
    for t, rows in offs.items():
        first = int(rows.loc[rows["mu_dom"].notna(), "year"].min())
        parts = []
        for y in seasons:
            inner = [s for s in range(first + 1, y)]
            a = choose_alpha(rows, t, inner, grid) if inner else grid[len(grid) // 2]
            picks.append({"target": t, "season": y, "alpha": a})
            p = layer.walk_forward(rows, t, [y], alpha=a, min_train_seasons=2)["rows"]
            parts.append(p.assign(alpha=a))
        preds[t] = pd.concat(parts, ignore_index=True)
    return {"preds": preds, "picks": pd.DataFrame(picks)}


def coefficient_intervals(rows: pd.DataFrame, t: str, alpha: float, n_boot: int = 300,
                          seed: int = 2026) -> pd.DataFrame:
    """Point estimates and fixture-bootstrap 90% intervals for the layer's coefficients.

    Resampled by fixture, not by team-row: the two sides of a match share
    everything about it, and resampling them apart would understate the spread.
    """
    r = rows[rows["mu_dom"].notna() & rows["actual"].notna()]
    point = layer.fit(r, t, alpha=alpha).coefficients()
    fx = r["fixture_id"].unique()
    groups = {f: g for f, g in r.groupby("fixture_id")}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(fx, size=len(fx), replace=True)
        b = pd.concat([groups[f] for f in pick], ignore_index=True)
        draws.append(layer.fit(b, t, alpha=alpha).coefficients())
    d = pd.DataFrame(draws)
    return pd.DataFrame({"estimate": point, "lo90": d.quantile(0.05), "hi90": d.quantile(0.95)})


# --- Holdout -------------------------------------------------------------------

HOLDOUT_LOG = EURO_ARTIFACTS / "holdout_scored.json"


def holdout(offs: dict[str, pd.DataFrame], config: dict[str, float | None], *,
            force: bool = False) -> dict:
    """Score the held-out season, once. Refuses a second time unless forced.

    ``config`` is ``{target: alpha}``, with ``None`` meaning the at-par league
    model alone. Settings must be decided before calling: the point of a holdout
    is that nothing about the model is chosen after looking at it.
    """
    y = season_sort_key(test_season())
    if HOLDOUT_LOG.exists() and not force:
        raise RuntimeError(f"holdout already scored: {HOLDOUT_LOG.read_text()}")
    table, preds = [], {}
    for t, rows in offs.items():
        a = config[t]
        pred = layer.walk_forward(rows, t, [y], alpha=a if a is not None else 0.3,
                                  min_train_seasons=1)["rows"]
        final = "layer" if a is not None else "off"
        sc = layer.score(pred, t).set_index("variant")["logloss"]
        s_final = layer.significance(pred, t, final, "base")
        s_step = layer.significance(pred, t, "layer", "off")
        table.append({
            "target": t, "config": f"layer, alpha={a}" if a is not None else "at par only",
            "n": int(s_final["n"]), "ll_base": sc["base"], "ll_off": sc["off"], "ll_layer": sc["layer"],
            "final_vs_base": f"{(sc[final] / sc['base'] - 1) * 100:+.1f}%",
            "final_over_base": describe_significance(s_final),
            "layer_over_off": describe_significance(s_step),
        })
        preds[t] = pred
    HOLDOUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    HOLDOUT_LOG.write_text(json.dumps({"date": date.today().isoformat(), "season": test_season(),
                                       "config": config, "artifact": RunContext.load().version}))
    return {"table": pd.DataFrame(table), "preds": preds}


def counts(stats: pd.DataFrame | None = None) -> pd.DataFrame:
    """Top-five-versus-top-five matches per competition and season -- the step-0 gate."""
    stats = ingest.load_match_stats() if stats is None else stats
    rows = model.team_rows(stats, load_clean_table())
    m = rows.drop_duplicates("fixture_id")
    tab = m.pivot_table(index="season", columns="comp", values="fixture_id", aggfunc="count", fill_value=0)
    allm = stats.pivot_table(index="season", columns="league_key", values="espn_event_id",
                             aggfunc="count", fill_value=0).add_prefix("all_")
    return tab.join(allm)


__all__ = ["WALK_FORWARD", "ALPHAS", "ALPHA_GRID", "load_offsets", "walk_forward", "choose_alpha", "nested",
           "coefficient_intervals", "holdout", "counts", "HOLDOUT_LOG"]
