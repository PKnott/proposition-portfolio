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


HOLDOUT_LOG = EURO_ARTIFACTS / "holdout_scored.json"


def holdout(offs: dict[str, pd.DataFrame], *, alpha: float, form: bool, force: bool = False) -> pd.DataFrame:
    """Score the held-out season, once. Refuses a second time unless forced.

    Settings must be decided before calling: the point of a holdout is that
    nothing about the model is chosen after looking at it.
    """
    y = season_sort_key(test_season())
    if HOLDOUT_LOG.exists() and not force:
        raise RuntimeError(f"holdout already scored: {HOLDOUT_LOG.read_text()}")
    out = []
    for t, rows in offs.items():
        pred = layer.walk_forward(rows, t, [y], alpha=alpha, form=form,
                                  min_train_seasons=1)["rows"]
        sc = layer.score(pred, t)
        out.append({"target": t, **{f"ll_{r.variant}": r.logloss for r in sc.itertuples()},
                    "n": int(sc["n"].iat[0]),
                    "offset_over_base": describe_significance(layer.significance(pred, t, "off", "base")),
                    "layer_over_offset": describe_significance(layer.significance(pred, t, "layer", "off"))})
    HOLDOUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    HOLDOUT_LOG.write_text(json.dumps({"date": date.today().isoformat(), "season": test_season(),
                                       "alpha": alpha, "form": form,
                                       "artifact": RunContext.load().version}))
    return pd.DataFrame(out)


def counts(stats: pd.DataFrame | None = None) -> pd.DataFrame:
    """Top-five-versus-top-five matches per competition and season -- the step-0 gate."""
    stats = ingest.load_match_stats() if stats is None else stats
    rows = model.team_rows(stats, load_clean_table())
    m = rows.drop_duplicates("fixture_id")
    tab = m.pivot_table(index="season", columns="comp", values="fixture_id", aggfunc="count", fill_value=0)
    allm = stats.pivot_table(index="season", columns="league_key", values="espn_event_id",
                             aggfunc="count", fill_value=0).add_prefix("all_")
    return tab.join(allm)


__all__ = ["WALK_FORWARD", "ALPHAS", "load_offsets", "walk_forward", "holdout", "counts", "HOLDOUT_LOG"]
