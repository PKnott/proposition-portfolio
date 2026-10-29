"""How sure is each model probability? Two interval methods, for selection on a lower bound.

The builder qualifies a proposition when ``p * price >= 1``, i.e. when the model
sits above the market. Lines that only just clear it are as likely to be there by
model error as by edge, and the selected set inherits the optimistic errors (the
winner's curse that `study.selection_check` measures). The alternative tested
here: give every ``p`` an interval and qualify on its **lower** bound.

Two ways to size the interval:

* **A, model uncertainty** (`refit_probabilities`, `bootstrap_width`). Refit the
  production models as of each slate's run date on match-resampled history, and
  read the spread of each line's probability across the refits. Nothing about
  the market enters it. It measures how much the estimate itself wobbles.
* **B, track record** (`track_record_width`). How far, in probability, the model
  has typically been from what happened, as a function of how far it sits from
  the market price. Only error that shows up in aggregate is identifiable: a
  single yes/no outcome says nothing about the spread of that one line's true
  probability. So the width is the slate-level error at each distance, net of
  binomial noise.

Both produce a lower bound with `lower_bound`, at a one-sided confidence level
(0.5 means no interval: today's rule). `sizing_probability` picks the point
between the lower bound and ``p`` that positions are sized at.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm

# One-sided confidence levels the replay walks; 0.5 is today's rule.
LEVELS = (0.5, 0.6, 0.75, 0.9)
# Distance-from-market bins the track-record width is measured in (probability).
GAP_BINS = (0.0, 0.03, 0.06, 0.10, 0.15, 1.0)
_EPS = 1e-6


# --- A: refit the models on resampled history -------------------------------


def _team_probability(target: str, mu: float, line: float, dispersion) -> float:
    from .metrics import over_prob, pmf_for, poisson_pmf
    from .report.markets import GOALS_MATRIX_MAX
    from .spec import target_spec

    if target == "goals":
        # The workbook's goals ladder reads the scoreline matrix's marginal,
        # which is a Poisson pmf with a folded "6+" bucket.
        return float(over_prob(poisson_pmf(mu, GOALS_MATRIX_MAX), line))
    s = target_spec(target)
    return float(over_prob(pmf_for(s.dist, mu, s.total_cap, dispersion), line))


def _refit_job(args) -> pd.DataFrame:
    """One target, one run: B+1 fits as of `cutoff`, priced at the run's lines."""
    target, run, cutoff, rows, n_boot, seed = args
    import fpp
    from .build import build_feature_table, scorable_mask
    from .models import fit_model, predict_mu
    from .priors import BufferWindows

    tm = fpp.clean.load_clean_table()
    ctx = fpp.RunContext.load()
    w, feats = ctx.windows[target], ctx.features[target]
    bw = BufferWindows(tm, L_max=max(50, int(w["L"])))
    ftr = build_feature_table(tm, bw, int(w["L"]), float(w["alpha"]), features=feats)
    fsc = build_feature_table(tm, bw, int(w["L"]), float(w["alpha"]), features=feats,
                              drop_warmup=False)
    params = dict(ctx.params[target])
    params["n_estimators"] = ctx.n_estimators[target]
    params.pop("early_stopping_rounds", None)
    disp = ctx.dispersion.get(target, {})

    train = scorable_mask(ftr, target) & (ftr.meta["date"].to_numpy() < np.datetime64(cutoff))
    Xtr, ytr = ftr.X.loc[train, feats], ftr.y[target][train]
    fid = ftr.meta.loc[train, "fixture_id"].to_numpy()
    matches, inverse = np.unique(fid, return_inverse=True)

    # The scoring row of each proposition's team: same team and opponent, kickoff
    # within a day (results carry UTC dates, fixtures local ones).
    meta = fsc.meta.reset_index(drop=True)
    key = {}
    for i, (d, team, opp) in enumerate(zip(meta["date"], meta["team"], meta["opponent"])):
        key.setdefault((team, opp), []).append((pd.Timestamp(d), i))
    want, row_of = [], []
    for r in rows.itertuples():
        opp = r.away_team if r.team == r.home_team else r.home_team
        hit = [i for d, i in key.get((r.team, opp), [])
               if abs((d - pd.Timestamp(r.fixture_date)).days) <= 1]
        row_of.append(hit[0] if hit else -1)
    row_of = np.array(row_of)
    need = np.unique(row_of[row_of >= 0])
    Xsc = fsc.X.iloc[need][feats]
    pos = {j: k for k, j in enumerate(need)}

    rng = np.random.default_rng([seed, int(run[1:]), len(target)])
    out = np.full((len(rows), n_boot + 1), np.nan)
    for b in range(n_boot + 1):
        if b == 0:
            model = fit_model(Xtr, ytr, params)
        else:
            counts = np.bincount(rng.integers(0, len(matches), len(matches)), minlength=len(matches))
            take = np.repeat(np.arange(len(ytr)), counts[inverse])
            model = fit_model(Xtr.iloc[take], ytr[take], params)
        mu = predict_mu(model, Xsc)
        for k, r in enumerate(rows.itertuples()):
            if row_of[k] < 0:
                continue
            d = disp.get(r.league_key) if isinstance(disp, dict) else disp
            out[k, b] = _team_probability(target, float(mu[pos[row_of[k]]]), float(r.line), d)
    frame = pd.DataFrame(out, columns=[f"p_{b}" for b in range(n_boot + 1)])
    frame.insert(0, "prop_key", rows["prop_key"].to_numpy())
    frame.insert(0, "run_code", run)
    return frame


def refit_probabilities(rows: pd.DataFrame, cutoffs: dict[str, pd.Timestamp], *,
                        n_boot: int = 50, seed: int = 2026, workers: int = 8) -> pd.DataFrame:
    """Every row's probability under the refit (`p_0`) and `n_boot` resampled refits.

    `rows` needs run_code, prop_key, fixture_date, league_key, home_team,
    away_team, target, team and line. Each run is refit on matches strictly before
    its cutoff -- the date the run was priced -- so no fit sees the matches it
    prices. Resampling is by match: both sides of a fixture come and go together.
    """
    from concurrent.futures import ProcessPoolExecutor

    cols = ["run_code", "prop_key", "fixture_date", "league_key", "home_team", "away_team",
            "target", "team", "line"]
    jobs = []
    for run, g in rows[cols].groupby("run_code"):
        for target, gt in g.groupby("target"):
            jobs.append((str(target), str(run), cutoffs[run], gt.reset_index(drop=True), n_boot, seed))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return pd.concat(list(ex.map(_refit_job, jobs)), ignore_index=True)


def bootstrap_width(boot: pd.DataFrame) -> pd.Series:
    """Method A's width: the standard deviation of logit(p) across the resampled refits."""
    cols = [c for c in boot.columns if c.startswith("p_") and c != "p_0"]
    x = logit(np.clip(boot[cols].to_numpy(float), _EPS, 1 - _EPS))
    return pd.Series(np.nanstd(x, axis=1, ddof=1), index=boot.index, name="width_a")


# --- B: the track record against distance from the market --------------------


def gap_error_table(s: pd.DataFrame, *, p_col: str = "p_raw", group: str = "run_code",
                    bins=GAP_BINS) -> pd.DataFrame:
    """Typical size of the model's error at each distance above the market.

    For each slate and gap bin, the error ``mean(y) - mean(p)``; its mean square
    across slates, less the binomial noise each one carries
    (``mean p(1-p) / n``), is the squared error that is really there. Only lines
    where the model sits above the market, the side selection draws from.
    """
    d = s[s["o"].notna()].copy()
    d["gap"] = d[p_col] - 1.0 / d["o"]
    d = d[d["gap"] > 0]
    d["bin"] = pd.cut(d["gap"], list(bins))
    d["pq"] = d[p_col] * (1 - d[p_col])
    g = d.groupby(["bin", group], observed=True).agg(n=("y", "size"), y=("y", "mean"),
                                                     p=(p_col, "mean"), pq=("pq", "mean"),
                                                     gap=("gap", "mean"))
    g["e2"] = (g["y"] - g["p"]) ** 2 - g["pq"] / g["n"]
    out = g.groupby(level="bin", observed=True).apply(
        lambda x: pd.Series({"gap": np.average(x["gap"], weights=x["n"]),
                             "n": x["n"].sum(), "slates": len(x),
                             "bias": np.average(x["y"] - x["p"], weights=x["n"]),
                             "err2": np.average(x["e2"], weights=x["n"])}))
    out["width"] = np.sqrt(out["err2"].clip(lower=0))
    return out.reset_index()


def track_record_width(train: pd.DataFrame, *, p_col: str = "p_raw") -> tuple[float, float]:
    """Method B's width as ``a + b * gap`` (probability), fitted on `train`.

    Weighted by lines per bin, and constrained to be non-negative everywhere on
    the fitted range: a width cannot be below zero, and a fit that tilts down
    would read as more certainty the further the model strays from the price.
    """
    t = gap_error_table(train, p_col=p_col)
    t = t[t["n"] > 0]
    if len(t) < 2:
        return float(t["width"].mean() if len(t) else 0.0), 0.0
    b, a = np.polyfit(t["gap"], t["width"], 1, w=np.sqrt(t["n"]))
    if b < 0:
        a, b = float(np.average(t["width"], weights=t["n"])), 0.0
    return max(float(a), 0.0), float(b)


# --- the lower bound and the sizing point ------------------------------------


def z_for(level: float) -> float:
    """One-sided normal quantile; 0 at level 0.5."""
    return float(norm.ppf(level))


def lower_bound(p, *, level: float, width_a=None, width_b=None) -> np.ndarray:
    """The one-sided lower bound of ``p`` at `level`, from exactly one width.

    `width_a` is on the logit scale (method A); `width_b` is in probability
    (method B).
    """
    p = np.asarray(p, float)
    z = z_for(level)
    if (width_a is None) == (width_b is None):
        raise ValueError("pass exactly one of width_a / width_b")
    if width_a is not None:
        x = logit(np.clip(p, _EPS, 1 - _EPS)) - z * np.asarray(width_a, float)
        return expit(x)
    return np.clip(p - z * np.asarray(width_b, float), _EPS, 1 - _EPS)


def sizing_probability(p, lower, tau: float) -> np.ndarray:
    """The point positions are sized at: ``p`` at tau 0, the lower bound at tau 1."""
    p, lower = np.asarray(p, float), np.asarray(lower, float)
    return p - tau * (p - lower)


__all__ = ["LEVELS", "GAP_BINS", "refit_probabilities", "bootstrap_width", "gap_error_table",
           "track_record_width", "z_for", "lower_bound", "sizing_probability"]
