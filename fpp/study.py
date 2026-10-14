"""Correlation and calibration, measured on the slates the current model priced.

Three questions, each answered on the same read-only dataset:

1. `pair_correlation` -- how correlated are two propositions on the same match?
   This is the evidence behind `config.PAIR_CORRELATION`.
2. `reliability_frame` / `gap_table` / `market_comparison` -- are the model's
   probabilities better calibrated than the price, worse, or no different?
3. `recalibration_study` -- would a dampening or inflation constant, global or
   per context, have improved the probabilities *out of sample*?

**Read-only.** `load_study_set` reads the ledger tables and the Pending inbox and
grades unsettled rows in memory against `espn_match_stats.csv`. It never absorbs,
settles or writes anything, so running it can never disturb the ledger.

**Which slates.** Only runs from `config.STUDY_FIRST_RUN` on. Earlier runs were
priced by a superseded model (the ledger excludes R001-R005 for exactly that
reason), and pooling them would describe a model that no longer exists.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import ledger
from .analysis import model_vs_market
from .calibration import MIN_BIN_COUNT, N_BINS, quantile_bins, wilson_interval
from .config import LEAGUES, PAIR_CORRELATION, STUDY_FIRST_RUN

EPS = 1e-6
TARGETS: tuple[str, ...] = ("goals", "shots", "sot", "corners")
ODDS_BANDS: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 5.0, np.inf)


def _run_number(code) -> int:
    return int(str(code).lstrip("R"))


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(x) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _ll(y, p) -> np.ndarray:
    """Per-row log loss."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# --- 0. the dataset -------------------------------------------------------


def grade(props: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """`actual` and `won` for every row, in memory. The same rule as `ledger.settle`.

    Rows the ledger has already settled keep their banked result; only rows with
    `won` null are looked up. A proposition wins on ``actual > line``, strictly.
    """
    out = props.copy()
    todo = out[out["won"].isna()]
    if todo.empty:
        return out
    on = ["fixture_date", "league_key", "home_team", "away_team"]
    fixtures = todo[on].drop_duplicates().reset_index(drop=True)
    matched = ledger.match_fixtures(fixtures, results)
    joined = todo[on + ["scope", "target", "line"]].merge(matched, on=on, how="left")
    actual = np.full(len(joined), np.nan)
    for scope in ("home", "away"):
        for target in TARGETS:
            m = ((joined["scope"] == scope) & (joined["target"] == target)).to_numpy()
            if m.any():
                actual[m] = joined.loc[m, f"{scope}_{target}"].to_numpy(dtype=float)
    won = np.where(np.isfinite(actual), actual > joined["line"].to_numpy(dtype=float), None)
    out.loc[todo.index, "actual"] = actual
    out.loc[todo.index, "won"] = pd.array(won, dtype="boolean")
    return out


def run_models(root=None) -> pd.DataFrame:
    """The model each run was priced with, from `runs` and the Pending `run.json` files.

    Runs captured before 14 Oct 2026 carry no stamp and come back with nulls; for
    those, `config.STUDY_FIRST_RUN` is still the evidence.
    """
    import json

    cols = ["run_code", "model_version", "model_fingerprint", "stamp_origin"]
    runs = ledger.read_table("runs", root)
    rows = [] if runs.empty else runs.reindex(columns=cols).to_dict("records")
    for path in ledger.pending_slates(root):
        f = path / "run.json"
        if f.exists():
            rec = json.loads(f.read_text(encoding="utf-8"))
            rows.append({c: rec.get(c) for c in cols})
    return pd.DataFrame(rows, columns=cols).drop_duplicates("run_code", keep="last")


def load_study_set(first_run: str = STUDY_FIRST_RUN, last_run: str | None = None, *,
                   dedupe: bool = True, root=None, results: pd.DataFrame | None = None,
                   model_version: str | None = None
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every graded proposition the current model priced, and a coverage table.

    Reads ledger `propositions` and every `Pending/slate_*/propositions.parquet`.
    Drops excluded runs, runs before `first_run`, and anything outside the five
    domestic leagues.

    `model_version`, when given, drops every *stamped* run priced by a different
    model. Unstamped runs (before 14 Oct 2026) are kept and rely on `first_run`.

    `dedupe` keeps one row per `prop_key`, from the latest run. Re-runs on the same
    matchday (R009/R010, R012/R013, R014/R015, R018/R019) offer the same
    proposition two or three times, and counting each copy would treat one
    outcome as several. The latest run is kept because its price is the one
    closest to kickoff.
    """
    held = ledger.read_table("propositions", root)
    pending = [pd.read_parquet(p / "propositions.parquet")
               for p in ledger.pending_slates(root)
               if (p / "propositions.parquet").exists()]
    props = pd.concat([held, *pending], ignore_index=True)
    # Runs captured before the shots shift existed carry no `p_raw`: their `p`
    # is the model's own value.
    props["p_raw"] = (props["p_raw"].fillna(props["p"]) if "p_raw" in props
                      else props["p"])

    runs = ledger.read_table("runs", root)
    excluded = set()
    if not runs.empty and "status" in runs:
        excluded = set(runs.loc[runs["status"] == "excluded", "run_code"])
    n = props["run_code"].map(_run_number)
    keep = (n >= _run_number(first_run)) & ~props["run_code"].isin(excluded)
    if last_run is not None:
        keep &= n <= _run_number(last_run)
    keep &= props["league_key"].isin(list(LEAGUES))
    if model_version is not None:
        models = run_models(root).set_index("run_code")["model_version"]
        stamped = props["run_code"].map(models)
        keep &= stamped.isna() | (stamped == model_version)
    props = props[keep].copy()
    props["fixture_date"] = pd.to_datetime(props["fixture_date"]).dt.normalize()

    if dedupe:
        props["_n"] = props["run_code"].map(_run_number)
        props = (props.sort_values("_n").drop_duplicates("prop_key", keep="last")
                 .drop(columns="_n"))

    graded = grade(props.reset_index(drop=True),
                   results if results is not None else ledger.load_results())

    cov = (graded.assign(graded=graded["won"].notna())
           .groupby("league_key", observed=True)
           .agg(propositions=("prop_key", "size"), graded=("graded", "sum"),
                fixtures=("home_team", lambda s: 0), slates=("run_code", "nunique")))
    cov["fixtures"] = (graded.drop_duplicates(["fixture_date", "league_key", "home_team"])
                       .groupby("league_key").size())
    cov["ungraded"] = cov["propositions"] - cov["graded"]
    cov.loc["all"] = [len(graded), int(graded["won"].notna().sum()),
                      int(cov["fixtures"].sum()), graded["run_code"].nunique(),
                      int(graded["won"].isna().sum())]

    s = graded[graded["won"].notna()].copy()
    s["y"] = s["won"].astype(float)
    s["market_p"] = 1.0 / s["o"]
    s["league"] = s["league_key"].astype(str)
    s["target"] = s["target"].astype(str)
    s["venue"] = s["scope"].astype(str)
    s["event"] = (s["fixture_date"].dt.strftime("%Y-%m-%d") + "|" + s["league"]
                  + "|" + s["home_team"].astype(str) + "|" + s["away_team"].astype(str))
    s["odds_band"] = pd.cut(s["o"], ODDS_BANDS, right=False).astype(str)
    s["target_league"] = s["target"] + " · " + s["league"]
    return s.reset_index(drop=True), cov.reset_index()


# --- 1. same-match correlation --------------------------------------------


def relation(same_team, same_stat, gap) -> np.ndarray:
    """The `PAIR_CORRELATION` class of each pair, by the same rule as `portfolio.pair_rho`."""
    same_team = np.asarray(same_team, bool)
    same_stat = np.asarray(same_stat, bool)
    gap = np.asarray(gap, float)
    return np.select(
        [~same_team, ~same_stat, gap <= 1.0, gap <= 2.0],
        ["opposite_teams", "same_team_diff_stat",
         "same_team_same_stat_adjacent", "same_team_same_stat_near"],
        "same_team_same_stat_far")


def pair_frame(s: pd.DataFrame, *, qualified_only: bool = False) -> pd.DataFrame:
    """Every pair of propositions on the same match, with ``zz = z_i * z_j``.

    ``z = (won - p) / sqrt(p(1-p))`` is a proposition's standardised surprise. If
    `p` is right, `z` has mean 0 and variance 1, so the mean of `z_i z_j` over
    pairs estimates their correlation. This is the estimator the original
    R006-R007 study used, rebuilt as a function so the numbers can be refreshed.

    Stats are put in a canonical order (`TARGETS`) within each pair so a 4x4
    table is symmetric.
    """
    d = s[s["qualified"].astype(bool)] if qualified_only else s
    d = d.reset_index(drop=True)
    p = d["p"].clip(EPS, 1 - EPS)
    d = d.assign(z=(d["y"] - p) / np.sqrt(p * (1 - p)), _i=np.arange(len(d)))
    cols = ["event", "_i", "z", "team", "target", "line", "run_code"]
    m = d[cols].merge(d[cols], on="event", suffixes=("_a", "_b"))
    m = m[m["_i_a"] < m["_i_b"]]

    order = {t: k for k, t in enumerate(TARGETS)}
    swap = m["target_a"].map(order) > m["target_b"].map(order)
    stat_a = np.where(swap, m["target_b"], m["target_a"])
    stat_b = np.where(swap, m["target_a"], m["target_b"])
    same_team = (m["team_a"] == m["team_b"]).to_numpy()
    same_stat = (m["target_a"] == m["target_b"]).to_numpy()
    gap = (m["line_a"] - m["line_b"]).abs().to_numpy()
    return pd.DataFrame({
        "event": m["event"].to_numpy(),
        "run_code": m["run_code_a"].to_numpy(),
        "stat_a": stat_a, "stat_b": stat_b,
        "same_team": same_team, "same_stat": same_stat,
        "line_gap": np.where(same_team & same_stat, gap, np.nan),
        "relation": relation(same_team, same_stat, gap),
        "zz": (m["z_a"] * m["z_b"]).to_numpy(),
    })


def _cluster_mean(zz: np.ndarray, cluster: np.ndarray) -> tuple[float, float]:
    """Mean and its cluster-robust standard error, clustered by match.

    Pairs on one match share propositions, so they are not independent. The
    naive `std/sqrt(n)` the first study reported treats them as if they were and
    is too narrow; this one treats the match as the unit of evidence.
    """
    n = len(zz)
    if n == 0:
        return float("nan"), float("nan")
    m = float(zz.mean())
    resid = pd.Series(zz - m).groupby(cluster).sum().to_numpy()
    g = len(resid)
    if g < 2:
        return m, float("nan")
    return m, float(np.sqrt((resid ** 2).sum() * g / (g - 1)) / n)


def rho_table(pairs: pd.DataFrame, by) -> pd.DataFrame:
    """ρ̂, its match-clustered standard error, and how many pairs and matches back it."""
    by = [by] if isinstance(by, str) else list(by)
    rows = []
    for key, g in pairs.groupby(by, observed=True, dropna=True):
        key = key if isinstance(key, tuple) else (key,)
        rho, se = _cluster_mean(g["zz"].to_numpy(), g["event"].to_numpy())
        rows.append({**dict(zip(by, key)), "rho": rho, "se": se,
                     "lo": rho - 1.96 * se, "hi": rho + 1.96 * se,
                     "pairs": len(g), "matches": g["event"].nunique()})
    return pd.DataFrame(rows)


def placebo_correlation(s: pd.DataFrame, *, n: int = 20_000, seed: int = 2,
                        qualified_only: bool = False) -> dict:
    """The same estimator on random pairs from *different* matches. Should be ~0.

    If it is not, the estimator itself is biased -- for instance because `p` is
    miscalibrated, which shifts every `z` the same way -- and the same-match
    numbers have to be read relative to it rather than relative to zero.
    """
    d = s[s["qualified"].astype(bool)] if qualified_only else s
    p = d["p"].clip(EPS, 1 - EPS).to_numpy()
    z = (d["y"].to_numpy() - p) / np.sqrt(p * (1 - p))
    ev = d["event"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(z), size=(n, 2))
    ok = ev[idx[:, 0]] != ev[idx[:, 1]]
    zz = z[idx[ok, 0]] * z[idx[ok, 1]]
    se = float(zz.std() / np.sqrt(ok.sum()))
    return {"rho": float(zz.mean()), "se": se, "pairs": int(ok.sum())}


def correlation_vs_config(pairs: pd.DataFrame) -> pd.DataFrame:
    """Measured ρ̂ per `PAIR_CORRELATION` class, against the value in config.

    `differs` flags a class whose configured value sits outside the 95% interval,
    which makes it worth considering a config change. Inside the interval, the
    data has no quarrel with the config.
    """
    t = rho_table(pairs, "relation")
    t["configured"] = t["relation"].map(PAIR_CORRELATION)
    t["differs"] = (t["configured"] < t["lo"]) | (t["configured"] > t["hi"])
    order = list(PAIR_CORRELATION)
    return t.set_index("relation").reindex(order).reset_index()


# --- 2. calibration -------------------------------------------------------


def reliability_frame(s: pd.DataFrame, by: str = "target", *, n_bins: int = N_BINS,
                      min_count: int = MIN_BIN_COUNT) -> pd.DataFrame:
    """Reliability bins for the model and the price, long format, ready to plot.

    Equal-count bins (`calibration.quantile_bins`) with Wilson intervals on the
    hit rate. Each source is binned on its own predictions, so a model bin and a
    market bin at the same x describe the same predicted probability, not the
    same propositions.
    """
    rows = []
    for key, g in s.groupby(by, observed=True):
        y = g["y"].to_numpy()
        for source, col in (("model", "p"), ("market", "market_p")):
            p = g[col].to_numpy()
            idx = quantile_bins(p, n_bins, min_count)
            for b in np.unique(idx):
                m = idx == b
                k, n = int(y[m].sum()), int(m.sum())
                lo, hi = wilson_interval(k, n)
                rows.append({by: key, "source": source, "bin": int(b), "n": n,
                             "predicted": float(p[m].mean()), "observed": k / n,
                             "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def gap_table(s: pd.DataFrame, rows: str = "target", cols: str = "league",
              source: str = "p") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hit rate minus mean predicted probability, and the count behind each cell.

    Positive means the propositions won more often than predicted (under-
    prediction, which -- every proposition being an Over -- is in our favour). Negative is the
    costly direction.
    """
    g = s.groupby([rows, cols], observed=True)
    gap = (g["y"].mean() - g[source].mean()).unstack(cols)
    n = g.size().unstack(cols)
    return gap, n


def market_comparison(s: pd.DataFrame, groups=("target", "league")) -> pd.DataFrame:
    """`analysis.model_vs_market` for each grouping, stacked into one frame.

    `ll_diff` is model minus market log loss per proposition, so **negative means
    the model beat the price**. The interval is `model_vs_market`'s paired
    bootstrap, with its sign flipped to match.
    """
    out = []
    for by in groups:
        t = model_vs_market(s, by=by)
        col = by if by in t else "segment"
        t = t.rename(columns={col: "level"})
        t["grouping"] = by
        out.append(t)
    t = pd.concat(out, ignore_index=True)
    t = t[~((t["level"] == "all") & (t["grouping"] != groups[0]))]
    t["ll_diff"] = -t["mean_gain"]
    t["ll_lo"], t["ll_hi"] = -t["ci_hi"], -t["ci_lo"]
    t.loc[t["level"] == "all", "grouping"] = "all"
    return t[["grouping", "level", "n", "n_slates", "base_rate", "model_p", "market_p",
              "model_logloss", "market_logloss", "ll_diff", "ll_lo", "ll_hi", "winner"]]


def selection_check(s: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Calibration on the propositions that qualified (``e >= 1``) against the rest.

    Selecting on ``p * o >= 1`` picks out the rows where the model sits above the
    price, and those are disproportionately the rows where the model is too
    high. A model can therefore under-predict across the board and still
    over-predict on exactly the positions it takes. This shows whether that is
    happening, and it is the table any constant has to be checked against: a
    shift that fixes the whole book can make the qualified set worse.
    """
    keys = ["qualified"] + ([by] if by else [])
    rows = []
    for key, g in s.groupby(keys, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        k, n = int(g["y"].sum()), len(g)
        lo, hi = wilson_interval(k, n)
        rows.append({**dict(zip(keys, key)), "n": n, "hit_rate": k / n,
                     "hit_lo": lo, "hit_hi": hi, "model_p": g["p"].mean(),
                     "market_p": g["market_p"].mean(),
                     "gap_pp": (k / n - g["p"].mean()) * 100,
                     "flat_roi": float((g["o"] * g["y"] - 1).mean())})
    out = pd.DataFrame(rows)
    out["qualified"] = out["qualified"].map({True: "qualified (held)", False: "not qualified"})
    return out


# --- 3. should p be dampened or inflated? ---------------------------------
#
# The adjustment is a logistic recalibration, ``logit p' = a + b * logit p``:
#
# * ``b < 1`` dampens -- pulls every probability towards 50%;
# * ``b > 1`` inflates -- pushes them away from it;
# * ``a`` shifts everything up (``a > 0``) or down without changing the spread.
#
# A *contextual* constant lets ``(a, b)`` differ by context (per target, per
# league, ...), as offsets from the global fit with a ridge penalty pulling them
# back towards it. A context with little data then barely moves, which is the
# protection against fitting noise in a thin cell.
#
# Everything is scored **leave-one-slate-out**: fit on every slate but one,
# score the one held out, repeat. A constant that only improves the slates it
# was fitted on has learnt nothing.


def _design(levels: np.ndarray | None, x: np.ndarray) -> tuple[np.ndarray, int]:
    if levels is None:
        return np.zeros((len(x), 0)), 0
    k = int(levels.max()) + 1
    onehot = np.zeros((len(x), k))
    onehot[np.arange(len(x)), levels] = 1.0
    return onehot, k


def fit_recalibration(y, p, levels=None, *, ridge: float = 0.0, market_p=None) -> np.ndarray:
    """Fitted parameters ``[a, b, da_1..da_K, db_1..db_K, (c)]``.

    `levels` are integer context codes; with none, this is the global fit. When
    `market_p` is given, ``c * logit(market_p)`` is added to the linear predictor,
    which turns this into the "lean on the price" benchmark.
    """
    y = np.asarray(y, float)
    x = logit(p)
    onehot, k = _design(None if levels is None else np.asarray(levels), x)
    xm = None if market_p is None else logit(market_p)
    n_par = 2 + 2 * k + (0 if xm is None else 1)

    def eta(w):
        e = w[0] + w[1] * x
        if k:
            e = e + onehot @ w[2:2 + k] + (onehot * x[:, None]) @ w[2 + k:2 + 2 * k]
        if xm is not None:
            e = e + w[-1] * xm
        return e

    def loss(w):
        e = eta(w)
        q = expit(e)
        nll = np.sum(np.logaddexp(0, e) - y * e)
        r = q - y
        grad = np.empty_like(w)
        grad[0], grad[1] = r.sum(), (r * x).sum()
        if k:
            grad[2:2 + k] = onehot.T @ r
            grad[2 + k:2 + 2 * k] = onehot.T @ (r * x)
            pen = w[2:2 + 2 * k]
            nll += 0.5 * ridge * (pen ** 2).sum()
            grad[2:2 + 2 * k] += ridge * pen
        if xm is not None:
            grad[-1] = (r * xm).sum()
        return nll, grad

    w0 = np.zeros(n_par)
    w0[1] = 1.0
    return minimize(loss, w0, jac=True, method="L-BFGS-B").x


def apply_recalibration(w, p, levels=None, market_p=None) -> np.ndarray:
    x = logit(p)
    e = w[0] + w[1] * x
    k = (len(w) - 2 - (0 if market_p is None else 1)) // 2
    if k and levels is not None:
        lv = np.asarray(levels)
        e = e + w[2:2 + k][lv] + w[2 + k:2 + 2 * k][lv] * x
    if market_p is not None:
        e = e + w[-1] * logit(market_p)
    return expit(e)


RIDGE_GRID: tuple[float, ...] = (3.0, 10.0, 30.0, 100.0, 300.0)


def _choose_ridge(y, p, levels, groups, grid=RIDGE_GRID) -> float:
    """The ridge strength with the lowest leave-one-slate-out loss on *training* data only."""
    best, best_loss = grid[-1], np.inf
    for lam in grid:
        tot = 0.0
        for g in np.unique(groups):
            tr, te = groups != g, groups == g
            w = fit_recalibration(y[tr], p[tr], levels[tr], ridge=lam)
            tot += _ll(y[te], apply_recalibration(w, p[te], levels[te])).sum()
        if tot < best_loss:
            best, best_loss = lam, tot
    return best


def _cluster_boot(diff: np.ndarray, cluster: np.ndarray, n_boot: int = 2000,
                  seed: int = 2026) -> tuple[float, float]:
    """Percentile CI for a mean per-row difference, resampling whole matches."""
    sums = pd.Series(diff).groupby(cluster).agg(["sum", "size"])
    s, c = sums["sum"].to_numpy(), sums["size"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    boot = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def loso_predictions(s: pd.DataFrame, contexts=("target", "league", "venue",
                                                 "odds_band", "target_league"),
                     *, group: str = "run_code") -> pd.DataFrame:
    """Out-of-slate predictions for raw, global, each contextual fit, and the market blend.

    One row per proposition. Each column is what that proposition's probability
    would have been under the adjustment, fitted without its own slate. The ridge
    for each contextual fit is chosen inside each training fold, so the
    held-out slate never influences it.
    """
    y = s["y"].to_numpy()
    p = s["p"].to_numpy()
    mp = s["market_p"].to_numpy()
    groups = s[group].astype(str).to_numpy()
    codes = {c: pd.Categorical(s[c]).codes.astype(int) for c in contexts}

    out = pd.DataFrame({"y": y, "raw": p, "market": mp,
                        "event": s["event"].to_numpy(), "slate": groups}, index=s.index)
    for c in contexts:
        out[c] = s[c].to_numpy()
    out["global"] = np.nan
    out["blend"] = np.nan
    for c in contexts:
        out[f"ctx:{c}"] = np.nan

    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        w = fit_recalibration(y[tr], p[tr])
        out.loc[te, "global"] = apply_recalibration(w, p[te])
        wb = fit_recalibration(y[tr], p[tr], market_p=mp[tr])
        out.loc[te, "blend"] = apply_recalibration(wb, p[te], market_p=mp[te])
        for c in contexts:
            lv = codes[c]
            lam = _choose_ridge(y[tr], p[tr], lv[tr], groups[tr])
            wc = fit_recalibration(y[tr], p[tr], lv[tr], ridge=lam)
            out.loc[te, f"ctx:{c}"] = apply_recalibration(wc, p[te], lv[te])
    return out


def recalibration_summary(cv: pd.DataFrame) -> pd.DataFrame:
    """Out-of-slate log-loss gain of every adjustment, with match-clustered intervals.

    `gain` is how much lower the log loss is than the comparison, per
    proposition: **positive means the adjustment helped**. Each contextual fit is
    compared with the global one, because the question is whether a context
    deserves its *own* constant rather than whether any constant helps.
    """
    y, ev = cv["y"].to_numpy(), cv["event"].to_numpy()
    ll = {c: _ll(y, cv[c].to_numpy()) for c in cv.columns
          if c in ("raw", "market", "global", "blend") or c.startswith("ctx:")}
    comps = [("global", "raw"), ("blend", "global"), ("market", "raw")]
    comps += [(c, "global") for c in ll if c.startswith("ctx:")]
    rows = []
    for new, base in comps:
        d = ll[base] - ll[new]
        lo, hi = _cluster_boot(d, ev)
        rows.append({"adjustment": new.replace("ctx:", "per "), "versus": base,
                     "gain": float(d.mean()), "lo": lo, "hi": hi,
                     "logloss": float(ll[new].mean()),
                     "helps": bool(lo > 0), "hurts": bool(hi < 0)})
    return pd.DataFrame(rows)


def recalibration_by_level(cv: pd.DataFrame, context: str) -> pd.DataFrame:
    """Within each level of one context: does its own constant beat the global one?"""
    rows = []
    for lvl, g in cv.groupby(context, observed=True):
        y = g["y"].to_numpy()
        for new, base in ((f"ctx:{context}", "global"), ("global", "raw")):
            d = _ll(y, g[base].to_numpy()) - _ll(y, g[new].to_numpy())
            lo, hi = _cluster_boot(d, g["event"].to_numpy())
            rows.append({"context": context, "level": lvl, "n": len(g),
                         "comparison": f"{new.replace('ctx:', 'per ')} vs {base}",
                         "gain": float(d.mean()), "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def fitted_constants(s: pd.DataFrame, context: str | None = None,
                     ridge: float | None = None) -> pd.DataFrame:
    """``(a, b)`` fitted on every slate, globally or per level of `context`.

    These are the numbers that would go into the pipeline. They are fitted on
    all the data, so they come from the same fit the cross-validation scored but
    are not themselves validated; the CV tables say whether to trust them.
    `p_at_30` / `p_at_50` translate the pair into what it does to a typical
    proposition.
    """
    y, p = s["y"].to_numpy(), s["p"].to_numpy()
    if context is None:
        w = fit_recalibration(y, p)
        rows = [{"level": "global", "n": len(s), "a": w[0], "b": w[1]}]
    else:
        cat = pd.Categorical(s[context])
        lv = cat.codes.astype(int)
        lam = ridge if ridge is not None else _choose_ridge(
            y, p, lv, s["run_code"].astype(str).to_numpy())
        w = fit_recalibration(y, p, lv, ridge=lam)
        k = len(cat.categories)
        rows = [{"level": lvl, "n": int((lv == i).sum()),
                 "a": w[0] + w[2 + i], "b": w[1] + w[2 + k + i], "ridge": lam}
                for i, lvl in enumerate(cat.categories)]
    t = pd.DataFrame(rows)
    for q in (0.30, 0.50):
        t[f"p_at_{int(q * 100)}"] = expit(t["a"] + t["b"] * logit(q))
    t["reads_as"] = np.where(t["b"] < 0.97, "dampen",
                             np.where(t["b"] > 1.03, "inflate", "no spread change"))
    t["reads_as"] += np.where(t["a"] > 0.03, ", shift up",
                              np.where(t["a"] < -0.03, ", shift down", ""))
    return t


# --- 4. push, select, deflate, size ---------------------------------------
#
# The model leans low across the board, but the propositions that clear
# `p * o >= 1` are the ones where it happened to lean high. Two adjustments,
# applied at two different points:
#
# 1. **Push** before selection: ``logit p_push = logit p + k * (a + (b - 1) logit p)``,
#    the recalibration fitted above, scaled by ``k``. ``k = 0`` is today's model;
#    ``k = 1`` is the full calibration fit.
# 2. **Deflate** before sizing: ``p_stake = p_push - d``. ``d`` is the
#    selection bias, i.e. how far the pushed probability sat above the hit rate *on
#    the propositions it selected*, measured on slates the push never saw. It is
#    an additive shift in probability, the same form as `config.PESSIMISM_B`.
#
# Both are learnt **without the slate being scored**. For each held-out slate:
# an inner leave-one-slate-out over the remaining slates gives every training
# row a pushed probability from a fit that did not see it. Those rows give `d`
# for every `k`, and (when `k` is tuned) the inner profit that picks `k`. Then
# the push is refitted on all training slates and applied to the held-out one.


def push(w, p, k: float) -> np.ndarray:
    """`p` moved a fraction `k` of the way along the fitted recalibration `w = (a, b)`."""
    x = logit(p)
    return expit(x + k * (w[0] + (w[1] - 1.0) * x))


def select(df: pd.DataFrame, p: np.ndarray) -> np.ndarray:
    """`portfolio.qualify`'s filter on probability `p`: ``p * o >= 1``, then the
    ``(p, o)`` frontier within each match. Returns a boolean mask over `df`."""
    from .staking import undominated

    d = pd.DataFrame({"p": np.asarray(p, float), "o": df["o"].to_numpy(),
                      "event": df["event"].to_numpy(), "_r": np.arange(len(df))})
    d = d[d["p"] * d["o"] >= 1.0]
    keep = np.zeros(len(df), bool)
    for _, g in d.groupby("event", sort=False):
        keep[undominated(g)["_r"].to_numpy()] = True
    return keep


def kelly_stakes(p_stake, o, frac: float = 0.25) -> np.ndarray:
    """Fractional Kelly per position; zero where the sizing probability shows no edge."""
    p_stake, o = np.asarray(p_stake, float), np.asarray(o, float)
    return frac * np.clip((p_stake * o - 1.0) / (o - 1.0), 0.0, None)


SLATE_CAP = 0.20


def slate_growth(f, ret, cap: float = SLATE_CAP) -> float:
    """Log growth of capital that takes every position on one slate at sizes `f`.

    Sizes are fractions of capital, scaled down together if they would sum
    to more than `cap` of it -- a slate is never allowed to risk the whole pot,
    as the protective stake never does. Unlike summed profit this punishes
    over-sizing: a probability that is too high sizes too large, and the log
    charges for it.
    """
    f = np.asarray(f, float)
    total = f.sum()
    if total > cap:
        f = f * (cap / total)
    return float(np.log1p((f * ret).sum()))


def _position_stats(y, o, p_stake, sel, frac, cap=SLATE_CAP):
    f = kelly_stakes(p_stake, o, frac) * sel
    ret = o * y - 1.0
    return {"log_growth": slate_growth(f, ret, cap), "positions": int(sel.sum()), "sized_positions": int((f > 0).sum()),
            "flat_pnl": float(ret[sel].sum()),
            "stake": float(f.sum()), "kelly_pnl": float((f * ret).sum()),
            "kelly_expected": float((f * (p_stake * o - 1.0)).sum()),
            "said": float(p_stake[sel].sum()), "landed": float(y[sel].sum())}


PUSH_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5)


def push_deflate_backtest(s: pd.DataFrame, *, ks=PUSH_GRID, frac: float = 0.25,
                          per_target: bool = False, ridge: float = 300.0,
                          cap: float = SLATE_CAP, push_targets: tuple[str, ...] | None = None,
                          deflate_by_target: bool = False, deflate_prior: float = 50.0,
                          group: str = "run_code") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-slate result of every push `k`, each with its own learnt deflation.

    Returns ``(per_slate, tuning)``:

    * `per_slate`: one row per held-out slate × strategy. The strategies are
      ``k=<k> push+deflate`` for every `k`, ``k=<k> push only`` (no deflation),
      and ``tuned`` (the `k` with the best inner capital growth, chosen per fold
      from the training slates alone).
    * `tuning`: per fold, the push `(a, b)`, the deflation `d` for every `k`,
      and the `k` chosen. Their spread across folds is how stable the
      amounts are.

    `push_targets` restricts the push to those stats: it is fitted on their rows
    alone and every other proposition keeps its raw `p`. The deflation is still
    learnt on, and applied to, every selected position. `deflate_by_target` learns one
    deflation per stat instead, each pulled towards the pooled value by
    `deflate_prior` pseudo-observations so a stat with few selections cannot run wild.
    """
    y_all, p_all, o_all = s["y"].to_numpy(), s["p"].to_numpy(), s["o"].to_numpy()
    groups = s[group].astype(str).to_numpy()
    lv_all = pd.Categorical(s["target"]).codes.astype(int) if per_target else None
    tcode = pd.Categorical(s["target"]).codes.astype(int)
    on = (s["target"].isin(push_targets).to_numpy() if push_targets is not None
          else np.ones(len(s), bool))

    def learn_d(pp, y, sel, tc) -> np.ndarray:
        """Deflation per stat code, as an array indexed by code."""
        n_codes = int(tcode.max()) + 1
        if not sel.any():
            return np.zeros(n_codes)
        gap = pp[sel] - y[sel]
        pooled = float(gap.mean())
        if not deflate_by_target:
            return np.full(n_codes, pooled)
        out = np.full(n_codes, pooled)
        for c in range(n_codes):
            m = tc[sel] == c
            out[c] = (gap[m].sum() + deflate_prior * pooled) / (m.sum() + deflate_prior)
        return out

    def fit(idx):
        idx = idx[on[idx]]
        if per_target:
            return fit_recalibration(y_all[idx], p_all[idx], lv_all[idx], ridge=ridge)
        return fit_recalibration(y_all[idx], p_all[idx])

    def pushed(w, idx, k):
        if not per_target:
            out = push(w, p_all[idx], k)
        else:
            K = (len(w) - 2) // 2
            lv = lv_all[idx]
            ww = np.column_stack([w[0] + w[2:2 + K][lv], w[1] + w[2 + K:2 + 2 * K][lv]])
            x = logit(p_all[idx])
            out = expit(x + k * (ww[:, 0] + (ww[:, 1] - 1.0) * x))
        return np.where(on[idx], out, p_all[idx])

    rows, tune = [], []
    slates = np.unique(groups)
    for g in slates:
        tr = np.flatnonzero(groups != g)
        te = np.flatnonzero(groups == g)

        # Inner loop: out-of-sample pushed p for every training row, per k.
        inner = {k: np.full(len(tr), np.nan) for k in ks}
        for h in np.unique(groups[tr]):
            fit_idx = tr[groups[tr] != h]
            pos = np.flatnonzero(groups[tr] == h)
            w = fit(fit_idx)
            for k in ks:
                inner[k][pos] = pushed(w, tr[pos], k)

        sub = s.iloc[tr]
        d_by_k, inner_profit = {}, {}
        for k in ks:
            sel = select(sub, inner[k])
            d_by_k[k] = learn_d(inner[k], y_all[tr], sel, tcode[tr])
            p_st = np.clip(inner[k] - d_by_k[k][tcode[tr]], EPS, 1 - EPS)
            f = kelly_stakes(p_st, o_all[tr], frac) * sel
            ret = o_all[tr] * y_all[tr] - 1
            inner_profit[k] = sum(slate_growth(f[groups[tr] == h], ret[groups[tr] == h], cap)
                                  for h in np.unique(groups[tr]))
        k_best = max(ks, key=lambda k: inner_profit[k])

        w = fit(tr)
        cats = pd.Categorical(s["target"]).categories
        tune.append({"slate": g, "a": float(w[0]), "b": float(w[1]), "k_chosen": k_best,
                     **{f"d@{k:g}": float(np.average(d_by_k[k][tcode[tr]])) for k in ks},
                     **({f"d@{k:g}:{c}": float(d_by_k[k][i]) for k in ks
                         for i, c in enumerate(cats)} if deflate_by_target else {})})

        y, o = y_all[te], o_all[te]
        for k in ks:
            pp = pushed(w, te, k)
            sel = select(s.iloc[te], pp)
            d_row = d_by_k[k][tcode[te]]
            for label, d in ((f"k={k:g} push only", 0.0 * d_row),
                             (f"k={k:g} push+deflate", d_row)):
                p_st = np.clip(pp - d, EPS, 1 - EPS)
                rows.append({"slate": g, "strategy": label, "k": k,
                             "deflate": float(d[sel].mean()) if sel.any() else 0.0,
                             **_position_stats(y, o, p_st, sel, frac, cap)})
            if k == k_best:
                p_st = np.clip(pp - d_row, EPS, 1 - EPS)
                rows.append({"slate": g, "strategy": "tuned", "k": k,
                             "deflate": float(d_row[sel].mean()) if sel.any() else 0.0,
                             **_position_stats(y, o, p_st, sel, frac, cap)})
    return pd.DataFrame(rows), pd.DataFrame(tune)


def backtest_summary(per_slate: pd.DataFrame, baseline: str = "k=0 push only",
                     n_boot: int = 5000, seed: int = 2026) -> pd.DataFrame:
    """Totals per strategy, with slate-bootstrap intervals on the growth gain over `baseline`.

    `growth` is the summed log growth over every slate (`slate_growth`), so
    ``exp(growth)`` is the capital multiple. `gain_*` compare it with `baseline`.

    The slate is the unit resampled because it is the unit the strategy is
    tuned and applied on, and the positions inside one slate share a matchday.
    """
    piv = per_slate.pivot_table(index="slate", columns="strategy", values="log_growth")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(piv), size=(n_boot, len(piv)))
    rows = []
    for name, g in per_slate.groupby("strategy", sort=False):
        t = g[["positions", "sized_positions", "flat_pnl", "stake", "kelly_pnl", "kelly_expected",
               "said", "landed"]].sum()
        diff = (piv[name] - piv[baseline]).to_numpy()
        boot = diff[idx].sum(axis=1)
        rows.append({
            "strategy": name, "k": g["k"].iloc[0] if name != "tuned" else np.nan,
            "mean_deflate": float(g["deflate"].mean()),
            "positions": int(t["positions"]), "sized_positions": int(t["sized_positions"]),
            "said": t["said"] / t["positions"], "landed": t["landed"] / t["positions"],
            "flat_roi": t["flat_pnl"] / t["positions"],
            "kelly_roi": t["kelly_pnl"] / t["stake"] if t["stake"] else np.nan,
            "kelly_pnl": t["kelly_pnl"], "kelly_expected": t["kelly_expected"],
            "growth": float(g["log_growth"].sum()),
            "capital_x": float(np.exp(g["log_growth"].sum())),
            "gain_vs_today": float(diff.sum()),
            "gain_lo": float(np.percentile(boot, 2.5)), "gain_hi": float(np.percentile(boot, 97.5)),
            "slates_better": int((diff > 1e-12).sum()), "slates_worse": int((diff < -1e-12).sum()),
        })
    return pd.DataFrame(rows)



# --- 5. the same question through the real portfolio builder --------------
#
# Everything above scores propositions one at a time. A slate is not traded that
# way: `06_Split` runs `portfolio.search` (qualify, at most two legs per match
# priced with `PAIR_CORRELATION`, growth split, dominance), picks one portfolio
# with the notebook's filter, and sizes it at `growth.suggested_fraction` --
# where `PESSIMISM_B` is the deflation hook. This section replays each held-out
# slate through exactly that code.
#
# The push and the deflation are learnt at proposition level on the *other*
# slates, as in section 4; only the scoring goes through the builder. That is
# deliberate: learning them through the builder would need a full search per
# training slate per fold.

# The slates as they were actually run, one per matchday: R018 is dropped
# because R019 re-ran the same 28 matches. Nothing is deduplicated, so each
# replayed slate is exactly the card that was priced.
AS_RUN_SLATES: tuple[str, ...] = ("R006", "R007", "R010", "R011", "R012", "R013", "R015",
                                  "R016", "R017", "R019", "R020", "R021", "R022")

PICK_RULE = dict(min_pct_expected_return=1.05, min_p_over_100=0.70, max_pct_sd=0.25,
                 sort_by="p_over_100")
LEG_VAR = 11
# `portfolio.GRID` is 16,384, and at that resolution one slate's search takes ~15
# minutes, almost all of it in `threshold_probs_batch`. The replay uses a coarser
# grid; `validate_grid` checks it picks the same portfolio as the full one.
REPLAY_GRID = 2048


def _filled(rows: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """A slate's rows as the filled odds form `portfolio.search` reads.

    The best price goes in the first book column and the rest are left blank, so
    `best_price` recovers exactly the `o` the ledger banked. `_rid` rides along
    so each leg can be traced back to its outcome.
    """
    from .staking import BOOK_COLUMNS

    out = pd.DataFrame({
        "sheet_code": rows["sheet_code"].astype(str).to_numpy(),
        "label": rows["label"].astype(str).to_numpy(),
        "p": np.asarray(p, float),
        "date": rows["fixture_date"].to_numpy(),
        "league": rows["league"].to_numpy(),
        "home_team": rows["home_team"].astype(str).to_numpy(),
        "away_team": rows["away_team"].astype(str).to_numpy(),
        "_rid": np.arange(len(rows)),
    })
    for c in BOOK_COLUMNS:
        out[c] = np.nan
    out[BOOK_COLUMNS[0]] = rows["o"].to_numpy()
    return out


def replay_slate(rows: pd.DataFrame, p_select: np.ndarray, biases: dict[str, np.ndarray],
                 *, leg_var: int = LEG_VAR, pick_rule: dict | None = None,
                 grid: int = REPLAY_GRID) -> list[dict]:
    """One slate through `06_Split`: search, pick, size, settle.

    `p_select` is the probability the builder sees -- it qualifies, pairs,
    splits and picks on it. `biases` maps a strategy name to a per-row deflation
    passed to `growth.suggested_fraction` as its `bias`, which is what
    `PESSIMISM_B` does today with one number for every leg. One search serves
    every bias, because deflation only changes the position size.

    Capital moves by ``f * (R - 1)``, where ``R`` is the portfolio's realised
    return per unit of capital allocated, and growth is ``log(1 + f (R - 1))``.
    """
    from . import portfolio as pf
    from .growth import growth_metrics, suggested_fraction

    rule = PICK_RULE if pick_rule is None else pick_rule
    res = pf.search(_filled(rows, p_select), leg_var=leg_var, grid=grid)
    scored = res["scored"]
    if scored.empty:
        return [{"strategy": k, "legs": 0, "f": 0.0, "R": np.nan, "growth": 0.0,
                 "expected_R": np.nan, "shots_legs": 0, "pairs": 0} for k in biases]
    picked = pf.filter_portfolios(scored, **rule)
    row = (picked if len(picked) else scored[scored["undominated"]]).iloc[0]
    combo = int(row["combo"])
    opts, picks = res["options"], res["picks"]
    stakes, prob, odds, rho = pf.stakes_for(picks[[combo]], opts, row["split"])

    rid = np.full(stakes.shape[1], -1)
    for j in range(opts.n_events):
        k = int(picks[combo, j])
        if k < 0:
            continue
        for slot, prop in enumerate(opts.members[j][k]):
            src = opts.props[(opts.props["event"] == j) & (opts.props["option"] == prop)]
            rid[opts.width * j + slot] = int(src["_rid"].iloc[0])
    live = rid >= 0
    y = np.zeros(stakes.shape[1])
    y[live] = rows["y"].to_numpy()[rid[live]]
    pay = stakes[0] * odds[0]
    R = float((pay * y).sum())
    targets = np.array([""] * stakes.shape[1], dtype=object)
    targets[live] = rows["target"].to_numpy()[rid[live]]

    gm = growth_metrics(stakes * odds, prob, rho=rho).iloc[0]
    out = []
    for name, b in biases.items():
        bias = np.zeros(stakes.shape[1])
        bias[live] = np.asarray(b, float)[rid[live]]
        f, _ = suggested_fraction(float(gm["f_drawdown"]), pay, prob[0], bias=bias)
        out.append({"strategy": name, "legs": int(live.sum()), "f": float(f), "R": R,
                    "picks": str(row["picks"]),
                    "growth": float(np.log1p(f * (R - 1.0))),
                    "expected_R": float((pay * prob[0]).sum()),
                    "shots_legs": int((targets == "shots").sum()),
                    # Matches holding two legs, and three: counted off the slot
                    # layout so it holds at either width.
                    "pairs": int(((prob[0].reshape(-1, opts.width) > 0).sum(axis=1) == 2).sum()),
                    "triples": int(((prob[0].reshape(-1, opts.width) > 0).sum(axis=1) == 3).sum()),
                    "f_drawdown": float(gm["f_drawdown"])})
    return out


def _fold_plan(train: pd.DataFrame, test: pd.DataFrame, *, shots_k: float = 1.0,
               all_k: float = 0.5, group: str = "run_code") -> dict:
    """Push and deflation for one held-out slate, learnt on `train` only.

    `test` is the held-out slate as it was run; `train` is every other slate's
    deduplicated rows with the held-out slate's matches removed, so a match
    re-priced on two overlapping runs can never sit on both sides.

    Returns, per *variant* (the probability the builder selects on), that
    probability for the held-out rows and the per-row deflation of each strategy
    using it.
    """
    s = pd.concat([train.assign(_te=False), test.assign(_te=True)], ignore_index=True)
    groups = s[group].astype(str).to_numpy()
    y, p = s["y"].to_numpy(), s["p"].to_numpy()
    shots = (s["target"] == "shots").to_numpy()
    te = s["_te"].to_numpy()
    tr = ~te

    def oos_push(mask_fit, k):
        """Out-of-sample pushed p for every training row (inner leave-one-slate-out)."""
        pp = p.copy()
        for h in np.unique(groups[tr]):
            fit_rows = tr & (groups != h) & mask_fit
            w = fit_recalibration(y[fit_rows], p[fit_rows])
            m = (groups == h) & mask_fit
            pp[m] = push(w, p[m], k)
        return pp

    def gap(pp, rows_mask):
        sel = select(s[tr], pp[tr]) & rows_mask[tr]
        return float((pp[tr][sel] - y[tr][sel]).mean()) if sel.any() else 0.0

    plan = {}
    # V0: today's probabilities.
    d_raw = gap(p, np.ones(len(s), bool))
    plan["raw"] = (p[te], {"today": np.zeros(te.sum()),
                           "deflate all": np.full(te.sum(), d_raw)})
    # V1: push shots only.
    pp_in = oos_push(shots, shots_k)
    w = fit_recalibration(y[tr & shots], p[tr & shots])
    p_te = np.where(shots[te], push(w, p[te], shots_k), p[te])
    d_other = gap(pp_in, ~shots)
    d_pooled = gap(pp_in, np.ones(len(s), bool))
    plan["shots push"] = (p_te, {
        "shots push, no deflation": np.zeros(te.sum()),
        "shots push, deflate others": np.where(shots[te], 0.0, d_other),
        "shots push, deflate all": np.full(te.sum(), d_pooled)})
    # V2: push every stat half way.
    pp_in = oos_push(np.ones(len(s), bool), all_k)
    w = fit_recalibration(y[tr], p[tr])
    plan["all push"] = (push(w, p[te], all_k),
                        {"half push all, deflate all": np.full(te.sum(),
                                                               gap(pp_in, np.ones(len(s), bool)))})
    # V3/V4: the production rule -- a logit shift on shots only, per league
    # (partially pooled) or pooled -- fitted on the training slates' raw p.
    trd = s[tr]
    fit = shots_shift(trd, ridges=(30.0,))
    lk = s["league"].astype(str).to_numpy()
    by_league = np.array([fit["shift"].get(l, fit["pooled"]) for l in lk[te]])
    pooled_c, _ = fit_logit_shift(y[tr & shots], p[tr & shots])
    plan["shots shift by league"] = (
        np.where(shots[te], expit(logit(p[te]) + by_league), p[te]),
        {"shots shift by league, no deflation": np.zeros(te.sum())})
    plan["shots shift pooled"] = (
        np.where(shots[te], expit(logit(p[te]) + pooled_c), p[te]),
        {"shots shift pooled, no deflation": np.zeros(te.sum())})
    plan["_deflation"] = {"deflate all": d_raw, "deflate others (shots pushed)": d_other,
                          "deflate all (shots pushed)": d_pooled}
    return plan


def _replay_task(args):
    rows, p_sel, biases, held, variant = args
    out = replay_slate(rows, p_sel, biases)
    for r in out:
        r.update(slate=held, variant=variant, matches=int(rows["event"].nunique()))
    return out


def portfolio_backtest(s: pd.DataFrame, *, slates=AS_RUN_SLATES, as_run: pd.DataFrame | None = None,
                       workers: int = 4, cache: Path | str | None = None,
                       refresh: bool = False, group: str = "run_code",
                       variants: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Every held-out slate through the real builder, under each strategy.

    Each slate is replayed **as it was run** (`as_run`, default
    `load_study_set(dedupe=False)`), the card that was actually priced; the push
    and deflation are learnt on `s`, the deduplicated set, minus that slate's
    matches. Slow -- one full `portfolio.search` per slate per variant -- so it
    runs in `workers` processes and is cached. A cache is reused only if it
    covers the same slates; pass `refresh=True` to recompute regardless.
    """
    from concurrent.futures import ProcessPoolExecutor

    slates = sorted(slates, key=_run_number)
    path = Path(cache) if cache else None
    if path is not None and path.exists() and not refresh:
        cached = pd.read_parquet(path)
        if sorted(cached["slate"].unique(), key=_run_number) == slates:
            return cached

    raw = load_study_set(dedupe=False)[0] if as_run is None else as_run
    tasks, deflations = [], []
    for held in slates:
        rows = raw[raw[group].astype(str) == held].reset_index(drop=True)
        train = s[(s[group].astype(str) != held) & ~s["event"].isin(rows["event"])]
        plan = _fold_plan(train, rows, group=group)
        deflations.append({"slate": held, **plan.pop("_deflation")})
        for variant, (p_sel, biases) in plan.items():
            if variants is None or variant in variants:
                tasks.append((rows, p_sel, biases, held, variant))

    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = [r for batch in ex.map(_replay_task, tasks) for r in batch]
    out = pd.DataFrame(results).merge(pd.DataFrame(deflations), on="slate", how="left")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path, index=False)
    return out


def portfolio_summary(bt: pd.DataFrame, baseline: str = "today", n_boot: int = 5000,
                      seed: int = 2026) -> pd.DataFrame:
    """Capital multiple per strategy, and its slate-bootstrap gain over `baseline`."""
    piv = bt.pivot_table(index="slate", columns="strategy", values="growth")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(piv), size=(n_boot, len(piv)))
    rows = []
    for name in piv.columns:
        g = bt[bt["strategy"] == name]
        diff = (piv[name] - piv[baseline]).to_numpy()
        boot = diff[idx].sum(axis=1)
        rows.append({"strategy": name, "slates": len(g),
                     "mean_legs": g["legs"].mean(), "mean_shots_legs": g["shots_legs"].mean(),
                     "mean_f": g["f"].mean(), "mean_R": g["R"].mean(),
                     "mean_expected_R": g["expected_R"].mean(),
                     "capital_x": float(np.exp(g["growth"].sum())),
                     "gain_lo": float(np.exp(np.percentile(boot, 2.5))),
                     "gain_hi": float(np.exp(np.percentile(boot, 97.5))),
                     "slates_better": int((diff > 1e-12).sum()),
                     "slates_worse": int((diff < -1e-12).sum())})
    return pd.DataFrame(rows).sort_values("capital_x", ascending=False).reset_index(drop=True)



# --- 6. a whole frontier, settled -----------------------------------------


def frontier_outcomes(rows: pd.DataFrame, *, triples: bool, leg_var: int = LEG_VAR,
                      grid: int = REPLAY_GRID, pick_rule: dict | None = None,
                      max_leg_stake: float | None = None) -> dict:
    """One slate through `portfolio.search`, with every undominated portfolio settled.

    Returns ``{"frontier": DataFrame, "pick": Series, "info": dict}``. The frontier
    frame carries each portfolio's promised return (`pct_expected_return`), its
    realised return per unit allocated (`realised_R`), how many legs it holds and
    how many of its matches it holds as a pair or a triple. The pick is the 06
    rule's choice, sized at `growth.suggested_fraction`.
    """
    from . import portfolio as pf
    from .growth import growth_metrics, suggested_fraction

    from .config import MAX_LEG_STAKE

    rule = PICK_RULE if pick_rule is None else pick_rule
    cap = MAX_LEG_STAKE if max_leg_stake is None else max_leg_stake
    res = pf.search(_filled(rows, rows["p"].to_numpy()), leg_var=leg_var, grid=grid,
                    triples=triples, max_leg_stake=cap)
    scored, opts, picks = res["scored"], res["options"], res["picks"]
    front = scored[scored["undominated"]].copy()
    W = opts.width

    # Outcome of every option slot, laid out exactly as `stakes_for` lays out p.
    y_rows = rows["y"].to_numpy()
    rid_of = {(int(e), int(o)): int(r) for e, o, r in
              zip(opts.props["event"], opts.props["option"], opts.props["_rid"])}
    combos = front["combo"].to_numpy()
    stakes, prob, odds, rho = pf.stakes_for(picks[combos], opts, front["split"].iloc[0], cap)
    Y = np.zeros_like(prob)
    multi = np.zeros(len(combos), int)
    tri = np.zeros(len(combos), int)
    for r_i, c in enumerate(combos):
        for j in range(opts.n_events):
            k = int(picks[c, j])
            if k < 0:
                continue
            mem = opts.members[j][k]
            multi[r_i] += len(mem) >= 2
            tri[r_i] += len(mem) == 3
            for slot, prop in enumerate(mem):
                Y[r_i, W * j + slot] = y_rows[rid_of[(j, prop)]]
    front["realised_R"] = (stakes * odds * Y).sum(axis=1)
    front["multi_events"] = multi
    front["triple_events"] = tri

    picked = pf.filter_portfolios(scored, **rule)
    row = (picked if len(picked) else front).iloc[0]
    pos = int(np.flatnonzero(combos == int(row["combo"]))[0])
    gm = growth_metrics((stakes * odds)[[pos]], prob[[pos]],
                        rho=None if rho is None else rho[[pos]]).iloc[0]
    f, _ = suggested_fraction(float(gm["f_drawdown"]), (stakes * odds)[pos], prob[pos])
    pick = front.iloc[pos].copy()
    pick["f"] = f
    pick["growth"] = float(np.log1p(f * (pick["realised_R"] - 1.0)))
    info = {**res["info"], "options": int(opts.sizes.sum()), "width": W,
            "triple_options": int(sum((x > 0).sum() for x in opts.p3))}
    return {"frontier": front, "pick": pick, "info": info}



# --- 7. triples on and off, through the builder ---------------------------



def _triples_job(args) -> dict:
    run, tri = args
    s, _ = load_study_set(dedupe=False)
    rows = s[s["run_code"] == run].reset_index(drop=True)
    out = frontier_outcomes(rows, triples=tri)
    f, pk = out["frontier"], out["pick"]
    return {"slate": run, "triples": tri, "matches": int(rows["event"].nunique()),
            "options": out["info"]["options"], "triple_options": out["info"]["triple_options"],
            "frontier": len(f), "frontier_with_triple": int((f["triple_events"] > 0).sum()),
            "front_promised_med": float(f["pct_expected_return"].median()),
            "front_delivered_med": float(f["realised_R"].median()),
            "front_delivered_mean": float(f["realised_R"].mean()),
            "front_profitable": float((f["realised_R"] > 1).mean()),
            **{f"pick_{k}": pk[k] for k in ("legs", "events", "multi_events", "triple_events",
                                            "pct_expected_return", "p_over_100", "realised_R",
                                            "f", "growth", "picks")}}


def triples_backtest(slates=AS_RUN_SLATES, *, workers: int = 4, cache: Path | str | None = None,
                     refresh: bool = False) -> pd.DataFrame:
    """Every slate through the builder twice, triples off and on, settled.

    One row per slate x setting: the 06 pick (legs, promised, delivered, size,
    growth) and the whole frontier (median promised and delivered, share
    profitable). ~25 minutes on 4 cores, so cached like `portfolio_backtest`.
    """
    from concurrent.futures import ProcessPoolExecutor

    path = Path(cache) if cache else None
    if path is not None and path.exists() and not refresh:
        cached = pd.read_parquet(path)
        if sorted(cached["slate"].unique()) == sorted(slates):
            return cached
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = pd.DataFrame(list(ex.map(_triples_job, [(r, t) for r in slates
                                                      for t in (False, True)])))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path, index=False)
    return out


def triples_summary(tb: pd.DataFrame, n_boot: int = 5000, seed: int = 2026) -> pd.DataFrame:
    """Per slate, off against on: the pick's growth and the frontier's median delivered."""
    w = tb.pivot(index="slate", columns="triples",
                 values=["pick_growth", "pick_realised_R", "pick_legs", "pick_triple_events",
                         "front_delivered_med", "triple_options", "matches"])
    out = pd.DataFrame({
        "matches": w[("matches", False)], "triple_options": w[("triple_options", True)],
        "pick_triples": w[("pick_triple_events", True)],
        "legs_off": w[("pick_legs", False)], "legs_on": w[("pick_legs", True)],
        "R_off": w[("pick_realised_R", False)], "R_on": w[("pick_realised_R", True)],
        "growth_off": w[("pick_growth", False)], "growth_on": w[("pick_growth", True)],
        "front_med_off": w[("front_delivered_med", False)],
        "front_med_on": w[("front_delivered_med", True)],
    })
    out = out.reindex(sorted(out.index, key=_run_number))
    d = (out["growth_on"] - out["growth_off"]).to_numpy()
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), size=(n_boot, len(d)))].sum(axis=1)
    out.attrs.update(capital_off=float(np.exp(out["growth_off"].sum())),
                     capital_on=float(np.exp(out["growth_on"].sum())),
                     gain=float(np.exp(d.sum())),
                     gain_lo=float(np.exp(np.percentile(boot, 2.5))),
                     gain_hi=float(np.exp(np.percentile(boot, 97.5))),
                     better=int((d > 1e-9).sum()), worse=int((d < -1e-9).sum()),
                     same=int((np.abs(d) <= 1e-9).sum()))
    return out



# --- 8. the per-leg cap, across every frontier portfolio ------------------

LEG_CAPS: tuple[float, ...] = (1.0, 0.25, 0.20, 0.15, 0.10, 0.07)


def _leg_cap_job(args) -> dict:
    run, caps = args
    from . import portfolio as pf

    s, _ = load_study_set(dedupe=False)
    rows = s[s["run_code"] == run].reset_index(drop=True)
    res = pf.search(_filled(rows, rows["p"].to_numpy()), leg_var=LEG_VAR, grid=REPLAY_GRID,
                    triples=False)
    sc, opts, picks = res["scored"], res["options"], res["picks"]
    front = sc[sc["undominated"]]
    combos = front["combo"].to_numpy()
    raw, p, o, rho = pf.stakes_for(picks[combos], opts, front["split"].iloc[0], max_leg_stake=1.0)
    act = p > 0
    rid = {(int(e), int(k)): int(r) for e, k, r in
           zip(opts.props["event"], opts.props["option"], opts.props["_rid"])}
    Y = np.zeros_like(p)
    for i, c in enumerate(combos):
        for j in range(opts.n_events):
            k = int(picks[c, j])
            if k < 0:
                continue
            for slot, prop in enumerate(opts.members[j][k]):
                Y[i, opts.width * j + slot] = rows["y"].to_numpy()[rid[(j, prop)]]
    out = {"slate": run, "frontier": len(combos)}
    for cap in caps:
        stk = pf.cap_stakes(raw, act, cap)
        er, var = pf.metrics(stk, p, o, rho)
        R = (stk * o * Y).sum(axis=1)
        out.update({f"R@{cap:g}": float(R.mean()), f"E@{cap:g}": float(er.mean()),
                    f"sd@{cap:g}": float(np.sqrt(var).mean()),
                    f"profitable@{cap:g}": float((R > 1).mean())})
    return out


def leg_cap_backtest(slates=AS_RUN_SLATES, caps=LEG_CAPS, *, workers: int = 4,
                     cache: Path | str | None = None, refresh: bool = False) -> pd.DataFrame:
    """Every undominated portfolio on every slate, re-weighted under each per-leg cap and settled.

    The growth split is computed uncapped, then `portfolio.cap_stakes` applies
    each cap (with the usual `CAP_RELIEF` for short rows). The legs are held
    fixed, so this isolates what the cap does to the stakes, not which portfolio
    the pick rule would then choose. One row per slate; columns per cap are the
    frontier's mean delivered return (`R@`), mean promised (`E@`), mean sd, and
    share profitable. Pairs only, as the slates were run.
    """
    from concurrent.futures import ProcessPoolExecutor

    path = Path(cache) if cache else None
    if path is not None and path.exists() and not refresh:
        cached = pd.read_parquet(path)
        if sorted(cached["slate"]) == sorted(slates):
            return cached
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = pd.DataFrame(list(ex.map(_leg_cap_job, [(r, tuple(caps)) for r in slates])))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path, index=False)
    return out



def _cap_replay_job(args) -> dict:
    run, cap = args
    s, _ = load_study_set(dedupe=False)
    rows = s[s["run_code"] == run].reset_index(drop=True)
    out = frontier_outcomes(rows, triples=True, max_leg_stake=cap)
    f, pk = out["frontier"], out["pick"]
    return {"slate": run, "cap": cap, "frontier": len(f),
            "front_R_mean": float(f["realised_R"].mean()),
            "front_E_mean": float(f["pct_expected_return"].mean()),
            "front_profitable": float((f["realised_R"] > 1).mean()),
            **{f"pick_{k}": pk[k] for k in ("legs", "pct_expected_return", "p_over_100",
                                            "realised_R", "f", "growth", "max_leg_stake")}}


def leg_cap_replay(slates=AS_RUN_SLATES, caps=(0.15, 0.10), *, workers: int = 4,
                   cache: Path | str | None = None, refresh: bool = False) -> pd.DataFrame:
    """Every slate searched, picked and sized under each per-leg cap, then settled.

    Unlike `leg_cap_backtest`, the cap is in force *during* the search, so
    dominance and the 06 pick respond to it -- the test that decides whether to
    change `MAX_LEG_STAKE`. Triples on. Cached; ~35 minutes fresh.
    """
    from concurrent.futures import ProcessPoolExecutor

    path = Path(cache) if cache else None
    if path is not None and path.exists() and not refresh:
        cached = pd.read_parquet(path)
        if sorted(cached["slate"].unique()) == sorted(slates):
            return cached
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = pd.DataFrame(list(ex.map(_cap_replay_job, [(r, c) for r in slates for c in caps])))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path, index=False)
    return out


def leg_cap_summary(cr: pd.DataFrame, baseline: float = 0.15, n_boot: int = 5000,
                    seed: int = 2026) -> pd.DataFrame:
    """Per cap: the pick's capital multiple and its gain over `baseline`, and the frontier's averages."""
    piv = cr.pivot(index="slate", columns="cap", values="pick_growth")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(piv), size=(n_boot, len(piv)))
    rows = []
    for cap in sorted(piv.columns, reverse=True):
        g = cr[cr["cap"] == cap]
        d = (piv[cap] - piv[baseline]).to_numpy()
        boot = d[idx].sum(axis=1)
        rows.append({"cap": cap, "pick_capital_x": float(np.exp(piv[cap].sum())),
                     "gain_vs_baseline": float(np.exp(d.sum())),
                     "gain_lo": float(np.exp(np.percentile(boot, 2.5))),
                     "gain_hi": float(np.exp(np.percentile(boot, 97.5))),
                     "slates_better": int((d > 1e-9).sum()), "slates_worse": int((d < -1e-9).sum()),
                     "frontier_delivered": g["front_R_mean"].mean(),
                     "frontier_promised": g["front_E_mean"].mean(),
                     "frontier_profitable": g["front_profitable"].mean()})
    return pd.DataFrame(rows)



# --- 9. the production shots shift ----------------------------------------
#
# What `config.SHOTS_LOGIT_SHIFT` holds: one additive shift on the logit of every
# shots line probability, per league, pulled towards the pooled shift by a ridge
# penalty. A plain shift rather than ``a + b logit p`` because it scores as well
# out of sample (0.5946 against 0.5955 log loss on R006-R022) with one number per
# league instead of two.

SHIFT_RIDGE_GRID: tuple[float, ...] = (0.0, 10.0, 30.0, 100.0, 300.0, 1000.0)


def fit_logit_shift(y, p, levels=None, *, ridge: float = 0.0) -> tuple[float, np.ndarray]:
    """``(pooled, per_level)`` shifts maximising likelihood of ``y`` under ``expit(logit p + c)``.

    With `levels`, each level's shift is the pooled one plus an offset that the
    ridge penalty ``ridge/2 * offset**2`` pulls back to zero -- the same partial
    pooling `fit_recalibration` uses for contexts.
    """
    y = np.asarray(y, float)
    x = logit(p)
    lv = np.zeros(len(y), int) if levels is None else np.asarray(levels, int)
    k = int(lv.max()) + 1

    def loss(w):
        e = x + w[0] + w[1:][lv]
        q = expit(e)
        r = q - y
        g = np.empty_like(w)
        g[0] = r.sum()
        g[1:] = np.bincount(lv, weights=r, minlength=k) + ridge * w[1:]
        return np.sum(np.logaddexp(0, e) - y * e) + 0.5 * ridge * np.sum(w[1:] ** 2), g

    if levels is None:
        w = minimize(loss, np.zeros(1 + k), jac=True, method="L-BFGS-B",
                     bounds=[(None, None)] + [(0, 0)] * k).x
    else:
        w = minimize(loss, np.zeros(1 + k), jac=True, method="L-BFGS-B").x
    return float(w[0]), w[0] + w[1:]


def shots_shift(s: pd.DataFrame, *, group: str = "run_code", ridges=SHIFT_RIDGE_GRID) -> dict:
    """The per-league shots shift for `config.SHOTS_LOGIT_SHIFT`, with its evidence.

    Fitted on **raw** model probabilities (`p_raw`), so a refit after the shift
    is live measures the model, not the model plus the last shift. The ridge is
    chosen by leave-one-slate-out log loss. Returns the shifts, the pooled shift
    (the default for leagues with no shots prices), the ridge, and the CV log
    loss of raw, pooled and per-league.
    """
    sh = s[s["target"] == "shots"].reset_index(drop=True)
    y = sh["y"].to_numpy()
    p = (sh["p_raw"] if "p_raw" in sh else sh["p"]).to_numpy()
    cat = pd.Categorical(sh["league"])
    lv = cat.codes.astype(int)
    groups = sh[group].astype(str).to_numpy()

    def cv(ridge):
        tot = 0.0
        for g in np.unique(groups):
            tr, te = groups != g, groups == g
            pooled, per = fit_logit_shift(y[tr], p[tr], lv[tr], ridge=ridge)
            tot += _ll(y[te], expit(logit(p[te]) + per[lv[te]])).sum()
        return tot / len(y)

    scores = {r: cv(r) for r in ridges}
    best = min(scores, key=scores.get)
    pooled, per = fit_logit_shift(y, p, lv, ridge=best)
    raw_ll = float(_ll(y, p).mean())
    pooled_cv = 0.0
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        c, _ = fit_logit_shift(y[tr], p[tr])
        pooled_cv += _ll(y[te], expit(logit(p[te]) + c)).sum()
    return {"shift": {str(c): round(float(v), 3) for c, v in zip(cat.categories, per)},
            "pooled": round(float(pooled), 3), "ridge": best,
            "n": {str(c): int((lv == i).sum()) for i, c in enumerate(cat.categories)},
            "cv_logloss": {"raw": raw_ll, "pooled": pooled_cv / len(y),
                           "per_league": scores[best]},
            "cv_by_ridge": scores}



def shift_monitor(s: pd.DataFrame, by=("run_code", "league")) -> pd.DataFrame:
    """Shots propositions priced *with* the shift live: raw model, shifted value, outcome.

    One row per run and league, for runs where `p` differs from `p_raw` on shots
    (i.e. captured after `SHOTS_LOGIT_SHIFT` went live). `gap_raw` and
    `gap_shifted` are hit rate minus the mean of each: if the shift is right the
    second sits near zero while the first stays positive. Empty until the first
    run priced with the shift has been graded.
    """
    sh = s[(s["target"] == "shots") & ((s["p"] - s["p_raw"]).abs() > 1e-9)]
    if sh.empty:
        return pd.DataFrame(columns=[*by, "n", "hit_rate", "p_raw", "p", "gap_raw", "gap_shifted"])
    g = sh.groupby(list(by), observed=True)
    out = g.agg(n=("y", "size"), hit_rate=("y", "mean"), p_raw=("p_raw", "mean"), p=("p", "mean"))
    out["gap_raw"] = out["hit_rate"] - out["p_raw"]
    out["gap_shifted"] = out["hit_rate"] - out["p"]
    return out.reset_index()


__all__ = [
    "load_study_set", "run_models", "grade", "pair_frame", "rho_table", "placebo_correlation",
    "correlation_vs_config", "reliability_frame", "gap_table", "market_comparison",
    "selection_check",
    "fit_recalibration", "apply_recalibration", "loso_predictions",
    "recalibration_summary", "recalibration_by_level", "fitted_constants",
    "push", "select", "kelly_stakes", "slate_growth", "SLATE_CAP",
    "replay_slate", "portfolio_backtest", "portfolio_summary", "frontier_outcomes",
    "AS_RUN_SLATES", "triples_backtest", "triples_summary", "LEG_CAPS",
    "leg_cap_backtest", "leg_cap_replay", "leg_cap_summary", "fit_logit_shift",
    "shots_shift", "shift_monitor", "push_deflate_backtest", "backtest_summary",
]
