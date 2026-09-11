"""What the ledger has to say once the matches have been played.

Four questions, in the order they are worth asking. Each is a pure function over
the settled tables `fpp.ledger` holds, and each returns a frame `07_Analysis`
writes out and displays.

1. `model_vs_market` -- is the model better than the price it is betting into?
   Nothing below matters if it is not: no portfolio rule rescues probabilities
   that lose to `1/o`.
2. `live_calibration` -- which markets and which lines can be trusted. Graded by
   `fpp.calibration`, the same code `04_Evaluation` uses, so a verdict here and a
   verdict there mean the same thing.
3. `edge_buckets` / `odds_buckets` -- does a claimed edge turn into money.
4. `strategy_returns` / `frontier_attribution` / `growth_check` -- which way of
   choosing a portfolio off the frontier actually pays.

**On sample size.** A slate contributes thousands of proposition rows and
thousands of portfolios, but its portfolios are overlapping combinations of the
same few dozen propositions and they rise and fall together. The effective sample
for anything portfolio-level is therefore the number of *slates*, not the number
of rows, and every frame below reports `n_slates` next to `n` so a reader cannot
mistake one for the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from . import ledger
from .calibration import grade, tables_from_stream, wilson_interval
from .metrics import binary_metrics, bootstrap_ci

# Edge bins straddle 1.0 because that is where the meaning changes: below it the
# model says the price is short, above it the proposition is backable at all.
EDGE_BINS: tuple[float, ...] = (0.0, 0.90, 0.95, 1.00, 1.02, 1.05, 1.10, 1.20, np.inf)
ODDS_BINS: tuple[float, ...] = (1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0, np.inf)

@dataclass(frozen=True)
class Rule:
    """One way of picking a single portfolio off the frontier.

    A `(column, sense)` pair covers "take the biggest" and "take the smallest"
    and nothing else, which is why `quantile` exists: "the portfolio at the 75th
    percentile of expected return" is not an extreme, and backing off the extreme
    is exactly the hypothesis the first real slate argues for.
    """
    column: str
    sense: int = +1                  # +1 take the max, -1 take the min
    quantile: float | None = None    # or: the portfolio nearest this quantile

    def pick(self, pool: pd.DataFrame):
        col = pool[self.column]
        if col.isna().all():
            return None
        if self.quantile is None:
            return pool.loc[col.idxmax() if self.sense > 0 else col.idxmin()]
        # Nearest to the target rather than interpolated: the answer has to be a
        # portfolio that exists and can be backed, not a point between two.
        return pool.loc[(col - col.quantile(self.quantile)).abs().idxmin()]


# One portfolio off the frontier per rule per slate. These are the rules worth
# arguing about -- `06_Split` deliberately ends in a set to filter rather than a
# pick, and this is the table that says which filter would have paid.
STRATEGIES: dict[str, Rule] = {
    "max expected return": Rule("expected_return_pct", +1),
    "max P(profit)": Rule("p_over_100", +1),
    "min variance": Rule("variance", -1),
    "max P(>110%)": Rule("p_over_110", +1),
    "most legs": Rule("legs", +1),
    "fewest legs": Rule("legs", -1),
    "max protective growth": Rule("g_g_protective", +1),
    "max sharpe": Rule("_sharpe", +1),
    # Edge capacity: the quantity the growth split maximises, so this is the
    # strategy the model itself would pick. Null on runs before it existed, which
    # `strategy_picks` skips the same way it skips any other missing column.
    "max capacity": Rule("capacity", +1),
    # Back the likelier things, independent of price and of the model's own
    # expected-return arithmetic.
    "likeliest legs": Rule("median_leg_p", +1),
    "longest-shot legs": Rule("median_leg_p", -1),
    # The favourite-longshot question, asked at portfolio level.
    "shortest average odds": Rule("mean_leg_odds", -1),
    "longest average odds": Rule("mean_leg_odds", +1),
    # Off the extreme rather than on it.
    "75th pct expected return": Rule("expected_return_pct", quantile=0.75),
    "median expected return": Rule("expected_return_pct", quantile=0.50),
}


# --- the settled view -----------------------------------------------------


def settled(root=None) -> pd.DataFrame:
    """Every settled proposition, with its run's date attached.

    `won` arrives as a nullable boolean and every metric below wants a float, so
    it is cast once here rather than at eight call sites. `market_p` is `1/o` and
    is **not** normalised -- see `model_vs_market` for why that is the honest
    thing to do rather than a shortcut.
    """
    props = ledger.read_table("propositions", root)
    runs = ledger.read_table("runs", root)
    if props.empty:
        return props
    s = props[props["won"].notna()].copy()
    if s.empty:
        return s
    if not runs.empty:
        s = s.merge(runs[["run_code", "run_date"]], on="run_code", how="left")
    s["y"] = s["won"].astype(float)
    s["market_p"] = 1.0 / s["o"]
    s["roi"] = s["o"] * s["y"] - 1.0        # return on one unit staked flat
    s["venue"] = s["scope"]
    s["league"] = s["league_key"]
    return s.reset_index(drop=True)


def _slates(s: pd.DataFrame) -> int:
    return int(s["run_code"].nunique()) if "run_code" in s else 0


# --- 1. the model against the price ---------------------------------------


def model_vs_market(s: pd.DataFrame, *, by: str = "target", alpha: float = 0.05) -> pd.DataFrame:
    """Log loss and Brier for the model and for the price, on the same rows.

    `market_p` is the raw `1/o` of the best price on the form, which still
    carries the book's margin and therefore sums to more than one across a
    two-way market. That flatters the model, and deliberately so: we cannot strip
    the overround without the other side of the line, which the form never
    captures, and a comparison that is biased *towards* the thing being tested is
    worth more than one biased away from it. If the model still loses here, it
    loses.

    Significance follows `evaluate._significance`: a paired bootstrap CI on the
    mean of the differences and a Wilcoxon on their median, and `winner` names a
    side only when both agree. Log-loss differences are near zero on most rows
    and large on the minority where model and market genuinely disagree, so the
    mean can sit clearly away from zero while the median does not -- reporting
    one of them alone would misdescribe exactly the case this table exists to
    find.
    """
    if s.empty:
        return pd.DataFrame()
    groups: list[tuple[str, pd.DataFrame]] = [("all", s)]
    if by in s:
        groups += [(str(k), g) for k, g in s.groupby(by, observed=True)]

    rows = []
    for name, g in groups:
        y = g["y"].to_numpy()
        m = binary_metrics(y, g["p"].to_numpy(), "model")
        k = binary_metrics(y, g["market_p"].to_numpy(), "market")
        eps = 1e-15
        ll = lambda p: -(y * np.log(np.clip(p, eps, 1 - eps))
                         + (1 - y) * np.log(np.clip(1 - p, eps, 1 - eps)))
        diff = ll(g["market_p"].to_numpy()) - ll(g["p"].to_numpy())  # + => model wins
        lo, hi = bootstrap_ci(diff)
        try:
            _, pv = wilcoxon(ll(g["market_p"].to_numpy()), ll(g["p"].to_numpy()))
        except ValueError:
            pv = float("nan")
        wilcoxon_significant = bool(pv < alpha) if np.isfinite(pv) else False
        # Which side won, if either. A single `significant` boolean would read
        # False both when nothing separates them and when the *market* beats the
        # model decisively -- and those are opposite findings.
        if lo > 0 and wilcoxon_significant:
            winner = "model"
        elif hi < 0 and wilcoxon_significant:
            winner = "market"
        else:
            winner = "neither"
        rows.append({
            by if by in s else "segment": name,
            "n": m["n"], "n_slates": _slates(g),
            "base_rate": float(y.mean()), "model_p": float(g["p"].mean()),
            "market_p": float(g["market_p"].mean()),
            "model_logloss": m["logloss"], "market_logloss": k["logloss"],
            "model_brier": m["brier"], "market_brier": k["brier"],
            "model_auc": m["auc"], "market_auc": k["auc"],
            "mean_gain": float(diff.mean()), "median_gain": float(np.median(diff)),
            "win_rate": float((diff > 0).mean()),
            "ci_lo": lo, "ci_hi": hi, "wilcoxon_p": float(pv),
            "winner": winner,
            "model_beats_market": bool(winner == "model"),
        })
    return pd.DataFrame(rows)


# --- 2. calibration, live and forward -------------------------------------


def _direction(bins: pd.DataFrame) -> dict:
    """Split a market's calibration error into the safe half and the costly half.

    Every proposition on this book is an **Over**, so the two directions are not
    equivalent. Under-predicting means the bets win more often than the price you
    took implies -- error in your favour. Over-predicting is the one that costs
    money. `ece` weights both the same and cannot tell them apart, which on the
    first real slate graded two markets AVOID on error that was entirely in the
    safe direction.
    """
    if bins.empty:
        return {"ece_under": np.nan, "ece_over": np.nan, "bands_under": 0,
                "bands_over": 0, "net_direction": "unknown"}
    w = bins["n"] / bins["n"].sum()
    gap = bins["obs_freq"] - bins["pred_mean"]        # + => under-predicts
    under = float((w * gap.clip(lower=0)).sum())
    over = float((w * (-gap).clip(lower=0)).sum())
    return {
        "ece_under": under, "ece_over": over,
        "bands_under": int((bins["verdict"] == "under-predicts").sum()),
        "bands_over": int((bins["verdict"] == "over-predicts").sum()),
        "net_direction": "under (safe)" if under >= over else "over (costly)",
    }


def live_calibration(s: pd.DataFrame, *, pooled: bool = False
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(summary, bins)`` per target, graded by `fpp.calibration`.

    This is the first calibration number in the project that describes the lines
    actually bet. `04_Evaluation` grades `spec`'s fixed ladders -- 0.5/1.5/2.5/3.5
    for goals -- over a held-out season, but propositions are priced on dynamic
    eleven-line ladders centred on each team's own predicted rate, so no existing
    verdict covers `Shots - Hull - Over 9.5`.

    `pooled` collapses the line dimension. Early on that is the only view with
    enough rows per bin to say anything: a single slate spreads eleven shots
    lines across a dozen distinct values and every one of them is `insufficient`
    on its own.
    """
    if s.empty:
        return pd.DataFrame(), pd.DataFrame()
    summaries, bins = [], []
    for target, g in s.groupby("target", observed=True):
        stream = g[["p", "y", "league", "venue"]].copy()
        stream["line"] = "all" if pooled else g["line"].to_numpy()
        su, bi = tables_from_stream(stream, str(target), "live")

        # The symmetric verdict is left exactly as `04_Evaluation` computes it so
        # the two remain comparable; the directional one sits beside it.
        extra = []
        for row in su.to_dict("records"):
            b = bi[(bi["segment"] == row["segment"]) & (bi["line"] == row["line"])] \
                if not bi.empty else bi
            d = _direction(b)
            v_over, c_over = grade(d["ece_over"] if np.isfinite(d["ece_over"]) else np.nan,
                                   row["slope"], row["slope_se"], d["bands_over"])
            seg = g if row["segment"] == "all" else g[g["league"] == row["segment"].split(":")[-1]]
            d["mean_gap"] = float(seg["y"].mean() - seg["p"].mean()) if len(seg) else np.nan
            d["verdict_over"] = v_over
            d["confidence_over"] = c_over
            extra.append(d)
        su = pd.concat([su.reset_index(drop=True), pd.DataFrame(extra)], axis=1)
        summaries.append(su)
        bins.append(bi)
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    if not summary.empty:
        summary.insert(2, "n_slates", _slates(s))
    return summary, (pd.concat(bins, ignore_index=True) if bins else pd.DataFrame())


# --- 3. does a claimed edge turn into money -------------------------------


def _bucket_roi(s: pd.DataFrame, col: str, bins) -> pd.DataFrame:
    """Realised flat-stake return per bucket, with an interval on each.

    Two intervals because they answer different questions. Wilson is on the *hit
    rate*, a proportion, and is exact enough near 0 and 1 where these bins live.
    The bootstrap is on *mean ROI*, which is not a proportion at all -- a bucket
    of 10.0 shots is mostly zeros with the occasional tenfold payout, and a
    normal interval around that mean would be nonsense.
    """
    cut = pd.cut(s[col], bins=list(bins), right=False)
    rows = []
    for b, g in s.groupby(cut, observed=True):
        y = g["y"].to_numpy()
        k, n = int(y.sum()), len(y)
        lo, hi = wilson_interval(k, n)
        rlo, rhi = bootstrap_ci(g["roi"].to_numpy()) if n > 1 else (np.nan, np.nan)
        rows.append({
            "bucket": str(b), "lo": float(b.left), "hi": float(b.right),
            "n": n, "n_slates": _slates(g),
            "mean_p": float(g["p"].mean()), "mean_odds": float(g["o"].mean()),
            "mean_e": float(g["e"].mean()),
            "hit_rate": k / n if n else np.nan, "hit_lo": lo, "hit_hi": hi,
            "flat_roi": float(g["roi"].mean()),
            "roi_lo": rlo, "roi_hi": rhi,
            "profitable": bool(rlo > 0) if np.isfinite(rlo) else False,
        })
    return pd.DataFrame(rows)


def edge_buckets(s: pd.DataFrame, bins=EDGE_BINS) -> pd.DataFrame:
    """Realised return by claimed edge.

    The shape is the finding. Rising in `e` means the filter selects winners.
    Flat means `E >= 1` is selecting noise and the threshold is doing nothing.
    Falling means it is selecting the propositions where the model most disagrees
    with the market *because the model is wrong there* -- the market being right
    about the same propositions the model is most confident about is the failure
    mode this whole ledger exists to detect.
    """
    return _bucket_roi(s, "e", bins) if not s.empty else pd.DataFrame()


def odds_buckets(s: pd.DataFrame, bins=ODDS_BINS) -> pd.DataFrame:
    """Realised return by price, i.e. the favourite-longshot question."""
    return _bucket_roi(s, "o", bins) if not s.empty else pd.DataFrame()


def by_market(s: pd.DataFrame) -> pd.DataFrame:
    """Hit rate and realised return per target, and per target within league."""
    if s.empty:
        return pd.DataFrame()
    rows = []
    for keys, g in [(("all", "all"), s)] + \
            [((str(t), "all"), gg) for t, gg in s.groupby("target", observed=True)] + \
            [((str(t), str(l)), gg) for (t, l), gg in s.groupby(["target", "league"], observed=True)]:
        y = g["y"].to_numpy()
        lo, hi = wilson_interval(int(y.sum()), len(y))
        rlo, rhi = bootstrap_ci(g["roi"].to_numpy()) if len(g) > 1 else (np.nan, np.nan)
        rows.append({
            "target": keys[0], "league": keys[1], "n": len(g), "n_slates": _slates(g),
            "mean_p": float(g["p"].mean()), "hit_rate": float(y.mean()),
            "hit_lo": lo, "hit_hi": hi, "gap": float(y.mean() - g["p"].mean()),
            "mean_odds": float(g["o"].mean()), "flat_roi": float(g["roi"].mean()),
            "roi_lo": rlo, "roi_hi": rhi,
        })
    return pd.DataFrame(rows)


# --- 4. the frontier ------------------------------------------------------


def frontier(root=None) -> pd.DataFrame:
    """Every fully settled portfolio, with its run date and a Sharpe column."""
    pf = ledger.read_table("portfolios", root)
    runs = ledger.read_table("runs", root)
    if pf.empty:
        return pf
    out = pf[pf["settled"].fillna(False).astype(bool)].copy()
    if out.empty:
        return out
    if not runs.empty:
        out = out.merge(runs[["run_code", "run_date"]], on="run_code", how="left")
    # Return over spread, on the same scale the Portfolio Book shows both.
    out["_sharpe"] = (out["expected_return_pct"] - 1.0) / out["sd_pct"].replace(0, np.nan)
    return out.reset_index(drop=True)


def frontier_distribution(f: pd.DataFrame) -> pd.DataFrame:
    """What the whole frontier returned, per slate.

    The headline is `spread`. Every portfolio on one slate is built from the same
    few dozen propositions, so the distance between the best and the worst is
    what choosing between them was worth that day -- and on the first real slate
    that was 273 points, from wiped out to nearly tripled. No other number in
    this report makes the case for caring about selection at all.
    """
    if f.empty:
        return pd.DataFrame()
    rows = []
    for run_code, g in list(f.groupby("run_code", observed=True)) + [("all", f)]:
        r = g["realised_return"].dropna()
        if r.empty:
            continue
        rows.append({
            "run_code": str(run_code),
            "run_date": g["run_date"].iloc[0] if "run_date" in g else None,
            "n": len(r), "min": r.min(), "p5": r.quantile(.05), "p25": r.quantile(.25),
            "median": r.median(), "p75": r.quantile(.75), "p95": r.quantile(.95),
            "max": r.max(), "mean": r.mean(), "sd": r.std(),
            "spread": r.max() - r.min(),
            "share_profitable": float((r > 0).mean()),
            "share_wiped_out": float((r <= -0.999).mean()),
        })
    return pd.DataFrame(rows)


# Fixed bands, not quantile bins: two slates have to be readable side by side,
# and a quantile bin means something different on every one of them.
RETURN_BANDS: tuple[float, ...] = (-1.0001, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5,
                                   0.75, 1.0, 1.25, 1.5, 2.0, np.inf)


def frontier_histogram(f: pd.DataFrame, bands=RETURN_BANDS) -> pd.DataFrame:
    """Portfolio counts per return band, per slate and pooled."""
    if f.empty:
        return pd.DataFrame()
    rows = []
    for run_code, g in list(f.groupby("run_code", observed=True)) + [("all", f)]:
        r = g["realised_return"].dropna()
        if r.empty:
            continue
        counts = pd.cut(r, bins=list(bands)).value_counts().sort_index()
        for iv, n in counts.items():
            rows.append({"run_code": str(run_code), "band": str(iv),
                         "lo": float(iv.left), "hi": float(iv.right),
                         "n": int(n), "share": float(n / len(r))})
    return pd.DataFrame(rows)


def percentile_of(f: pd.DataFrame, run_code: str, value: float) -> float:
    """Share of that slate's frontier a return beat.

    Rank rather than return, because returns are not comparable between slates --
    a good slate lifts every portfolio on it. Rank answers the question that
    actually transfers: out of everything that was available that day, did this
    pick land well?
    """
    r = f.loc[f["run_code"] == run_code, "realised_return"].dropna().to_numpy()
    if not len(r) or not np.isfinite(value):
        return np.nan
    return float((r < value).mean())


def focus_run(root=None) -> str | None:
    """The slate everything defaults to showing: the one the last bet is on.

    The last line of the placed table, not the newest run. Those differ the
    moment `06` is run for tomorrow -- the newest run is then a slate with no
    result and no bet on it, and defaulting to it would draw an empty chart over
    a live one. Falls back to the newest settled run when nothing has been backed
    yet.
    """
    placed = ledger.read_table("placed", root)
    runs = ledger.read_table("runs", root)
    if not placed.empty:
        if not runs.empty and "run_date" in runs:
            m = placed.merge(runs[["run_code", "run_date"]], on="run_code", how="left")
            m = m.sort_values(["run_date", "recorded_at"], na_position="first")
            return str(m.iloc[-1]["run_code"])
        return str(placed.iloc[-1]["run_code"])
    pf = ledger.read_table("portfolios", root)
    if pf.empty:
        return None
    done = pf[pf["settled"].fillna(False).astype(bool)]
    return str(sorted(done["run_code"].unique())[-1]) if not done.empty else None


def placed_rank(f: pd.DataFrame, placed: pd.DataFrame | None = None) -> pd.DataFrame:
    """Where the bet actually struck landed in its own slate's frontier."""
    placed = ledger.read_table("placed") if placed is None else placed
    if f.empty or placed is None or placed.empty:
        return pd.DataFrame()
    rows = []
    for b in placed.to_dict("records"):
        hit = f[(f["run_code"] == b["run_code"]) & (f["portfolio_id"] == b["portfolio_id"])]
        if hit.empty:
            continue
        pick = hit.iloc[0]
        rows.append({
            "run_code": b["run_code"], "ref": f"{b['run_code']}-{b['portfolio_id']}",
            "split": pick["split"], "legs": pick["legs"],
            "expected": pick["expected_return_pct"] - 1.0,
            "realised": pick["realised_return"],
            "percentile": percentile_of(f, b["run_code"], pick["realised_return"]),
            "slate_median": f.loc[f["run_code"] == b["run_code"], "realised_return"].median(),
            "stake": b.get("stake"), "pnl": b.get("stake", 0) * pick["realised_return"],
        })
    return pd.DataFrame(rows)


def strategy_returns(f: pd.DataFrame) -> pd.DataFrame:
    """What each rule for picking off the frontier actually returned, per slate.

    Run once across the whole frontier and once inside each split, because the
    split is part of the portfolio rather than a presentation of it. `percentile`
    is always measured against the **whole** slate, even for a rule applied
    inside one split, so every row answers one question: out of everything
    available that day, how good was this pick?
    """
    if f.empty:
        return pd.DataFrame()
    rows = []
    for split_name, pool in [("all", f)] + [(str(k), g) for k, g in f.groupby("split", observed=True)]:
        for run_code, g in pool.groupby("run_code", observed=True):
            for label, rule in STRATEGIES.items():
                if rule.column not in g.columns:
                    continue
                pick = rule.pick(g)
                if pick is None:
                    continue
                rows.append({
                    "strategy": label, "split": split_name, "run_code": run_code,
                    "run_date": pick.get("run_date"), "portfolio_id": int(pick["portfolio_id"]),
                    "legs": pick["legs"], "expected_return_pct": pick["expected_return_pct"],
                    "sd_pct": pick["sd_pct"], "p_over_100": pick["p_over_100"],
                    "median_leg_p": pick.get("median_leg_p"),
                    "mean_leg_odds": pick.get("mean_leg_odds"),
                    "g_f_protective": pick.get("g_f_protective"),
                    "g_g_protective": pick.get("g_g_protective"),
                    "g_f_suggested": pick.get("g_f_suggested"),
                    "realised_return": pick["realised_return"],
                    "percentile": percentile_of(f, run_code, pick["realised_return"]),
                })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["split", "strategy", "run_date"]).reset_index(drop=True)


def strategy_summary(picks: pd.DataFrame, f: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (strategy, split), best first, ranked on percentile.

    `mean_percentile` leads and `mean_return` follows, because only the first is
    comparable across slates. A rule that returned +18% on a slate where the
    median portfolio returned +11% picked well; the same +18% on a slate where
    the median was +25% picked badly, and the return column alone cannot tell
    those apart.

    `n` here is slates, not portfolios. Fourteen rules over one slate is fourteen
    observations of one Saturday.
    """
    if picks.empty:
        return pd.DataFrame()
    picks = picks.copy()
    if "percentile" not in picks:
        # A caller that assembled picks by hand, or an older stored table. Rank
        # is unavailable rather than zero -- reporting it as 0.0 would sort every
        # such rule to the bottom as if it had picked the worst portfolio.
        picks["percentile"] = np.nan
    g = picks.groupby(["split", "strategy"], observed=True)
    out = g.agg(
        n_slates=("realised_return", "size"),
        mean_percentile=("percentile", "mean"),
        slates_above_median=("percentile", lambda s: int((s > 0.5).sum())),
        mean_return=("realised_return", "mean"),
        median_return=("realised_return", "median"),
        worst=("realised_return", "min"),
        best=("realised_return", "max"),
        slates_profitable=("realised_return", lambda s: int((s > 0).sum())),
        mean_expected=("expected_return_pct", lambda s: float(s.mean() - 1.0)),
    ).reset_index()
    out["hit_rate"] = out["slates_profitable"] / out["n_slates"]
    # What the portfolio promised against what it paid. Positive means the rule
    # over-promised, which is the direction that matters.
    out["expected_minus_realised"] = out["mean_expected"] - out["mean_return"]
    return out.sort_values("mean_percentile", ascending=False,
                           na_position="last").reset_index(drop=True)


def strategy_reference(f: pd.DataFrame, placed: pd.DataFrame | None = None) -> pd.DataFrame:
    """The rows a strategy table has to be read against.

    Without them the percentile column is a number with no anchor: the median
    portfolio is 0.500 by construction, and the bet actually placed is the only
    row on the table that cost anything.
    """
    if f.empty:
        return pd.DataFrame()
    rows = [{"strategy": "-- median portfolio", "split": "all",
             "n_slates": int(f["run_code"].nunique()),
             "mean_percentile": 0.5,
             "mean_return": float(f.groupby("run_code")["realised_return"].median().mean())},
            {"strategy": "-- random pick", "split": "all",
             "n_slates": int(f["run_code"].nunique()),
             "mean_percentile": np.nan,
             "mean_return": float(f.groupby("run_code")["realised_return"].mean().mean())}]
    pr = placed_rank(f, placed)
    if not pr.empty:
        rows.append({"strategy": "-- YOUR BET", "split": "all", "n_slates": len(pr),
                     "mean_percentile": float(pr["percentile"].mean()),
                     "mean_return": float(pr["realised"].mean())})
    return pd.DataFrame(rows)


# Old and new alike. A column absent from a slate is skipped rather than dropped
# from the list: R001-R007 carry `g_f_star`, later runs carry `capacity`, and the
# whole point of this table is to compare characteristics across every slate that
# has one.
CHARACTERISTICS: tuple[str, ...] = (
    "expected_return_pct", "sd_pct", "variance", "p_over_100", "p_over_110",
    "legs", "capacity", "capacity_used", "n_eff", "max_leg_stake",
    "g_f_suggested", "g_g_suggested",
    "g_f_protective", "g_g_protective", "g_f_star", "g_g_star", "g_p0",
    "median_leg_p", "mean_leg_p", "mean_leg_odds", "min_leg_odds", "max_leg_odds",
    "mean_leg_e",
)


def frontier_attribution(f: pd.DataFrame) -> pd.DataFrame:
    """Does any ex-ante characteristic predict what a portfolio actually returned?

    Two correlations per characteristic, and they are not interchangeable. The
    pooled one runs over every settled portfolio at once and will look
    impressively significant on almost nothing: portfolios inside a slate share
    legs, so they are not independent draws. The within-slate mean ranks
    portfolios only against others from the same day and then averages those
    ranks, which is the number to read.
    """
    if f.empty or f["realised_return"].isna().all():
        return pd.DataFrame()
    rows = []
    for col in CHARACTERISTICS:
        if col not in f or f[col].isna().all():
            continue
        # A characteristic that never varies has no rank correlation with
        # anything. `g_f_protective` is genuinely constant on some slates, and
        # scipy warns and returns nan rather than raising -- which would put a
        # warning in the middle of the notebook's output every run.
        pooled = (f[col].corr(f["realised_return"], method="spearman")
                  if f[col].nunique() > 1 else np.nan)
        per = [g[col].corr(g["realised_return"], method="spearman")
               for _, g in f.groupby("run_code", observed=True)
               if g[col].nunique() > 1 and g["realised_return"].nunique() > 1]
        per = [x for x in per if np.isfinite(x)]
        rows.append({
            "characteristic": col, "n_portfolios": int(f[col].notna().sum()),
            "n_slates": len(per),
            "spearman_pooled": float(pooled) if np.isfinite(pooled) else np.nan,
            "spearman_within_slate": float(np.mean(per)) if per else np.nan,
            "slates_positive": int(sum(x > 0 for x in per)),
        })
    return pd.DataFrame(rows).sort_values("spearman_within_slate", ascending=False,
                                          na_position="last").reset_index(drop=True)


def characteristic_deciles(f: pd.DataFrame, characteristics=None) -> pd.DataFrame:
    """Mean, median and hit rate by decile of each characteristic.

    The companion `frontier_attribution` reports Spearman, which is rank-based
    and therefore describes the **typical** portfolio rather than the average
    one. That distinction is not pedantry: on the first real slate
    `expected_return_pct` correlated -0.76 with realised return, which reads as
    "high expected return is bad" and is not what happened. High-EV portfolios
    are skewed -- most disappoint, a few own the whole right tail -- so the rank
    falls while the mean need not.

    A decile view shows both halves at once, and is the table to read before
    concluding anything from a correlation.
    """
    if f.empty or f["realised_return"].isna().all():
        return pd.DataFrame()
    rows = []
    for col in (characteristics or CHARACTERISTICS):
        if col not in f or f[col].nunique() < 10:
            continue
        d = f.dropna(subset=[col, "realised_return"]).copy()
        if d.empty:
            continue
        d["decile"] = pd.qcut(d[col].rank(method="first"), 10, labels=False, duplicates="drop") + 1
        for dec, g in d.groupby("decile", observed=True):
            rows.append({
                "characteristic": col, "decile": int(dec), "n": len(g),
                "value_lo": float(g[col].min()), "value_hi": float(g[col].max()),
                "mean_return": float(g["realised_return"].mean()),
                "median_return": float(g["realised_return"].median()),
                "best": float(g["realised_return"].max()),
                "share_profitable": float((g["realised_return"] > 0).mean()),
            })
    return pd.DataFrame(rows)


def growth_check(picks: pd.DataFrame) -> pd.DataFrame:
    """Realised compounding against what the Projection Book projected.

    The Projection Book says a portfolio played every week at the protective
    stake grows at `g_protective` a round. This is the same arithmetic run
    backwards over the rounds that actually happened: stake `f` of the pot on
    each slate's pick, and `log(1 + f * realised)` is the growth that round.

    It is the one number in this file that is directly falsifiable against a
    claim the pipeline already makes on screen.
    """
    if picks.empty:
        return pd.DataFrame()
    rows = []
    for (split_name, strategy), g in picks.groupby(["split", "strategy"], observed=True):
        g = g.dropna(subset=["realised_return", "g_f_protective"])
        if g.empty:
            continue
        f = g["g_f_protective"].to_numpy(dtype=float)
        r = g["realised_return"].to_numpy(dtype=float)
        mult = 1.0 + f * r
        # A round that takes the pot to zero or below is not a growth rate, it is
        # the end of the series. Log it as such rather than letting -inf average
        # into a number that reads like a rate.
        ruined = bool((mult <= 0).any())
        rows.append({
            "split": split_name, "strategy": strategy, "n_slates": len(g),
            "mean_f_protective": float(f.mean()),
            "projected_g_protective": float(g["g_g_protective"].mean()),
            "realised_g": float(np.log(mult).mean()) if not ruined else np.nan,
            "pot_multiple": float(np.prod(mult)) if not ruined else 0.0,
            "ruined": ruined,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["gap"] = out["realised_g"] - out["projected_g_protective"]
    return out.sort_values("realised_g", ascending=False, na_position="last").reset_index(drop=True)


# --- the whole report -----------------------------------------------------


def run_all(root=None) -> dict[str, pd.DataFrame]:
    """Every analysis over whatever the ledger currently holds, keyed by name.

    One call so the notebook stays a driver, and one dict so `write_reports` and
    the cells that display a table are reading the same objects rather than each
    recomputing from the store.
    """
    s = settled(root)
    f = frontier(root)
    picks = strategy_returns(f)
    placed = ledger.read_table("placed", root)
    cal_line, cal_line_bins = live_calibration(s)
    cal_pool, cal_pool_bins = live_calibration(s, pooled=True)
    return {
        "focus_run": focus_run(root),
        "model_vs_market": model_vs_market(s),
        "by_market": by_market(s),
        "edge_buckets": edge_buckets(s),
        "odds_buckets": odds_buckets(s),
        "calibration_by_market": cal_pool,
        "calibration_by_market_bins": cal_pool_bins,
        "calibration_by_line": cal_line,
        "calibration_by_line_bins": cal_line_bins,
        "frontier_distribution": frontier_distribution(f),
        "frontier_histogram": frontier_histogram(f),
        "placed_rank": placed_rank(f, placed),
        "strategy_picks": picks,
        "strategy_summary": strategy_summary(picks, f),
        "strategy_reference": strategy_reference(f, placed),
        "frontier_attribution": frontier_attribution(f),
        "characteristic_deciles": characteristic_deciles(f),
        "growth_check": growth_check(picks),
        "bankroll": ledger.bankroll(root),
    }


def view(frames: dict[str, pd.DataFrame], name: str, cols=None,
         where=None) -> pd.DataFrame:
    """One report frame, cut down to `cols`, tolerant of an empty ledger.

    Selecting columns straight off `frames[name]` raises `KeyError` when nothing
    has settled, because an empty analysis returns a frame with no columns at
    all -- so the first run of `07`, before any match has been played, died on a
    display cell rather than saying there was nothing to display yet. Which is
    the normal state of this notebook the day it is set up.
    """
    df = frames.get(name)
    if not isinstance(df, pd.DataFrame) or df.empty:
        print(f"{name}: nothing to show yet -- no settled propositions in the ledger")
        return pd.DataFrame()
    if where is not None:
        df = df[where(df)]
    keep = [c for c in (cols or list(df.columns)) if c in df.columns]
    missing = [c for c in (cols or []) if c not in df.columns]
    if missing:
        print(f"{name}: no column {', '.join(missing)}")
    return df[keep].round(4)


def write_reports(frames: dict[str, pd.DataFrame], out_dir=None) -> list:
    """Write every non-empty frame as a CSV, plus the calibration figures.

    CSVs rather than one workbook, and undated, exactly as `04_Evaluation`
    writes `Outputs/Evaluation/`: these are a view of the ledger at a moment,
    and the ledger is the thing that keeps history.
    """
    from .paths import ANALYSIS_REPORTS
    from .report.figures import save_calibration_figures

    out = Path(out_dir) if out_dir else ANALYSIS_REPORTS
    out.mkdir(parents=True, exist_ok=True)

    # Clear first. These are a view of the ledger at a moment, not a history, and
    # an analysis that has become empty -- because runs were excluded, or because
    # nothing has settled yet -- would otherwise leave the previous answer on disk
    # looking current. The ledger is what keeps history; this directory does not.
    for stale in list(out.glob("*.csv")) + list((out / "Figures").glob("*.png")):
        stale.unlink()

    written = []
    for name, df in frames.items():
        # `frames` is mostly tables but carries `focus_run`, a bare string; a
        # scalar has no `.empty` and would take the whole report down.
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        path = out / f"{name}.csv"
        df.to_csv(path, index=False)
        written.append(path)

    summary, bins = frames.get("calibration_by_line"), frames.get("calibration_by_line_bins")
    if summary is not None and bins is not None and not bins.empty:
        made = save_calibration_figures(summary, bins, out / "Figures")
        written += [p for paths in made.values() for p in paths]

    hist = frames.get("frontier_histogram")
    if hist is not None and not hist.empty:
        import matplotlib.pyplot as plt

        from .report.figures import frontier_return_figure

        # One per slate. Pooling two slates whose medians are 55 points apart
        # draws a mixture that describes neither, so `all` is written too but
        # never as the headline file.
        dist, placed = frames.get("frontier_distribution"), frames.get("placed_rank")
        runs = sorted(r for r in hist["run_code"].unique() if r != "all")
        focus = frames.get("focus_run") or (runs[-1] if runs else None)
        # A stable name for the one to look at, so nothing has to know the run
        # code to find today's chart, plus one per slate for going back.
        jobs = [(focus, "frontier_returns.png")]
        jobs += [(rc, f"frontier_returns_{rc}.png") for rc in runs]
        if len(runs) > 1:
            jobs.append(("all", "frontier_returns_all.png"))
        for rc, name in jobs:
            if rc is None:
                continue
            fig = frontier_return_figure(hist, dist, placed, run_code=rc)
            if fig is None:
                continue
            path = out / "Figures" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=140, bbox_inches="tight")
            plt.close(fig)
            written.append(path)
    return written


__all__ = [
    "EDGE_BINS", "ODDS_BINS", "STRATEGIES", "CHARACTERISTICS",
    "run_all", "write_reports", "view", "focus_run",
    "settled", "model_vs_market", "live_calibration", "Rule",
    "frontier_distribution", "frontier_histogram", "percentile_of", "placed_rank",
    "strategy_reference", "characteristic_deciles", "RETURN_BANDS",
    "edge_buckets", "odds_buckets", "by_market",
    "frontier", "strategy_returns", "strategy_summary", "frontier_attribution",
    "growth_check",
]
