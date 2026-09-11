"""Is a stated probability worth what it says?

Log loss answers "is this model better than that one". It does not answer "when
this model says 62%, does the thing happen 62% of the time" -- and that second
question is the one that decides whether a priced ladder is usable. A model can
win on log loss and still be systematically overconfident at exactly the lines
worth betting.

So every market is reduced to a stream of (predicted probability, outcome) pairs
and interrogated three ways:

* **overall** -- ECE, Brier and its Murphy decomposition, and a calibration slope.
  The slope is the headline: regress the outcome on ``logit(p)`` and a slope
  below 1 means the probabilities are too extreme in both directions, which no
  single-number error metric will tell you.
* **by range** -- which *bands* of predicted probability are trustworthy. A model
  can be well calibrated in aggregate while being badly wrong at the tails,
  because the middle carries most of the weight and averages the tails away.
* **by segment** -- per league, and per venue. Venue is here deliberately: the
  fixture workbook's baseline pooled home and away for a long time and it took a
  visible artefact to notice, so the evaluation now measures that axis directly.

Binning is by **quantile, not equal width**. Predicted probabilities cluster --
a shots Over line sits between .3 and .7 on nearly every fixture -- so ten
equal-width bins leave most of them empty and hide all the structure inside the
two or three that fill up. Equal-count bins put the resolution where the
predictions actually live.

One caveat on reading the per-band verdicts: the interval is 95%, so roughly one
band in twenty falls outside it by chance alone even when the model is perfect.
A single flagged band is noise; a run of them in the same direction, or a segment
where most bands are flagged, is the signal. That is why the summary verdict is
driven by ECE and the slope -- aggregate quantities -- and reports
``n_bands_off`` against ``n_bins`` rather than treating any flag as a failure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import EPS, binary_metrics, over_prob, pmf_for
from .models import total_pmfs
from .spec import target_spec

N_BINS = 10
MIN_BIN_COUNT = 50
Z95 = 1.959963984540054

# Verdict thresholds. Round numbers, chosen to be legible rather than derived --
# the underlying figures always sit in the row beside the verdict, so nothing is
# hidden behind the exact cut.
#
# `SLOPE_SEVERE` earns its own rule. A slope of 0.32 (corners, match total) means
# the stated probabilities are wildly too extreme, and yet its ECE is a mild
# 0.027: when predictions cluster near the base rate, an average error stays
# small while the *confident* predictions -- the only ones worth acting on -- are
# badly wrong. ECE alone would wave that through.
ECE_TRUST, ECE_AVOID = 0.025, 0.05
ECE_SLOPE_COMBINED = 0.035  # a significant slope miss plus this much ECE is fatal
SLOPE_TOL = 0.10
SLOPE_SEVERE = 0.35

TRUST, CAUTION, AVOID = "TRUST", "CAUTION", "AVOID"


# --- Streams --------------------------------------------------------------


def probability_stream(paired: pd.DataFrame, target: str, dispersion, *, metric: str) -> pd.DataFrame:
    """Every (line, predicted probability, outcome) triple, long-form.

    ``metric`` is ``"team"`` or ``"total"`` and is never inferred. The two answer
    different questions on different bucket grids -- the discipline `evaluate.py`
    already enforces -- so the column travels with the numbers and every frame
    downstream carries it.

    Team pmfs are built at ``total_cap`` for the same reason `report.markets`
    does: the cap is a scoring device, and any cap above the top line yields an
    identical ``over_prob``.
    """
    s = target_spec(target)
    if metric not in ("team", "total"):
        raise ValueError(f"metric must be 'team' or 'total', got {metric!r}")

    rows: list[dict] = []
    if metric == "total":
        pmfs = total_pmfs(paired, target, dispersion)
        totals = paired["actual_home"].to_numpy() + paired["actual_away"].to_numpy()
        leagues = paired["league_key"].to_numpy()
        for pmf, actual, lk in zip(pmfs, totals, leagues):
            for line in s.match_lines:
                rows.append({"line": line, "p": over_prob(pmf, line),
                             "y": int(actual > line), "league": lk, "venue": "match"})
    else:
        for r in paired.itertuples():
            disp = dispersion.get(r.league_key) if isinstance(dispersion, dict) else dispersion
            for venue, mu, actual in (("home", r.mu_home, r.actual_home),
                                      ("away", r.mu_away, r.actual_away)):
                if not np.isfinite(actual):
                    continue
                pmf = pmf_for(s.dist, mu, s.total_cap, disp)
                for line in s.team_lines:
                    rows.append({"line": line, "p": over_prob(pmf, line),
                                 "y": int(actual > line), "league": r.league_key, "venue": venue})

    out = pd.DataFrame(rows)
    out["target"] = target
    out["metric"] = metric
    return out


def _segments(stream: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """``all``, then one per league, then one per venue where venue is meaningful.

    A single flat ``segment`` column rather than a league column and a venue
    column: most rows would be null in one or the other, and a reader filtering a
    CSV should not have to know which.
    """
    out: list[tuple[str, pd.DataFrame]] = [("all", stream)]
    for lk, g in stream.groupby("league", observed=True):
        out.append((f"league:{lk}", g))
    if stream["venue"].nunique() > 1:
        for v, g in stream.groupby("venue", observed=True):
            out.append((f"venue:{v}", g))
    return out


# --- Binning --------------------------------------------------------------


def quantile_bins(p: np.ndarray, n_bins: int = N_BINS, min_count: int = MIN_BIN_COUNT) -> np.ndarray:
    """Equal-count bin index per prediction, with no bin below ``min_count``.

    The bin count is capped at ``len(p) // min_count`` up front, then any bin
    left short by ties is merged into its neighbour. A verdict issued off nine
    fixtures is noise wearing a number, so the floor is enforced rather than
    hoped for.
    """
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=int)

    k = max(1, min(n_bins, n // max(min_count, 1)))
    edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, k + 1)))
    if len(edges) < 3:
        return np.zeros(n, dtype=int)

    idx = np.clip(np.searchsorted(edges, p, side="left") - 1, 0, len(edges) - 2)

    # Merge undersized bins leftwards (the first one rightwards), so the floor
    # holds even when ties collapse a quantile boundary.
    while True:
        counts = np.bincount(idx, minlength=idx.max() + 1)
        small = np.flatnonzero((counts > 0) & (counts < min_count))
        if not small.size or (counts > 0).sum() == 1:
            break
        b = int(small[0])
        nonempty = np.flatnonzero(counts > 0)
        neighbours = nonempty[nonempty != b]
        target_bin = int(neighbours[np.abs(neighbours - b).argmin()])
        idx[idx == b] = target_bin

    # Renumber to a dense 0..m-1 in ascending probability order.
    order = {old: new for new, old in enumerate(sorted(np.unique(idx)))}
    return np.array([order[i] for i in idx], dtype=int)


def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for an observed frequency.

    Not the normal approximation: these bins routinely sit near 0 or 1, where
    ``p +/- z*sqrt(p(1-p)/n)`` produces bounds outside [0, 1] and declares
    miscalibration that is really just a bad approximation.
    """
    if n <= 0:
        return float("nan"), float("nan")
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


# --- Metrics --------------------------------------------------------------


def calibration_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    """``(slope, intercept, slope_se)`` from a logistic fit of outcome ~ logit(p).

    Slope 1 / intercept 0 is perfect. Slope below 1 means the probabilities are
    too extreme -- the model is more confident than the evidence supports -- and
    that is invisible to ECE, which can be near zero while the tails are wrong in
    opposite directions.

    The standard error is returned because the slope alone is over-readable. Its
    precision depends on how *wide* the predicted probabilities spread, not just
    how many there are: a market whose predictions all sit between .45 and .60
    gives the regression almost no leverage, and will happily report a slope of
    1.15 on perfectly calibrated input. Without the error bar that reads as a
    finding. With it, it reads as noise.
    """
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    nan3 = (float("nan"), float("nan"), float("nan"))
    if len(np.unique(y)) < 2:
        return nan3
    x = np.log(p / (1 - p)).reshape(-1, 1)
    # `C=np.inf` rather than `penalty=None`: same unpenalised fit, but the latter
    # is deprecated in scikit-learn 1.8 and removed in 1.10, and any shrinkage at
    # all would bias the slope towards zero -- which is the exact quantity being
    # measured.
    m = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000).fit(x, y)

    # Standard error from the inverse Fisher information of the fitted model.
    fitted = np.clip(m.predict_proba(x)[:, 1], EPS, 1 - EPS)
    w = fitted * (1 - fitted)
    design = np.column_stack([np.ones(len(x)), x[:, 0]])
    try:
        cov = np.linalg.inv(design.T @ (design * w[:, None]))
        se = float(np.sqrt(cov[1, 1]))
    except np.linalg.LinAlgError:
        se = float("nan")
    return float(m.coef_[0][0]), float(m.intercept_[0]), se


def _bin_frame(y: np.ndarray, p: np.ndarray, idx: np.ndarray) -> pd.DataFrame:
    rows = []
    for b in range(idx.max() + 1 if idx.size else 0):
        m = idx == b
        if not m.any():
            continue
        n = int(m.sum())
        k = int(y[m].sum())
        obs = k / n
        pred = float(p[m].mean())
        lo, hi = wilson_interval(k, n)
        if not np.isfinite(lo):
            verdict = "insufficient"
        elif lo <= pred <= hi:
            verdict = "calibrated"
        elif pred > hi:
            verdict = "over-predicts"
        else:
            verdict = "under-predicts"
        rows.append({
            "bin": b, "lo": float(p[m].min()), "hi": float(p[m].max()),
            "n": n, "pred_mean": pred, "obs_freq": obs, "gap": obs - pred,
            "ci_lo": lo, "ci_hi": hi, "verdict": verdict,
        })
    return pd.DataFrame(rows)


def slope_is_off(slope: float, slope_se: float) -> bool:
    """Is the calibration slope both materially and *significantly* away from 1?

    Requiring two standard errors as well as the tolerance stops narrow-spread
    markets -- where the regression has little leverage and the estimate is noisy
    -- from being labelled miscalibrated on nothing.
    """
    return bool(np.isfinite(slope) and abs(slope - 1) > SLOPE_TOL
                and (not np.isfinite(slope_se) or abs(slope - 1) > 2 * slope_se))


def grade(ece_value: float, slope: float, slope_se: float, bands_off: int) -> tuple[str, str]:
    """``(verdict, confidence)`` for one market. The single definition of both.

    Kept as a free function rather than inlined into `_summarise` so that
    `line_verdicts` regrades a stored summary instead of trusting whatever label
    was written into it -- a CSV from an earlier run would otherwise carry old
    thresholds silently into a new report.
    """
    off_one = slope_is_off(slope, slope_se)
    severe = off_one and abs(slope - 1) > SLOPE_SEVERE

    if not np.isfinite(ece_value):
        verdict = "insufficient"
    elif ece_value >= ECE_AVOID or severe or (off_one and ece_value >= ECE_SLOPE_COMBINED):
        verdict = AVOID
    elif ece_value < ECE_TRUST and not off_one and bands_off <= 1:
        verdict = TRUST
    else:
        verdict = CAUTION

    if not np.isfinite(slope):
        confidence = "unknown"
    elif not off_one:
        confidence = "well-scaled"
    elif slope < 1:
        confidence = "overconfident"
    else:
        confidence = "underconfident"
    return verdict, confidence


def _summarise(y: np.ndarray, p: np.ndarray, bins: pd.DataFrame) -> dict:
    """Overall metrics for one (line, segment), from its own quantile bins."""
    n = len(y)
    base = float(y.mean()) if n else float("nan")
    w = bins["n"] / bins["n"].sum() if not bins.empty else pd.Series(dtype=float)

    gaps = (bins["pred_mean"] - bins["obs_freq"]).abs() if not bins.empty else pd.Series(dtype=float)
    out = binary_metrics(y, p, "")
    out.pop("market", None)

    # Murphy: BS = reliability - resolution + uncertainty, on these same bins.
    reliability = float((w * (bins["pred_mean"] - bins["obs_freq"]) ** 2).sum()) if not bins.empty else np.nan
    resolution = float((w * (bins["obs_freq"] - base) ** 2).sum()) if not bins.empty else np.nan
    uncertainty = base * (1 - base) if np.isfinite(base) else np.nan

    slope, intercept, slope_se = calibration_slope(y, p)
    e = float((w * gaps).sum()) if not bins.empty else np.nan

    bands_off = int((bins["verdict"].isin(["over-predicts", "under-predicts"])).sum()) if not bins.empty else 0
    verdict, confidence = grade(e, slope, slope_se, bands_off)

    out.update({
        "n": n, "n_bins": int(len(bins)), "base_rate": base, "pred_mean": float(p.mean()) if n else np.nan,
        "ece": e, "mce": float(gaps.max()) if not bins.empty else np.nan,
        "brier_reliability": reliability, "brier_resolution": resolution,
        "brier_uncertainty": uncertainty,
        "slope": slope, "slope_se": slope_se, "intercept": intercept,
        "worst_band_gap": float(bins.loc[bins["gap"].abs().idxmax(), "gap"]) if not bins.empty else np.nan,
        "n_bands_off": bands_off,
        "verdict": verdict, "confidence": confidence,
    })
    return out


# --- Public API -----------------------------------------------------------


def tables_from_stream(
    stream: pd.DataFrame, target: str, metric: str, *,
    n_bins: int = N_BINS, min_count: int = MIN_BIN_COUNT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(summary, bins)`` for a ready-made ``(line, p, y, league, venue)`` stream.

    Split out from `calibration_tables` so a stream that did not come from a pmf
    can be graded by the same code. `fpp.ledger` has one: settled propositions
    carry a probability and a realised outcome directly, and grading those on a
    second implementation of ECE and the slope is exactly the quiet disagreement
    this pipeline keeps designing out.
    """
    summary_rows, bin_frames = [], []

    for segment, seg in _segments(stream):
        for line, g in seg.groupby("line", observed=True):
            y = g["y"].to_numpy(dtype=int)
            p = g["p"].to_numpy(dtype=float)
            idx = quantile_bins(p, n_bins, min_count)
            bins = _bin_frame(y, p, idx)

            # A line is a number everywhere the pmf path uses this, and a label
            # ("all") where a caller has pooled the ladder. Coerce the first and
            # leave the second alone rather than refusing it.
            keys = {"target": target, "metric": metric, "segment": segment,
                    "line": float(line) if isinstance(line, (int, float, np.number)) else line}
            summary_rows.append({**keys, **_summarise(y, p, bins)})
            if not bins.empty:
                bin_frames.append(bins.assign(**keys))

    lead = ["target", "metric", "segment", "line"]
    summary = pd.DataFrame(summary_rows)
    summary = summary[lead + [c for c in summary.columns if c not in lead]]
    bins_out = (pd.concat(bin_frames, ignore_index=True) if bin_frames
                else pd.DataFrame(columns=lead + ["bin", "n"]))
    bins_out = bins_out[lead + [c for c in bins_out.columns if c not in lead]]
    return summary, bins_out


def calibration_tables(
    paired: pd.DataFrame, target: str, dispersion, *, metric: str,
    n_bins: int = N_BINS, min_count: int = MIN_BIN_COUNT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(summary, bins)`` for one target and one metric, across every segment."""
    stream = probability_stream(paired, target, dispersion, metric=metric)
    return tables_from_stream(stream, target, metric, n_bins=n_bins, min_count=min_count)


def calibration_report(
    paired_by_target: dict[str, pd.DataFrame], dispersions: dict[str, object],
    *, metrics: tuple[str, ...] = ("team", "total"),
    n_bins: int = N_BINS, min_count: int = MIN_BIN_COUNT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every target, both metrics, every segment -- two frames ready to export."""
    summaries, bins = [], []
    for target, paired in paired_by_target.items():
        for metric in metrics:
            s, b = calibration_tables(paired, target, dispersions[target],
                                      metric=metric, n_bins=n_bins, min_count=min_count)
            summaries.append(s)
            bins.append(b)
    return (pd.concat(summaries, ignore_index=True),
            pd.concat(bins, ignore_index=True))


# --- The headline: which lines can be trusted ------------------------------


def _pct(x: float) -> str:
    return "-" if not np.isfinite(x) else f"{x:.0%}"


def _span(frame: pd.DataFrame) -> str:
    """Where on the probability scale these bands sit, by their mean prediction."""
    lo, hi = frame["pred_mean"].min(), frame["pred_mean"].max()
    return f"around {_pct(lo)}" if len(frame) == 1 else f"{_pct(lo)}-{_pct(hi)}"


def _reading(row: pd.Series | dict, bins: pd.DataFrame) -> str:
    """One sentence saying what is wrong with this line, and where.

    The band tables answer "which slice of the probability scale is off"; this
    turns that into the sentence a person can act on, because "band 7" is not
    something anyone can act on and "over-predicts once above 60%" is.

    Counts are always reported as ``k of n`` bands. A span alone would imply the
    whole stretch is bad even when two scattered bands happen to sit at either
    end of it -- and at a 95% interval, one flagged band in twenty is expected
    noise rather than a fault.
    """
    n_bands = len(bins)
    off = bins[bins["verdict"].isin(["over-predicts", "under-predicts"])]
    conf = row.get("confidence", "")
    slope_note = "" if conf == "well-scaled" or not conf else (
        f"{conf} overall (slope {row['slope']:.2f})")

    if off.empty:
        base = "reliable across the whole probability range"
        return f"{base}, but {slope_note}" if slope_note else base

    over = off[off["verdict"] == "over-predicts"]
    under = off[off["verdict"] == "under-predicts"]

    # Both directions with the over-predictions higher up the scale is the
    # signature of probabilities that are simply too extreme -- one fault worth
    # naming as such, not two unrelated ones.
    if (not over.empty and not under.empty
            and over["pred_mean"].mean() > under["pred_mean"].mean()):
        parts = [f"too confident at both ends - under-predicts {_span(under)}, "
                 f"over-predicts {_span(over)}"]
    else:
        parts = []
        if not over.empty:
            parts.append(f"over-predicts in {len(over)} of {n_bands} bands ({_span(over)})")
        if not under.empty:
            parts.append(f"under-predicts in {len(under)} of {n_bands} bands ({_span(under)})")

    worst = off.loc[off["gap"].abs().idxmax()]
    parts.append(f"worst says {_pct(worst['pred_mean'])} where it happens {_pct(worst['obs_freq'])}")
    if slope_note:
        parts.append(slope_note)
    return "; ".join(parts)


def line_verdicts(summary: pd.DataFrame, bins: pd.DataFrame,
                  segment: str | None = "all") -> pd.DataFrame:
    """One row per over/under line: can this market be trusted, and why not.

    This is the table to read first. The per-band frames underneath it are the
    evidence; this is the finding. ``segment=None`` keeps every league and venue
    split rather than only the pooled rows.
    """
    sel = summary if segment is None else summary[summary["segment"] == segment]
    rows = []
    for _, r in sel.iterrows():
        b = bins[(bins["target"] == r["target"]) & (bins["metric"] == r["metric"])
                 & (bins["segment"] == r["segment"]) & (bins["line"] == r["line"])]
        verdict, confidence = grade(r["ece"], r["slope"], r["slope_se"], r["n_bands_off"])
        r = {**r, "verdict": verdict, "confidence": confidence}
        rows.append({
            "target": r["target"],
            "metric": r["metric"],
            "segment": r["segment"],
            "market": f"Over {r['line']}",
            "line": r["line"],
            "n": r["n"],
            "predicted": r["pred_mean"],
            "actual": r["base_rate"],
            "ece": r["ece"],
            "worst_band": r["mce"],
            "slope": r["slope"],
            "slope_se": r["slope_se"],
            "bands_off": r["n_bands_off"],
            "bands": r["n_bins"],
            "verdict": verdict,
            "confidence": confidence,
            "reading": _reading(r, b),
        })
    out = pd.DataFrame(rows)
    order = {TRUST: 0, CAUTION: 1, AVOID: 2, "insufficient": 3}
    return (out.assign(_o=out["verdict"].map(order))
               .sort_values(["target", "metric", "line"])
               .drop(columns="_o")
               .reset_index(drop=True))


def summarise_by_target(verdicts: pd.DataFrame) -> pd.DataFrame:
    """Per target and metric: how many of its lines survive, and which.

    The four-line answer to "which of these can I actually bet".
    """
    rows = []
    for (t, m), g in verdicts.groupby(["target", "metric"], observed=True):
        keep = g[g["verdict"] == TRUST]["market"].tolist()
        avoid = g[g["verdict"] == AVOID]["market"].tolist()
        rows.append({
            "target": t, "metric": m,
            "lines": len(g),
            "trust": len(keep), "caution": int((g["verdict"] == CAUTION).sum()), "avoid": len(avoid),
            "mean_ece": g["ece"].mean(),
            "mean_slope": g["slope"].mean(),
            "trusted_markets": ", ".join(keep) or "-",
            "avoid_markets": ", ".join(avoid) or "-",
        })
    return pd.DataFrame(rows).sort_values(["target", "metric"]).reset_index(drop=True)


def describe(row: pd.Series | dict) -> str:
    """One-line plain reading of a summary row."""
    head = f"{row['target']} [{row['metric']}] Over {row['line']} ({row['segment']})"
    if row["verdict"] == "insufficient":
        return f"{head}: too few observations to judge"
    return (f"{head}: {row['verdict']} -- ECE {row['ece']:.3f}, worst band off by "
            f"{row['worst_band_gap']:+.3f}, {row['n_bands_off']}/{row['n_bins']} bands outside "
            f"their interval, slope {row['slope']:.2f}+/-{row['slope_se']:.2f} "
            f"({row['confidence']})")


__all__ = [
    "probability_stream", "quantile_bins", "wilson_interval", "calibration_slope",
    "calibration_tables", "tables_from_stream", "calibration_report", "describe",
    "line_verdicts", "summarise_by_target", "grade", "slope_is_off",
    "TRUST", "CAUTION", "AVOID",
    "N_BINS", "MIN_BIN_COUNT", "ECE_GOOD", "ECE_FAIR", "SLOPE_TOL",
]
