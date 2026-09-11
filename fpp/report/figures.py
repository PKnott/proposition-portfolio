"""Calibration figures.

Three plots, in the order a reader wants them.

1. ``ece_by_line_figure`` -- the headline. One bar per market, coloured by
   verdict. This is the "which of these can I trust" answer, and it should be
   readable in two seconds without decoding anything.
2. ``reliability_figure`` -- the classic diagram, restored: the diagonal, the
   points, ECE in the title. Wilson error bars are added because a point that
   wobbles should look like it wobbles rather than reading as a finding.
3. ``gap_heatmap`` -- where on the probability scale a market goes wrong.

The heatmap is drawn on **fixed probability deciles**, not on the quantile bands
used for the statistics. The two need different things from a bin: the metrics
need a guaranteed count per bin, which is what quantile binning delivers; a chart
axis needs to mean something, and "band 7" does not. Deciles give a shared,
literal x-axis across every line, at the cost of leaving thinly-populated cells
blank -- which is the honest rendering anyway.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from matplotlib.patches import Patch

from ..calibration import (
    AVOID,
    CAUTION,
    ECE_AVOID,
    ECE_TRUST,
    TRUST,
    grade,
    slope_is_off,
)

INK = "#1F4E79"
VERDICT_COLOUR = {TRUST: "#2E7D32", CAUTION: "#F9A825", AVOID: "#B71C1C",
                  "insufficient": "#9E9E9E"}
GAP_CMAP = "RdYlGn"
N_DECILES = 10


def _lines(frame: pd.DataFrame) -> list[float]:
    return sorted(frame["line"].unique())


def _slice(frame: pd.DataFrame, target: str, metric: str | None, segment: str) -> pd.DataFrame:
    m = (frame["target"] == target) & (frame["segment"] == segment)
    if metric is not None:
        m &= frame["metric"] == metric
    return frame[m]


# --- 1. The headline ------------------------------------------------------


def ece_by_line_figure(summary: pd.DataFrame, target: str, segment: str = "all"):
    """One bar per market, coloured by verdict -- which lines can be trusted."""
    import matplotlib.pyplot as plt

    s = _slice(summary, target, None, segment).sort_values(["metric", "line"])
    if s.empty:
        return None

    labels, values, colours, hatches = [], [], [], []
    for _, r in s.iterrows():
        verdict, _ = grade(r["ece"], r["slope"], r["slope_se"], r["n_bands_off"])
        labels.append(f"{r['metric']}  Over {r['line']}")
        values.append(r["ece"])
        colours.append(VERDICT_COLOUR.get(verdict, "#9E9E9E"))
        # A bar can be red while sitting left of the red line, because a bad
        # slope condemns a market on its own (see `calibration.grade`). Hatching
        # says which rule fired, so the chart never looks like it contradicts
        # its own threshold.
        hatches.append("///" if verdict != TRUST and r["ece"] < ECE_AVOID
                       and slope_is_off(r["slope"], r["slope_se"]) else "")

    fig, ax = plt.subplots(figsize=(9.0, 0.44 * len(labels) + 2.4))
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colours, edgecolor="black", linewidth=.4)
    for bar, h in zip(bars, hatches):
        if h:
            bar.set_hatch(h)
    ax.set_yticks(y, labels, fontsize=9)
    ax.invert_yaxis()

    ax.axvline(ECE_TRUST, ls="--", lw=1, color="#2E7D32")
    ax.axvline(ECE_AVOID, ls="--", lw=1, color="#B71C1C")
    ax.text(ECE_TRUST, -0.85, " trust", color="#2E7D32", fontsize=8, va="center")
    ax.text(ECE_AVOID, -0.85, " avoid", color="#B71C1C", fontsize=8, va="center")

    for yi, (v, r) in enumerate(zip(values, s.to_dict("records"))):
        ax.text(v + 0.0015, yi, f"{v:.3f}  (slope {r['slope']:.2f})",
                va="center", fontsize=8, color="#333333")

    ax.set_xlabel("expected calibration error  (lower is better)")
    ax.set_xlim(0, max(max(values) * 1.45, ECE_AVOID * 1.35))
    ax.set_title(f"{target} - which markets are trustworthy", fontweight="bold", color=INK)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [Patch(facecolor=VERDICT_COLOUR[v], edgecolor="black", label=v)
               for v in (TRUST, CAUTION, AVOID)]
    if any(hatches):
        handles.append(Patch(facecolor="white", edgecolor="black", hatch="///",
                             label="flagged on slope, not ECE"))
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=.95)
    fig.tight_layout()
    return fig


# --- 2. Reliability -------------------------------------------------------


def reliability_figure(bins: pd.DataFrame, target: str, metric: str, segment: str = "all"):
    """The classic reliability diagram, one panel per line, with error bars."""
    import matplotlib.pyplot as plt

    b = _slice(bins, target, metric, segment)
    lines = _lines(b)
    if not lines:
        return None

    fig, axes = plt.subplots(1, len(lines), figsize=(3.5 * len(lines), 3.8), squeeze=False)
    for ax, line in zip(axes[0], lines):
        v = b[b["line"] == line].sort_values("pred_mean")
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#54008B", zorder=1)

        err = np.vstack([
            (v["obs_freq"] - v["ci_lo"]).clip(lower=0).to_numpy(),
            (v["ci_hi"] - v["obs_freq"]).clip(lower=0).to_numpy(),
        ])
        ax.errorbar(v["pred_mean"], v["obs_freq"], yerr=err, fmt="none",
                    ecolor="#999999", elinewidth=1, capsize=2.5, zorder=2)
        ax.scatter(v["pred_mean"], v["obs_freq"],
                   s=(v["n"] / max(v["n"].max(), 1) * 170).clip(24),
                   color="#027B5B", edgecolor="k", linewidth=.5, alpha=.85, zorder=3)

        ece = float((v["n"] / v["n"].sum() * (v["pred_mean"] - v["obs_freq"]).abs()).sum())
        ax.set_title(f"Over {line}\nECE = {ece:.3f}", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("predicted")
    axes[0][0].set_ylabel("observed")

    fig.suptitle(f"{target} - reliability [{metric}] ({segment})", fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


# --- 3. Where it goes wrong -----------------------------------------------


def _decile_grid(b: pd.DataFrame, lines: list[float]) -> np.ndarray:
    """Mean gap per (line, probability decile), n-weighted; NaN where unpopulated.

    Bands are re-placed onto fixed deciles by their mean prediction. Two bands
    can land in one decile, so the merge is weighted by count rather than a
    plain mean -- otherwise a 60-fixture band would pull as hard as a 600.
    """
    grid = np.full((len(lines), N_DECILES), np.nan)
    for i, line in enumerate(lines):
        v = b[b["line"] == line]
        d = np.clip((v["pred_mean"] * N_DECILES).astype(int), 0, N_DECILES - 1)
        for col in range(N_DECILES):
            m = d == col
            if m.any():
                w = v.loc[m, "n"].to_numpy(dtype=float)
                grid[i, col] = float(np.average(v.loc[m, "gap"].to_numpy(), weights=w))
    return grid


def gap_heatmap(bins: pd.DataFrame, target: str, metric: str, segment: str = "all"):
    """Observed minus predicted, by line and by real probability range."""
    import matplotlib.pyplot as plt

    b = _slice(bins, target, metric, segment)
    lines = _lines(b)
    if not lines:
        return None

    grid = _decile_grid(b, lines)
    lim = float(np.nanmax(np.abs(grid))) if np.isfinite(grid).any() else 0.1

    fig, ax = plt.subplots(figsize=(1.05 * N_DECILES + 3, 0.62 * len(lines) + 2.4))
    im = ax.imshow(grid, cmap=GAP_CMAP, vmin=-lim, vmax=lim, aspect="auto")

    ax.set_yticks(range(len(lines)), [f"Over {x}" for x in lines])
    ax.set_xticks(range(N_DECILES),
                  [f"{i * 100 // N_DECILES}-{(i + 1) * 100 // N_DECILES}%" for i in range(N_DECILES)],
                  rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("what the model predicted")

    for i in range(len(lines)):
        for j in range(N_DECILES):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:+.02f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="observed - predicted  (green = happens more often than stated)")
    ax.set_title(f"{target} - where it goes wrong [{metric}] ({segment})",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


# --- Saving ---------------------------------------------------------------


def frontier_return_figure(hist: pd.DataFrame, dist: pd.DataFrame | None = None,
                           placed: pd.DataFrame | None = None,
                           run_code: str | None = None):
    """What every portfolio on one slate's frontier actually returned.

    The one picture that argues for caring about selection: same slate, same
    propositions, and a bar chart that runs from wiped out to nearly tripled. The
    median and the bet actually placed are drawn on it, because the distance
    between those two is the part that is yours to control.

    **One slate at a time**, and `run_code` defaults to the most recent rather
    than to the pooled `"all"` row. Pooling was the original behaviour and it was
    wrong twice over: two slates with medians 55 points apart make a mixture that
    describes neither, and the marker showed the *first* bet ever placed rather
    than the one from the slate being drawn -- so the chart kept reporting an
    old result against a new distribution.
    """
    import matplotlib.pyplot as plt

    runs = [r for r in hist["run_code"].unique() if r != "all"]
    if run_code is None:
        run_code = max(runs) if runs else "all"
    h = hist[hist["run_code"] == run_code]
    if h.empty:
        return None
    h = h.sort_values("lo")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(range(len(h)), h["n"], color="#027B5B", alpha=.85, edgecolor="k", linewidth=.5)
    ax.set_xticks(range(len(h)))
    ax.set_xticklabels([f"{lo:+.0%}" for lo in h["lo"]], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("realised return (band lower edge)")
    ax.set_ylabel("portfolios")

    los, his = h["lo"].to_numpy(), h["hi"].to_numpy()
    drawn: list[float] = []

    def rule(x, color, label):
        """Place a return on the categorical bar scale.

        Interpolated *within* its band, not snapped to a band edge: the median
        and the bet placed routinely fall in the same 25-point band, and snapping
        drew both lines on top of each other with their labels overprinted --
        hiding the one comparison the chart exists to make.
        """
        i = int(np.clip(np.searchsorted(his, x, side="left"), 0, len(h) - 1))
        lo, hi = los[i], his[i]
        t = 0.5 if not np.isfinite(hi - lo) or hi <= lo else (x - lo) / (hi - lo)
        pos = i - 0.5 + float(np.clip(t, 0, 1))
        ax.axvline(pos, color=color, ls="--", lw=1.4, zorder=5)
        # Stack labels when two markers land close together, for the same reason.
        row = sum(abs(pos - q) < 1.2 for q in drawn)
        drawn.append(pos)
        top = ax.get_ylim()[1]
        ax.annotate(label, xy=(pos, top * .97), xytext=(4, -12 - row * 12),
                    textcoords="offset points", color=color, fontsize=8,
                    va="top", ha="left", zorder=6)

    label = run_code
    if dist is not None and not dist.empty:
        row = dist[dist["run_code"] == run_code]
        if not row.empty:
            r = row.iloc[0]
            if r.get("run_date"):
                label = f"{run_code}  {r['run_date']}"
            rule(r["median"], "#54008B", f"median {r['median']:+.1%}")
            ax.set_title(f"{label}   {int(r['n']):,} portfolios   "
                         f"{r['min']:+.0%} to {r['max']:+.0%}   "
                         f"{r['share_profitable']:.0%} profitable", fontsize=10)
    # The bet from *this* slate, not row zero of every bet ever placed.
    if placed is not None and not placed.empty and "run_code" in placed:
        mine = placed[placed["run_code"] == run_code]
        if not mine.empty:
            v = float(mine["realised"].iloc[0])
            rule(v, "#D16002", f"your bet {v:+.1%}  (p{mine['percentile'].iloc[0]:.0%})")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    return fig


def save_calibration_figures(summary: pd.DataFrame, bins: pd.DataFrame, out_dir: Path,
                             segment: str = "all") -> dict[str, list[Path]]:
    """Write every figure to ``out_dir``; return the paths keyed by target."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made: dict[str, list[Path]] = {}

    for target in sorted(bins["target"].unique()):
        paths = []

        def _save(fig, name):
            if fig is None:
                return
            p = out_dir / f"calibration_{target}_{name}.png"
            fig.savefig(p, dpi=140, bbox_inches="tight")
            plt.close(fig)
            paths.append(p)

        # Verdict bar first: it is the summary the other two support.
        _save(ece_by_line_figure(summary, target, segment), "verdicts")
        for metric in sorted(bins[bins["target"] == target]["metric"].unique()):
            _save(reliability_figure(bins, target, metric, segment), f"{metric}_reliability")
            _save(gap_heatmap(bins, target, metric, segment), f"{metric}_gap")
        if paths:
            made[target] = paths
    return made


__all__ = ["ece_by_line_figure", "reliability_figure", "gap_heatmap",
           "frontier_return_figure", "save_calibration_figures"]
