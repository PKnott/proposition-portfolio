"""Seaborn figures for `07_Analysis`, one function per chart.

Every function takes the frames `fpp.study` builds and returns a matplotlib
`Figure`, so the notebook can show it and `save` can write it to
`Outputs/Analysis/Reports/Figures/`.

Colour carries one meaning throughout:

* **blue = the model, orange = the price**, wherever the two are compared;
* aqua = a measured correlation, yellow = the value currently in config;
* diverging maps have a grey zero, and red is always the side that costs return:
  outcomes landing together (correlation) or landing less than predicted
  (calibration);
* the four stats get their own colours only in the dampening section.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from .paths import ANALYSIS_REPORTS
from .study import TARGETS, apply_recalibration, expit, fit_recalibration, logit

MODEL = "#2a78d6"
MARKET = "#eb6834"
MEASURED = "#1baf7a"
CONFIGURED = "#eda100"
NEUTRAL = "#898781"
INK = "#52514e"
SOURCE_PALETTE = {"model": MODEL, "market": MARKET}
# One per stat, only in the dampening section, from slots that mean nothing elsewhere.
STAT_COLOURS = ("#008300", "#e87ba4", "#4a3aa7", "#8a5a00")

# Grey midpoint, never a hue: zero has to read as "nothing".
DIVERGING = LinearSegmentedColormap.from_list(
    "fpp_div", ["#184f95", "#6da7ec", "#f0efec", "#ec8b8a", "#b02a2a"])
DIVERGING_R = DIVERGING.reversed()
SEQUENTIAL = LinearSegmentedColormap.from_list("fpp_seq", ["#e6f0fc", "#86b6ef", "#1c5cab"])

FIGURES = ANALYSIS_REPORTS / "Figures"


def theme() -> None:
    sns.set_theme(style="whitegrid", context="notebook", rc={
        "axes.edgecolor": "#d6d5d0", "grid.color": "#ebeae6", "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.titleweight": "bold",
        "axes.titlesize": 12, "figure.dpi": 110, "savefig.dpi": 150,
        "savefig.bbox": "tight"})


def save(fig, name: str, out_dir: Path | None = None) -> Path:
    out = Path(out_dir) if out_dir else FIGURES
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"study_{name}.png"
    fig.savefig(path)
    return path


def _title(fig, title: str, subtitle: str | None = None) -> None:
    """Title and subtitle at fixed distances from the top, then lay the axes out below them.

    Positioned in inches rather than figure fractions so the gap is the same on a
    short figure as on a tall one.
    """
    h = fig.get_figheight()
    fig.text(0.01, 1 - 0.12 / h, title, ha="left", va="top", fontweight="bold", fontsize=13)
    if subtitle:
        fig.text(0.01, 1 - 0.42 / h, subtitle, ha="left", va="top", color=INK, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.75 if subtitle else 0.5) / h))


# --- data ----------------------------------------------------------------


def coverage(s: pd.DataFrame):
    """Graded propositions per league and stat: where the evidence actually is."""
    t = s.pivot_table(index="league", columns="target", values="y", aggfunc="size",
                      fill_value=0).reindex(columns=list(TARGETS))
    t.loc["all"] = t.sum()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(t, annot=True, fmt=",d", cmap=SEQUENTIAL, cbar=False, linewidths=2,
                linecolor="white", ax=ax, mask=t.index.to_series().eq("all")
                .to_numpy()[:, None].repeat(t.shape[1], 1))
    for j, v in enumerate(t.loc["all"]):
        ax.text(j + 0.5, len(t) - 0.5, f"{v:,d}", ha="center", va="center",
                fontweight="bold", color=INK)
    ax.set(xlabel="", ylabel="")
    ax.tick_params(left=False, bottom=False)
    _title(fig, "What the study is built on",
           f"{len(s):,} graded propositions · {s['event'].nunique()} matches · "
           f"{s['run_code'].nunique()} slates ({s['run_code'].min()}–{s['run_code'].max()})")
    return fig


# --- correlation ----------------------------------------------------------


def correlation_heatmaps(pairs: pd.DataFrame, rho_table, min_pairs: int = 30):
    """ρ̂ for every stat × stat combination, same team on the left, opposite teams on the right.

    Each cell is annotated with ρ̂ and the number of matches behind it. Cells with
    fewer than `min_pairs` pairs are left blank rather than shown as a number
    nobody should read.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.4), sharey=True)
    for ax, same, name in ((axes[0], True, "Same team"), (axes[1], False, "Opposite teams")):
        t = rho_table(pairs[pairs["same_team"] == same], ["stat_a", "stat_b"])
        full = pd.concat([t, t.rename(columns={"stat_a": "stat_b", "stat_b": "stat_a"})])
        full = full.drop_duplicates(["stat_a", "stat_b"])
        rho = full.pivot(index="stat_a", columns="stat_b", values="rho").reindex(
            index=list(TARGETS), columns=list(TARGETS))
        n = full.pivot(index="stat_a", columns="stat_b", values="pairs").reindex(
            index=list(TARGETS), columns=list(TARGETS))
        m = full.pivot(index="stat_a", columns="stat_b", values="matches").reindex(
            index=list(TARGETS), columns=list(TARGETS))
        annot = rho.map(lambda v: "" if pd.isna(v) else f"{v:+.2f}") + "\n" + m.map(
            lambda v: "" if pd.isna(v) else f"{int(v)} matches")
        sns.heatmap(rho.where(n >= min_pairs), annot=annot.where(n >= min_pairs, ""),
                    fmt="", cmap=DIVERGING, vmin=-0.8, vmax=0.8, center=0, square=True,
                    linewidths=2, linecolor="white", cbar=ax is axes[1],
                    cbar_kws={"label": "ρ", "shrink": 0.8}, ax=ax,
                    annot_kws={"fontsize": 9})
        ax.set(title=name, xlabel="", ylabel="")
        ax.tick_params(left=False, bottom=False)
    _title(fig, "Which outcomes move together on the same match",
           "Red: both tend to land together (less diversification). "
           "Blue: one landing makes the other less likely (a natural hedge).")
    return fig


def correlation_by_gap(pairs: pd.DataFrame, rho_table, placebo: dict, configured: dict):
    """Same team, same stat: how quickly the overlap fades as the two lines move apart."""
    sub = pairs[pairs["same_team"] & pairs["same_stat"]].copy()
    sub["gap"] = sub["line_gap"].clip(upper=4).round()
    t = rho_table(sub, ["gap"])
    t = t[t["pairs"] >= 30]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhspan(placebo["rho"] - 1.96 * placebo["se"], placebo["rho"] + 1.96 * placebo["se"],
               color=NEUTRAL, alpha=0.18, lw=0)
    ax.axhline(0, color=NEUTRAL, lw=1)
    ax.errorbar(t["gap"], t["rho"], yerr=1.96 * t["se"], fmt="o-", color=MEASURED, lw=2,
                ms=8, capsize=4, label="measured (95% CI)")
    cfg = [(1, configured["same_team_same_stat_adjacent"]),
           (2, configured["same_team_same_stat_near"]),
           (3, configured["same_team_same_stat_far"]),
           (4, configured["same_team_same_stat_far"])]
    ax.plot(*zip(*cfg), "s", mfc="none", mec=CONFIGURED, mew=2.2, ms=11,
            label="config.PAIR_CORRELATION")
    for _, r in t.iterrows():
        ax.annotate(f"{int(r['pairs']):,} pairs", (r["gap"], r["hi"]), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.text(4.35, placebo["rho"], "placebo\n(different\nmatches)", va="center", fontsize=8,
            color=NEUTRAL)
    ax.set(xlabel="distance between the two lines", ylabel="ρ", xlim=(0.5, 4.8),
           xticks=[1, 2, 3, 4], xticklabels=["1", "2", "3", "4+"])
    ax.legend(loc="upper right", frameon=False)
    _title(fig, "Two lines on the same team's stat: overlap by distance",
           "Adjacent lines are close to the same position; the further apart, the more "
           "independent.")
    return fig


def correlation_vs_config(tables: dict[str, pd.DataFrame]):
    """Measured ρ̂ per `PAIR_CORRELATION` class against config, for each sample in `tables`."""
    rows = []
    for name, t in tables.items():
        rows.append(t.assign(sample=name))
    t = pd.concat(rows, ignore_index=True)
    labels = {
        "same_team_same_stat_adjacent": "same team, same stat\nlines ≤1 apart",
        "same_team_same_stat_near": "same team, same stat\nlines ~2 apart",
        "same_team_same_stat_far": "same team, same stat\nlines 3+ apart",
        "same_team_diff_stat": "same team,\ndifferent stat",
        "opposite_teams": "opposite\nteams",
    }
    order = list(labels)
    samples = list(tables)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = 0.8 / len(samples)
    shades = [MEASURED, "#0f6e4c"]
    for k, name in enumerate(samples):
        d = t[t["sample"] == name].set_index("relation").reindex(order)
        x = np.arange(len(order)) + (k - (len(samples) - 1) / 2) * width
        ax.bar(x, d["rho"], width * 0.9, color=shades[k % 2], label=name)
        ax.errorbar(x, d["rho"], yerr=[d["rho"] - d["lo"], d["hi"] - d["rho"]], fmt="none",
                    ecolor=INK, capsize=3, lw=1.2)
    cfg = t.drop_duplicates("relation").set_index("relation").reindex(order)["configured"]
    ax.scatter(np.arange(len(order)), cfg, marker="D", s=70, facecolor="white",
               edgecolor=CONFIGURED, linewidth=2.2, zorder=5, label="currently in config")
    ax.axhline(0, color=NEUTRAL, lw=1)
    ax.set_xticks(np.arange(len(order)), [labels[o] for o in order], fontsize=9)
    ax.set_ylabel("ρ (95% CI, clustered by match)")
    ax.legend(frameon=False, loc="upper right")
    _title(fig, "Measured correlation against what the portfolio builder assumes",
           "A diamond outside its bar's interval is a config value the data disagrees with.")
    return fig


# --- calibration ----------------------------------------------------------


def reliability(rel: pd.DataFrame, s: pd.DataFrame, by: str = "target"):
    """Predicted against observed, model and price side by side, one panel per group."""
    levels = [t for t in TARGETS if t in set(rel[by])] if by == "target" else sorted(rel[by].unique())
    fig, axes = plt.subplots(1, len(levels), figsize=(3.6 * len(levels), 4.2),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    top = float(min(1.0, max(rel["predicted"].max(), rel["hi"].max()) + 0.05))
    for ax, lvl in zip(axes, levels):
        ax.plot([0, top], [0, top], color=NEUTRAL, lw=1, ls="--")
        ax.fill_between([0, top], [0, top], [top, top], color=MODEL, alpha=0.04, lw=0)
        for src, colour in SOURCE_PALETTE.items():
            d = rel[(rel[by] == lvl) & (rel["source"] == src)].sort_values("predicted")
            ax.fill_between(d["predicted"], d["lo"], d["hi"], color=colour, alpha=0.15, lw=0)
            ax.plot(d["predicted"], d["observed"], "o-", color=colour, lw=2, ms=6, label=src)
        g = s[s[by] == lvl]
        sns.rugplot(x=g["p"], ax=ax, color=MODEL, alpha=0.08, height=0.03)
        ax.set(title=f"{lvl}  (n={len(g):,})", xlim=(0, top), ylim=(0, top),
               xlabel="predicted probability", ylabel="how often it landed")
        ax.set_aspect("equal")
    axes[0].text(0.03, top * 0.93, "above the line:\nlanded more often\nthan predicted",
                 fontsize=8, color=INK, va="top")
    axes[-1].legend(frameon=False, loc="lower right")
    _title(fig, "Reliability: does a 30% proposition land 30% of the time?",
           "Shaded bands are 95% Wilson intervals. Blue = the model's p, orange = 1/odds "
           "(still carrying the market's margin).")
    return fig


def gap_heatmap(gaps: dict[str, tuple[pd.DataFrame, pd.DataFrame]]):
    """Hit rate minus mean predicted probability, by stat × league, for each source.

    Positive (blue): the propositions landed more often than predicted, which on
    a portfolio of Overs is error in our favour. Negative (red): they landed less
    often, the direction that costs return.
    """
    fig, axes = plt.subplots(1, len(gaps), figsize=(6.2 * len(gaps), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    lim = max(float(np.nanmax(np.abs(g.to_numpy()))) for g, _ in gaps.values())
    lim = np.ceil(lim * 100) / 100
    for ax, (name, (gap, n)) in zip(axes, gaps.items()):
        gap = gap.reindex(index=[t for t in TARGETS if t in gap.index])
        n = n.reindex(index=gap.index, columns=gap.columns)
        annot = gap.map(lambda v: f"{v * 100:+.1f}pp") + "\n" + n.map(
            lambda v: "" if pd.isna(v) else f"n={int(v):,}")
        sns.heatmap(gap, annot=annot, fmt="", cmap=DIVERGING_R, vmin=-lim, vmax=lim, center=0,
                    linewidths=2, linecolor="white", cbar=ax is axes[-1],
                    cbar_kws={"label": "hit rate − mean predicted"}, ax=ax,
                    annot_kws={"fontsize": 8.5})
        ax.set(title=name, xlabel="", ylabel="")
        ax.tick_params(left=False, bottom=False)
    _title(fig, "Where the probabilities are too low, and where too high",
           "Blue: landed more than predicted (in our favour, as every position is an Over). "
           "Red: landed less than predicted (the costly direction).")
    return fig


def _forest(ax, t: pd.DataFrame, value: str, lo: str, hi: str, label: str,
            good_side: str = "left"):
    t = t.reset_index(drop=True)
    y = np.arange(len(t))[::-1]
    sig_good = (t[hi] < 0) if good_side == "left" else (t[lo] > 0)
    sig_bad = (t[lo] > 0) if good_side == "left" else (t[hi] < 0)
    colours = np.where(sig_good, MODEL, np.where(sig_bad, MARKET, NEUTRAL))
    ax.axvline(0, color=INK, lw=1)
    ax.hlines(y, t[lo], t[hi], color=colours, lw=2.5)
    ax.scatter(t[value], y, color=colours, s=60, zorder=3, edgecolor="white", linewidth=1.5)
    ax.set_yticks(y, t[label])
    ax.grid(axis="y", visible=False)


def market_forest(comp: pd.DataFrame):
    """Model minus market log loss, with its 95% interval. Left of zero: the model priced better."""
    t = comp.copy()
    t["label"] = np.where(t["grouping"] == "all", "ALL",
                          t["level"].astype(str)) + t["n"].map(lambda n: f"  ({n:,})")
    order = {"all": 0, "target": 1, "league": 2}
    t = t.sort_values(["grouping"], key=lambda c: c.map(order), kind="stable")
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(t) + 1.8))
    _forest(ax, t, "ll_diff", "ll_lo", "ll_hi", "label")
    for k, g in enumerate(["target", "league"]):
        idx = np.flatnonzero(t["grouping"].to_numpy() == g)
        if len(idx):
            y = len(t) - 1 - idx.max() - 0.5
            ax.axhline(y, color="#d6d5d0", lw=1)
    xmax = float(np.nanmax(np.abs(t[["ll_lo", "ll_hi"]].to_numpy()))) * 1.15
    ax.set_xlim(-xmax, xmax)
    ax.text(-xmax * 0.97, len(t) - 0.4, "← model priced better", color=MODEL, fontsize=9,
            va="bottom")
    ax.text(xmax * 0.97, len(t) - 0.4, "price was better →", color=MARKET, fontsize=9,
            ha="right", va="bottom")
    ax.set(xlabel="log loss, model − market (per proposition)", ylim=(-0.7, len(t) + 0.2))
    _title(fig, "Is the model sharper than the price?",
           "Coloured where the 95% interval clears zero; grey means no detectable difference.")
    return fig


def selection(sel_all: pd.DataFrame, sel_by: pd.DataFrame):
    """Predicted against landed, for the propositions held and the ones passed over.

    Each row is a dumbbell: blue is the model's mean p, orange the price's, and
    the black dot (with its 95% interval) is how often they actually landed. A
    black dot left of blue means the model was too high on that group.
    """
    t = pd.concat([sel_all.assign(target="all"), sel_by], ignore_index=True)
    order = ["all", *[x for x in TARGETS if x in set(t["target"])]]
    groups = ["not qualified", "qualified (held)"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)
    for ax, grp in zip(axes, groups):
        d = t[t["qualified"] == grp].set_index("target").reindex(order).dropna(subset=["n"])
        y = np.arange(len(d))[::-1]
        lo = d[["model_p", "hit_rate"]].min(axis=1)
        hi = d[["model_p", "hit_rate"]].max(axis=1)
        ax.hlines(y, lo, hi, color="#d6d5d0", lw=6, zorder=1)
        ax.scatter(d["market_p"], y, color=MARKET, s=55, zorder=3, label="price (1/odds)")
        ax.scatter(d["model_p"], y, color=MODEL, s=70, zorder=3, label="model p")
        ax.errorbar(d["hit_rate"], y, xerr=[d["hit_rate"] - d["hit_lo"], d["hit_hi"] - d["hit_rate"]],
                    fmt="o", color="#0b0b0b", ms=6, capsize=3, zorder=4, label="landed (95% CI)")
        ax.set_yticks(y, [f"{i}  ({int(n):,})" for i, n in zip(d.index, d["n"])])
        ax.grid(axis="y", visible=False)
        ax.set(title=grp, xlabel="probability")
    axes[1].legend(frameon=False, fontsize=8.5, loc="lower right")
    _title(fig, "On the positions actually taken, is the model still too low?",
           "Black dot right of blue: landed more than the model said. "
           "Left of blue: the model was too high on that group.")
    return fig


# --- dampening / inflation ------------------------------------------------


def adjustment_curves(s: pd.DataFrame, constants: pd.DataFrame, glob: pd.DataFrame,
                      by: str = "target"):
    """What each fitted constant does to a probability: p in, p' out.

    On the identity line nothing changes. Above it the constant raises the
    probability; a curve flatter than the identity dampens, steeper inflates.
    The marginal shows where the model's probabilities actually sit, which is
    the only part of each curve that matters.
    """
    grid = np.linspace(0.03, 0.9, 200)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
    ax.plot(grid, grid, color=NEUTRAL, ls="--", lw=1, label="no adjustment")
    pal = dict(zip(TARGETS, STAT_COLOURS))
    for _, r in constants.iterrows():
        ax.plot(grid, expit(r["a"] + r["b"] * logit(grid)), lw=1.6,
                color=pal.get(r["level"], NEUTRAL), label=f"{r['level']}")
    g = glob.iloc[0]
    ax.plot(grid, expit(g["a"] + g["b"] * logit(grid)), color=INK, lw=2.6, label="global")
    ax.set(xlabel="model p", ylabel="adjusted p′", xlim=(0, 0.9), ylim=(0, 0.95))
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("The fitted adjustments")

    # Right: the same thing as a shift in percentage points, which is easier to read.
    for _, r in constants.iterrows():
        ax2.plot(grid, (expit(r["a"] + r["b"] * logit(grid)) - grid) * 100, lw=1.6,
                 color=pal.get(r["level"], NEUTRAL))
    ax2.plot(grid, (expit(g["a"] + g["b"] * logit(grid)) - grid) * 100, color=INK, lw=2.6)
    ax2.axhline(0, color=NEUTRAL, ls="--", lw=1)
    ax2b = ax2.twinx()
    sns.histplot(s["p"], bins=40, ax=ax2b, color=MODEL, alpha=0.12, edgecolor="none",
                 stat="density")
    ax2b.set(yticks=[], ylabel="")
    ax2b.grid(False)
    ax2.set_zorder(ax2b.get_zorder() + 1)
    ax2.patch.set_visible(False)
    ax2.set(xlabel="model p (shaded: where the model's p actually fall)",
            ylabel="change, percentage points", xlim=(0, 0.9))
    ax2.set_title("…as a change in percentage points")
    _title(fig, "What a dampening / inflation constant would do",
           "Fitted on every slate: logit p′ = a + b·logit p. Flatter than the diagonal = "
           "dampen, steeper = inflate, above it = shift up.")
    return fig


def recalibration_forest(summary: pd.DataFrame, by_level: pd.DataFrame | None = None):
    """Out-of-slate log-loss gain of each adjustment. Right of zero: it helped."""
    names = {"global": "one global constant  vs  none",
             "blend": "global + lean on the price  vs  global",
             "market": "use the price instead  vs  model",
             }
    t = summary.copy()
    t["label"] = [names.get(a, f"{a.replace('_', ' ')}  vs  global") for a in t["adjustment"]]
    panels = 1 if by_level is None else 2
    fig, axes = plt.subplots(1, panels, figsize=(7 * panels, 0.45 * max(len(t), 8) + 2))
    axes = np.atleast_1d(axes)
    _forest(axes[0], t, "gain", "lo", "hi", "label", good_side="right")
    axes[0].set(xlabel="log-loss gain per proposition (out of slate)")
    axes[0].set_title("Every adjustment")
    if by_level is not None:
        b = by_level.copy()
        b["label"] = b["level"].astype(str) + "  ·  " + b["comparison"].str.replace(
            " vs ", "  vs  ")
        _forest(axes[1], b, "gain", "lo", "hi", "label", good_side="right")
        axes[1].set(xlabel="log-loss gain per proposition (out of slate)")
        axes[1].set_title(f"Within each {b['context'].iloc[0]}")
    for ax in axes:
        lim = max(abs(x) for x in ax.get_xlim())
        ax.set_xlim(-lim, lim)
    _title(fig, "Would the adjustment have helped on slates it never saw?",
           "Leave-one-slate-out. Blue: the interval clears zero, so it helped; "
           "orange: it hurt; grey: no detectable effect.")
    return fig


def push_deflate(summary: pd.DataFrame, tuning: pd.DataFrame):
    """The backtest on one page: what each push does to growth and to honesty.

    Left: capital multiple over every held-out slate against the push `k`, with
    and without the learnt deflation; the dashed line is today. Right: on the
    positions each push selects, what the sizing probability said against what
    landed. Deflation exists to close that gap.
    """
    t = summary[summary["strategy"] != "tuned"].copy()
    t["mode"] = np.where(t["strategy"].str.endswith("push+deflate"), "push + deflate", "push only")
    today = float(t.loc[t["strategy"] == "k=0 push only", "capital_x"].iloc[0])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    styles = {"push + deflate": dict(color=MEASURED, marker="o"),
              "push only": dict(color=NEUTRAL, marker="s", ls="--")}
    for mode, g in t.groupby("mode"):
        g = g.sort_values("k")
        ax.plot(g["k"], g["capital_x"], lw=2, ms=7, label=mode, **styles[mode])
    ax.axhline(today, color=INK, lw=1, ls=":")
    ax.text(t["k"].max(), today, " today", va="bottom", ha="right", fontsize=8.5, color=INK)
    ax.set(xlabel="push k  (0 = none, 1 = the full calibration fit)",
           ylabel="capital multiple, all held-out slates",
           title="Growth, out of slate")
    ax.legend(frameon=False, loc="center right")

    for mode, g in t.groupby("mode"):
        g = g.sort_values("k")
        ax2.plot(g["k"], (g["said"] - g["landed"]) * 100, lw=2, ms=7, label=mode, **styles[mode])
    ax2.axhline(0, color=INK, lw=1)
    ax2.set(xlabel="push k", ylabel="sizing p − hit rate on the positions (pp)",
            title="How over-confident the sizing p is")
    d = tuning.filter(like="d@")
    ks = [float(c[2:]) for c in d.columns]
    ax2b = ax2.twinx()
    ax2b.errorbar(ks, d.mean() * 100, yerr=d.std() * 100, fmt="none", ecolor=CONFIGURED,
                  capsize=3, lw=1.5)
    ax2b.plot(ks, d.mean() * 100, color=CONFIGURED, lw=1.5, marker="D", ms=5,
              label="deflation learnt (± sd across folds)")
    ax2b.set_ylabel("deflation d (pp)", color=CONFIGURED)
    ax2b.grid(False)
    lo = min(ax2.get_ylim()[0], 0)
    hi = max(ax2.get_ylim()[1], ax2b.get_ylim()[1])
    ax2.set_ylim(lo, hi); ax2b.set_ylim(lo, hi)
    ax2b.legend(frameon=False, loc="upper left", fontsize=8.5)
    _title(fig, "Push before selecting, deflate before sizing",
           "Every amount learnt without the slate it is scored on. "
           "Quarter-Kelly per position, at most 20% of capital per slate.")
    return fig


REPLAY_COLOURS = {"today": "#0b0b0b", "deflate all": NEUTRAL,
                  "half push all, deflate all": "#4a3aa7",
                  "shots push, deflate others": "#008300",
                  "shots push, deflate all": "#e87ba4",
                  "shots push, no deflation": "#8a5a00"}


def portfolio_replay(bt: pd.DataFrame):
    """The real builder, slate by slate: the capital path, and promised against delivered.

    Left: capital under each strategy across the held-out slates in order.
    Right: today's pick on each slate, the return it promised per £1 against the
    return it delivered. The gap is the leg-level optimism compounded by picking
    the best-looking portfolio.
    """
    from .study import _run_number

    order = sorted(bt["slate"].unique(), key=_run_number)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 6.0), gridspec_kw={"width_ratios": [1.25, 1]})
    for name, g in bt.groupby("strategy"):
        g = g.set_index("slate").reindex(order)
        path = np.exp(np.concatenate([[0.0], g["growth"].fillna(0).cumsum().to_numpy()]))
        lw = 2.8 if name == "today" else 1.8
        ax.plot(range(len(order) + 1), path, lw=lw, color=REPLAY_COLOURS.get(name, NEUTRAL),
                label=f"{name}  ×{path[-1]:.2f}", zorder=3 if name == "today" else 2)
    ax.axhline(1, color=NEUTRAL, lw=1, ls=":")
    ax.set_xticks(range(len(order) + 1), ["start", *order], rotation=60, fontsize=8)
    ax.set(ylabel="capital (start = 1)", title="Capital through the real builder")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2),
              ncol=2)

    t = bt[bt["strategy"] == "today"].set_index("slate").reindex(order).dropna(subset=["R"])
    y = np.arange(len(t))[::-1]
    ax2.hlines(y, t["R"], t["expected_R"], color="#d6d5d0", lw=5, zorder=1)
    ax2.scatter(t["expected_R"], y, color=MODEL, s=55, zorder=3, label="promised (expected return)")
    ax2.scatter(t["R"], y, color="#0b0b0b", s=45, zorder=4, label="delivered")
    ax2.axvline(1, color=NEUTRAL, lw=1, ls=":")
    ax2.set_yticks(y, [f"{s}  ({int(n)} legs)" for s, n in zip(t.index, t["legs"])], fontsize=8.5)
    ax2.grid(axis="y", visible=False)
    ax2.set(xlabel="return per £1 allocated", title="Today's pick: promised vs delivered")
    ax2.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2),
               ncol=2)
    _title(fig, "Every slate through 06_Split's own search, pick and sizing",
           "Push and deflation learnt on the other slates. At most two legs per match, "
           "pairs priced with PAIR_CORRELATION.")
    return fig


def triples_replay(summary: pd.DataFrame):
    """Triples off against on, slate by slate: the capital path and each slate's swing."""
    a = summary.attrs
    order = list(summary.index)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.6), gridspec_kw={"width_ratios": [1.2, 1]})
    for col, colour, lw, name in (("growth_off", "#0b0b0b", 2.6, "triples off"),
                                  ("growth_on", "#008300", 2.0, "triples on")):
        path = np.exp(np.r_[0.0, summary[col].cumsum().to_numpy()])
        ax.plot(range(len(order) + 1), path, color=colour, lw=lw, label=f"{name}  ×{path[-1]:.2f}")
    ax.axhline(1, color=NEUTRAL, lw=1, ls=":")
    ax.set_xticks(range(len(order) + 1), ["start", *order], rotation=60, fontsize=8)
    ax.set(ylabel="capital (start = 1)", title="The 06 pick, sized as 06 sizes it")
    ax.legend(frameon=False, loc="upper left")

    d = (np.exp(summary["growth_on"] - summary["growth_off"]) - 1) * 100
    y = np.arange(len(order))[::-1]
    colours = np.where(d > 1e-6, "#008300", np.where(d < -1e-6, MARKET, NEUTRAL))
    ax2.barh(y, d, color=colours, height=0.6)
    ax2.axvline(0, color=INK, lw=1)
    ax2.set_yticks(y, [f"{s}  ({int(n)} triple options)" for s, n in
                       zip(order, summary["triple_options"])], fontsize=8.5)
    ax2.grid(axis="y", visible=False)
    ax2.set(xlabel="capital change from switching triples on (%)",
            title="Where it mattered")
    _title(fig, "Three legs per match: two on one team, one on the other",
           f"Off ×{a['capital_off']:.2f} → on ×{a['capital_on']:.2f}; gain ×{a['gain']:.2f} "
           f"(×{a['gain_lo']:.2f}–×{a['gain_hi']:.2f}); {a['better']} slates better, "
           f"{a['worse']} worse, {a['same']} unchanged.")
    return fig


def leg_cap(lc: pd.DataFrame):
    """What the per-leg cap does across every frontier portfolio: delivered, promised, profitable."""
    caps = sorted({float(c.split("@")[1]) for c in lc.columns if c.startswith("R@")}, reverse=True)
    x = np.arange(len(caps))
    labels = ["none" if c >= 1 else f"{c:.0%}" for c in caps]
    R = [lc[f"R@{c:g}"].mean() for c in caps]
    E = [lc[f"E@{c:g}"].mean() for c in caps]
    P = [lc[f"profitable@{c:g}"].mean() * 100 for c in caps]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax.plot(x, E, "o-", color=MODEL, lw=2, ms=7, label="promised")
    ax.plot(x, R, "o-", color="#0b0b0b", lw=2, ms=7, label="delivered")
    ax.set_xticks(x, labels)
    ax.set(xlabel="per-leg cap", ylabel="mean return per £1 (slate average)",
           title="Promised vs delivered, every frontier portfolio")
    ax.legend(frameon=False)
    cur = labels.index("15%") if "15%" in labels else None
    for a_ in (ax, ax2):
        if cur is not None:
            a_.axvline(cur, color=CONFIGURED, lw=1.5, ls="--")
    ax2.plot(x, P, "o-", color="#008300", lw=2, ms=7)
    ax2.set_xticks(x, labels)
    ax2.set(xlabel="per-leg cap", ylabel="% of frontier portfolios profitable",
            title="Share with a positive return")
    _title(fig, "The per-leg cap, tested on every viable portfolio",
           "Same legs restaked at each cap and settled; dashed line is today's 15%. "
           "Averages of per-slate means over 13 slates.")
    return fig


__all__ = [
    "theme", "save", "coverage", "correlation_heatmaps", "correlation_by_gap",
    "correlation_vs_config", "reliability", "gap_heatmap", "market_forest", "selection",
    "adjustment_curves", "recalibration_forest", "push_deflate",
    "portfolio_replay", "triples_replay", "leg_cap",
]
