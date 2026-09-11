"""Scale a promoted club's cold-start seed by what it actually did coming up.

`priors.promoted_seeds` answers "what do promoted sides in this league do", which
is a per-league constant: every promoted Premier League club is currently seeded
at 0.806x an established side on shots, whoever it is. That is already far better
than the league average it replaced -- the average was 32% too kind on goals --
but it still cannot tell a runaway champion from a play-off winner who scraped up.

This scales it per club:

    seed(team, stat, side) =
        promoted_seed(league, stat, side)
        x shrink(team_lower[stat, side] / benchmark_lower[stat, side])

The denominator is the benchmark of **other promoted clubs from that same
division**, not the division average. That distinction is the whole design. A
promoted club looks good against its old division by construction -- it just beat
it -- so dividing by the division average would mostly measure "was promoted",
which is already what `promoted_seeds` encodes. Dividing by what *other promoted
clubs* did in the same division isolates what is left: was this one better than
the sides that usually come up.

Attack and defence are measured separately
------------------------------------------
The ratio carries a ``side``, and both sides are read off what the club actually
recorded: ``for`` from the stats it produced in the division below, ``against``
from the stats it conceded there. A team's conceded value is its opponent's
produced value on the same match row, so the second tier supplies both at the
same sample size -- 10,675 team-rows on the current cache, no nulls on either
side and no extra ingest.

This used to be one number. `multipliers` returned a ``for`` ratio keyed on the
stat alone and `build.build_feature_table` applied it to every buffer of that
stat, defensive ones included, which inverted the meaning: a club that outscored
the promoted cohort in the second tier was seeded to *concede* proportionally
more in the top flight, and a club that underscored to concede less. Hull came up
having scored 11% below the benchmark and conceded 33% above it, and was seeded
to concede 11% below.

Nothing needs inverting to fix it, because the quantity being scaled is itself a
conceded rate: conceded less than the cohort gives a ratio below 1, which lowers
the seeded concession. The two ratios are simply independent, which is the whole
point -- a promoted side can come up on the back of its attack, its defence, or
neither.

Walk-forward throughout. The benchmark for a season reads strictly earlier
seasons, matching `priors.promoted_seeds`. `priors._ordered_seasons` documents a
leak of exactly this kind that went unnoticed for a while -- a cold start seeded
from five seasons into its own future -- so it is asserted here, not assumed.

xG has no lower-league source
-----------------------------
Understat covers the top five divisions and nothing below, so `xg` and `npxg` --
alone among the seven seeded stats -- cannot be measured for a promotion season.
They are estimated instead, from the three that ESPN does provide, fitted at
**season aggregate** on top-five data where all four coexist:

    predictor              match R2   season R2
    goals only                0.398       0.829
    shots only                0.416       0.709
    shots + SoT               0.529       0.819
    shots + SoT + goals       0.633       0.875

Match by match, goals is a poor stand-in for xG -- which is the entire reason xG
exists. Aggregated over a season it reaches 0.829, because finishing luck is what
separates them and finishing luck averages out over forty-odd games. The fit uses
all three for 0.875, and a season aggregate is the only level this is ever used
at.

A systematic difference in chance quality per shot between divisions would bias
the proxy, and largely cancels: it lands on the club's estimate and the
benchmark's estimate alike, and only their ratio survives.

One fit serves both sides. The relationship is linear, so ``E[f(x)] = f(E[x])``
and coefficients fitted on ``_for`` transfer to ``_against`` exactly at season
aggregate -- a club's conceded xG is the mean of its opponents' produced xG, and
the mean of a linear function is that function of the mean. Measured on the top
five divisions, the transferred coefficients score R2 0.788 on ``xg_against`` and
0.762 on ``npxg_against``, against 0.797 and 0.772 for a fit trained on the
against side directly. A 0.008 gain does not justify a second fit, and only the
club/benchmark ratio survives either way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LOWER_LEAGUES, LOWER_OF, season_sort_key
from .paths import ESPN_MATCH_STATS

# One promotion season is a small sample -- 46 games in the Championship, fewer
# elsewhere -- and the ratio of two noisy means is noisier than either. Shrink
# toward 1 by how much of a full season the club actually played, then clip.
#
# The clip is not a safety net for a bad fit, it is a statement about what the
# quantity can mean: a promoted club that was twice as good as the usual promoted
# club is a claim about 46 games of second-tier football, and letting it double a
# seed would put more weight on that than the entire promoted profile behind it.
SHRINK_MATCHES = 46.0
CLIP_LO, CLIP_HI = 0.75, 1.33

# Stats ESPN provides below the top flight, and the ones that must be estimated.
MEASURED = ("goals", "shots", "sot", "corners")
ESTIMATED = ("xg", "npxg")

# Produced and conceded, measured separately. Same spelling as `spec.SIDES`, and
# the same meaning as the `_for`/`_against` suffixes everywhere else in the
# pipeline, so a multiplier keyed on a side drops straight onto a buffer.
SIDES = ("for", "against")


def lower_league_stats(stats: pd.DataFrame | None = None) -> pd.DataFrame:
    """Second-tier matches, one row per team, ready for `multipliers`.

    The gap this closes: `clean.to_team_perspective` runs on the Understat match
    table, which stops at the top five divisions, so nothing turned the ESPN
    second-tier rows into the shape the rest of this module reads. Returns an
    empty frame when none have been ingested, which is the state before
    `ingest.scoreboard.backfill_lower_leagues` has been run and the reason every
    caller degrades to the blanket promoted seed rather than failing.

    Both sides, as `clean.to_team_perspective` does upstream: ``{stat}_for`` from
    this team's column and ``{stat}_against`` from its opponent's on the same
    match. Only the *name* of an opponent can be missing on a second-tier row --
    `ingest.scoreboard.build_match_stats` keeps rows where either side maps, and
    the counts are parsed for both sides regardless -- so the conceded columns
    come free, at the same sample size, with nothing extra to fetch.

    Season is derived from the date rather than carried: ESPN scoreboards are
    dated, not seasoned. A European season is labelled by the calendar year it
    starts in, so anything from July belongs to that year's season -- the same
    convention `config.season_sort_key` reads.
    """
    if stats is None:
        if not ESPN_MATCH_STATS.exists():
            return pd.DataFrame()
        stats = pd.read_csv(ESPN_MATCH_STATS)
    if stats.empty or "league_key" not in stats.columns:
        return pd.DataFrame()

    sub = stats[stats["league_key"].isin(LOWER_LEAGUES)].copy()
    if sub.empty:
        return pd.DataFrame()

    six = [f"{s}_{c}" for s in ("home", "away") for c in ("shots", "sot", "corners")]
    have = [c for c in six if c in sub.columns]
    if have:
        # the same empty-stats sentinel `scoreboard._is_sentinel` rejects at parse
        sub = sub[~(sub[have] == 0).all(axis=1)]

    d = pd.to_datetime(sub["date"])
    start = np.where(d.dt.month >= 7, d.dt.year, d.dt.year - 1)
    sub["season"] = [f"{y}/{y + 1}" for y in start]

    frames = []
    for venue, other in (("home", "away"), ("away", "home")):
        cols = {"team": sub[f"{venue}_team"], "league_key": sub["league_key"],
                "season": sub["season"], "date": sub["date"]}
        for stat in MEASURED:
            for side, src_venue in (("for", venue), ("against", other)):
                src = f"{src_venue}_{stat}"
                if src in sub.columns:
                    cols[f"{stat}_{side}"] = sub[src].to_numpy()
        frames.append(pd.DataFrame(cols))
    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=["team"]).reset_index(drop=True)


def _profile_columns(df: pd.DataFrame, stats) -> list[str]:
    """The ``{stat}_{side}`` columns of `df` worth averaging, in a stable order.

    Tolerant of a side being absent rather than requiring both. That is what keeps
    the module safe to run against a partial frame: a missing column yields no
    multiplier for that ``(stat, side)``, and `priors.Seeds.multiplier` reads an
    absent entry as 1.0, so the blanket promoted profile stands unscaled there.
    """
    return [f"{s}_{side}" for s in stats for side in SIDES if f"{s}_{side}" in df.columns]


def _per_match(df: pd.DataFrame, stats) -> pd.DataFrame:
    """``(team, league_key, season)`` -> per-match mean of each stat, plus a count."""
    cols = _profile_columns(df, stats)
    g = df.groupby(["team", "league_key", "season"], sort=False)
    out = g[cols].mean()
    out["matches"] = g.size()
    return out.reset_index()


def promotion_seasons(top: pd.DataFrame) -> pd.DataFrame:
    """Every ``(team, parent_league, season)`` a club was new to the top flight.

    Read off `clean.attach_movement_flags`' own `team_is_new_to_league`, so
    "promoted" means exactly what the feature already means and there is not a
    second definition to drift from it.
    """
    if "team_is_new_to_league" not in top.columns:
        raise KeyError("expected team_is_new_to_league -- run attach_movement_flags first")
    new = top[top["team_is_new_to_league"] == 1]
    out = new[["team", "league_key", "season"]].drop_duplicates()
    out = out.rename(columns={"league_key": "parent"})
    out["lower"] = out["parent"].map(LOWER_OF)
    out["year"] = out["season"].map(season_sort_key)
    return out.dropna(subset=["lower"]).reset_index(drop=True)


def fit_xg_proxy(top: pd.DataFrame) -> dict[str, np.ndarray]:
    """Season-aggregate xG and npxG from shots, SoT and goals. ``{stat: coeffs}``.

    Fitted on the top five divisions, the only place all four exist together, and
    applied below them. Least squares on ``[1, shots, sot, goals]`` per match,
    averaged over a team-season -- never per match, where the relationship is
    much weaker and the noise is the part xG is meant to remove.
    """
    need = ("shots_for", "sot_for", "goals_for")
    agg = _per_match(top, ("shots", "sot", "goals", "xg", "npxg"))
    # No inputs, no proxy. Returning empty rather than raising is what lets
    # `multipliers` run on a table carrying only the measured stats -- the
    # promoted ratio for goals, shots, SOT and corners is still perfectly
    # computable without xG, and refusing all four because two are missing would
    # be the wrong trade.
    if any(c not in agg.columns for c in need):
        return {}
    agg = agg[agg["matches"] >= 25]
    X = np.column_stack([np.ones(len(agg)), agg["shots_for"], agg["sot_for"], agg["goals_for"]])
    fits: dict[str, np.ndarray] = {}
    for stat in ESTIMATED:
        col = f"{stat}_for"
        if col not in agg.columns:
            continue
        m = np.isfinite(agg[col].to_numpy()) & np.isfinite(X).all(axis=1)
        if m.sum() < 50:
            continue
        beta, *_ = np.linalg.lstsq(X[m], agg.loc[m, col].to_numpy(), rcond=None)
        fits[stat] = beta
    return fits


def apply_xg_proxy(profile: pd.DataFrame, fits: dict[str, np.ndarray]) -> pd.DataFrame:
    """Add estimated ``xg``/``npxg`` columns to a lower-league profile, both sides.

    The same coefficients serve produced and conceded: the fit is linear, so a
    club's conceded xG -- the mean of its opponents' produced xG -- is that linear
    function of its conceded shots, SoT and goals. See the module docstring for
    the measured transfer.

    A side whose inputs are absent is skipped rather than estimated from the
    other, which would reintroduce exactly the coupling this module exists to
    remove.
    """
    out = profile.copy()
    for side in SIDES:
        need = [f"{s}_{side}" for s in ("shots", "sot", "goals")]
        if any(c not in out.columns for c in need):
            continue
        X = np.column_stack([np.ones(len(out)), *(out[c] for c in need)])
        for stat, beta in fits.items():
            out[f"{stat}_{side}"] = np.clip(X @ beta, 0.0, None)
    return out


def _benchmark(profiles: pd.DataFrame, promos: pd.DataFrame,
               lower: str, before_year: int, stats) -> dict[tuple[str, str], float]:
    """Mean profile of clubs promoted out of `lower` in seasons strictly earlier.

    Keyed ``(stat, side)``: a club is judged against what the usual promoted club
    produced *and* against what it conceded, which are different questions with
    different answers.

    Strictly earlier is the whole causal guarantee: a club's own promotion season,
    and every later one, is invisible to the benchmark it is measured against.
    """
    earlier = promos[(promos["lower"] == lower) & (promos["year"] < before_year)]
    if earlier.empty:
        return {}
    # a club promoted for season Y played its promotion season in Y-1
    want = {(r.team, r.year - 1) for r in earlier.itertuples()}
    sub = profiles[profiles["lower"] == lower]
    sel = sub[[(t, y) in want for t, y in zip(sub["team"], sub["year"])]]
    if sel.empty:
        return {}
    return {(s, side): float(sel[f"{s}_{side}"].mean())
            for s in stats for side in SIDES if f"{s}_{side}" in sel.columns}


def multipliers(top: pd.DataFrame, lower_stats: pd.DataFrame) -> dict[tuple, float]:
    """``{(team, parent_league, season, stat, side): multiplier}``, shrunk and clipped.

    ``lower_stats`` is a team-perspective table of second-tier matches -- the same
    shape `clean.to_team_perspective` produces, carrying a ``league_key`` from
    `config.LOWER_LEAGUES`. Absent, the result is empty and every seed keeps the
    blanket promoted ratio, which is why this is safe to wire in before the
    second-tier backfill has run.

    ``for`` and ``against`` are computed independently from the club's own
    produced and conceded rates, and neither is derived from the other. The
    shrink and the clip apply per side, on the same terms.
    """
    stats = MEASURED + ESTIMATED
    promos = promotion_seasons(top)
    if promos.empty or lower_stats is None or lower_stats.empty:
        return {}

    profiles = _per_match(lower_stats, MEASURED)
    profiles["year"] = profiles["season"].map(season_sort_key)
    profiles = profiles.rename(columns={"league_key": "lower"})
    fits = fit_xg_proxy(top)
    if fits:
        profiles = apply_xg_proxy(profiles, fits)

    idx = {(t, l, y): i for i, (t, l, y)
           in enumerate(zip(profiles["team"], profiles["lower"], profiles["year"]))}

    out: dict[tuple, float] = {}
    for r in promos.itertuples():
        i = idx.get((r.team, r.lower, r.year - 1))     # the club's promotion season
        if i is None:
            continue                                   # never seen down there
        row = profiles.iloc[i]
        bench = _benchmark(profiles, promos, r.lower, r.year, stats)
        if not bench:
            continue                                   # no earlier promoted cohort
        w = min(1.0, float(row["matches"]) / SHRINK_MATCHES)
        for stat in stats:
            for side in SIDES:
                col = f"{stat}_{side}"
                # No inversion on the conceded side: the seed being scaled is
                # itself a conceded rate, so conceding less than the cohort gives
                # a ratio below 1 and lowers it, which is the direction wanted.
                if col not in row or (stat, side) not in bench or not bench[(stat, side)]:
                    continue
                raw = float(row[col]) / bench[(stat, side)]
                if not np.isfinite(raw):
                    continue
                # shrink toward 1 by how much of a season backs it, then clip
                out[(r.team, r.parent, r.season, stat, side)] = float(
                    np.clip(1.0 + w * (raw - 1.0), CLIP_LO, CLIP_HI))
    return out


__all__ = ["lower_league_stats", "promotion_seasons", "fit_xg_proxy", "apply_xg_proxy", "multipliers",
           "MEASURED", "ESTIMATED", "SIDES", "SHRINK_MATCHES", "CLIP_LO", "CLIP_HI"]
