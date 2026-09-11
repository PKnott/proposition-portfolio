"""Walk-forward rolling priors, vectorised.

Semantics (identical to the reference framework, and to the old
``decayed_mean_last_L``): for each row, each buffer's prior is the
recency-weighted mean of that team's most recent up-to-``L`` observations
**strictly before** this match. Weight of an observation ``age`` matches back is
``exp(-alpha * age)``, so the most recent carries weight 1 and ``alpha = 0``
gives a plain mean. Record-before-update, so a match never sees itself.

Two buffer families per stat:

* **all-venue**  -- every prior match, window ``L``
* **venue**      -- prior matches at the same venue only, window ``L_V = ceil(L/2)``

Why it is written this way
--------------------------
The old ``compute_priors`` looped per row with ``.loc`` scalar assignment (~60-90s
per build) and was re-run for every grid point -- the dominant cost of the whole
pipeline. Here the rows are laid out *team-contiguous with a front pad of
``L_max`` NaN rows per team*, which turns "the observation ``k`` matches ago" into
a plain **contiguous slice shift**, not fancy indexing. The whole pass is then
``L`` vectorised multiply-adds over one float32 array.

The mask denominator is what reproduces the partial-window semantics exactly: a
team with only 3 prior matches averages over 3, not over ``L``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import MIN_PROMOTED_SEASONS, PROMOTED_LOOKBACK, PROMOTED_WINDOW, season_sort_key
from .spec import BUFFER_NAMES, BUFFERS, SIDES

SORT_KEYS = ["date", "fixture_id", "is_home"]


def _weights(window: int, alpha: float) -> np.ndarray:
    """Weight by age: index 0 is the most recent prior match."""
    return np.exp(-alpha * np.arange(window, dtype=np.float64))


@dataclass
class _Blocks:
    """Group-contiguous, front-padded layout for one grouping (team, or team+venue)."""

    values: np.ndarray  # (n_padded, n_buffers) float32, NaN where unobserved/pad
    mask: np.ndarray  # (n_padded, n_buffers) float32, 1 where observed
    row_pos: np.ndarray  # (n_rows,) padded position of each *original* row
    n_prior: np.ndarray  # (n_rows,) count of that group's earlier rows
    pad: int


def _season_span(grp: np.ndarray, y: np.ndarray) -> float:
    """Typical rows per season for this grouping.

    Derived rather than hard-coded because it differs by grouping: ~38 for an
    all-venue team block, ~19 for a venue block, ~380 for a league block. Taking
    the median over (group, season) makes it adapt to a 34-match Bundesliga
    season as well as a 38-match one.
    """
    counts = pd.DataFrame({"g": grp, "y": y}).groupby(["g", "y"], sort=False).size().to_numpy()
    return float(max(1.0, np.median(counts))) if counts.size else 1.0


def _gap_slots(df: pd.DataFrame, order: np.ndarray, grp: np.ndarray, pad: int) -> np.ndarray:
    """NaN slots to insert *before* each row so an absence cannot be read across.

    The buffers count age in matches, not time: an observation ``age`` matches
    back carries ``exp(-alpha * age)`` whether that match was last month or nine
    years ago. So a club that leaves the league and returns reads its pre-exit
    form as though it were current -- Hull's May-2017 match was "1 match back"
    from August 2026, which is what drove their 2026-08-22 stat line to near zero.

    Materialising the absence fixes it at the source. Insert one slot per match
    the club would have played while it was away, and the rolling window simply
    cannot reach across the gap: the mask denominator sees NaN, not a value.

    Two properties worth keeping:

    * **The threshold is emergent, not set here.** How much of a gap it takes to
      empty a window is decided by that target's own ``L`` -- a two-season absence
      empties goals (L=35) but only dents SOT (L=85). That is the intended
      behaviour, and it is why this is not a staleness cutoff: a cutoff at two or
      three seasons would truncate SOT's legitimate continuous lookback.
    * **The gap to the season being predicted counts too.** Unplayed fixtures are
      appended as rows before the buffers are built, so a club whose last real
      season is nine years behind the fixture gets padded by the same rule, with
      no special case. That is the case that actually bites.

    Capped at ``pad + 1``: past a full window's worth of NaN the result is
    identical and the extra slots only cost memory.
    """
    n = len(order)
    gap = np.zeros(n, dtype=np.int64)
    if n < 2 or "season" not in df.columns:
        return gap
    y = df["season"].map(season_sort_key).to_numpy(dtype=float)[order]
    same = grp[1:] == grp[:-1]
    jump = np.zeros(n, dtype=float)
    jump[1:] = np.where(same, y[1:] - y[:-1], 0.0)
    missing = np.maximum(np.nan_to_num(jump) - 1.0, 0.0)
    return np.minimum(missing * _season_span(grp, y), pad + 1).astype(np.int64)


def _build_blocks(df: pd.DataFrame, group_cols: list[str], obs: np.ndarray, pad: int) -> _Blocks:
    order = np.lexsort(
        tuple(df[c].values for c in reversed(group_cols + SORT_KEYS))
    )
    grp = df.iloc[order][group_cols].astype(str).agg("\x1f".join, axis=1).values
    n = len(order)

    # Group boundaries in the sorted order.
    starts = np.flatnonzero(np.concatenate(([True], grp[1:] != grp[:-1])))
    group_ord = np.zeros(n, dtype=np.int64)
    group_ord[starts[1:]] = 1
    group_ord = np.cumsum(group_ord)
    pos_in_group = np.arange(n) - starts[group_ord]

    # Extra NaN slots standing in for seasons the group was absent, so a rolling
    # window cannot read across an absence. See `_gap_slots`.
    gap_before = np.cumsum(_gap_slots(df, order, grp, pad))
    total_gap = int(gap_before[-1]) if n else 0

    n_groups = len(starts)
    n_padded = n + n_groups * pad + total_gap
    # Each group occupies `pad` NaN rows followed by its own rows, plus a run of
    # NaN wherever that group skipped one or more seasons.
    row_pos_sorted = np.arange(n) + (group_ord + 1) * pad + gap_before

    n_buf = obs.shape[1]
    values = np.full((n_padded, n_buf), np.nan, dtype=np.float32)
    values[row_pos_sorted] = obs[order]
    mask = (~np.isnan(values)).astype(np.float32)

    # Scatter back to original row order.
    row_pos = np.empty(n, dtype=np.int64)
    row_pos[order] = row_pos_sorted
    n_prior = np.empty(n, dtype=np.int64)
    n_prior[order] = pos_in_group

    return _Blocks(values=values, mask=mask, row_pos=row_pos, n_prior=n_prior, pad=pad)


class BufferWindows:
    """Precomputed layout, reusable across every ``(L, alpha)`` in a grid.

    The expensive part -- sorting, grouping, padding -- happens once here. Each
    ``(L, alpha)`` then costs only the weighted slice-sum in ``priors()``.
    """

    def __init__(self, df: pd.DataFrame, L_max: int = 50):
        self.n_rows = len(df)
        self.L_max = int(L_max)
        self.buffer_names = list(BUFFER_NAMES)

        # Observation matrix: column j is the value that feeds buffer j.
        obs = np.empty((len(df), len(BUFFERS)), dtype=np.float32)
        for j, b in enumerate(BUFFERS):
            col = f"{b.stat}_{b.side}"
            obs[:, j] = df[col].to_numpy(dtype=np.float32, na_value=np.nan)
        self._obs = obs

        # All-venue buffers group by team; venue buffers by (team, venue).
        self.all_blocks = _build_blocks(df, ["team"], obs, self.L_max)
        self.venue_blocks = _build_blocks(df, ["team", "is_home"], obs, self.L_max)
        # League priors use the identical machinery, grouped by (league, venue):
        # "what does a home side in this league typically do lately". This is the
        # pooled model's handle on the fact that Serie A and the Bundesliga sit at
        # different levels, without normalising the raw team buffers.
        self.league_blocks = _build_blocks(df, ["league_key", "is_home"], obs, self.L_max)

        # Warm-up gate needs prior counts at each venue separately.
        self.n_prior_all = self.all_blocks.n_prior
        home = df["is_home"].to_numpy() == 1
        self.n_prior_home = np.where(home, self.venue_blocks.n_prior, 0)
        self.n_prior_away = np.where(~home, self.venue_blocks.n_prior, 0)
        # Prior matches at the *other* venue = total prior minus prior at this venue.
        other = self.n_prior_all - self.venue_blocks.n_prior
        self.n_prior_home = np.where(home, self.venue_blocks.n_prior, other)
        self.n_prior_away = np.where(home, other, self.venue_blocks.n_prior)

    @staticmethod
    def _rolling(blocks: _Blocks, window: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        """Weighted mean of the previous `window` observations, and its denominator.

        `den` is returned rather than discarded because it is the honest measure
        of how much usable history sits behind a row: the weighted count of
        *observed* values in the window, on the same scale as the weights. It is
        what the warm-up gate and the seed blend both key off -- see `warmup_mask`.
        """
        w = _weights(window, alpha)
        vals = np.nan_to_num(blocks.values, nan=0.0)
        num = np.zeros_like(vals)
        den = np.zeros_like(vals)
        for k in range(1, window + 1):
            wk = np.float32(w[k - 1])
            num[k:] += wk * vals[:-k]
            den[k:] += wk * blocks.mask[:-k]
        with np.errstate(invalid="ignore", divide="ignore"):
            out = num / den
        out[den == 0] = np.nan
        return out, den

    def _assemble(self, arr_all: np.ndarray, arr_ven: np.ndarray) -> dict[str, np.ndarray]:
        """Pick the all-venue or venue array per buffer, in row order."""
        return {b.name: (arr_ven if b.venue == "venue" else arr_all)[:, j]
                for j, b in enumerate(BUFFERS)}

    def priors(self, L: int, alpha: float) -> dict[str, np.ndarray]:
        """``{buffer_name: (n_rows,) float32}`` for this ``(L, alpha)``."""
        return self.priors_and_denominators(L, alpha)[0]

    def priors_and_denominators(
        self, L: int, alpha: float
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Buffers and their denominators in one pass.

        Callers that need both -- the cold-start path does -- should use this
        rather than calling `priors` and `denominators` separately, which would
        run the rolling sum twice.
        """
        L = int(L)
        if L > self.L_max:
            raise ValueError(f"L={L} exceeds L_max={self.L_max}; rebuild BufferWindows")
        L_V = math.ceil(0.5 * L)

        all_p, all_d = self._rolling(self.all_blocks, L, alpha)
        ven_p, ven_d = self._rolling(self.venue_blocks, L_V, alpha)
        rp_a, rp_v = self.all_blocks.row_pos, self.venue_blocks.row_pos
        return (self._assemble(all_p[rp_a], ven_p[rp_v]),
                self._assemble(all_d[rp_a], ven_d[rp_v]))

    def denominators(self, L: int, alpha: float) -> dict[str, np.ndarray]:
        """Weighted count of usable observations behind each row's buffer."""
        return self.priors_and_denominators(L, alpha)[1]

    def full_denominator(self, L: int, alpha: float) -> dict[str, float]:
        """What a completely full window weighs, per buffer.

        The saturation point of `den`, and the scale any pseudo-count must be
        expressed in. It differs sharply by target and by buffer family -- 7.16
        for an all-venue buffer at alpha=0.15, 16.94 at alpha=0.05 -- which is
        exactly why a fixed pseudo-count means something different on every
        target and must not be used.
        """
        L = int(L)
        L_V = math.ceil(0.5 * L)
        full_all = float(_weights(L, alpha).sum())
        full_ven = float(_weights(L_V, alpha).sum())
        return {b.name: (full_ven if b.venue == "venue" else full_all) for b in BUFFERS}

    def league_priors(self, L: int, alpha: float) -> dict[str, np.ndarray]:
        """``{'league_avg_<stat>_<side>': (n_rows,)}`` -- venue-aware league rates.

        League priors move slowly, so they use a wider window than the team
        buffers: ``L`` matches of team history is a handful of game weeks, but a
        league sees ~10 matches per round. We use the full ``L_max`` window here
        so the league rate is a stable level, not another noisy short-run signal.
        """
        lg = self._rolling(self.league_blocks, self.L_max, alpha)[0][self.league_blocks.row_pos]
        out: dict[str, np.ndarray] = {}
        for j, b in enumerate(BUFFERS):
            if b.venue != "all":
                continue  # one league prior per (stat, side), not per venue variant
            out[f"league_avg_{b.stat}_{b.side}"] = lg[:, j]
        return out


def warmup_mask(den: dict[str, np.ndarray], min_effective: float = 0.0) -> dict[str, np.ndarray]:
    """Per buffer, True where there is too little usable history to read it.

    Gates on the weighted count of *observed* values behind the row (`den`), not
    on how many rows precede it.

    The bug this replaces
    ---------------------
    This used to be ``(n_prior_home < 1) | (n_prior_away < 1)`` -- "has this team
    ever played at both venues". That asks the wrong question. It is satisfied by
    any history at all, however old and however unusable, so a club returning to a
    league after years away read False and its cold-start seed was never applied:
    the promoted profile was consulted inside a branch such a club could not
    reach. Hull carried 38 home and 38 away rows from 2014-2017, so `warm` was
    False in 2026 and their seed never fired.

    `n_prior` is now actively wrong for this, not merely weak: it counts rows
    including the NaN gap slots `_gap_slots` inserts, so a longer absence makes a
    club look like it has *more* history. `den` moves the right way -- gap slots
    are unobserved, contribute nothing, and drive it to zero.

    Returned per buffer because usability is per stat: a club can have readable
    goals history and no readable shots history at all, wherever the ESPN join
    failed for its era.
    """
    return {name: d <= min_effective for name, d in den.items()}


# --- Cold start -----------------------------------------------------------
#
# A side with no usable history has no priors to read, so they are seeded. Which
# numbers to seed with is a modelling choice, and it was being made badly.
#
# The old seed was the league's previous-season per-team average -- i.e. a
# promoted side was initialised as a perfectly average top-flight team. Measured
# over 150 promoted team-seasons, that is far too generous:
#
#     PROMOTED / ESTABLISHED, per league (ratio of per-match means)
#            goals_for  xg_for  shots_for  sot_for | goals_ag  shots_ag | points
#     Prem       0.676   0.741      0.806    0.753 |    1.314     1.153 |  0.619
#     Serie      0.747   0.780      0.822    0.795 |    1.334     1.182 |  0.637
#     Liga       0.775   0.804      0.890    0.830 |    1.195     1.091 |  0.749
#     Bund       0.771   0.800      0.893    0.827 |    1.185     1.098 |  0.760
#     Ligue      0.869   0.860      0.910    0.862 |    1.131     1.046 |  0.843
#
# A promoted Premier League side was being seeded ~32% high on goals scored and
# ~24% low on goals conceded, and the size of the error differs enough between
# leagues -- Ligue 1's promoted clubs are nearly twice as competitive as Serie
# A's -- that one pooled correction would also be wrong.
#
# So there are two seed tables, `season_seeds` for everyone and `promoted_seeds`
# for sides flagged `is_new_to_league`, and `Seeds.value` picks between them. The
# deficit barely fades within the first season (goals-for ratio 0.70 over games
# 1-3 against 0.78 over games 20-38), which is why the promoted profile is
# measured over the whole first season by default rather than an opening window:
# three to four times the sample, and no less accurate at game one.
#
# Both read strictly earlier seasons, so both stay causal, and the seed decays out
# of the feature on its own as real matches arrive -- `warmup_mask` stops firing
# once a side has played at both venues.

SCOPES = ("all", "home", "away")


def _scope_means(g: pd.DataFrame, stats: list[str]) -> dict[tuple[str, str, str], float]:
    """``{(stat, side, scope): mean}`` over one block of rows.

    Three scopes, because a venue buffer needs a venue-specific number. The old
    seeding filled *every* buffer of a stat -- ``_for``, ``_against`` and both
    venue variants -- from one league-wide ``_for`` mean. For the all-venue pair
    that is right by symmetry, since a league's goals for and goals against are
    the same goals counted twice. For the venue pair it is not: home sides score
    materially more, so ``goals_for_v`` on a home row wants the league's
    home-scoring rate and ``goals_against_v`` its away-scoring rate, and they
    differ by home advantage. `report.markets.LeagueBaseline` already holds home
    and away separately for exactly this reason.
    """
    home = (g["is_home"] == 1).to_numpy()
    out: dict[tuple[str, str, str], float] = {}
    for stat in stats:
        for side in SIDES:
            col = f"{stat}_{side}"
            if col not in g.columns:
                continue
            v = g[col].to_numpy(dtype=float)
            for scope, m in (("all", np.ones(len(v), bool)), ("home", home), ("away", ~home)):
                sel = v[m]
                sel = sel[np.isfinite(sel)]
                if sel.size:
                    out[(stat, side, scope)] = float(sel.mean())
    return out


def _ordered_seasons(g: pd.DataFrame) -> list[str]:
    """This league's seasons, oldest first.

    Ordered by the year each season *starts*, not lexically. Lexically ``'1920'``
    sorts before ``'2014/2015'``, so Ligue 1 -- whose curtailed 2019/20 carries
    that label -- had its 2014/15 cold start seeded from **2019/20**, five
    seasons into its own future. A small leak, on warm-up rows only, and entirely
    invisible.
    """
    return sorted(g["season"].unique(), key=season_sort_key)


def season_seeds(df: pd.DataFrame, stats: list[str]) -> dict[tuple, float]:
    """League averages from the *previous* season, per stat, side and venue.

    Returns ``{(league_key, season, stat, side, scope): value}``. Always reads
    strictly earlier seasons, so it stays causal.
    """
    seeds: dict[tuple, float] = {}
    for lk, g in df.groupby("league_key"):
        seasons = _ordered_seasons(g)
        for i, season in enumerate(seasons):
            if i == 0:
                continue
            prev = g[g["season"] == seasons[i - 1]]
            for (stat, side, scope), v in _scope_means(prev, stats).items():
                seeds[(lk, season, stat, side, scope)] = v
    return seeds


def promoted_seeds(df: pd.DataFrame, stats: list[str], lookback: int = PROMOTED_LOOKBACK,
                   window: int | None = PROMOTED_WINDOW,
                   min_seasons: int = MIN_PROMOTED_SEASONS) -> dict[tuple, float]:
    """What newly promoted sides in this league actually did, in earlier seasons.

    Same key as `season_seeds`, so the two are interchangeable at the point of
    use. Built from the ``lookback`` seasons *before* the target one, over rows
    whose team was flagged `is_new_to_league` -- so it is as walk-forward as the
    buffers it seeds.

    ``window`` restricts each promoted side to its first ``window`` matches;
    ``None`` uses its whole first season, which is the default because the
    deficit does not meaningfully fade within one.

    A league-season with fewer than ``min_seasons`` promoted team-seasons behind
    it contributes nothing, and `Seeds.value` falls back to the ordinary league
    average there. The Bundesliga is the binding case: 23 promoted team-seasons
    across twelve years, so a five-season lookback yields about nine.

    Returns an empty dict when the movement flag is absent, which makes this
    additive -- a caller without it gets exactly the old behaviour.
    """
    if "team_is_new_to_league" not in df.columns:
        return {}

    prom = df[df["team_is_new_to_league"] == 1]
    if prom.empty:
        return {}
    if window is not None:
        prom = prom.sort_values(["team", "season", "date"])
        prom = prom[prom.groupby(["team", "season"]).cumcount() < int(window)]

    seeds: dict[tuple, float] = {}
    for lk, g in df.groupby("league_key"):
        seasons = _ordered_seasons(g)
        sub = prom[prom["league_key"] == lk]
        for i, season in enumerate(seasons):
            if i == 0:
                continue
            hist = sub[sub["season"].isin(seasons[max(0, i - lookback):i])]
            if hist.empty or hist.groupby(["team", "season"]).ngroups < min_seasons:
                continue
            for (stat, side, scope), v in _scope_means(hist, stats).items():
                seeds[(lk, season, stat, side, scope)] = v
    return seeds


@dataclass(frozen=True)
class Seeds:
    """Both cold-start tables, and the rule for choosing between them."""

    league: dict[tuple, float] = field(default_factory=dict)
    promoted: dict[tuple, float] = field(default_factory=dict)
    # ``(team, league_key, season, stat, side) -> multiplier`` from `fpp.promoted`,
    # scaling the blanket promoted profile by what that specific club did in the
    # division below. Empty until the second-tier stats are ingested, and an
    # absent entry is 1.0, so the blanket ratio is what everyone keeps until then.
    #
    # The ``side`` is load-bearing. `fpp.promoted` measures produced and conceded
    # separately, and a caller that drops it applies an attacking ratio to a
    # defensive buffer -- which reverses its meaning, since scoring more in the
    # division below says nothing about conceding more in this one.
    multipliers: dict[tuple, float] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.league or self.promoted)

    def multiplier(self, team: str, league_key: str, season: str,
                   stat: str, side: str) -> float:
        """Per-club scaling for a promoted seed. 1.0 when nothing is known.

        Kept separate from `value` rather than folded into it because the two are
        looked up on different keys -- the seed on ``(league, season, ...)``, this
        on ``(team, ...)``. Multiplying afterwards keeps the seed lookup small
        enough to resolve once per league-season instead of once per club.
        """
        return self.multipliers.get((team, league_key, season, stat, side), 1.0)

    def value(self, league_key: str, season: str, stat: str, side: str,
              scope: str, is_promoted: bool) -> float:
        """The seed for one buffer on one row, promoted profile first.

        Falling back to the league average rather than to NaN is deliberate: a
        promoted side in a league-season with too little promoted history behind
        it is still better served by an average than by being dropped, which is
        what a NaN prior does downstream.
        """
        key = (league_key, season, stat, side, scope)
        if is_promoted:
            v = self.promoted.get(key)
            if v is not None:
                return v
        return self.league.get(key, np.nan)


def build_seeds(df: pd.DataFrame, stats: list[str], *,
                lower_stats: pd.DataFrame | str | None = "auto", **kwargs) -> Seeds:
    """Both seed tables from one pass over the history."""
    # "auto" reads whatever second-tier stats have been ingested, so the
    # multiplier reaches production without every caller having to thread it --
    # `build_feature_table` builds seeds itself and has no route to pass one.
    # Before the backfill has run this is an empty frame and every seed keeps the
    # blanket promoted ratio. Pass None to switch it off outright.
    if isinstance(lower_stats, str):
        from .promoted import lower_league_stats
        lower_stats = lower_league_stats()

    mult: dict[tuple, float] = {}
    if lower_stats is not None and len(lower_stats):
        from .promoted import multipliers as _multipliers
        mult = _multipliers(df, lower_stats)
    return Seeds(league=season_seeds(df, stats),
                 promoted=promoted_seeds(df, stats, **kwargs),
                 multipliers=mult)


__all__ = ["BufferWindows", "warmup_mask", "Seeds", "SCOPES",
           "season_seeds", "promoted_seeds", "build_seeds"]
