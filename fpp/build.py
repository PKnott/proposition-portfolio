"""Assemble a model-ready feature matrix from the buffer engine.

``FeatureTable`` holds one float32 matrix plus the metadata needed to slice it by
season and re-pair rows into fixtures. All four target families share a single
table -- the buffers depend on ``(L, alpha)``, not on which stat is being
predicted -- which is the saving that makes a four-family search affordable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import PRIOR_SEED_KAPPA
from .priors import BufferWindows, Seeds, build_seeds, warmup_mask
from .spec import (
    ALL_FEATURE_NAMES,
    BUFFERS,
    CONTEXT_FEATURES,
    FEATURES,
    STATS,
    TARGETS,
)

SEED_STATS = [s.key for s in STATS]


@dataclass
class FeatureTable:
    """A built feature matrix plus everything needed to evaluate against it."""

    X: pd.DataFrame  # (n_rows, n_features); `league` is categorical
    meta: pd.DataFrame  # fixture_id, date, season, league, league_key, team, opponent, is_home
    y: dict[str, np.ndarray] = field(default_factory=dict)  # target key -> values
    opp_idx: np.ndarray | None = None  # partner row of each row
    L: int = 0
    alpha: float = 0.0

    def __len__(self) -> int:
        return len(self.X)

    def subset(self, columns: list[str]) -> "FeatureTable":
        return FeatureTable(
            X=self.X[list(columns)], meta=self.meta, y=self.y,
            opp_idx=self.opp_idx, L=self.L, alpha=self.alpha,
        )

    def mask(self, m: np.ndarray) -> "FeatureTable":
        """Row subset. ``opp_idx`` is dropped: partner indices no longer resolve."""
        return FeatureTable(
            X=self.X.loc[m].reset_index(drop=True),
            meta=self.meta.loc[m].reset_index(drop=True),
            y={k: v[m] for k, v in self.y.items()},
            opp_idx=None, L=self.L, alpha=self.alpha,
        )


def partner_index(df: pd.DataFrame) -> np.ndarray:
    """For each row, the positional index of the other team's row in the fixture.

    Precomputed once and reused for every ``(L, alpha)``: attaching opponent
    priors then costs a single fancy-index gather, replacing the old
    ``df.merge`` self-join inside the inner loop.
    """
    pos = np.arange(len(df))
    order = np.lexsort((df["is_home"].to_numpy(), df["fixture_id"].to_numpy()))
    if len(order) % 2 != 0:
        raise ValueError("odd number of rows -- every fixture must have exactly two")
    paired = order.reshape(-1, 2)
    opp = np.empty(len(df), dtype=np.int64)
    opp[paired[:, 0]] = paired[:, 1]
    opp[paired[:, 1]] = paired[:, 0]
    # Guard: partners must share a fixture and differ in venue.
    fid = df["fixture_id"].to_numpy()
    assert (fid[opp] == fid).all(), "partner_index paired rows across fixtures"
    assert (df["is_home"].to_numpy()[opp] != df["is_home"].to_numpy()).all(), "partner has same venue"
    return opp[pos]


def build_feature_table(
    df: pd.DataFrame,
    bw: BufferWindows,
    L: int,
    alpha: float,
    *,
    features: list[str] | None = None,
    seeds: Seeds | None = None,
    drop_warmup: bool = True,
    kappa: float | None = None,
    target_shrink: float = 0.0,
) -> FeatureTable:
    """Build the feature matrix for one ``(L, alpha)``.

    Cold start: sides new to a league have no usable history, so their priors are
    seeded. ~150 team-seasons across the dataset need this; without it every
    promoted side is dropped for its warm-up window, every season.

    A side flagged `is_new_to_league` is seeded from what promoted sides in that
    league *actually did*, not from the league average -- the average is far too
    kind, by ~32% on goals scored in the Premier League. Everyone else, and any
    league-season without enough promoted history behind it, still gets the
    league average. See the block comment above `priors.season_seeds`.

    ``kappa`` overrides `config.PRIOR_SEED_KAPPA` for one build, which is what the
    sweep that sets that constant needs; None uses the configured value.

    ``target_shrink`` pulls every team buffer toward its league average by a fixed
    fraction, regardless of how full that buffer is -- the separate lever
    `config.TARGET_SHRINK` documents. Zero by default and zero in config; it
    exists to be swept.
    """
    wanted = list(features) if features else list(ALL_FEATURE_NAMES)
    unknown = set(wanted) - set(ALL_FEATURE_NAMES)
    if unknown:
        raise ValueError(f"unknown feature(s): {sorted(unknown)}")

    team_p, den = bw.priors_and_denominators(L, alpha)
    opp = partner_index(df)

    # --- cold-start seeding -------------------------------------------------
    seeds = seeds if seeds is not None else build_seeds(df, SEED_STATS)
    warm = warmup_mask(den)
    kappa = float(PRIOR_SEED_KAPPA if kappa is None else kappa)
    if seeds and (kappa > 0 or any(w.any() for w in warm.values())):
        lk = df["league_key"].to_numpy()
        ssn = df["season"].to_numpy()
        home = df["is_home"].to_numpy() == 1
        promoted = (df["team_is_new_to_league"].to_numpy() == 1
                    if "team_is_new_to_league" in df.columns else np.zeros(len(df), bool))

        # A seed is a pure function of (league, season, stat, side, scope,
        # promoted). Only the first two and the last vary by row, and they take a
        # few hundred distinct combinations against tens of thousands of rows, so
        # resolve each combination once and index into it. The previous form
        # called `Seeds.value` per row, which was affordable only because it ran
        # on warm-up rows alone; the blend below touches every row.
        codes, uniq = pd.factorize(pd.MultiIndex.from_arrays([lk, ssn, promoted]))

        def seed_for(stat: str, side: str, scope: str) -> np.ndarray:
            table = np.array(
                [seeds.value(u[0], u[1], stat, side, scope, bool(u[2])) for u in uniq],
                dtype=np.float32,
            )
            return table[codes]

        if seeds.multipliers:
            teams = df["team"].to_numpy()
            _mult_cache: dict[tuple[str, str], np.ndarray] = {}

            def mult_for(stat: str, side: str) -> np.ndarray:
                arr = _mult_cache.get((stat, side))
                if arr is None:
                    arr = np.array([seeds.multiplier(t, l, sn, stat, side)
                                    for t, l, sn in zip(teams, lk, ssn)], dtype=np.float32)
                    _mult_cache[stat, side] = arr
                return arr

        full = bw.full_denominator(L, alpha)
        for b in BUFFERS:
            vals = team_p[b.name]
            d = den[b.name]
            need = warm[b.name]
            if not need.any() and kappa <= 0:
                continue
            # An all-venue buffer wants the league's pooled rate; a venue buffer
            # wants the rate for the venue *this row* is at, which is why the
            # scope is per row rather than per buffer.
            if b.venue == "all":
                seed_vals = seed_for(b.stat, b.side, "all")
            else:
                seed_vals = np.where(home,
                                     seed_for(b.stat, b.side, "home"),
                                     seed_for(b.stat, b.side, "away"))

            # Scale the blanket promoted profile by what this particular club did
            # in the division below. Applied as a separate vectorised pass rather
            # than inside `seed_for`, because the seed resolves once per
            # (league, season, promoted) -- a few hundred combinations -- while
            # this is per club, and folding them together would multiply that
            # lookup by the number of teams for a factor that is 1.0 on all but
            # the promoted rows.
            #
            # Keyed on the buffer's side as well as its stat. It was not, and the
            # produced ratio landed on the conceded buffers unchanged: a club that
            # outscored the promoted cohort in the division below was seeded to
            # concede proportionally more in this one. See `fpp.promoted`.
            if seeds.multipliers:
                seed_vals = seed_vals * mult_for(b.stat, b.side)

            # No usable history at all: take the seed outright. This is the branch
            # a returning club could not previously reach.
            vals[need] = seed_vals[need]

            # Some usable history: shrink toward the seed with a pseudo-count,
            # which is what "backfill the log with the seed" amounts to without
            # inventing rows. It decays out on its own as real matches arrive,
            # instead of switching hard from seed to a two-match average.
            #
            # `k` scales with the buffer's saturated denominator and is never a
            # fixed constant: `den` saturates at 7.16 for shots (alpha=0.15) and
            # 16.94 for goals (alpha=0.05), so one constant would weigh the seed
            # 41% on one target and 23% on another. See `full_denominator`.
            if kappa > 0:
                k = kappa * full[b.name]
                blended = (d * np.nan_to_num(vals) + k * seed_vals) / (d + k)
                vals[~need] = blended[~need]

    # --- uniform shrinkage toward the league average ------------------------
    # Applied after seeding and to every row, full buffer or not -- that is the
    # whole point of it being a different lever from `kappa`, which only ever
    # touches rows with little usable history. `league_priors` is the venue-aware
    # league rate the buffer engine already computes.
    if target_shrink:
        lg = bw.league_priors(L, alpha)
        for b in BUFFERS:
            avg = lg.get(f"league_avg_{b.stat}_{b.side}")
            if avg is None:
                continue
            vals = team_p[b.name]
            ok = np.isfinite(vals) & np.isfinite(avg)
            vals[ok] = (1.0 - target_shrink) * vals[ok] + target_shrink * avg[ok]

    # --- assemble -----------------------------------------------------------
    cols: dict[str, np.ndarray] = {}
    for f in FEATURES:
        if f.name not in wanted:
            continue
        v = team_p[f.buffer.name]
        cols[f.name] = v if f.perspective == "team" else v[opp]

    # League priors are no longer candidate features -- see the note in spec.py.
    # `BufferWindows.league_priors()` still works if they are ever reinstated.

    for name in CONTEXT_FEATURES:
        if name not in wanted:
            continue
        if name == "league":
            continue  # categorical, added below
        cols[name] = df[name].to_numpy(dtype=np.float32, na_value=np.nan)

    X = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in cols.items()})
    if "league" in wanted:
        X["league"] = pd.Categorical(df["league_key"].to_numpy())
    X = X[[c for c in wanted if c in X.columns]]

    meta = df[["fixture_id", "date", "season", "league", "league_key", "team", "opponent", "is_home"]].reset_index(drop=True)
    y = {t: df[f"{t}_for"].to_numpy(dtype=np.float64, na_value=np.nan) for t in TARGETS}

    ft = FeatureTable(X=X, meta=meta, y=y, opp_idx=opp, L=int(L), alpha=float(alpha))

    if drop_warmup:
        prior_cols = [c for c in X.columns if c in {f.name for f in FEATURES}]
        keep = ~X[prior_cols].isna().any(axis=1).to_numpy() if prior_cols else np.ones(len(X), bool)
        # Keep fixtures whole: a fixture survives only if both of its rows do.
        fid = meta["fixture_id"].to_numpy()
        bad = set(fid[~keep])
        keep &= ~np.isin(fid, list(bad))
        if keep.sum() < len(keep):
            ft = ft.mask(keep)
            ft.opp_idx = partner_index(ft.meta)
    return ft


def scorable_mask(ft: FeatureTable, target: str) -> np.ndarray:
    """Rows with a usable target value -- shots/SOT/corners can be missing."""
    return ~np.isnan(ft.y[target])


__all__ = ["FeatureTable", "partner_index", "build_feature_table", "scorable_mask"]
