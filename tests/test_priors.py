"""Regression guard for the vectorised walk-forward buffer engine.

The engine is ~1000x faster than the row-at-a-time version it replaces, which is
what makes the four-target-family search tractable. That speedup is only worth
anything if the semantics are identical, so this test re-implements the priors
the slow, obvious way and demands agreement.

It also pins the leakage property directly: a prior may only ever see strictly
earlier matches.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

from fpp.config import season_sort_key
from fpp.priors import BufferWindows
from fpp.spec import BUFFERS

STAT_SIDES = sorted({(b.stat, b.side) for b in BUFFERS})


def _naive_priors(df: pd.DataFrame, L: int, alpha: float) -> dict[str, np.ndarray]:
    """Row-at-a-time reference implementation. Deliberately slow and obvious."""
    L_V = math.ceil(0.5 * L)
    gen: dict = defaultdict(list)
    ven: dict = defaultdict(list)
    out = {b.name: np.full(len(df), np.nan) for b in BUFFERS}

    def wmean(values: list[float], window: int) -> float:
        recent = values[-window:]
        observed = [(i, v) for i, v in enumerate(recent) if not np.isnan(v)]
        if not observed:
            return np.nan
        k = len(recent)
        num = den = 0.0
        for i, v in observed:
            w = math.exp(-alpha * (k - 1 - i))
            num += w * v
            den += w
        return num / den

    for i, row in enumerate(df.to_dict("records")):
        team, home = row["team"], int(row["is_home"])
        for b in BUFFERS:
            if b.venue == "all":
                out[b.name][i] = wmean(gen[(team, b.stat, b.side)], L)
            else:
                out[b.name][i] = wmean(ven[(team, home, b.stat, b.side)], L_V)
        # Record AFTER reading -- one append per (stat, side), not per buffer.
        for stat, side in STAT_SIDES:
            v = float(row[f"{stat}_{side}"])
            gen[(team, stat, side)].append(v)
            ven[(team, home, stat, side)].append(v)
    return out


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    from fpp.clean import load_clean_table

    df = load_clean_table()
    teams = sorted(df["team"].unique())[:12]
    return (
        df[df["team"].isin(teams)]
        .sort_values(["date", "fixture_id", "is_home"])
        .reset_index(drop=True)
    )


def _gap_free(df: pd.DataFrame, L: int) -> tuple[np.ndarray, np.ndarray]:
    """``(all_venue, venue)`` masks of rows whose window contains no season gap.

    `_naive_priors` reads a team's history as one uninterrupted stream, which is
    the semantics the engine had before `_gap_slots`. It is still the right
    reference wherever no absence falls inside the window, and that is most of the
    table -- so the equality guard is scoped to those rows rather than weakened.
    Gap behaviour is pinned separately by `test_gap_empties_the_window`.

    Computed per buffer family, not once: a venue buffer looks back ``L_V`` of the
    team's matches *at that venue*, which reaches a different distance into the
    table than ``L`` of its matches overall.
    """
    y = df["season"].map(season_sort_key).to_numpy(dtype=float)

    def mask(group_cols: list[str], window: int) -> np.ndarray:
        ok = np.ones(len(df), bool)
        for _, g in df.groupby(group_cols, sort=False):
            idx = g.index.to_numpy()
            for j in np.flatnonzero(np.diff(y[idx]) > 1) + 1:   # first row after a gap
                ok[idx[j:j + window + 1]] = False
        return ok

    return mask(["team"], L), mask(["team", "is_home"], math.ceil(0.5 * L))


@pytest.mark.parametrize(("L", "alpha"), [(5, 0.0), (10, 0.05), (20, 0.2), (40, 0.1)])
def test_matches_naive_implementation(sample: pd.DataFrame, L: int, alpha: float) -> None:
    fast = BufferWindows(sample, L_max=50).priors(L, alpha)
    slow = _naive_priors(sample, L, alpha)
    keep_all, keep_ven = _gap_free(sample, L)
    assert min(keep_all.mean(), keep_ven.mean()) > 0.9, \
        "fixture became mostly gap-affected; scope is too broad to be a guard"

    for b in BUFFERS:
        keep = keep_ven if b.venue == "venue" else keep_all
        a = np.asarray(fast[b.name], dtype=float)[keep]
        c = np.asarray(slow[b.name], dtype=float)[keep]
        assert np.array_equal(np.isnan(a), np.isnan(c)), f"NaN pattern differs for {b.name}"
        m = ~np.isnan(a)
        if m.any():
            # float32 accumulation; 1e-4 is comfortably tight at these magnitudes.
            assert np.max(np.abs(a[m] - c[m])) < 1e-4, f"values differ for {b.name}"


def test_priors_never_see_the_current_match(sample: pd.DataFrame) -> None:
    """A team's first-ever row has no history, so every prior must be NaN."""
    bw = BufferWindows(sample, L_max=50)
    p = bw.priors(L=10, alpha=0.0)
    first_rows = sample.groupby("team", observed=True).head(1).index.to_numpy()
    for b in BUFFERS:
        assert np.isnan(p[b.name][first_rows]).all(), f"{b.name} leaked into a team's first match"


def test_alpha_zero_is_a_plain_mean(sample: pd.DataFrame) -> None:
    """With alpha=0 the prior is the unweighted mean of available observations."""
    bw = BufferWindows(sample, L_max=50)
    p = bw.priors(L=5, alpha=0.0)["goals_for"]
    keep, _ = _gap_free(sample, 5)   # goals_for is an all-venue buffer

    hist: dict = defaultdict(list)
    for i, row in enumerate(sample.to_dict("records")):
        prior = hist[row["team"]][-5:]
        expected = float(np.mean(prior)) if prior else np.nan
        got = p[i]
        if keep[i]:
            if np.isnan(expected):
                assert np.isnan(got)
            else:
                assert abs(got - expected) < 1e-4
        hist[row["team"]].append(float(row["goals_for"]))


def test_L_above_L_max_is_rejected(sample: pd.DataFrame) -> None:
    bw = BufferWindows(sample, L_max=10)
    with pytest.raises(ValueError, match="exceeds L_max"):
        bw.priors(L=20, alpha=0.0)


def _synthetic(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """Minimal team-match table: (team, season, goals_for) per row, in date order."""
    from fpp.spec import STATS

    recs = []
    for i, (team, season, gf) in enumerate(rows):
        rec = {
            "fixture_id": i, "date": pd.Timestamp("2000-01-01") + pd.Timedelta(days=i),
            "season": season, "league": "Premier League", "league_key": "Prem",
            "team": team, "opponent": "Other", "is_home": i % 2,
        }
        for st in STATS:
            rec[f"{st.key}_for"] = gf
            rec[f"{st.key}_against"] = gf
        recs.append(rec)
    return pd.DataFrame(recs)


def test_gap_empties_the_window() -> None:
    """A season gap must stop a buffer reading across it.

    The engine counts age in matches, so without `_gap_slots` a club returning
    after years away reads its pre-exit form as though it were last week -- the
    defect that drove Hull's 2026-08-22 corners to 0.23. `Steady` and `Gapper`
    have identical match histories; only the season labels differ.
    """
    early = [("Steady", "2019/2020", 1.0)] * 6 + [("Gapper", "2019/2020", 1.0)] * 6
    late_steady = [("Steady", "2020/2021", 5.0)] * 6      # contiguous
    late_gapper = [("Gapper", "2023/2024", 5.0)] * 6      # three seasons away
    df = _synthetic(early + late_steady + late_gapper).sort_values(
        ["date", "fixture_id", "is_home"]).reset_index(drop=True)

    p = BufferWindows(df, L_max=50).priors(L=4, alpha=0.0)["goals_for"]
    steady = df.index[(df.team == "Steady") & (df.season == "2020/2021")].to_numpy()
    gapper = df.index[(df.team == "Gapper") & (df.season == "2023/2024")].to_numpy()

    # Contiguous seasons still read straight through, unchanged.
    assert abs(p[steady[0]] - 1.0) < 1e-6

    # Across a gap there is nothing usable to read.
    assert np.isnan(p[gapper[0]]), "prior read across a three-season absence"

    # Once the window refills from real post-return matches it is uncontaminated
    # by the pre-gap era -- 5.0 throughout, never a blend with the old 1.0.
    assert abs(p[gapper[4]] - 5.0) < 1e-6
    assert abs(p[steady[4]] - 5.0) < 1e-6


def test_gap_does_not_inflate_n_prior_based_warmup() -> None:
    """`den` must fall across a gap even though the row count rises.

    This is why the warm-up gate moved off `n_prior`: gap slots are rows, so a
    longer absence makes a returning club look like it has *more* history, while
    `den` correctly reports none.
    """
    rows = [("Gapper", "2019/2020", 1.0)] * 6 + [("Gapper", "2023/2024", 5.0)] * 1
    df = _synthetic(rows).sort_values(["date", "fixture_id", "is_home"]).reset_index(drop=True)

    bw = BufferWindows(df, L_max=50)
    den = bw.denominators(L=4, alpha=0.0)["goals_for"]
    from fpp.priors import warmup_mask

    back = df.index[df.season == "2023/2024"].to_numpy()[0]
    assert bw.n_prior_all[back] == 6, "row count should still see the old matches"
    assert den[back] == 0, "but no usable observation should be readable across the gap"
    assert warmup_mask(bw.denominators(L=4, alpha=0.0))["goals_for"][back]
