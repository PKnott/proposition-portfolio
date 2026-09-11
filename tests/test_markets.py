"""Pricing invariants: ladder placement, baseline lookup, and the cap change."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpp.metrics import over_prob, pmf_for
from fpp.report.markets import (
    LADDER_LINES,
    count_markets,
    dynamic_team_lines,
    league_baselines,
    team_pmf,
)
from fpp.report.markets import _season_sort_key
from fpp.spec import STAT_BY_KEY, TARGETS, target_spec


# --- Dynamic ladders ------------------------------------------------------


# Below this the window is shifted up rather than centred (see the clamp test),
# so the centring invariants only apply above it. The lowest line is
# ``floor(mu) + 0.5 - LADDER_LINES // 2``, which clears 0.5 exactly when
# ``floor(mu) >= LADDER_LINES // 2`` -- so the boundary moves every time the
# ladder is widened, and is derived rather than typed for that reason.
UNCLAMPED_MIN = LADDER_LINES // 2

CENTRED_CASES = [
    (15.06, 15.5), (9.52, 9.5), (6.34, 6.5), (4.90, 4.5), (3.67, 3.5),
    (12.0, 12.5), (13.0, 13.5),  # integer rates must not collide -- see below
]


def test_the_centring_cases_have_not_all_been_skipped():
    """Widening the ladder pushes low-rate cases below ``UNCLAMPED_MIN``.

    That is correct -- a clamped ladder has no centre to check -- but it means
    the centring test quietly loses cases every time ``LADDER_LINES`` grows.
    This fails loudly rather than letting it become vacuous.
    """
    assert sum(mu >= UNCLAMPED_MIN for mu, _ in CENTRED_CASES) >= 4


@pytest.mark.parametrize("mu, centre", CENTRED_CASES)
def test_ladder_is_centred_on_the_nearest_half_integer(mu, centre):
    if mu < UNCLAMPED_MIN:
        pytest.skip(f"mu={mu} clamps at LADDER_LINES={LADDER_LINES}; see the clamp test")
    half = LADDER_LINES // 2
    lines = dynamic_team_lines(mu)
    assert len(lines) == LADDER_LINES
    assert lines[half] == centre
    assert lines == tuple(centre + i for i in range(-half, half + 1))


def test_integer_rates_do_not_share_a_centre():
    """Banker's rounding put mu=12.0 and mu=13.0 on the same 12.5 centre."""
    assert dynamic_team_lines(12.0) != dynamic_team_lines(13.0)
    for i in range(UNCLAMPED_MIN, 20):
        assert dynamic_team_lines(float(i))[LADDER_LINES // 2] == i + 0.5


def test_centre_is_within_half_a_goal_of_the_rate():
    for mu in np.linspace(UNCLAMPED_MIN, 25, 300):
        centre = dynamic_team_lines(float(mu))[LADDER_LINES // 2]
        assert abs(centre - mu) <= 0.5 + 1e-9


@pytest.mark.parametrize("mu", [0.0, 0.4, 0.79, 2.27, 2.94])
def test_low_rates_shift_the_window_rather_than_clipping_it(mu):
    """A low-rate side still gets a full ladder -- shifted up, not truncated."""
    lines = dynamic_team_lines(mu)
    assert len(lines) == LADDER_LINES
    assert min(lines) >= 0.5
    assert lines == tuple(0.5 + i for i in range(LADDER_LINES))


def test_ladder_lines_are_always_half_integers():
    for mu in np.linspace(0, 25, 200):
        for line in dynamic_team_lines(float(mu)):
            assert (line * 2) % 2 == 1, f"{line} is not a half-integer"


# --- The cap change -------------------------------------------------------


@pytest.mark.parametrize("target", [t for t in TARGETS if STAT_BY_KEY[t].dist == "nbinom"])
def test_wide_cap_leaves_canonical_prices_unchanged(target):
    """`team_pmf` moved from `team_cap` to `total_cap`; prices must not move.

    The caps are scoring buckets, and any cap above the top line folds the same
    tail mass. This is what licenses pricing the dynamic ladders off the wider
    pmf, so it is pinned rather than assumed.
    """
    s = target_spec(target)
    for mu in (1.0, 5.0, 12.0, 20.0):
        narrow = pmf_for(s.dist, mu, s.team_cap, 8.0)
        wide = pmf_for(s.dist, mu, s.total_cap, 8.0)
        for line in s.team_lines:
            assert over_prob(narrow, line) == pytest.approx(over_prob(wide, line), abs=1e-12)


@pytest.mark.parametrize("target", [t for t in TARGETS if STAT_BY_KEY[t].dist == "nbinom"])
def test_every_dynamic_line_clears_the_pmf_cap(target):
    """The reason for the cap change: `sot` tops out at 12 and a ladder reaches 10.5."""
    s = target_spec(target)
    for mu in np.linspace(0.5, s.team_cap, 40):
        lines = dynamic_team_lines(float(mu))
        pmf = team_pmf(target, float(mu), 8.0)
        assert max(lines) < len(pmf) - 1, (
            f"{target}: line {max(lines)} at mu={mu:.1f} reaches the folded tail bucket"
        )


def test_count_markets_gives_each_side_its_own_ladder():
    mk = count_markets("shots", 15.06, 9.52, dispersion=8.0)
    assert mk["home_lines"] == dynamic_team_lines(15.06)
    assert mk["away_lines"] == dynamic_team_lines(9.52)
    assert mk["home_lines"] != mk["away_lines"]
    for line in mk["home_lines"]:
        assert 0.0 <= mk[f"home_over_{line}"] <= 1.0
    # Probabilities fall as the line rises.
    probs = [mk[f"home_over_{ln}"] for ln in mk["home_lines"]]
    assert probs == sorted(probs, reverse=True)


# --- Season ordering ------------------------------------------------------


def test_season_sort_key_orders_mixed_label_formats():
    """The cleaned table mixes '2014/2015', '1920' and '2526'."""
    seasons = ["2526", "1920", "2014/2015", "2024/2025", "2020/2021"]
    assert sorted(seasons, key=_season_sort_key) == [
        "2014/2015", "1920", "2020/2021", "2024/2025", "2526",
    ]


# --- Baselines ------------------------------------------------------------


def _synthetic_matches(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        for is_home in (1, 0):
            mult = 1.3 if is_home else 1.0  # a deliberate venue effect to detect
            rows.append({
                "fixture_id": i, "season": "2024/2025", "league_key": "Prem",
                "is_home": is_home,
                "goals_for": rng.poisson(1.5 * mult),
                "shots_for": rng.poisson(12 * mult),
                "sot_for": rng.poisson(4 * mult),
                "corners_for": rng.poisson(5 * mult),
            })
    return pd.DataFrame(rows)


def test_over_rate_matches_a_brute_force_count():
    df = _synthetic_matches()
    base = league_baselines(df, seasons=1)["Prem"]

    for target in TARGETS:
        for scope, sel in (("home", df["is_home"] == 1), ("away", df["is_home"] == 0)):
            vals = df.loc[sel, f"{target}_for"].to_numpy(dtype=float)
            for line in (0.5, 2.5, 5.5, 9.5):
                assert base.over_rate(target, scope, line) == pytest.approx(
                    float((vals > line).mean()), abs=1e-12
                )


def test_home_and_away_baselines_are_genuinely_separate():
    """The bug this replaced: pooling made the colour report venue, not edge."""
    base = league_baselines(_synthetic_matches(), seasons=1)["Prem"]
    for target in TARGETS:
        h = base.over_rate(target, "home", 4.5)
        a = base.over_rate(target, "away", 4.5)
        assert h > a, f"{target}: expected a home advantage, got home={h:.3f} away={a:.3f}"


def test_match_baseline_sits_above_either_side():
    base = league_baselines(_synthetic_matches(), seasons=1)["Prem"]
    for target in TARGETS:
        assert base.over_rate(target, "match", 6.5) >= base.over_rate(target, "home", 6.5)


def test_unknown_target_or_scope_returns_nan_rather_than_raising():
    """A missing baseline must colour neutral, not abort a whole workbook."""
    base = league_baselines(_synthetic_matches(), seasons=1)["Prem"]
    assert np.isnan(base.over_rate("goals", "nowhere", 2.5))
    assert np.isnan(base.over_rate("unknown", "home", 2.5))
