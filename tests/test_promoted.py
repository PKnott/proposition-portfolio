"""Guards for the per-club promoted multiplier.

Built on synthetic divisions rather than real ones, deliberately: the second-tier
stats this consumes need an ~89-minute ESPN backfill, and the logic has to be
right *before* that runs, not after. Every property below is a statement about
the arithmetic, so a fixture is the honest place to check it.

The one thing checked against real data is the xG proxy, because its whole claim
is a measured R2 on the top five divisions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fpp.promoted as pr


def _top(rows: list[tuple[str, str, str, int]]) -> pd.DataFrame:
    """(team, league_key, season, is_new) -> a minimal top-flight table."""
    recs = []
    for team, lk, season, new in rows:
        recs.append({"team": team, "league_key": lk, "season": season,
                     "team_is_new_to_league": new})
    return pd.DataFrame(recs)


def _lower(rows: list[tuple[str, str, str, float, int]]) -> pd.DataFrame:
    """(team, lower_key, season, shots_per_match, n_matches[, conceded]) -> matches.

    Every produced stat is tied to `shots` so one number drives a club's attacking
    profile and the assertions stay about the mechanism rather than about
    arithmetic on five independent columns. `conceded` drives the defensive side
    the same way, and defaults to the cohort's 10.0 so a club that only varies its
    attack has an unremarkable defence -- which is what makes "the two ratios move
    independently" testable at all.
    """
    recs = []
    for row in rows:
        team, lk, season, shots, n = row[:5]
        conceded = row[5] if len(row) > 5 else 10.0
        for _ in range(n):
            recs.append({"team": team, "league_key": lk, "season": season,
                         "shots_for": shots, "sot_for": shots / 3.0,
                         "goals_for": shots / 10.0, "corners_for": shots / 2.0,
                         "shots_against": conceded, "sot_against": conceded / 3.0,
                         "goals_against": conceded / 10.0,
                         "corners_against": conceded / 2.0})
    return pd.DataFrame(recs)


# A cohort promoted in earlier seasons sets the benchmark; two clubs promoted for
# 2024/2025 are then measured against it.
BENCH_ROWS = [
    ("OldA", "Prem", "2022/2023", 1), ("OldB", "Prem", "2023/2024", 1),
    ("Strong", "Prem", "2024/2025", 1), ("Weak", "Prem", "2024/2025", 1),
    ("Established", "Prem", "2024/2025", 0),
]
BENCH_LOWER = [
    ("OldA", "Prem2", "2021/2022", 10.0, 46),
    ("OldB", "Prem2", "2022/2023", 10.0, 46),
    ("Strong", "Prem2", "2023/2024", 12.0, 46),     # 20% above the cohort
    ("Weak", "Prem2", "2023/2024", 8.0, 46),        # 20% below
]


def _mult():
    return pr.multipliers(_top(BENCH_ROWS), _lower(BENCH_LOWER))


def test_multiplier_differs_by_club():
    """The point of the whole module: two promoted clubs stop being identical."""
    m = _mult()
    strong = m[("Strong", "Prem", "2024/2025", "shots", "for")]
    weak = m[("Weak", "Prem", "2024/2025", "shots", "for")]
    assert strong > 1.0 > weak, f"expected strong>1>weak, got {strong} / {weak}"
    assert strong == pytest.approx(1.2, abs=0.02)
    assert weak == pytest.approx(0.8, abs=0.02)


def test_established_clubs_get_no_multiplier():
    """Only sides flagged new to the league are scaled; everyone else is untouched."""
    m = _mult()
    assert not any(k[0] == "Established" for k in m)


def test_absent_lower_data_leaves_every_seed_alone():
    """Safe to wire in before the backfill: no data means no multipliers at all."""
    assert pr.multipliers(_top(BENCH_ROWS), None) == {}
    assert pr.multipliers(_top(BENCH_ROWS), pd.DataFrame()) == {}


def test_benchmark_reads_only_strictly_earlier_seasons():
    """A club must not help set the benchmark it is judged against.

    `priors._ordered_seasons` documents a leak of exactly this shape that stood
    unnoticed -- a cold start seeded from five seasons into its own future -- so
    this is a known failure mode rather than a hypothetical one.
    """
    m = _mult()
    strong = m[("Strong", "Prem", "2024/2025", "shots", "for")]

    # Make the *other* 2024/2025 promotion enormous. If the benchmark leaked its
    # own cohort, Strong's multiplier would fall; it must not move at all.
    louder = list(BENCH_LOWER)
    louder[3] = ("Weak", "Prem2", "2023/2024", 40.0, 46)
    m2 = pr.multipliers(_top(BENCH_ROWS), _lower(louder))
    assert m2[("Strong", "Prem", "2024/2025", "shots", "for")] == pytest.approx(strong)


def test_no_earlier_cohort_means_no_multiplier():
    """The first promoted season in the data has nothing to be measured against."""
    rows = [("First", "Prem", "2022/2023", 1)]
    low = [("First", "Prem2", "2021/2022", 12.0, 46)]
    assert pr.multipliers(_top(rows), _lower(low)) == {}


def test_clip_binds_on_an_extreme_season():
    m = pr.multipliers(_top(BENCH_ROWS),
                       _lower(BENCH_LOWER[:2] + [("Strong", "Prem2", "2023/2024", 100.0, 46)]))
    v = m[("Strong", "Prem", "2024/2025", "shots", "for")]
    assert v == pytest.approx(pr.CLIP_HI), f"expected the clip at {pr.CLIP_HI}, got {v}"


def test_short_sample_shrinks_further_toward_one():
    """A ratio backed by six games must move a seed less than one backed by 46."""
    long_season = pr.multipliers(_top(BENCH_ROWS), _lower(BENCH_LOWER))
    short = list(BENCH_LOWER)
    short[2] = ("Strong", "Prem2", "2023/2024", 12.0, 6)
    short_season = pr.multipliers(_top(BENCH_ROWS), _lower(short))

    a = long_season[("Strong", "Prem", "2024/2025", "shots", "for")]
    b = short_season[("Strong", "Prem", "2024/2025", "shots", "for")]
    assert 1.0 < b < a, f"six games should shrink harder toward 1 than 46: {b} vs {a}"


def test_seeds_multiplier_defaults_to_one():
    from fpp.priors import Seeds
    s = Seeds()
    assert s.multiplier("Anyone", "Prem", "2024/2025", "shots", "for") == 1.0
    assert s.multiplier("Anyone", "Prem", "2024/2025", "shots", "against") == 1.0


# --- attack and defence are measured separately -----------------------------
#
# The regression these exist for: `multipliers` returned one ratio per stat, built
# from produced stats alone, and `build.build_feature_table` applied it to the
# conceded buffers too. A club that outscored the promoted cohort in the division
# below was seeded to concede proportionally more in the top flight. Nothing here
# looked at the `against` side at all, which is why it stood.

# Same cohort as BENCH_ROWS, with the defensive dimension varied independently of
# the attacking one. Strong outscores the cohort and also keeps more out; Leaky
# matches the cohort's attack exactly and concedes far more than it.
SIDED_LOWER = [
    ("OldA", "Prem2", "2021/2022", 10.0, 46, 10.0),
    ("OldB", "Prem2", "2022/2023", 10.0, 46, 10.0),
    ("Strong", "Prem2", "2023/2024", 12.0, 46, 8.0),   # +20% attack, -20% conceded
    ("Weak", "Prem2", "2023/2024", 10.0, 46, 12.0),    # cohort attack, +20% conceded
]


def _sided():
    return pr.multipliers(_top(BENCH_ROWS), _lower(SIDED_LOWER))


def test_for_and_against_are_not_the_same_number():
    """The bug in one line: a strong attack must not scale the conceded seed up."""
    m = _sided()
    for stat in pr.MEASURED:
        f = m[("Strong", "Prem", "2024/2025", stat, "for")]
        a = m[("Strong", "Prem", "2024/2025", stat, "against")]
        assert f > 1.0 > a, f"{stat}: expected for>1>against, got {f} / {a}"


def test_conceding_less_lowers_the_conceded_seed():
    """Direction. The seed being scaled is a conceded rate, so no inversion."""
    m = _sided()
    assert m[("Strong", "Prem", "2024/2025", "goals", "against")] == pytest.approx(0.8, abs=0.02)
    assert m[("Weak", "Prem", "2024/2025", "goals", "against")] == pytest.approx(1.2, abs=0.02)


def test_the_two_sides_move_independently():
    """Changing only what a club conceded must leave its attacking ratio alone."""
    m = _sided()
    leakier = list(SIDED_LOWER)
    leakier[2] = ("Strong", "Prem2", "2023/2024", 12.0, 46, 13.0)
    m2 = pr.multipliers(_top(BENCH_ROWS), _lower(leakier))

    key_for = ("Strong", "Prem", "2024/2025", "shots", "for")
    key_ag = ("Strong", "Prem", "2024/2025", "shots", "against")
    assert m2[key_for] == pytest.approx(m[key_for]), "the produced ratio must not move"
    assert m2[key_ag] > m[key_ag], "conceding more must raise the conceded ratio"


def test_against_benchmark_reads_only_strictly_earlier_seasons():
    """The causal guarantee holds on the conceded side too, not just the produced one."""
    m = _sided()
    strong = m[("Strong", "Prem", "2024/2025", "shots", "against")]

    leakier = list(SIDED_LOWER)
    leakier[3] = ("Weak", "Prem2", "2023/2024", 10.0, 46, 40.0)
    m2 = pr.multipliers(_top(BENCH_ROWS), _lower(leakier))
    assert m2[("Strong", "Prem", "2024/2025", "shots", "against")] == pytest.approx(strong)


def test_estimated_stats_are_sided_too():
    """xG and npxG are proxied, but they are proxied per side like everything else."""
    # A top table carrying real xG, so `fit_xg_proxy` has something to fit on.
    # Enough matches per team-season to clear its `matches >= 25` gate, and
    # enough team-seasons to clear the 50-row minimum on the fit itself.
    rng = np.random.default_rng(0)
    teams, seasons, per_season = 20, 5, 30
    recs = []
    for t in range(teams):
        for s in range(seasons):
            for _ in range(per_season):
                sf, sa = rng.uniform(5, 20), rng.uniform(5, 20)
                recs.append({
                    "team": f"T{t}", "league_key": "Prem",
                    "season": f"{2014 + s}/{2015 + s}",
                    "shots_for": sf, "sot_for": sf / 3, "goals_for": sf / 10,
                    "xg_for": 0.1 * sf, "npxg_for": 0.09 * sf,
                    "shots_against": sa, "sot_against": sa / 3, "goals_against": sa / 10,
                    "xg_against": 0.1 * sa, "npxg_against": 0.09 * sa,
                })
    fits = pr.fit_xg_proxy(pd.DataFrame(recs))
    assert set(fits) == {"xg", "npxg"}

    profile = pr._per_match(_lower(SIDED_LOWER).assign(lower="Prem2"), pr.MEASURED)
    out = pr.apply_xg_proxy(profile, fits)
    for stat in pr.ESTIMATED:
        assert f"{stat}_for" in out.columns and f"{stat}_against" in out.columns
    strong = out[out["team"] == "Strong"].iloc[0]
    assert strong["xg_for"] > strong["xg_against"], \
        "Strong out-shot what it faced, so its proxied xG must exceed its proxied xGA"


# --- the one claim that needs real data -------------------------------------


def test_xg_proxy_recovers_season_aggregate_xg(clean_table):
    """The proxy exists because xG is unavailable below the top flight.

    Its licence is the measured season-aggregate fit -- 0.875 with shots, SoT and
    goals together. Checked on the real table, because a synthetic one would only
    reproduce whatever relationship it was built with.
    """
    top = clean_table
    fits = pr.fit_xg_proxy(top)
    assert set(fits) == {"xg", "npxg"}

    agg = pr._per_match(top, ("shots", "sot", "goals", "xg"))
    agg = agg[agg["matches"] >= 25].dropna(subset=["shots_for", "sot_for", "goals_for", "xg_for"])
    X = np.column_stack([np.ones(len(agg)), agg["shots_for"], agg["sot_for"], agg["goals_for"]])
    pred = X @ fits["xg"]
    y = agg["xg_for"].to_numpy()
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    assert r2 > 0.85, f"season-aggregate xG proxy fell to R2={r2:.3f}"

    # and it must beat goals alone, which is the cheaper thing it replaces
    Xg = np.column_stack([np.ones(len(agg)), agg["goals_for"]])
    bg, *_ = np.linalg.lstsq(Xg, y, rcond=None)
    r2_goals = 1 - ((y - Xg @ bg) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    assert r2 > r2_goals, "the three-term proxy must beat goals alone"


# --- the plumbing from ingest to multiplier ---------------------------------


def _espn_rows() -> pd.DataFrame:
    """An `espn_match_stats.csv` slice: two divisions, one of them second tier."""
    return pd.DataFrame([
        # second tier, real
        {"league_key": "Prem2", "date": "2025-10-25", "home_team": "Strong", "away_team": "Weak",
         "home_shots": 18, "away_shots": 6, "home_sot": 7, "away_sot": 2,
         "home_corners": 9, "away_corners": 3, "home_goals": 3, "away_goals": 0},
        # second tier, in the second half of the season -> same season label
        {"league_key": "Prem2", "date": "2026-02-14", "home_team": "Weak", "away_team": "Strong",
         "home_shots": 8, "away_shots": 14, "home_sot": 3, "away_sot": 6,
         "home_corners": 4, "away_corners": 7, "home_goals": 1, "away_goals": 2},
        # second tier, the all-zero sentinel -- must be rejected
        {"league_key": "Prem2", "date": "2025-11-22", "home_team": "Strong", "away_team": "Weak",
         "home_shots": 0, "away_shots": 0, "home_sot": 0, "away_sot": 0,
         "home_corners": 0, "away_corners": 0, "home_goals": 0, "away_goals": 0},
        # top tier -- must not be picked up
        {"league_key": "Prem", "date": "2025-10-25", "home_team": "Established", "away_team": "Other",
         "home_shots": 15, "away_shots": 9, "home_sot": 5, "away_sot": 3,
         "home_corners": 6, "away_corners": 4, "home_goals": 2, "away_goals": 1},
    ])


def test_lower_league_stats_selects_and_reshapes():
    out = pr.lower_league_stats(_espn_rows())
    assert set(out["league_key"]) == {"Prem2"}, "top-tier rows must not be picked up"
    assert "Established" not in set(out["team"])
    # two real matches x two teams; the sentinel row is gone
    assert len(out) == 4, f"expected 4 team-rows, got {len(out)}"
    assert {"goals_for", "shots_for", "sot_for", "corners_for"} <= set(out.columns)
    assert {"goals_against", "shots_against", "sot_against",
            "corners_against"} <= set(out.columns)

    # and the conceded side is the opponent's column on the same match, not a
    # copy of the produced one
    strong_home = out[(out["team"] == "Strong") & (out["date"] == "2025-10-25")].iloc[0]
    assert float(strong_home["shots_for"]) == 18
    assert float(strong_home["shots_against"]) == 6


def test_lower_league_stats_labels_the_season_from_the_date():
    """October 2025 and February 2026 are one season, not two."""
    out = pr.lower_league_stats(_espn_rows())
    assert set(out["season"]) == {"2025/2026"}


def test_lower_league_stats_is_empty_before_the_backfill():
    assert pr.lower_league_stats(pd.DataFrame()).empty


def test_multipliers_run_off_the_reshaped_frame():
    """End to end: ESPN-shaped rows in, a per-club multiplier out."""
    top = _top([("OldA", "Prem", "2023/2024", 1),
                ("Strong", "Prem", "2026/2027", 1)])
    espn = pd.concat([
        _espn_rows(),
        pd.DataFrame([{"league_key": "Prem2", "date": "2022-10-25",
                       "home_team": "OldA", "away_team": "Weak",
                       "home_shots": 10, "away_shots": 10, "home_sot": 4, "away_sot": 4,
                       "home_corners": 5, "away_corners": 5, "home_goals": 1, "away_goals": 1}]),
    ], ignore_index=True)
    m = pr.multipliers(top, pr.lower_league_stats(espn))
    assert ("Strong", "Prem", "2026/2027", "shots", "for") in m
    assert m[("Strong", "Prem", "2026/2027", "shots", "for")] > 1.0, "Strong out-shot the cohort"


# --- the two ingest rules the second tier depends on ------------------------
#
# Both were found by asking why the promoted multiplier was doing so little:
# the second tier had 28% of its cached matches surviving ingest, and what did
# survive included two seasons of malformed Spanish and French blocks.


def _rec(hs, as_, hsot, asot, hc, ac):
    return {"home_shots": hs, "away_shots": as_, "home_sot": hsot,
            "away_sot": asot, "home_corners": hc, "away_corners": ac}


def test_degenerate_block_is_rejected():
    """Every shot on target on both sides *and* no corners -- the 2019/20 layout."""
    from fpp.ingest.scoreboard import _is_degenerate, _is_sentinel
    bad = _rec(1, 2, 1, 2, 0, 0)
    assert _is_degenerate(bad)
    assert not _is_sentinel(bad), "not all-zero, so the older check cannot catch it"


def test_degenerate_needs_both_conditions():
    """Either condition alone is a real match and must survive."""
    from fpp.ingest.scoreboard import _is_degenerate
    # a genuine goalless, cornerless match in which both sides had off-target shots
    assert not _is_degenerate(_rec(8, 5, 2, 1, 0, 0))
    # one side converted everything it hit, but corners were won
    assert not _is_degenerate(_rec(3, 11, 3, 4, 5, 6))
    # every shot on target for one side only
    assert not _is_degenerate(_rec(2, 9, 2, 3, 0, 0))


def test_second_tier_keeps_half_mapped_rows():
    """An unmapped opponent must not delete the mapped club's match.

    The second tier is read one team at a time, so a row with one nameless side
    is still a usable observation for the other. Requiring both sides -- which
    the top flight does need, for the Understat join -- was discarding the
    promotion seasons this module exists to measure.
    """
    import pandas as pd
    rows = pd.DataFrame([
        {"league_key": "Prem2", "date": "2025-10-25", "home_team": "Promoted",
         "away_team": None, "home_shots": 14, "away_shots": 7, "home_sot": 5,
         "away_sot": 2, "home_corners": 6, "away_corners": 3,
         "home_goals": 2, "away_goals": 1},
    ])
    out = pr.lower_league_stats(rows)
    assert list(out["team"]) == ["Promoted"], "the named side must survive alone"
    assert float(out.iloc[0]["shots_for"]) == 14
    # Only the opponent's *name* is missing. Its counts were parsed, so the
    # conceded side of a half-mapped row is a real observation, not a gap.
    assert float(out.iloc[0]["shots_against"]) == 7
