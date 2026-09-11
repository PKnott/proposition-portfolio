"""Guards on the cold start and on the schedule-context features.

The cold start decides what a side with no history looks like to the model, and
it is the only place in the pipeline where a number is invented rather than
measured. Three things had gone wrong there, all silently:

1. every buffer of a stat was filled from one league-wide ``_for`` mean, so a
   venue buffer got the pooled rate instead of its own venue's;
2. the "previous season" was found by lexical sort, so Ligue 1 -- whose curtailed
   2019/20 is labelled ``'1920'`` -- seeded its 2014/15 from its own future;
3. a promoted side was seeded as a perfectly average top-flight team.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpp import clean
from fpp.priors import Seeds, build_seeds, promoted_seeds, season_seeds

STATS = ["goals", "shots"]


def _rows(league, season, team, is_home, goals_for, goals_against, new=0, shots_for=10.0):
    return {"league_key": league, "season": season, "team": team, "is_home": is_home,
            "goals_for": goals_for, "goals_against": goals_against,
            "shots_for": shots_for, "shots_against": shots_for,
            "team_is_new_to_league": new,
            "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(goals_for))}


def _frame(records) -> pd.DataFrame:
    return pd.DataFrame(records)


# --- Venue scopes -----------------------------------------------------------


def test_home_and_away_seeds_are_different_numbers():
    """The bug: every buffer of a stat got one pooled `_for` mean.

    Home sides score more, so a venue buffer wants its own venue's rate --
    `goals_for_v` on a home row the league's home-scoring rate, `goals_against_v`
    its away-scoring rate. Pooling them made the seed report the league instead
    of the venue, which is the same mistake `report.markets.LeagueBaseline`
    documents having made and fixed.
    """
    prev = [_rows("Prem", "2014/2015", "A", 1, 2.0, 1.0),
            _rows("Prem", "2014/2015", "B", 0, 1.0, 2.0),
            _rows("Prem", "2014/2015", "C", 1, 2.0, 1.0),
            _rows("Prem", "2014/2015", "D", 0, 1.0, 2.0)]
    now = [_rows("Prem", "2015/2016", "A", 1, 0.0, 0.0)]
    seeds = season_seeds(_frame(prev + now), STATS)

    assert seeds[("Prem", "2015/2016", "goals", "for", "home")] == 2.0
    assert seeds[("Prem", "2015/2016", "goals", "for", "away")] == 1.0
    assert seeds[("Prem", "2015/2016", "goals", "for", "all")] == 1.5
    # And the against side is the mirror, not a copy.
    assert seeds[("Prem", "2015/2016", "goals", "against", "home")] == 1.0
    assert seeds[("Prem", "2015/2016", "goals", "against", "away")] == 2.0


# --- Causality --------------------------------------------------------------


def test_seeds_read_the_previous_season_and_never_a_later_one():
    df = _frame([_rows("Prem", "2014/2015", "A", 1, 1.0, 1.0),
                 _rows("Prem", "2015/2016", "A", 1, 5.0, 5.0),
                 _rows("Prem", "2016/2017", "A", 1, 9.0, 9.0)])
    seeds = season_seeds(df, STATS)
    assert seeds[("Prem", "2015/2016", "goals", "for", "all")] == 1.0
    assert seeds[("Prem", "2016/2017", "goals", "for", "all")] == 5.0
    # The first season has nothing before it, so it gets nothing.
    assert ("Prem", "2014/2015", "goals", "for", "all") not in seeds


def test_the_curtailed_ligue_1_season_does_not_seed_the_past():
    """`'1920'` sorts before `'2014/2015'` lexically, and is five seasons later.

    Under the old ordering Ligue 1's 2014/15 cold start was seeded from 2019/20 --
    real future leakage, on warm-up rows, entirely invisible.
    """
    df = _frame([_rows("Ligue", "2014/2015", "A", 1, 1.0, 1.0),
                 _rows("Ligue", "2015/2016", "A", 1, 2.0, 2.0),
                 _rows("Ligue", "1920", "A", 1, 9.0, 9.0)])       # = 2019/20
    seeds = season_seeds(df, STATS)
    assert ("Ligue", "2014/2015", "goals", "for", "all") not in seeds, "2014/15 is the first season"
    assert seeds[("Ligue", "2015/2016", "goals", "for", "all")] == 1.0
    assert seeds[("Ligue", "1920", "goals", "for", "all")] == 2.0   # seeded from 2015/16


# --- Promoted profile -------------------------------------------------------


def _promoted_history(n_seasons=6, promoted_goals=0.5, established_goals=2.0):
    """`n_seasons` seasons, each with one promoted club and three established."""
    recs = []
    for y in range(2010, 2010 + n_seasons):
        season = f"{y}/{y + 1}"
        recs.append(_rows("Prem", season, f"up{y}", 1, promoted_goals, 3.0, new=1))
        recs.append(_rows("Prem", season, f"up{y}", 0, promoted_goals, 3.0, new=1))
        for k in range(3):
            recs.append(_rows("Prem", season, f"est{k}", 1, established_goals, 1.0))
            recs.append(_rows("Prem", season, f"est{k}", 0, established_goals, 1.0))
    y = 2010 + n_seasons
    recs.append(_rows("Prem", f"{y}/{y + 1}", "newcomer", 1, 0.0, 0.0, new=1))
    return _frame(recs), f"{y}/{y + 1}"


def test_a_promoted_side_is_seeded_from_promoted_history_not_the_league_average():
    df, target = _promoted_history()
    seeds = build_seeds(df, STATS, lookback=10, min_seasons=3)
    promoted = seeds.value("Prem", target, "goals", "for", "all", is_promoted=True)
    league = seeds.value("Prem", target, "goals", "for", "all", is_promoted=False)
    assert promoted == pytest.approx(0.5)
    assert league == pytest.approx(1.625)      # 1 promoted + 3 established, both venues
    assert promoted < league, "the whole point: the league average is too kind"


def test_an_established_side_still_gets_the_league_average():
    df, target = _promoted_history()
    seeds = build_seeds(df, STATS, lookback=10, min_seasons=3)
    assert seeds.value("Prem", target, "goals", "for", "all", is_promoted=False) == pytest.approx(1.625)


def test_too_little_promoted_history_falls_back_rather_than_guessing():
    """The Bundesliga binds here -- 23 promoted team-seasons across twelve years."""
    df, target = _promoted_history(n_seasons=2)
    seeds = build_seeds(df, STATS, lookback=10, min_seasons=6)
    assert seeds.promoted == {}
    league = seeds.value("Prem", target, "goals", "for", "all", is_promoted=False)
    assert seeds.value("Prem", target, "goals", "for", "all", is_promoted=True) == league


def test_the_promoted_profile_is_walk_forward():
    """A season's profile must not contain that season, or any later one."""
    recs = []
    for y, g in ((2010, 1.0), (2011, 1.0), (2012, 1.0), (2013, 9.0)):
        season = f"{y}/{y + 1}"
        for k in range(3):
            recs.append(_rows("Prem", season, f"up{y}_{k}", k % 2, g, 1.0, new=1))
            recs.append(_rows("Prem", season, f"est{k}", k % 2, 2.0, 1.0))
    seeds = promoted_seeds(_frame(recs), STATS, lookback=10, min_seasons=3)
    # 2013/14 is the outlier season; the profile used *for* it must not see it.
    assert seeds[("Prem", "2013/2014", "goals", "for", "all")] == pytest.approx(1.0)


def test_the_lookback_window_forgets_old_seasons():
    recs = []
    for y, g in ((2010, 9.0), (2011, 1.0), (2012, 1.0), (2013, 1.0)):
        season = f"{y}/{y + 1}"
        for k in range(3):
            recs.append(_rows("Prem", season, f"up{y}_{k}", k % 2, g, 1.0, new=1))
    wide = promoted_seeds(_frame(recs), STATS, lookback=10, min_seasons=3)
    narrow = promoted_seeds(_frame(recs), STATS, lookback=2, min_seasons=3)
    assert wide[("Prem", "2013/2014", "goals", "for", "all")] > narrow[("Prem", "2013/2014", "goals", "for", "all")]
    assert narrow[("Prem", "2013/2014", "goals", "for", "all")] == pytest.approx(1.0)


def test_the_window_restricts_to_a_promoted_sides_opening_matches():
    # Four seasons, so the target has three promoted team-seasons behind it and
    # clears `min_seasons`. Each promoted club scores 0 in its first two matches
    # and 8 in the next two, so the window is visible in the mean.
    recs = []
    for y in (2010, 2011, 2012, 2013):
        season = f"{y}/{y + 1}"
        for k in range(4):
            r = _rows("Prem", season, f"up{y}", k % 2, 0.0 if k < 2 else 8.0, 1.0, new=1)
            r["date"] = pd.Timestamp("2020-01-01") + pd.Timedelta(days=k)
            recs.append(r)
    whole = promoted_seeds(_frame(recs), STATS, lookback=10, window=None, min_seasons=3)
    opener = promoted_seeds(_frame(recs), STATS, lookback=10, window=2, min_seasons=3)
    assert whole[("Prem", "2013/2014", "goals", "for", "all")] == pytest.approx(4.0)
    assert opener[("Prem", "2013/2014", "goals", "for", "all")] == pytest.approx(0.0)


def test_no_movement_flag_means_no_promoted_seeds_and_the_old_behaviour():
    """Additive: a caller without the flag gets exactly what it got before."""
    df, _ = _promoted_history()
    assert promoted_seeds(df.drop(columns=["team_is_new_to_league"]), STATS) == {}


def test_an_empty_seeds_object_is_falsey_and_yields_nan():
    s = Seeds()
    assert not s
    assert np.isnan(s.value("Prem", "2015/2016", "goals", "for", "all", True))


# --- On the real table ------------------------------------------------------


def test_promoted_seeds_are_materially_below_the_league_average(clean_table):
    """Measured, not assumed: the Prem promoted profile is ~0.64 of the average."""
    seeds = build_seeds(clean_table, ["goals", "shots", "sot", "corners"])
    assert seeds.promoted, "no promoted seeds built from the real table"
    for lk in ("Prem", "Liga", "Bund", "Serie", "Ligue"):
        league = seeds.value(lk, "2024/2025", "goals", "for", "all", False)
        promoted = seeds.value(lk, "2024/2025", "goals", "for", "all", True)
        assert 0.5 < promoted / league < 0.95, f"{lk}: ratio {promoted / league:.3f}"
        # And conceding more, not less.
        assert (seeds.value(lk, "2024/2025", "goals", "against", "all", True)
                > seeds.value(lk, "2024/2025", "goals", "against", "all", False))


def test_home_and_away_seeds_differ_on_the_real_table(clean_table):
    seeds = build_seeds(clean_table, ["goals"])
    home = seeds.value("Prem", "2024/2025", "goals", "for", "home", False)
    away = seeds.value("Prem", "2024/2025", "goals", "for", "away", False)
    assert home > away, "home advantage should make these differ"
    assert home - away > 0.1


# --- Schedule context -------------------------------------------------------


def test_congestion_counts_matches_strictly_before_the_fixture(tmp_path, monkeypatch):
    """Including the fixture itself would make the feature unusable on an
    unplayed row, which is exactly where it has to work."""
    monkeypatch.setattr(clean, "HISTORIC_ALL_COMP", tmp_path / "none.csv")
    monkeypatch.setattr(clean, "all_comp_fixtures", lambda _label: tmp_path / "none2.csv")
    dates = pd.to_datetime(["2020-01-01", "2020-01-05", "2020-01-09", "2020-01-30"])
    # Two rows per fixture, as the real table has: A at home against a different
    # opponent each time, so only A accumulates a schedule.
    df = pd.DataFrame({
        "fixture_id": [0, 0, 1, 1, 2, 2, 3, 3],
        "is_home": [1, 0] * 4,
        "team": ["A", "B0", "A", "B1", "A", "B2", "A", "B3"],
        "date": dates.repeat(2),
    })
    out = clean.attach_schedule_context(df)
    a = out[out["team"] == "A"].sort_values("date")
    # 1 Jan: nothing before. 5 Jan: one. 9 Jan: two. 30 Jan: none within 14 days.
    assert a["team_matches_14d"].tolist() == [0, 1, 2, 0]
    assert a["team_rest_days"].tolist() == [pytest.approx(np.nan, nan_ok=True), 4, 4, 21]
    # Each opponent is playing its only match, so its own count is zero -- and A
    # sees that, not a copy of its own.
    assert a["opp_matches_14d"].tolist() == [0, 0, 0, 0]


def test_congestion_is_present_and_varies_on_the_real_table(clean_table):
    for col in ("team_matches_14d", "opp_matches_14d"):
        assert col in clean_table.columns
        v = clean_table[col]
        assert v.notna().all(), f"{col} has gaps"
        assert v.nunique() > 3, f"{col} is nearly constant"
    assert clean_table["team_matches_14d"].mean() == pytest.approx(1.96, abs=0.1)


def test_congestion_is_symmetric_across_a_fixture(clean_table):
    """`opp_matches_14d` must be the partner row's own count, not a copy."""
    df = clean_table
    home = df[df["is_home"] == 1].set_index("fixture_id")
    away = df[df["is_home"] == 0].set_index("fixture_id")
    common = home.index.intersection(away.index)
    assert (home.loc[common, "team_matches_14d"].to_numpy()
            == away.loc[common, "opp_matches_14d"].to_numpy()).all()


def test_upcoming_fixtures_get_real_schedule_context_not_nan(clean_table):
    """The bug this catches was invisible in training and total in production.

    `append_fixtures_as_rows` used to *create* the schedule columns as NaN on the
    unplayed rows rather than computing them, so `team_rest_days` was missing on
    every row the model was actually asked to predict -- present and informative
    throughout the training data, absent at the moment of use. XGBoost routes
    NaN down a missing branch rather than failing, so nothing anywhere said so.

    Both quantities are well defined for a scheduled match, which is the point:
    the gap since the last match and the count of matches strictly before it are
    known as soon as the fixture has a date.
    """
    from fpp import predict

    fixtures = predict.load_upcoming_fixtures()
    if fixtures.empty:
        pytest.skip("no upcoming fixtures on disk")

    combined = predict.append_fixtures_as_rows(clean_table, fixtures)
    upcoming = combined[combined["is_scored"] == 0]
    assert len(upcoming) == 2 * len(fixtures)

    for col in ("team_matches_14d", "opp_matches_14d"):
        assert upcoming[col].notna().all(), f"{col} is NaN on rows being predicted"
        assert upcoming[col].nunique() > 1, f"{col} is constant on upcoming fixtures"
    # Rest days can legitimately be missing for a club with no schedule history
    # pulled yet -- a promoted side ESPN has no id row for -- but not for most.
    assert upcoming["team_rest_days"].notna().mean() > 0.8


def test_congestion_measures_something_rest_days_does_not(clean_table):
    """The justification for adding it: within team-season, more matches in the
    prior fortnight means fewer shots, and raw rest days shows no such thing."""
    df = clean_table.dropna(subset=["shots_for", "team_rest_days"]).copy()
    g = df.groupby(["team", "season"])["shots_for"]
    df["demeaned"] = df["shots_for"] - g.transform("mean")

    by_congestion = df.groupby("team_matches_14d")["demeaned"].mean()
    busy = by_congestion.loc[[3.0, 4.0]].mean()
    fresh = by_congestion.loc[[0.0, 1.0]].mean()
    assert busy < fresh, "congestion should reduce shots once quality is controlled for"

    # Raw rest days, same treatment, shows essentially nothing.
    assert abs(df["team_rest_days"].corr(df["demeaned"])) < 0.02
