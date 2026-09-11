"""Guards on promotion detection and on fixture-to-history coverage.

Both cover failures that were **silent**: `is_new_to_league` returned zero for
every row in the dataset, and an unmapped fixture team was scored from the league
average with nothing printed. Neither showed up as an error, a NaN, or a failing
test -- which is why the tests here assert on *counts and identities*, not just
on shapes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpp import predict
from fpp.clean import attach_movement_flags
from fpp.config import season_sort_key


def _table(rows) -> pd.DataFrame:
    """``(league_key, season, team)`` triples -> a minimal team-match frame."""
    df = pd.DataFrame(rows, columns=["league_key", "season", "team"])
    df["league"] = df["league_key"].map(
        {"Prem": "Premier League", "Liga": "La Liga", "Bund": "Bundesliga"})
    df["opponent"] = "someone"
    return df


# --- Season ordering --------------------------------------------------------


def test_season_sort_key_handles_every_label_shape_in_the_table():
    """The table really does mix all three, from three different sources."""
    assert season_sort_key("2014/2015") == 2014
    assert season_sort_key("1920") == 2019       # Ligue 1's curtailed 2019/20
    assert season_sort_key("2526") == 2025       # the current-season pull's label
    labels = ["2526", "1920", "2014/2015", "2024/2025"]
    assert sorted(labels, key=season_sort_key) == ["2014/2015", "1920", "2024/2025", "2526"]
    # Lexical order gets this wrong, which is the whole reason the key exists.
    assert sorted(labels) != ["2014/2015", "1920", "2024/2025", "2526"]


# --- is_new_to_league -------------------------------------------------------


def test_a_club_promoted_back_after_years_away_is_new_again():
    """The bug: a `groupby(team).shift(1)` compared 2025 against 2016.

    A relegated club is simply absent for the seasons it spends down, so the
    previous *row* is its own last season in the same league -- same league, so
    "not new". Sunderland, present 2014-2016 and back in 2025, is the real case.
    """
    out = attach_movement_flags(_table([
        ("Prem", "2014/2015", "Sunderland"),
        ("Prem", "2015/2016", "Sunderland"),
        ("Prem", "2016/2017", "Sunderland"),
        ("Prem", "2024/2025", "Sunderland"),   # away for eight seasons
        ("Prem", "2014/2015", "Arsenal"), ("Prem", "2015/2016", "Arsenal"),
        ("Prem", "2016/2017", "Arsenal"), ("Prem", "2023/2024", "Arsenal"),
        ("Prem", "2024/2025", "Arsenal"),
    ]))
    flags = dict(zip(out["season"], out["team_is_new_to_league"]))
    assert out.loc[(out["team"] == "Sunderland") & (out["season"] == "2024/2025"),
                   "team_is_new_to_league"].iloc[0] == 1
    assert flags is not None


def test_an_unbroken_run_in_one_league_is_never_new():
    out = attach_movement_flags(_table([
        ("Prem", s, "Arsenal") for s in ("2014/2015", "2015/2016", "2016/2017")
    ]))
    assert out["team_is_new_to_league"].tolist() == [0, 0, 0]


def test_the_leagues_first_season_flags_nobody():
    """Nobody is new to a league we have no previous season of."""
    out = attach_movement_flags(_table([
        ("Prem", "2014/2015", "Arsenal"), ("Prem", "2014/2015", "Chelsea"),
        ("Prem", "2015/2016", "Arsenal"), ("Prem", "2015/2016", "Watford"),
    ]))
    first = out[out["season"] == "2014/2015"]
    assert first["team_is_new_to_league"].sum() == 0
    later = out[(out["season"] == "2015/2016") & (out["team"] == "Watford")]
    assert later["team_is_new_to_league"].iloc[0] == 1


def test_a_gap_year_in_the_labels_does_not_flag_everyone():
    """`'1920'` and `'2526'` must order as seasons, not as strings.

    Ligue 1's 2019/20 is labelled `'1920'`. Ordered lexically it lands before
    2014/2015, which would make every club in 2020/21 look promoted.
    """
    rows = [("Prem", s, t) for s in ("2018/2019", "1920", "2020/2021")
            for t in ("Arsenal", "Chelsea")]
    out = attach_movement_flags(_table(rows))
    assert out["team_is_new_to_league"].sum() == 0


def test_the_flag_is_not_degenerate_on_the_real_table(clean_table):
    """It was identically zero across all 43,178 rows, and nothing noticed.

    Pinned as a real-world fact rather than a shape: the Premier League promotes
    exactly three clubs every season, so the flag has to find exactly three in
    every season bar the first one in the data.
    """
    df = clean_table.drop(columns=[c for c in clean_table.columns
                                   if "new_to_league" in c or "rank_delta" in c])
    out = attach_movement_flags(df)
    assert out["team_is_new_to_league"].sum() > 0, "the flag is constant zero again"

    ts = out[["league_key", "season", "team", "team_is_new_to_league"]].drop_duplicates()
    ts = ts.assign(year=ts["season"].map(season_sort_key))
    prem = ts[ts["league_key"] == "Prem"].groupby("year")["team_is_new_to_league"].sum()
    assert prem.iloc[0] == 0, "the first season in the data cannot flag anyone"
    assert (prem.iloc[1:] == 3).all(), f"the Prem promotes three a season, got {prem.to_dict()}"

    # Every league, every season after its first: at least one promotion.
    per = ts.groupby(["league_key", "year"])["team_is_new_to_league"].sum()
    for lk in ts["league_key"].unique():
        counts = per[lk].sort_index()
        assert (counts.iloc[1:] > 0).all(), f"{lk} has a season with no promoted club"


def test_the_opponent_flag_mirrors_the_team_flag(clean_table):
    df = clean_table.drop(columns=[c for c in clean_table.columns
                                   if "new_to_league" in c or "rank_delta" in c])
    out = attach_movement_flags(df)
    assert out["opp_is_new_to_league"].sum() == out["team_is_new_to_league"].sum()


def test_rank_delta_is_still_zero_while_every_league_shares_a_rank(clean_table):
    """Documented behaviour, not a bug -- it starts firing when lower tiers land."""
    df = clean_table.drop(columns=[c for c in clean_table.columns
                                   if "new_to_league" in c or "rank_delta" in c])
    out = attach_movement_flags(df)
    assert set(out["team_rank_delta"].unique()) == {0}


# --- Fixture coverage -------------------------------------------------------


def _history(teams, season="2024/2025"):
    return pd.DataFrame({"team": list(teams), "season": season})


def test_a_team_with_no_history_is_reported_as_missing(capsys):
    fx = pd.DataFrame({"home_team": ["Arsenal"], "away_team": ["Coventry City"]})
    rep = predict.check_fixture_coverage(fx, _history(["Arsenal", "Chelsea"]))
    assert rep["missing"] == ["Coventry City"]
    assert rep["stale"] == []
    out = capsys.readouterr().out
    assert "Coventry City" in out
    # A club with no history is necessarily flagged `is_new_to_league` once its
    # fixture rows are appended, so it takes the promoted branch of `Seeds.value`,
    # not the league average. The message said "league average" for a long time
    # and this test pinned it there -- on the Bundesliga that understated the seed
    # by 1.618 goals against 1.906.
    assert "promoted profile" in out
    assert "league average" not in out
    # what is actually lost is the per-club scaling from the division below
    assert "per-club" in out


def test_a_team_whose_history_is_years_old_is_reported_as_stale(capsys):
    hist = pd.concat([
        _history(["Arsenal"], "2024/2025"),
        _history(["Arsenal", "Hull"], "2016/2017"),
        _history(["Arsenal"], "2023/2024"),
    ])
    fx = pd.DataFrame({"home_team": ["Arsenal"], "away_team": ["Hull"]})
    rep = predict.check_fixture_coverage(fx, hist, max_age=2)
    assert rep["missing"] == []
    assert rep["stale"] == [("Hull", "2016/2017")]
    assert "2016/2017" in capsys.readouterr().out


def test_missing_and_stale_are_different_answers():
    """Stale is the quieter failure: real numbers from a squad that is gone."""
    hist = pd.concat([_history(["A", "B"], "2024/2025"), _history(["C"], "2015/2016")])
    fx = pd.DataFrame({"home_team": ["A", "C"], "away_team": ["B", "D"]})
    rep = predict.check_fixture_coverage(fx, hist, max_age=2, verbose=False)
    assert rep["missing"] == ["D"]
    assert [t for t, _ in rep["stale"]] == ["C"]
    assert rep["ok"] == 2


def test_a_clean_fixture_list_says_so(capsys):
    fx = pd.DataFrame({"home_team": ["A"], "away_team": ["B"]})
    rep = predict.check_fixture_coverage(fx, _history(["A", "B"]))
    assert rep == {"missing": [], "stale": [], "ok": 2}
    assert "have history within" in capsys.readouterr().out


def test_coverage_survives_the_mixed_season_labels():
    """`'2526'` is the most recent season, so a club last seen there is current."""
    hist = pd.concat([_history(["A"], "2526"), _history(["A", "B"], "2014/2015")])
    rep = predict.check_fixture_coverage(
        pd.DataFrame({"home_team": ["A"], "away_team": ["B"]}), hist, max_age=2, verbose=False)
    assert rep["missing"] == []
    assert [t for t, _ in rep["stale"]] == ["B"]


def test_no_fixtures_is_not_an_error():
    empty = pd.DataFrame(columns=["home_team", "away_team"])
    assert predict.check_fixture_coverage(empty, _history(["A"])) == {
        "missing": [], "stale": [], "ok": 0}
