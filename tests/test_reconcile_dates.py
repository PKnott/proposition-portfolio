"""Guards on which result dates the pipeline goes looking for.

The failure these exist for: every ESPN path was driven off the Understat match
table, so a date Understat had not published yet was never checked and never
refetched -- even with ESPN's final stats sitting there. On a Saturday night,
with Understat seven matches into a 182-match Bundesliga season, that was every
fixture played that day.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fpp import reconcile
from fpp.config import current_season
from fpp.ingest import scoreboard


@pytest.fixture
def fixtures(tmp_path, monkeypatch):
    """A fixtures directory holding one Bundesliga matchday."""
    # The freshness rule lives in the scoreboard layer: results are ESPN's, and
    # nothing about deciding when one is final involves Understat.
    monkeypatch.setattr(scoreboard, "FIXTURES_DIR", tmp_path)
    pd.DataFrame({
        "Date": ["2026-08-29", "2026-08-29", "2026-08-30"],
        "Home Team": ["Union Berlin", "Mainz 05", "Bayern Munich"],
        "Away Team": ["Eintracht Frankfurt", "Paderborn", "VfB Stuttgart"],
        "Start Time (UTC)": ["13:30:00", "16:30:00", "15:00:00"],
    }).to_csv(tmp_path / "bundesliga_fixtures.csv", index=False)
    return tmp_path


def test_kickoffs_are_the_latest_start_on_each_date(fixtures):
    ko = scoreboard.fixture_kickoffs()
    assert ko[("Bund", "20260829")] == pd.Timestamp("2026-08-29 16:30", tz="UTC")
    assert ko[("Bund", "20260830")] == pd.Timestamp("2026-08-30 15:00", tz="UTC")


def test_a_date_understat_has_not_published_is_still_looked_for(fixtures):
    """The bug, stated directly."""
    understat_has_nothing = pd.DataFrame(columns=["league_key", "date", "season"])
    dates = reconcile.current_result_dates(understat_has_nothing)
    assert "20260829" in dates["Bund"]


def test_understat_dates_and_fixture_dates_are_unioned(fixtures):
    matches = pd.DataFrame({"league_key": ["Bund"], "date": [pd.Timestamp("2026-08-15")],
                            "season": [current_season()]})
    dates = reconcile.current_result_dates(matches)
    assert {"20260815", "20260829", "20260830"} <= dates["Bund"]


def test_a_scoreboard_is_trusted_one_lag_after_the_last_kickoff(fixtures):
    """Waiting for midnight meant a game finishing at nine could not be banked
    until the next day, which is the whole reason a match-night run found
    nothing.

    Anchored on `FINAL_LAG` rather than a literal: the constant is meant to be
    tuned, and a test that restates its value turns every tune into a failure
    that says nothing about behaviour.
    """
    ko = scoreboard.fixture_kickoffs()
    at = scoreboard.settled_after("Bund", "20260829", ko)
    last_kickoff = pd.Timestamp("2026-08-29 16:30", tz="UTC")
    assert at == (last_kickoff + scoreboard.FINAL_LAG).timestamp()
    assert scoreboard.FINAL_LAG >= pd.Timedelta(hours=2), "shorter than a match"


def test_a_date_with_no_fixture_row_falls_back_to_midnight_following(fixtures):
    at = scoreboard.settled_after("Bund", "20260101", scoreboard.fixture_kickoffs())
    assert at == pd.Timestamp("2026-01-02").timestamp()


# --- the ESPN-only results pull -------------------------------------------


def test_refresh_results_waits_until_the_last_kickoff_has_passed(fixtures, tmp_path,
                                                                 monkeypatch):
    """A live fixture must never be banked as a final score."""
    called = []
    monkeypatch.setattr(scoreboard, "fetch_scoreboard_dates",
                        lambda slug, days, cd=None, **k: called.append((slug, set(days))) or 0)
    monkeypatch.setattr(scoreboard, "build_match_stats", lambda **k: None)
    # 17:00, half an hour after the day's last kickoff at 16:30.
    n = scoreboard.refresh_results({"Bund": {"20260829"}}, cache_dir=tmp_path,
                                   now=pd.Timestamp("2026-08-29 17:00", tz="UTC"))
    assert n == 0 and called == []


def test_refresh_results_fetches_once_the_lag_has_elapsed(fixtures, tmp_path, monkeypatch):
    monkeypatch.setattr(scoreboard, "fetch_scoreboard_dates",
                        lambda slug, days, cd=None, **k: len(days))
    monkeypatch.setattr(scoreboard, "build_match_stats", lambda **k: None)
    just_after = pd.Timestamp("2026-08-29 16:30", tz="UTC") + scoreboard.FINAL_LAG \
        + pd.Timedelta(minutes=1)
    n = scoreboard.refresh_results({"Bund": {"20260829"}}, cache_dir=tmp_path, now=just_after)
    assert n == 1


def test_refresh_results_deletes_a_capture_taken_before_full_time(fixtures, tmp_path,
                                                                  monkeypatch):
    """`fetch_scoreboard_dates` never refetches a file that exists, so a
    pre-match capture is kept forever unless it is removed first."""
    stale = tmp_path / "Schedule_ger.1_20260829.json"
    stale.write_text("{}")
    import os
    early = pd.Timestamp("2026-08-29 00:31", tz="UTC").timestamp()
    os.utime(stale, (early, early))

    monkeypatch.setattr(scoreboard, "fetch_scoreboard_dates",
                        lambda slug, days, cd=None, **k: len(days))
    monkeypatch.setattr(scoreboard, "build_match_stats", lambda **k: None)
    n = scoreboard.refresh_results({"Bund": {"20260829"}}, cache_dir=tmp_path,
                                   now=pd.Timestamp("2026-08-30 02:00", tz="UTC"))
    assert n == 1
    assert not stale.exists()


def test_refresh_results_leaves_a_capture_taken_after_full_time_alone(fixtures, tmp_path,
                                                                      monkeypatch):
    good = tmp_path / "Schedule_ger.1_20260829.json"
    good.write_text("{}")
    import os
    late = pd.Timestamp("2026-08-30 02:00", tz="UTC").timestamp()
    os.utime(good, (late, late))

    monkeypatch.setattr(scoreboard, "fetch_scoreboard_dates",
                        lambda slug, days, cd=None, **k: len(days))
    monkeypatch.setattr(scoreboard, "build_match_stats", lambda **k: None)
    assert scoreboard.refresh_results({"Bund": {"20260829"}}, cache_dir=tmp_path,
                                      now=pd.Timestamp("2026-08-30 02:00", tz="UTC")) == 0
    assert good.exists()
