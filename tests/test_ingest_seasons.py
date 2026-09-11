"""Guards for the season boundaries, the cache scan, and the name suggestions.

These three things share a failure mode: they go wrong quietly. A stale season
constant produces a valid-looking split a year out of date; a backfill that checks
the wrong list reports success having fetched nothing; a poisoned cache file reads
back as "already done". None of them raise on their own, so they are pinned here.

Everything in this module is offline -- no network, no soccerdata, no real Inputs.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from fpp import config, reconcile
from fpp.ingest import espn, scoreboard, understat


@pytest.fixture(autouse=True)
def restore_seasons():
    """Season state is process-global, so put it back after every test."""
    current, previous = config.current_season(), config.previous_season()
    yield
    config.set_seasons(current=current, previous=previous)


# --- Season boundaries ----------------------------------------------------


def test_defaults_are_self_consistent() -> None:
    assert config.test_season() == config.previous_season()
    assert config.historic_seasons()[-1] == config.previous_season()
    assert config.historic_seasons()[0] == "2014/2015"
    assert config.season_label(config.current_season()) == config.current_season_label()


def test_one_edit_moves_every_boundary() -> None:
    """The whole point: a rollover is one change, not four."""
    config.set_seasons(current="2027/2028", previous="2026/2027")

    assert config.current_season_label() == "2027-28"
    assert config.test_season() == "2026/2027"
    assert config.first_val_season() == "2021/2022"
    assert config.historic_seasons()[-1] == "2026/2027"


@pytest.mark.parametrize(
    "current, previous, expected_first_val",
    [
        ("2026/2027", "2025/2026", "2020/2021"),
        ("2030/2031", "2029/2030", "2024/2025"),
    ],
)
def test_first_val_season_tracks_the_test_boundary(
    current: str, previous: str, expected_first_val: str
) -> None:
    """Five seasons back, so the window slides instead of widening each rollover."""
    config.set_seasons(current=current, previous=previous)
    assert config.first_val_season() == expected_first_val


@pytest.mark.parametrize(
    "current, previous",
    [
        ("2026/2028", "2025/2026"),   # not consecutive years
        ("2026/2027", "2024/2025"),   # two-year gap -- the rollover bug
        ("2026/2027", "2026/2027"),   # no gap at all
        ("not-a-season", "2025/2026"),
        ("2026-2027", "2025/2026"),   # label form, not season form
        ("2014/2015", "2013/2014"),   # at/before the first historic year
    ],
)
def test_set_seasons_rejects_bad_pairs(current: str, previous: str) -> None:
    with pytest.raises(ValueError):
        config.set_seasons(current=current, previous=previous)


def test_rejected_override_leaves_state_untouched() -> None:
    """Validation happens before mutation -- a bad call must not half-apply."""
    before = (config.current_season(), config.previous_season())
    with pytest.raises(ValueError):
        config.set_seasons(current="2026/2027", previous="2020/2021")
    assert (config.current_season(), config.previous_season()) == before


# --- Backfill cache scan --------------------------------------------------


def _matches(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"league_key": k, "season": s, "date": pd.Timestamp(d)} for k, s, d in rows]
    )


def test_missing_dates_reports_what_is_absent(tmp_path) -> None:
    """The bug this replaces: only ever checking a hardcoded list."""
    matches = _matches([
        ("Prem", "2020/2021", "2020-09-12"),
        ("Bund", "2019/2020", "2019-08-16"),
    ])
    # Seed the cache so Prem's window is complete but Bund's is not.
    for d in ("20200911", "20200912", "20200913"):
        (tmp_path / f"Schedule_eng.1_{d}.json").write_text(json.dumps({"events": []}))

    gaps = scoreboard.missing_dates(matches, tmp_path)

    assert ("Prem", "2020/2021") not in gaps, "complete combinations should be omitted"
    assert gaps[("Bund", "2019/2020")] == {"20190815", "20190816", "20190817"}


def test_missing_dates_covers_every_combination_not_a_fixed_list(tmp_path) -> None:
    """Any incomplete league-season is picked up, not just once-known ones.

    The deleted `KNOWN_MISSING` listed four combinations; Ligue 2022/23 was never
    one of them, and is exactly the kind of gap that used to go unnoticed.
    """
    matches = _matches([
        ("Ligue", "2022/2023", "2023-01-14"),
        ("Serie", "2015/2016", "2015-08-22"),
    ])
    gaps = scoreboard.missing_dates(matches, tmp_path)

    assert set(gaps) == {("Ligue", "2022/2023"), ("Serie", "2015/2016")}
    assert not hasattr(scoreboard, "KNOWN_MISSING"), "the hardcoded list must stay deleted"


def test_missing_dates_honours_scoping(tmp_path) -> None:
    matches = _matches([
        ("Prem", "2020/2021", "2020-09-12"),
        ("Bund", "2019/2020", "2019-08-16"),
    ])
    assert set(scoreboard.missing_dates(matches, tmp_path, leagues=["Bund"])) == {("Bund", "2019/2020")}
    assert set(scoreboard.missing_dates(matches, tmp_path, seasons=["2020/2021"])) == {("Prem", "2020/2021")}


def test_dry_run_fetches_nothing(tmp_path) -> None:
    matches = _matches([("Prem", "2020/2021", "2020-09-12")])
    n = scoreboard.backfill_from_matches(matches, cache_dir=tmp_path, dry_run=True)

    assert n == 3  # the match date, widened by +/- 1
    assert list(tmp_path.glob("*.json")) == [], "a dry run must not write"


def test_backfill_is_a_noop_when_the_cache_is_complete(tmp_path) -> None:
    matches = _matches([("Prem", "2020/2021", "2020-09-12")])
    for d in ("20200911", "20200912", "20200913"):
        (tmp_path / f"Schedule_eng.1_{d}.json").write_text(json.dumps({"events": []}))

    # No monkeypatching of the fetch: if it tried to hit the network, this would fail.
    assert scoreboard.backfill_from_matches(matches, cache_dir=tmp_path) == 0


def test_max_calls_caps_a_run(tmp_path, monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(scoreboard, "get_json", lambda url: calls.append(url) or {"events": []})
    monkeypatch.setattr(scoreboard, "SCOREBOARD_DELAY", 0)

    matches = _matches([("Prem", "2020/2021", f"2020-09-{d:02d}") for d in range(1, 20)])
    written = scoreboard.backfill_from_matches(matches, cache_dir=tmp_path, max_calls=5)

    assert written == 5
    assert len(calls) == 5
    assert len(list(tmp_path.glob("*.json"))) == 5


# --- Understat empty-fetch guard ------------------------------------------


class _FakeUnderstat:
    """Stands in for soccerdata's reader. `frame` is whatever the site returned."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def read_team_match_stats(self) -> pd.DataFrame:
        return self._frame


def _patch_fetch(monkeypatch, frame: pd.DataFrame) -> None:
    """Replace the soccerdata call inside pull_season with a canned frame."""
    def fake_with_retry(fn, *args, **kwargs):
        if kwargs.get("leagues") is not None or args:
            return _FakeUnderstat(frame)
        return fn()
    monkeypatch.setattr(understat, "with_retry", fake_with_retry)


def test_empty_fetch_is_not_cached(tmp_path, monkeypatch) -> None:
    """An empty early-season result must not pin the season to empty."""
    _patch_fetch(monkeypatch, pd.DataFrame())
    out = understat.pull_season("Prem", "2026/2027", out_dir=tmp_path)

    assert out.empty
    assert list(tmp_path.glob("*.csv")) == [], "an empty frame must not be cached"


def test_column_less_fetch_is_not_cached(tmp_path, monkeypatch) -> None:
    _patch_fetch(monkeypatch, pd.DataFrame({"something_else": [1, 2]}))
    understat.pull_season("Prem", "2026/2027", out_dir=tmp_path)

    assert list(tmp_path.glob("*.csv")) == []


def test_a_poisoned_cache_file_is_refetched(tmp_path, monkeypatch) -> None:
    """The live failure: a cached file with no date column raised on every run."""
    cached = tmp_path / "2026_2027.csv"
    cached.write_text("home_team,away_team\nArsenal,Chelsea\n")  # no date column

    good = pd.DataFrame({"date": ["2026-08-15"], "home_team": ["Arsenal"], "away_team": ["Chelsea"]})
    _patch_fetch(monkeypatch, good)

    out = understat.pull_season("Prem", "2026/2027", out_dir=tmp_path)

    assert "date" in out.columns
    assert "date" in pd.read_csv(cached).columns, "the bad file should have been replaced"


def test_a_good_cache_file_is_reused(tmp_path, monkeypatch) -> None:
    cached = tmp_path / "2026_2027.csv"
    cached.write_text("date,home_team,away_team,season\n2026-08-15,Arsenal,Chelsea,2026/2027\n")

    def explode(*a, **k):
        raise AssertionError("should not refetch a usable cache file")
    monkeypatch.setattr(understat, "with_retry", explode)

    assert len(understat.pull_season("Prem", "2026/2027", out_dir=tmp_path)) == 1


# --- Name suggestions -----------------------------------------------------


def test_best_match_picks_the_closest_name() -> None:
    assert espn.best_match("Coventry City", {"Coventry", "Cardiff"})[0] == "Coventry"
    assert espn.best_match("Manchester Utd", {"Manchester United", "Manchester City"})[0] == "Manchester United"


def test_best_match_on_an_empty_pool_is_none() -> None:
    assert espn.best_match("anything", set()) is None


def test_report_unmapped_suggests_without_applying(capsys, tmp_path, monkeypatch) -> None:
    """It must print a candidate and touch nothing on disk, at any score."""
    mapping = tmp_path / "Club_Mapping_All.csv"
    mapping.write_text("espn_team,understat_team\nArsenal,Arsenal\n")
    before = mapping.read_text()
    monkeypatch.setattr(espn, "CLUB_MAPPING_ALL", mapping)

    espn.report_unmapped({"Coventry City"}, pool={"Coventry", "Cardiff"})

    out = capsys.readouterr().out
    assert "Coventry City" in out and "'Coventry'" in out and "similarity" in out
    assert mapping.read_text() == before, "a suggestion must never edit the mapping"


def test_report_unmapped_is_silent_when_nothing_is_unmapped(capsys) -> None:
    espn.report_unmapped(set(), pool={"Coventry"})
    assert capsys.readouterr().out == ""


def test_report_unmapped_caps_the_list(capsys) -> None:
    espn.report_unmapped({f"Team {i}" for i in range(15)}, pool={"Coventry"}, cap=3)
    out = capsys.readouterr().out
    assert "... and 12 more" in out


# --- Completeness: "real data", not "a file exists" -----------------------


def _season_frame(n_teams: int, n_matches: int | None = None) -> pd.DataFrame:
    """A synthetic league-season: every team plays every other home and away."""
    teams = [f"Team {i}" for i in range(n_teams)]
    rows = [
        {"date": pd.Timestamp("2025-08-15"), "home_team": h, "away_team": a}
        for h in teams for a in teams if h != a
    ]
    return pd.DataFrame(rows[:n_matches] if n_matches is not None else rows)


@pytest.mark.parametrize("n_teams, expected", [(20, 380), (18, 306)])
def test_expected_matches_is_derived_not_configured(n_teams: int, expected: int) -> None:
    """380 and 306 fall out of the team count -- neither is written down anywhere."""
    assert understat.expected_matches(_season_frame(n_teams)) == expected


def test_a_complete_season_has_no_shortfall() -> None:
    assert understat.season_shortfall(_season_frame(20)) == 0
    assert understat.season_shortfall(_season_frame(18)) == 0


def test_frozen_season_is_detected() -> None:
    """Regression: last season froze at 370 of 380 in four of the five leagues.

    The cached current-season file was served forever because it existed and
    parsed, so the final matchday never arrived and nothing said so.
    """
    frozen = _season_frame(20, n_matches=370)
    assert understat.expected_matches(frozen) == 380
    assert understat.season_shortfall(frozen) == 10
    assert understat.is_usable(frozen), "it is real data -- just not all of it"


def test_unusable_frames_report_no_expectation() -> None:
    """A shortfall against an unknown target would be meaningless."""
    poisoned = pd.DataFrame({"season": ["2026/2027"]})   # the real poisoned-CSV shape
    assert not understat.is_usable(poisoned)
    assert understat.expected_matches(poisoned) == 0
    assert understat.season_shortfall(poisoned) == 0


def test_unusable_is_distinct_from_incomplete(tmp_path) -> None:
    """Two different failures that need two different responses."""
    (tmp_path / "2026_2027.csv").write_text("season\n2026/2027\n")
    assert understat.read_cached_season("Prem", "2026/2027", out_dir=tmp_path) is None


# --- The expected grid ----------------------------------------------------


def test_expected_combinations_covers_history_plus_current() -> None:
    combos = reconcile.expected_combinations()
    assert len(combos) == len(config.LEAGUES) * (len(config.historic_seasons()) + 1)
    assert ("Prem", config.current_season()) in combos
    assert ("Bund", config.historic_seasons()[0]) in combos


def test_expected_combinations_moves_with_the_season() -> None:
    """It cannot go stale, because there is nothing to update."""
    before = len(reconcile.expected_combinations())
    config.set_seasons(current="2027/2028", previous="2026/2027")
    after = reconcile.expected_combinations()

    assert len(after) == before + len(config.LEAGUES)   # one more historic season
    assert ("Prem", "2027/2028") in after
    assert ("Prem", "2026/2027") in after               # last season, now history


def test_expected_combinations_scoping() -> None:
    combos = reconcile.expected_combinations(leagues=["Bund"])
    assert {k for k, _ in combos} == {"Bund"}


def test_audit_never_fetches(tmp_path, monkeypatch) -> None:
    """The audit is safe to run at any time; only reconcile() may fetch."""
    def explode(*a, **k):
        raise AssertionError("audit must not hit the network")
    monkeypatch.setattr(scoreboard, "get_json", explode)
    monkeypatch.setattr(understat, "with_retry", explode)
    monkeypatch.setattr(understat, "INPUTS", tmp_path)          # empty tree

    state = reconcile.audit(leagues=["Bund"])

    assert len(state) == len(config.historic_seasons()) + 1
    # An absent historic season is a real gap; an absent current one has simply
    # not started. Same empty disk, two different findings.
    historic = state[state["season"] != config.current_season()]
    current = state[state["season"] == config.current_season()]
    assert (historic["action"] == "unusable").all()
    assert (current["action"] == "pending").all()


def test_pending_current_season_is_not_a_residual() -> None:
    """An August run must not report five phantom failures."""
    pending = reconcile.SeasonStatus(
        league_key="Prem", season=config.current_season(), action="pending")
    assert pending.understat_ok and pending.espn_ok

    missing_history = reconcile.SeasonStatus(
        league_key="Prem", season="2019/2020", action="unusable")
    assert not missing_history.understat_ok


def test_a_short_season_is_a_residual() -> None:
    """The frozen-season case must still surface."""
    st = reconcile.SeasonStatus(
        league_key="Prem", season="2025/2026",
        matches=370, expected=380, with_stats=370, pct=100.0)
    assert st.shortfall == 10
    assert not st.understat_ok
    assert st.espn_ok, "ESPN covered everything it was given -- the gap is upstream"


# --- the league index that pinned 2026/2027 to invisible ---------------------


def test_expire_league_index_respects_age(tmp_path, monkeypatch):
    """Fresh index is kept; stale one is removed so soccerdata refetches it.

    soccerdata reads `leagues.json` with `no_cache=False` forever, and
    `read_seasons` derives the set of existing seasons from it. A season that
    starts after that file was written is invisible until it is dropped.
    """
    import os, time
    from fpp.ingest import understat as U

    idx = tmp_path / "leagues.json"
    idx.write_text("{}")
    monkeypatch.setattr(U, "SOCCERDATA_CACHE", tmp_path)

    assert U.expire_league_index(1.0) is False, "a fresh index must be kept"
    assert idx.exists()

    old = time.time() - 3 * 86400
    os.utime(idx, (old, old))
    assert U.expire_league_index(1.0) is True, "a 3-day-old index must be dropped"
    assert not idx.exists()


def test_expire_league_index_missing_is_not_an_error(tmp_path, monkeypatch):
    from fpp.ingest import understat as U
    monkeypatch.setattr(U, "SOCCERDATA_CACHE", tmp_path)
    assert U.expire_league_index(0) is False


def test_pull_season_retries_once_when_season_missing_from_index(tmp_path, monkeypatch):
    """A fresh-but-pre-season index must not pin the current season to empty.

    The age timer alone leaves a gap on the one day it matters: refresh the index
    in the morning, before Understat publishes the new month, and every run for
    the next 24 hours reads an index that predates the season. The miss itself
    has to be the trigger.
    """
    import sys, types
    import pandas as pd
    from fpp.ingest import understat as U

    idx = tmp_path / "leagues.json"
    idx.write_text("{}")
    monkeypatch.setattr(U, "SOCCERDATA_CACHE", tmp_path)
    monkeypatch.setattr(U, "current_season", lambda: "2026/2027")

    good = pd.DataFrame({"date": ["2026-08-21"], "home_team": ["Arsenal"],
                         "away_team": ["Coventry"], "home_goals": [3], "away_goals": [0]})
    calls = []

    class FakeReader:
        def __init__(self, **kw): pass
        def read_team_match_stats(self):
            calls.append(idx.exists())
            # First call: index present and stale-but-fresh -> season unlisted.
            # soccerdata rewrites the index on read, so recreate it.
            if len(calls) == 1:
                idx.write_text("{}")
                return pd.DataFrame(index=["league", "season", "game"])  # (3, 0)
            return good.copy()

    monkeypatch.setitem(sys.modules, "soccerdata", types.SimpleNamespace(Understat=FakeReader))

    out = U.pull_season("Prem", "2026/2027", out_dir=tmp_path, force=True)
    assert len(calls) == 2, "an unlisted current season must trigger exactly one retry"
    assert U.is_usable(out), "the retry's result must be the one returned"
    assert not idx.exists() or idx.read_text() == "{}"


def test_pull_season_does_not_retry_for_a_historical_season(tmp_path, monkeypatch):
    """Only the current season can be missing because the index went stale."""
    import sys, types
    import pandas as pd
    from fpp.ingest import understat as U

    (tmp_path / "leagues.json").write_text("{}")
    monkeypatch.setattr(U, "SOCCERDATA_CACHE", tmp_path)
    monkeypatch.setattr(U, "current_season", lambda: "2026/2027")
    calls = []

    class FakeReader:
        def __init__(self, **kw): pass
        def read_team_match_stats(self):
            calls.append(1)
            return pd.DataFrame(index=["league", "season", "game"])

    monkeypatch.setitem(sys.modules, "soccerdata", types.SimpleNamespace(Understat=FakeReader))
    U.pull_season("Prem", "2019/2020", out_dir=tmp_path, force=True)
    assert len(calls) == 1, "a historical season must not spend a second fetch"
