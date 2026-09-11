"""Verify that the data we should have is actually there, and fetch what isn't.

Every gap this pipeline has hit came from the same place: nothing checked. The
checks that existed were either a hand-maintained list (``KNOWN_MISSING``, a
snapshot of what was absent whenever someone last looked) or a bare
``path.exists()``. Both can be wrong indefinitely without anything noticing,
because both answer a question adjacent to the one that matters.

The question that matters is "for every league-season I should have, is there
real, complete data?" -- and it is answered by measurement, not memory:

* **Understat** -- a season is complete when it holds ``teams * (teams - 1)``
  matches. Self-calibrating, so a 20-team league wants 380 and an 18-team one
  306, and Ligue 1's 20 -> 18 change needs no edit anywhere. This is the check
  that catches a season frozen at 370 of 380, which is what happened to four of
  the five leagues last season and went unreported for months.
* **ESPN** -- coverage of the Understat matches, via the same
  ``clean.coverage_report`` the notebook already displayed.

The two are checked separately because they fail separately: Understat can be
short while ESPN is fine, and vice versa.

This module sits above both ``clean`` and ``ingest`` deliberately. ``clean``
already imports from ``ingest``, so reusing ``attach_espn_stats`` and
``coverage_report`` from inside ``ingest`` would be a circular import.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from .clean import attach_espn_stats, coverage_report
from .config import LEAGUES, current_season, historic_seasons
from .ingest import scoreboard, understat
from .ingest.scoreboard import fixture_kickoffs, settled_after
from .paths import SOCCERDATA_CACHE

# Below this, a league-season is worth naming in the residual report.
MIN_COVERAGE = 95.0


@dataclass
class SeasonStatus:
    """What we have versus what we should have, for one league-season."""

    league_key: str
    season: str
    matches: int = 0          # Understat rows present
    expected: int = 0         # teams * (teams - 1); 0 when unusable
    with_stats: int = 0       # matches carrying ESPN shots/SOT/corners
    pct: float = 0.0
    missing_dates: int = 0    # scoreboard date files not on disk
    action: str = "ok"        # ok | pending | unusable | short | refetched

    @property
    def shortfall(self) -> int:
        return max(0, self.expected - self.matches)

    @property
    def understat_ok(self) -> bool:
        # A current season with nothing played yet is pending, not broken. It is
        # still refetched every run -- this only keeps it out of the residuals,
        # so an August run does not report five phantom failures.
        if self.action == "pending":
            return True
        return self.expected > 0 and self.shortfall == 0

    @property
    def espn_ok(self) -> bool:
        # Nothing to cover is not a coverage failure.
        return self.matches == 0 or self.pct >= MIN_COVERAGE


@dataclass
class ReconcileReport:
    statuses: list[SeasonStatus] = field(default_factory=list)
    understat_refetched: list[tuple[str, str]] = field(default_factory=list)
    dates_fetched: int = 0
    residual: list[SeasonStatus] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"Reconcile: {len(self.statuses)} league-season(s) checked, "
            f"{len(self.understat_refetched)} Understat refetch(es), "
            f"{self.dates_fetched:,} scoreboard file(s) fetched"
        ]
        if not self.residual:
            lines.append("  every league-season has complete Understat data and "
                         f"{MIN_COVERAGE:.0f}%+ ESPN coverage")
            return "\n".join(lines)

        # Residuals are named with a cause. A gap that survives reconciliation is
        # a fact about the sources, not something to go and re-run.
        lines.append(f"  {len(self.residual)} still short:")
        for s in self.residual:
            bits = []
            if not s.understat_ok:
                bits.append(f"Understat {s.matches}/{s.expected}" if s.expected
                            else "Understat unusable")
            if not s.espn_ok:
                cause = "fetch incomplete" if s.missing_dates else "not in ESPN's data"
                bits.append(f"ESPN {s.pct:.1f}% ({cause})")
            lines.append(f"    {s.league_key} {s.season}: {'; '.join(bits)}")
        return "\n".join(lines)


def expected_combinations(
    leagues: list[str] | None = None, seasons: list[str] | None = None
) -> list[tuple[str, str]]:
    """Every ``(league, season)`` that should have data, re-derived every call.

    This is what replaces the hand-maintained lists. It cannot go stale, because
    there is nothing to update -- it follows ``config``'s season boundaries.
    """
    all_seasons = [*historic_seasons(), current_season()]
    return [
        (k, s)
        for k in (leagues or list(LEAGUES))
        for s in (seasons or all_seasons)
        if s in all_seasons and k in LEAGUES
    ]


def _understat_status(league_key: str, season: str) -> SeasonStatus:
    """Read-only look at one season's cached Understat data."""
    st = SeasonStatus(league_key=league_key, season=season)
    df = understat.read_cached_season(league_key, season)
    if df is None:
        # Nothing usable. For the current season that is the normal state before
        # a ball is kicked; for a historic one it is a real gap.
        st.action = "pending" if season == current_season() else "unusable"
        return st
    st.matches = len(df)
    st.expected = understat.expected_matches(df)
    if st.shortfall:
        st.action = "short"
    return st


def audit(
    leagues: list[str] | None = None, seasons: list[str] | None = None
) -> pd.DataFrame:
    """What exists versus what should, for the whole grid. Fetches nothing.

    Safe to call any time -- it only reads the caches. Use it before a reconcile
    to see the size of the job, and after one to confirm it did what it claimed.
    """
    combos = expected_combinations(leagues, seasons)
    statuses = [_understat_status(k, s) for k, s in combos]

    # ESPN coverage needs the joined table, which needs a usable match table.
    try:
        matches = understat.load_matches()
    except (FileNotFoundError, ValueError):
        matches = None

    if matches is not None and not matches.empty:
        gaps = scoreboard.missing_dates(matches)
        cov = coverage_report(attach_espn_stats(matches))
        by_combo = {(r.league_key, r.season): r for r in cov.itertuples()}
        for st in statuses:
            key = (st.league_key, st.season)
            if (row := by_combo.get(key)) is not None:
                st.with_stats, st.pct = int(row.with_stats), float(row.pct)
            st.missing_dates = len(gaps.get(key, ()))

    return pd.DataFrame([vars(st) | {"shortfall": st.shortfall} for st in statuses])


def current_result_dates(matches: pd.DataFrame, leagues: list[str] | None = None
                         ) -> dict[str, set[str]]:
    """``league_key`` -> the current-season dates whose results we should have.

    Understat's published dates **union our own fixtures**, and the union is a
    fix for a real gap rather than belt and braces. Every ESPN path here used to
    be driven off `matches`, which is Understat -- so a date Understat had not
    published yet was never checked, never refetched, and never parsed, even
    though ESPN had the final stats sitting there. On a Saturday night with
    Understat 7 matches into a 182-match Bundesliga season, that is every fixture
    played that day.
    """
    season = current_season()
    out: dict[str, set[str]] = {}
    sub = matches[matches["season"] == season] if len(matches) else matches
    if len(sub):
        for league_key, grp in sub.groupby("league_key"):
            if leagues and league_key not in leagues:
                continue
            out.setdefault(league_key, set()).update(
                pd.to_datetime(grp["date"]).dt.strftime("%Y%m%d"))
    for (league_key, day) in fixture_kickoffs(leagues):
        out.setdefault(league_key, set()).add(day)
    return out


def _refresh_stale_current_dates(matches: pd.DataFrame, cache_dir=None) -> int:
    """Refetch current-season scoreboards captured before their match finished.

    A scoreboard pulled on match day can hold a live fixture with no final stats,
    and the file existing afterwards makes it look done. Staleness is decided by
    mtime against the time the last match on that date should have ended, so this
    terminates: once a file is written after full time it is never fetched again.
    """
    cache_dir = cache_dir or SOCCERDATA_CACHE
    wanted = current_result_dates(matches)
    if not wanted:
        return 0
    kickoffs = fixture_kickoffs()
    now = pd.Timestamp.now(tz="UTC").timestamp()

    stale: dict[str, set[str]] = {}
    for league_key, days in wanted.items():
        slug = LEAGUES[league_key].espn_slug
        for day in days:
            settled = settled_after(league_key, day, kickoffs)
            if now < settled:
                continue  # still being played; nothing to refetch yet
            path = cache_dir / f"Schedule_{slug}_{day}.json"
            if not path.exists():
                # Absent and already finished: the normal backfill only covers
                # dates Understat has, so this is where a played-but-unpublished
                # date gets picked up.
                stale.setdefault(slug, set()).add(day)
                continue
            if path.stat().st_mtime < settled:
                path.unlink()
                stale.setdefault(slug, set()).add(day)

    fetched = 0
    for slug, dates in stale.items():
        fetched += scoreboard.fetch_scoreboard_dates(slug, dates, cache_dir)
    if fetched:
        print(f"  refetched {fetched} current-season scoreboard(s) captured mid-match")
    return fetched


def reconcile(
    *,
    dry_run: bool = False,
    leagues: list[str] | None = None,
    seasons: list[str] | None = None,
    cache_dir=None,
) -> ReconcileReport:
    """Check every league-season that should exist, and fetch whatever doesn't.

    Idempotent: a second run immediately after a first fetches nothing, because
    every decision is a measurement of the current state rather than a record of
    what was done last time.

    ``dry_run`` reports the same findings without fetching. From a cold cache a
    real run is ~10,000 requests; it is safe to interrupt, since existing files
    are never refetched and a resumed run continues where it stopped.
    """
    rep = ReconcileReport()
    combos = expected_combinations(leagues, seasons)
    started = time.time()

    # --- Pass 1: Understat, per season ------------------------------------
    needs_pull: list[tuple[str, str]] = []
    for league_key, season in combos:
        st = _understat_status(league_key, season)
        # The current season is never "complete" -- it is still being played, so
        # a shortfall is expected and refetching is the whole point. This is the
        # case that silently froze at 370/380 last season.
        if st.action in ("unusable", "short") or season == current_season():
            needs_pull.append((league_key, season))
        rep.statuses.append(st)

    if needs_pull and not dry_run:
        print(f"  Understat: {len(needs_pull)} season(s) to refetch")
        for league_key, season in needs_pull:
            understat.pull_season(league_key, season, force=True)
            rep.understat_refetched.append((league_key, season))
        # Rebuild the combined CSVs `load_matches` actually reads.
        for league_key in sorted({k for k, _ in needs_pull}):
            current_touched = any(s == current_season() for k, s in needs_pull if k == league_key)
            historic = [s for k, s in needs_pull if k == league_key and s != current_season()]
            if historic:
                understat.refresh_league(league_key)
            if current_touched:
                understat.refresh_league(league_key, current=True)
    elif needs_pull:
        print(f"  Understat: {len(needs_pull)} season(s) would be refetched")

    # --- Pass 2: ESPN scoreboards -----------------------------------------
    try:
        matches = understat.load_matches()
    except (FileNotFoundError, ValueError) as e:
        print(f"  cannot check ESPN coverage: {type(e).__name__}: {e}")
        return rep

    gaps = scoreboard.missing_dates(matches, cache_dir, leagues=leagues, seasons=seasons)
    outstanding = sum(len(d) for d in gaps.values())

    if dry_run:
        if outstanding:
            print(f"  ESPN: {len(gaps)} league-season(s) incomplete, "
                  f"{outstanding:,} date file(s) would be fetched "
                  f"(~{outstanding * scoreboard.SCOREBOARD_DELAY / 60:.0f} min)")
        for st in rep.statuses:
            st.missing_dates = len(gaps.get((st.league_key, st.season), ()))
        rep.residual = [s for s in rep.statuses if not s.understat_ok or s.missing_dates]
        return rep

    if outstanding:
        print(f"  ESPN: fetching {outstanding:,} missing date file(s) "
              f"across {len(gaps)} league-season(s)")
        rep.dates_fetched = scoreboard.backfill_from_matches(
            matches, cache_dir=cache_dir, leagues=leagues, seasons=seasons)

    # --- Pass 3: current-season staleness ---------------------------------
    rep.dates_fetched += _refresh_stale_current_dates(matches, cache_dir)

    # --- Pass 4: verify ---------------------------------------------------
    if rep.dates_fetched:
        scoreboard.build_match_stats(save=True)

    joined = attach_espn_stats(matches)
    cov = coverage_report(joined)
    by_combo = {(r.league_key, r.season): r for r in cov.itertuples()}
    gaps_after = scoreboard.missing_dates(matches, cache_dir, leagues=leagues, seasons=seasons)

    for st in rep.statuses:
        key = (st.league_key, st.season)
        if (row := by_combo.get(key)) is not None:
            st.with_stats, st.pct = int(row.with_stats), float(row.pct)
        st.missing_dates = len(gaps_after.get(key, ()))
        if (st.league_key, st.season) in rep.understat_refetched:
            fresh = understat.read_cached_season(st.league_key, st.season)
            if fresh is not None:
                st.matches, st.expected = len(fresh), understat.expected_matches(fresh)
            st.action = "refetched"
        elif st.espn_ok and st.understat_ok:
            st.action = "ok"

    # --- Pass 5: residuals ------------------------------------------------
    rep.residual = [s for s in rep.statuses if not (s.understat_ok and s.espn_ok)]
    print(f"  reconciled in {time.time() - started:.0f}s")
    return rep


__all__ = [
    "MIN_COVERAGE", "SeasonStatus", "ReconcileReport",
    "expected_combinations", "audit", "reconcile",
]
