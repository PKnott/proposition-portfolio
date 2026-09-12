"""Data ingestion.

Neither the Data Pull notebook nor the Run notebook owns this code -- both are
thin drivers over ``refresh()``. That is how "Run reuses Data Pull's functions
rather than keeping its own copy" is satisfied structurally rather than by
discipline.

Two cadences, matching the README's existing split:

* ``mode="current"`` -- day to day, before a prediction run
* ``mode="all"``     -- start of season / first build, the expensive one
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..config import (LEAGUES, current_season_label, historic_seasons, season_label,
                      season_sort_key)
from . import espn, scoreboard, understat
from .retry import with_retry

STEPS_CURRENT = ("understat_current", "results_current", "match_stats", "reconcile",
                 "fixtures", "all_comp_current")
# `club_mapping` sits after the Understat pulls and before anything that reads a
# club name, because it is built *from* those pulls. It had no step at all until
# now -- the builder lived only in the archived v1 notebook, so the mapping was a
# frozen snapshot that decayed a little at every promotion window and
# `load_name_map` raised pointing at a step that no longer existed.
STEPS_ALL = ("understat_history", "understat_current", "club_mapping", "team_ids",
             "results_current", "match_stats", "reconcile", "all_comp_history",
             "all_comp_current", "fixtures")


@dataclass
class IngestReport:
    mode: str
    ran: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [f"Ingest ({self.mode}): {len(self.ran)} step(s) ran"]
        for s in self.ran:
            lines.append(f"  ok      {s}")
        for s in self.skipped:
            lines.append(f"  skipped {s}")
        for s, e in self.errors.items():
            lines.append(f"  FAILED  {s}: {e}")
        return "\n".join(lines)


def refresh(
    mode: str = "current",
    steps: tuple[str, ...] | None = None,
    *,
    date_from: str | None = None,
    date_to: str | None = None,
    force: bool = False,
) -> IngestReport:
    """Run the ingestion steps for a cadence.

    ``date_from``/``date_to`` (``dd-mm-yyyy``) bound the fixture pull; without
    them the fixture step is skipped, which is what you want when only refreshing
    results.
    """
    chosen = steps or (STEPS_ALL if mode == "all" else STEPS_CURRENT)
    rep = IngestReport(mode=mode)

    def run(name: str, fn) -> None:
        if name not in chosen:
            return
        try:
            print(f"[{name}]")
            fn()
            rep.ran.append(name)
        except Exception as e:  # noqa: BLE001 -- one failing step should not abort the rest
            print(f"  FAILED: {type(e).__name__}: {e}")
            rep.errors[name] = f"{type(e).__name__}: {e}"

    def _understat_history() -> None:
        for key in LEAGUES:
            print(f"  {key}...")
            understat.refresh_league(key, historic_seasons())

    def _understat_current() -> None:
        for key in LEAGUES:
            understat.refresh_league(key, current=True)

    def _results_current() -> None:
        # A scoreboard fetched *before* kickoff holds no final score, and
        # `scoreboard.missing_dates` never refetches a date whose file already
        # exists -- so something has to go back for it. That used to happen by
        # accident: `espn.pull_fixtures` refetched every date of the season on
        # every run, and quietly repaired the current season's results as a side
        # effect of looking for fixtures. It no longer does -- that was 257
        # requests a run to find one day's fixtures -- so the repair is its own
        # step now, which is the honest place for it either way: a side effect
        # nobody names is a side effect nobody notices losing.
        #
        # Driven off ESPN's own season calendar rather than Understat's published
        # match dates, for the reason `refresh_results` sets out at length.
        today = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d")
        season_start = season_sort_key(current_season_label().replace("-", "/"))
        dates: dict[str, set[str]] = {}
        for key, lg in LEAGUES.items():
            live, _, _ = espn.refresh_season_calendar(lg.espn_slug, season_start)
            played = {d for d in live if d <= today}
            if played:
                dates[key] = played
        # `match_stats` runs next and rebuilds from the whole cache regardless,
        # so the per-call rebuild would just be the same parse done twice.
        scoreboard.refresh_results(dates, rebuild=False)

    def _reconcile() -> None:
        # Imported here, not at module scope: `clean` imports from this package,
        # so a top-level `from ..reconcile import reconcile` would be circular.
        # Same reasoning as `import soccerdata` inside `understat.pull_season`.
        from ..reconcile import reconcile as _run
        print(_run())

    run("understat_history", _understat_history)
    run("understat_current", _understat_current)
    run("results_current", _results_current)
    run("club_mapping", espn.build_club_mapping)
    run("team_ids", lambda: espn.build_team_id_cache(force=force))
    run("match_stats", lambda: scoreboard.build_match_stats(save=True))
    run("reconcile", _reconcile)
    # Derived from the history rather than hardcoded: the old `range(2014, 2025)`
    # stopped at 2024/25 and silently missed whatever season had rolled into
    # HISTORIC_SEASONS since someone last edited it.
    run("all_comp_history", lambda: espn.pull_all_competition_fixtures(
        [int(season_label(s).split("-")[0]) for s in historic_seasons()], label="Historic"))
    run("all_comp_current", lambda: espn.pull_all_competition_fixtures(
        [int(current_season_label().split("-")[0])], label="Current"))

    if date_from and date_to:
        run("fixtures", lambda: espn.pull_fixtures(date_from, date_to))
    elif "fixtures" in chosen:
        rep.skipped.append("fixtures (no date range given)")

    return rep


__all__ = [
    "refresh", "IngestReport", "STEPS_CURRENT", "STEPS_ALL",
    "espn", "scoreboard", "understat", "with_retry",
]
