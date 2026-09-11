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

from ..config import LEAGUES, current_season_label, historic_seasons, season_label
from . import espn, scoreboard, understat
from .retry import with_retry

STEPS_CURRENT = ("understat_current", "match_stats", "reconcile", "fixtures", "all_comp_current")
# `club_mapping` sits after the Understat pulls and before anything that reads a
# club name, because it is built *from* those pulls. It had no step at all until
# now -- the builder lived only in the archived v1 notebook, so the mapping was a
# frozen snapshot that decayed a little at every promotion window and
# `load_name_map` raised pointing at a step that no longer existed.
STEPS_ALL = ("understat_history", "understat_current", "club_mapping", "team_ids",
             "match_stats", "reconcile", "all_comp_history", "all_comp_current", "fixtures")


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

    def _reconcile() -> None:
        # Imported here, not at module scope: `clean` imports from this package,
        # so a top-level `from ..reconcile import reconcile` would be circular.
        # Same reasoning as `import soccerdata` inside `understat.pull_season`.
        from ..reconcile import reconcile as _run
        print(_run())

    run("understat_history", _understat_history)
    run("understat_current", _understat_current)
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
