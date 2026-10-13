"""European scoreboards: fetch, parse, map. Reads the league model's tools, never its data.

ESPN's ``uefa.champions``, ``uefa.europa`` and ``uefa.europa.conf`` scoreboards
carry the same per-competitor statistics block the domestic ones do, so the
event-level parsers in `ingest.scoreboard` -- `_parse_event` and its two
malformed-block filters -- are reused as they stand. What is *not* reused is
`scoreboard.parse_cache`: it filters on the domestic `SLUG_TO_KEY`, and widening
that would put European rows one step from the league model. This module runs its
own loop over its own cache directory instead.

Which dates to fetch
--------------------
There is no Understat match table to read European dates off, and a whole-season
sweep is ~300 calls per competition per season. ESPN's season calendar narrows
it: the season-opener scoreboard lists every round with its window. Inside a
round, European football is played Tuesday to Thursday, so those are the only
days fetched -- except the final, which has been played on a Saturday, and the
August 2020 tournament in Lisbon and Germany, which was played every day.
Qualifying rounds are skipped: top-five clubs rarely meet there, and nothing
is priced in them.

Season comes from the event, not the date. The 2019/20 finals were played in
August 2020, which a July cut-off would file under the following season.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..ingest.espn import load_name_map
from ..ingest.retry import ESPN_SITE_API, get_json
from ..ingest.scoreboard import (SCOREBOARD_DELAY, _is_degenerate, _is_sentinel, _parse_event,
                                 fetch_scoreboard_dates)
from .config import EURO_CACHE, EURO_COMPS, EURO_INPUTS, EURO_MATCH_STATS, EURO_SLUG_TO_KEY

# Calendar rounds that are qualifiers. "Knockout Round Playoffs" is not one of
# them -- it is the first knockout round proper -- so match on the label start.
_QUALIFYING = ("qualifying", "playoff round", "play-off round", "preliminary")
_MATCH_DAYS = {1, 2, 3}  # Tue, Wed, Thu
# The 2019/20 competitions finished as a single-venue tournament in August 2020.
_EVERY_DAY_FROM = {2019: pd.Timestamp("2020-07-01")}


def _opener_path(slug: str, year: int, cache_dir: Path) -> Path:
    return cache_dir / f"Schedule_{slug}_{year}0701.json"


def season_calendar(comp: str, year: int, cache_dir: Path | None = None) -> dict:
    """The season-opener scoreboard for one competition-season, cached."""
    cache_dir = cache_dir or EURO_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    slug = EURO_COMPS[comp].espn_slug
    path = _opener_path(slug, year, cache_dir)
    if path.exists():
        return json.loads(path.read_text())
    payload = get_json(f"{ESPN_SITE_API}/{slug}/scoreboard?dates={year}0701")
    path.write_text(json.dumps(payload))
    return payload


def season_dates(payload: dict, year: int) -> set[str]:
    """``YYYYMMDD`` dates worth fetching for one competition-season."""
    cal = (payload.get("leagues") or [{}])[0].get("calendar") or []
    entries = [e for block in cal if isinstance(block, dict) for e in block.get("entries") or []]
    out: set[str] = set()
    every_day_from = _EVERY_DAY_FROM.get(year)
    for e in entries:
        label = str(e.get("label", "")).lower()
        if label.startswith(_QUALIFYING):
            continue
        try:
            lo = pd.Timestamp(e["startDate"]).tz_localize(None).normalize()
            hi = pd.Timestamp(e["endDate"]).tz_localize(None).normalize()
        except (KeyError, ValueError, TypeError):
            continue
        final = label == "final"
        for d in pd.date_range(lo, hi, freq="D"):
            if final or d.dayofweek in _MATCH_DAYS or (every_day_from is not None and d >= every_day_from):
                out.add(d.strftime("%Y%m%d"))
    return out


def backfill(seasons: list[int], *, comps: list[str] | None = None,
             cache_dir: Path | None = None, dry_run: bool = True,
             max_calls: int | None = None) -> int:
    """Cache European scoreboards for ``seasons`` (start years). Resumable; dry by default.

    Mirrors `scoreboard.backfill_lower_leagues`: existing files are never
    refetched, so an interrupted run resumes for free. The calendar files
    themselves are fetched even on a dry run -- one call per competition-season
    -- because without them there is no plan to report.
    """
    cache_dir = cache_dir or EURO_CACHE
    plan: dict[tuple[str, int], set[str]] = {}
    for comp in comps or list(EURO_COMPS):
        c = EURO_COMPS[comp]
        for year in seasons:
            if year < c.first_season:
                continue
            want = season_dates(season_calendar(comp, year, cache_dir), year)
            plan[(comp, year)] = {d for d in want
                                  if not (cache_dir / f"Schedule_{c.espn_slug}_{d}.json").exists()}

    outstanding = sum(len(v) for v in plan.values())
    if dry_run:
        print(f"  {outstanding:,} date file(s) to fetch, ~{outstanding * SCOREBOARD_DELAY / 60:.0f} min "
              f"of delay alone. Dry run -- nothing fetched.")
        return outstanding

    total = 0
    for (comp, year), dates in plan.items():
        room = None if max_calls is None else max(0, max_calls - total)
        if room == 0:
            break
        n = fetch_scoreboard_dates(EURO_COMPS[comp].espn_slug, dates, cache_dir, limit=room)
        total += n
        print(f"  {comp} {year}/{year + 1}: {n:,} new files")
    return total


def _stage(event: dict) -> tuple[str | None, int | None, int | None]:
    """``(stage slug, leg, season start year)`` from a scoreboard event."""
    season = event.get("season") or {}
    comp = (event.get("competitions") or [{}])[0]
    leg = (comp.get("leg") or {}).get("value")
    try:
        leg = int(leg) if leg is not None else None
    except (TypeError, ValueError):
        leg = None
    year = season.get("year")
    return season.get("slug"), leg, int(year) if year is not None else None


def parse_cache(cache_dir: Path | None = None) -> pd.DataFrame:
    """Every European event with full stats, one row per match, de-duplicated by event id."""
    cache_dir = cache_dir or EURO_CACHE
    by_event: dict[str, dict] = {}
    dropped = 0
    for f in sorted(cache_dir.glob("Schedule_*.json")):
        slug = f.name.split("Schedule_", 1)[1].rsplit("_", 1)[0]
        if slug not in EURO_SLUG_TO_KEY:
            continue
        try:
            payload = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for event in payload.get("events") or []:
            rec = _parse_event(event)
            if rec is None:
                continue
            if _is_sentinel(rec) or _is_degenerate(rec):
                dropped += 1
                continue
            stage, leg, year = _stage(event)
            if year is None:
                continue
            rec.update(espn_slug=slug, league_key=EURO_SLUG_TO_KEY[slug], stage=stage, leg=leg,
                       season=f"{year}/{year + 1}")
            by_event[rec["espn_event_id"]] = rec
    if dropped:
        print(f"  Dropped {dropped:,} European event(s) with a malformed stats block")
    df = pd.DataFrame(list(by_event.values()))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "league_key"]).reset_index(drop=True)


COLUMNS = [
    "espn_event_id", "espn_slug", "league_key", "season", "stage", "leg", "date",
    "home_espn", "away_espn", "home_team", "away_team",
    "home_goals", "away_goals", "home_shots", "away_shots",
    "home_sot", "away_sot", "home_corners", "away_corners",
]


def build_match_stats(cache_dir: Path | None = None, save: bool = True,
                      name_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Parse -> map to Understat names -> save. Every European match is kept.

    Unmapped clubs keep a NaN name -- they are almost all from outside the five
    leagues and are expected to be. Restricting to top-five-versus-top-five is
    done later, against the clean table, because "plays in a modelled league"
    is a fact about a season, not about a club.
    """
    raw = parse_cache(cache_dir)
    if raw.empty:
        return pd.DataFrame(columns=COLUMNS)
    name_map = name_map if name_map is not None else load_name_map()
    raw["home_team"] = raw["home_espn"].map(name_map)
    raw["away_team"] = raw["away_espn"].map(name_map)
    df = raw[COLUMNS]
    if save:
        EURO_INPUTS.mkdir(parents=True, exist_ok=True)
        df.to_csv(EURO_MATCH_STATS, index=False)
        print(f"  Saved {len(df):,} European matches -> {EURO_MATCH_STATS}")
    return df


def load_match_stats() -> pd.DataFrame:
    if not EURO_MATCH_STATS.exists():
        raise FileNotFoundError(f"{EURO_MATCH_STATS} not found -- run europe.ingest.build_match_stats()")
    df = pd.read_csv(EURO_MATCH_STATS, dtype={"espn_event_id": str}, parse_dates=["date"])
    return df


__all__ = ["season_calendar", "season_dates", "backfill", "parse_cache", "build_match_stats",
           "load_match_stats", "COLUMNS"]
