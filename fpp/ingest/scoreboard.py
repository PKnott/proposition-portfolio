"""Shots, shots on target and corners, parsed out of the local soccerdata cache.

The brief assumed these needed a new ESPN *summary/boxscore* pull -- one HTTP
call per historical fixture, budgeted at tens of thousands of calls. They do not.

``soccerdata``'s ESPN reader already caches raw *scoreboard* JSON to
``Inputs/soccerdata_cache/Schedule_<slug>_<YYYYMMDD>.json``, and every completed
event in those files carries per-competitor statistics::

    events[].competitions[0].competitors[i].statistics[]
        -> {"name": "totalShots"|"shotsOnTarget"|"wonCorners", "displayValue": "14"}

``soccerdata`` does not surface them (``read_schedule`` parses fixture metadata
only; ``read_matchsheet`` hits a different endpoint), so we parse the cache
ourselves. Measured on the current cache: 20,429 events with all three stats,
across all five leagues, back to 2013/14 -- earlier than Understat's history
starts. Zero HTTP calls.

``backfill_from_matches`` covers whatever the cache is missing. It compares the
Understat match table against the files actually on disk, so a league-season is
fetched because it is absent, not because it appears on a list. ``fpp.reconcile``
drives it as part of checking that every league-season we should have is really
there.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from ..config import LEAGUES, LOWER_LEAGUES, SLUG_TO_KEY, season_sort_key
from ..paths import ESPN_MATCH_STATS, FIXTURES_DIR, SOCCERDATA_CACHE, STATS_DIR
from .espn import MAPPING_GAP_FILL, load_name_map, report_unmapped
from .retry import ESPN_SITE_API, get_json

# ESPN stat name -> our column stem
STAT_KEYS = {"totalShots": "shots", "shotsOnTarget": "sot", "wonCorners": "corners"}
REQUIRED = frozenset(STAT_KEYS)


# `load_name_map` and `MAPPING_GAP_FILL` live in `espn` now, next to the mapping
# they read. They were defined here as well, which meant two functions of the
# same name resolving ESPN club names two slightly different ways -- exactly the
# collision this codebase's naming rules exist to prevent.


def _parse_event(event: dict) -> dict | None:
    """One scoreboard event -> a flat record, or None if it lacks full stats."""
    comps = event.get("competitions") or [{}]
    competitors = comps[0].get("competitors") or []
    if len(competitors) != 2:
        return None

    rec: dict = {"espn_event_id": event.get("id"), "date": (event.get("date") or "")[:10]}
    for c in competitors:
        side = c.get("homeAway")
        if side not in ("home", "away"):
            return None
        stats = {s.get("name"): s.get("displayValue") for s in c.get("statistics") or []}
        if not REQUIRED.issubset(stats):
            return None
        rec[f"{side}_espn"] = (c.get("team") or {}).get("displayName")
        for espn_key, stem in STAT_KEYS.items():
            try:
                rec[f"{side}_{stem}"] = float(stats[espn_key])
            except (TypeError, ValueError):
                return None
        # Goals come from Understat for the top five divisions and are not needed
        # from here. Below them Understat does not reach, and `fpp.promoted` needs
        # goals both for its own ratio and as the strongest term in the xG proxy,
        # so the scoreline is carried through as well. NaN rather than a drop when
        # it is missing: a match with stats but no parsable score is still a
        # usable shots/SOT/corners row, and the top tier ignores the column.
        try:
            rec[f"{side}_goals"] = float(c.get("score"))
        except (TypeError, ValueError):
            rec[f"{side}_goals"] = float("nan")
    return rec


def _is_sentinel(rec: dict) -> bool:
    """Every count zero on *both* sides -- ESPN's empty-stats marker, not a result.

    Some older seasons come back with a structurally complete statistics block
    whose values are all "0": `_parse_event` sees `totalShots`, `shotsOnTarget`
    and `wonCorners` present and accepts the event. A side can genuinely take no
    corners, and one side can genuinely fail to register a shot -- those are kept.
    Neither side registering a shot, a shot on target *and* a corner does not
    happen in a played match.

    It matters because the zeros are indistinguishable from real observations once
    they reach the rolling priors: a NaN leaves an honest gap that the mask
    denominator skips, a 0.0 is a confident statement that the team took none.
    Hull's last ten cached matches are all of this form, which drove their whole
    2026-08-22 stat line to near zero.
    """
    # Judged on shots/SOT/corners only. Goals are deliberately excluded: a real
    # 0-0 in which both sides had shots is a legitimate result, and a sentinel is
    # recognised by the *stats* being uniformly absent, not by the scoreline.
    return all(rec[f"{side}_{stem}"] == 0
               for side in ("home", "away") for stem in STAT_KEYS.values())


def _is_degenerate(rec: dict) -> bool:
    """Every shot on target on both sides, and no corners at all -- a bad block.

    A second malformed layout, distinct from `_is_sentinel` and not caught by it
    because the values are not all zero. ESPN's 2019 and 2020 second-tier Spanish
    and French scoreboards return blocks like::

        Troyes 1-2 Clermont   shots 1/2   sot 1/2   corners 0/0
        Lens   2-0 Guingamp   shots 2/0   sot 2/0   corners 0/0

    -- shots identical to shots on target on both sides, corners uniformly zero,
    and the magnitudes tracking the scoreline. Whatever field is being read, it
    is not shots. Left in, it dragged the Ligue 2 benchmark to 1.63 shots per
    match, and every French club promoted since divided by it and hit the
    `promoted.CLIP_HI` ceiling -- recorded as "exceptional in the second tier"
    when the truth was "measured against nonsense".

    The two conditions only convict together. One side can have every shot on
    target, and a match can have no corners; both at once, on both sides, does
    not happen -- zero of the 21,200 top-flight matches in the cache match this,
    across all five leagues and fourteen seasons.
    """
    both_all_on_target = all(rec[f"{side}_shots"] == rec[f"{side}_sot"]
                             for side in ("home", "away"))
    no_corners = all(rec[f"{side}_corners"] == 0 for side in ("home", "away"))
    return both_all_on_target and no_corners


def parse_cache(cache_dir: Path | None = None) -> pd.DataFrame:
    """Parse every cached scoreboard file into one de-duplicated frame.

    Events are keyed on the ESPN event id, so the same fixture appearing in
    several cached date-files is counted once.
    """
    cache_dir = cache_dir or SOCCERDATA_CACHE
    files = sorted(cache_dir.glob("Schedule_*.json"))
    if not files:
        raise FileNotFoundError(f"No Schedule_*.json files in {cache_dir}")

    by_event: dict[str, dict] = {}
    n_sentinel = 0
    n_degenerate = 0
    for f in files:
        slug = f.name.split("Schedule_", 1)[1].split("_", 1)[0]
        if slug not in SLUG_TO_KEY:
            continue  # a competition we do not model
        try:
            payload = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for event in payload.get("events") or []:
            rec = _parse_event(event)
            if rec is None:
                continue
            if _is_sentinel(rec):
                n_sentinel += 1
                continue
            if _is_degenerate(rec):
                n_degenerate += 1
                continue
            rec["espn_slug"] = slug
            rec["league_key"] = SLUG_TO_KEY[slug]
            by_event[rec["espn_event_id"]] = rec

    if n_sentinel:
        print(f"  Dropped {n_sentinel:,} event(s) with all-zero stats "
              f"(ESPN empty-block sentinel, not a 0-0 result)")
    if n_degenerate:
        print(f"  Dropped {n_degenerate:,} event(s) with every shot on target and "
              f"no corners (malformed ESPN stat block, not a played match)")

    df = pd.DataFrame(list(by_event.values()))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["league_key", "date"]).reset_index(drop=True)


def to_understat_names(df: pd.DataFrame, name_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Attach Understat club names; report anything unmapped rather than dropping silently."""
    name_map = name_map if name_map is not None else load_name_map()
    out = df.copy()
    out["home_team"] = out["home_espn"].map(name_map)
    out["away_team"] = out["away_espn"].map(name_map)

    unmapped = (
        set(out.loc[out["home_team"].isna(), "home_espn"].dropna())
        | set(out.loc[out["away_team"].isna(), "away_espn"].dropna())
    )

    # Split by whether the club ever appears in a top-flight fixture, because the
    # two halves need opposite responses and one undifferentiated list of 94 hid
    # that. Understat covers the top five divisions only, so a club that plays
    # exclusively in the second tier has no Understat name to be mapped *to* --
    # it is unmapped permanently, by design, and since `build_match_stats` stopped
    # requiring both sides down there it costs nothing. Reporting it next to a
    # genuine miss invites someone to accept a fuzzy suggestion like
    # "Barnsley -> Burnley (0.52)", which would silently corrupt a real club.
    #
    # A club in a top-flight fixture is the opposite: it *will* have an Understat
    # name, and until it is mapped every one of its matches is dropped outright.
    # That is the list worth acting on, and it is normally empty or newly promoted.
    seen_top: set[str] = set()
    for side in ("home", "away"):
        top = out.loc[out["league_key"].isin(LEAGUES), f"{side}_espn"].dropna()
        seen_top.update(top)

    # Suggestion only -- see `report_unmapped`. Nothing here edits the mapping.
    report_unmapped(sorted(unmapped & seen_top), label="top flight, mapping needed")
    lower_only = sorted(unmapped - seen_top)
    if lower_only:
        print(f"  {len(lower_only)} second-tier-only club(s) unmapped -- expected, "
              f"Understat does not cover those divisions; their matches are still "
              f"kept for the mapped opponent")
    return out


def build_match_stats(cache_dir: Path | None = None, save: bool = True) -> pd.DataFrame:
    """Full parse -> mapped -> saved. This is the function Data Pull calls."""
    raw = parse_cache(cache_dir)
    print(f"  Parsed {len(raw):,} events with full shots/SOT/corners from the cache")
    df = to_understat_names(raw)

    # Top flight needs both sides: the row exists to be joined onto the Understat
    # match table, and half a fixture cannot be.
    #
    # Second tier is read one team at a time -- `promoted.lower_league_stats`
    # explodes each match into two team-perspective rows and drops the ones with
    # no name -- so requiring both sides there discards the *mapped* club's match
    # along with the unmapped opponent's. That is not a small loss: the second
    # tier is full of clubs that never come up and so were never worth a mapping
    # row, and every one of them was deleting a promotion-season match from a
    # club that did. Of 10,528 second-tier matches on disk, both-sides kept
    # 2,954; either-side keeps 8,371, and the median promotion season a promoted
    # club is measured over goes from 18 matches to 34 -- which is the difference
    # between `promoted.multipliers` shrinking 61% of its signal away and 26%.
    both = df["home_team"].notna() & df["away_team"].notna()
    either = df["home_team"].notna() | df["away_team"].notna()
    is_lower = df["league_key"].isin(LOWER_LEAGUES)
    df = df[both | (is_lower & either)].reset_index(drop=True)

    cols = [
        "espn_event_id", "espn_slug", "league_key", "date",
        "home_espn", "away_espn", "home_team", "away_team",
        "home_shots", "away_shots", "home_sot", "away_sot", "home_corners", "away_corners",
        "home_goals", "away_goals",
    ]
    df = df[cols]
    validate(df)

    if save:
        STATS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(ESPN_MATCH_STATS, index=False)
        print(f"  Saved {len(df):,} rows -> {ESPN_MATCH_STATS}")
    return df


def validate(df: pd.DataFrame) -> None:
    """Cheap integrity assertions -- counts are non-negative, fixtures unique."""
    stat_cols = [f"{side}_{stem}" for side in ("home", "away") for stem in STAT_KEYS.values()]
    for c in stat_cols:
        assert df[c].notna().all(), f"nulls in {c}"
        assert (df[c] >= 0).all(), f"negative counts in {c}"
    dupes = df.duplicated(subset=["espn_event_id"]).sum()
    assert dupes == 0, f"{dupes} duplicate espn_event_id rows"
    # Named rows keep the original check. It cannot be the only one any more:
    # second-tier rows may carry one unmapped side, and `duplicated` treats NaN
    # as equal to NaN, so two unrelated fixtures would compare equal on name.
    named = df[df["home_team"].notna() & df["away_team"].notna()]
    dupes = named.duplicated(subset=["date", "home_team", "away_team"]).sum()
    assert dupes == 0, f"{dupes} duplicate (date, home, away) rows"


# --- Gap filling ----------------------------------------------------------

# There was a `KNOWN_MISSING` tuple here: the four league-seasons found absent on
# the day someone last checked. It was the default target list, so the backfill
# restored those four and reported success no matter what else had gone missing
# since. Deleted rather than demoted to a fallback -- a snapshot of a moving
# quantity is the bug, and keeping one around invites its reuse. `missing_dates`
# re-derives the answer from the cache on every call.

# A full backfill from a cold cache is ~10,000 requests. ESPN has no published
# rate limit, but firing that many back to back is asking to be throttled, and
# `retry.get_json` only backs off *after* a failure. Applied here rather than in
# `get_json` so the short existing pull paths stay at full speed.
SCOREBOARD_DELAY = 0.5  # seconds between scoreboard requests


def _expanded_dates(sub: pd.DataFrame) -> set[str]:
    """Match dates as ``YYYYMMDD``, widened by +/- 1 day.

    ESPN dates are UTC kickoff and can straddle midnight relative to Understat's,
    so a match can land in either neighbour's scoreboard file.
    """
    dates: set[str] = set()
    for d in pd.to_datetime(sub["date"]).dt.normalize().unique():
        ts = pd.Timestamp(d)
        for shift in (-1, 0, 1):
            dates.add((ts + pd.Timedelta(days=shift)).strftime("%Y%m%d"))
    return dates


def missing_dates(
    matches: pd.DataFrame,
    cache_dir: Path | None = None,
    *,
    leagues: list[str] | None = None,
    seasons: list[str] | None = None,
) -> dict[tuple[str, str], set[str]]:
    """``(league_key, season)`` -> the date files not yet on disk.

    Every combination present in ``matches`` is considered, so this reports what
    is *actually* absent rather than what someone remembered to list. Combinations
    already complete are omitted entirely.
    """
    cache_dir = cache_dir or SOCCERDATA_CACHE
    out: dict[tuple[str, str], set[str]] = {}

    for (league_key, season), sub in matches.groupby(["league_key", "season"], sort=True):
        if leagues and league_key not in leagues:
            continue
        if seasons and season not in seasons:
            continue
        slug = LEAGUES[league_key].espn_slug
        absent = {
            d for d in _expanded_dates(sub)
            if not (cache_dir / f"Schedule_{slug}_{d}.json").exists()
        }
        if absent:
            out[(league_key, season)] = absent
    return out


def season_date_range(season: str) -> set[str]:
    """Every date a season could have a match on, as ``YYYYMMDD``.

    The second tiers have no Understat coverage, so unlike `missing_dates` there
    is no match table to read their fixture dates off -- the thing that drives
    the top-tier backfill does not exist for them. Enumerating the season window
    is the honest substitute: a date with no matches returns an empty event list,
    which caches like any other and is never refetched.

    August to May, which covers every European second-tier season including the
    play-off tail, at about 300 dates a season.
    """
    start_year = season_sort_key(season)
    lo = pd.Timestamp(year=start_year, month=8, day=1)
    hi = pd.Timestamp(year=start_year + 1, month=5, day=31)
    return {d.strftime("%Y%m%d") for d in pd.date_range(lo, hi, freq="D")}


def backfill_lower_leagues(seasons: list[str], *, leagues: list[str] | None = None,
                           cache_dir: Path | None = None, dry_run: bool = True,
                           max_calls: int | None = None) -> int:
    """Cache ESPN scoreboards for the second tiers. Resumable; dry by default.

    Deliberately dry by default. A cold run is roughly ``len(leagues) x
    len(seasons) x 300`` requests against a third party at
    ``SCOREBOARD_DELAY`` apart -- five leagues over seven seasons is ~10,500
    calls and about an hour and a half -- so it is opted into, not stumbled into.
    Existing files are never refetched, so an interrupted run resumes for free.
    """
    cache_dir = cache_dir or SOCCERDATA_CACHE
    keys = leagues or list(LOWER_LEAGUES)
    plan: dict[str, set[str]] = {}
    for key in keys:
        slug = LOWER_LEAGUES[key].espn_slug
        want: set[str] = set()
        for season in seasons:
            want |= season_date_range(season)
        plan[key] = {d for d in want
                     if not (cache_dir / f"Schedule_{slug}_{d}.json").exists()}

    outstanding = sum(len(v) for v in plan.values())
    if dry_run:
        mins = outstanding * SCOREBOARD_DELAY / 60
        print(f"  {outstanding:,} date file(s) to fetch across {len(keys)} league(s):")
        for key, dates in plan.items():
            print(f"    {key:<7} {LOWER_LEAGUES[key].espn_slug:<6} {len(dates):,}")
        print(f"  dry run -- nothing fetched. A real run is ~{mins:.0f} min.")
        return outstanding

    total = 0
    for key, dates in plan.items():
        if not dates:
            continue
        room = None if max_calls is None else max(0, max_calls - total)
        if room == 0:
            break
        n = fetch_scoreboard_dates(LOWER_LEAGUES[key].espn_slug, dates,
                                   cache_dir, limit=room)
        total += n
        print(f"  {key}: {n:,} new files")
    return total


def fetch_scoreboard_dates(
    slug: str,
    dates: set[str],
    cache_dir: Path | None = None,
    *,
    delay: float | None = None,
    limit: int | None = None,
) -> int:
    """Fetch and cache one scoreboard file per date. ``dates`` are ``YYYYMMDD``.

    Written in soccerdata's own cache format so both ``parse_cache`` and
    ``soccerdata`` itself pick the files up. Existing files are never refetched,
    which is what makes an interrupted backfill safe to resume.
    """
    # Read at call time, not in the signature: a default argument is evaluated
    # once at import, so `SCOREBOARD_DELAY` could never be adjusted afterwards.
    delay = SCOREBOARD_DELAY if delay is None else delay
    cache_dir = cache_dir or SOCCERDATA_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for d in sorted(dates):
        if limit is not None and written >= limit:
            break
        out = cache_dir / f"Schedule_{slug}_{d}.json"
        if out.exists():
            continue
        payload = get_json(f"{ESPN_SITE_API}/{slug}/scoreboard?dates={d}")
        out.write_text(json.dumps(payload))
        written += 1
        if delay:
            time.sleep(delay)
    return written


# Kickoff to a settled scoreboard. 90 minutes plus half time and stoppage runs a
# little over two hours, so this is tight: a delayed start or a long stoppage can
# still be in play when the window opens. `refresh_results` will simply refetch
# on the next run if it catches one mid-match.
FINAL_LAG = pd.Timedelta(hours=2)


def fixture_kickoffs(leagues: list[str] | None = None) -> dict[tuple[str, str], pd.Timestamp]:
    """``(league_key, YYYYMMDD)`` -> the last kickoff we know of on that date.

    Read from our own fixture files rather than from Understat, and that is the
    whole point -- see `current_result_dates`.
    """
    out: dict[tuple[str, str], pd.Timestamp] = {}
    for key, lg in LEAGUES.items():
        if leagues and key not in leagues:
            continue
        path = FIXTURES_DIR / f"{lg.fixtures_stem}_fixtures.csv"
        if not path.exists():
            continue
        fx = pd.read_csv(path)
        if "Date" not in fx.columns:
            continue
        start = fx.get("Start Time (UTC)", pd.Series([""] * len(fx)))
        when = pd.to_datetime(fx["Date"].astype(str) + " " + start.fillna("").astype(str),
                              errors="coerce", utc=True)
        when = when.fillna(pd.to_datetime(fx["Date"], errors="coerce", utc=True))
        for day, grp in when.dropna().groupby(when.dt.strftime("%Y%m%d")):
            k = (key, str(day))
            out[k] = max(grp.max(), out.get(k, grp.max()))
    return out


def settled_after(league_key: str, day: str, kickoffs: dict) -> float:
    """The moment a scoreboard for this date can be believed, as a timestamp.

    Three hours after the day's last kickoff where we know it, and otherwise
    midnight following -- the original rule, kept as the fallback for a date with
    no fixture row. The kickoff form is what makes a match-night run work: waiting
    for midnight meant the results of games that finished at nine could not be
    banked until the next day.
    """
    ko = kickoffs.get((league_key, day))
    if ko is not None:
        return float((ko + FINAL_LAG).timestamp())
    return float((pd.Timestamp(day) + pd.Timedelta(days=1)).timestamp())


def refresh_results(dates_by_league: dict[str, set[str]], cache_dir: Path | None = None,
                    *, rebuild: bool = True, now: pd.Timestamp | None = None) -> int:
    """Refetch finished scoreboards for the dates given, and rebuild the stats.

    ESPN only. Everything a proposition settles against -- goals, shots, shots on
    target, corners -- comes from this scoreboard, so results need nothing from
    Understat, which supplies xG for the model and nothing this reads.

    That distinction is the fix for a real failure: the results path used to be
    driven off Understat's published match dates, so on a night when Understat
    was seven matches into a 182-match season, every fixture played that day was
    invisible and nothing settled -- with ESPN's final stats sitting there
    unasked-for.

    A date is only fetched once its last kickoff is `FINAL_LAG` behind us, and a
    file captured before then is deleted first, because `fetch_scoreboard_dates`
    never refetches a file that exists. Both halves matter: without the delete a
    pre-match capture is kept forever, and without the wait a live fixture is
    banked as a final score.

    `now` is injectable because the whole behaviour turns on it, and a test that
    has to patch the clock out of pandas to check "does this wait" is a test
    nobody trusts.
    """
    cache_dir = cache_dir or SOCCERDATA_CACHE
    kickoffs = fixture_kickoffs()
    now = (pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)).timestamp()

    stale: dict[str, set[str]] = {}
    waiting: list[str] = []
    for league_key, days in dates_by_league.items():
        if league_key not in LEAGUES:
            continue
        slug = LEAGUES[league_key].espn_slug
        for day in sorted(days):
            if now < settled_after(league_key, day, kickoffs):
                waiting.append(f"{league_key} {day}")
                continue
            path = cache_dir / f"Schedule_{slug}_{day}.json"
            if path.exists() and path.stat().st_mtime >= settled_after(league_key, day, kickoffs):
                continue                      # already captured after full time
            if path.exists():
                path.unlink()                 # captured before kickoff or mid-match
            stale.setdefault(slug, set()).add(day)

    if waiting:
        print(f"  {len(waiting)} date(s) not finished yet: {', '.join(waiting[:6])}"
              + (" ..." if len(waiting) > 6 else ""))
    fetched = sum(fetch_scoreboard_dates(slug, days, cache_dir)
                  for slug, days in stale.items())
    print(f"  fetched {fetched} scoreboard file(s) from ESPN")
    if fetched and rebuild:
        build_match_stats(save=True)
    return fetched


def backfill_from_matches(
    matches: pd.DataFrame,
    cache_dir: Path | None = None,
    *,
    dry_run: bool = False,
    max_calls: int | None = None,
    leagues: list[str] | None = None,
    seasons: list[str] | None = None,
) -> int:
    """Fill cache gaps using the match dates we already know from Understat.

    We know exactly which dates matter -- they are in the Understat match table --
    so we fetch one scoreboard per *date* (~150-200 per season) rather than one
    summary per *fixture* (380 per season). Roughly half the calls, and no event
    enumeration needed.

    Targets are always derived from the cache by ``missing_dates``: every
    ``(league_key, season)`` in ``matches`` whose date files are not all present.
    There is deliberately no way to pass a fixed list.

    ``dry_run`` reports what would be fetched without fetching it; from a cold
    cache the real run is ~10,000 requests, so it is worth looking first.
    ``max_calls`` caps a single run -- existing files are never refetched, so
    stopping early and resuming later loses nothing.

    ``matches`` needs columns ``league_key``, ``season``, ``date``.
    Returns the number of new cache files written (or that would be, if ``dry_run``).
    """
    gaps = missing_dates(matches, cache_dir, leagues=leagues, seasons=seasons)

    if not gaps:
        print("  nothing to fetch -- every target is already complete")
        return 0

    outstanding = sum(len(d) for d in gaps.values())
    if dry_run:
        print(f"  {len(gaps)} league-season(s) incomplete, {outstanding:,} date file(s) to fetch:")
        for (league_key, season), dates in sorted(gaps.items()):
            print(f"    {league_key} {season}: {len(dates):,}")
        mins = outstanding * SCOREBOARD_DELAY / 60
        print(f"  dry run -- nothing fetched. A real run is ~{outstanding:,} requests (~{mins:.0f} min).")
        return outstanding

    total = 0
    for (league_key, season), dates in sorted(gaps.items()):
        if max_calls is not None and total >= max_calls:
            break
        slug = LEAGUES[league_key].espn_slug
        remaining = None if max_calls is None else max_calls - total
        written = fetch_scoreboard_dates(slug, dates, cache_dir, limit=remaining)
        total += written
        print(f"  {league_key} {season}: {len(dates):,} dates missing, {written:,} new files")

    if max_calls is not None and total >= max_calls:
        print(f"  stopped at max_calls={max_calls:,}; {outstanding - total:,} date file(s) still outstanding")
    return total


__all__ = [
    "STAT_KEYS",
    "SCOREBOARD_DELAY",
    "load_name_map",
    "parse_cache",
    "to_understat_names",
    "build_match_stats",
    "validate",
    "missing_dates",
    "fixture_kickoffs",
    "settled_after",
    "refresh_results",
    "FINAL_LAG",
    "fetch_scoreboard_dates",
    "backfill_from_matches",
]
