"""Understat match data: goals, xG, npxG, points.

The pull itself is unchanged from the original Data Pull notebook (cells 6 and 8,
deduplicated into one function). ``load_matches`` is the read side -- it stitches
the five leagues' previous-season and current-season CSVs into one frame.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from ..config import LEAGUES, canonical_team, current_season, historic_seasons
from ..paths import INPUTS, SOCCERDATA_CACHE
from .retry import with_retry

# Columns we need out of Understat's team-match stats.
REQUIRED_COLS = [
    "date", "home_team", "away_team",
    "home_goals", "away_goals",
    "home_xg", "away_xg",
    "home_np_xg", "away_np_xg",
    "season",
]


# --- What "real data" means -----------------------------------------------
#
# One definition, used by the cache guards here *and* by the reconciler. Three
# separate checks that each decide for themselves what "good" means is how a
# frozen season goes unnoticed for months -- see `expected_matches`.

SHAPE_COLS = ("date", "home_team", "away_team")


def is_usable(df: pd.DataFrame) -> bool:
    """Real data: it has rows, and the columns everything downstream keys on.

    Deliberately stronger than "the file exists". An empty or column-less frame
    reads back as a perfectly good file, which is exactly how one gets cached and
    then served forever.
    """
    return not df.empty and all(c in df.columns for c in SHAPE_COLS)


def expected_matches(df: pd.DataFrame) -> int:
    """How many matches a complete season of this league holds: ``T * (T - 1)``.

    Derived from the teams present rather than configured, so a 20-team league
    gives 380 and an 18-team one 306 with nothing to keep up to date -- Ligue 1's
    20 -> 18 change needs no edit here. Returns 0 when the frame is unusable,
    since a shortfall against an unknown target is meaningless.
    """
    if not is_usable(df):
        return 0
    teams = set(df["home_team"].dropna()) | set(df["away_team"].dropna())
    return len(teams) * (len(teams) - 1)


def season_shortfall(df: pd.DataFrame) -> int:
    """Matches missing from a complete season, floored at 0.

    This is the check that would have caught last season freezing at 370 of 380
    in four of the five leagues.
    """
    return max(0, expected_matches(df) - len(df))


def season_cache_dir(league_key: str, season: str):
    """Where a season's ``_temp`` CSV lives -- current seasons sit apart from history."""
    sub = "Current Season" if season == current_season() else "Previous Seasons"
    return INPUTS / league_key / sub / "_temp"


# soccerdata caches Understat's league index -- the `/getStatData` response that
# says which (league, season) pairs exist at all -- to `leagues.json`, and reads
# it back with `no_cache=False` forever. It never expires on its own.
#
# `read_seasons` derives the available season ids from that file, so a season
# that began after it was last written is invisible: `read_seasons` returns an
# empty frame, `read_team_match_stats` returns a (3, 0) frame with no columns,
# and every guard below correctly refuses to cache the result. The pipeline then
# reports "no usable rows" all season while the data sits on Understat, reachable
# by URL. That is exactly what happened to 2026/2027 -- the index was written on
# 17 May 2026, listed EPL months up to (2026, 5), and pinned the newest visible
# season to 2025/2026 for three months.
#
# Expire it on age rather than deleting it every time: the index is one request,
# but it is also the thing every league-season read starts from, and refetching
# it on each of a hundred historical pulls is waste. A day is the right cadence
# for a file whose only volatile content is "has the new season started yet".
LEAGUE_INDEX_MAX_AGE_DAYS = 1.0


def expire_league_index(max_age_days: float = LEAGUE_INDEX_MAX_AGE_DAYS) -> bool:
    """Drop soccerdata's cached league index if it is stale. True if removed.

    Removing rather than refetching in place: soccerdata rewrites the file on its
    next read, so this hands the refresh back to the library instead of reaching
    into a private method whose name is not ours to depend on.
    """
    import time

    idx = SOCCERDATA_CACHE / "leagues.json"
    if not idx.exists():
        return False
    age_days = (time.time() - idx.stat().st_mtime) / 86400
    if age_days < max_age_days:
        return False
    idx.unlink()
    # Only narrate an age-triggered drop. A forced one (`max_age_days=0`) has a
    # caller that explains itself better than an age can, and "was 0.0 days old"
    # next to its message reads like a contradiction.
    if max_age_days > 0:
        print(f"  Understat league index was {age_days:.1f} days old -- refetching "
              f"(a season that started since it was written is invisible until this)")
    return True


def read_cached_season(league_key: str, season: str, *, out_dir=None) -> pd.DataFrame | None:
    """The cached per-season frame, or None if there isn't a usable one.

    Read-only and never fetches -- this is what lets the reconciler audit the
    whole grid without touching the network.
    """
    out_dir = out_dir or season_cache_dir(league_key, season)
    cached = out_dir / f"{season.replace('/', '_')}.csv"
    if not cached.exists():
        return None
    try:
        df = pd.read_csv(cached, parse_dates=["date"])
    except (ValueError, OSError, pd.errors.EmptyDataError):
        return None  # empty, malformed, or missing the date column
    return df if is_usable(df) else None


def pull_season(league_key: str, season: str, *, out_dir=None, force: bool = False) -> pd.DataFrame:
    """Fetch one league-season from Understat, caching to a per-season CSV.

    Mirrors the notebook's ``_temp/`` cache: if the season file exists we read it
    rather than refetching. That is the one true resume point in the old pipeline
    and it is worth keeping.

    ``force`` refetches over a usable cache file. Without it the cache is a
    one-way door, which is how the *current* season used to freeze at whatever
    was captured the first time it was pulled -- it never expires on its own,
    because "the file exists" stays true all season.

    An empty or column-less result is never cached, so a barren early-season
    fetch cannot pin the season to empty.
    """
    import soccerdata as sd

    lg = LEAGUES[league_key]
    out_dir = out_dir or season_cache_dir(league_key, season)
    out_dir.mkdir(parents=True, exist_ok=True)
    cached = out_dir / f"{season.replace('/', '_')}.csv"

    if not force:
        prev = read_cached_season(league_key, season, out_dir=out_dir)
        if prev is not None:
            return prev
        if cached.exists():
            print(f"  {league_key} {season}: cached file unusable -- refetching")

    # Only for the current season: a historical season's existence was settled
    # long ago and no index refresh can change it, so the hundred-odd backfill
    # pulls stay on the cached index.
    is_current = season == current_season()
    if is_current:
        expire_league_index()

    def fetch():
        us = with_retry(sd.Understat, leagues=lg.understat, seasons=season,
                        data_dir=SOCCERDATA_CACHE)
        return with_retry(us.read_team_match_stats)

    df = fetch()

    # Zero columns is the exact shape `read_seasons` produces when it matched
    # nothing -- the season is absent from Understat's index. Read it here,
    # before the `season` column below is added, or the test can never be true.
    not_listed = len(df.columns) == 0

    # Self-correct rather than wait for the age timer. The timer alone leaves a
    # gap on exactly the day that matters: refresh the index at 10:00 on the
    # opening day, before Understat has published anything for the new month, and
    # every run for the next 24 hours is pinned to an index that predates the
    # season. Here the miss *is* the trigger -- one forced refresh and one retry,
    # only when the current season came back unlisted, so a genuinely unstarted
    # season (no columns missing, just no rows) costs nothing.
    if is_current and not_listed and expire_league_index(0):
        print(f"  {league_key} {season}: not in Understat's index -- forcing a "
              f"refresh and retrying once")
        df = fetch()
        not_listed = len(df.columns) == 0

    df = df.reset_index() if df.index.nlevels > 1 else df
    if "season" not in df.columns:
        df["season"] = season

    if not is_usable(df):
        # Two very different causes, and the original message covered both with a
        # shrug: "no usable rows". Which one it is decides whether to act or wait.
        why = ("still not in Understat's index after a forced refresh -- the "
               "season may not be published yet"
               if not_listed else
               "the season is listed but returned no matches -- none played yet?")
        print(f"  {league_key} {season}: no usable rows ({why}) -- not cached, "
              f"will refetch next run")
        return df

    df.to_csv(cached, index=False)
    return df


def refresh_league(
    league_key: str,
    seasons: list[str] | None = None,
    current: bool = False,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Pull and write the combined previous-seasons (or current-season) CSV.

    ``force`` refetches every season rather than trusting the per-season cache.
    The reconciler uses it to repair a season it has judged incomplete.
    """
    lg = LEAGUES[league_key]
    if current:
        label = current_season()
        df = pull_season(league_key, label, force=force)
        dest = INPUTS / league_key / "Current Season" / f"{lg.prefix} Current Season.csv"
    else:
        label = "history"
        frames = [pull_season(league_key, s, force=force) for s in (seasons or historic_seasons())]
        df = pd.concat(frames, ignore_index=True)
        dest = INPUTS / league_key / "Previous Seasons" / f"{lg.prefix} Previous Seasons.csv"

    # Same reasoning as the per-season cache, one level up. `_read_one` requires
    # REQUIRED_COLS and raises without them, so overwriting a good combined CSV
    # with an empty early-season pull would break `load_matches` for all five
    # leagues at once. Keeping the last good file degrades this to "no new data".
    if not is_usable(df):
        kept = "kept existing file" if dest.exists() else "nothing written"
        print(f"  {league_key} {label}: no usable rows -- {dest.name} {kept}")
        return df

    dest.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(dest, index=False)
    return df


# --- Read side ------------------------------------------------------------


def _read_one(league_key: str) -> pd.DataFrame:
    lg = LEAGUES[league_key]
    frames = []
    for sub in ("Previous Seasons", "Current Season"):
        p = INPUTS / league_key / sub / f"{lg.prefix} {sub}.csv"
        if not p.exists():
            print(f"  missing {p} -- skipped")
            continue
        frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(f"No Understat CSVs found for {league_key}")

    df = pd.concat(frames, ignore_index=True)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{league_key}: Understat CSV missing columns {missing}")

    df = df[REQUIRED_COLS].copy()
    df["league_key"] = league_key
    df["league"] = lg.display
    return df


def load_matches(league_keys: list[str] | None = None) -> pd.DataFrame:
    """All five leagues' matches in one frame, typed, played-only, canonical names.

    One row per match. This is the base the clean step builds on.
    """
    keys = league_keys or list(LEAGUES)
    df = pd.concat([_read_one(k) for k in keys], ignore_index=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    for c in ("home_goals", "away_goals", "home_xg", "away_xg", "home_np_xg", "away_np_xg"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Unplayed fixtures carry no result -- drop them here, not downstream.
    before = len(df)
    df = df.dropna(subset=["date", "home_team", "away_team", "season", "home_goals", "away_goals"])
    if before != len(df):
        print(f"  dropped {before - len(df):,} unplayed/incomplete rows")

    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df["season"] = df["season"].astype(str)
    for c in ("home_team", "away_team"):
        df[c] = df[c].map(canonical_team)

    return df.sort_values("date").reset_index(drop=True)


@lru_cache(maxsize=1)
def team_names_by_league() -> dict[str, frozenset[str]]:
    """``{league_key: {club names}}`` from Understat's match history, canonicalised.

    Per league because the club mapping is built per league: matching a Spanish
    club against the pooled pool of all five would let a fuzzy score pair it with
    a similarly-spelled German one.

    Never raises. This backs a warning path and the mapping builder, and a
    warning that can itself fail is worse than no suggestion -- a missing file
    just narrows the pool.
    """
    out: dict[str, frozenset[str]] = {}
    for key, lg in LEAGUES.items():
        names: set[str] = set()
        for sub in ("Previous Seasons", "Current Season"):
            path = INPUTS / key / sub / f"{lg.prefix} {sub}.csv"
            try:
                df = pd.read_csv(path, usecols=["home_team", "away_team"])
            except (OSError, ValueError):
                continue  # absent, unreadable, or missing those columns
            names.update(df["home_team"].dropna())
            names.update(df["away_team"].dropna())
        out[key] = frozenset(canonical_team(n) for n in names)
    return out


@lru_cache(maxsize=1)
def known_team_names() -> frozenset[str]:
    """Every club name in Understat's match history, across all leagues.

    The candidate pool for name suggestions. Deliberately wider than the mapping
    file: a newly promoted club is missing from the mapping exactly when a
    suggestion would be most useful, but it is already in the match data.
    """
    return frozenset().union(*team_names_by_league().values())


__all__ = [
    "REQUIRED_COLS", "SHAPE_COLS",
    "is_usable", "expected_matches", "season_shortfall",
    "season_cache_dir", "read_cached_season",
    "expire_league_index", "LEAGUE_INDEX_MAX_AGE_DAYS",
    "pull_season", "refresh_league", "load_matches",
    "known_team_names", "team_names_by_league",
]
