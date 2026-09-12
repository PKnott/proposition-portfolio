"""ESPN: upcoming fixtures, club-name mapping, team IDs, all-competition history.

Why the mapping is keyed on the numeric id
------------------------------------------
ESPN returns a club's name from two endpoints and does not promise they agree.
`ESPN_Team_IDs.csv` and the generated `Club_Mapping_All.csv` were both built from
`/teams`; `read_schedule` -- and therefore `pull_fixtures` -- reads `/scoreboard`.
For one club those diverged: id **90** is ``Deportivo La Coruna`` in the mapping
and ``Deportivo`` in every current scoreboard file, so the row was correct and
unreachable, and the club silently lost eight seasons of history.

Sweeping the whole scoreboard cache: 173 club ids, ``name`` equal to
``displayName`` for every one, no name used by two ids, and exactly one id --
90 -- that has ever carried two names. So the name is *nearly* a key, which is
the worst kind: it works until it does not, and it fails silently.

The id is the real key. Both endpoints carry it, it never changes, and
`espn_id_by_name` resolves whichever name a caller happens to hold back to it
from files already on disk.

Ported from the original Data Pull notebook (cells 10, 11, 13, 14) with three
fixes carried over from the plan:

* the raw-``requests`` calls move to ``site.web.api.espn.com``; the old host
  ``site.api.espn.com`` now returns 403
* a pooled session with browser headers replaces per-call ``requests.get``
* checkpoints are actually *read back*, so a long pull resumes instead of
  restarting (the old ``_checkpoint_*.csv`` were written every 50 calls and never
  read)
"""

from __future__ import annotations

import json
import time
import unicodedata
from collections.abc import Iterable
from difflib import SequenceMatcher
from functools import lru_cache

import pandas as pd

from ..config import (LEAGUES, SLUG_TO_KEY, canonical_team, current_season_label,
                      season_sort_key)
from ..paths import CLUB_MAPPING_ALL, ESPN_TEAM_IDS, FIXTURES_DIR, MAPPING_DIR, SOCCERDATA_CACHE, all_comp_fixtures
from .retry import ESPN_SITE_API, get_json
from .understat import known_team_names, team_names_by_league

# --- Name normalisation (unchanged behaviour from the notebook) -----------

STOPWORDS = {"fc", "cf", "ac", "afc", "as", "ssc", "sc", "sv", "vfb", "vfl", "tsg",
             "fsv", "rc", "stade", "1", "1846", "calcio", "2013", "1913"}
TOKEN_MAP = {"monchengladbach": "mgladbach", "rasenballsport": "rb",
             "internazionale": "inter", "saint": "st", "hellas": ""}
OVERRIDES = {"AFC Bournemouth": "Bournemouth", "Brighton & Hove Albion": "Brighton"}


def normalise_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = s.lower().replace("&", " and ")
    for ch in ".-'":
        s = s.replace(ch, "")
    tokens = [TOKEN_MAP.get(t, t) for t in s.split()]
    return " ".join(t for t in tokens if t and t not in STOPWORDS)


def similarity(a: str, b: str) -> float:
    an, bn = normalise_name(a), normalise_name(b)
    seq = SequenceMatcher(None, an, bn).ratio()
    at, bt = set(an.split()), set(bn.split())
    tok = len(at & bt) / len(at | bt) if (at or bt) else 0.0
    return 0.65 * seq + 0.35 * tok


def best_match(name: str, pool: Iterable[str]) -> tuple[str, float] | None:
    """Closest name in ``pool`` by ``similarity``, or None if the pool is empty."""
    scored = [(candidate, similarity(name, candidate)) for candidate in pool]
    return max(scored, key=lambda pair: pair[1]) if scored else None


def report_unmapped(
    unmapped: Iterable[str],
    *,
    pool: Iterable[str] | None = None,
    cap: int = 10,
    indent: str = "  ",
    label: str | None = None,
) -> None:
    """Print each unmapped club name alongside its closest known Understat name.

    A suggestion, never an action. Nothing here writes to ``Club_Mapping_All.csv``
    or ``MAPPING_GAP_FILL`` at any similarity score, and it must stay that way --
    a fuzzy match is a starting point for a person, not a decision. Confirming and
    adding the mapping is still manual; this only removes the guesswork of finding
    the right name from scratch.
    """
    names = sorted(unmapped)
    if not names:
        return
    if pool is None:
        pool = known_team_names()

    head = f"{label}: " if label else ""
    print(f"{indent}{head}{len(names)} unmapped ESPN club name(s):")
    for name in names[:cap]:
        guess = best_match(name, pool)
        if guess is None:
            print(f"{indent}  {name}: unmapped -- no known names to compare against")
        else:
            candidate, score = guess
            print(f"{indent}  {name}: unmapped -- best guess {candidate!r} (similarity {score:.2f})")
    if len(names) > cap:
        print(f"{indent}  ... and {len(names) - cap} more")


# --- Club identity, keyed on the ESPN id ----------------------------------

# Below this a fuzzy pairing is not worth writing down even for review. Carried
# over from the original mapping build, where it was the same number.
MIN_FUZZY_SCORE = 0.35

MAPPING_COLUMNS = ("league", "espn_id", "espn_team", "understat_team", "score", "source")

# Clubs no fuzzy pass can pair, because they sit outside Understat's history in
# our window: they appear in the ESPN cache with no Understat name to match
# against. Used twice, both additively -- `build_club_mapping` seeds rows for them
# so they stop being reported as unresolved every run, and `load_name_map` fills
# them in for a mapping file written before they were listed. Never overriding:
# ESPN's "Parma" must stay pointed at Understat's "Parma Calcio 1913", which
# `canonical_team` then folds back to "Parma".
#
# Two directions of the same gap. Carpi and the rest *predate* our window. A club
# newly promoted from the second tier is the mirror case: Understat has nothing
# for it until it plays a top-flight match, so the fuzzy pass has no name to pair
# with, and until then its second-tier record -- the thing `fpp.promoted` needs to
# scale its cold-start seed -- is dropped for want of a name.
#
# These entries are keyed on the *name*, and ESPN can serve two spellings of one
# id from two endpoints -- the scoreboard JSON and the schedule reader behind
# `pull_fixtures` -- so a promoted club needs both listed. They are meant to
# retire: the moment Understat publishes the club, put its real name in
# `Club_Mapping_All.csv` as an `override` row and delete the entry here. Leaving a
# retired one would be worse than useless, because `build_club_mapping` seeds from
# this dict and would write the invented name back the next time the mapping was
# rebuilt from scratch.
#
# SV Elversberg was the worked example: promoted to the Bundesliga for 2026/27 and
# carried here as "SV Elversberg" until it played a top-flight match. Understat now
# lists it as "Elversberg", so id 10388 has an override row and the entry is gone.
# `canonical_team` folds the old invented spelling onto the real one, which is what
# settles the rows already written under it.
MAPPING_GAP_FILL = {
    "Carpi": "Carpi",        # Serie A 2015/16 -- 38 fixtures
    "Catania": "Catania",    # pre-2014/15 only
    "Livorno": "Livorno",    # pre-2014/15 only
}


@lru_cache(maxsize=4)
def _scoreboard_teams(cache_dir=None) -> tuple[dict, dict]:
    """Parse the scoreboard cache once: ``({name: id}, {league_key: {id: name}})``.

    Reads files `soccerdata` has already written, so this costs no HTTP calls at
    all. The name index folds in every name an id has ever carried, which is what
    lets a mapping row written under an old name still resolve.
    """
    from ..paths import SOCCERDATA_CACHE as _default

    cache_dir = cache_dir or _default
    by_name: dict[str, str] = {}
    # Keyed on what `SLUG_TO_KEY` can actually return, not on `LEAGUES`. The two
    # were the same set until the second tier was added to `SLUG_TO_KEY` so that
    # `parse_cache` would stop skipping its files; the loop below resolves slugs
    # through it, so a `Schedule_eng.2_*.json` in the cache then indexed a key
    # that was never seeded and this raised `KeyError: 'Prem2'`. `build_club_mapping`
    # iterates `LEAGUES` and reads with `.get`, so the extra keys are collected
    # and ignored -- no second-tier club gains a mapping row from this.
    by_league: dict[str, dict[str, str]] = {k: {} for k in set(SLUG_TO_KEY.values())}
    for path in sorted(cache_dir.glob("Schedule_*.json")):
        slug = path.name.split("Schedule_", 1)[1].split("_", 1)[0]
        key = SLUG_TO_KEY.get(slug)
        if key is None:
            continue  # a competition we do not model
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for event in payload.get("events") or []:
            for c in (event.get("competitions") or [{}])[0].get("competitors") or []:
                team = c.get("team") or {}
                tid = team.get("id")
                if not tid:
                    continue
                tid = str(tid)
                for field in ("name", "displayName"):
                    if team.get(field):
                        by_name[team[field]] = tid
                        # Overwrite rather than setdefault: files sort by date, so
                        # the last one wins and the mapping file carries the name
                        # the club goes by *now*. Every earlier name still resolves
                        # through `by_name`, which is what makes the id the key.
                        by_league[key][tid] = team[field]
    return by_name, by_league


def espn_id_by_name(cache_dir=None) -> dict[str, str]:
    """``{any ESPN name ever seen: team id}``, from the scoreboard cache."""
    return dict(_scoreboard_teams(cache_dir)[0])


def build_club_mapping(cache_dir=None, out_path=None) -> pd.DataFrame:
    """Regenerate ``Club_Mapping_All.csv``, keyed on the ESPN team id.

    Ported out of ``Notebooks/Archive/Data Pull.ipynb`` cell 10, which is where it
    had been left: nothing in `fpp` regenerated the file, `scoreboard.load_name_map`
    raised pointing at "the club-mapping step in Data Pull" that no longer
    existed, and the mapping therefore decayed a little at every promotion window.

    Three differences from the archived version, all of them the point:

    * **Keyed on the id**, with the name kept alongside for a human to read.
    * **No HTTP.** ESPN clubs come from the scoreboard cache already on disk,
      Understat clubs from the season CSVs, so this is a local rebuild.
    * **Existing rows are carried forward, not re-derived.** Every hand-checked
      pairing in the current file survives a rebuild; only ids with no row yet are
      fuzzy-matched. A regeneration that could silently re-decide a mapping
      somebody had already fixed would be worse than not having one.

    A fuzzy pairing is written with its score and ``source="fuzzy"`` so it can be
    audited, exactly as the current file records them. Anything below
    `MIN_FUZZY_SCORE`, or with no Understat name left to pair with, is reported
    rather than written -- `report_unmapped`'s rule that a guess is a starting
    point for a person, not a decision.
    """
    out_path = out_path or CLUB_MAPPING_ALL
    _, by_league = _scoreboard_teams(cache_dir)
    understat = team_names_by_league()

    existing = pd.DataFrame(columns=list(MAPPING_COLUMNS))
    if CLUB_MAPPING_ALL.exists():
        existing = pd.read_csv(CLUB_MAPPING_ALL, dtype={"espn_id": str})
        if "espn_id" not in existing.columns:
            # First rebuild after the column was added: recover the id from the
            # name the row was written under, which the cache still remembers.
            names = espn_id_by_name(cache_dir)
            ids = pd.read_csv(ESPN_TEAM_IDS, dtype=str) if ESPN_TEAM_IDS.exists() else None
            from_ids = dict(zip(ids["espn_team"], ids["espn_id"])) if ids is not None else {}
            existing["espn_id"] = [
                names.get(n) or from_ids.get(n) for n in existing["espn_team"].astype(str)
            ]

    rows: list[dict] = []
    unresolved: dict[str, list[str]] = {}
    remaining: dict[str, set[str]] = {}
    for key, lg in LEAGUES.items():
        seen = by_league.get(key, {})
        prior = existing[existing["league"] == lg.understat] if len(existing) else existing
        kept = {str(r["espn_id"]): r for r in prior.to_dict("records")
                if pd.notna(r.get("espn_id")) and str(r["espn_id"]) in seen}
        taken = {r["understat_team"] for r in kept.values()}

        for tid, row in kept.items():
            rows.append({"league": lg.understat, "espn_id": tid,
                         # Refresh the display name -- this is what fixes id 90.
                         "espn_team": seen[tid], "understat_team": row["understat_team"],
                         "score": row.get("score"), "source": row.get("source")})

        pool = set(understat.get(key, ())) - taken
        missing = []
        for tid in sorted(set(seen) - set(kept)):
            name = seen[tid]
            if name in MAPPING_GAP_FILL:
                rows.append({"league": lg.understat, "espn_id": tid, "espn_team": name,
                             "understat_team": MAPPING_GAP_FILL[name],
                             "score": None, "source": "gap_fill"})
                continue
            guess = best_match(name, pool) if pool else None
            if guess is None or guess[1] < MIN_FUZZY_SCORE:
                missing.append(name)
                continue
            und, score = guess
            rows.append({"league": lg.understat, "espn_id": tid, "espn_team": name,
                         "understat_team": und, "score": round(score, 4), "source": "fuzzy"})
            pool.discard(und)
        remaining[key] = pool
        if missing:
            unresolved[key] = missing

    out = pd.DataFrame(rows, columns=list(MAPPING_COLUMNS)).sort_values(
        ["league", "espn_team"]).reset_index(drop=True)
    MAPPING_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"  club mapping: {len(out)} rows over {out['league'].nunique()} leagues -> {out_path.name}")
    for key, names in unresolved.items():
        # The pool reported against is the one actually searched -- names already
        # claimed by another club are removed. Suggesting a taken name reads as a
        # near miss when it is not available at all.
        report_unmapped(names, pool=remaining[key], label=key, indent="    ")
    if unresolved:
        print("    these have no Understat history yet (or need a hand-written row); "
              "they will map themselves once Understat publishes them")
    return out


def load_name_map(cache_dir=None) -> dict[str, str]:
    """``{any ESPN club name: canonical Understat name}``.

    Resolution is **id first, name second**. A row is looked up by the id its
    club currently reports, so a club ESPN has since renamed still resolves;
    falling back to the row's own name keeps a hand-written row working even
    before the cache has seen that club.
    """
    if not CLUB_MAPPING_ALL.exists():
        raise FileNotFoundError(
            f"{CLUB_MAPPING_ALL} not found -- run `ingest.espn.build_club_mapping()`."
        )
    m = pd.read_csv(CLUB_MAPPING_ALL, dtype={"espn_id": str}).dropna(subset=["understat_team"])
    by_id = {}
    if "espn_id" in m.columns:
        by_id = {str(i): canonical_team(u)
                 for i, u in zip(m["espn_id"], m["understat_team"]) if pd.notna(i)}

    out: dict[str, str] = {}
    for name, tid in espn_id_by_name(cache_dir).items():
        if tid in by_id:
            out[name] = by_id[tid]
    for name, und in zip(m["espn_team"], m["understat_team"]):
        if pd.notna(name):
            out.setdefault(str(name), canonical_team(und))
    for name, und in MAPPING_GAP_FILL.items():
        out.setdefault(name, canonical_team(und))
    return out


# --- Fixtures -------------------------------------------------------------

# ESPN publishes a season's match-date list in the ``calendar`` block of *any*
# scoreboard response for that season. `soccerdata.ESPN.read_schedule` reads that
# list out of the **cached** season-opener file and then fetches every date on
# it. Two consequences, and the second one is the expensive one:
#
# * ~55 requests per league per run (~257 across the five) for a pull that
#   normally wants a single day.
# * The calendar it trusts is whatever was cached the first time the season was
#   pulled. When La Liga moved its September 2026 fixtures after the calendar was
#   cached on 19 Aug, 12 Sept was simply not on the list, so no request for it
#   was ever made. The frame came back empty, the old CSV stayed on disk, and
#   four matches vanished from the workbook, the odds form and the edge book
#   without a single error.
#
# So we drive the calendar ourselves: refetch it every run -- one ~2KB request
# per league, which is what makes this *cheaper* rather than more expensive --
# diff it against the cached copy so a reschedule is reported rather than
# silently obeyed, and then fetch only the dates that fall inside the requested
# range. Roughly 10 requests a run instead of 257.


def _calendar_dates(payload: dict) -> list[str]:
    """The ``YYYYMMDD`` match dates in a scoreboard response's calendar block.

    Entries are ISO strings today; the dict form is handled because ESPN returns
    one for some competitions and a silent empty calendar is the failure mode
    this whole function exists to stop.
    """
    cal = (payload.get("leagues") or [{}])[0].get("calendar") or []
    out: set[str] = set()
    for entry in cal:
        raw = entry.get("startDate") if isinstance(entry, dict) else entry
        if isinstance(raw, str) and len(raw) >= 10:
            out.add(raw[:10].replace("-", ""))
    return sorted(out)


def _scoreboard_url(slug: str, day: str) -> str:
    return f"{ESPN_SITE_API}/{slug}/scoreboard?dates={day}"


def refresh_season_calendar(
    slug: str, season_start: int, cache_dir=None
) -> tuple[list[str], set[str], set[str]]:
    """Refetch a league's season match-date list. ``(live, added, removed)``.

    The season-opener file is rewritten in place, so `soccerdata` -- which reads
    the same path and would otherwise keep the stale list indefinitely -- picks
    the correction up too.
    """
    cache_dir = cache_dir or SOCCERDATA_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"Schedule_{slug}_{season_start}0701.json"

    cached: set[str] = set()
    if path.exists():
        try:
            cached = set(_calendar_dates(json.loads(path.read_text())))
        except (json.JSONDecodeError, OSError):
            cached = set()  # unreadable cache is the same as no cache

    payload = get_json(_scoreboard_url(slug, f"{season_start}0701"))
    live = _calendar_dates(payload)
    path.write_text(json.dumps(payload))
    return live, set(live) - cached, cached - set(live)


def pull_fixtures(
    date_from: str, date_to: str, season: str | None = None,
    league_keys: list[str] | None = None, cache_dir=None,
) -> dict[str, pd.DataFrame]:
    """Upcoming fixtures per league, name-mapped to Understat, written to Inputs/Fixtures.

    Dates are ``dd-mm-yyyy``, matching the original notebook's convention.
    """
    # Imported here rather than at module scope: `ingest.scoreboard` imports from
    # this module, so a top-level import would be circular. Same reasoning as the
    # deferred `import soccerdata` in `understat.pull_season`.
    from .scoreboard import SCOREBOARD_DELAY

    # Resolved here, not in the signature: a default argument is evaluated once at
    # import time and would pin the season for the life of the process.
    season = season or current_season_label()
    # `current_season_label` emits the ESPN '2026-27' shape, which `season_sort_key`
    # does not read; swapping the dash for a slash puts it in a shape it does, and
    # leaves the '2026/2027' and '2627' forms other callers may pass untouched.
    season_start = season_sort_key(season.replace("-", "/"))
    start = pd.to_datetime(date_from, format="%d-%m-%Y", utc=True)
    end = pd.to_datetime(date_to, format="%d-%m-%Y", utc=True) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    in_range = {d.strftime("%Y%m%d")
                for d in pd.date_range(start.normalize(), end.normalize(), freq="D")}

    name_map = load_name_map() if CLUB_MAPPING_ALL.exists() else {}

    cache_dir = cache_dir or SOCCERDATA_CACHE
    out: dict[str, pd.DataFrame] = {}
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    for key in (league_keys or list(LEAGUES)):
        lg = LEAGUES[key]
        live, added, removed = refresh_season_calendar(lg.espn_slug, season_start, cache_dir)

        # Only drift that touches the window we are pulling can change this run's
        # answer, so that is what gets reported loudly; the rest is a count.
        hit = sorted((added | removed) & in_range)
        if hit:
            moved = ", ".join(f"+{d}" if d in added else f"-{d}" for d in hit)
            print(f"  {key}: calendar changed in range ({moved}) -- stale cache would have missed these")
        elif added or removed:
            print(f"  {key}: calendar moved outside range (+{len(added)}/-{len(removed)})")

        days = sorted(set(live) & in_range)
        rows: list[dict[str, str]] = []
        for i, day in enumerate(days):
            if i:
                time.sleep(SCOREBOARD_DELAY)
            payload = get_json(_scoreboard_url(lg.espn_slug, day))
            # Always rewritten, never read from cache: a fixture inside the pull
            # window is exactly the thing whose kickoff time is still moving.
            (cache_dir / f"Schedule_{lg.espn_slug}_{day}.json").write_text(json.dumps(payload))

            for e in payload.get("events") or []:
                comp = (e.get("competitions") or [{}])[0]
                sides = {c.get("homeAway"): (c.get("team") or {}).get("name")
                         for c in comp.get("competitors") or []}
                when = pd.to_datetime(e.get("date"), utc=True, errors="coerce")
                if pd.isna(when) or not sides.get("home") or not sides.get("away"):
                    continue
                if not (start <= when <= end):
                    continue  # a date file can carry a neighbouring day's late kickoff
                rows.append({
                    "Date": when.strftime("%Y-%m-%d"),
                    "Home Team": sides["home"],
                    "Away Team": sides["away"],
                    "Start Time (UTC)": when.strftime("%H:%M:%S"),
                })

        fixtures = pd.DataFrame(rows, columns=["Date", "Home Team", "Away Team", "Start Time (UTC)"])
        if not fixtures.empty:
            unmapped = {t for t in set(fixtures["Home Team"]) | set(fixtures["Away Team"])
                        if t not in name_map}
            report_unmapped(unmapped, cap=6, label=key)
            for col in ("Home Team", "Away Team"):
                fixtures[col] = fixtures[col].map(lambda x: name_map.get(x, x))
            fixtures = fixtures.sort_values(["Date", "Start Time (UTC)", "Home Team"]).reset_index(drop=True)

        # Written even when empty. The old code `continue`d here, which left the
        # previous run's file on disk -- so "no fixtures found" and "last week's
        # fixtures" were the same state to every reader downstream.
        dest = FIXTURES_DIR / f"{lg.fixtures_stem}_fixtures.csv"
        fixtures.to_csv(dest, index=False)
        out[key] = fixtures
        note = f"{len(fixtures)} fixtures" if len(fixtures) else "no fixtures in range"
        print(f"  {key}: {note} -> {dest.name}")
    return out


# --- Team IDs -------------------------------------------------------------

LEAGUE_SLUGS_ALL_TIERS = ["eng.1", "eng.2", "eng.3", "eng.4", "esp.1", "esp.2",
                          "ger.1", "ger.2", "ita.1", "ita.2", "fra.1", "fra.2"]


def fetch_team_ids(slug: str, season: int) -> dict[str, str]:
    try:
        data = get_json(f"{ESPN_SITE_API}/{slug}/teams?season={season}")
        teams = data["sports"][0]["leagues"][0]["teams"]
        return {t["team"]["displayName"]: t["team"]["id"] for t in teams}
    except Exception:
        return {}


def build_team_id_cache(seasons: list[int] | None = None, force: bool = False) -> pd.DataFrame:
    """Resolve ESPN numeric ids for every club we know about.

    Unlike the original, this reads the existing cache first and only looks up
    names still missing -- so a re-run is nearly free instead of ~144 requests.
    """
    known: dict[str, str] = {}
    if ESPN_TEAM_IDS.exists() and not force:
        prev = pd.read_csv(ESPN_TEAM_IDS)
        known = dict(zip(prev["espn_team"].astype(str), prev["espn_id"].astype(str)))
        print(f"  {len(known)} ids already cached")

    targets: set[str] = set()
    if CLUB_MAPPING_ALL.exists():
        targets = set(pd.read_csv(CLUB_MAPPING_ALL)["espn_team"].dropna())
    missing = targets - set(known)
    if not missing:
        print("  nothing to resolve")
        return pd.DataFrame({"espn_team": list(known), "espn_id": [known[k] for k in known]})

    for season in (seasons or list(range(2025, 2013, -1))):
        if not missing:
            break
        for slug in LEAGUE_SLUGS_ALL_TIERS:
            if not missing:
                break
            for name, tid in fetch_team_ids(slug, season).items():
                if name in missing:
                    known[name] = tid
                    missing.discard(name)

    MAPPING_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"espn_team": list(known), "espn_id": [known[k] for k in known]}).sort_values("espn_team")
    out.to_csv(ESPN_TEAM_IDS, index=False)
    print(f"  resolved {len(out)} ids, {len(missing)} still unresolved")
    return out


# --- All-competition schedules (rest days) --------------------------------


def team_schedule(team_id: str, season: int) -> dict | None:
    """A team's fixtures across *all* competitions -- the rest-day source."""
    try:
        return get_json(f"{ESPN_SITE_API}/all/teams/{team_id}/schedule?season={season}")
    except Exception:
        return None


def pull_all_competition_fixtures(seasons: list[int], label: str, resume: bool = True) -> pd.DataFrame:
    """Per-team fixture history across all competitions, checkpointed and resumable."""
    if not ESPN_TEAM_IDS.exists():
        raise FileNotFoundError("Run build_team_id_cache() first.")
    ids = pd.read_csv(ESPN_TEAM_IDS)

    name_map = load_name_map() if CLUB_MAPPING_ALL.exists() else {}

    ckpt = FIXTURES_DIR / f"_checkpoint_{label.lower()}.csv"
    done: set[tuple[str, int]] = set()
    records: list[dict] = []
    if resume and ckpt.exists():
        prev = pd.read_csv(ckpt, parse_dates=["date"])
        records = prev.to_dict("records")
        if {"_team_id", "_season"} <= set(prev.columns):
            done = set(zip(prev["_team_id"].astype(str), prev["_season"].astype(int)))
        print(f"  resuming: {len(records):,} records, {len(done)} (team, season) pairs done")

    today = pd.Timestamp.today().normalize()
    todo = [(str(r.espn_id), int(s), r.espn_team) for r in ids.itertuples() for s in seasons
            if (str(r.espn_id), int(s)) not in done]
    print(f"  {len(todo)} (team, season) pulls to do")

    for i, (tid, season, espn_name) in enumerate(todo, 1):
        data = team_schedule(tid, season)
        und = name_map.get(espn_name, espn_name)
        for event in (data or {}).get("events", []):
            try:
                d = pd.Timestamp(event["date"][:10])
                if d > today:
                    continue
                comps = event["competitions"][0]["competitors"]
                entry = next((c for c in comps if c["team"].get("displayName") == espn_name), None)
                if entry is None:
                    continue
                records.append({"date": d, "team": und, "venue": entry.get("homeAway", "unknown"),
                                "_team_id": tid, "_season": season})
            except Exception:
                continue
        if i % 50 == 0:
            pd.DataFrame(records).to_csv(ckpt, index=False)
            print(f"    [{i}/{len(todo)}] checkpointed {len(records):,} records")

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["date", "team", "venue"])
    pd.DataFrame(records).to_csv(ckpt, index=False)

    out = df[["date", "team", "venue"]].drop_duplicates(["date", "team"]).sort_values(["team", "date"])
    dest = all_comp_fixtures(current_season_label()) if label.lower() == "current" else FIXTURES_DIR / "historic_competitions_fixtures.csv"
    out.to_csv(dest, index=False)
    print(f"  {len(out):,} rows -> {dest.name}")
    return out


__all__ = [
    "normalise_name", "similarity", "best_match", "report_unmapped",
    "MIN_FUZZY_SCORE", "MAPPING_COLUMNS", "MAPPING_GAP_FILL",
    "espn_id_by_name", "build_club_mapping", "load_name_map",
    "pull_fixtures", "refresh_season_calendar", "fetch_team_ids",
    "build_team_id_cache", "team_schedule", "pull_all_competition_fixtures",
]
