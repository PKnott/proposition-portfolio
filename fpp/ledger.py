"""What we said, and what actually happened.

The rest of the pipeline has no memory. `06_Split` writes one day's payload and
`_write_json` deletes yesterday's the instant it lands, so every earlier run's
propositions, prices and frontier are gone and nothing anywhere records how any
of it turned out. This module is the store that fixes that, and it is the only
thing in the project that **cannot be rebuilt from source**: a model, a clean
table and both payloads all regenerate, but a result nobody captured is gone.

Three steps, deliberately split across two notebooks:

- `capture` runs inside `06_Split`, the moment the frontier exists. It cannot
  wait for `07`: run `06` twice before `07` and the first slate has already been
  deleted. It writes a self-contained slate into `Analysis/Pending/`.
- `absorb` runs in `07_Analysis`, drains the inbox into the permanent tables and
  deletes what it consumed. Re-absorbing a run it already holds is a no-op.
- `settle` joins unsettled propositions to the results that have since arrived.
  It only ever fills rows where `won` is null, so it is safe to run every time.

**Every row is keyed structurally**, on
``(fixture_date, league_key, home_team, away_team, target, team, line)``, never
on `sheet_code` or a `D1-05#0` pick id. Those are positional within one run:
`E0-02` was Everton v Palace, then Palace v Man City, then Coventry v Hull inside
eight days. `prop_key` joins that tuple into one string so legs resolve against
propositions without carrying seven columns.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import LEAGUES, canonical_team
from .paths import ANALYSIS, ANALYSIS_PENDING, ESPN_MATCH_STATS
from .spec import STATS
# `parse_label` moved next to `proposition_label`, which it is the inverse of.
# Re-exported here because every caller and every test knows it by this name.
from .staking import BOOK_COLUMNS, add_edge, best_price, parse_label

# --- the store ------------------------------------------------------------

TABLES: tuple[str, ...] = ("runs", "propositions", "portfolios", "legs", "placed", "cash")

# Sheet-code prefix -> league key. `sheet_codes` builds `D1-05` from
# `LEAGUES[...].prefix`, so inverting that dict is the only definition needed.
_KEY_BY_PREFIX: dict[str, str] = {lg.prefix: key for key, lg in LEAGUES.items()}

# Proposition display name -> target key, e.g. "Shots on Target" -> "sot". The
# labels are built by `staking.proposition_label` from `StatSpec.display`, so
# this inverts one registry rather than restating it.
_TARGET_BY_DISPLAY: dict[str, str] = {s.display: s.key for s in STATS}

_LABEL = re.compile(r"^(?P<head>.+) - Over (?P<line>-?\d+(?:\.\d+)?)$")

# The four stats a proposition can be about, as they are named in
# `espn_match_stats.csv`: `<scope>_<target>`.
_STAT_COLUMNS: tuple[str, ...] = tuple(
    f"{side}_{t}" for side in ("home", "away") for t in ("goals", "shots", "sot", "corners")
)


def store_dir(root: Path | None = None) -> Path:
    return Path(root) if root else ANALYSIS


def table_path(name: str, root: Path | None = None) -> Path:
    if name not in TABLES:
        raise ValueError(f"unknown table {name!r}; expected one of {', '.join(TABLES)}")
    return store_dir(root) / f"{name}.parquet"


def read_table(name: str, root: Path | None = None) -> pd.DataFrame:
    """One ledger table, or an empty frame if it has never been written."""
    path = table_path(name, root)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def write_table(df: pd.DataFrame, name: str, root: Path | None = None) -> Path:
    """Replace a table on disk, via a temporary file.

    Parquet has no append, so every write is a whole-table rewrite. Writing
    through a temp file and renaming means an interrupted write leaves the
    previous table intact rather than a half-written one -- and this is the store
    whose loss cannot be undone by re-running anything.
    """
    path = table_path(name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


# --- keys -----------------------------------------------------------------


def league_of(sheet_code: str) -> str:
    """``"D1-05"`` -> ``"Bund"``. Within-run positional codes still name a league."""
    prefix = str(sheet_code).split("-")[0]
    if prefix not in _KEY_BY_PREFIX:
        raise ValueError(f"unknown league prefix {prefix!r} in sheet code {sheet_code!r}")
    return _KEY_BY_PREFIX[prefix]


def prop_key(fixture_date, league_key: str, home: str, away: str,
             target: str, team: str, line: float) -> str:
    """The structural identity of one proposition, as one string.

    Everything in it is a fact about the fixture and the bet. Nothing in it is a
    fact about the run that happened to produce it, which is the whole point --
    the same proposition offered on two runs is one row here.
    """
    d = pd.Timestamp(fixture_date).date().isoformat()
    return f"{d}|{league_key}|{home}|{away}|{target}|{team}|{float(line):g}"


def _keys_for(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [prop_key(*r) for r in zip(df["fixture_date"], df["league_key"], df["home_team"],
                                   df["away_team"], df["target"], df["team"], df["line"])],
        index=df.index, dtype="string",
    )


# --- run codes ------------------------------------------------------------

_RUN_CODE = re.compile(r"^R(\d+)$")


def next_run_code(root: Path | None = None) -> str:
    """``R001``, ``R002``, ... counting absorbed runs *and* pending slates.

    Counting both matters: capture assigns the code and 07 may not run for days,
    so two captures in that window must not both be handed `R008`. The date
    alone would not do either -- `06` can be run twice in one day.
    """
    used = set()
    runs = read_table("runs", root)
    if not runs.empty:
        used |= set(runs["run_code"].astype(str))
    used |= {p.name.split("_")[1] for p in pending_slates(root)}
    n = max((int(m.group(1)) for c in used if (m := _RUN_CODE.match(c))), default=0)
    return f"R{n + 1:03d}"


def pending_slates(root: Path | None = None) -> list[Path]:
    """Captured slates waiting to be absorbed, oldest code first."""
    pend = store_dir(root) / "Pending"
    if not pend.exists():
        return []
    return sorted(p for p in pend.iterdir() if p.is_dir() and p.name.startswith("slate_"))


# --- capture (runs inside 06_Split) ---------------------------------------


@dataclass(frozen=True)
class Slate:
    """One captured run, sitting in the inbox."""
    run_code: str
    path: Path
    n_propositions: int
    n_portfolios: int

    def __str__(self) -> str:
        return f"{self.run_code} -> {self.path} ({self.n_propositions} props, {self.n_portfolios} portfolios)"


def proposition_rows(filled: pd.DataFrame, run_code: str,
                     qualified_ids: dict[tuple[str, str], str] | None = None) -> pd.DataFrame:
    """Every priced proposition on one form, as ledger rows.

    `filled` carries the fixture on every row -- `read_filled` reads `date`,
    `league`, `home_team` and `away_team` off the Contents sheet -- so this needs
    nothing else to build a structural key. Rows no book priced are dropped by
    `best_price`, which is the sample the ledger banks.
    """
    priced = add_edge(best_price(filled))
    if priced.empty:
        return pd.DataFrame()

    parsed = [parse_label(x) for x in priced["label"]]
    out = pd.DataFrame({
        "run_code": run_code,
        "fixture_date": pd.to_datetime(priced["date"]).dt.normalize(),
        "league_key": [league_of(c) for c in priced["sheet_code"]],
        "home_team": priced["home_team"].astype("string"),
        "away_team": priced["away_team"].astype("string"),
        "target": pd.Series([t for t, _, _ in parsed], index=priced.index, dtype="string"),
        "team": pd.Series([tm for _, tm, _ in parsed], index=priced.index, dtype="string"),
        "line": [ln for _, _, ln in parsed],
        "label": priced["label"].astype("string"),
        "sheet_code": priced["sheet_code"].astype("string"),
        "p": priced["p"].astype(float),
        "o": priced["o"].astype(float),
        "book": priced["book"].astype("string"),
        "e": priced["e"].astype(float),
    })
    # `scope` is which side of the fixture the proposition is about, and it is
    # what picks the column out of the results file. Derived rather than read:
    # the form does not carry it, but the team name settles it.
    out["scope"] = np.where(out["team"] == out["home_team"], "home", "away")
    for col in BOOK_COLUMNS:
        out[col] = priced[col].astype(float).to_numpy() if col in priced else np.nan

    out["prop_key"] = _keys_for(out)
    # Which of these became a pick, keyed on `(sheet_code, label)` -- the pair
    # the payload and the form both carry verbatim. Matching on the rendered
    # `"Home vs Away"` string instead would work right up until a club name
    # containing " vs " or a stray space made it quietly match nothing.
    ids = qualified_ids or {}
    out["prop_id"] = pd.Series(
        [ids.get((c, l)) for c, l in zip(out["sheet_code"], out["label"])],
        index=out.index, dtype="string")
    out["qualified"] = out["prop_id"].notna()
    out["actual"] = pd.Series(np.nan, index=out.index, dtype=float)
    out["won"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out["settled_at"] = pd.Series(pd.NA, index=out.index, dtype="string")
    return out.reset_index(drop=True)


def _growth_columns(growth: dict | None) -> dict:
    """The growth scalars, old names and new.

    Runs before 2026-09 wrote `f_star`/`f_protective`; runs after write
    `f_suggested` and the factors behind it. **Both sets of columns are emitted**,
    each null where the payload does not carry it, because this table is the one
    store in the project that cannot be rebuilt -- dropping the old columns would
    make R001-R007 unreadable, and renaming in place would silently merge two
    different quantities into one column.

    `g_f_protective` is filled from the new `f_drawdown` where it exists: they are
    the same number under the same constraint, which is what makes the settled
    history comparable across the change.
    """
    g = growth or {}
    dd = g.get("drawdown") or {}
    return {
        "g_p0": g.get("p0"),
        # new
        "g_f_suggested": g.get("f_suggested"), "g_g_suggested": g.get("g_suggested"),
        "g_edge_factor": g.get("edge_factor"), "g_var_factor": g.get("var_factor"),
        "g_k_leverage": g.get("k_leverage"),
        "g_breakeven_shift": g.get("breakeven_shift"),
        # retained: `f_drawdown` is the same constrained optimum the old column held
        "g_f_protective": g.get("f_drawdown", g.get("f_protective")),
        "g_g_protective": g.get("g_protective"),
        "g_f_star": g.get("f_star"), "g_g_star": g.get("g_star"),
        "g_growth_given_up": g.get("growth_given_up"),
        "g_drawdown_d": dd.get("d"), "g_drawdown_p": dd.get("p"),
    }


# Copied verbatim off each payload portfolio. The `p_over_*` keys are read from
# the record rather than listed, because `staking.THRESHOLDS` is meant to be
# tuned and a hardcoded list silently writes a column of nulls when it is.
#
# `capacity` and its companions are null on runs written before they existed;
# `capacity` is recomputable from `legs` joined to `propositions`, the other two
# are not, because they depend on the stakes as allocated.
_PORTFOLIO_FIELDS = ("split", "legs", "expected_return_pct", "sd_pct", "variance",
                     "capacity", "capacity_used", "max_leg_stake", "n_eff",
                     "median_leg_p")


def _frontier_from_payload(payload: dict, run_code: str,
                           key_by_pid: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(portfolios, legs)`` from a payload's portfolio list.

    The `projection` block is dropped on the way in. `g_curve`, the percentile
    bands and the histogram are plotting arrays that `growth.growth_metrics`
    rebuilds from the same stakes and prices; the scalars analysis actually needs
    -- the suggested stake, the factors behind it, `p0` and `drawdown` -- are in
    `growth` and are kept. That is 9 MB a slate down to under one.
    """
    prow, lrow = [], []
    for p in payload.get("portfolios", []):
        pid = int(p["id"])
        thresholds = {k: v for k, v in p.items() if k.startswith("p_over_")}
        prow.append({"run_code": run_code, "portfolio_id": pid,
                     **{f: p.get(f) for f in _PORTFOLIO_FIELDS}, **thresholds,
                     **_growth_columns(p.get("growth"))})
        stakes = p.get("stakes") or []
        for i, pick in enumerate(p.get("picks", [])):
            lrow.append({
                "run_code": run_code, "portfolio_id": pid, "leg": i,
                "prop_key": key_by_pid.get(pick),
                "stake_frac": float(stakes[i]) if i < len(stakes) else np.nan,
            })

    portfolios = pd.DataFrame(prow)
    if not portfolios.empty:
        portfolios["split"] = portfolios["split"].astype("string")
        portfolios["realised_return"] = np.nan
        portfolios["n_settled"] = 0
        portfolios["settled"] = False
    legs = pd.DataFrame(lrow)
    if not legs.empty:
        legs["prop_key"] = legs["prop_key"].astype("string")
    return portfolios, legs


def capture(result: dict, filled: pd.DataFrame, *, source: str | None = None,
            run_code: str | None = None, root: Path | None = None) -> Slate:
    """Write one run into the inbox, and hand back the code it was given.

    Call this *before* `write_portfolios_json` so the run code can travel in the
    payload and the Edge Book can print `R007-770` on the betting slip -- that
    string is what `record_bet` wants back, and reading it off the page you are
    already looking at beats remembering which run it was.
    """
    from .report.edge_book import portfolios_payload

    run_code = run_code or next_run_code(root)
    payload = portfolios_payload(result, filled, source=source, with_projection=False)

    ids = {(rec["event"], rec["proposition"]): rec["id"]
           for rec in payload.get("propositions", [])}
    props = proposition_rows(filled, run_code, ids)
    # Legs point at pick ids; the ledger keys on the fixture. `props` holds both
    # columns, so the crosswalk is the frame itself rather than a second parse.
    picked = props[props["prop_id"].notna()]
    key_by_pid = dict(zip(picked["prop_id"], picked["prop_key"]))
    missing = {i for i in ids.values()} - set(key_by_pid)
    if missing:
        raise ValueError(
            f"{len(missing)} pick(s) on the frontier have no priced row on the form "
            f"-- e.g. {sorted(missing)[:3]}. The payload and the form have drifted.")
    portfolios, legs = _frontier_from_payload(payload, run_code, key_by_pid)

    dest = store_dir(root) / "Pending" / f"slate_{run_code}_{_run_date(payload)}"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    run = {
        "run_code": run_code,
        "run_date": _run_date(payload),
        "generated_at": payload.get("generated_at"),
        "source_odds_file": payload.get("source_odds_file"),
        "min_legs": payload.get("min_legs"),
        "max_legs": payload.get("max_legs"),
        "origin": "capture",
        "captured_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        **{f"n_{k}": v for k, v in (payload.get("counts") or {}).items()},
    }
    (dest / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    props.to_parquet(dest / "propositions.parquet", index=False)
    portfolios.to_parquet(dest / "portfolios.parquet", index=False)
    legs.to_parquet(dest / "legs.parquet", index=False)
    return Slate(run_code, dest, len(props), len(portfolios))


def _run_date(payload: dict) -> str:
    """The date the payload's own filename would carry.

    `generated_at` is UTC and the filename is local, so the two can disagree by a
    day around midnight -- and this pipeline runs near midnight. The source form
    name is the tiebreak, because that is the file the run actually consumed.
    """
    src = str(payload.get("source_odds_file") or "")
    if m := re.search(r"(\d{4}-\d{2}-\d{2})", src):
        return m.group(1)
    gen = str(payload.get("generated_at") or "")
    return gen[:10] or dt.date.today().isoformat()


# --- backfill (the runs that happened before this module existed) ----------


def _unwrap(html: str, element_id: str) -> dict:
    """One inlined payload out of a built Edge Book page.

    `write_edge_book` escapes `<` to its `\u003c` form on the way in, which JSON
    reads back as the same string; undoing it is not required for correctness but
    keeps the recovered payload byte-identical to the one that was written.
    """
    m = re.search(rf'<script id="{element_id}" type="application/json">(.*?)</script>',
                  html, re.S)
    if not m:
        raise ValueError(f"no {element_id} block in this page")
    return json.loads(m.group(1).replace("\\u003c", "<"))


def slate_from_payloads(predictions: dict, portfolios: dict, run_code: str,
                        root: Path | None = None, *, origin: str = "backfill") -> Slate:
    """Rebuild a slate from the two payloads a run left behind.

    The price rows carry only `event` and `label`, so the fixture comes from the
    predictions payload -- which is why this needs both. What cannot be recovered
    is the per-book column for the five books that did not win: `best_price` had
    already collapsed them before either payload was written. `o`, `book` and `e`
    are intact, so everything the analyses read is here; only "which book was
    sharpest" has to start from the next captured run.
    """
    fixtures = {f["event"]: f for f in predictions.get("fixtures", [])}
    ids = {(r["event"], r["proposition"]): r["id"] for r in portfolios.get("propositions", [])}

    rows = []
    for r in portfolios.get("prices", []):
        fx = fixtures.get(r["event"])
        if fx is None:
            continue
        target, team, line = parse_label(r["label"])
        rows.append({
            "run_code": run_code,
            "fixture_date": pd.Timestamp(fx["date"]).normalize(),
            "league_key": fx["league_key"], "home_team": fx["home"], "away_team": fx["away"],
            "target": target, "team": team, "line": line,
            "scope": "home" if team == fx["home"] else "away",
            "label": r["label"], "sheet_code": r["event"],
            "p": r.get("p"), "o": r.get("odds"), "book": r.get("book"), "e": r.get("e"),
            "prop_id": ids.get((r["event"], r["label"])),
        })
    props = pd.DataFrame(rows)
    if props.empty:
        raise ValueError(f"{run_code}: no price rows could be placed against a fixture")

    for col in BOOK_COLUMNS:
        props[col] = np.nan
    for col in ("home_team", "away_team", "target", "team", "label", "sheet_code",
                "book", "prop_id"):
        props[col] = props[col].astype("string")
    props["prop_key"] = _keys_for(props)
    props["qualified"] = props["prop_id"].notna()
    props["actual"] = np.nan
    props["won"] = pd.Series(pd.NA, index=props.index, dtype="boolean")
    props["settled_at"] = pd.Series(pd.NA, index=props.index, dtype="string")

    picked = props[props["prop_id"].notna()]
    key_by_pid = dict(zip(picked["prop_id"], picked["prop_key"]))
    frontier, legs = _frontier_from_payload(portfolios, run_code, key_by_pid)

    run_date = _run_date(portfolios)
    dest = store_dir(root) / "Pending" / f"slate_{run_code}_{run_date}"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    run = {
        "run_code": run_code, "run_date": run_date,
        "generated_at": portfolios.get("generated_at"),
        "source_odds_file": portfolios.get("source_odds_file"),
        "min_legs": portfolios.get("min_legs"), "max_legs": portfolios.get("max_legs"),
        "origin": origin,
        "captured_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        **{f"n_{k}": v for k, v in (portfolios.get("counts") or {}).items()},
    }
    (dest / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    props.to_parquet(dest / "propositions.parquet", index=False)
    frontier.to_parquet(dest / "portfolios.parquet", index=False)
    legs.to_parquet(dest / "legs.parquet", index=False)
    return Slate(run_code, dest, len(props), len(frontier))


def backfill(root: Path | None = None, *, pages: list[Path] | None = None,
             include_current: bool = True) -> list[Slate]:
    """Recover every run still on disk into the inbox, oldest first.

    Two sources, and they do not overlap. Every superseded page is archived whole
    with both payloads inlined, which is the only surviving copy of those runs --
    `_write_json` deleted the JSON itself the day after. The current run is still
    in `Data/` as JSON and is read from there.

    Ordering matters: run codes are handed out in the order slates are made, so
    walking oldest page first makes `R001` the oldest run rather than whichever
    file the filesystem listed first.
    """
    from .paths import OUTPUTS_DATA, archive_dir
    from .report.edge_book import latest_json

    if pages is None:
        pages = sorted(archive_dir("edge_book").glob("edge_book_*.html"))
    # Dates, not codes. Comparing `run_date` against a set of `run_code`s never
    # matched, so a second `backfill` re-imported every archived page and `absorb`
    # gave the copies fresh codes -- one silent duplication of the whole ledger.
    # A run that was deliberately excluded is in `runs` too, so it stays out.
    known = read_table("runs", root)
    seen = set(known["run_date"].astype(str)) if not known.empty else set()
    seen |= {p.name.split("_", 2)[2] for p in pending_slates(root)}
    # Why a date is being skipped, so "excluded on purpose" and "already have it"
    # do not print the same sentence. They are very different reassurances.
    excluded: dict[str, str] = {}
    if not known.empty and "status" in known:
        gone = known[known["status"].fillna("active") == "excluded"]
        excluded = dict(zip(gone["run_date"].astype(str),
                            gone["excluded_reason"].fillna("excluded by hand")))

    jobs: list[tuple[str, dict, dict]] = []
    for page in sorted(pages):
        m = re.search(r"(\d{4}-\d{2}-\d{2})", Path(page).name)
        html = Path(page).read_text(encoding="utf-8")
        try:
            pred, port = _unwrap(html, "predictions-data"), _unwrap(html, "portfolios-data")
        except ValueError as exc:
            print(f"  skipping {Path(page).name}: {exc}")
            continue
        jobs.append((m.group(1) if m else _run_date(port), pred, port))

    if include_current:
        pj, oj = latest_json("predictions"), latest_json("portfolios")
        if pj and oj:
            pred = json.loads(Path(pj).read_text(encoding="utf-8"))
            port = json.loads(Path(oj).read_text(encoding="utf-8"))
            jobs.append((_run_date(port), pred, port))

    out = []
    for run_date, pred, port in sorted(jobs, key=lambda j: j[0]):
        if run_date in excluded:
            print(f"  {run_date} excluded ({excluded[run_date]}); NOT re-importing")
            continue
        if run_date in seen:
            print(f"  {run_date} already in the ledger; skipping")
            continue
        slate = slate_from_payloads(pred, port, next_run_code(root), root)
        seen.add(run_date)
        out.append(slate)
        print(f"  {slate}")
    return out


# --- absorb (runs inside 07_Analysis) -------------------------------------


def absorb(root: Path | None = None, *, keep_source: bool = False) -> pd.DataFrame:
    """Drain the inbox into the permanent tables. Returns the runs taken in.

    Idempotent by run code: a slate whose code is already in `runs` is dropped
    rather than appended, so a half-finished 07 can simply be re-run.
    """
    slates = pending_slates(root)
    if not slates:
        print("Nothing pending.")
        return pd.DataFrame()

    runs = read_table("runs", root)
    known = set(runs["run_code"].astype(str)) if not runs.empty else set()
    new_runs, taken = [], []

    for path in slates:
        run = json.loads((path / "run.json").read_text(encoding="utf-8"))
        code = str(run["run_code"])
        if code in known:
            print(f"  {code} already absorbed; discarding the pending copy")
            if not keep_source:
                shutil.rmtree(path)
            continue
        run["absorbed_at"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        run.setdefault("status", "active")
        run.setdefault("excluded_reason", None)
        new_runs.append(run)
        taken.append(path)
        known.add(code)

    if not new_runs:
        return pd.DataFrame()

    for name in ("propositions", "portfolios", "legs"):
        frames = [read_table(name, root)] + [
            pd.read_parquet(p / f"{name}.parquet") for p in taken]
        frames = [f for f in frames if not f.empty]
        if frames:
            write_table(pd.concat(frames, ignore_index=True), name, root)

    added = pd.DataFrame(new_runs)
    write_table(pd.concat([runs, added], ignore_index=True) if not runs.empty else added,
                "runs", root)

    if not keep_source:
        for p in taken:
            shutil.rmtree(p)
    print(f"Absorbed {len(added)} run(s): {', '.join(added['run_code'])}")
    return added


def purge(*run_codes: str, reason: str = "", root: Path | None = None) -> pd.DataFrame:
    """Drop a run's data, and remember that it was dropped on purpose.

    For runs that are real but must not be pooled with the rest -- most obviously
    a slate priced by a superseded model, where the probabilities came from
    something that no longer exists and averaging them into a calibration table
    would describe a model nobody is running.

    The `runs` row **stays**, marked `excluded`, and only its propositions,
    portfolios, legs and placed bet are removed. That is deliberate on two
    counts. It keeps `backfill` from cheerfully re-importing the run the next
    time it walks the archive, since it skips any date `runs` already names. And
    it leaves the ledger able to say why its history starts where it does, which
    a silent deletion could not.

    Nothing outside the ledger is touched: the archived pages these came from are
    still on disk, so an exclusion is reversible by hand if it turns out to be
    wrong.
    """
    runs = read_table("runs", root)
    if runs.empty:
        print("Nothing to purge.")
        return runs
    codes = {str(c) for c in run_codes}
    unknown = codes - set(runs["run_code"].astype(str))
    if unknown:
        raise KeyError(f"not in the ledger: {', '.join(sorted(unknown))}")

    for name in ("propositions", "portfolios", "legs", "placed"):
        df = read_table(name, root)
        if df.empty or "run_code" not in df:
            continue
        kept = df[~df["run_code"].astype(str).isin(codes)]
        if len(kept) != len(df):
            write_table(kept.reset_index(drop=True), name, root)
            print(f"  {name}: dropped {len(df) - len(kept):,}, kept {len(kept):,}")

    if "status" not in runs:
        runs["status"] = "active"
    if "excluded_reason" not in runs:
        runs["excluded_reason"] = None
    mask = runs["run_code"].astype(str).isin(codes)
    runs.loc[mask, "status"] = "excluded"
    runs.loc[mask, "excluded_reason"] = reason or "excluded by hand"
    write_table(runs, "runs", root)
    print(f"Excluded {len(codes)} run(s): {', '.join(sorted(codes))}"
          + (f" -- {reason}" if reason else ""))
    return runs


def active_runs(root: Path | None = None) -> pd.DataFrame:
    """The runs whose data the ledger is actually holding."""
    runs = read_table("runs", root)
    if runs.empty or "status" not in runs:
        return runs
    return runs[runs["status"].fillna("active") != "excluded"].reset_index(drop=True)


# --- settle ---------------------------------------------------------------


def pull_results(root: Path | None = None, dates: dict[str, set[str]] | None = None,
                 *, cache_dir: Path | None = None) -> int:
    """Fetch the results the ledger is still waiting on. ESPN only.

    Everything a proposition settles against -- goals, shots, shots on target,
    corners -- is in ESPN's scoreboard. Understat supplies xG, which the model
    uses and settlement never touches, so this deliberately does not go near it.

    That is not a tidiness argument. The old route was `ingest.refresh("current")`,
    whose ESPN passes were all driven off Understat's *published* match dates --
    so a matchday Understat had not caught up on was never looked for, and 819
    propositions sat unsettled with the results already on ESPN's server. Asking
    for the dates the ledger is actually waiting on removes that dependency
    entirely.

    Passing `dates` overrides the derivation, for a date the ledger has no
    proposition on.
    """
    from .ingest.scoreboard import refresh_results

    if dates is None:
        props = read_table("propositions", root)
        if props.empty or props["won"].notna().all():
            print("Nothing waiting on results.")
            return 0
        pending = props[props["won"].isna()]
        dates = {}
        for lk, grp in pending.groupby("league_key"):
            dates.setdefault(str(lk), set()).update(
                pd.to_datetime(grp["fixture_date"]).dt.strftime("%Y%m%d"))
        n = len(pending)
        days = sorted({d for v in dates.values() for d in v})
        print(f"{n:,} unsettled proposition(s) across {len(days)} date(s): {', '.join(days)}")
    return refresh_results(dates, cache_dir)


def load_results(path: Path | None = None) -> pd.DataFrame:
    """The results table, one row per played match.

    `espn_match_stats.csv` is the only file that carries all four statistics a
    proposition can be about. It keeps both the mapped club name and ESPN's own,
    which is what makes the fallback join below possible.
    """
    stats = pd.read_csv(Path(path) if path else ESPN_MATCH_STATS, parse_dates=["date"])
    keep = ["league_key", "date", "home_team", "away_team", "home_espn", "away_espn",
            *_STAT_COLUMNS]
    stats = stats[[c for c in keep if c in stats.columns]].copy()
    stats["date"] = stats["date"].dt.normalize()
    return stats


def match_fixtures(fixtures: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Attach every statistic to each fixture, or leave it null.

    Joined on `(league_key, home, away)` with a +/-1 day tolerance, exactly as
    `clean.attach_espn_stats` does and for the same reason: ESPN timestamps are
    UTC kickoff and land either side of the local date.

    The name is tried three ways -- as the ledger holds it, against ESPN's own
    display name, and through `canonical_team`. A form carrying `Coventry City`
    against a results file carrying `Coventry` is the single reason the first
    pass is not enough, and it costs one fixture a slate to get wrong.
    """
    left = fixtures.copy().reset_index(drop=True)
    left["_row"] = np.arange(len(left))
    left["_home_c"] = [canonical_team(x) for x in left["home_team"]]
    left["_away_c"] = [canonical_team(x) for x in left["away_team"]]

    found: dict[int, dict] = {}
    attempts = (("home_team", "away_team", "home_team", "away_team"),
                ("home_team", "away_team", "home_espn", "away_espn"),
                ("_home_c", "_away_c", "home_team", "away_team"))
    for lh, la, rh, ra in attempts:
        if rh not in results.columns or ra not in results.columns:
            continue
        for shift in (0, -1, 1):
            pending = left[~left["_row"].isin(found)]
            if pending.empty:
                break
            r = results.rename(columns={rh: "_rh", ra: "_ra"}).copy()
            r["date"] = r["date"] + pd.Timedelta(days=shift)
            j = pending.merge(
                r[["league_key", "date", "_rh", "_ra", *_STAT_COLUMNS]],
                left_on=["league_key", "fixture_date", lh, la],
                right_on=["league_key", "date", "_rh", "_ra"], how="inner")
            for rec in j.to_dict("records"):
                found.setdefault(rec["_row"], {c: rec[c] for c in _STAT_COLUMNS})

    for c in _STAT_COLUMNS:
        left[c] = left["_row"].map(lambda i, col=c: found.get(i, {}).get(col, np.nan))
    left["played"] = left["_row"].isin(found)
    return left.drop(columns=["_row", "_home_c", "_away_c"])


def settle(root: Path | None = None, results: pd.DataFrame | None = None) -> dict:
    """Fill `actual` and `won` for every proposition whose match has been played.

    Only ever writes rows where `won` is null, so running it again after more
    results have landed picks up the new ones and leaves the rest alone. A
    proposition wins on ``actual > line``, strictly -- the same comparison
    `metrics.over_prob` integrates and `calibration.probability_stream` grades,
    so a settled row and a modelled one can never disagree about what "over"
    means.
    """
    props = read_table("propositions", root)
    if props.empty:
        print("Nothing to settle.")
        return {"settled": 0, "pending": 0}

    todo = props[props["won"].isna()]
    if todo.empty:
        print(f"All {len(props):,} propositions already settled.")
        return {"settled": 0, "pending": 0}

    res = results if results is not None else load_results()
    fixtures = (todo[["fixture_date", "league_key", "home_team", "away_team"]]
                .drop_duplicates().reset_index(drop=True))
    matched = match_fixtures(fixtures, res)

    on = ["fixture_date", "league_key", "home_team", "away_team"]
    joined = todo[on + ["scope", "target", "line"]].merge(matched, on=on, how="left")
    actual = np.full(len(joined), np.nan)
    for scope in ("home", "away"):
        for target in ("goals", "shots", "sot", "corners"):
            m = ((joined["scope"] == scope) & (joined["target"] == target)).to_numpy()
            if m.any():
                actual[m] = joined.loc[m, f"{scope}_{target}"].to_numpy(dtype=float)

    stamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    won = pd.Series(
        pd.array(np.where(np.isfinite(actual), actual > joined["line"].to_numpy(), None),
                 dtype="boolean"), index=todo.index)
    props.loc[todo.index, "actual"] = actual
    props.loc[todo.index, "won"] = won
    props.loc[todo.index, "settled_at"] = won.notna().map({True: stamp, False: pd.NA})
    write_table(props, "propositions", root)

    n = int(won.notna().sum())
    pending = int(len(props) - props["won"].notna().sum())
    print(f"Settled {n:,} proposition(s); {pending:,} still waiting on results.")
    return {"settled": n, "pending": pending}


# --- roll up --------------------------------------------------------------


def roll_up(root: Path | None = None) -> pd.DataFrame:
    """Realised return for every frontier portfolio whose legs are all settled.

    ``sum(stake_frac * odds * won) - 1``: what a pound spread across the legs in
    those proportions came back as, net. A plain sum is right because a portfolio
    takes at most one proposition per event -- the rule the README names as the
    one that makes leg independence hold rather than an assumption about it.
    """
    portfolios = read_table("portfolios", root)
    legs = read_table("legs", root)
    props = read_table("propositions", root)
    if portfolios.empty or legs.empty or props.empty:
        print("Nothing to roll up.")
        return portfolios

    # Price and probability come off every leg, settled or not; the outcome only
    # off the settled ones. Two merges rather than one because the aggregates
    # below describe what the portfolio *was*, and that is knowable the moment it
    # is built -- waiting for results to compute a median leg probability would
    # make the characteristic unavailable exactly when it is wanted.
    j = legs.merge(props[["run_code", "prop_key", "p", "o", "e"]],
                   on=["run_code", "prop_key"], how="left")
    j = j.merge(props[props["won"].notna()][["run_code", "prop_key", "won"]],
                on=["run_code", "prop_key"], how="left")
    j["ret"] = j["stake_frac"] * j["o"] * j["won"].astype("float")

    g = j.groupby(["run_code", "portfolio_id"], observed=True).agg(
        n_legs=("leg", "size"),
        n_settled=("won", lambda s: int(s.notna().sum())),
        gross=("ret", "sum"),
        median_leg_p=("p", "median"),
        mean_leg_p=("p", "mean"),
        mean_leg_odds=("o", "mean"),
        min_leg_odds=("o", "min"),
        max_leg_odds=("o", "max"),
        mean_leg_e=("e", "mean"),
    ).reset_index()
    g["settled"] = g["n_settled"] == g["n_legs"]
    g["realised_return"] = np.where(g["settled"], g["gross"] - 1.0, np.nan)

    carried = ["n_settled", "settled", "realised_return", *LEG_AGGREGATES]
    out = portfolios.drop(columns=carried, errors="ignore")
    out = out.merge(g[["run_code", "portfolio_id", *carried]],
                    on=["run_code", "portfolio_id"], how="left")
    write_table(out, "portfolios", root)
    n = int(out["settled"].fillna(False).sum())
    print(f"Rolled up {n:,} of {len(out):,} portfolios.")
    return out


# --- the money ------------------------------------------------------------
#
# The Projection Book's whole claim is about a bankroll compounding round over
# round, so the ledger has to know what the bankroll *is* rather than being told
# a pot each time. `cash` holds the money that arrives and leaves by hand; the
# bets move the rest, and `balance` is the two folded together.

# The one the model computes, named as the Edge Book's betting slip labels it so
# the word you read on the page is the word you type here. `custom` is the other:
# a fraction you chose, which by definition no column can supply.
#
# `protective` and `max` stay as aliases. Bets were recorded under those names and
# `record_bet` has to keep accepting them; `max` resolves to a column that new
# runs no longer fill, which is correct -- it was the growth optimum, and that
# number sat at its own ceiling on two thirds of portfolios.
STAKE_MODES: dict[str, str] = {
    "suggested": "g_f_suggested",
    "protective": "g_f_protective",
    "max": "g_f_star",
}
CUSTOM = "custom"

# What a portfolio's legs look like, folded onto the portfolio row. Stored rather
# than recomputed because every selection rule reads a portfolio column, so a
# characteristic that lived only in a join could not be one.
LEG_AGGREGATES: tuple[str, ...] = (
    "median_leg_p", "mean_leg_p", "mean_leg_odds", "min_leg_odds", "max_leg_odds",
    "mean_leg_e",
)

# Ordering within one day: money in before the bet it funded, money out after the
# bet it was funded by. Without it a same-day deposit and stake settle in
# whatever order the rows happen to sit in.
_CASH_RANK: dict[str, int] = {"opening": 0, "deposit": 1, "bet": 2, "withdrawal": 3}


def _cash_row(kind: str, amount: float, when, note: str) -> dict:
    return {
        "kind": kind, "when": str(pd.Timestamp(when).date() if when else dt.date.today()),
        "amount": float(amount), "note": note,
        "recorded_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def _add_cash(row: dict, root: Path | None, *, replace_kind: str | None = None) -> pd.Series:
    cash = read_table("cash", root)
    if not cash.empty and replace_kind:
        cash = cash[cash["kind"] != replace_kind]
    rec = pd.DataFrame([row])
    write_table(pd.concat([cash, rec], ignore_index=True) if not cash.empty else rec,
                "cash", root)
    return rec.iloc[0]


def open_account(amount: float, when=None, note: str = "", root: Path | None = None) -> pd.Series:
    """Set the opening balance. Calling it again moves it rather than adding one.

    There is only ever one opening balance -- a second would be a deposit, and
    the difference matters: an opening balance is where the record starts, a
    deposit is money that arrived during it.
    """
    rec = _add_cash(_cash_row("opening", amount, when, note), root, replace_kind="opening")
    print(f"Opening balance {amount:,.2f} as at {rec['when']}")
    return rec


def deposit(amount: float, when=None, note: str = "", root: Path | None = None) -> pd.Series:
    """Money put in, so a rising balance is not mistaken for a winning run."""
    rec = _add_cash(_cash_row("deposit", abs(amount), when, note), root)
    print(f"Deposit {abs(amount):,.2f} on {rec['when']}  ->  balance {balance(root):,.2f}")
    return rec


def withdraw(amount: float, when=None, note: str = "", root: Path | None = None) -> pd.Series:
    """Money taken out. Stored positive; `bankroll` is what subtracts it."""
    rec = _add_cash(_cash_row("withdrawal", abs(amount), when, note), root)
    print(f"Withdrawal {abs(amount):,.2f} on {rec['when']}  ->  balance {balance(root):,.2f}")
    return rec


def balance(root: Path | None = None) -> float:
    """The pot right now: opening, plus and minus cash, plus every settled bet.

    An unsettled bet does not move it. The stake is with the bookmaker and the
    outcome is unknown, so counting it either way would be a guess -- `bankroll`
    marks those rows `pending` instead.
    """
    book = bankroll(root)
    if book.empty:
        return 0.0
    return float(book["balance"].iloc[-1])


def record_bet(ref: str, *, pot: float | None = None, mode: str | None = None,
               f: float | None = None, stake: float | None = None,
               root: Path | None = None) -> pd.Series:
    """Record which portfolio was backed, and for how much.

    `ref` is what the Edge Book prints: ``"R007-770"``. Optional by design --
    skip a slate and the ledger holds no bet for that run, which is the truth.
    Recording the same run twice replaces rather than doubles, so a corrected pot
    is one call, not a hand edit.

    Three ways to say how much, matching the slip's three buttons:

    - ``mode="protective"`` (the default) or ``mode="max"`` -- the fraction the
      model computed, read off this portfolio's own growth block.
    - ``f=0.15`` -- a fraction you chose. Recorded as `custom`, because a stake
      the model did not pick must not be filed under a mode that says it did.
    - ``stake=150`` -- the cash you actually put on, when that is what you know.
      The fraction is derived from the pot, so the record still carries both.

    `pot` defaults to the **current balance**, which is the number a stake
    fraction is supposed to be a fraction of. Retyping it every round is how a
    ledger ends up staking 21% of a pot that stopped existing four losses ago.
    """
    run_code, _, pid = str(ref).partition("-")
    if not pid.isdigit():
        raise ValueError(f"expected a reference like 'R007-770', got {ref!r}")
    portfolio_id = int(pid)

    portfolios = read_table("portfolios", root)
    hit = portfolios[(portfolios["run_code"] == run_code)
                     & (portfolios["portfolio_id"] == portfolio_id)]
    if hit.empty:
        raise KeyError(f"{ref} is not in the ledger -- absorb the run first")
    row = hit.iloc[0]

    if f is not None and stake is not None:
        raise ValueError("pass f= or stake=, not both -- one is derived from the other")

    # The pot is resolved first because `stake=` needs it to produce a fraction.
    if pot is None:
        pot = balance(root)
        if not pot > 0:
            raise ValueError(
                "no balance to stake from -- call `ledger.open_account(amount)` "
                "first, or pass an explicit `pot=`")
    pot = float(pot)

    if stake is not None:
        f, mode = float(stake) / pot, CUSTOM
    elif f is not None:
        # An explicit fraction *is* the custom mode. Letting `mode` keep its
        # default here filed a hand-picked stake under "protective", which is a
        # record that says the model chose something it did not.
        mode = CUSTOM
    else:
        mode = mode or "protective"
        if mode == CUSTOM:
            raise ValueError("custom staking needs f= (a fraction) or stake= (cash)")
        if mode not in STAKE_MODES:
            raise ValueError(
                f"mode must be {', '.join(STAKE_MODES)} or {CUSTOM}; got {mode!r}")
        f = row[STAKE_MODES[mode]]
        if pd.isna(f):
            raise ValueError(f"{ref} has no {mode} stake -- it carries no growth block")
        f = float(f)

    if not 0 < f <= 1:
        raise ValueError(f"stake fraction must be above 0 and at most 1; got {f:.4g}")
    cash_stake = pot * f

    placed = read_table("placed", root)
    if not placed.empty:
        placed = placed[placed["run_code"] != run_code]
    rec = pd.DataFrame([{
        "run_code": run_code, "portfolio_id": portfolio_id,
        "pot": pot, "mode": mode, "f": f, "stake": cash_stake,
        "recorded_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }])
    write_table(pd.concat([placed, rec], ignore_index=True) if not placed.empty else rec,
                "placed", root)
    print(f"Recorded {ref}: {mode} stake {f:.1%} of {pot:,.2f} = {cash_stake:,.2f}")
    return rec.iloc[0]


def bankroll(root: Path | None = None) -> pd.DataFrame:
    """A bank statement: every cash movement and every bet, in order, with a
    running balance.

    One chronological table rather than a cash table beside a bets table,
    because the question it answers -- what is the pot now, and what moved it --
    is answered by the interleaving. Deposits show up beside the bets they
    funded, which is the only way a rising balance can be told apart from a
    winning run.

    A bet moves the balance by `stake * realised_return`, once its legs have all
    settled. Until then the row is `pending` and the balance is unchanged: the
    stake is with the bookmaker and the outcome is unknown, so booking it either
    way would be a guess dressed as a number.
    """
    cash = read_table("cash", root)
    placed = read_table("placed", root)
    portfolios = read_table("portfolios", root)
    runs = read_table("runs", root)

    events = []
    if not cash.empty:
        for r in cash.to_dict("records"):
            signed = -r["amount"] if r["kind"] == "withdrawal" else r["amount"]
            events.append({"when": r["when"], "kind": r["kind"], "ref": "",
                           "cash": signed, "note": r.get("note") or ""})

    if not placed.empty:
        cols = ["run_code", "portfolio_id", "realised_return", "settled", "legs", "split",
                "expected_return_pct", "g_g_protective"]
        bets = placed.merge(portfolios[[c for c in cols if c in portfolios]],
                            on=["run_code", "portfolio_id"], how="left")
        if not runs.empty:
            bets = bets.merge(runs[["run_code", "run_date"]], on="run_code", how="left")
        for r in bets.to_dict("records"):
            done = bool(r.get("settled")) and pd.notna(r.get("realised_return"))
            events.append({
                "when": str(r.get("run_date") or r.get("recorded_at", ""))[:10],
                "kind": "bet", "ref": f"{r['run_code']}-{int(r['portfolio_id'])}",
                "cash": 0.0, "note": r.get("split") or "",
                "mode": r.get("mode"), "f": r.get("f"), "pot": r.get("pot"),
                "stake": r.get("stake"), "legs": r.get("legs"),
                "expected_return_pct": r.get("expected_return_pct"),
                "realised_return": r.get("realised_return") if done else np.nan,
                "pending": not done,
            })

    if not events:
        return pd.DataFrame()

    out = pd.DataFrame(events)
    for col in ("stake", "realised_return", "pot", "f", "pending", "mode"):
        if col not in out:
            out[col] = np.nan if col != "pending" else False
    out["pending"] = out["pending"].fillna(False).astype(bool)
    out["_rank"] = out["kind"].map(_CASH_RANK).fillna(9)
    out = out.sort_values(["when", "_rank"], kind="stable").drop(columns="_rank")

    out["pnl"] = out["stake"] * out["realised_return"]
    out["returned"] = out["stake"] * (1.0 + out["realised_return"])
    # A pending bet contributes nothing to the balance, and NaN would poison the
    # cumulative sum for every row after it.
    out["balance"] = (out["cash"].fillna(0.0) + out["pnl"].fillna(0.0)).cumsum()
    lead = ["when", "kind", "ref", "note", "cash", "pot", "mode", "f", "stake",
            "expected_return_pct", "realised_return", "returned", "pnl", "pending", "balance"]
    return out[[c for c in lead if c in out]].reset_index(drop=True)


__all__ = [
    "TABLES", "STAKE_MODES", "CUSTOM", "LEG_AGGREGATES", "Slate",
    "store_dir", "table_path", "read_table", "write_table",
    "parse_label", "league_of", "prop_key", "next_run_code", "pending_slates",
    "proposition_rows", "capture", "slate_from_payloads", "backfill",
    "absorb", "purge", "active_runs", "pull_results", "load_results",
    "match_fixtures",
    "settle", "roll_up", "record_bet", "bankroll",
    "open_account", "deposit", "withdraw", "balance",
]
