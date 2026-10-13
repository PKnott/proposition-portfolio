"""The league model's view of a European fixture: "what would this be as a league game?"

Each frozen league model predicts one team's count from that team's rolling
priors, its opponent's rolling priors, venue and schedule. None of the frozen
selections use the `league` feature, so a European fixture can be put to the
league model directly, once two questions are answered:

**What are the priors at a European date?** The buffers are computed
record-before-update on domestic rows only, so the state of a club's buffers at a
European date is exactly the prior carried by its *next* domestic row: no domestic
match falls between the two, by definition of "next". Venue buffers read the next
domestic row at the same venue, because a match at the other venue in between does
not enter a venue buffer either. Nothing is inserted into the buffer table. Buffers
count age in rows, so an inserted row would age every observation behind it, and
the league model's table must not be touched anyway.

**What are the opponent's priors worth in this team's league?** Converted at par:
``prior_j x rate[L_i] / rate[L_j]`` for the same (stat, side, venue), where `rate`
is the league's mean over the previous completed season. That assumes the leagues
are equally strong. They are not, and measuring by how much is the job of the
European layer that sits on top (`fpp.europe.layer`); par is the neutral starting
point it corrects.

**Which model?** For a European match in season *s*, a league model trained on
seasons strictly before *s*, at the frozen settings. The production model has seen
the domestic matches played after that European fixture, and an offset built from
it would carry them into the European fit.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..artifacts import RunContext
from ..build import build_feature_table, scorable_mask
from ..clean import attach_schedule_context
from ..config import season_sort_key
from ..models import fit_model, predict_mu
from ..priors import BufferWindows
from ..spec import TARGETS
from .config import EURO_ARTIFACTS

COUNTS = ("goals", "shots", "sot", "corners")

# How far ahead a club's next domestic row may be and still stand for its state
# now. The summer break is the longest legitimate gap (a final in early June to a
# mid-August opener). Past this, the club has left the five leagues.
MAX_LOOKAHEAD = pd.Timedelta(days=120)

_PRIOR = re.compile(r"^(team|opp)_avg_(?P<stat>[a-z]+)_(?P<side>for|against)(?P<v>_v)?$")


def _year(season) -> int:
    return season_sort_key(season)


# --- European team rows -------------------------------------------------------

def membership(clean: pd.DataFrame) -> dict[tuple[str, int], str]:
    """``{(team, season start year): league_key}`` for every club-season in the five leagues."""
    m = clean[["team", "season", "league_key"]].drop_duplicates(["team", "season"])
    return {(t, _year(s)): lk for t, s, lk in zip(m["team"], m["season"], m["league_key"])}


def team_rows(stats: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """European matches between two clubs from the five leagues, one row per team.

    A club counts only if it plays in a modelled league *that season*. A club
    relegated the season before still gets a European place sometimes, and it has
    no current domestic priors to read.
    """
    member = membership(clean)
    s = stats.copy()
    s["year"] = s["season"].map(_year)
    s["home_league"] = [member.get((t, y)) for t, y in zip(s["home_team"], s["year"])]
    s["away_league"] = [member.get((t, y)) for t, y in zip(s["away_team"], s["year"])]
    s = s[s["home_league"].notna() & s["away_league"].notna()]

    frames = []
    for venue, other in (("home", "away"), ("away", "home")):
        cols = {
            "fixture_id": s["espn_event_id"].astype(str), "date": pd.to_datetime(s["date"]),
            "season": s["season"], "year": s["year"], "comp": s["league_key"],
            "stage": s["stage"], "leg": s["leg"],
            "team": s[f"{venue}_team"], "opponent": s[f"{other}_team"],
            "team_league": s[f"{venue}_league"], "opp_league": s[f"{other}_league"],
            "is_home": int(venue == "home"),
        }
        for stat in COUNTS:
            cols[f"{stat}_for"] = s[f"{venue}_{stat}"].to_numpy()
            cols[f"{stat}_against"] = s[f"{other}_{stat}"].to_numpy()
        frames.append(pd.DataFrame(cols))
    out = pd.concat(frames, ignore_index=True)
    out["neutral"] = (out["stage"] == "final").astype(int)
    return out.sort_values(["date", "fixture_id", "is_home"], ascending=[True, True, False]).reset_index(drop=True)


# --- As-of priors -------------------------------------------------------------

def _parse(name: str) -> dict | None:
    m = _PRIOR.match(name)
    return m.groupdict() | {"who": m.group(1)} if m else None


def _team_name(name: str) -> str:
    """The team-side buffer an opponent feature is read from: ``opp_avg_x`` -> ``team_avg_x``."""
    return "team_" + name[len("opp_"):] if name.startswith("opp_avg_") else name


def lookup_features(selected: list[str]) -> list[str]:
    """Team-side buffer features that must be looked up to build `selected`."""
    need = {_team_name(f) for f in selected if _parse(f)}
    return sorted(need)


def asof_priors(rows: pd.DataFrame, dom: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Each team's buffers as of each European date, read off its next domestic row.

    ``dom`` is a domestic feature frame (meta + the ``team_avg_*`` columns in
    ``cols``). All-venue columns come from the next domestic row at any venue;
    ``_v`` columns from the next one at the same venue as the European match.
    Returned in ``rows`` order, NaN where a club has no next domestic row within
    `MAX_LOOKAHEAD`.
    """
    left = rows[["team", "date", "is_home"]].copy()
    left["_i"] = np.arange(len(left))
    left = left.sort_values("date")
    right = dom[["team", "date", "is_home", *cols]].sort_values("date")

    out = pd.DataFrame(index=np.arange(len(rows)))
    for venue_cols, by in (([c for c in cols if not c.endswith("_v")], ["team"]),
                           ([c for c in cols if c.endswith("_v")], ["team", "is_home"])):
        if not venue_cols:
            continue
        m = pd.merge_asof(left, right[[*by, "date", *venue_cols]], on="date", by=by,
                          direction="forward", allow_exact_matches=False, tolerance=MAX_LOOKAHEAD)
        out.loc[m["_i"].to_numpy(), venue_cols] = m[venue_cols].to_numpy()
    return out[cols]


# --- Par conversion -------------------------------------------------------------

def league_rates(clean: pd.DataFrame) -> dict[tuple, float]:
    """``{(league, season year, stat, side, venue): mean}``, venue in {1, 0, 'all'}."""
    df = clean.copy()
    df["year"] = df["season"].map(_year)
    out: dict[tuple, float] = {}
    stats = ("goals", "xg", "npxg", "shots", "sot", "corners")
    cols = [f"{s}_{side}" for s in stats for side in ("for", "against") if f"{s}_{side}" in df.columns]
    for (lk, y), g in df.groupby(["league_key", "year"]):
        means = {"all": g[cols].mean()}
        for v, gv in g.groupby("is_home"):
            means[int(v)] = gv[cols].mean()
        for venue, m in means.items():
            for c in cols:
                stat, side = c.rsplit("_", 1)
                out[(lk, int(y), stat, side, venue)] = float(m[c])
    return out


def par_ratio(rates: dict, to_league: str, from_league: str, year: int,
              stat: str, side: str, venue) -> float:
    """``rate[to] / rate[from]`` over the previous completed season; 1.0 when unknown."""
    a = rates.get((to_league, year - 1, stat, side, venue))
    b = rates.get((from_league, year - 1, stat, side, venue))
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b) or b <= 0:
        return 1.0
    return a / b


# --- Feature rows -------------------------------------------------------------

def feature_rows(rows: pd.DataFrame, priors: pd.DataFrame, selected: list[str],
                 rates: dict, clean: pd.DataFrame) -> pd.DataFrame:
    """The league model's feature matrix for European team rows, in this team's league frame."""
    partner = (rows.groupby("fixture_id").cumcount() == 0).to_numpy()
    # each fixture is two adjacent rows after `team_rows`' sort; index of the other one
    idx = np.arange(len(rows))
    other = np.where(partner, idx + 1, idx - 1)
    assert (rows["fixture_id"].to_numpy()[other] == rows["fixture_id"].to_numpy()).all()

    X = pd.DataFrame(index=rows.index)
    for f in selected:
        p = _parse(f)
        if p is None:
            continue
        src = priors[_team_name(f)].to_numpy(dtype=float)
        if p["who"] == "team":
            X[f] = src
            continue
        val = src[other]
        # the opponent's buffer at the opponent's venue, converted into this team's league
        opp_home = rows["is_home"].to_numpy()[other]
        ratio = np.array([
            par_ratio(rates, li, lj, y, p["stat"], p["side"], int(oh) if p["v"] else "all")
            for li, lj, y, oh in zip(rows["team_league"], rows["opp_league"], rows["year"], opp_home)
        ])
        X[f] = val * ratio

    sched = attach_schedule_context(rows[["fixture_id", "team", "date", "is_home"]])
    new = (clean.assign(year=clean["season"].map(_year))
           .groupby(["team", "year"])["team_is_new_to_league"].max())
    for f in selected:
        if f in X.columns:
            continue
        if f == "is_home":
            X[f] = rows["is_home"].to_numpy()
        elif f in sched.columns:
            X[f] = sched[f].to_numpy()
        elif f == "team_is_new_to_league":
            X[f] = [new.get((t, y), 0) for t, y in zip(rows["team"], rows["year"])]
        elif f == "opp_is_new_to_league":
            X[f] = [new.get((t, y), 0) for t, y in zip(rows["opponent"], rows["year"])]
        else:
            raise KeyError(f"no European construction for league-model feature {f!r}")
    return X[selected].astype(float)


def feature_matrix(rows: pd.DataFrame, table: pd.DataFrame, clean: pd.DataFrame,
                   ctx: RunContext, target: str, rates: dict | None = None,
                   bw: BufferWindows | None = None) -> pd.DataFrame:
    """One target's league-model features for European rows, read off ``table``.

    ``table`` is the domestic team-match table the buffers run over: the clean
    table for history, or the clean table with unplayed league fixtures appended
    (`predict.append_fixtures_as_rows`) for live fixtures -- which is what gives a
    club a "next domestic row" after its last played match. ``clean`` supplies the
    promoted flags and the league rates, which only ever need played matches.
    """
    w, feats = ctx.windows[target], ctx.features[target]
    bw = bw or BufferWindows(table, L_max=max(50, int(w["L"])))
    need = lookup_features(feats)
    look = build_feature_table(table, bw, int(w["L"]), float(w["alpha"]),
                               features=need, drop_warmup=False)
    dom = pd.concat([look.meta[["team", "date", "is_home"]], look.X[need]], axis=1)
    dom["date"] = pd.to_datetime(dom["date"])
    rates = rates if rates is not None else league_rates(clean)
    return feature_rows(rows, asof_priors(rows, dom, need), feats, rates, clean)


# --- Offsets ------------------------------------------------------------------

def offsets(stats: pd.DataFrame, clean: pd.DataFrame, ctx: RunContext | None = None,
            targets: tuple[str, ...] = TARGETS, seasons: list[int] | None = None,
            cache: bool = True) -> dict[str, pd.DataFrame]:
    """``{target: rows + mu_dom + actual}`` for every European top-five fixture.

    ``mu_dom`` for season *s* comes from a league model fitted on domestic
    seasons ``< s`` at the frozen settings. Cached per artifact version under
    `EURO_ARTIFACTS`, because it is ~10 fits per target.
    """
    ctx = ctx or RunContext.load()
    rows = team_rows(stats, clean)
    if seasons is not None:
        rows = rows[rows["year"].isin(seasons)].reset_index(drop=True)
    rates = league_rates(clean)
    out: dict[str, pd.DataFrame] = {}
    for t in targets:
        path = EURO_ARTIFACTS / ctx.version / f"offsets_{t}.parquet"
        if cache and path.exists():
            cached = pd.read_parquet(path)
            if set(rows["fixture_id"]) <= set(cached["fixture_id"]):
                out[t] = cached[cached["fixture_id"].isin(set(rows["fixture_id"]))].reset_index(drop=True)
                continue

        w, feats = ctx.windows[t], ctx.features[t]
        bw = BufferWindows(clean, L_max=max(50, int(w["L"])))
        X = feature_matrix(rows, clean, clean, ctx, t, rates, bw)

        train = build_feature_table(clean, bw, int(w["L"]), float(w["alpha"]), features=feats)
        m = scorable_mask(train, t)
        ty = train.meta["season"].map(_year).to_numpy()
        params = dict(ctx.params[t])
        params["n_estimators"] = ctx.n_estimators[t]
        params.pop("early_stopping_rounds", None)

        res = rows.copy()
        res["actual"] = res[f"{t}_for"]
        res["mu_dom"] = np.nan
        ok = X.notna().all(axis=1).to_numpy()
        for y in sorted(res["year"].unique()):
            fit_rows = m & (ty < y)
            if fit_rows.sum() == 0:
                # Production training drops warm-up rows, which removes the first
                # domestic season entirely -- so the first European season has no
                # leak-free league model, and stays unscored rather than borrowing one.
                print(f"  {t:8} {y}/{y + 1}: no earlier domestic season to fit on -- unscored")
                continue
            score = ok & (res["year"].to_numpy() == y)
            if not score.any():
                continue
            model = fit_model(train.X.loc[fit_rows, feats], train.y[t][fit_rows], params)
            res.loc[score, "mu_dom"] = predict_mu(model, X.loc[score, feats])
            print(f"  {t:8} {y}/{y + 1}: fitted on {int(fit_rows.sum()):,} domestic rows, "
                  f"scored {int(score.sum())} European rows")
        for f in feats:
            res[f"x_{f}"] = X[f].to_numpy()
        if cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            res.to_parquet(path, index=False)
        out[t] = res
    return out


__all__ = ["COUNTS", "MAX_LOOKAHEAD", "membership", "team_rows", "lookup_features", "asof_priors", "feature_matrix",
           "league_rates", "par_ratio", "feature_rows", "offsets"]
