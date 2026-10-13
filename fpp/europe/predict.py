"""Live European predictions, in the shape the rest of the pipeline already reads.

`score` returns the frame `fpp.predict.score_fixtures` returns -- one row per
fixture, ``{target}_home`` / ``{target}_away`` means -- with ``league_key`` set to
the competition (`UCL`, `UEL`, `UECL`). From there the domestic report, staking,
portfolio, ledger and Edge Book code runs unchanged; this module only decides
the inputs it is handed:

* **dispersion** keyed by competition, fitted on European residuals;
* **baselines** from a European team-match table, so a fixture is coloured
  against that competition's realised rates rather than a domestic league's.

How a fixture is priced
-----------------------
1. The production league models (`predict.train_production`, exactly what
   `05_Run` trains) predict each side as though it were a league game, with the
   opponent converted at par -- `fpp.europe.model`.
2. The European layer, fitted on every European season through the last
   completed one, applies the exchange rates -- `fpp.europe.layer`, at
   `config.PRODUCTION`. Corners stay at par.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..artifacts import RunContext
from ..clean import load_clean_table
from ..config import LEAGUES, season_sort_key
from ..ingest.espn import load_name_map
from ..ingest.retry import ESPN_SITE_API, get_json
from ..models import predict_mu
from ..predict import append_fixtures_as_rows
from ..spec import TARGETS, target_spec
from . import ingest, layer, model
from .config import EURO_CACHE, EURO_COMPS, PRODUCTION

# --- Fixtures -----------------------------------------------------------------


def _fixture_record(event: dict, comp: str) -> dict | None:
    comps = event.get("competitions") or [{}]
    sides = {c.get("homeAway"): (c.get("team") or {}).get("displayName")
             for c in comps[0].get("competitors") or []}
    if set(sides) != {"home", "away"}:
        return None
    stage, leg, year = ingest._stage(event)
    if year is None:
        return None
    return {
        "espn_event_id": str(event.get("id")), "comp": comp,
        "date": pd.Timestamp((event.get("date") or "")[:10]),
        "kickoff_utc": event.get("date"), "season": f"{year}/{year + 1}", "year": year,
        "stage": stage, "leg": leg, "home_espn": sides["home"], "away_espn": sides["away"],
        "status": ((event.get("status") or {}).get("type") or {}).get("name"),
    }


def fixtures(date_from, date_to, *, refresh: bool = True) -> pd.DataFrame:
    """Every European fixture between two dates, mapped to Understat names where possible.

    Fetched fresh by default: a cached scoreboard from before a draw or a
    rescheduling is exactly the stale input to avoid. The files land in the
    European cache, where `ingest.build_match_stats` will later read the results.
    """
    days = pd.date_range(pd.Timestamp(date_from), pd.Timestamp(date_to), freq="D")
    EURO_CACHE.mkdir(parents=True, exist_ok=True)
    recs = []
    for comp, c in EURO_COMPS.items():
        for d in days:
            path = EURO_CACHE / f"Schedule_{c.espn_slug}_{d:%Y%m%d}.json"
            if refresh or not path.exists():
                payload = get_json(f"{ESPN_SITE_API}/{c.espn_slug}/scoreboard?dates={d:%Y%m%d}")
                path.write_text(json.dumps(payload))
            else:
                payload = json.loads(path.read_text())
            for ev in payload.get("events") or []:
                rec = _fixture_record(ev, comp)
                if rec is not None:
                    recs.append(rec)
    fx = pd.DataFrame(recs).drop_duplicates("espn_event_id") if recs else pd.DataFrame()
    if fx.empty:
        return fx
    names = load_name_map()
    fx["home_team"] = fx["home_espn"].map(names)
    fx["away_team"] = fx["away_espn"].map(names)
    return fx.sort_values(["date", "comp", "home_espn"]).reset_index(drop=True)


def eligible(fx: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """Fixtures between two clubs from the five leagues this season, with each club's league."""
    if fx.empty:
        return fx
    member = model.membership(clean)
    out = fx.copy()
    out["home_league"] = [member.get((t, y)) for t, y in zip(out["home_team"], out["year"])]
    out["away_league"] = [member.get((t, y)) for t, y in zip(out["away_team"], out["year"])]
    keep = out["home_league"].notna() & out["away_league"].notna()
    print(f"  {int(keep.sum())} of {len(out)} European fixture(s) are between two clubs "
          f"from the five leagues")
    return out[keep].reset_index(drop=True)


def _team_rows(fx: pd.DataFrame) -> pd.DataFrame:
    """Fixtures -> the team-row layout `model.feature_rows` reads (home row first)."""
    frames = []
    for venue, other in (("home", "away"), ("away", "home")):
        frames.append(pd.DataFrame({
            "fixture_id": fx["espn_event_id"].astype(str), "date": fx["date"],
            "season": fx["season"], "year": fx["year"], "comp": fx["comp"],
            "stage": fx["stage"], "leg": fx["leg"],
            "team": fx[f"{venue}_team"], "opponent": fx[f"{other}_team"],
            "team_league": fx[f"{venue}_league"], "opp_league": fx[f"{other}_league"],
            "is_home": int(venue == "home"),
        }))
    rows = pd.concat(frames, ignore_index=True)
    rows["neutral"] = (rows["stage"] == "final").astype(int)
    return rows.sort_values(["date", "fixture_id", "is_home"],
                            ascending=[True, True, False]).reset_index(drop=True)


# --- The European layer, fitted for production ------------------------------------


def production_layers(offs: dict[str, pd.DataFrame] | None = None,
                      settings: dict[str, float | None] = PRODUCTION) -> dict[str, layer.Layer]:
    """One fitted layer per target, on every European season with an offset."""
    offs = offs if offs is not None else model.offsets(ingest.load_match_stats(), load_clean_table())
    return {t: layer.fit(offs[t], t, alpha=a if a is not None else 0.0, offset_only=a is None)
            for t, a in settings.items()}


def dispersion(layers: dict[str, layer.Layer]) -> dict[str, dict[str, float]]:
    """``{target: {competition: r}}`` -- the shape `report.markets` reads.

    One European value per target, repeated per competition: the European sample
    cannot support a separate dispersion for each, and the Conference League's
    alone is a few dozen rows. Poisson targets carry the same "effectively
    Poisson" sentinel the league artifacts use.
    """
    out = {}
    for t, lay in layers.items():
        r = lay.dispersion if target_spec(t).dist == "nbinom" else 1e6
        out[t] = {c: float(r) for c in EURO_COMPS}
    return out


# --- Scoring ------------------------------------------------------------------------

_PLACEHOLDER = "__euro_placeholder__ {}"


def _live_table(clean: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """The clean table plus, per club, one unplayed home row and one unplayed away row.

    `model.asof_priors` reads a club's buffers off its next domestic row after the
    European date. Historically that row exists; for an upcoming fixture it may
    not, because the league fixture files hold only a run's own date range. No
    league match is played between a club's last result and the European night,
    so the buffers' current state *is* the answer -- and an unplayed row reads
    exactly that state, contributing nothing to any buffer.

    The home row comes a day before the away row, each against its own
    placeholder opponent. Each club's first placeholder row is therefore the next
    row at any venue, and the away row is the next row at its own venue with
    nothing unplayed in that venue's block ahead of it: neither read is aged by
    another placeholder. A private copy, used for the read and discarded.
    """
    after = pd.Timestamp(rows["date"].max()) + pd.Timedelta(days=1)
    clubs = rows[["team", "team_league"]].drop_duplicates("team")
    names = {lk: lg.display for lk, lg in LEAGUES.items()}
    fx = []
    for team, lk in zip(clubs["team"], clubs["team_league"]):
        ph = _PLACEHOLDER.format(team)
        fx.append({"date": after, "home_team": team, "away_team": ph, "league_key": lk, "league": names[lk]})
        fx.append({"date": after + pd.Timedelta(days=1), "home_team": ph, "away_team": team,
                   "league_key": lk, "league": names[lk]})
    return append_fixtures_as_rows(clean, pd.DataFrame(fx))



def score(fx: pd.DataFrame, clean: pd.DataFrame, models: dict, layers: dict[str, layer.Layer],
          ctx: RunContext | None = None) -> pd.DataFrame:
    """European fixtures -> the per-fixture prediction frame the pipeline reads.

    ``models`` is `predict.train_production(clean, ctx)` -- the league models
    `05_Run` trains, reused rather than refitted.
    """
    if fx.empty:
        return pd.DataFrame()
    ctx = ctx or RunContext.load()
    rows = _team_rows(fx)

    table = _live_table(clean, rows)
    rates = model.league_rates(clean)

    out = rows.copy()
    for t in TARGETS:
        X = model.feature_matrix(rows, table, clean, ctx, t, rates)
        if X.isna().any().any():
            bad = rows.loc[X.isna().any(axis=1), ["team", "date"]].drop_duplicates()
            raise ValueError(f"{t}: no domestic priors for {bad.to_dict('records')} -- "
                             f"is the league fixtures file current?")
        rows_t = rows.assign(mu_dom=predict_mu(models[t]["model"], X[ctx.features[t]]))
        out[f"mu_{t}"] = layers[t].mu(rows_t)

    home = out[out["is_home"] == 1].set_index("fixture_id")
    away = out[out["is_home"] == 0].set_index("fixture_id")
    common = [f for f in home.index if f in away.index]
    meta = fx.set_index(fx["espn_event_id"].astype(str))
    res = pd.DataFrame({
        "fixture_id": common,
        "date": home.loc[common, "date"].to_numpy(),
        "league_key": home.loc[common, "comp"].to_numpy(),
        "league": [EURO_COMPS[c].display for c in home.loc[common, "comp"]],
        "home_team": home.loc[common, "team"].to_numpy(),
        "away_team": away.loc[common, "team"].to_numpy(),
        "stage": meta.loc[common, "stage"].to_numpy(),
        "kickoff_utc": meta.loc[common, "kickoff_utc"].to_numpy(),
    })
    for t in TARGETS:
        res[f"{t}_home"] = home.loc[common, f"mu_{t}"].to_numpy()
        res[f"{t}_away"] = away.loc[common, f"mu_{t}"].to_numpy()
    assert not res[[f"{t}_{s}" for t in TARGETS for s in ("home", "away")]].isna().any().any()
    return res.sort_values(["date", "league_key", "home_team"]).reset_index(drop=True)


# --- Baselines ------------------------------------------------------------------------


def competition_table(stats: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every European match with full stats, one row per team, keyed by competition.

    The shape `report.markets.league_baselines` groups on, so each fixture is
    coloured against its own competition's realised rates. All European matches
    count here, not only the top-five ones: a baseline is a description of the
    competition, and the narrower sample would leave the Conference League's with
    a handful of matches.
    """
    stats = ingest.load_match_stats() if stats is None else stats
    frames = []
    for venue, other in (("home", "away"), ("away", "home")):
        cols = {"fixture_id": stats["espn_event_id"].astype(str), "season": stats["season"],
                "league_key": stats["league_key"], "date": stats["date"],
                "is_home": int(venue == "home")}
        for t in TARGETS:
            cols[f"{t}_for"] = stats[f"{venue}_{t}"].to_numpy()
            cols[f"{t}_against"] = stats[f"{other}_{t}"].to_numpy()
        frames.append(pd.DataFrame(cols))
    out = pd.concat(frames, ignore_index=True)
    out["_k"] = out["season"].map(season_sort_key)
    return out.sort_values(["_k", "date"]).drop(columns="_k").reset_index(drop=True)


__all__ = ["fixtures", "eligible", "production_layers", "dispersion", "score", "competition_table"]
