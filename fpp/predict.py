"""Production training and fixture scoring.

The important structural point is how upcoming fixtures get their features.

The old pipeline had ``latest_team_priors_before_date``, which returned the prior
*recorded at* a team's last played match -- and that prior was computed
*excluding* that match's own result. Upcoming-fixture priors were therefore one
match stale. It also had ``predict_lambdas_from_priors``, which hand-assembled
feature dicts and filled unknown columns with ``.get(c, np.nan)``, so any rename
silently produced a NaN column instead of an error.

Both functions are gone. Instead, upcoming fixtures are appended to the
team-match table as rows with unknown outcomes and pushed through the *same*
buffer pass as history. Record-before-update on such a row is, by construction,
the correct as-of-now prior -- and there is exactly one code path from
``(team_matches, L, alpha)`` to a feature matrix, used identically in tuning and
in production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .artifacts import RunContext
from .build import build_feature_table, scorable_mask
from .clean import SCHEDULE_COLS, attach_game_week, attach_movement_flags, attach_schedule_context
from .config import LEAGUES, canonical_team, season_sort_key
from .models import fit_model, predict_mu
from .paths import FIXTURES_DIR
from .priors import BufferWindows
from .spec import STATS, TARGETS

STAT_COLS = [f"{s.key}_{side}" for s in STATS for side in ("for", "against")]

# Seasons a team's most recent appearance may be behind the current one before
# its priors stop being worth trusting. Two, so last season and the one before it
# are fine and anything older is called out. Deportivo and Malaga last played a
# league we cover in 2017/18 and Hull in 2016/17 -- mapping those clubs
# successfully hands the model eight-year-old form, which is a different kind of
# wrong from having no form at all, and quieter.
MAX_PRIOR_AGE = 2


def load_upcoming_fixtures(
    date_from: str | pd.Timestamp | None = None,
    date_to: str | pd.Timestamp | None = None,
    league_keys: list[str] | None = None,
) -> pd.DataFrame:
    """Read the per-league fixture CSVs Data Pull writes, as one frame."""
    frames = []
    for key in (league_keys or list(LEAGUES)):
        lg = LEAGUES[key]
        p = FIXTURES_DIR / f"{lg.fixtures_stem}_fixtures.csv"
        if not p.exists():
            print(f"  no fixtures file for {key} ({p.name}) -- skipped")
            continue
        f = pd.read_csv(p)
        f = f.rename(columns={
            "Date": "date", "Home Team": "home_team",
            "Away Team": "away_team", "Start Time (UTC)": "kickoff_utc",
        })
        f["league_key"] = key
        f["league"] = lg.display
        frames.append(f)

    if not frames:
        return pd.DataFrame(columns=["date", "home_team", "away_team", "league_key", "league"])

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # The club mapping is applied *again* here, not just when `pull_fixtures`
    # wrote the file. The CSV is a cache of an already-mapped result, so a fix to
    # the mapping would otherwise not reach the fixtures until the next pull --
    # and a club whose ESPN name has changed is exactly the case where the file
    # on disk is the stale thing. Re-applying is safe because the map is
    # idempotent: no Understat name is also an ESPN name pointing somewhere else,
    # so a name that has already been mapped maps to itself.
    try:
        from .ingest.espn import load_name_map

        name_map = load_name_map()
    except (FileNotFoundError, OSError, KeyError):
        name_map = {}  # no mapping on disk: fall through to the raw names
    for c in ("home_team", "away_team"):
        df[c] = df[c].map(lambda x: canonical_team(name_map.get(x, x)))
    if date_from is not None:
        df = df[df["date"] >= pd.Timestamp(date_from)]
    if date_to is not None:
        df = df[df["date"] <= pd.Timestamp(date_to)]
    return df.sort_values(["date", "league_key", "home_team"]).reset_index(drop=True)


def check_fixture_coverage(fixtures: pd.DataFrame, team_matches: pd.DataFrame,
                           max_age: int = MAX_PRIOR_AGE, verbose: bool = True) -> dict:
    """Which fixture teams the model has no usable history for. Returns a report.

    Two failure modes, and they are different:

    * **missing** -- the name appears nowhere in the clean table. Every prior is
      NaN, so `build_feature_table`'s cold start seeds the whole team from the
      league's **promoted** profile, not its average. A club with no history is
      necessarily flagged `is_new_to_league` once its fixture rows are appended --
      present in (league, this season), absent in (league, last season) -- so it
      takes the promoted branch of `Seeds.value` by construction. This message
      used to claim the league average, which understated the seeding badly: the
      Bundesliga's promoted profile is 1.171 goals for and 1.906 against, against
      a league average of 1.618 either way.

      What is genuinely lost is the *per-club* scaling. `fpp.promoted` would
      narrow the blanket profile using the club's own record in the division
      below, and cannot when the club has no mapped name to find it under -- which
      is what the mapping hint below is for.
    * **stale** -- the club is there, but last appeared more than `max_age`
      seasons ago. Its priors are real numbers from a squad that no longer
      exists.

    Neither raises, and neither used to be *detectable*. Nothing checked that a
    fixture team existed in the history: an unmapped club fell straight through
    to the league-average seed, and the NaN assertion at the end of
    `score_fixtures` could not catch it because the seeding is what prevents the
    NaN. So a mis-mapped club produced a confident, plausible, meaningless
    prediction, and the run said nothing.

    Each name comes with its closest known match, so the usual fix -- a row in
    `Club_Mapping_All.csv` -- does not start with guessing what the club is
    called on the other side.
    """
    from .ingest.espn import best_match
    from .ingest.understat import known_team_names

    if fixtures.empty:
        return {"missing": [], "stale": [], "ok": 0}

    used = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))
    years = team_matches["season"].map(season_sort_key)
    last = years.groupby(team_matches["team"]).max()
    label = team_matches.assign(_y=years).sort_values("_y").groupby("team")["season"].last()
    latest = int(years.max())

    # Age is the gap in *years*, not the number of seasons in between. Those
    # coincide while the table is contiguous, and stop coinciding the moment one
    # is not -- at which point rank distance would silently under-report.
    missing, stale = [], []
    for team in used:
        if team not in last.index:
            missing.append(team)
        elif latest - int(last[team]) > max_age:
            stale.append((team, str(label[team])))

    if verbose and (missing or stale):
        pool = known_team_names()
        if missing:
            print(f"  {len(missing)} fixture team(s) have NO history and will be scored "
                  f"from the league's promoted profile alone, with no per-club scaling:")
            for team in missing:
                guess = best_match(team, pool)
                hint = f"closest known: {guess[0]!r} ({guess[1]:.2f})" if guess else "no pool to compare"
                print(f"    {team} -- {hint}")
        if stale:
            print(f"  {len(stale)} fixture team(s) have history older than {max_age} season(s):")
            for team, season in stale:
                print(f"    {team} -- last seen {season}")
        print("  fix a missing team with a row in Inputs/Mapping/Club_Mapping_All.csv "
              "(or wait for Understat to publish it), then rebuild the clean table")
    elif verbose:
        print(f"  all {len(used)} fixture teams have history within {max_age} season(s)")

    return {"missing": missing, "stale": stale, "ok": len(used) - len(missing) - len(stale)}


def append_fixtures_as_rows(team_matches: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Extend the team-match table with unplayed fixtures, marked ``is_scored=0``.

    Every stat is NaN on these rows, so they contribute nothing to any buffer --
    but they still *read* the buffers, which is exactly what scoring needs.
    """
    if fixtures.empty:
        out = team_matches.copy()
        out["is_scored"] = 1
        return out

    hist = team_matches.copy()
    hist["is_scored"] = 1

    season = hist["season"].max()
    next_fid = int(hist["fixture_id"].max()) + 1
    rows = []
    for i, fx in enumerate(fixtures.itertuples()):
        fid = next_fid + i
        for is_home, (team, opp) in enumerate([(fx.away_team, fx.home_team), (fx.home_team, fx.away_team)]):
            rec = {
                "fixture_id": fid, "date": fx.date, "season": season,
                "league": fx.league, "league_key": fx.league_key,
                "team": team, "opponent": opp, "is_home": is_home,
                "has_espn_stats": 0, "is_scored": 0,
            }
            for c in STAT_COLS:
                rec[c] = np.nan
            rows.append(rec)

    combined = pd.concat([hist, pd.DataFrame(rows)], ignore_index=True)
    combined = combined.sort_values(["date", "fixture_id", "is_home"], ascending=[True, True, False]).reset_index(drop=True)

    # Recompute the derived context so the new rows get real values, not NaN.
    combined = attach_game_week(combined)
    combined = attach_movement_flags(combined)
    # Recomputed over history *and* fixtures, not carried over. The unplayed rows
    # have dates, so both the gap since the last match and the count of matches
    # before it are knowable -- and they were previously left NaN on exactly the
    # rows being predicted, which made the schedule features dead weight in
    # production while looking fine in training.
    combined = combined.drop(columns=[c for c in combined.columns
                                      if c.endswith(SCHEDULE_COLS)], errors="ignore")
    combined = attach_schedule_context(combined)
    return combined


def train_production(
    team_matches: pd.DataFrame, ctx: RunContext, targets: tuple[str, ...] = TARGETS
) -> dict[str, dict]:
    """Retrain every target family on the full history, at frozen settings.

    Deliberately includes the seasons held out during research. That period
    existed to get an honest read on the model, not to be preserved forever --
    once a model is evaluated and accepted there is no reason to keep starving it
    of the most recent matches.
    """
    models: dict[str, dict] = {}
    for t in targets:
        w = ctx.windows[t]
        feats = ctx.features[t]
        bw = BufferWindows(team_matches, L_max=max(50, int(w["L"])))
        ft = build_feature_table(team_matches, bw, int(w["L"]), float(w["alpha"]), features=feats)

        m = scorable_mask(ft, t)
        params = dict(ctx.params[t])
        params["n_estimators"] = ctx.n_estimators[t]
        params.pop("early_stopping_rounds", None)

        model = fit_model(ft.X.loc[m][feats], ft.y[t][m], params)
        models[t] = {"model": model, "features": feats, "window": w,
                     "dispersion": ctx.dispersion.get(t, {})}
        print(f"  {t:8} trained on {int(m.sum()):,} rows, {len(feats)} features, "
              f"{params['n_estimators']} trees")
    return models


def score_fixtures(
    team_matches: pd.DataFrame, fixtures: pd.DataFrame, models: dict[str, dict]
) -> pd.DataFrame:
    """Predict every target family for each upcoming fixture.

    Returns one row per fixture with ``mu_home``/``mu_away`` per target family.
    """
    if fixtures.empty:
        return pd.DataFrame()

    combined = append_fixtures_as_rows(team_matches, fixtures)
    upcoming = combined["is_scored"] == 0

    out = combined.loc[upcoming, ["fixture_id", "date", "league_key", "league", "team", "opponent", "is_home"]].copy()

    for t, spec in models.items():
        feats, w = spec["features"], spec["window"]
        bw = BufferWindows(combined, L_max=max(50, int(w["L"])))
        ft = build_feature_table(combined, bw, int(w["L"]), float(w["alpha"]),
                                 features=feats, drop_warmup=False)
        mu = predict_mu(spec["model"], ft.X[feats])
        out[f"mu_{t}"] = mu[upcoming.to_numpy()]

    home = out[out["is_home"] == 1].set_index("fixture_id")
    away = out[out["is_home"] == 0].set_index("fixture_id")
    common = home.index.intersection(away.index)

    res = pd.DataFrame({
        "fixture_id": common,
        "date": home.loc[common, "date"].to_numpy(),
        "league_key": home.loc[common, "league_key"].to_numpy(),
        "league": home.loc[common, "league"].to_numpy(),
        "home_team": home.loc[common, "team"].to_numpy(),
        "away_team": away.loc[common, "team"].to_numpy(),
    })
    for t in models:
        res[f"{t}_home"] = home.loc[common, f"mu_{t}"].to_numpy()
        res[f"{t}_away"] = away.loc[common, f"mu_{t}"].to_numpy()

    assert not res[[c for c in res.columns if c.endswith(("_home", "_away")) and c != "home_team"]].isna().any().any(), \
        "NaN predictions -- a feature is missing from the scoring path"
    return res.sort_values(["date", "league_key", "home_team"]).reset_index(drop=True)


__all__ = [
    "load_upcoming_fixtures", "check_fixture_coverage", "append_fixtures_as_rows",
    "MAX_PRIOR_AGE",
    "train_production", "score_fixtures",
]
