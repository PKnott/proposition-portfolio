"""Build the one canonical modelling table, and check it is sane.

Output is a *long* (team-perspective) frame: two rows per match, one per team,
carrying every stat both for and against. That shape is what the walk-forward
buffer engine consumes and what all four target families share.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .config import LEAGUE_RANK, LEAGUES, canonical_team, current_season_label, season_sort_key
from .ingest.scoreboard import build_match_stats
from .ingest.understat import load_matches
from .paths import CLEAN_CACHE, ESPN_MATCH_STATS, HISTORIC_ALL_COMP, all_comp_fixtures
from .spec import STATS

# --- Join -----------------------------------------------------------------


def attach_espn_stats(matches: pd.DataFrame, stats: pd.DataFrame | None = None) -> pd.DataFrame:
    """Left-join shots/SOT/corners onto the match table.

    Joined on ``(league, home_team, away_team)`` with a +/-1 day date tolerance:
    ESPN timestamps are UTC kickoff and can land either side of the Understat
    local date. Unmatched fixtures keep NaN stats and are flagged, never dropped
    here -- the goals model can still use them.
    """
    stats = stats if stats is not None else pd.read_csv(ESPN_MATCH_STATS, parse_dates=["date"])
    stat_cols = [f"{side}_{s}" for side in ("home", "away") for s in ("shots", "sot", "corners")]

    left = matches.copy()
    left["_row"] = np.arange(len(left))
    right = stats[["league_key", "date", "home_team", "away_team", *stat_cols]].copy()

    # ESPN's empty-stats sentinel: every count zero on both sides. `scoreboard`
    # rejects these at parse time, but an `espn_match_stats.csv` written before
    # that check existed still carries them, so drop them here too. Dropping from
    # `right` rather than nulling in place is deliberate -- the fixture then takes
    # the ordinary unmatched path below (NaN stats, `has_espn_stats` 0), instead of
    # claiming stats were joined and handing on a row of NaNs.
    sentinel = (right[stat_cols] == 0).all(axis=1)
    if sentinel.any():
        print(f"  Dropped {int(sentinel.sum()):,} all-zero ESPN row(s) before the join")
        right = right[~sentinel]

    found: dict[int, dict] = {}
    for shift in (0, -1, 1):
        pending = left[~left["_row"].isin(found)]
        if pending.empty:
            break
        r = right.copy()
        r["date"] = r["date"] + pd.Timedelta(days=shift)
        j = pending.merge(r, on=["league_key", "date", "home_team", "away_team"], how="inner")
        for rec in j.to_dict("records"):
            found.setdefault(rec["_row"], {c: rec[c] for c in stat_cols})

    for c in stat_cols:
        left[c] = left["_row"].map(lambda i, col=c: found.get(i, {}).get(col, np.nan))

    left["has_espn_stats"] = left["_row"].isin(found).astype(int)
    n = int(left["has_espn_stats"].sum())
    print(f"  ESPN stats joined to {n:,}/{len(left):,} matches ({100 * n / len(left):.1f}%)")
    return left.drop(columns=["_row"])


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per league-season join coverage -- the thing to look at when a gap appears."""
    g = df.groupby(["league_key", "season"]).agg(
        matches=("has_espn_stats", "size"), with_stats=("has_espn_stats", "sum")
    ).reset_index()
    g["pct"] = (100 * g["with_stats"] / g["matches"]).round(1)
    return g.sort_values(["league_key", "season"])


# --- Wide -> long ---------------------------------------------------------


def to_team_perspective(df: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match -- one per team -- with every stat for and against.

    Points are derived here (3/1/0) so they are available as a prior source.
    """
    df = df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    df = df.copy()
    df["fixture_id"] = np.arange(len(df))

    base = ["fixture_id", "date", "season", "league", "league_key"]
    # (our stat key) -> (home-side source column, away-side source column)
    src = {
        "goals": ("home_goals", "away_goals"),
        "xg": ("home_xg", "away_xg"),
        "npxg": ("home_np_xg", "away_np_xg"),
        "shots": ("home_shots", "away_shots"),
        "sot": ("home_sot", "away_sot"),
        "corners": ("home_corners", "away_corners"),
    }

    def side(is_home: bool) -> pd.DataFrame:
        me, opp = ("home", "away") if is_home else ("away", "home")
        out = df[base].copy()
        out["team"] = df[f"{me}_team"].values
        out["opponent"] = df[f"{opp}_team"].values
        out["is_home"] = int(is_home)
        out["has_espn_stats"] = df["has_espn_stats"].values
        for key, (hcol, acol) in src.items():
            out[f"{key}_for"] = df[hcol if is_home else acol].values
            out[f"{key}_against"] = df[acol if is_home else hcol].values
        gf, ga = out["goals_for"], out["goals_against"]
        out["points_for"] = np.where(gf > ga, 3, np.where(gf == ga, 1, 0))
        out["points_against"] = np.where(ga > gf, 3, np.where(gf == ga, 1, 0))
        return out

    long = pd.concat([side(True), side(False)], ignore_index=True)
    return long.sort_values(["date", "fixture_id", "is_home"], ascending=[True, True, False]).reset_index(drop=True)


# --- Derived context ------------------------------------------------------


def attach_game_week(df: pd.DataFrame) -> pd.DataFrame:
    """Approximate game week as the dense rank of match dates within a league-season.

    Understat does not ship a game-week column. The rank is a faithful stand-in
    for "how far into the season are we", which is all the normalised feature is
    used for.
    """
    out = df.copy()
    key = ["league_key", "season"]
    gw = (
        out.groupby(key)["date"].rank(method="dense").astype(int).rename("game_week")
    )
    out["game_week"] = gw
    max_gw = out.groupby(key)["game_week"].transform("max")
    out["game_week_normalised"] = out["game_week"] / max_gw
    return out


# Window, in days, for the fixture-congestion count. Fourteen because it spans a
# midweek-plus-weekend block either side of the fixture, which is what "congested"
# actually means here; a shorter window collapses into `rest_days` and a longer
# one is measuring the calendar rather than the schedule.
CONGESTION_DAYS = 14

# The per-row schedule columns, before they are split into team_/opp_ pairs.
SCHEDULE_COLS = ("rest_days", "matches_14d")


def attach_schedule_context(df: pd.DataFrame) -> pd.DataFrame:
    """Days since the last match, and matches in the previous fortnight.

    Both from the all-competitions fixture history the pipeline already pulls,
    because a league-only view of a team's schedule is not its schedule.

    Why there are two of these
    --------------------------
    ``rest_days`` was here first and does not measure what its name says. Against
    goals scored it correlates at **-0.033** -- more rest, *fewer* goals -- which
    is backwards for fatigue and exactly right for team quality: the sides
    playing every three days are the sides in Europe. Demean the outcome within
    team-season and the correlation is **+0.002**: the effect vanishes entirely
    once you condition on who is playing. The team-minus-opponent differential is
    stronger at -0.082 and still backwards, being a sharper quality-differential
    proxy.

    That explains a standing puzzle in the feature search. ``rest_days`` survives
    stage 1, which tests each group in isolation, because with nothing else
    present a crude quality proxy has real signal. It dies in stage 2, once the
    rolling xG and goals priors are there, because they carry quality far better.
    It was never carrying fatigue at all.

    ``matches_14d`` asks the question the other one was being credited for. Within
    team-season, with the outcome demeaned:

        matches in prior 14d:      0       1       2       3       4
        shots_for  (demeaned)  +0.089  +0.067  +0.007  -0.098  -0.086
        goals_for  (demeaned)  -0.004  +0.006  +0.007  -0.008  -0.040

    Monotone and correctly signed for shots, which raw rest days never was -- and
    about 1.5% of the mean, which is small enough that the selection search may
    well still prune it at a tolerance of 0.001. That would be a real answer
    rather than an artefact, which is the point of adding it.

    ``rest_days`` stays a candidate. Dropping a feature is the search's job, and
    it now has something to lose to.
    """
    frames = []
    for p in (HISTORIC_ALL_COMP, all_comp_fixtures(current_season_label())):
        if p.exists():
            frames.append(pd.read_csv(p, parse_dates=["date"]))
        else:
            print(f"  {p.name} not found -- schedule context falls back to league-only spacing")

    out = df.copy()
    if frames:
        allcomp = pd.concat(frames, ignore_index=True)
        allcomp["team"] = allcomp["team"].map(canonical_team)
        allcomp = allcomp[["date", "team"]].drop_duplicates()
    else:
        allcomp = pd.DataFrame(columns=["date", "team"])

    league_dates = out[["date", "team"]].drop_duplicates()
    timeline = pd.concat([allcomp, league_dates], ignore_index=True).drop_duplicates()
    # Concatenating an empty frame yields an object-dtype column, and both `.dt`
    # and `searchsorted` need real datetimes -- so the one case with no
    # all-competition files on disk, which is also the fallback path, would be
    # the one that raised.
    timeline["date"] = pd.to_datetime(timeline["date"])
    timeline = timeline.sort_values(["team", "date"]).reset_index(drop=True)
    timeline["rest_days"] = timeline.groupby("team")["date"].diff().dt.days

    # Matches in the `CONGESTION_DAYS` days **strictly before** this one. Each
    # team's dates are already sorted, so the count is the gap between two
    # searchsorted positions -- and excluding the fixture itself is what keeps it
    # usable on an unplayed row, where the buffers are read but nothing is known.
    window = pd.Timedelta(days=CONGESTION_DAYS)
    counts = np.empty(len(timeline), dtype=float)
    for _, g in timeline.groupby("team", sort=False):
        d = g["date"].to_numpy()
        lo = np.searchsorted(d, d - window, side="left")
        hi = np.searchsorted(d, d, side="left")
        counts[g.index.to_numpy()] = hi - lo
    timeline["matches_14d"] = counts

    out = out.merge(timeline, on=["team", "date"], how="left")
    out["rest_days"] = out["rest_days"].clip(upper=30)  # long gaps are all "rested"

    # Split into team_/opp_ pairs here rather than in the caller, so that the one
    # code path serves both the history build and `predict.append_fixtures_as_rows`.
    # It did not, and the consequence was invisible: upcoming fixtures were given
    # the columns as NaN rather than computed, so `team_rest_days` was missing on
    # every row the model was actually asked to predict. Both quantities are
    # defined for an unplayed match -- the gap since the last match and the count
    # of matches strictly before it are known the moment the fixture is scheduled
    # -- so there was never a reason for them to be absent.
    partner = out[["fixture_id", "is_home", *SCHEDULE_COLS]].copy()
    partner["is_home"] = 1 - partner["is_home"]
    partner = partner.rename(columns={c: f"opp_{c}" for c in SCHEDULE_COLS})
    out = out.merge(partner, on=["fixture_id", "is_home"], how="left")
    return out.rename(columns={c: f"team_{c}" for c in SCHEDULE_COLS})


def attach_movement_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Promotion/relegation signals, in both the live and the future-proof form.

    ``is_new_to_league``  -- team is in this league this season and was not in it
    *last* season. This is the flag that carries signal today and mirrors what
    the existing per-league notebooks used.

    ``rank_delta`` -- movement in the divisional pyramid, from LEAGUE_RANK. With
    all five leagues at the same rank this is identically zero right now; it
    starts firing the day English lower tiers are added, with no other change.

    The bug this replaces
    ---------------------
    "Last season" used to mean **the previous row this team had**, via a
    ``groupby("team").shift(1)``. But a relegated club is simply absent from the
    data for the seasons it spends down, so a club promoted back to the same
    league compared itself against its own last season *in that league*, saw the
    same league, and read "not new". A club appearing for the first time was then
    explicitly zeroed. Between them those two rules made the flag **identically
    zero across all 43,178 rows** -- 152 promoted team-seasons, none of them
    found. Combined with ``rank_delta`` being constant-zero by design, the entire
    ``movement`` feature group was two constant columns, which is why it won
    nothing at stage 1 in any of the four target families and why the README's
    "the flag that actually carries signal" carried none.

    The fix is to compare against the season that actually precedes this one,
    which needs a real ordering rather than a row offset --
    ``config.season_sort_key``, because the table mixes ``'2014/2015'``,
    ``'1920'`` and ``'2526'`` labels.

    A league's own first season in the data still reads 0: nobody can be new to a
    league we have no previous season of, and guessing would flag an entire
    division at once.
    """
    out = df.copy()
    ts = out[["team", "season", "league", "league_key"]].drop_duplicates()
    ts["_year"] = ts["season"].map(season_sort_key)
    ts = ts.sort_values(["team", "_year"])

    # Present in (league, season) -- the set membership the flag is really about.
    present = set(zip(ts["league_key"], ts["_year"], ts["team"]))
    first_year = ts.groupby("league_key")["_year"].min().to_dict()

    ts["is_new_to_league"] = [
        0 if year == first_year[lk] else int((lk, year - 1, team) not in present)
        for lk, year, team in zip(ts["league_key"], ts["_year"], ts["team"])
    ]

    # `rank_delta` still reads the team's previous *appearance* anywhere, because
    # that is the question it asks: which division did this club come up from.
    # An absent season carries no division to compare against.
    ts["prev_league"] = ts.groupby("team")["league"].shift(1)
    ts["curr_rank"] = ts["league"].map(LEAGUE_RANK)
    ts["prev_rank"] = ts["prev_league"].map(LEAGUE_RANK)
    ts["rank_delta"] = (ts["curr_rank"] - ts["prev_rank"]).fillna(0).astype(int)

    lookup = ts.set_index(["team", "season"])[["is_new_to_league", "rank_delta"]]

    for prefix, col in (("team", "team"), ("opp", "opponent")):
        j = out[[col, "season"]].merge(lookup, left_on=[col, "season"], right_index=True, how="left")
        out[f"{prefix}_is_new_to_league"] = j["is_new_to_league"].fillna(0).astype(int).values
        out[f"{prefix}_rank_delta"] = j["rank_delta"].fillna(0).astype(int).values

    return out


# --- Integrity ------------------------------------------------------------


def check_no_team_name_collisions(df: pd.DataFrame) -> None:
    """No team name may appear in two leagues in the same season.

    The pooled walk-forward buffer is keyed by team name across all leagues, so a
    collision would silently merge two clubs' histories. Currently clean (168
    distinct names, zero collisions) -- this assertion is insurance for when the
    English lower tiers, with far more scope for incidental name overlap, land.
    """
    per = df.groupby(["team", "season"])["league_key"].nunique()
    bad = per[per > 1]
    if len(bad):
        raise AssertionError(
            f"{len(bad)} team-season(s) appear in more than one league: {bad.index.tolist()[:10]}"
        )


def check_integrity(df: pd.DataFrame) -> None:
    """Structural checks that must hold before anything downstream runs."""
    assert (df.groupby("fixture_id").size() == 2).all(), "a fixture is not exactly 2 rows"

    # Per-team values must reconstruct the match: my 'for' is my opponent's 'against'.
    for s in STATS:
        if s.key in ("points",):
            continue
        g = df.groupby("fixture_id")[[f"{s.key}_for", f"{s.key}_against"]].sum()
        both = g[f"{s.key}_for"].notna() & g[f"{s.key}_against"].notna()
        assert np.allclose(
            g.loc[both, f"{s.key}_for"], g.loc[both, f"{s.key}_against"], equal_nan=False
        ), f"{s.key}: sum(for) != sum(against) across the two rows of a fixture"

    for s in STATS:
        for col in (f"{s.key}_for", f"{s.key}_against"):
            v = df[col].dropna()
            assert (v >= 0).all(), f"negative values in {col}"

    check_no_team_name_collisions(df)


def dispersion_report(df: pd.DataFrame) -> pd.DataFrame:
    """Variance/mean per stat per league -- the check behind the objective choice.

    A ratio near 1 means Poisson is adequate (goals); materially above 1 means
    overdispersion, which is what justifies ``reg:tweedie`` plus a Negative
    Binomial pmf for shots/SOT/corners.

    This is the *marginal* ratio. Some of it is explained by team strength and
    will shrink once the model conditions on priors -- see
    ``conditional_dispersion`` for the residual version.
    """
    rows = []
    for lk in sorted(df["league_key"].unique()):
        sub = df[df["league_key"] == lk]
        for s in STATS:
            if not s.is_target:
                continue
            v = sub[f"{s.key}_for"].dropna()
            if v.empty:
                continue
            m, var = float(v.mean()), float(v.var(ddof=0))
            rows.append({
                "league": lk, "stat": s.key, "n": len(v),
                "mean": round(m, 3), "var": round(var, 3),
                "var_over_mean": round(var / m, 3) if m else np.nan,
            })
    return pd.DataFrame(rows)


def conditional_dispersion(df: pd.DataFrame, by: tuple[str, ...] = ("league_key", "team", "season")) -> pd.DataFrame:
    """Residual dispersion after conditioning on team identity within a season.

    A crude but honest stand-in for "how much of the overdispersion survives once
    the model knows who is playing". If these ratios collapse to ~1, Poisson
    would do; if they stay well above 1, the Tweedie/NB choice is earned.
    """
    rows = []
    for s in STATS:
        if not s.is_target:
            continue
        col = f"{s.key}_for"
        sub = df[["league_key", "team", "season", col]].dropna()
        if sub.empty:
            continue
        grp = sub.groupby(list(by))[col]
        resid = sub[col] - grp.transform("mean")
        mean = float(sub[col].mean())
        # Within-group variance, adjusted for the degrees of freedom used by the means.
        n, k = len(sub), grp.ngroups
        var_resid = float((resid**2).sum() / max(n - k, 1))
        rows.append({
            "stat": s.key, "n": n, "groups": k,
            "mean": round(mean, 3),
            "residual_var": round(var_resid, 3),
            "residual_var_over_mean": round(var_resid / mean, 3) if mean else np.nan,
        })
    return pd.DataFrame(rows)


# --- Entry point ----------------------------------------------------------


def _data_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(str(len(df)).encode())
    h.update(str(df["date"].max()).encode())
    h.update(",".join(sorted(df.columns)).encode())
    return h.hexdigest()[:12]


def build_clean_table(force: bool = False, save: bool = True, refresh_stats: bool = False) -> pd.DataFrame:
    """Understat + ESPN -> one validated long table. Cached to parquet by data hash."""
    matches = load_matches()
    if refresh_stats or not ESPN_MATCH_STATS.exists():
        build_match_stats(save=True)

    matches = attach_espn_stats(matches)

    cov = coverage_report(matches)
    weak = cov[cov["pct"] < 95]
    if len(weak):
        print(f"  {len(weak)} league-season(s) below 95% ESPN coverage:")
        print(weak.to_string(index=False))

    long = to_team_perspective(matches)
    long = attach_game_week(long)
    long = attach_schedule_context(long)
    long = attach_movement_flags(long)

    check_integrity(long)

    if save:
        CLEAN_CACHE.mkdir(parents=True, exist_ok=True)
        dest = CLEAN_CACHE / f"team_matches_{_data_hash(long)}.parquet"
        long.to_parquet(dest, index=False)
        (CLEAN_CACHE / "latest.txt").write_text(dest.name)
        print(f"  Saved {len(long):,} rows -> {dest}")

    return long


def load_clean_table() -> pd.DataFrame:
    """Read the most recent cached clean table."""
    pointer = CLEAN_CACHE / "latest.txt"
    if not pointer.exists():
        raise FileNotFoundError("No cached clean table -- run build_clean_table() first.")
    return pd.read_parquet(CLEAN_CACHE / pointer.read_text().strip())


__all__ = [
    "attach_espn_stats", "coverage_report", "to_team_perspective",
    "attach_game_week", "attach_schedule_context", "attach_movement_flags",
    "CONGESTION_DAYS", "SCHEDULE_COLS",
    "check_no_team_name_collisions", "check_integrity",
    "dispersion_report", "conditional_dispersion",
    "build_clean_table", "load_clean_table",
]
