"""Turn predicted means into the prices the workbook shows.

Goals keep the existing treatment: two Poisson rates -> an N x N scoreline
matrix -> every goal market read off one coherent object, so all the prices stay
mutually consistent.

Shots, shots on target and corners get per-team Negative Binomial distributions
and a convolved match total. A full N x N grid is not useful there -- the count
range is far too wide to read -- so those blocks price over/under ladders
instead. Those ladders are **centred on each team's own predicted rate** rather
than fixed: a shared 8.5/10.5/12.5/14.5 shots ladder is nearly all signal for one
side and nearly all noise for the other whenever the two teams differ, which is
most fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..metrics import convolve_pmf, over_prob, pmf_for
from ..config import season_sort_key
from ..spec import STAT_BY_KEY, TARGETS, target_spec

GOALS_MATRIX_MAX = 6  # rows/cols 0..5 plus a "6+" bucket

# Per team, centred on its own rate. Eleven rather than seven so the ladder
# reaches further into both tails: the model's edge against a book is often on a
# line two or three clear of the mean, and a seven-line window simply never asked
# about those. `dynamic_team_lines` keeps the centre on `floor(mu) + 0.5` either
# way, so widening moves the ends without moving anything already priced.
#
# The ceiling on this is `total_cap`, not taste: `team_pmf` builds at `total_cap`
# and `over_prob` reads the folded tail bucket for any line at or above it. At 11
# the top line is `floor(mu) + 5.5`, which clears every cap for realistic rates --
# `tests/test_markets.py::test_every_dynamic_line_clears_the_pmf_cap` is what
# holds that true rather than this comment.
LADDER_LINES = 11


def safe_odds(p: float) -> float:
    """Fair odds. No overround -- these are model prices, not a book."""
    p = float(p)
    return float("nan") if p <= 0 else 1.0 / p


def team_pmf(target: str, mu: float, dispersion: float | None = None) -> np.ndarray:
    """One team's count distribution, built wide enough for any priced line.

    Capped at `total_cap`, not `team_cap`. The two caps are bucketing devices for
    *scoring* -- `score_team_logloss` reads one, `score_total_logloss` the other --
    and neither is a claim about pricing. `over_prob` sums the buckets above a
    line and the pmf folds all tail mass into its last cell, so any cap above the
    top line gives an identical answer; `tests/test_markets.py` pins that.

    Pricing therefore uses the wider of the two, because the ladders are now
    centred per team (`dynamic_team_lines`) and a high-rate side can reach lines
    that `team_cap` does not clear -- shots on target caps at 12 and a ladder can
    reach 10.5. A line at or above the cap would silently read the folded tail
    bucket and come back wrong rather than raising.
    """
    s = target_spec(target)
    return pmf_for(s.dist, mu, s.total_cap, dispersion)


def dynamic_team_lines(mu: float, n: int = LADDER_LINES) -> tuple[float, ...]:
    """``n`` half-integer lines centred on one team's own predicted rate.

    The centre is ``floor(mu) + 0.5`` -- the half-integer nearest ``mu``, and so
    the line whose over/under is closest to a coin flip for this team -- with
    ``n // 2`` lines either side.

    Written as a floor rather than ``round(mu - 0.5) + 0.5``, which looks
    equivalent and is not: ``round`` is banker's rounding, so ``mu`` of exactly
    12.0 and of exactly 13.0 both landed on a 12.5 centre, one half-line high and
    one half-line low. Integer rates are common enough in a fitted mean not to
    leave that to chance.

    When the window would run below zero it is **shifted up**, not clipped, so a
    low-rate side still gets ``n`` lines rather than a stub: a 2.9-SoT team is
    priced 0.5 through 6.5 rather than -0.5 through 5.5.
    """
    centre = np.floor(mu) + 0.5
    half = n // 2
    lines = [centre + i for i in range(-half, half + 1)]
    if lines[0] < 0.5:
        lines = [0.5 + i for i in range(n)]
    return tuple(float(x) for x in lines)


def ladder_lines(target: str, mk: dict) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """``(home_lines, away_lines)`` -- the single definition of which lines a
    fixture is priced at.

    Goals keep the fixed spec ladder: the range is small enough that one shared
    set is genuinely readable, and 0.5-3.5 are the lines actually quoted. The
    three count families use the per-team ladders `count_markets` centred on each
    side's own rate, which travel in ``mk``.

    This lives here rather than in the workbook writer because two files now need
    the answer -- the fixture sheets and the odds capture form -- and the two must
    agree exactly or the workbooks stop lining up row for row. One definition
    makes that true by construction rather than by both happening to encode the
    same rule.
    """
    if target == "goals":
        lines = target_spec(target).team_lines
        return lines, lines
    return tuple(mk["home_lines"]), tuple(mk["away_lines"])


def scoreline_matrix(mu_home: float, mu_away: float, max_goals: int = GOALS_MATRIX_MAX) -> np.ndarray:
    """P(home=i, away=j) under goal independence. Sums to 1."""
    from ..metrics import poisson_pmf

    ph = poisson_pmf(mu_home, max_goals)
    pa = poisson_pmf(mu_away, max_goals)
    return np.outer(ph, pa)


def matrix_labels(max_goals: int = GOALS_MATRIX_MAX) -> list[str]:
    """``["0", "1", ... "5", "6+"]`` -- the axis of a `scoreline_matrix`.

    The last row and column are a *bucket*, not a score, and the `+` is the only
    thing that says so. Two readers now draw this axis -- the fixture sheet and
    the Edge Book heatmap -- and one of them quietly labelling the bucket "6"
    would misreport every high-scoring cell.
    """
    return [str(i) for i in range(max_goals)] + [f"{max_goals}+"]


def goal_markets(mu_home: float, mu_away: float) -> dict:
    """1X2, BTTS, and over/under -- every one a reduction of the same matrix."""
    M = scoreline_matrix(mu_home, mu_away)
    s = target_spec("goals")

    # Rows are home goals, so home wins strictly below the diagonal.
    p_home = float(np.tril(M, k=-1).sum())
    p_draw = float(np.trace(M))
    p_away = float(np.triu(M, k=1).sum())
    p_btts = float(M[1:, 1:].sum())

    ph, pa = M.sum(axis=1), M.sum(axis=0)
    total = convolve_pmf(ph, pa, s.total_cap)

    out = {
        "matrix": M, "pmf_home": ph, "pmf_away": pa, "pmf_total": total,
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "p_btts_yes": p_btts, "p_btts_no": 1.0 - p_btts,
        "most_likely": np.unravel_index(int(np.argmax(M)), M.shape),
        "most_likely_p": float(M.max()),
    }
    for line in s.match_lines:
        out[f"match_over_{line}"] = over_prob(total, line)
    for line in s.team_lines:
        out[f"home_over_{line}"] = over_prob(ph, line)
        out[f"away_over_{line}"] = over_prob(pa, line)
    return out


def count_markets(target: str, mu_home: float, mu_away: float, dispersion: float | None = None,
                  n_lines: int = LADDER_LINES) -> dict:
    """Per-team ladders -- one per side, each centred on that side's own rate.

    The two ladders generally do **not** share their lines, which is the point:
    the sheet asks a useful question of each team rather than the same question
    of both. ``home_lines`` and ``away_lines`` travel with the prices so callers
    never have to re-derive them.

    Both pmfs are built at ``total_cap``, so the convolution is over untruncated
    tails -- feeding a tail-folded team pmf into it would move real mass down
    before the sides are combined and shift the match total. One wide pair now
    serves both the ladders and the convolution.
    """
    s = target_spec(target)
    ph = team_pmf(target, mu_home, dispersion)
    pa = team_pmf(target, mu_away, dispersion)
    total = convolve_pmf(ph, pa, s.total_cap)

    home_lines = dynamic_team_lines(mu_home, n_lines)
    away_lines = dynamic_team_lines(mu_away, n_lines)

    out = {"pmf_home": ph, "pmf_away": pa, "pmf_total": total,
           "mu_home": mu_home, "mu_away": mu_away, "mu_total": mu_home + mu_away,
           "home_lines": home_lines, "away_lines": away_lines}
    for line in home_lines:
        out[f"home_over_{line}"] = over_prob(ph, line)
    for line in away_lines:
        out[f"away_over_{line}"] = over_prob(pa, line)
    for line in s.match_lines:
        out[f"match_over_{line}"] = over_prob(total, line)
    return out


def fixture_markets(row: pd.Series, dispersion: dict[str, dict[str, float]] | None = None) -> dict:
    """Every market for one fixture, across all four target families."""
    out = {"goals": goal_markets(float(row["goals_home"]), float(row["goals_away"]))}
    for t in TARGETS:
        if t == "goals":
            continue
        d = (dispersion or {}).get(t, {})
        disp = d.get(row["league_key"]) if isinstance(d, dict) else d
        out[t] = count_markets(t, float(row[f"{t}_home"]), float(row[f"{t}_away"]), disp)
    return out


# --- League baselines -----------------------------------------------------


# Season ordering lives in `config.season_sort_key` -- `clean` and `predict` need
# the same answer, and three copies of "which season came first" is how one of
# them ends up disagreeing.
_season_sort_key = season_sort_key


def _survival(values: np.ndarray, cap: int) -> np.ndarray:
    """``out[k] = P(X >= k)`` for k in 0..cap+1, from observed counts.

    One array answers ``P(X > line)`` at *any* half-integer line, which is what
    the per-team ladders now need: their lines move with each team's rate, so
    there is no fixed set of thresholds to precompute.
    """
    v = values[np.isfinite(values)]
    out = np.zeros(cap + 2)
    if v.size == 0:
        return np.full(cap + 2, np.nan)
    for k in range(cap + 2):
        out[k] = float((v >= k).mean())
    return out


@dataclass(frozen=True)
class LeagueBaseline:
    """One league's realised rates -- the baseline the fixture colours mean.

    Team rates are held **separately for home and away**. Pooling them, as the
    first version did, made the colour report venue rather than edge: home sides
    take materially more shots, so every home ladder came out green and every
    away ladder red regardless of the fixture. Prem P(shots > 12.5) is .551 at
    home against .379 away -- a gap wider than most of the delta scale.

    Match totals stay pooled, having no venue.
    """

    league_key: str
    seasons: tuple[str, ...]
    n_matches: int
    p_home: float
    p_draw: float
    p_away: float
    p_btts_yes: float
    p_btts_no: float
    # surv[(target, scope)][k] = P(count >= k); scope in {"home", "away", "match"}
    surv: dict[tuple[str, str], np.ndarray]

    def over_rate(self, target: str, scope: str, line: float) -> float:
        """P(count > line) for a half-integer line, i.e. P(count >= floor+1)."""
        a = self.surv.get((target, scope))
        if a is None:
            return float("nan")
        k = int(np.floor(line)) + 1
        return float(a[min(k, len(a) - 1)])

    def result_rate(self, key: str) -> float:
        return float(getattr(self, key, np.nan))


def league_baselines(team_matches: pd.DataFrame, seasons: int = 2) -> dict[str, LeagueBaseline]:
    """Realised rates per league -- the comparison baseline for colouring.

    Computed **per league**, deliberately, even though the model is pooled: a
    shots total that is "above average" has to mean above *that* league's own
    average, not the blend across five very different environments.

    Uses the most recent ``seasons`` seasons so the baseline reflects the current
    game rather than a decade-old one.
    """
    df = team_matches
    keep = sorted(df["season"].unique(), key=_season_sort_key)[-seasons:]
    df = df[df["season"].isin(keep)]

    out: dict[str, LeagueBaseline] = {}
    for lk, g in df.groupby("league_key", observed=True):
        home = g[g["is_home"] == 1].set_index("fixture_id")
        away = g[g["is_home"] == 0].set_index("fixture_id")
        common = home.index.intersection(away.index)

        hg = home.loc[common, "goals_for"].to_numpy()
        ag = away.loc[common, "goals_for"].to_numpy()
        p_btts = float(((hg >= 1) & (ag >= 1)).mean()) if common.size else np.nan

        surv: dict[tuple[str, str], np.ndarray] = {}
        for t in TARGETS:
            s = STAT_BY_KEY[t]
            h = home.loc[common, f"{t}_for"].to_numpy(dtype=float)
            a = away.loc[common, f"{t}_for"].to_numpy(dtype=float)
            surv[(t, "home")] = _survival(h, s.total_cap)
            surv[(t, "away")] = _survival(a, s.total_cap)
            surv[(t, "match")] = _survival(h + a, s.total_cap)

        out[lk] = LeagueBaseline(
            league_key=str(lk),
            seasons=tuple(str(x) for x in keep),
            n_matches=int(common.size),
            p_home=float((hg > ag).mean()) if common.size else np.nan,
            p_draw=float((hg == ag).mean()) if common.size else np.nan,
            p_away=float((hg < ag).mean()) if common.size else np.nan,
            p_btts_yes=p_btts,
            p_btts_no=1.0 - p_btts,
            surv=surv,
        )
    return out


__all__ = [
    "safe_odds", "team_pmf", "dynamic_team_lines", "ladder_lines", "scoreline_matrix",
    "matrix_labels", "goal_markets",
    "count_markets", "fixture_markets", "league_baselines", "LeagueBaseline",
    "GOALS_MATRIX_MAX", "LADDER_LINES",
]
