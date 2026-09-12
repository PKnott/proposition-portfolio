"""Registries and constants. Everything league- or season-specific lives here.

The five old league notebooks were ~99% identical; the only real differences were
five configuration strings each. Those differences are now rows in ``LEAGUES``.
"""

from __future__ import annotations

from dataclasses import dataclass

RANDOM_SEED = 2026

# --- Leagues --------------------------------------------------------------


@dataclass(frozen=True)
class League:
    key: str  # our short key, also the Inputs/ folder name
    understat: str  # soccerdata Understat league code
    espn_slug: str  # ESPN league slug
    prefix: str  # legacy file prefix, also the Excel sheet code
    display: str
    rank: int  # LEAGUE_RANK -- see note below
    fixtures_stem: str  # Inputs/Fixtures/<stem>_fixtures.csv


LEAGUES: dict[str, League] = {
    "Prem": League("Prem", "ENG-Premier League", "eng.1", "E0", "Premier League", 1, "premier_league"),
    "Liga": League("Liga", "ESP-La Liga", "esp.1", "SP1", "La Liga", 1, "la_liga"),
    "Bund": League("Bund", "GER-Bundesliga", "ger.1", "D1", "Bundesliga", 1, "bundesliga"),
    "Serie": League("Serie", "ITA-Serie A", "ita.1", "I1", "Serie A", 1, "serie_a"),
    "Ligue": League("Ligue", "FRA-Ligue 1", "fra.1", "F1", "Ligue 1", 1, "ligue_1"),
}

# The division each top-tier league promotes *from*. Separate from `LEAGUES`, not
# a row in it, because the two are not the same kind of thing: every consumer of
# `LEAGUES` treats it as "leagues we pull Understat for" -- `predict.upcoming`,
# `ingest.pull_understat` and `reconcile` all iterate it that way, and
# `UNDERSTAT_TO_KEY` inverts it on a code these do not have. Understat covers the
# top five divisions and nothing below, so a second tier put in `LEAGUES` would
# be asked for data that does not exist and would collide on a blank code.
#
# ESPN does cover them, which is the whole point: shots, shots on target and
# corners for a promoted club's final season down, to scale the blanket promoted
# seed by what that club actually did. See `fpp.promoted`.
@dataclass(frozen=True)
class LowerLeague:
    key: str          # our key, and the `league_key` its stat rows carry
    parent: str       # the top-tier key a promoted club enters
    espn_slug: str
    display: str


LOWER_LEAGUES: dict[str, LowerLeague] = {
    "Prem2":  LowerLeague("Prem2",  "Prem",  "eng.2", "Championship"),
    "Liga2":  LowerLeague("Liga2",  "Liga",  "esp.2", "LaLiga 2"),
    "Serie2": LowerLeague("Serie2", "Serie", "ita.2", "Serie B"),
    "Bund2":  LowerLeague("Bund2",  "Bund",  "ger.2", "2. Bundesliga"),
    "Ligue2": LowerLeague("Ligue2", "Ligue", "fra.2", "Ligue 2"),
}
PARENT_OF = {k: lg.parent for k, lg in LOWER_LEAGUES.items()}
LOWER_OF = {lg.parent: k for k, lg in LOWER_LEAGUES.items()}

# Both tiers, because `scoreboard.parse_cache` filters cached files on this and a
# second-tier file must be parsed, not skipped.
SLUG_TO_KEY = ({lg.espn_slug: lg.key for lg in LEAGUES.values()}
               | {lg.espn_slug: lg.key for lg in LOWER_LEAGUES.values()})
UNDERSTAT_TO_KEY = {lg.understat: lg.key for lg in LEAGUES.values()}

# Promotion/relegation is evaluated against a team's own previous season.
#
# All five leagues share rank 1 because none sits below another in a shared
# pyramid -- so `rank_delta` is identically zero today and the flag never fires.
# That is expected, and it is *why* `is_new_to_league` (below, in the feature
# spec) exists as the feature that actually carries signal right now.
#
# When English lower tiers are added, insert them beneath the Premier League
# with decreasing rank (Championship 0, League One -1, League Two -2) and
# `rank_delta` starts firing with no other change.
LEAGUE_RANK: dict[str, int] = ({lg.display: lg.rank for lg in LEAGUES.values()}
                               | {lg.display: 0 for lg in LOWER_LEAGUES.values()})

# --- Club identity --------------------------------------------------------

# Understat uses two different names for the same club across eras, which would
# otherwise split the club's rolling-prior history in two and break the join to
# ESPN (which uses one name throughout).
#
# Parma FC went bankrupt in 2015 and was refounded as Parma Calcio 1913; both
# eras are the same club for our purposes, and ESPN calls them "Parma"
# throughout. Applied to every team name on the way into the clean table.
#
# NB: 'Ajaccio' and 'GFC Ajaccio' look similar but are genuinely different clubs
# (AC Ajaccio and Gazélec FC Ajaccio) -- do not merge them.
TEAM_ALIASES: dict[str, str] = {
    "Parma Calcio 1913": "Parma",
    # The name this club was carried under while it sat outside Understat's
    # history -- see `ingest.espn.MAPPING_GAP_FILL`. Understat lists it as
    # "Elversberg" now, and folding the old spelling here is what lets rows
    # already written under it -- 81 second-tier stat rows, and any ledger
    # proposition captured before today -- still resolve to one club.
    "SV Elversberg": "Elversberg",
}


def canonical_team(name: str) -> str:
    return TEAM_ALIASES.get(name, name)


# --- Seasons & splits -----------------------------------------------------
#
# Every season boundary derives from ONE value: the current season. These used to
# be four hand-maintained constants that had to move together at each rollover,
# and they drifted -- `FIRST_VAL_SEASON` sat six seasons behind the test boundary
# instead of five, because it was not rolled forward last time.
#
# They are functions, not constants, because `from .config import CURRENT_SEASON`
# binds at import time: a caller that did that would never see `set_seasons`.
# Read them through the accessors and an override is always visible.

_CURRENT_SEASON = "2026/2027"
_PREVIOUS_SEASON = "2025/2026"  # last completed season; upper bound of the history
_FIRST_HISTORIC_YEAR = 2014
_VAL_LOOKBACK = 5  # seasons between the test-season start and the first val season


def _season_start(season: str) -> int:
    """``"2026/2027"`` -> ``2026``. Raises on anything that is not a season."""
    try:
        start, end = str(season).split("/")
        start_year, end_year = int(start), int(end)
    except (ValueError, AttributeError) as e:
        raise ValueError(f"season must look like '2026/2027', got {season!r}") from e
    if end_year != start_year + 1:
        raise ValueError(f"season must span consecutive years, got {season!r}")
    return start_year


def _season(start_year: int) -> str:
    return f"{start_year}/{start_year + 1}"


def season_sort_key(season: str) -> int:
    """Start year of a season label, whatever shape the label happens to be.

    The cleaned table carries three formats side by side -- ``'2014/2015'``,
    ``'1920'`` and ``'2526'`` -- because they arrive from different sources.
    Plain lexical ``sorted()`` puts ``'1920'`` first and ``'2526'`` last, which
    happens to give the right answer today and would stop doing so the moment a
    ``'0910'``-style label appears. Sort on the year the season *starts* instead,
    so the ordering is right by construction.

    Lives here rather than beside any one caller: `clean` decides which season
    precedes which, `predict` decides how stale a club's history is, and
    `report.markets` picks the most recent seasons for its baselines. Three
    copies of "which season came first" is how one of them ends up disagreeing.
    """
    s = str(season)
    if "/" in s:
        return int(s.split("/")[0])
    if len(s) == 4:  # 'YYnn' -> 20YY, e.g. '2526' -> 2025
        return 2000 + int(s[:2])
    return int(s)


def season_label(season: str) -> str:
    """``"2026/2027"`` -> ``"2026-27"``, the ESPN/file-naming form."""
    start = _season_start(season)
    return f"{start}-{str(start + 1)[2:]}"


def current_season() -> str:
    return _CURRENT_SEASON


def current_season_label() -> str:
    return season_label(_CURRENT_SEASON)


def previous_season() -> str:
    """The last completed season -- the upper bound of ``historic_seasons()``."""
    return _PREVIOUS_SEASON


def historic_seasons() -> list[str]:
    """2014/2015 through the previous season, inclusive."""
    return [_season(y) for y in range(_FIRST_HISTORIC_YEAR, _season_start(_PREVIOUS_SEASON) + 1)]


def test_season() -> str:
    """Sacred boundary. Seasons >= this are the held-out test set, scored once.

    Always the same value as ``previous_season()``, and necessarily so: the
    contiguity rule in ``set_seasons`` means "the last completed season" and "the
    season immediately before the current one" cannot differ. Two names, one
    value, deliberately -- ingest and modelling each read it under the name that
    fits, but there is only one thing to keep correct.
    """
    return previous_season()


def first_val_season() -> str:
    """First validation fold of the expanding-window walk-forward CV.

    Five seasons before the test boundary, so the validation window slides
    forward with it rather than silently widening at every rollover.
    """
    return _season(_season_start(test_season()) - _VAL_LOOKBACK)


def holdout_season() -> str:
    """The most recent development season, withheld from tuning.

    `03_Tuning` searches the folds strictly before this one and then scores the
    settled configuration against it exactly once. The gap between the two is the
    overfitting read -- and it is the only one available without spending the test
    season, which by design is looked at once ever.

    Derived from the test boundary rather than typed, so it rolls forward with
    everything else: the season immediately before the first test season.
    """
    return _season(_season_start(test_season()) - 1)


def set_seasons(current: str | None = None, previous: str | None = None) -> None:
    """Override the season boundaries for this process.

    Validates rather than trusts: each season must span consecutive years, and
    ``previous`` must start exactly one year before ``current``. Both of the
    season-boundary bugs this replaced would have been caught here.
    """
    global _CURRENT_SEASON, _PREVIOUS_SEASON
    new_current = _CURRENT_SEASON if current is None else current
    new_previous = _PREVIOUS_SEASON if previous is None else previous

    current_start = _season_start(new_current)
    previous_start = _season_start(new_previous)
    if previous_start != current_start - 1:
        raise ValueError(
            f"previous season {new_previous!r} must start one year before current "
            f"{new_current!r} -- got a gap of {current_start - previous_start} year(s)"
        )
    if current_start <= _FIRST_HISTORIC_YEAR:
        raise ValueError(
            f"current season {new_current!r} must start after the first historic "
            f"year, {_FIRST_HISTORIC_YEAR}"
        )

    _CURRENT_SEASON, _PREVIOUS_SEASON = new_current, new_previous

# --- Cold start for promoted sides ----------------------------------------
#
# See the block comment above `season_seeds` in priors.py for the measurements
# these are set from. All three are dials rather than constants because the right
# values are an empirical question this project has only just started asking.

# Seasons of promoted-side history to average over. Five is a compromise: long
# enough for the smaller leagues to reach a usable sample, short enough that the
# profile still reflects the current game.
PROMOTED_LOOKBACK = 5

# Matches of each promoted side's first season to use. `None` = the whole season.
# The promoted deficit barely fades within one -- goals-for runs 0.70 of the
# league average over games 1-3 and 0.78 over games 20-38 -- so the whole season
# gives three to four times the sample at essentially no cost in accuracy at game
# one. Set an int to test the alternative.
PROMOTED_WINDOW: int | None = None

# Promoted team-seasons a league-season needs behind it before its promoted
# profile is used at all; below this it falls back to the league average. The
# Bundesliga binds: 23 promoted team-seasons across twelve years, so a
# five-season lookback yields about nine.
MIN_PROMOTED_SEASONS = 6

# How hard a cold-start seed pulls on a buffer that has *some* usable history.
#
# Expressed as a fraction of the buffer's saturated denominator, never as an
# absolute pseudo-count: `den` is a weighted sum, not a match count, and it
# saturates at very different levels per target (7.16 for shots at alpha=0.15
# against 16.94 for goals at alpha=0.05). A fixed count would weigh the seed 41%
# on shots and 23% on goals for the same side -- see `priors.full_denominator`.
#
# With `k = kappa * den_full`, seed share by real matches played at kappa=0.05:
#
#     target      0m      3m      5m     10m     19m     38m
#     goals    100.0%   22.9%   15.7%    9.5%    6.3%    4.8%
#     shots    100.0%   12.1%    8.6%    6.0%    5.0%    4.8%
#     sot      100.0%   16.2%   11.3%    7.3%    5.6%    4.9%
#     corners  100.0%   12.1%    8.6%    6.0%    5.0%    4.8%
#
# Swept on walk-forward CV against frozen (L, alpha, features, params). Change
# from kappa=0, in fold SE -- negative is better:
#
#     kappa      0.01    0.02    0.05    0.10    0.20    0.35    0.50
#     goals     +0.11   +0.06   -0.10   +0.00   -0.00   +0.13
#     sot       -0.05   -0.07   -0.12   -0.19   -0.14   +0.01   +0.27
#     corners   -0.01   -0.02   -0.03   -0.04   -0.01   -0.02
#     shots     -0.02   -0.02   -1.13   -2.57   -2.60   -3.13   -3.48
#
# Three of the four families are flat inside noise and this is set for robustness,
# not for the CV: a buffer whose window holds one heavily decayed observation --
# Troyes' SOT in August 2026 carried den=0.003, 0.03% of a full window -- is not
# zero, so the warm-up gate does not fire, and it would otherwise be read at full
# confidence. Any positive kappa closes that continuously: at 0.02 such a row is
# 98.6% seed, while a fully established side moves 2%. That is preferred to a
# hand-set `min_effective` floor, which would be another threshold to choose.
#
# Shots is the exception and is deliberately *not* followed here. Its optimum is
# real (-3.48 SE) but interior and far out at 0.5, and 62% of shots rows sit at a
# full window, so the gain is being earned on established sides -- it is general
# regularisation of a noisy prior, not the promoted-team correction this constant
# exists for. Worth pursuing separately, re-tuning L/alpha/params jointly rather
# than bolting 0.5 onto settings frozen at kappa=0.
PRIOR_SEED_KAPPA = 0.02

# Uniform shrinkage of a team's prior toward its league average, per target and
# **independent of `den`**. A different lever from `PRIOR_SEED_KAPPA`: kappa asks
# "how much do I trust this buffer given how sparse it is", this asks "even at
# full sample, how far toward the population mean should any estimate be pulled".
# Folding the two into one number would conflate two mechanisms under one dial.
#
# All zero: this is an investigation, not a shipped setting. `build_feature_table`
# takes a `target_shrink` override so the sweep needs no edit here, and a non-zero
# value must not be committed without the calibration evidence to go with it --
# a shrunk estimate can improve log loss while degrading the tails, which is the
# failure mode log loss alone cannot see. See `fpp.calibration`.
TARGET_SHRINK: dict[str, float] = {"goals": 0.0, "shots": 0.0, "sot": 0.0, "corners": 0.0}

# --- Growth-optimal staking (`fpp.growth`) --------------------------------
#
# How much of a bankroll to commit to one portfolio, repeated round after round.
# A different question from either existing one: `staking` splits stake *within*
# a portfolio and `portfolio` chooses which propositions go together; neither
# asks what fraction of the pot the whole portfolio should get.
#
# The `f` grid stops strictly below 1 because f* cannot reach it. Every exported
# portfolio has `p0 = P(return == 0) > 0` -- the smallest seen is 2.7e-09 -- so
# `g(f) -> -inf` as `f -> 1`, and `g` is concave, which puts the optimum strictly
# inside. Measured on 2026-08-22: 0 of 500 sampled portfolios had f* > 1, the
# largest was 0.98. A grid reaching past 1 would only ever return -inf there.
GROWTH_F_MAX = 0.99
GROWTH_F_COARSE = 0.05
GROWTH_F_FINE = 0.01

# Drawdown constraint `f_protective` must satisfy: P(ever down >= D from a peak)
# below `GROWTH_DRAWDOWN_P`. A risk-tolerance choice, not an engineering one, and
# a live dial rather than a hardcode so it can be revisited without a redeploy.
#
# 30% rather than 50%: median f* on real data is 0.96 of bankroll, so the
# unconstrained optimum is not usable and the constraint is what makes the number
# actionable. Full-Kelly is growth-optimal in the limit and brutal on the way.
GROWTH_DRAWDOWN_D = 0.30
GROWTH_DRAWDOWN_P = 0.05

# Multi-round paths behind the drawdown estimate. The one sampled quantity in the
# module -- "ever down D% over many rounds" is path-dependent with no closed form
# -- but each round is drawn from the exact grid pmf, not from a re-approximation
# of the legs, and it is confined to a shortlist of candidate `f` values.
GROWTH_PATHS = 4_000
GROWTH_ROUNDS = 100

# The tolerances the Edge Book offers as a selector, beside the one above that
# sets the published stake.
#
# `GROWTH_DRAWDOWN_D`/`_P` are a taste, not a derivation: 30% at 5% is one point
# in this space, and it is the point that makes the published stake one-ninth
# Kelly. Shipping the neighbours is how that choice becomes visible rather than
# buried -- a reader who can see that 50%-at-25% would double the stake can tell
# a recommendation from a house rule.
#
# Nine cells cost about a third more than the one they surround, not nine times:
# a path's worst fall is a single number, so every drawdown level is a tail count
# of one array and every probability a quantile of it. See `growth.max_drawdowns`.
GROWTH_DRAWDOWN_GRID_D: tuple[float, ...] = (0.10, 0.30, 0.50)
GROWTH_DRAWDOWN_GRID_P: tuple[float, ...] = (0.05, 0.10, 0.25)

# Points in the shared `f` sweep the grid reads its crossings off. Coarse on
# purpose -- each crossing is then bisected inside the interval the sweep
# bracketed it in, which costs six more passes and lands far finer than 24
# points alone would.
GROWTH_DRAWDOWN_GRID_POINTS = 24

# --- Projection block (`growth.growth_metrics(projection=True)`) ----------
#
# What the Edge Book's Projection page draws. Percentiles rather than a literal
# best and worst path: the extremes of 4,000 samples are the two draws that got
# luckiest and unluckiest, they move with `rng_seed`, and quoting them as "the
# maximum" would give a sampling artefact the authority of a bound.
GROWTH_BAND_QUANTILES: tuple[int, ...] = (5, 50, 95)

# Rounds the bands are exported at, log-spaced over 1..GROWTH_ROUNDS. The fan is
# smooth in `n` -- log wealth is a sum of iid rounds, so its quantiles grow like
# `n*g + z*sqrt(n*v)` -- and a client drawing straight lines between these points
# is indistinguishable from one handed all 100. This is a *transport* decision,
# not an accuracy one: every exported point is the exact percentile of the full
# sample at that round. It exists because the block ships once per portfolio and
# the undominated set runs to thousands.
GROWTH_BAND_POINTS = 25

# Bins for the single-round outcome histogram. The pmf behind it has GRID=16384
# cells, which is resolution for reading thresholds off, not for drawing.
GROWTH_HIST_BINS = 40

# --- Model risk (`growth.suggested_fraction`) ------------------------------
#
# The drawdown constraint above prices *variance* -- the week going badly. These
# two price the model being wrong, which is a different risk and the one leg
# count cannot diversify away. Both are slate-wide: they are the mean and the
# spread of one quantity, the error shared by every leg on the card.
#
# Measured (R006-R007, 286 settled propositions): independent per-leg error costs
# nothing at all -- the extra uncertainty in `p` is exactly offset by less
# coin-flip variance in the outcome, and simulated books at 6 to 96 legs are
# indistinguishable from a perfect model. A *shared* error is not diversified:
# a 2pp common bias cuts a 10% edge to 6.3% at every leg count, and a 5pp common
# wobble widens the spread by 2.7% at 6 legs and 41% at 96.
#
# Both default to zero because there is no evidence for either. The two settled
# slates ran *hot*, not optimistic (63 hits against 58.2 expected, 51 against
# 46.5), so a positive `PESSIMISM_B` today would be caution wearing the clothes
# of a measurement. They become live numbers when the calibration study fills
# them in; until then the suggested stake is exactly the drawdown constraint.
PESSIMISM_B = 0.0
SLATE_TAU = 0.0

# There is deliberately no flat ceiling on the suggested stake.
#
# One was tried and removed. It priced nothing the rest of the chain did not
# already price: the drawdown constraint bounds the *path* from the portfolio's
# own distribution, `PESSIMISM_B` and `SLATE_TAU` bound the model being wrong,
# and a ceiling on the stake is a ceiling on the worst round by definition --
# lose every leg and you are out exactly what you staked -- so it restated the
# number it was capping and added no information.
#
# Nothing structural needs it either: `growth.protective_fraction` searches over
# `f <= f_star <= GROWTH_F_MAX`, so the stake is bounded below 1 regardless, and
# the drawdown constraint bites long before that (P(30% drawdown) reaches 1.00 by
# f = 0.9 on a real book). If a large card returns a stake that feels too big, the
# honest lever is `SLATE_TAU` -- it shrinks in proportion to the risk it guards
# against, which grows with leg count, where a flat cap does not.

# Largest share of one round's stake any single leg may carry.
#
# Growth weights concentrate by design: on R006's full book the best leg takes
# 24% of stake and effective legs fall from 12.6 to 5.6 against minimum variance.
# That is correct for compounding and uncomfortable for model risk, because it
# means one mispriced proposition moves the week. Capping costs a few percent of
# capacity and is reported as `capacity_used` so the cost is visible.
MAX_LEG_STAKE = 0.15

# How much room the cap leaves a short portfolio, as a multiple of equal weight.
#
# `MAX_LEG_STAKE` is unreachable below seven legs -- six legs cannot each hold
# under 15% -- so `cap_stakes` relaxes it per row. Relaxing to exactly `1 / n`
# is feasible and *uniquely* feasible: with six legs capped at a sixth, the only
# allocation summing to one is six equal stakes, and the growth split is erased
# rather than constrained. Measured on the 12 Sept slate, a three-fixture card
# whose largest portfolio is six legs, that flattened all 194 exported
# portfolios to equal stakes and cost 13% of capacity on the top row.
#
# At 2.0 the floor is twice equal weight, so a short row keeps the ordering the
# growth weights give it while still refusing to put a third of the stake on one
# proposition. Below `n = CAP_RELIEF` the cap cannot bind at all and the row is
# effectively uncapped, which is the honest answer for two legs: there is no
# diversification left to protect.
CAP_RELIEF = 2.0

# --- Paired propositions (`portfolio.event_options`) ----------------------
#
# Whether a portfolio may take two propositions from one match, and how many
# candidate pairs each event offers the search.
#
# The one-per-event rule was adopted on the assumption that same-match
# propositions are too correlated to count separately. Measured over 1,438
# same-match pairs across R006-R007, they are correlated but far from duplicated,
# and propositions on *opposite* teams are negatively correlated -- worth more
# than an unrelated leg, not less. A placebo pairing propositions from different
# matches returns -0.013, so the estimator is clean.
#
# `PAIR_KEEP` bounds how many pairs survive per event after the (capacity,
# normaliser) frontier. Eight reaches the full available capacity on both settled
# slates (1.32x on R006, 1.29x on R007) while keeping a five-event card inside
# `portfolio.EXHAUSTIVE_MAX`, so small midweek slates still enumerate provably.
PAIRS_ENABLED = True
PAIR_KEEP = 8

# Ceiling on how many undominated portfolios are exported to the Edge Book.
#
# Inert on a normal slate -- 2,360 cleared the frontier on 11 Sept -- and there
# for the shape that is not normal. A 37-event card at `leg_var = 2` put 12% of
# its pool on the frontier, because portfolios differing by one dropped event
# have return and variance correlated at 0.83 and so almost never dominate each
# other. That run wrote 35 MB and no page could be built from it; 9.4 MB is the
# largest that ever worked.
EXPORT_MAX = 2_500

# Correlation of two propositions in the same match, by what they share.
#
# Provenance: 1,438 same-match pairs, R006-R007, as the mean of z_i*z_j with
# z = (won - p)/sqrt(p(1-p)). Standard errors run 0.04-0.09 and every band
# excludes zero. Re-estimated by `analysis.pair_correlation` as slates
# accumulate -- five numbers from two weeks is where this starts, not where it
# should stay.
#
# Same team and same stat is graded by how far apart the two lines are, because
# the headline 0.455 averages over a real gradient: adjacent lines are nearly the
# same bet, distant ones much less so.
PAIR_CORRELATION: dict[str, float] = {
    "same_team_same_stat_adjacent": 0.630,   # lines <= 1 apart
    "same_team_same_stat_near": 0.400,       # lines ~2 apart
    "same_team_same_stat_far": 0.300,        # lines 3+ apart
    "same_team_diff_stat": 0.267,
    "opposite_teams": -0.185,
}

# --- Search grids ---------------------------------------------------------
# Mirrors the reference framework's budget. The brief is explicit that we are
# not trying to be more exhaustive than the framework being copied.

# The window grid. Extended past 45 because the optimum is a broad *plateau* that
# reaches a little further than the grid did: measured on goals at alpha=0.05,
# L=45 scores 1.455809 and L=80 scores 1.455285. That is a fifth of a fold SE --
# small, and the reason `_plateau_entry` picks the near edge of the plateau rather
# than its lowest point. What matters is being inside the zone at all: the spread
# across the whole grid is 10-12 fold SE, its interior under a third of one.
L_GRID = [2,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,
          50,55,60,70,85,100]
ALPHA_GRID = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5]

# `n_estimators` is derived via early stopping rather than tuned, so block 1 is a
# learning-rate sweep instead of a 7x5 grid.
PAIR_BLOCKS: list[tuple[str, list[tuple[str, list]]]] = [
    ("learning_rate", [("learning_rate", [0.01, 0.02, 0.05, 0.1, 0.2])]),
    ("max_depth+min_child_weight", [("max_depth", [2, 3, 4, 5,6]), ("min_child_weight", [1,2,4,6,8, 10, 20, 30])]),
    ("subsample+colsample_bytree", [("subsample", [0.5, 0.6, 0.7, 0.8,0.85, 0.9,0.95, 1.0]), ("colsample_bytree", [0.5, 0.6, 0.7, 0.8, 0.85,0.9,0.95, 1.0])]),
    ("reg_alpha+reg_lambda", [("reg_alpha", [0, 1, 5, 10, 20]), ("reg_lambda", [0.1, 0.5, 1.0, 5.0])]),
]

# Extra block, Tweedie families only.
TWEEDIE_BLOCK = ("tweedie_variance_power", [("tweedie_variance_power", [1.1, 1.3, 1.5, 1.7])])

# `max_bin` as a tuned axis rather than something inherited from the FitConfig.
#
# It used to arrive via `base_params(target, fit)`, which copies `fit.max_bin` into
# the params dict; the descent carried it in `current` and it was frozen. Every
# production model therefore ran at 64 -- the *probe's* cheap binning -- not
# because anything chose it but because that was the search scaffolding. Measured,
# 64 is in fact fine here (256 scored marginally worse on goals and cost ~10%
# more), which is the point: it should be a result, not a leftover.
MAX_BIN_BLOCK = ("max_bin", [("max_bin", [64, 128, 256])])

# Rounds of the alternating window <-> hyperparameter loop, and descent passes
# inside each round. Both are caps rather than targets: the loop stops as soon as
# a round changes nothing, which is the real stopping rule.
MAX_TUNING_ROUNDS = 3
DESCENT_PASSES = 2

# How far either side of the incumbent a later round re-searches the window.
# Round 1 searches the full grid; after that the question is only whether the
# tuned learner moved the answer, and a full re-sweep costs 261 points to
# re-confirm a plateau it has already found.
WINDOW_RECHECK_L_SPAN = 12
WINDOW_RECHECK_ALPHA_SPAN = 0.1

# --- Fit configurations ---------------------------------------------------


@dataclass(frozen=True)
class FitConfig:
    """Search fits are cheap probes; final fits are full strength.

    Using a faster learning rate during search is legitimate because ``(L, alpha)``
    and feature subsets are questions about data *representation*, which
    rank-orders near-identically under a faster learner. The guard is that the
    top candidates are re-verified at ``FULL`` before anything is frozen.
    """

    learning_rate: float
    n_estimators: int
    early_stopping_rounds: int | None
    max_bin: int
    nthread: int = 1


PROBE = FitConfig(learning_rate=0.05, n_estimators=400, early_stopping_rounds=30, max_bin=64)
FULL = FitConfig(learning_rate=0.014, n_estimators=2400, early_stopping_rounds=50, max_bin=256)

# What `03_Tuning` searches at. Distinct from PROBE, which `02_Features` uses and
# which must not move under it.
#
# `n_estimators` is a **ceiling, not a budget**. Early stopping decides the actual
# count for each candidate, so every learning rate is judged at the number of
# trees it actually wants: measured on goals, 0.05 wants 213 and 0.01 wants 994.
# PROBE's 400 lets the first have what it needs and denies the second, so the
# descent was rejecting low learning rates for a reason that had nothing to do
# with the data. Raising the ceiling costs nothing at lr >= 0.05 (the cap was
# never reached) and about 3x at lr = 0.01, which is the point.
#
# `max_bin` here is only a starting value: `MAX_BIN_GRID` makes it a tuned axis,
# so what gets frozen is chosen rather than inherited from this line.
SEARCH = FitConfig(learning_rate=0.05, n_estimators=3000, early_stopping_rounds=50, max_bin=64)
SEARCH_TREE_CEILING = SEARCH.n_estimators

N_WORKERS = 6  # 8-core M2: 6 single-threaded fits beat 1 multi-threaded fit

# What `cv.evaluate` optimises, and therefore what every search stage selects on.
#
#   "team"   -- each side scored against its own actual (the training objective)
#   "total"  -- the convolved match total (what the priced product delivers)
#
# This is a *scoring definition*, so it is fingerprinted into checkpoint filenames
# and artifact provenance: a cached score measured under one mode is meaningless
# under the other, and `point_id` encodes neither. See `spec.scoring_hash()`.
#
# Changing this string is enough to make every existing checkpoint unreachable
# rather than silently reusable -- which is the point.
SCORING_MODE = "team"

# Selection tolerance: absolute log-loss slack, per target family. A candidate is
# eligible if it scores within this much of the reference score; among the
# eligible, the smallest feature set wins.
#
# Absolute rather than scaled by fold noise. Our measured fold standard error is
# ~0.0098 for goals, so a 1-SE band was ~10x wider than the 0.001 the reference
# framework used in every selection stage -- wide enough that against ~76 other
# features almost anything looks like noise, which is how whole groups were being
# eliminated. The achievable edge in this domain is itself only a few thousandths
# of a log-loss point, so the tolerance has to be smaller than that, not larger.
#
# TIGHTER keeps MORE features (harder to justify dropping one); looser prunes
# harder. These are the package defaults -- 02_Features.ipynb sets its own copy in
# its config cell, which is the one to edit when tuning.
SELECTION_TOL: dict[str, float] = {
    "goals": 0.001,
    "shots": 0.001,
    "sot": 0.001,
    "corners": 0.001,
}

# --- Two-tier tolerance ---------------------------------------------------
#
# SELECTION_TOL above is the *inner* tolerance: it governs each group's own
# culling inside stages 1-3, and nothing else. It shrinks on retry.
#
# The *outer* check is separate, fixed, and never shrinks. After each stage's
# groups are culled and recombined, the combined set is scored against the
# **full model** -- every candidate feature -- and must come in within a fixed
# multiple of the fold standard error of it:
#
#     combined_ll <= full_model_ll + MULT * fold_se
#
# NOT the naive per-league baseline. "Baseline" already means something specific
# in this codebase (`models.league_mean_baseline`, the Baseline section of
# 02_Features, and every comparison in evaluate.py), and the two numbers are far
# apart -- for goals, a full model of ~1.4568 against a naive baseline of
# ~1.5212. Gating on the naive baseline would be a much looser bar than intended
# and would let real compression damage through. This check says "the compressed
# set is still as good as using everything", which is the actual question.
#
# The full model is the right reference because it is the only fixed point the
# search cannot move: the anchor is itself stage 1's output, so gating later
# stages on it means an over-pruned stage 1 quietly moves the bar for everything
# downstream instead of being caught. ANCHOR stays computed and printed, and is
# informational past stage 1.
#
# A failing stage does not fail the run: it shrinks its inner tolerance and culls
# again from full membership. Only a stage that cannot pass even at tol=0 gives
# up, and it gives up by keeping everything rather than by raising.
#
# Named for what they measure against, deliberately: a `*_BASELINE_MULT` guarding
# a full-model check is exactly the collision these comments exist to prevent.
# Per target, like `SELECTION_TOL`, because the four do not need the same bar and
# one of them demonstrably cannot meet it. Shots' cheapest available stage-1 cull
# costs 2.18x its fold SE, so at 1.0 no culling can ever pass and stage 1 falls
# back to full membership -- which then hands stage 2 the maximal 12-member merged
# groups the search was designed to avoid.
#
# These were scalars, and `run_feature_selection` read them straight off this
# module rather than taking them as arguments -- so the copies in
# 02_Features.ipynb's config cell were dead names that changed nothing while
# looking exactly like the dials beside them that do work. They are parameters
# now; these are the defaults.
ANCHOR_FULL_MULT: dict[str, float] = {   # stage 1, whose output becomes the anchor
    "goals": 1.0,
    "shots": 1.0,
    "sot": 1.0,
    "corners": 1.0,
}
STAGE_FULL_MULT: dict[str, float] = {    # stages 2-5
    "goals": 2.0,
    "shots": 2.0,
    "sot": 2.0,
    "corners": 2.0,
}

# Fraction to shrink the inner tolerance by on each retry. 0.20 -> each attempt
# is 80% of the last, so a tolerance reaches negligible in ~60 attempts; in
# practice a stage passes within the first few or is not going to.
RETRY_SHRINK_PCT = 0.20

# Stage 3 (group-level elimination) enumerates every non-empty subset of the
# surviving groups when there are few enough -- 2**10 - 1 = 1,023 evaluations at
# the cap. Above it, fall back to greedy backward elimination, which is quadratic
# rather than exponential. With 20 groups the greedy path is the likely one
# unless stages 1-2 drop half of them.
GROUP_EXHAUSTIVE_MAX = 10

# Stage 4 (exhaustive over individual features), same shape one level down.
# 2**10 - 1 = 1,023 evaluations at the cap; the cost becomes awkward past ~12.
#
# Note the interaction with SELECTION_TOL: a tighter tolerance keeps more
# features, so this gate opens less often. A skipped stage 4 says so explicitly
# with the surviving count -- it is not silent.
FEATURE_EXHAUSTIVE_MAX = 10

__all__ = [
    "RANDOM_SEED",
    "League",
    "LEAGUES",
    "LOWER_LEAGUES",
    "LowerLeague",
    "PARENT_OF",
    "LOWER_OF",
    "SLUG_TO_KEY",
    "UNDERSTAT_TO_KEY",
    "LEAGUE_RANK",
    "season_sort_key",
    "season_label",
    "current_season",
    "current_season_label",
    "previous_season",
    "historic_seasons",
    "test_season",
    "first_val_season",
    "holdout_season",
    "set_seasons",
    "PROMOTED_LOOKBACK",
    "PROMOTED_WINDOW",
    "MIN_PROMOTED_SEASONS",
    "PRIOR_SEED_KAPPA",
    "TARGET_SHRINK",
    "GROWTH_F_MAX",
    "GROWTH_F_COARSE",
    "GROWTH_F_FINE",
    "GROWTH_DRAWDOWN_D",
    "GROWTH_DRAWDOWN_P",
    "GROWTH_PATHS",
    "GROWTH_ROUNDS",
    "GROWTH_DRAWDOWN_GRID_D",
    "GROWTH_DRAWDOWN_GRID_P",
    "GROWTH_DRAWDOWN_GRID_POINTS",
    "PESSIMISM_B",
    "SLATE_TAU",
    "MAX_LEG_STAKE",
    "CAP_RELIEF",
    "PAIRS_ENABLED",
    "PAIR_KEEP",
    "EXPORT_MAX",
    "PAIR_CORRELATION",
    "L_GRID",
    "ALPHA_GRID",
    "PAIR_BLOCKS",
    "TWEEDIE_BLOCK",
    "MAX_BIN_BLOCK",
    "MAX_TUNING_ROUNDS",
    "DESCENT_PASSES",
    "WINDOW_RECHECK_L_SPAN",
    "WINDOW_RECHECK_ALPHA_SPAN",
    "FitConfig",
    "PROBE",
    "FULL",
    "SEARCH",
    "SEARCH_TREE_CEILING",
    "N_WORKERS",
    "SCORING_MODE",
    "SELECTION_TOL",
    "ANCHOR_FULL_MULT",
    "STAGE_FULL_MULT",
    "RETRY_SHRINK_PCT",
    "GROUP_EXHAUSTIVE_MAX",
    "FEATURE_EXHAUSTIVE_MAX",
]
