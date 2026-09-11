"""Which propositions to back together, and in what mix.

`staking` answers a question about one proposition: is this price wrong in our
favour. This module answers the question after it -- given every proposition
whose price is wrong in our favour, which *combination* of them, staked which
way, best balances expected return against spread against the chance of ending
the week up.

There is no single answer, and that is the point. The three quantities trade
against each other, so what comes out is a set of portfolios none of which beats
another on all three, for a person to choose between.

There is, however, a single number that says which selection is worth the most.
``capacity`` -- ``C``, the sum of each leg's squared standalone Sharpe ratio --
is the maximum squared Sharpe any weighting of those legs can reach, it is
additive over legs, and the stake in `growth` is proportional to it. It is
reported and sortable and it is deliberately *not* a fourth dominance criterion:
the largest leg set always wins it, so adding it as an axis would leave almost
everything undominated. The frontier says what shapes are available; ``C`` says
which of them is worth the most.

The rules
---------
* At most one proposition per event. Two propositions from the same match -- a
  team's goals line and its own corners line -- are strongly correlated, and
  every variance and threshold figure here assumes independence. The rule and
  the assumption stand or fall together.

  Measured since, over 1,438 same-match pairs: +0.455 for the same team's same
  stat, +0.267 for its different stat, and **-0.185 for opposite teams**, against
  a -0.013 placebo across matches. So the correlation is real but far from
  duplication, and a hedged cross-team pair is worth more than an unrelated leg.
  Relaxing this rule is worth ~1.3x capacity and needs the joint distribution
  priced properly, not assumed away -- it is not done here yet.
* Events may be skipped. A portfolio is any non-empty selection of events with
  one proposition chosen from each.
* Only propositions with ``e >= 1`` are eligible, and within an event only those
  on the ``(p, o)`` frontier (see `staking.undominated`).

Why this is not a loop over combinations
----------------------------------------
On a real filled form -- 810 propositions, 180 priced, 57 clearing ``e >= 1``
across 26 events -- taking one from every event is ``prod(k_i)`` = 442,368
combinations, which enumerates in a hundredth of a second. Allowing events to be
skipped is ``prod(k_i + 1) - 1`` = **2.7e11**, which does not enumerate at all.
And that was a week with a 22% odds fill rate.

So the search is structural instead, and it works because every quantity a
portfolio is judged on is built from **sums over its legs**.

The stake split is ``s_i ~ mu_i / v_i`` with ``mu_i = o_i p_i - 1`` and
``v_i = o_i^2 p_i (1-p_i)`` -- inverse variance times edge. Write
``c_i = mu_i^2 / v_i`` and ``w_i = mu_i / v_i``, and let ``C = sum c_i``,
``W = sum w_i``. Then::

    stake_i              =  w_i / W
    expected return - 1  =  C / W
    variance             =  C / W^2
    Sharpe               =  sqrt(C)

That last line is why there is one split rather than the two this module used to
carry. ``mu/v`` weighting maximises portfolio Sharpe over *every* weighting, and
the maximum equals the root of the summed squared leg Sharpes exactly -- checked
against the discrete distribution on both settled slates, agreeing to 4e-16.
The two retired splits (``1/E``, minimum variance) both allocated on how quiet a
leg is and read neither its edge nor its Sharpe; on R006's placed book, minimum
variance put 25.5% of stake on the worst proposition in it and 4.3% on the best.

Two sums, each additive over legs, and a skipped event contributes ``(0, 0)``. So
the reachable space is a Minkowski sum over events and can be walked with a
dynamic program instead of enumerated: process events one at a time, and among
partial selections that agree on leg count and on ``W``, keep the ones with the
largest ``C``. Those partial selections are interchangeable under any common
remainder, so discarding the rest is exact.

`MAX_LEG_STAKE` is applied after the fact, in `cap_stakes`, and is the one place
the reported allocation departs from the closed forms above. The search ranks on
uncapped ``C`` as an upper bound; the cap lands at scoring time, where the numbers
are exact, and ``capacity_used`` reports what it cost.

What the grid does and does not cost
------------------------------------
``W`` is bucketed to make "agree on ``W``" finite. That costs **coverage, not
accuracy**: every surviving state carries its own true ``C`` and ``W``, so every
number this module reports is exact for the portfolio it belongs to. A finer
grid considers more portfolios; it never changes the score of one already
considered. `KEEP_PER_CELL` is the other half of the same dial -- keeping
several states per cell rather than one is what widens the search from the
frontier to a band around it, which matters (see `undominated_portfolios`).

The coverage loss is small and it is measured, not assumed. Run against the same
real 26-event form both ways -- enumerated, and with the search forced -- the
enumerated frontier has 90 portfolios and the searched one 87. The best expected
return is identical to the last bit, and each of the three misses sits within
7.4e-05 of a portfolio the search did find. So what the grid costs is a handful
of portfolios indistinguishable from their neighbours, not a corner of the
space.

Thresholds
----------
``P(return > t)`` needs the whole return distribution, which is the one thing
that does not reduce to a pair of sums. `staking.threshold_probs` enumerates
``2**n`` paths and cannot be used here -- twenty-six legs is 67 million paths for
*one* portfolio. `threshold_probs_batch` convolves the legs onto a value grid
instead: exact up to the grid, ~1 ms per portfolio, and vectorised across the
whole pool at once. It is pinned against the enumerating version in
`tests/test_portfolio.py`.

A saddlepoint approximation was tried first and rejected. It is faster, but the
return distribution is lumpy -- a handful of legs at meaningfully different
payouts -- and the continuous Lugannani-Rice formula smooths the steps away,
giving 0.003 to 0.02 absolute error against exact enumeration. The grid gives
0.0005 to 0.0025. One to two percentage points is too much when the whole
exercise is choosing between portfolios that differ by less than that.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import (
    EXPORT_MAX,
    MAX_LEG_STAKE,
    PAIR_CORRELATION,
    PAIR_KEEP,
    PAIRS_ENABLED,
)
from .staking import (
    P_CLAMP,
    label_parts,
    THRESHOLDS,
    add_edge,
    best_price,
    positive_edge,
    undominated,
)

# The stake split every portfolio is scored under. Closed form; see the module
# docstring for the algebra. The name is what the sheets display.
#
# `SPLIT_EVEN` and `SPLIT_MINVAR` are the two this replaced. They are kept as
# names, and out of `SPLITS`, because the ledger holds runs scored under them and
# `report.SPLIT_KEYS` has to resolve those rows -- not because either is still an
# option. Both allocate on how *quiet* a leg is and neither reads its edge, which
# is what made leg count look harmful: on R006's placed book, minimum variance
# put 25.5% of stake on the worst proposition in it (edge 1.003, Sharpe 0.006)
# and 4.3% on the best (Sharpe 0.367).
SPLIT_EVEN = "1/E"
SPLIT_MINVAR = "Min variance"
SPLIT_GROWTH = "Growth"
SPLITS: tuple[str, ...] = (SPLIT_GROWTH,)
LEGACY_SPLITS: tuple[str, ...] = (SPLIT_EVEN, SPLIT_MINVAR)

# Enumerate exactly rather than search when the whole space fits. Two million
# combinations cost about a second and a few hundred MB, and the exhaustive path
# is what the dynamic program is validated against, so it earns its keep even
# when it rarely fires.
EXHAUSTIVE_MAX = 2_000_000

# Portfolios carried forward to the threshold stage. This is the real budget:
# thresholds cost ~1 ms each and both splits are scored, so it sets the runtime
# almost single-handedly. Everything upstream is fast enough not to matter.
#
# It is also the single biggest determinant of how much of the answer you see,
# which is why it is not smaller. Ground truth on a 9-event slice of a real form
# with skipping allowed -- 408,239 portfolios, all of them scored, 496 genuinely
# undominated on all three criteria:
#
#     cap        portfolios reaching the threshold stage   true winners found
#     25,000                    19,016                          424 / 496
#     unlimited                115,992                          491 / 496
#
# The 72 that a 25,000 cap was hiding were not lost by the search or by the band
# width -- both paths missed exactly the same ones -- they were thrown away by
# this number. On the full 26-event form the cost of raising it is:
#
#     cap        pool     undominated shown    runtime
#     25,000    24,950            912             61 s
#     100,000  ~100,000         ~1,400           ~140 s
#     250,000   235,758          1,847            521 s
#
# 100,000 is where the curve bends: it buys back most of the coverage for a
# little over two minutes. Best expected return and best profitability are
# identical at every setting -- the extremes were never at risk. What grows is
# the middle of the list, which is the part you actually choose from.
POOL_MAX = 100_000

# Resolution of the `A` (and `W`) axis in the dynamic program, and how many
# states to keep per cell. See "What the grid does and does not cost" above:
# BUCKETS widens which portfolios are considered, KEEP_PER_CELL widens the band
# around the frontier. Eight is deliberately generous -- on real data 11 of the
# 65 portfolios that are undominated on all three criteria sit *off* the
# two-dimensional frontier, so keeping only the best per cell would miss them.
#
# 4096 rather than 1024 because 1024 is not enough to be exact. Measured against
# full enumeration of a real 26-event form (442,368 portfolios, 90 on the
# frontier), forcing the searched path:
#
#     buckets   frontier found   best expected return   worst miss     time
#        256          32 / 90            exact            1.7e-03      0.1s
#       1024          87 / 90            exact            7.4e-05      0.6s
#       4096          90 / 90            exact            0            3.1s
#      16384          90 / 90            exact            0           23.3s
#
# A 26-event form packs its `A` values into a narrow band -- every edge is a
# little over 1, so `A` sits just under `n` -- and 1024 buckets cannot separate
# them. Smaller spaces are exact at 1024; this one is not, and this one is the
# real one.
#
# End to end on that form with skips allowed the difference is 35s against 62s,
# and the *answer improves*: a wider search finds more dominators, so the
# undominated set falls from 989 to 912. The 77 that dropped out were never
# undominated -- the search had simply not found what beat them.
BUCKETS = 4096
KEEP_PER_CELL = 8

# Cells of the return grid the leg payouts are convolved onto. 4096 measures at
# 0.0005-0.0025 absolute against exact enumeration, which is an order of
# magnitude finer than the model probabilities feeding it.
# Value-grid resolution for `threshold_probs_batch`. Raised from 4096 after the
# quantisation error was measured against `threshold_probs_exact` on the *real*
# 8-16 leg shapes rather than on synthetic ones:
#
#     grid    worst err   p_over_100   rows over the 0.005 test tolerance
#     4096     1.42e-02     1.42e-02   6 of 40
#     8192     1.42e-02     1.42e-02   1 of 40
#    16384     3.20e-03     1.37e-04   0 of 40
#    32768     7.51e-04     5.55e-16   0 of 40
#
# 4096 was not merely imprecise, it was deciding the frontier: `p_over_100` is a
# dominance criterion, and rebuilding dominance over one day's exported set at
# 16384 moved **260 of 2,740** portfolios in or out. An error of 1.4pp on the
# axis that ranks them is not a rounding detail.
#
# It costs about 5x -- ~131s to ~684s to score 200,000 rows -- which is the price
# of the frontier being the real one. 8192 is not the cheaper compromise it looks
# like: it leaves the worst case untouched.
GRID = 16384

# Rows per chunk in the threshold convolution: `CHUNK * GRID` float64 is the
# working set, so this is a memory dial, not a correctness one.
CHUNK = 2048

_SKIP = -1  # a portfolio's pick for an event it does not back


# --- Qualifying propositions ----------------------------------------------


def qualify(filled: pd.DataFrame) -> pd.DataFrame:
    """Filled odds form -> the propositions a portfolio may be built from.

    Best price across the books, then edge, then ``e >= 1``, then the ``(p, o)``
    frontier **within each event**. An event that loses everything simply has no
    options and can never appear in a portfolio; that is a normal outcome.
    """
    priced = best_price(filled)
    scored = positive_edge(add_edge(priced))
    if scored.empty:
        return scored
    kept = [undominated(g) for _, g in scored.groupby("sheet_code", sort=False)]
    out = pd.concat(kept, ignore_index=True)
    # What a proposition is *about* only exists in its name -- the odds form
    # carries no separate columns for it -- and pairing two of them needs to know
    # whether they share a team or a market. Attached here, where the canonical
    # proposition frame is made, so `event_options`, `legs` and the payload all
    # read one derivation. Tolerant: a label this cannot read is a proposition
    # that cannot be paired, which is a normal outcome rather than an error.
    for col, values in label_parts(out["label"]).items():
        if col not in out:
            out[col] = values
    return out.sort_values(["sheet_code", "p"], ascending=[True, False]).reset_index(drop=True)


@dataclass(frozen=True)
class Options:
    """The per-event menu, in the layout the search wants.

    Everything is indexed ``[event][option]``. `props` keeps the original rows so
    a portfolio can be expanded back into propositions with prices and books
    attached, which is what makes the output actionable rather than abstract.

    An **option is one proposition or two**. ``p``/``o`` are its first slot and
    ``p2``/``o2`` its second, zero where the option is a single; ``rho`` is the
    pair's correlation and is zero there too. ``members`` records which rows of
    `props` each option covers, so `legs` can expand it back.

    The second slot is carried as zeros rather than as ragged per-option lists
    because a zero-probability leg is already a genuine no-op in every consumer --
    `metrics`, `return_pmf`, `ruin_prob` -- so singles and pairs score through one
    path with no branching.

    All four pair fields default to empty and are filled with the single-option
    shape, so an `Options` built the old way (three tuples and a frame) still
    constructs and behaves exactly as it did.
    """

    codes: tuple[str, ...]              # sheet_code per event, in search order
    p: tuple[np.ndarray, ...]           # model probability, first slot
    o: tuple[np.ndarray, ...]           # best bookmaker price, first slot
    props: pd.DataFrame                 # the source rows, with `event` and `option`
    p2: tuple[np.ndarray, ...] = ()     # second slot, 0 where the option is single
    o2: tuple[np.ndarray, ...] = ()
    rho: tuple[np.ndarray, ...] = ()    # the pair's correlation, 0 for a single
    members: tuple[tuple[tuple[int, ...], ...], ...] = ()   # [event][option] -> prop rows
    n_events: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_events", len(self.codes))
        zeros = tuple(np.zeros_like(x) for x in self.p)
        if not self.p2:
            object.__setattr__(self, "p2", zeros)
        if not self.o2:
            object.__setattr__(self, "o2", tuple(np.zeros_like(x) for x in self.p))
        if not self.rho:
            object.__setattr__(self, "rho", tuple(np.zeros_like(x) for x in self.p))
        if not self.members:
            object.__setattr__(self, "members", tuple(
                tuple((i,) for i in range(len(x))) for x in self.p))

    @property
    def sizes(self) -> np.ndarray:
        return np.array([len(x) for x in self.p], dtype=np.int64)

    @property
    def has_pairs(self) -> bool:
        return any(bool((x > 0).any()) for x in self.p2)

    def min_legs_for(self, leg_var: int) -> int:
        """The leg floor sitting ``leg_var`` events below everything on offer.

        The floor has to be **relative**, because the number it is relative to is
        not knowable when the run is configured. A weekend card of 45 fixtures may
        qualify 43 events or 37, depending on which markets got priced and which
        cleared ``e >= 1``; typing ``min_legs = 38`` against a guess of 43 quietly
        becomes "skip nothing" if only 38 qualify, and fails outright if 36 do.

        ``leg_var`` says the thing actually meant: *how many events am I willing to
        leave out*. Zero is "back every qualifying event", which is `min_legs="all"`
        by another name. Anything larger opens that many rungs below it.

        Floored at 1, so an over-large `leg_var` opens the search up rather than
        asking for a portfolio with no legs.
        """
        if leg_var < 0:
            raise ValueError(f"leg_var must be >= 0, got {leg_var}")
        return max(1, self.n_events - int(leg_var))

    def space(self, min_legs: int = 1, max_legs: int | None = None) -> float:
        """How many portfolios have between ``min_legs`` and ``max_legs`` legs.

        Exact, via the generating polynomial ``prod(1 + k_j x)``: the coefficient
        of ``x**n`` counts the selections with exactly ``n`` legs, so summing the
        coefficients in range counts the legal ones. That matters because the
        naive ``prod(k_j + 1) - 1`` is only right when any number of legs is
        allowed -- pinning the count to every event makes the real space
        ``prod(k_j)``, which on a real form is 442 thousand rather than 2.7e11,
        and the difference decides whether this is enumerable.

        Float, because the unbounded count overflows int64 routinely.
        """
        max_legs = self.n_events if max_legs is None else max_legs
        poly = np.zeros(self.n_events + 1)
        poly[0] = 1.0
        for k in self.sizes:
            poly[1:] += poly[:-1] * k          # take one of k, or skip
        return float(poly[max(min_legs, 0):max_legs + 1].sum())

    def enumeration_cost(self, min_legs: int = 1) -> float:
        """Rows a mixed-radix enumeration would materialise.

        Distinct from `space`: enumeration walks a rectangular product and then
        filters, so its cost is the product and not the count of survivors. The
        one exception is worth taking, because it is the mode this is most often
        run in -- when every event must be backed there is no skip digit at all,
        so the radix is ``k_j`` and the walk enumerates exactly the legal set.
        """
        radix = self.sizes if min_legs >= self.n_events else self.sizes + 1
        return float(np.prod(radix.astype(float)))


def pair_rho(a: pd.Series, b: pd.Series) -> float:
    """The measured correlation between two propositions in the same match.

    Graded by what they share, from `config.PAIR_CORRELATION`. Same team and same
    stat is graded again by how far apart the two lines are, because the headline
    0.455 averages over a real gradient: adjacent lines are nearly the same bet at
    0.63, three or more apart much less so at 0.30.
    """
    if a["team"] != b["team"]:
        return PAIR_CORRELATION["opposite_teams"]
    if a["target"] != b["target"]:
        return PAIR_CORRELATION["same_team_diff_stat"]
    gap = abs(float(a["line"]) - float(b["line"]))
    if gap <= 1.0:
        return PAIR_CORRELATION["same_team_same_stat_adjacent"]
    if gap <= 2.0:
        return PAIR_CORRELATION["same_team_same_stat_near"]
    return PAIR_CORRELATION["same_team_same_stat_far"]


def _pair_candidates(g: pd.DataFrame, keep: int) -> list[tuple[int, int, float]]:
    """``(i, j, rho)`` for the pairs of one event worth offering the search.

    Every pair is priced first -- one 2x2 solve each, a few thousand per slate, and
    the cost was never in *generating* pairs but in carrying them through the
    dynamic programme. Then the ``(c, w)``-undominated set is kept, capped at
    `keep`.

    **Ranking the pairs rather than the propositions is what makes this
    correlation-aware for free**, because ``c`` already contains ``rho``. Ranking
    propositions by standalone Sharpe and pairing the best few is not, and it
    discards exactly the pairs worth having: where a cross-team hedge wins, its
    weaker leg typically ranks fifth or worse on its own.

    ``(c, w)`` and not ``c`` alone, because the search's state is
    ``(n_events, bucket(W))`` ranked on ``C`` -- two options with equal capacity at
    different normalisers land in different buckets and are both useful. The same
    shape as `staking.undominated`'s ``(p, o)`` frontier, one level up.
    """
    n = len(g)
    if n < 2 or keep <= 0:
        return []
    # A proposition whose label would not parse cannot be classified against
    # another, so it is offered on its own and never in a pair.
    known = g[["team", "target", "line"]].notna().all(axis=1).to_numpy()
    idx = [(a, b) for a, b in itertools.combinations(range(n), 2)
           if known[a] and known[b]]
    if not idx:
        return []
    i = np.fromiter((a for a, _ in idx), int, len(idx))
    j = np.fromiter((b for _, b in idx), int, len(idx))
    rho = np.array([pair_rho(g.iloc[a], g.iloc[b]) for a, b in idx], dtype=float)
    p = g["p"].to_numpy(float); o = g["o"].to_numpy(float)
    t = pair_terms(p[i], o[i], p[j], o[j], rho)

    live = t["c"] > 0                       # a lay-requiring pair prices to zero
    order = np.argsort(-t["c"][live], kind="stable")
    c, w = t["c"][live][order], t["w"][live][order]
    ii, jj, rr = i[live][order], j[live][order], rho[live][order]

    out, w_min = [], np.inf
    for k in range(len(c)):
        if w[k] < w_min:                    # undominated: no better c at no more w
            out.append((int(ii[k]), int(jj[k]), float(rr[k])))
            w_min = w[k]
            if len(out) >= keep:
                break
    return out


def event_options(qualified: pd.DataFrame, *, pairs: bool = PAIRS_ENABLED,
                  pair_keep: int = PAIR_KEEP) -> Options:
    """Group qualifying propositions into the per-event menu.

    Each event offers every qualifying proposition on its own, plus -- with
    ``pairs`` on -- up to `pair_keep` of its two-proposition combinations. Singles
    are never restricted; only pairs are filtered.

    Pairing needs ``team``, ``target`` and ``line`` to know what two propositions
    share, which is what sets their correlation, and `qualify` attaches all three
    from the label. **A proposition missing any of them is never paired** -- not
    paired at a guessed correlation, because guessing here does not produce a
    slightly wrong number, it produces a capacity that can be 43% too high and a
    stake to match.
    """
    if not {"team", "target", "line"} <= set(qualified.columns):
        pairs = False
    codes, ps, os_, p2s, o2s, rhos, members, frames = [], [], [], [], [], [], [], []
    for i, (code, g) in enumerate(qualified.groupby("sheet_code", sort=False)):
        g = g.reset_index(drop=True)
        n = len(g)
        cand = _pair_candidates(g, pair_keep) if pairs else []
        p = g["p"].to_numpy(float); o = g["o"].to_numpy(float)

        codes.append(str(code))
        ps.append(np.r_[p, [p[a] for a, _, _ in cand]])
        os_.append(np.r_[o, [o[a] for a, _, _ in cand]])
        p2s.append(np.r_[np.zeros(n), [p[b] for _, b, _ in cand]])
        o2s.append(np.r_[np.zeros(n), [o[b] for _, b, _ in cand]])
        rhos.append(np.r_[np.zeros(n), [r for _, _, r in cand]])
        members.append(tuple([(k,) for k in range(n)] + [(a, b) for a, b, _ in cand]))
        # `option` indexes the *option* list, so a proposition's own row keeps the
        # index it always had and pairs are appended after. `picks_label`'s
        # `CODE#k` token therefore still resolves, and older payloads still read.
        frames.append(g.assign(event=i, option=np.arange(n)))
    props = (pd.concat(frames, ignore_index=True) if frames
             else qualified.assign(event=pd.Series(dtype=int), option=pd.Series(dtype=int)))
    return Options(tuple(codes), tuple(ps), tuple(os_), props,
                   tuple(p2s), tuple(o2s), tuple(rhos), tuple(members))


# --- Per-option separable quantities ---------------------------------------


def leg_terms(p: np.ndarray, o: np.ndarray) -> dict[str, np.ndarray]:
    """The two additive per-leg quantities the whole search is built on.

    ``c`` is the leg's **capacity** -- the square of its standalone Sharpe ratio
    -- and ``w`` its unnormalised growth weight. Both are sums over legs, and
    every quantity the search and the report need falls out of the pair::

        C = sum(c)                 W = sum(w)
        stake_i             = w_i / W
        expected return - 1 = C / W
        variance            = C / W**2
        Sharpe              = sqrt(C)

    That last line is the reason this module now carries one split rather than
    two. Weighting by ``mu / v`` maximises portfolio Sharpe over *all* weightings,
    and the maximum it reaches is ``sqrt(sum of squared leg Sharpes)`` exactly --
    checked against the discrete distribution on both settled slates at 4, 8, 12
    and full size, agreeing to 4e-16. So leg count enters the model through one
    additive scalar, and adding a leg can never lower it.

    ``mu`` is clamped at zero rather than allowed negative. `qualify` admits only
    ``e >= 1`` so a negative edge cannot arrive here, but ``e == 1`` exactly is
    reachable and must contribute nothing rather than a signed weight.

    For a **pair**, the same two quantities come from the 2x2 solve in
    `pair_terms`, and both stay additive over events -- which is what lets the
    dynamic programme stay exactly as it was.
    """
    pc = np.clip(p, P_CLAMP, 1.0 - P_CLAMP)
    mu = np.maximum(o * p - 1.0, 0.0)
    v = o**2 * pc * (1.0 - pc)
    return {"c": mu**2 / v, "w": mu / v}


def pair_terms(p1: np.ndarray, o1: np.ndarray, p2: np.ndarray, o2: np.ndarray,
               rho: np.ndarray) -> dict[str, np.ndarray]:
    """``(c, w, b1, b2)`` for options that may hold two correlated propositions.

    The generalisation of `leg_terms` to a pair. With edges ``mu`` and covariance
    ``S = [[v1, rho sqrt(v1 v2)], [rho sqrt(v1 v2), v2]]``::

        b = S^-1 mu        c = mu . b        w = b1 + b2

    which reduces to `leg_terms` exactly when the second slot is empty, so one
    function prices both and singles need no special case.

    **This is where the correlation earns its keep.** Two legs of equal Sharpe at
    ``rho`` are worth ``2/(1+rho)`` independent legs, not two: at the measured
    0.455 for a team's same stat, the second leg is worth 0.37 of one. Pricing the
    pair as if it were independent inflates capacity -- by 43% on the 11 Sept
    slate -- and since the suggested stake is proportional to capacity, it
    overstakes by the same margin.

    A pair whose solve wants a negative weight is rejected by returning zero
    capacity: that is a hedge requiring a lay, and a lay is not a bet available
    here. The caller drops those rather than offering them.
    """
    pc1 = np.clip(p1, P_CLAMP, 1.0 - P_CLAMP)
    pc2 = np.clip(p2, P_CLAMP, 1.0 - P_CLAMP)
    mu1 = np.maximum(o1 * p1 - 1.0, 0.0)
    mu2 = np.where(p2 > 0, np.maximum(o2 * p2 - 1.0, 0.0), 0.0)
    # Both variances are floored at 1 where their slot is empty. A skipped event
    # arrives here with o = p = 0 on *both* slots, which would otherwise make the
    # determinant vanish and divide by zero -- the result was discarded by the
    # `np.where` below, but only after numpy had computed and warned about it.
    v1 = np.where(p1 > 0, o1**2 * pc1 * (1.0 - pc1), 1.0)
    v2 = np.where(p2 > 0, o2**2 * pc2 * (1.0 - pc2), 1.0)
    cov = np.where(p2 > 0, rho * np.sqrt(v1 * v2), 0.0)

    # Closed form for the 2x2 inverse. `det` is strictly positive: |rho| < 1 is
    # enforced upstream and both variances are now strictly positive.
    det = v1 * v2 - cov**2
    b1 = (v2 * mu1 - cov * mu2) / det
    b2 = np.where(p2 > 0, (v1 * mu2 - cov * mu1) / det, 0.0)
    bad = (b1 < 0) | (b2 < 0)
    b1 = np.where(bad, 0.0, b1)
    b2 = np.where(bad, 0.0, b2)
    return {"c": mu1 * b1 + mu2 * b2, "w": b1 + b2, "b1": b1, "b2": b2}


def option_terms(opts: Options, j: int) -> dict[str, np.ndarray]:
    """The additive `(c, w)` pair for every option of event ``j``, pairs included."""
    return pair_terms(opts.p[j], opts.o[j], opts.p2[j], opts.o2[j], opts.rho[j])


# --- Stakes and exact metrics ----------------------------------------------


def stakes_for(picks: np.ndarray, opts: Options, split: str,
               max_leg_stake: float = MAX_LEG_STAKE
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(stakes, p, o, rho)`` per portfolio.

    ``stakes``, ``p`` and ``o`` are ``(n_portfolios, 2 * n_events)`` -- two slots
    per event, the second filled only where the chosen option is a pair. ``rho`` is
    ``(n_portfolios, n_events)``, one correlation per event block, zero for a
    single.

    A skipped event, and the empty slot of a single, get ``stake = 0`` and
    ``p = 0``, which makes them genuine no-ops everywhere downstream: they
    contribute nothing to any sum, and in the threshold convolution a leg with
    ``p = 0`` leaves the distribution untouched. Carrying them as zeros rather than
    as ragged per-portfolio arrays is what lets every portfolio be scored in one
    vectorised pass regardless of length -- and it is what lets a pair cost no
    special case.
    """
    m, n_ev = picks.shape
    p = np.zeros((m, 2 * n_ev)); o = np.zeros((m, 2 * n_ev)); rho = np.zeros((m, n_ev))
    for j in range(n_ev):
        taken = picks[:, j] >= 0
        idx = picks[taken, j]
        p[taken, 2 * j] = opts.p[j][idx]
        o[taken, 2 * j] = opts.o[j][idx]
        p[taken, 2 * j + 1] = opts.p2[j][idx]
        o[taken, 2 * j + 1] = opts.o2[j][idx]
        rho[taken, j] = opts.rho[j][idx]

    active = p > 0
    if split == SPLIT_GROWTH:
        # The same `mu/v` weighting, solved per event block so a pair's two legs
        # are weighted against each other *through* their correlation rather than
        # as if they were unrelated.
        t = pair_terms(p[:, 0::2], o[:, 0::2], p[:, 1::2], o[:, 1::2], rho)
        weight = np.zeros_like(p)
        weight[:, 0::2] = np.where(active[:, 0::2], t["b1"], 0.0)
        weight[:, 1::2] = np.where(active[:, 1::2], t["b2"], 0.0)
    elif split in LEGACY_SPLITS:
        raise ValueError(
            f"{split!r} was retired; portfolios are scored under {SPLIT_GROWTH!r} only. "
            "Ledger rows written under it still resolve by name.")
    else:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")

    # Every leg at exactly `e == 1` leaves nothing to weight by. Falling back to
    # equal stakes keeps the row scoreable and is honest about what it is: a
    # portfolio with no edge to allocate on, which the metrics will then say.
    total = weight.sum(axis=1, keepdims=True)
    flat = total[:, 0] <= 0
    if flat.any():
        n_props = active[flat].sum(axis=1, keepdims=True)
        weight[flat] = np.where(active[flat], 1.0, 0.0) / np.maximum(n_props, 1)
        total[flat] = 1.0
    return cap_stakes(weight / total, active, max_leg_stake), p, o, rho


def cap_stakes(stakes: np.ndarray, active: np.ndarray,
               limit: float = MAX_LEG_STAKE) -> np.ndarray:
    """No single leg carries more than `limit` of the stake. Excess spread pro-rata.

    Growth weights concentrate by design -- 24% of stake on one leg on R006's full
    book, and effective legs down from 12.6 to 5.6 against minimum variance. That
    is correct for compounding and uncomfortable for model risk, since it means one
    mispriced proposition moves the week.

    Redistributing pro-rata rather than equally keeps the *relative* ordering of
    everything under the cap, so the allocation stays growth-optimal among the legs
    that are not capped. Iterated because lifting the uncapped legs can push one of
    them over the line in turn; it converges because each pass either caps a new
    leg or stops, and there are finitely many legs.

    The limit is applied **per row as ``max(limit, 1 / n_legs)``**. Six legs cannot
    each hold under 15% -- the stake has to go somewhere -- so for short portfolios
    the cap relaxes to the equal-weight floor, which is the tightest constraint that
    can be satisfied. Clipping to an unreachable limit instead would silently leave
    the row staking less than the whole amount, which is a different bet from the
    one being scored.

    `score_pool` reports `capacity_used` alongside, so what the cap costs in
    capacity is always on the row rather than buried in it.
    """
    if not np.isfinite(limit) or limit >= 1.0:
        return stakes
    s = stakes.copy()
    n_legs = active.sum(axis=1, keepdims=True)
    row_limit = np.maximum(limit, 1.0 / np.maximum(n_legs, 1))
    for _ in range(64):
        over = (s > row_limit + 1e-15) & active
        if not over.any():
            break
        excess = np.where(over, s - row_limit, 0.0).sum(axis=1, keepdims=True)
        s = np.where(over, np.broadcast_to(row_limit, s.shape), s)
        room = np.where(~over & active, np.maximum(row_limit - s, 0.0), 0.0)
        room_total = room.sum(axis=1, keepdims=True)
        movable = room_total[:, 0] > 0
        if not movable.any():
            break
        share = np.zeros_like(s)
        share[movable] = room[movable] / room_total[movable]
        s = s + share * excess
    return s


def metrics(stakes: np.ndarray, p: np.ndarray, o: np.ndarray,
            rho: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(expected_return, variance)`` per portfolio, as fractions of one stake.

    Computed as the plain sums `staking.expected_return` and `staking.variance`
    define, one row at a time in parallel -- *not* through the ``n / A`` and
    ``B / A**2`` closed forms the search uses. The closed forms exist to make the
    search cheap; these exist to make the report obviously right, and
    `tests/test_portfolio.py` pins the two against each other. Deriving the
    report from the optimisation's own shortcut would mean an error in the
    algebra could never show up as a wrong number.
    """
    er = (stakes * o * p).sum(axis=1)
    var = (stakes**2 * o**2 * p * (1.0 - p)).sum(axis=1)
    if rho is not None:
        # The covariance term, one per event block: 2 s1 s2 rho sqrt(v1 v2). Zero
        # wherever the block holds one proposition, because `rho` is zero there.
        #
        # Without this the variance of a paired portfolio is the variance it would
        # have if its pairs were unrelated -- too low, and in the direction that
        # flatters. It is the second of the four places the correlation has to
        # land; see `pair_terms` for what missing one costs.
        s1, s2 = stakes[:, 0::2], stakes[:, 1::2]
        v1 = (o[:, 0::2] ** 2) * p[:, 0::2] * (1.0 - p[:, 0::2])
        v2 = (o[:, 1::2] ** 2) * p[:, 1::2] * (1.0 - p[:, 1::2])
        var = var + 2.0 * (s1 * s2 * np.asarray(rho) * np.sqrt(v1 * v2)).sum(axis=1)
    return er, var


# --- Threshold probabilities -----------------------------------------------


def threshold_probs_batch(pay: np.ndarray, p: np.ndarray,
                          thresholds=THRESHOLDS, grid: int = GRID,
                          chunk: int = CHUNK, *,
                          rho: np.ndarray | None = None) -> np.ndarray:
    """``P(return > t)`` for every portfolio and every ``t``. Shape ``(m, len(t))``.

    ``pay`` is each leg's payout as a fraction of the total stake -- ``s_i * o_i``
    -- and ``p`` the matching hit probability, both zero for a skipped event.

    Each portfolio's return is a sum of independent scaled Bernoullis, so its
    distribution is the convolution of ``n`` two-point distributions. Quantising
    the payouts onto a shared value grid turns each convolution into a shift and
    two multiply-adds, which vectorises across the whole batch.

    The grid step is **per portfolio** -- ``sum(pay) / (grid - 1)`` -- rather than
    shared. A common step would have to span the largest total return in the
    batch and would then waste most of its resolution on portfolios that cannot
    reach it; per-row scaling gives every portfolio the full grid over its own
    range, at no extra cost, because the shift amounts are per-row anyway.

    Quantisation is the only error, it is bounded by half a step per leg, and it
    is measured rather than assumed -- see the accuracy test.
    """
    pay = np.asarray(pay, dtype=float)
    p = np.asarray(p, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    m = len(pay)
    out = np.empty((m, len(thresholds)))

    for lo in range(0, m, chunk):
        hi = min(lo + chunk, m)
        dist, step = return_pmf(pay[lo:hi], p[lo:hi], grid,
                                rho=None if rho is None else rho[lo:hi])
        cdf = np.cumsum(dist, axis=1)
        # P(R > t) = 1 - P(R <= t); a threshold above the largest reachable
        # return lands past the last cell and correctly reads probability zero.
        for c, t in enumerate(thresholds):
            kt = np.clip(np.floor(t / step).astype(np.int64), 0, grid - 1)
            out[lo:hi, c] = 1.0 - np.take_along_axis(cdf, kt[:, None], axis=1)[:, 0]
    return np.clip(out, 0.0, 1.0)


def return_pmf(pay: np.ndarray, p: np.ndarray, grid: int = GRID,
               *, rho: np.ndarray | None = None,
               close_tail: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Each portfolio's return distribution, on its own value grid.

    ``(dist, step)``: ``dist[i, k]`` is the probability that portfolio ``i``
    returns ``k * step[i]``, and ``step[i] = sum(pay[i]) / (grid - 1)``. Rows are
    a full pmf, so ``dist.sum(axis=1)`` is 1 up to floating point.

    This is the convolution `threshold_probs_batch` has always run; it was inline
    and the distribution was discarded once the thresholds were read off it. It is
    exposed because `growth` needs the whole distribution rather than five points
    on it, and taking it from here keeps **one** engine: the growth block and the
    ``P(>t)`` columns are then two readings of the same numbers, which is the rule
    `staking.threshold_probs` sets out. Enumerating ``2**n`` separately would be a
    second engine, and 5x the work at the leg counts this actually runs at.

    ``close_tail`` folds quantisation loss back into the last cell. Each leg's
    index is ``rint(pay_j / step)``, so a row's indices can sum to as much as
    ``(grid - 1) + n/2`` while its largest *true* return is exactly ``sum(pay)``,
    which is the last cell. Paths that overrun are dropped by the shift, and the
    mass lost is therefore the all-win corner -- not probability that belongs
    anywhere else. On the 2,740 exported portfolios of 2026-08-22, 956 rows
    overrun by up to 4 cells, 51 lose more than 1e-3 of their mass and the worst
    loses 5.6e-2.

    It defaults **off** so `threshold_probs_batch` returns exactly what it always
    has -- every stored ``P(>t)`` stays reproducible. `growth` turns it on,
    because a pmf that integrates to 0.944 would silently under-weight ``g(f)``
    by the same 5.6%, and unlike a threshold read-off there is no clipping step
    downstream to absorb it.

    Does not chunk -- callers do, so the ``(rows, grid)`` array stays bounded.

    Convolved in **blocks of two columns**, not one at a time
    -------------------------------------------------------
    Columns arrive paired: slot 0 of an event and slot 1, the latter filled only
    where the portfolio takes two propositions from that match. ``rho`` carries
    each block's correlation and is zero everywhere a block holds one leg or two
    unrelated ones.

    A block convolves through its joint distribution::

        P11 = p1 p2 + rho sqrt(p1 q1 p2 q2)       (clamped to the Frechet bounds)
        P10 = p1 - P11    P01 = p2 - P11    P00 = 1 - p1 - p2 + P11

    which is one code path for every case, because it degenerates exactly:

    * ``rho = 0`` gives ``P11 = p1 p2`` -- independence, identical to shifting the
      two legs separately, which is what this did before and what the bit-for-bit
      test pins.
    * ``p2 = 0`` gives ``P11 = P01 = 0``, ``P10 = p1`` -- a single leg, exactly.

    That matters more than the tidiness. The correlation has to reach the
    *distribution*, not just the reported variance: ``P(>t)``, ``p0`` and every
    growth number are read off this array, and a pair charged its correlation in
    the variance but not here would understate risk everywhere it is looked at.
    Measured on the 11 Sept slate, ignoring the correlation inflates capacity by
    43% -- and since the stake is proportional to capacity, it overstakes by the
    same 43% while every figure still looks plausible.
    """
    pay = np.asarray(pay, dtype=float)
    p = np.asarray(p, dtype=float)
    m, n = pay.shape
    # Pad to an even column count so the reshape below is total. A trailing
    # all-zero column is the no-op every zero-probability column already is.
    if n % 2:
        pay = np.concatenate([pay, np.zeros((m, 1))], axis=1)
        p = np.concatenate([p, np.zeros((m, 1))], axis=1)
    blocks = pay.shape[1] // 2
    rho = np.zeros((m, blocks)) if rho is None else np.asarray(rho, dtype=float)

    idx = np.arange(grid)
    total = pay.sum(axis=1, keepdims=True)
    # A portfolio with no legs has nowhere to put its stake; MIN_LEGS keeps
    # them out, and this guard only stops the division from warning.
    step = np.where(total > 0, total, 1.0) / (grid - 1)
    k = np.rint(pay / step).astype(np.int64)

    def shifted(dist: np.ndarray, amount: np.ndarray) -> np.ndarray:
        src = idx[None, :] - amount[:, None]
        return np.where(src >= 0,
                        np.take_along_axis(dist, np.clip(src, 0, grid - 1), axis=1), 0.0)

    dist = np.zeros((m, grid))
    dist[:, 0] = 1.0
    for b in range(blocks):
        p1, p2 = p[:, 2 * b], p[:, 2 * b + 1]
        k1, k2 = k[:, 2 * b], k[:, 2 * b + 1]
        both = joint_both(p1, p2, rho[:, b])
        d = (dist * (1.0 - p1 - p2 + both)[:, None]
             + shifted(dist, k1) * (p1 - both)[:, None]
             + shifted(dist, k2) * (p2 - both)[:, None]
             + shifted(dist, k1 + k2) * both[:, None])
        dist = d
    if close_tail:
        dist[:, -1] += 1.0 - dist.sum(axis=1)
    return dist, step[:, 0]


def joint_both(p1: np.ndarray, p2: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """``P(both legs land)`` for a correlated pair, clamped to what is reachable.

    ``p1 p2 + rho sqrt(p1 q1 p2 q2)`` is the two-point joint with the requested
    correlation. Not every ``rho`` is attainable for a given pair of marginals --
    two legs at ``p = 0.2`` cannot correlate at 0.9 -- so the result is clamped to
    the Frechet bounds ``[max(0, p1 + p2 - 1), min(p1, p2)]``.

    Clamping rather than raising, because the correlations in `PAIR_CORRELATION`
    are population averages over a band of lines and a specific pair of marginals
    can sit outside what the average implies. The clamp is towards the *attainable*
    correlation nearest the estimate, which is the right answer rather than a
    convenient one.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    q1, q2 = 1.0 - p1, 1.0 - p2
    both = p1 * p2 + np.asarray(rho, dtype=float) * np.sqrt(
        np.clip(p1 * q1 * p2 * q2, 0.0, None))
    return np.clip(both, np.maximum(0.0, p1 + p2 - 1.0), np.minimum(p1, p2))


def threshold_probs_exact(pay: np.ndarray, p: np.ndarray, thresholds=THRESHOLDS) -> np.ndarray:
    """The same answer by enumerating all ``2**n`` outcomes. One portfolio.

    Costs ``2**n`` and exists to pin `threshold_probs_batch`, not to be called
    from a reporting path -- see the note on `staking.threshold_probs`.
    """
    pay = np.asarray(pay, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    live = p > 0
    pay, p = pay[live], p[live]
    n = len(p)
    bits = ((np.arange(2**n, dtype=np.int64)[:, None] >> np.arange(n)) & 1).astype(np.int8)
    probs = np.prod(np.where(bits == 1, p, 1.0 - p), axis=1)
    returns = bits @ pay
    return np.array([float(probs[returns > t].sum()) for t in thresholds])


# --- The search -------------------------------------------------------------


def _prune(n_legs: np.ndarray, key: np.ndarray, rank_on: np.ndarray,
           buckets: int, keep: int, span_lo: float | None = None,
           span_hi: float | None = None) -> np.ndarray:
    """Indices surviving "best `keep` by `rank_on` within each (n_legs, bucket)".

    ``key`` is the continuous axis being bucketed and ``rank_on`` the quantity
    minimised inside a bucket. Returned in no particular order.

    The bucket range is measured rather than fixed, and that is load-bearing.
    ``W`` is ``sum mu/v``, which is bounded only by the leg count in theory but in
    practice lands in a narrow band; bucketing against the theoretical range left
    ~40 of 1024 buckets occupied and threw away 96% of the resolution.

    ``span_lo``/``span_hi`` override that measurement, and `_search` must pass them
    whenever it prunes in chunks. Measuring per chunk would give each chunk its own
    bucket boundaries, so the same state could land in different cells depending on
    which block it arrived in -- and then the two-stage prune stops agreeing with
    the one-stage one. The caller knows the exact range of the full expansion
    before splitting it, so it passes that in and every chunk buckets identically.
    """
    lo = float(key.min()) if span_lo is None else span_lo
    hi = float(key.max()) if span_hi is None else span_hi
    span = max(hi - lo, 1e-12)
    cell = np.clip(((key - lo) / span * buckets).astype(np.int64), 0, buckets - 1)
    cell = n_legs * buckets + cell
    order = np.lexsort((rank_on, cell))
    c = cell[order]
    # Position within each run of equal `cell`, computed from run starts.
    starts = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
    run = np.zeros(len(c), dtype=np.int64)
    run[starts[1:]] = 1
    run = np.cumsum(run)
    within = np.arange(len(c)) - starts[run]
    return order[within < keep]


def _states_to_picks(paths: list[np.ndarray]) -> np.ndarray:
    return np.stack(paths, axis=1) if paths else np.empty((0, 0), dtype=np.int16)


def build_pool(opts: Options, *, min_legs: int | str | None = None,
               leg_var: int | None = None, max_legs: int | None = None,
               buckets: int = BUCKETS, keep_per_cell: int = KEEP_PER_CELL,
               pool_max: int = POOL_MAX,
               exhaustive_max: int = EXHAUSTIVE_MAX) -> tuple[np.ndarray, dict]:
    """Portfolios worth scoring, as an ``(m, n_events)`` array of option indices.

    ``-1`` means the event is not backed. Returns ``(picks, info)``, where
    ``info`` records how the pool was obtained -- exhaustive or searched, and at
    what settings -- because "these are all the portfolios" and "these are the
    portfolios the search reached" are different claims and the workbook says
    which one it is making.

    One pass, because there is one split: state ``(n_legs, bucket(W))`` keeping the
    largest ``C``. This used to be two passes unioned, one ranking for each split.

    The leg floor: ``leg_var`` or ``min_legs``
    -----------------------------------------
    ``leg_var=k`` asks for portfolios backing all but at most ``k`` of the events
    on offer, and is the setting to reach for. ``min_legs`` is the primitive it
    resolves to, and remains for callers that mean an absolute number.

    They are mutually exclusive, and the reason `leg_var` exists is that the
    absolute number is a guess: how many events qualify is not known until the
    form is priced. See `Options.min_legs_for`.

    A high floor costs nothing in the search itself -- `_search` walks the same
    states either way, and measured on R006 the pass takes 5.4 s at every setting
    -- but it cuts what comes out of it sharply, and that is what the rest of the
    pipeline pays for. On R006: 431,244 states at ``min_legs=1``, 184,250 at
    ``leg_var=5``, 30,593 at ``leg_var=0``. `score_pool` convolves every one of
    them, so the floor is the cheapest runtime lever available.
    """
    n_ev = opts.n_events
    if leg_var is not None and min_legs is not None:
        raise ValueError("pass leg_var or min_legs, not both -- they set the same floor")
    if leg_var is not None:
        min_legs = opts.min_legs_for(leg_var)
    elif min_legs is None:
        min_legs = 1
    # `"all"` means one bet from every event -- no skipping. Spelled as a word
    # because the number it stands for is the count of events that survived
    # qualifying, which changes every week: hardcoding last week's 26 would
    # silently mean "at least 26 of this week's 31". `leg_var=0` says the same
    # thing and is the form the notebook uses.
    if isinstance(min_legs, str):
        if min_legs != "all":
            raise ValueError(f"min_legs must be an int or 'all', got {min_legs!r}")
        min_legs = n_ev
    max_legs = n_ev if max_legs is None else min(max_legs, n_ev)
    if n_ev == 0 or min_legs > max_legs:
        return np.empty((0, n_ev), dtype=np.int16), {"mode": "empty", "space": 0.0}

    space = opts.space(min_legs, max_legs)
    terms = [option_terms(opts, j) for j in range(n_ev)]

    if opts.enumeration_cost(min_legs) <= exhaustive_max:
        picks = _enumerate_all(opts, min_legs, max_legs)
        mode = "exhaustive"
    else:
        picks = _search(opts, terms, "w", "c", min_legs, max_legs,
                        buckets, keep_per_cell, sign=-1)
        mode = "searched"

    info = {"mode": mode, "space": space, "n_events": n_ev, "leg_var": leg_var,
            "min_legs": min_legs, "max_legs": max_legs,
            "buckets": buckets, "keep_per_cell": keep_per_cell,
            "found": len(picks)}

    if len(picks) > pool_max:
        picks = _thin(picks, opts, terms, pool_max, buckets, keep_per_cell)
        info["thinned_to"] = len(picks)
    info["pool"] = len(picks)
    return picks, info


def _enumerate_all(opts: Options, min_legs: int, max_legs: int) -> np.ndarray:
    """Every portfolio, by mixed-radix decoding. Only called under `EXHAUSTIVE_MAX`."""
    no_skip = min_legs >= opts.n_events
    radix = opts.sizes if no_skip else opts.sizes + 1
    total = int(np.prod(radix))
    picks = np.empty((total, opts.n_events), dtype=np.int16)
    rem = np.arange(total, dtype=np.int64)
    for j in range(opts.n_events - 1, -1, -1):
        digit = rem % radix[j]
        # With a skip digit, 0 means "not backed" and 1..k mean options 0..k-1.
        picks[:, j] = digit if no_skip else digit - 1
        rem = rem // radix[j]
    n_legs = (picks >= 0).sum(axis=1)
    return picks[(n_legs >= min_legs) & (n_legs <= max_legs)]


def _search(opts: Options, terms: list[dict], key: str, rank: str,
            min_legs: int, max_legs: int, buckets: int, keep: int,
            sign: int, chunk: int = CHUNK) -> np.ndarray:
    """One dynamic-programming pass over the events.

    State is ``(n_legs, bucket(sum key))`` and the value kept is the best
    ``sign * sum rank`` -- state ``(n_legs, W)`` keeping the largest ``C``.
    Partial selections sharing a state are interchangeable under any common
    remainder, which is what makes discarding the rest exact within the bucket.

    ``max_legs`` prunes during the walk rather than at the end: a partial
    selection that already has too many legs cannot be rescued by later events.
    ``min_legs`` cannot prune the same way -- there are always more events to
    come -- so it is applied once at the end.

    Why the expansion is chunked
    ----------------------------
    Each step forms the cross product of every surviving state with every option,
    five arrays wide, and `_prune` then discards ~94% of it. On a 19-event card
    that peak is a few hundred MB and does not matter. On a Thursday-to-Sunday
    card of ~47 events with paired options it is ~1M states x 17 options = 17M
    rows built five times over, 47 times: measured at 3.9 GB of tracked Python
    allocation, and `tracemalloc` undercounts real RSS.

    So the states are expanded a block at a time and pruned per block, then the
    survivors are concatenated and pruned once more. Peak memory becomes
    ``chunk * options`` -- `threshold_probs_batch` solves the same problem the same
    way with the same constant.

    **Two-stage pruning is exact, not an approximation.** `_prune` keeps the best
    ``keep`` per ``(n_legs, bucket)`` cell; a block holds a subset of the rows, so
    anything in the global best-``keep`` for a cell ranks at least as well inside
    its own block and survives the first pass. Nothing the single-pass prune would
    have kept can be lost -- provided every block buckets identically, which is why
    the key range is computed up front and passed to `_prune` rather than measured
    from each block's own data.
    """
    n_ev = opts.n_events
    cur_key = np.zeros(1)
    cur_rank = np.zeros(1)
    cur_legs = np.zeros(1, dtype=np.int64)
    paths: list[np.ndarray] = []

    for j in range(n_ev):
        kj = np.r_[0.0, terms[j][key]]           # index 0 is "skip"
        rj = np.r_[0.0, terms[j][rank]]
        add = np.r_[0, np.ones(len(terms[j][key]), dtype=np.int64)]

        # The exact range of `cur_key + kj` over the whole expansion, known before
        # any of it is built. Every block then buckets against the same scale.
        span_lo = float(cur_key.min() + kj.min())
        span_hi = float(cur_key.max() + kj.max())

        blocks: list[tuple[np.ndarray, ...]] = []
        step = max(int(chunk), 1)
        for lo in range(0, len(cur_key), step):
            sl = slice(lo, min(lo + step, len(cur_key)))
            b_key = (cur_key[sl][:, None] + kj[None, :]).ravel()
            b_rank = (cur_rank[sl][:, None] + rj[None, :]).ravel()
            b_legs = (cur_legs[sl][:, None] + add[None, :]).ravel()
            b_parent = np.repeat(np.arange(sl.start, sl.stop), len(kj))
            b_choice = np.tile(np.arange(-1, len(kj) - 1, dtype=np.int16),
                               sl.stop - sl.start)

            ok = b_legs <= max_legs
            b_key, b_rank, b_legs = b_key[ok], b_rank[ok], b_legs[ok]
            b_parent, b_choice = b_parent[ok], b_choice[ok]
            if not len(b_key):
                continue
            sel = _prune(b_legs, b_key, sign * b_rank, buckets, keep, span_lo, span_hi)
            blocks.append((b_key[sel], b_rank[sel], b_legs[sel],
                           b_parent[sel], b_choice[sel]))

        if not blocks:
            return np.empty((0, n_ev), dtype=np.int16)
        new_key, new_rank, new_legs, parent, choice = (
            np.concatenate(x) for x in zip(*blocks))

        sel = _prune(new_legs, new_key, sign * new_rank, buckets, keep,
                     span_lo, span_hi)
        cur_key, cur_rank, cur_legs = new_key[sel], new_rank[sel], new_legs[sel]
        paths = [pth[parent[sel]] for pth in paths] + [choice[sel]]

    picks = _states_to_picks(paths)
    return picks[(picks >= 0).sum(axis=1) >= min_legs]


def _pareto2_superset(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mask over a **superset** of the frontier maximising ``x``, minimising ``y``.

    Sort by ``x`` descending and keep a row whose ``y`` is no worse than every
    ``y`` seen before it. Everything before it has ``x`` at least as large, so
    anything failing that test is genuinely dominated -- the test is exact in
    that direction, which is the direction that matters here.

    It is deliberately non-strict, so exact ties on both axes are all kept rather
    than all but one. That over-keeps by a handful of rows, which for a stage
    whose job is *not discarding a dominator* is the safe way to be wrong.
    `undominated_portfolios` is the exact test, and it runs afterwards.

    ``O(m log m)``, so it is affordable on the full space where the general
    dominance loop is not.
    """
    order = np.lexsort((y, -x))
    ys = y[order]
    best_before = np.minimum.accumulate(np.r_[np.inf, ys])[:-1]
    mask = np.zeros(len(x), dtype=bool)
    mask[order[ys <= best_before]] = True
    return mask


def _sum_terms(picks: np.ndarray, opts: Options, terms: list[dict]) -> dict[str, np.ndarray]:
    """The additive sums `leg_terms` defines, accumulated over each portfolio's legs."""
    out = {k: np.zeros(len(picks)) for k in terms[0]} if terms else {}
    for j in range(opts.n_events):
        taken = picks[:, j] >= 0
        idx = picks[taken, j]
        for k in out:
            out[k][taken] += terms[j][k][idx]
    return out


def _thin(picks: np.ndarray, opts: Options, terms: list[dict], pool_max: int,
          buckets: int, keep: int) -> np.ndarray:
    """Cut an oversized pool back without changing what is on the frontier.

    Two stages, and the first is the one that matters. Thinning by bucket alone
    can discard the portfolio that *dominates* another, which then reappears as
    undominated -- a pool that flatters itself. Measured: thinning 442,368
    portfolios by bucket gave an 88-portfolio frontier where the true one has 84.
    So the exact ``(expected return, variance)`` frontier is taken first, from
    closed forms over every candidate, and is never dropped.

    The rest of the budget then goes on the bucketed band around it, which is
    what supplies the portfolios that are undominated on *three* criteria while
    sitting just inside the two-criterion frontier. Thinning on the same axes the
    search ranks on keeps the pool's shape across every leg count, rather than
    shaving it off one end the way taking the first `pool_max` rows would.
    """
    n_legs = (picks >= 0).sum(axis=1)
    t = _sum_terms(picks, opts, terms)
    # `C / W` against `C / W**2` -- expected return against variance, both from
    # the closed forms in `leg_terms`. `W` is strictly positive wherever any leg
    # carries edge; a row of pure `e == 1` legs has C = W = 0 and is protected
    # by neither axis, which is correct: it has nothing to be on a frontier for.
    safe_w = np.where(t["w"] > 0, t["w"], np.inf)
    protected = _pareto2_superset(t["c"] / safe_w, t["c"] / safe_w**2)
    must_keep = np.flatnonzero(protected)
    if len(must_keep) >= pool_max:
        return picks[must_keep]

    def band(k: int) -> np.ndarray:
        return np.union1d(must_keep, _prune(n_legs, t["w"], -t["c"], buckets, k))

    wider = None
    for k in range(keep, 0, -1):
        sel = band(k)
        if len(sel) <= pool_max:
            # `pool_max` is a time budget -- thresholds cost ~1 ms a portfolio --
            # so leaving it unspent is lost coverage, not saved effort. Band
            # widths are whole notches, and dropping from one that overshoots to
            # one that fits can leave a lot on the table: on a real form a cap of
            # 100,000 was being filled to 61,096. Top up from the next notch out,
            # which is a superset, until the budget is actually used.
            if wider is not None and len(sel) < pool_max:
                extra = np.setdiff1d(wider, sel, assume_unique=True)
                sel = np.union1d(sel, extra[:pool_max - len(sel)])
            return picks[sel]
        wider = sel
    return picks[np.union1d(must_keep, wider[:max(pool_max - len(must_keep), 0)])]


# --- Scoring the pool -------------------------------------------------------


def score_pool(picks: np.ndarray, opts: Options, *, thresholds=THRESHOLDS,
               grid: int = GRID, splits: tuple[str, ...] = SPLITS) -> pd.DataFrame:
    """One row per (portfolio, split), with every reported quantity on it.

    Percentages throughout: every stake is a fixed fraction of the total, so the
    return distribution scales linearly with it and nothing here moves when the
    total does. Only a cash column would.

    ``capacity`` is the leg set's ``C = sum of squared leg Sharpes`` -- what the
    selection is worth if weighted ideally, and the one number leg count enters the
    model through. ``capacity_used`` is the fraction of it this row's stakes
    actually realise: 1.0 under growth weights unless `MAX_LEG_STAKE` bites, which
    is what makes the cap's cost visible rather than silent.

    ``capacity`` is deliberately **not** a dominance criterion. It is maximised by
    the largest leg set, so adding it as a fourth axis would leave almost every
    large portfolio undominated and blow up the export.
    """
    frames = []
    terms = [option_terms(opts, j) for j in range(opts.n_events)]
    capacity = _sum_terms(picks, opts, terms)["c"]
    for split in splits:
        stakes, p, o, rho = stakes_for(picks, opts, split)
        er, var = metrics(stakes, p, o, rho)
        th = threshold_probs_batch(stakes * o, p, thresholds, grid, rho=rho)
        # (mu/sigma)**2 against the best achievable, sqrt(C). Guarded because a
        # portfolio whose legs all sit at e == 1 has zero of both.
        realised = np.where(var > 0, (er - 1.0) ** 2 / np.where(var > 0, var, 1.0), 0.0)
        used = np.where(capacity > 0, realised / np.where(capacity > 0, capacity, 1.0), np.nan)
        # The typical leg's chance of landing. `stakes_for` returns `p` already
        # gathered per portfolio with 0 in the unused slots, so this is a masked
        # median over a column that is here anyway.
        #
        # A median rather than a mean: one 0.95 leg among eleven longshots moves
        # the mean and does not change what the portfolio is. It is reported
        # rather than scored -- nothing in the dominance comparison reads it.
        leg_p = np.where(p > 0, p, np.nan)
        df = pd.DataFrame({
            "combo": np.arange(len(picks)),
            "split": split,
            # Two different numbers, and they only coincide without pairs. `events`
            # is what `leg_var` governs and what the search counts; `legs` is how
            # many propositions are actually held, which is larger wherever an
            # option is a pair. Reporting one under both names was the bug.
            "events": (picks >= 0).sum(axis=1),
            "legs": (p > 0).sum(axis=1),
            "pct_expected_return": er,
            "variance": var,
            "pct_sd": np.sqrt(var),
            "capacity": capacity,
            "capacity_used": used,
            "max_leg_stake": stakes.max(axis=1),
            "n_eff": _effective_legs(stakes, p, o),
            "median_leg_p": np.nanmedian(leg_p, axis=1),
        })
        for c, t in enumerate(thresholds):
            df[f"p_over_{int(round(t * 100))}"] = th[:, c]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "id", np.arange(1, len(out) + 1))
    return out


def _effective_legs(stakes: np.ndarray, p: np.ndarray, o: np.ndarray) -> np.ndarray:
    """How many legs a portfolio is *really* carrying: inverse Herfindahl of variance.

    ``1 / sum((v_i / V)**2)`` over each leg's share of portfolio variance. Equals
    the leg count exactly when every leg contributes the same variance, and falls
    towards 1 as one leg dominates.

    Worth reporting because leg count on its own overstates the diversification
    growth weights deliver: R006's nineteen-leg book has 12.6 effective legs under
    minimum variance and 5.6 under growth weights. Both are nineteen-leg books, and
    only one of them behaves like it.
    """
    v = (stakes * o) ** 2 * p * (1.0 - p)
    total = v.sum(axis=1, keepdims=True)
    share = np.divide(v, total, out=np.zeros_like(v), where=total > 0)
    hhi = (share**2).sum(axis=1)
    return np.where(hhi > 0, 1.0 / np.where(hhi > 0, hhi, 1.0), 0.0)


# --- Portfolio-level dominance ---------------------------------------------

# The three criteria a portfolio is judged on, and which direction is better.
# Reported alongside them, but deliberately *not* part of the comparison, are the
# other four return thresholds: including them would make almost everything
# undominated, since a portfolio need only win on one of seven axes to survive.
# Passing a different tuple is how to change that without touching this code.
DOMINANCE_CRITERIA: tuple[str, ...] = ("pct_expected_return", "variance", "p_over_100")
DOMINANCE_SENSES: tuple[int, ...] = (+1, -1, +1)


def _pareto_mask(X: np.ndarray) -> np.ndarray:
    """True where a row is undominated. ``X`` is oriented so bigger is better.

    Rows are visited in descending order of the first criterion, so anything that
    could dominate the current row has already been seen and sits in `front`. The
    cost is ``O(m * |front|)`` rather than ``O(m**2)``, and the frontier of a set
    scattered in three dimensions grows like ``(ln m)**2 / 2`` -- a few hundred
    rows out of tens of thousands -- so that difference is the whole ballgame.

    Exact ties on every criterion dominate neither and both survive, matching
    `staking.undominated`.
    """
    m = len(X)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(-X[:, 0], kind="stable")
    front = np.empty((0, X.shape[1]))
    kept: list[int] = []
    for i in order:
        x = X[i]
        if len(front) and np.any(np.all(front >= x, axis=1) & np.any(front > x, axis=1)):
            continue
        # Not dominated -- but with ties on the first criterion it may dominate
        # something already on the frontier, so the frontier is pruned too.
        if len(front):
            dead = np.all(x >= front, axis=1) & np.any(x > front, axis=1)
            if dead.any():
                front = front[~dead]
                kept = [j for j, d in zip(kept, dead) if not d]
        front = np.vstack([front, x])
        kept.append(int(i))
    mask = np.zeros(m, dtype=bool)
    mask[kept] = True
    return mask


def undominated_portfolios(scored: pd.DataFrame,
                           criteria: tuple[str, ...] = DOMINANCE_CRITERIA,
                           senses: tuple[int, ...] = DOMINANCE_SENSES) -> np.ndarray:
    """Boolean mask over `scored`: True where nothing beats this row on all three.

    Run across **both splits at once**, deliberately. The choice of split is part
    of the portfolio, not a presentation of it, so a ``1/E`` allocation that is
    beaten on return, spread and probability of profit by some other
    combination's minimum-variance allocation is beaten, and saying so is the
    honest answer.
    """
    missing = [c for c in criteria if c not in scored.columns]
    if missing:
        raise ValueError(f"unknown dominance criteria {missing}; have {list(scored.columns)}")
    X = np.column_stack([scored[c].to_numpy(float) * s for c, s in zip(criteria, senses)])
    return _pareto_mask(X)


def filter_portfolios(scored: pd.DataFrame, *, undominated_only: bool = True,
                      split: str | None = None, sort_by: str = "p_over_100",
                      ascending: bool = False, **bounds) -> pd.DataFrame:
    """Explore the scored table by the same criteria the workbook's Query sheet uses.

    Bounds are given as ``min_<column>`` or ``max_<column>``, so::

        filter_portfolios(scored, min_pct_expected_return=1.05,
                          max_variance=0.02, min_p_over_100=0.75)

    reads as the sentence it is. Omitting one drops that constraint, exactly as
    leaving the cell blank does on the sheet.

    This exists so the notebook can stay a driver and so there is one place the
    predicate is written for pandas. The Excel `Match` column is the same
    predicate rendered as a formula, and `tests/test_portfolio_excel.py` pins the
    two to agree row for row.
    """
    out = scored
    if undominated_only and "undominated" in out.columns:
        out = out[out["undominated"]]
    if split is not None:
        out = out[out["split"] == split]
    for name, value in bounds.items():
        if value is None:
            continue
        side, _, col = name.partition("_")
        if side not in ("min", "max") or col not in scored.columns:
            raise ValueError(
                f"{name!r} is not a bound; use min_<column> or max_<column> with one of "
                f"{[c for c in scored.columns if c not in ('picks', 'split')]}"
            )
        out = out[out[col] >= value] if side == "min" else out[out[col] <= value]
    if sort_by:
        out = out.sort_values(sort_by, ascending=ascending)
    return out.reset_index(drop=True)


def thin_export(scored: pd.DataFrame, export_max: int = EXPORT_MAX,
                buckets: int = BUCKETS) -> np.ndarray:
    """Boolean mask over an undominated set, capped at `export_max`.

    The frontier is usually small -- 2,360 of 100,000 scored on a normal slate --
    and this does nothing. It exists for the slate shape that breaks that: on a
    37-event card at a tight `leg_var`, portfolios differ by which one or two
    events they drop, return and variance correlate at 0.83, almost nothing
    dominates anything, and 12% of the pool clears the frontier. That run wrote a
    35 MB payload and the Edge Book would not build from it.

    Thinning, not truncating. Taking the first `export_max` rows would lop off
    whichever end the sort favours -- every high-variance book, or every long one.
    Instead: **protect the extremes**, then keep the best few per bucket of the
    rest, so what survives spans the same range at lower density. The same shape
    `_thin` uses on the pool, one level down.
    """
    n = len(scored)
    if n <= export_max:
        return np.ones(n, dtype=bool)

    keep = np.zeros(n, dtype=bool)
    # The corners of the space, whatever else goes: the most capacity, the most
    # return, the least spread, the best chance of profit.
    for col, sense in (("capacity", +1), ("pct_expected_return", +1),
                       ("variance", -1), ("p_over_100", +1)):
        if col in scored:
            v = scored[col].to_numpy(float)
            keep[int(np.argmax(v) if sense > 0 else np.argmin(v))] = True

    legs = scored["legs"].to_numpy(np.int64) if "legs" in scored else np.zeros(n, np.int64)
    cap = (scored["capacity"].to_numpy(float) if "capacity" in scored
           else scored["pct_expected_return"].to_numpy(float))

    # Buckets sized to the budget, not to `BUCKETS`. The search's 4096 is
    # resolution for separating states; here it would put fewer than one row in
    # each cell, so "keep the best per cell" would keep everything and the cap
    # would never bind. One bucket per slot of the budget, split across the leg
    # counts present, puts roughly one row in each.
    n_legs = max(len(np.unique(legs)), 1)
    cells = max(1, export_max // n_legs)
    sel = _prune(legs, cap, -cap, cells, 1)
    keep[sel] = True

    # A bucketed sweep lands under the budget whenever cells collide, and leaving
    # the remainder unspent is lost coverage rather than saved effort -- the same
    # argument `_thin` makes about its own band. Top up with the highest-capacity
    # rows not already taken.
    if keep.sum() < export_max:
        spare = np.flatnonzero(~keep)
        extra = spare[np.argsort(-cap[spare], kind="stable")][:export_max - int(keep.sum())]
        keep[extra] = True
    return keep


# --- Expanding a portfolio back into bets ----------------------------------


def legs(picks: np.ndarray, opts: Options, combo: int, split: str,
         total: float = 1.0) -> pd.DataFrame:
    """The actual bets for one portfolio: proposition, price, book, stake.

    This is the step that turns a row of a results table into something you can
    go and place, so it carries `book` -- an edge you cannot find again is not
    actionable.

    **One row per proposition, not per event.** An option holding a pair expands
    to two rows from the same match, each with its own price, book and stake --
    which is what you go and place. `Options.members` says which propositions an
    option covers.
    """
    row = picks[[combo]]
    stakes, p, o, _ = stakes_for(row, opts, split)
    out = []
    for j in range(opts.n_events):
        k = int(row[0, j])
        if k < 0:
            continue
        for slot, prop in enumerate(opts.members[j][k]):
            col = 2 * j + slot
            if p[0, col] <= 0:
                continue
            src = opts.props[(opts.props["event"] == j)
                             & (opts.props["option"] == prop)].iloc[0]
            out.append({
                "event": opts.codes[j],
                "fixture": src.get("fixture", f"{src.get('home_team', '')} vs {src.get('away_team', '')}"),
                "label": src["label"],
                "p": float(p[0, col]),
                "o": float(o[0, col]),
                "book": src.get("book"),
                "e": float(p[0, col] * o[0, col]),
                "stake": float(stakes[0, col]) * float(total),
            })
    return pd.DataFrame(out)


def picks_label(picks_row: np.ndarray, opts: Options) -> str:
    """A portfolio as one compact string: ``SP1-01#2 E0-03#0 ...``.

    Enough to identify the selection at a glance on a crowded sheet, and enough
    to reconstruct it, without spending twenty-six columns on it.

    **One token per proposition, not per option**, so a pair emits both of its
    legs. That keeps every token a reference into `props` -- which is what the
    payload's proposition ids are built from -- and keeps this list the same
    length, and the same order, as the stakes `_stake_fractions` emits beside it.
    """
    return " ".join(
        f"{opts.codes[j]}#{prop}"
        for j, k in enumerate(picks_row) if k >= 0
        for prop in opts.members[j][int(k)])


# --- The whole thing, in order ---------------------------------------------


def search(filled: pd.DataFrame, *, min_legs: int | str | None = None,
           leg_var: int | None = None, max_legs: int | None = None,
           thresholds=THRESHOLDS, grid: int = GRID, **pool_kwargs) -> dict:
    """Filled odds form -> everything the workbook and the notebook need.

    ``leg_var=k`` allows portfolios that leave out up to ``k`` of the qualifying
    events, and is the setting the notebook drives. ``leg_var=0`` requires one bet
    from every one of them -- no skipping -- which is the only setting whose answer
    is provable: the space collapses to ``prod(k_i)``, which enumerates, so every
    portfolio is scored rather than searched.

    The floor has to be relative because how many events qualify is not knowable
    until the form is priced -- see `Options.min_legs_for`. ``min_legs`` remains
    for a caller that genuinely means an absolute number, and the two are mutually
    exclusive.

    Kept as one function so the notebook stays a driver and the order of the
    steps lives in the package, where a test can reach it.
    """
    qualified = qualify(filled)
    opts = event_options(qualified)
    picks, info = build_pool(opts, min_legs=min_legs, leg_var=leg_var,
                             max_legs=max_legs, **pool_kwargs)

    if len(picks) == 0:
        return {"qualified": qualified, "options": opts, "picks": picks,
                "info": info, "scored": pd.DataFrame(), "undominated": np.zeros(0, bool)}

    scored = score_pool(picks, opts, thresholds=thresholds, grid=grid)
    scored["undominated"] = undominated_portfolios(scored)
    scored["picks"] = [picks_label(picks[c], opts) for c in scored["combo"]]
    scored = scored.sort_values(
        ["undominated", "p_over_100"], ascending=[False, False]
    ).reset_index(drop=True)
    return {"qualified": qualified, "options": opts, "picks": picks,
            "info": info, "scored": scored,
            "undominated": scored["undominated"].to_numpy()}


__all__ = [
    "SPLIT_EVEN", "SPLIT_MINVAR", "SPLIT_GROWTH", "SPLITS", "LEGACY_SPLITS",
    "EXHAUSTIVE_MAX", "POOL_MAX", "BUCKETS", "KEEP_PER_CELL", "GRID",
    "DOMINANCE_CRITERIA", "DOMINANCE_SENSES",
    "Options", "qualify", "event_options", "leg_terms",
    "stakes_for", "cap_stakes", "metrics",
    "threshold_probs_batch", "threshold_probs_exact",
    "return_pmf",
    "build_pool", "score_pool", "undominated_portfolios", "filter_portfolios",
    "legs", "picks_label",
    "search",
]
