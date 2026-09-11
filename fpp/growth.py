"""What fraction of the bankroll one portfolio should get, round after round.

`06_Split` answers "what does one round look like": the ``(expected return,
variance, P(profit))`` frontier describes a single settlement. Reinvest, and a
third question appears that neither `staking` nor `portfolio` asks --
`staking` splits stake *within* a portfolio, `portfolio` chooses which
propositions go together, and this decides how much of the pot the whole thing
gets.

**One number comes out of it**: ``f_suggested``, the fraction of the pot this
portfolio should get. It is built in two steps, both reported alongside it so the
number can be argued with rather than only obeyed:

* ``f_drawdown`` -- the largest fraction that still keeps a serious drawdown
  unlikely, at the tolerance in `GROWTH_DRAWDOWN_D` / `GROWTH_DRAWDOWN_P` over
  `GROWTH_ROUNDS`. This is what the module used to publish as ``f_protective``.
* then ``edge_factor`` and ``var_factor``, the two model-risk dials in
  `suggested_fraction`. Both default to 1, so with the dials off the answer is
  exactly ``f_drawdown``.

There used to be a second published number, ``f_star``, the growth optimum. It is
still computed -- `protective_fraction` searches below it -- and no longer
reported, because it stopped varying: it sat at `GROWTH_F_MAX` in 66.4% of
exported portfolios and 99% of those with 16 or more legs. The unconstrained
optimum wants leverage it cannot have, so the column was reading back its own
ceiling. Offering a constant as one of two choices is worse than offering one
answer.

What the risk tolerance actually buys is worth stating, because it is invisible
in the output otherwise: measured across every portfolio in the leg-count study,
``f_drawdown`` is ``0.111 * mu / sigma**2`` with r = 0.998 and a worst residual
of 6.2%. The 4,000-path search is computing **one-ninth Kelly**, and the three
constants that set that fraction are a taste, not a derivation.

Why not the closed form
-----------------------
``f* = mu/sigma**2`` and ``g ~= mu - sigma**2/2`` assume something close to a
continuous, roughly symmetric return. A portfolio here is a sum of eight to
sixteen discrete win/lose outcomes -- lumpy, and lumpiest exactly in the tails
that decide drawdown. This codebase has already caught that assumption failing:
portfolio 2294's simulated ``P(>100%)`` was 75.64% against ~66% from a normal fit
to the same mean and SD. So every number here is read off the portfolio's actual
distribution. The closed form appears once, in a test, as a sanity bound on a
synthetic well-behaved case -- never in output.

Where the distribution comes from
---------------------------------
`portfolio.return_pmf` -- the same grid convolution that produces the ``P(>t)``
columns, which built the full distribution all along and discarded it once the
five thresholds were read off. Taking it from there keeps **one** engine, so the
growth block and the threshold columns are two readings of the same numbers.
Enumerating ``2**n`` separately would be a second engine answering the same
question to a different precision -- the thing `staking.threshold_probs` warns
against -- and at real leg counts it is 5x the work (55.7M outcome rows against
11.2M grid cells on the 2,740 portfolios of 2026-08-22).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    GROWTH_BAND_POINTS,
    GROWTH_BAND_QUANTILES,
    GROWTH_DRAWDOWN_D,
    GROWTH_DRAWDOWN_P,
    GROWTH_F_COARSE,
    GROWTH_F_FINE,
    GROWTH_F_MAX,
    GROWTH_HIST_BINS,
    GROWTH_PATHS,
    GROWTH_ROUNDS,
    PESSIMISM_B,
    SLATE_TAU,
)
from .portfolio import GRID, return_pmf


def suggested_fraction(f_drawdown: float, pay: np.ndarray, p: np.ndarray, *,
                       bias: float = PESSIMISM_B, tau: float = SLATE_TAU
                       ) -> tuple[float, dict]:
    """The one stake. ``(f, terms)``, where `terms` records what each dial cost.

    `f_drawdown` prices the week going badly: it is the largest fraction whose
    drawdown risk clears `GROWTH_DRAWDOWN_P`, and it is the number this project
    already published as ``f_protective``. What it does *not* price is the model
    being wrong, which is a different risk and the one leg count cannot diversify
    away. Two dials do that, and they are the mean and the spread of a single
    quantity -- the error shared by every leg on the card::

        f = f_drawdown * (mu_hat / mu) * sigma**2 / (sigma**2 + tau**2 K**2)

    ``bias`` shifts the centre: recompute expected return with every leg's ``p``
    lowered by ``b``, and scale ``f`` by the ratio, because ``f`` is linear in
    ``mu``. ``tau`` widens the spread: a shared shift of standard deviation
    ``tau`` adds ``tau**2 K**2`` to the round's variance, where
    ``K = sum(s_i o_i)`` is how far the portfolio's expected return moves per
    point of shift in every leg at once.

    Neither is a measurement yet. **With both at zero the result is exactly
    `f_drawdown`**, which is the regression test and the reason they can ship
    switched off: this is a re-description of today's behaviour with the risk
    appetite pulled out where it can be seen, not a new recommendation.

    Independent per-leg error is deliberately absent, because it costs nothing.
    If each leg's ``p`` is wrong by an independent draw, the extra uncertainty is
    exactly offset by less coin-flip variance in the outcome -- simulated books
    from 6 to 96 legs are indistinguishable from a perfect model. Only the shared
    part survives diversification, and only the shared part is priced here.
    """
    pay = np.asarray(pay, dtype=float)
    p = np.asarray(p, dtype=float)
    live = pay > 0
    mu = float((pay * p)[live].sum()) - 1.0
    var = float(((pay**2) * p * (1.0 - p))[live].sum())
    k = float(pay[live].sum())

    terms = {
        "f_drawdown": f_drawdown,
        "edge_factor": 0.0,
        "var_factor": 1.0,
        "k_leverage": k,
        "breakeven_shift": 0.0 if k <= 0 else -mu / k,
    }
    # No edge to stake on, before any haircut is applied. Stated as its own case
    # rather than falling out of a ratio, so that "this book is not worth backing"
    # never reads as "the dials shrank it to nothing".
    if mu <= 0 or not np.isfinite(f_drawdown) or f_drawdown <= 0:
        return 0.0, terms

    mu_hat = float((pay * np.clip(p - bias, 0.0, 1.0))[live].sum()) - 1.0
    terms["edge_factor"] = max(mu_hat, 0.0) / mu
    terms["var_factor"] = 1.0 if var <= 0 else var / (var + (tau * k) ** 2)

    return f_drawdown * terms["edge_factor"] * terms["var_factor"], terms


def ruin_prob(p: np.ndarray) -> np.ndarray:
    """``P(return == 0)`` per portfolio -- exact, closed form, no distribution.

    Every leg's payout is non-negative, so the return is zero if and only if no
    leg lands: ``prod(1 - p_j)`` over the live legs. Skipped events carry ``p =
    0`` and contribute a factor of 1, so they fall out on their own.

    Deliberately *not* read off the grid. It needs no distribution, it is exact
    where the grid is quantised, and it is the one number here that is worth
    being able to state without qualification -- on 2026-08-22 it ran to 0.6345,
    a portfolio with a 63% chance of returning nothing, and nothing on the Edge
    Book said so.
    """
    p = np.asarray(p, dtype=float)
    return np.prod(np.where(p > 0, 1.0 - p, 1.0), axis=1)


def growth_rate(f: np.ndarray, dist: np.ndarray, values: np.ndarray) -> np.ndarray:
    """``g(f) = E[ln((1 - f) + f * R)]`` for every ``f``, one portfolio.

    ``dist`` and ``values`` are one row of `portfolio.return_pmf` and its value
    axis. Returns shape ``(len(f),)``.

    Where any reachable return drives the bankroll multiplier to zero or below,
    the result is ``-inf`` and is returned as such. A clipped logarithm would
    hand back a large finite number and make ruin look merely unattractive
    instead of terminal: with ``p0 > 0``, ``f >= 1`` is ruin eventually however
    good the edge looks, and that has to stay visible.
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    w = (1.0 - f)[:, None] + f[:, None] * values[None, :]
    live = dist > 0
    bad = (w <= 0) & live[None, :]
    safe = np.where(w > 0, w, 1.0)
    g = (dist[None, :] * np.log(safe)).sum(axis=1)
    return np.where(bad.any(axis=1), -np.inf, g)


def _f_grid(f_max: float = GROWTH_F_MAX, step: float = GROWTH_F_COARSE) -> np.ndarray:
    return np.round(np.arange(step, f_max + 1e-12, step), 10)


def optimal_fraction(dist: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """``(f_star, g_star)`` by a coarse pass then a local refine.

    Two passes rather than one fine grid because the refine is only needed near
    the winner, and it *is* needed: measured ``f*`` clusters at 0.96-0.98, right
    against the edge where ``g`` falls off a cliff to ``-inf``. A coarse-only
    answer would round that corner badly.
    """
    coarse = _f_grid()
    gc = growth_rate(coarse, dist, values)
    if not np.isfinite(gc).any():
        return float("nan"), float("-inf")
    i = int(np.nanargmax(np.where(np.isfinite(gc), gc, -np.inf)))
    lo = max(GROWTH_F_FINE, coarse[i] - GROWTH_F_COARSE)
    hi = min(GROWTH_F_MAX, coarse[i] + GROWTH_F_COARSE)
    fine = np.round(np.arange(lo, hi + 1e-12, GROWTH_F_FINE), 10)
    gf = growth_rate(fine, dist, values)
    j = int(np.nanargmax(np.where(np.isfinite(gf), gf, -np.inf)))
    return float(fine[j]), float(gf[j])


def sample_rounds(dist: np.ndarray, values: np.ndarray, *, paths: int = GROWTH_PATHS,
                  rounds: int = GROWTH_ROUNDS, rng_seed: int = 0) -> np.ndarray:
    """``(paths, rounds)`` of returns drawn from the portfolio's own distribution.

    Separated from `drawdown_prob` because **the draw does not depend on ``f``**.
    Only the bankroll multiplier does. Sampling once per portfolio and reusing the
    same rounds across every ``f`` the search visits is the difference between one
    sample and seven, and it makes those seven answers common-random-number
    comparisons of one set of futures rather than seven independent ones -- so the
    drawdown curve is monotone in ``f`` in practice and not just in expectation,
    which is what `protective_fraction` binary-searches on.

    Sampled by inverse-CDF rather than ``rng.choice(p=...)``: same distribution,
    without rebuilding an alias table over ~2,500 occupied cells per call.
    """
    cdf = np.cumsum(dist / dist.sum())
    rng = np.random.default_rng(rng_seed)
    idx = np.searchsorted(cdf, rng.random((paths, rounds)), side="left")
    return values[np.clip(idx, 0, len(values) - 1)]


def drawdown_prob(f: float, returns: np.ndarray, *,
                  drawdown: float = GROWTH_DRAWDOWN_D) -> float:
    """``P(ever down `drawdown` from a running peak)`` over the sampled rounds.

    The one sampled quantity in this module, and unavoidably so: "ever below a
    prior peak" is path-dependent and has no closed form. What is sampled is the
    *sequence* -- each round comes from the exact pmf, never from a fresh
    approximation of the legs -- so this inherits no normal-approximation error,
    only path sampling noise.
    """
    if not np.isfinite(f) or f <= 0:
        return 0.0
    mult = (1.0 - f) + f * returns
    if np.any(mult <= 0):
        return 1.0                      # a reachable wipe-out; every path ruins
    curve = np.cumsum(np.log(mult), axis=1)         # log bankroll, starting at 0
    peak = np.maximum.accumulate(
        np.concatenate([np.zeros((len(curve), 1)), curve], axis=1), axis=1)
    below = curve - peak[:, :-1] <= np.log(1.0 - drawdown)
    return float(below.any(axis=1).mean())


def protective_fraction(returns: np.ndarray, f_star: float, *,
                        drawdown: float = GROWTH_DRAWDOWN_D,
                        max_prob: float = GROWTH_DRAWDOWN_P) -> tuple[float, float]:
    """Largest ``f <= f_star`` whose drawdown risk stays under `max_prob`.

    A **constrained optimum**, not a cautious guess. ``g`` rises all the way to
    ``f_star`` and drawdown risk rises with ``f``, so the best admissible stake is
    the highest one that still clears the constraint -- never smaller than the
    constraint requires, and equal to ``f_star`` when the constraint is slack, in
    which case there is no trade-off to make.

    Monotonicity is what makes it cheap: binary search over the grid costs ~6
    drawdown estimates instead of ~50 down the whole thing, and the drawdown
    estimate is the expensive piece.
    """
    if not np.isfinite(f_star):
        return float("nan"), float("nan")
    at_star = drawdown_prob(f_star, returns, drawdown=drawdown)
    if at_star < max_prob:
        return f_star, at_star           # constraint slack: no trade-off to make

    grid = np.round(np.arange(GROWTH_F_FINE, f_star + 1e-12, GROWTH_F_FINE), 10)
    lo, hi, best, best_p = 0, len(grid) - 1, float("nan"), float("nan")
    while lo <= hi:
        mid = (lo + hi) // 2
        pr = drawdown_prob(float(grid[mid]), returns, drawdown=drawdown)
        if pr < max_prob:
            best, best_p, lo = float(grid[mid]), pr, mid + 1
        else:
            hi = mid - 1
    return best, best_p


# --- Projection block ------------------------------------------------------
#
# What `growth_metrics(projection=True)` adds, for the Edge Book's Projection
# page. Every function below reads the *same* `dist`/`values` row and the *same*
# `sample_rounds` draw the growth block already built, so the fan, `g_star` and
# `drawdown_p` end up three readings of one set of numbers rather than three
# answers to the same question. That is the whole reason this lives here and not
# in the browser.


def sweep_prob(p: np.ndarray) -> np.ndarray:
    """``P(every leg lands)`` -- the mirror of `ruin_prob`.

    The probability attached to the largest return a portfolio can pay, and
    exact where the grid is quantised. Skipped events carry ``p = 0`` and
    contribute a factor of 1, exactly as they do in `ruin_prob`, rather than
    annihilating the product.
    """
    p = np.asarray(p, dtype=float)
    return np.prod(np.where(p > 0, p, 1.0), axis=1)


def f_curve(f_max: float = GROWTH_F_MAX, step: float = GROWTH_F_FINE) -> np.ndarray:
    """The stake fractions ``g_curve`` is evaluated at.

    `GROWTH_F_FINE` because that is the resolution `optimal_fraction` already
    refines to. A page reading a growth rate off this curve then reads it at the
    precision ``f_star`` was *chosen* at; interpolating across a coarser grid
    would let the curve disagree with the number printed beside it.
    """
    return _f_grid(f_max, step)


def band_rounds(rounds: int = GROWTH_ROUNDS,
                points: int = GROWTH_BAND_POINTS) -> np.ndarray:
    """Log-spaced rounds to report bands at -- dense early, sparse late.

    Where the fan bends is the first handful of rounds; past that log wealth is
    close to straight in ``n``. See `GROWTH_BAND_POINTS` for why this is a
    transport decision rather than an approximation.
    """
    if rounds <= points:
        return np.arange(1, rounds + 1)
    return np.unique(np.round(np.geomspace(1, rounds, points)).astype(int))


def wealth_bands(f: float, returns: np.ndarray, *, at: np.ndarray,
                 quantiles: tuple[int, ...] = GROWTH_BAND_QUANTILES
                 ) -> np.ndarray | None:
    """Percentiles of cumulative **log** wealth at each round in `at`.

    ``(len(quantiles), len(at))``, or ``None`` where `f` cannot be plotted: a
    non-finite fraction, or one that makes some reachable round wipe the bankroll
    out. `drawdown_prob` calls that second case every path ruining, and this
    agrees with it rather than drawing a line into ``-inf``.

    Log wealth rather than the multiple, because the multiple is the thing that
    gets unwieldy: ``g_star`` near 0.15 over 100 rounds is about 3e6x, which
    stores badly at fixed precision and reads as noise on the wire. The client
    exponentiates against its own pot, which is the only place the pot is known.
    """
    if not np.isfinite(f) or f <= 0:
        return None
    mult = (1.0 - f) + f * returns
    if np.any(mult <= 0):
        return None
    curve = np.cumsum(np.log(mult), axis=1)
    return np.percentile(curve[:, np.asarray(at) - 1], quantiles, axis=0)


def outcome_histogram(dist: np.ndarray, values: np.ndarray,
                      bins: int = GROWTH_HIST_BINS) -> tuple[np.ndarray, float]:
    """One round's return distribution, rebinned for drawing. ``(mass, step)``.

    Every grid cell lands in exactly one bin, so the histogram carries the same
    total mass the pmf does and bin ``k`` spans returns ``[k*step, (k+1)*step)``.
    Rebinned rather than sent whole because `portfolio.GRID` is 16,384 cells --
    resolution for reading thresholds off, not for drawing 40 bars.
    """
    hi = float(values[-1])
    if hi <= 0:
        return np.zeros(bins), 0.0
    mass, _ = np.histogram(values, bins=np.linspace(0.0, hi, bins + 1), weights=dist)
    return mass, hi / bins


def growth_metrics(pay: np.ndarray, p: np.ndarray, *, grid: int = GRID,
                   drawdown: float = GROWTH_DRAWDOWN_D,
                   max_prob: float = GROWTH_DRAWDOWN_P,
                   projection: bool = False, **kw) -> pd.DataFrame:
    """Growth block for a batch of portfolios. One row in, one row out.

    ``pay`` is each leg's payout as a fraction of the total stake (``s_j * o_j``)
    and ``p`` the matching hit probability, both zero on a skipped event -- the
    same pair `portfolio.stakes_for` already returns, so nothing here needs the
    ragged per-portfolio leg lists.

    Intended for the undominated set only, matching the rule that only
    undominated portfolios are exported at all. Dominance is settled before this
    runs and is not affected by it: the block is informational, not a fourth
    axis.

    ``projection=True`` adds the columns the Edge Book's Projection page draws --
    the ``g(f)`` curve, the percentile fan at both stakes, the outcome histogram
    and the best case. They are computed *here*, inside the same loop, rather
    than by a second function that would rebuild `dist` and redraw
    `sample_rounds`: sharing one draw is what makes the fan and ``drawdown_p``
    descriptions of the same 4,000 futures, and `sample_rounds` is written for
    exactly that reuse.
    """
    pay = np.asarray(pay, dtype=float)
    p = np.asarray(p, dtype=float)
    # `close_tail` because a pmf that integrates to less than one would
    # under-weight every g(f) by the shortfall -- see `portfolio.return_pmf`.
    dist, step = return_pmf(pay, p, grid, close_tail=True)
    idx = np.arange(grid)
    p0 = ruin_prob(p)
    p_max = sweep_prob(p)
    fs = f_curve() if projection else None
    at = band_rounds() if projection else None

    rows = []
    for i in range(len(pay)):
        values = idx * step[i]
        # `f_star` is still computed, and still not reported. `protective_fraction`
        # binary-searches over `f <= f_star`, so the growth optimum is needed as the
        # upper bound of that search -- but as a *number* it stopped varying: it sat
        # at `GROWTH_F_MAX` in 66.4% of exported portfolios and 99% of those with 16+
        # legs, because the unconstrained optimum wants leverage it cannot have. A
        # column that reads back the cap is not a recommendation.
        f_star, _ = optimal_fraction(dist[i], values)
        returns = sample_rounds(dist[i], values, **kw)
        f_prot, dd = protective_fraction(returns, f_star,
                                         drawdown=drawdown, max_prob=max_prob)
        f_sug, terms = suggested_fraction(f_prot, pay[i], p[i])
        # Staking nothing is a real answer, and its growth rate is exactly zero --
        # the bankroll does not move. Reporting NaN there would put a hole in the
        # payload for the one case the reader most needs stated plainly.
        g_sug = float(growth_rate(np.array([f_sug]), dist[i], values)[0]) \
            if f_sug > 0 else 0.0
        rec = {
            "p0": float(p0[i]),
            "f_suggested": f_sug,
            "g_suggested": g_sug,
            "f_drawdown": f_prot,
            "edge_factor": terms["edge_factor"],
            "var_factor": terms["var_factor"],
            "k_leverage": terms["k_leverage"],
            "breakeven_shift": terms["breakeven_shift"],
            "drawdown_d": drawdown,
            "drawdown_p": dd,
        }
        if projection:
            mass, hstep = outcome_histogram(dist[i], values)
            rec.update({
                "f_grid": fs,
                # -inf where ruin is reachable, and left as -inf: the caller
                # serialising it is the right place to decide how to say "this
                # stake is terminal", and any finite stand-in chosen here would
                # be a number someone plots.
                "g_curve": growth_rate(fs, dist[i], values),
                "band_rounds": at,
                # One fan, at the one stake. The page used to draw two and invite a
                # comparison between a recommendation and a boundary; the curve above
                # already shows what higher stakes buy, and what they cost.
                "bands_suggested": wealth_bands(f_sug, returns, at=at),
                "hist": mass,
                "hist_step": hstep,
                "max_return": float(values[-1]),
                "p_max": float(p_max[i]),
            })
        rows.append(rec)
    return pd.DataFrame(rows)


__all__ = [
    "ruin_prob", "sweep_prob", "growth_rate", "optimal_fraction",
    "sample_rounds", "drawdown_prob", "protective_fraction",
    "suggested_fraction", "growth_metrics",
    "f_curve", "band_rounds", "wealth_bands", "outcome_histogram",
]
