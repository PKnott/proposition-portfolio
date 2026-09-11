"""Guards for the growth-optimal staking metrics.

Two things need pinning here, and they are different in kind.

The **arithmetic** -- concavity, the closed-form sanity bound, the ruin guard --
is checked on synthetic distributions where the right answer is known
independently, so an error in the module cannot hide behind real data being
complicated.

The **engine choice** is checked against exact enumeration. `growth` reads its
distribution from `portfolio.return_pmf`, the quantised grid convolution, rather
than enumerating ``2**n`` outcomes. That is only allowed if the two agree, which
is the same bargain `test_grid_thresholds_match_exact_enumeration` strikes for
the threshold columns.
"""

from __future__ import annotations

import numpy as np
import pytest

import fpp.growth as gr
from fpp.config import GROWTH_F_FINE
from fpp.portfolio import return_pmf
from fpp.staking import outcome_paths


def _normalish(mean: float, sd: float, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """A well-behaved (dist, values) pair -- discretised normal, truncated positive."""
    values = np.linspace(max(1e-6, mean - 6 * sd), mean + 6 * sd, n)
    dist = np.exp(-0.5 * ((values - mean) / sd) ** 2)
    return dist / dist.sum(), values


def _legs(p: np.ndarray, o: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(pay, p)`` for one portfolio, staked 1/E and normalised to a unit bankroll."""
    s = 1.0 / (p * o)
    s = s / s.sum()
    return (s * o)[None, :], p[None, :]


# --- arithmetic -------------------------------------------------------------


def test_growth_rate_is_concave_in_f():
    """g(f) is concave, so the optimum the search finds is the only one."""
    dist, values = _normalish(1.08, 0.25)
    f = np.linspace(0.01, 0.99, 99)
    g = gr.growth_rate(f, dist, values)
    finite = np.isfinite(g)
    assert finite.sum() > 50
    second = np.diff(g[finite], 2)
    assert (second < 1e-9).all(), f"g(f) is not concave; worst second difference {second.max():.3e}"


def test_f_star_matches_the_closed_form_on_a_well_behaved_case():
    """Sanity bound only -- the closed form is never used in production.

    For a return multiple ``R`` with a small, roughly symmetric spread,
    ``f* ~= E[R-1] / E[(R-1)**2]``. It is a bad approximation on real portfolios,
    which is the whole reason this module reads the actual distribution, but on a
    near-normal low-variance case it should land close.

    The case is chosen so the optimum falls well inside the grid. At a larger
    edge -- mean 1.05, sd 0.15 -- the closed form asks for ``f* = 2.0``, which
    this distribution can support because nothing in it returns zero, but which
    `GROWTH_F_MAX` clamps to 0.99. That clamp is correct for real portfolios,
    where ``p0 > 0`` makes any ``f >= 1`` ruin, and it would make this a test of
    the cap rather than of the arithmetic.
    """
    dist, values = _normalish(1.02, 0.20)
    f_star, _ = gr.optimal_fraction(dist, values)
    excess = float((dist * (values - 1.0)).sum())
    second = float((dist * (values - 1.0) ** 2).sum())
    assert 0.05 < f_star < 0.95, "case must not sit against the grid edge"
    assert abs(f_star - excess / second) < 0.02


def test_ruin_prob_is_the_product_of_the_misses():
    p = np.array([[0.5, 0.25, 0.0, 0.8]])          # third event skipped
    assert gr.ruin_prob(p)[0] == pytest.approx(0.5 * 0.75 * 0.2)


def test_ruin_prob_matches_the_grids_zero_cell():
    """`ruin_prob` computes P(R == 0) without a distribution; the grid agrees."""
    p = np.array([0.3, 0.55, 0.4, 0.62, 0.48, 0.35, 0.7, 0.25])
    o = np.array([3.1, 1.9, 2.4, 1.7, 2.2, 2.9, 1.5, 4.0])
    pay, prob = _legs(p, o)
    dist, _ = return_pmf(pay, prob, close_tail=True)
    assert gr.ruin_prob(prob)[0] == pytest.approx(dist[0, 0], rel=1e-9)


# --- the ruin guard ---------------------------------------------------------


def test_full_stake_with_reachable_ruin_is_minus_infinity():
    """Never a finite number from a clipped log -- `f >= 1` with `p0 > 0` is ruin."""
    p = np.array([0.4, 0.6, 0.5, 0.55, 0.45, 0.5, 0.35, 0.65])
    o = np.array([2.8, 1.9, 2.2, 2.0, 2.5, 2.1, 3.0, 1.8])
    pay, prob = _legs(p, o)
    dist, step = return_pmf(pay, prob, close_tail=True)
    values = np.arange(dist.shape[1]) * step[0]

    assert gr.ruin_prob(prob)[0] > 0
    g = gr.growth_rate(np.array([1.0, 1.5]), dist[0], values)
    assert np.isneginf(g).all(), f"expected -inf at f >= 1, got {g}"


def test_f_star_is_always_below_one_when_ruin_is_possible():
    """Structural, not incidental: g(1) = -inf and g is concave, so the optimum
    is strictly inside. It is why the f grid stops below 1."""
    rng = np.random.default_rng(3)
    for _ in range(12):
        n = int(rng.integers(8, 17))
        p = rng.uniform(0.15, 0.85, n)
        o = rng.uniform(1.2, 6.0, n)
        pay, prob = _legs(p, o)
        dist, step = return_pmf(pay, prob, close_tail=True)
        values = np.arange(dist.shape[1]) * step[0]
        f_star, g_star = gr.optimal_fraction(dist[0], values)
        assert gr.ruin_prob(prob)[0] > 0
        assert f_star < 1.0, f"f* reached {f_star} with p0 > 0"
        assert np.isfinite(g_star)


# --- the constraint ---------------------------------------------------------


def test_protective_fraction_is_monotone_in_the_constraint():
    """Tightening the tolerated probability must never allow a *larger* stake."""
    p = np.array([0.35, 0.5, 0.45, 0.6, 0.4, 0.55, 0.3, 0.65, 0.42])
    o = np.array([3.2, 2.1, 2.4, 1.8, 2.7, 2.0, 3.6, 1.7, 2.5])
    pay, prob = _legs(p, o)
    dist, step = return_pmf(pay, prob, close_tail=True)
    values = np.arange(dist.shape[1]) * step[0]
    f_star, _ = gr.optimal_fraction(dist[0], values)
    returns = gr.sample_rounds(dist[0], values)

    prev = np.inf
    for max_prob in (0.20, 0.10, 0.05, 0.01):
        f_prot, _ = gr.protective_fraction(returns, f_star, max_prob=max_prob)
        assert f_prot <= prev + 1e-12, f"tightening to {max_prob} raised f_protective"
        assert f_prot <= f_star + 1e-12
        prev = f_prot


def test_protective_equals_star_when_the_constraint_is_slack():
    """No trade-off to make means no trade-off is made."""
    dist, values = _normalish(1.02, 0.03)
    f_star, _ = gr.optimal_fraction(dist, values)
    returns = gr.sample_rounds(dist, values)
    f_prot, _ = gr.protective_fraction(returns, f_star, drawdown=0.99, max_prob=0.99)
    assert f_prot == pytest.approx(f_star)


# --- the engine bargain -----------------------------------------------------


@pytest.mark.parametrize("n", [8, 10, 12])
def test_growth_matches_exact_enumeration(n):
    """The grid is only allowed because it agrees with counting every outcome."""
    rng = np.random.default_rng(11)
    worst_g, worst_f = 0.0, 0.0
    for _ in range(6):
        p = rng.uniform(0.2, 0.8, n)
        o = rng.uniform(1.3, 5.0, n)
        s = 1.0 / (p * o)
        s = s / s.sum()
        pay, prob = (s * o)[None, :], p[None, :]

        _, probs, returns = outcome_paths(p, o, s)
        dist, step = return_pmf(pay, prob, close_tail=True)
        values = np.arange(dist.shape[1]) * step[0]

        f = np.linspace(0.05, 0.9, 18)
        grid_g = gr.growth_rate(f, dist[0], values)
        exact_g = np.array([float((probs * np.log((1 - x) + x * returns)).sum()) for x in f])
        worst_g = max(worst_g, float(np.abs(grid_g - exact_g).max()))

        # Compare what the choice *achieves*, not where it lands. g is flat near
        # its optimum, so two f values a couple of grid steps apart can both be
        # optimal to floating point -- an argmax-location check would fail on a
        # tie while the decision it drives is identical.
        f_grid, _ = gr.optimal_fraction(dist[0], values)
        fine = np.round(np.arange(0.01, 0.99, 0.01), 10)
        best_exact = max(float((probs * np.log((1 - x) + x * returns)).sum()) for x in fine)
        at_grid = float((probs * np.log((1 - f_grid) + f_grid * returns)).sum())
        worst_f = max(worst_f, best_exact - at_grid)

    assert worst_g < 5e-3, f"g(f) drifted {worst_g:.4f} from exact enumeration"
    assert worst_f < 1e-6, f"grid f* forgoes {worst_f:.3e} of growth against exact enumeration"


def test_growth_metrics_batch_shape_and_invariants():
    rng = np.random.default_rng(5)
    m, n = 6, 9
    p = np.zeros((m, n)); o = np.zeros((m, n))
    for i in range(m):
        k = int(rng.integers(8, n + 1))
        j = rng.choice(n, size=k, replace=False)
        p[i, j] = rng.uniform(0.2, 0.8, k)
        o[i, j] = (1.0 / p[i, j]) * rng.uniform(1.05, 1.35, k)
    s = np.where(p > 0, 1.0 / np.where(p > 0, p * o, 1.0), 0.0)
    s = s / s.sum(axis=1, keepdims=True)

    out = gr.growth_metrics(s * o, p)
    assert len(out) == m
    assert (out["f_suggested"] > 0).all()
    assert (out["f_suggested"] <= out["f_drawdown"] + 1e-12).all()
    assert (out["f_suggested"] < 1.0).all(), "nothing may suggest staking the whole pot"
    assert (out["drawdown_p"] <= gr.GROWTH_DRAWDOWN_P + 1e-9).all()
    assert np.isfinite(out.to_numpy(dtype=float)).all(), "no NaN or inf may reach the payload"


def test_the_stake_is_exactly_the_drawdown_constraint_when_the_dials_are_off():
    """`PESSIMISM_B = SLATE_TAU = 0` must reproduce the old `f_protective` exactly.

    The two model-risk dials ship switched off, so this is the identity that says
    the new number is a re-description of the old behaviour rather than a new
    recommendation. If it ever fails, the stake changed for a reason nobody chose.
    """
    rng = np.random.default_rng(17)
    p = rng.uniform(0.2, 0.8, (4, 10))
    o = (1.0 / p) * rng.uniform(1.05, 1.35, (4, 10))
    s = (1.0 / (p * o)); s = s / s.sum(axis=1, keepdims=True)
    out = gr.growth_metrics(s * o, p)
    assert out["edge_factor"].to_numpy() == pytest.approx(1.0)
    assert out["var_factor"].to_numpy() == pytest.approx(1.0)
    assert out["f_suggested"].to_numpy() == pytest.approx(out["f_drawdown"].to_numpy())


def test_each_model_risk_dial_only_ever_shrinks_the_stake():
    """Both dials are one-sided: they can lower the stake and never raise it."""
    rng = np.random.default_rng(19)
    p = rng.uniform(0.25, 0.75, 12)
    o = (1.0 / p) * rng.uniform(1.05, 1.35, 12)
    s = (1.0 / (p * o)); s = s / s.sum()
    base, _ = gr.suggested_fraction(0.30, s * o, p, bias=0.0, tau=0.0)
    assert base == pytest.approx(0.30)
    for bias, tau in ((0.02, 0.0), (0.0, 0.03), (0.02, 0.03)):
        f, terms = gr.suggested_fraction(0.30, s * o, p, bias=bias, tau=tau)
        assert f < base, f"b={bias} tau={tau} did not shrink the stake"
        assert 0.0 <= terms["edge_factor"] <= 1.0
        assert 0.0 < terms["var_factor"] <= 1.0


def test_nothing_caps_the_stake_but_the_drawdown_constraint():
    """There is no flat ceiling, and none is needed.

    A cap on the stake is a cap on the worst round by definition -- lose every leg
    and you are out exactly what you staked -- so it restates the number it caps.
    The drawdown constraint bounds the path, `bias`/`tau` bound the model being
    wrong, and `protective_fraction` searches below `GROWTH_F_MAX`, so the stake is
    bounded below 1 without anyone imposing a number.
    """
    rng = np.random.default_rng(23)
    p = rng.uniform(0.25, 0.75, 12)
    o = (1.0 / p) * rng.uniform(1.05, 1.35, 12)
    s = (1.0 / (p * o)); s = s / s.sum()
    # whatever the drawdown search returns is passed through untouched
    for f_dd in (0.05, 0.30, 0.75, 0.98):
        f, _ = gr.suggested_fraction(f_dd, s * o, p, bias=0.0, tau=0.0)
        assert f == pytest.approx(f_dd)


# --- the projection block ---------------------------------------------------
#
# The Projection Book draws these and nothing else, so what needs pinning is not
# the arithmetic again -- it is that the drawn numbers and the printed numbers
# are the *same* numbers. Every test below is a form of that one claim.


@pytest.fixture(scope="module")
def projected():
    """A batch of six portfolios with the projection block computed."""
    rng = np.random.default_rng(11)
    m, n = 6, 9
    p = np.zeros((m, n)); o = np.zeros((m, n))
    for i in range(m):
        k = int(rng.integers(6, n + 1))
        j = rng.choice(n, size=k, replace=False)
        p[i, j] = rng.uniform(0.2, 0.8, k)
        # every leg clears e >= 1, as `qualify` guarantees in production
        o[i, j] = (1.0 / p[i, j]) * rng.uniform(1.05, 1.35, k)
    s = np.where(p > 0, 1.0 / np.where(p > 0, p * o, 1.0), 0.0)
    s = s / s.sum(axis=1, keepdims=True)
    return gr.growth_metrics(s * o, p, projection=True)


def test_projection_is_off_by_default():
    """The block costs bytes on every exported row, so it is opt-in."""
    p = np.array([[0.7, 0.5, 0.4]]); o = np.array([[1.5, 2.2, 2.8]])
    out = gr.growth_metrics(*_legs(p[0], o[0]))
    assert "g_curve" not in out.columns


def test_g_curve_passes_through_the_suggested_stake(projected):
    """The claim that justifies exporting the curve instead of fitting one.

    The page prints `g_suggested` beside a curve it reads other rates off. If the
    two disagreed at the suggested stake -- the one point where the answer is
    already known -- every other point on it would be unbelievable.

    Read off the nearest grid point rather than an exact match: `f_suggested` is
    the drawdown search's answer scaled by two shrinkage factors, so unlike the
    old `f_star` it does not land on `f_curve`'s grid.
    """
    fs = gr.f_curve()
    for _, r in projected.iterrows():
        j = int(np.argmin(np.abs(fs - r["f_suggested"])))
        assert abs(fs[j] - r["f_suggested"]) <= GROWTH_F_FINE
        assert r["g_curve"][j] == pytest.approx(r["g_suggested"], abs=2e-3)


def test_g_curve_rises_to_a_single_peak(projected):
    """`g(f)` is concave, so the exported curve must have exactly one summit.

    The old test pinned the peak to `f_star`. That column is gone -- it read back
    `GROWTH_F_MAX` on two thirds of real portfolios -- but the shape claim behind
    it still matters, because the page invites reading growth off this curve at
    stakes above the recommendation.
    """
    for _, r in projected.iterrows():
        g = np.asarray(r["g_curve"], dtype=float)
        finite = np.isfinite(g)
        top = int(np.argmax(np.where(finite, g, -np.inf)))
        assert np.all(np.diff(g[: top + 1]) >= -1e-9), "g(f) dips before its peak"
        assert r["g_suggested"] <= g[top] + 1e-9


def test_bands_are_ordered_at_every_round(projected):
    """p5 <= p50 <= p95, everywhere. A crossed fan is a transposed axis."""
    for _, r in projected.iterrows():
        b = r["bands_suggested"]
        if b is None:
            continue
        assert np.all(np.diff(b, axis=0) >= 0), "bands_suggested crosses itself"


def test_median_band_tracks_the_growth_rate(projected):
    """The fan's p50 is the growth rate compounding, not a separate estimate.

    Median log wealth after `n` rounds is close to `n * g` -- close, not equal,
    because the median of a sum is not the sum of medians, so this is a loose
    band around the claim rather than an identity.
    """
    rounds = gr.band_rounds()
    late = rounds >= 20
    for _, r in projected.iterrows():
        if r["bands_suggested"] is None:
            continue
        p50 = np.asarray(r["bands_suggested"])[1][late]
        assert p50 == pytest.approx(rounds[late] * r["g_suggested"], rel=0.35)


def test_bands_and_drawdown_read_the_same_futures(projected):
    """The consistency `sample_rounds` exists to make possible.

    `drawdown_p` and the fan are computed from one draw inside one call. Redoing
    the draw here with the same default seed must reproduce the exported band
    exactly -- if it does not, the two are describing different futures and the
    page would show a drawdown probability that its own chart contradicts.
    """
    p = np.array([0.72, 0.61, 0.55, 0.48, 0.40, 0.33])
    o = np.array([1.45, 1.72, 1.95, 2.25, 2.70, 3.30])
    pay, prob = _legs(p, o)
    out = gr.growth_metrics(pay, prob, projection=True).iloc[0]

    dist, step = return_pmf(pay, prob, close_tail=True)
    values = np.arange(dist.shape[1]) * step[0]
    again = gr.wealth_bands(out["f_suggested"], gr.sample_rounds(dist[0], values),
                            at=gr.band_rounds())
    assert np.allclose(again, out["bands_suggested"])


def test_histogram_conserves_the_pmf_mass(projected):
    """Rebinning for drawing must not quietly lose or invent probability."""
    for _, r in projected.iterrows():
        assert float(np.sum(r["hist"])) == pytest.approx(1.0, abs=1e-9)


def test_max_return_and_its_probability_are_the_mirror_of_ruin(projected):
    """`p_max` is to the best case what `p0` is to the worst.

    Both are closed form and exact where the grid is quantised, which is why
    neither is read off the histogram beside them.
    """
    for _, r in projected.iterrows():
        assert 0.0 < r["p_max"] < 1.0
        assert r["max_return"] > 1.0
        assert r["p_max"] + r["p0"] < 1.0


def test_sweep_prob_ignores_skipped_legs():
    """A skipped event carries `p = 0` and must not annihilate the product."""
    p = np.array([[0.5, 0.4, 0.0, 0.0]])
    assert gr.sweep_prob(p)[0] == pytest.approx(0.2)


def test_band_rounds_are_unique_and_span_the_horizon():
    r = gr.band_rounds()
    assert r[0] == 1 and r[-1] == gr.GROWTH_ROUNDS
    assert len(set(r.tolist())) == len(r)
    assert np.all(np.diff(r) > 0)


# --- the drawdown grid ------------------------------------------------------


def test_the_published_stake_is_the_grid_cell_not_a_second_search():
    """One computation, so the selector cannot disagree with the number beside it."""
    rng = np.random.default_rng(41)
    p = rng.uniform(0.25, 0.75, (3, 10))
    o = (1.0 / p) * rng.uniform(1.05, 1.35, (3, 10))
    s = (1.0 / (p * o)); s = s / s.sum(axis=1, keepdims=True)
    out = gr.growth_metrics(s * o, p)
    key = "dd_" + gr.default_drawdown_key()
    assert out["f_drawdown"].to_numpy() == pytest.approx(out[key].to_numpy())


def test_a_default_tolerance_outside_the_grid_is_refused():
    """A fallback here would restore the two answers the grid exists to remove."""
    with pytest.raises(ValueError, match="not a cell"):
        gr.default_drawdown_key(drawdown=0.37, max_prob=0.05)


def test_the_grid_is_monotone_in_both_directions():
    """A looser tolerance can only ever allow a larger stake."""
    rng = np.random.default_rng(43)
    p = rng.uniform(0.3, 0.7, 12)
    o = (1.0 / p) * rng.uniform(1.05, 1.3, 12)
    s = (1.0 / (p * o)); s = s / s.sum()
    dist, step = return_pmf((s * o)[None, :], p[None, :], close_tail=True)
    returns = gr.sample_rounds(dist[0], np.arange(dist.shape[1]) * step[0])
    grid = gr.drawdown_grid(returns, 0.9)
    ds, ps = gr.GROWTH_DRAWDOWN_GRID_D, gr.GROWTH_DRAWDOWN_GRID_P
    for d in ds:                       # a bigger tolerated fall allows more stake
        row = [grid[f"d{int(d * 100):02d}_p{int(q * 100):02d}"] for q in ps]
        assert row == sorted(row), f"not monotone in probability at d={d}"
    for q in ps:                       # a bigger tolerated chance allows more stake
        col = [grid[f"d{int(d * 100):02d}_p{int(q * 100):02d}"] for d in ds]
        assert col == sorted(col), f"not monotone in drawdown at p={q}"


def test_max_drawdowns_agrees_with_the_probability_it_replaced():
    """`drawdown_prob` is now a tail count of `max_drawdowns`; they must agree."""
    rng = np.random.default_rng(45)
    returns = rng.uniform(0.0, 2.5, (2000, 60))
    for f in (0.05, 0.2, 0.5):
        mdd = gr.max_drawdowns(f, returns)
        for d in (0.1, 0.3, 0.5):
            assert gr.drawdown_prob(f, returns, drawdown=d) == pytest.approx(
                float((mdd > d).mean()))
