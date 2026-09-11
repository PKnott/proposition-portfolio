"""Guards on the portfolio search.

Three things here are worth more than the rest, because they are the three
places the module trades exactness for speed and could be wrong without looking
wrong:

1. `threshold_probs_batch` convolves onto a grid instead of enumerating
   ``2**n`` outcomes, so it is pinned against `threshold_probs_exact`.
2. `metrics` reports plain sums while the search ranks on closed forms derived
   from them, so the two are pinned against each other.
3. `build_pool` searches instead of enumerating above `EXHAUSTIVE_MAX`, so on a
   space small enough for both, the searched pool's frontier is pinned against
   the enumerated one's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpp import portfolio as pf
from fpp import staking

THR = pf.THRESHOLDS


# --- Fixtures ---------------------------------------------------------------


def _form(p, o, codes, labels=None):
    """A filled odds form with one book, priced exactly as given."""
    labels = labels or [f"prop {i}" for i in range(len(p))]
    return pd.DataFrame({
        "sheet_code": list(codes),
        "label": labels,
        "p": np.asarray(p, dtype=float),
        "b365": np.asarray(o, dtype=float),
        "home_team": "H", "away_team": "A",
        "team": "H", "target": "corners",
        "line": np.arange(len(p), dtype=float) + 0.5,
    })


def _random_options(rng, n_events, k_max=3):
    codes, ps, os_, rows = [], [], [], []
    for j in range(n_events):
        k = int(rng.integers(1, k_max + 1))
        p = rng.uniform(0.15, 0.9, k)
        o = (1.0 / p) * rng.uniform(1.01, 1.25, k)   # every option clears e >= 1
        codes.append(f"EV-{j:02d}")
        ps.append(p)
        os_.append(o)
        for i in range(k):
            rows.append({"sheet_code": codes[-1], "label": f"EV-{j:02d} opt {i}",
                         "p": p[i], "o": o[i], "book": "Bet365",
                         "home_team": "H", "away_team": "A",
                         "event": j, "option": i})
    return pf.Options(tuple(codes), tuple(ps), tuple(os_), pd.DataFrame(rows))


def _er_var(picks, opts, split=pf.SPLIT_GROWTH, max_leg_stake=1.0):
    """Expected return and variance only -- no thresholds.

    `score_pool` computes threshold probabilities too, at ~1 ms a portfolio, so
    calling it on a million-portfolio pool is twenty minutes. The frontier tests
    below only compare expected return and variance, which are closed-form and
    vectorised, so they take that road instead.

    Uncapped by default: the frontier tests compare against the closed forms the
    search ranks on, and the per-leg cap is deliberately a departure from them.
    """
    stakes, p, o, rho = pf.stakes_for(picks, opts, split, max_leg_stake=max_leg_stake)
    return pf.metrics(stakes, p, o, rho)


def _frontier_er(picks, opts):
    """Expected returns on the exact (ER, variance) frontier, ascending.

    Two stages, because the exact dominance test is a Python loop over rows and
    the pools here have a million of them: `_pareto2_superset` cuts to a few
    hundred candidates with a sort, then the exact test runs on those.
    """
    er, var = _er_var(picks, opts)
    cand = np.flatnonzero(pf._pareto2_superset(er, var))
    df = pd.DataFrame({"pct_expected_return": er[cand], "variance": var[cand]})
    keep = pf.undominated_portfolios(df, ("pct_expected_return", "variance"), (+1, -1))
    return np.sort(er[cand][keep])


# --- 1. The threshold engine ------------------------------------------------


@pytest.mark.parametrize("n", [1, 3, 8, 14, 18])
def test_grid_thresholds_match_exact_enumeration(n):
    """The whole reason the grid is allowed: it agrees with counting outcomes."""
    rng = np.random.default_rng(n)
    worst = 0.0
    for _ in range(6):
        p = rng.uniform(0.1, 0.92, n)
        o = (1.0 / p) * rng.uniform(1.01, 1.3, n)
        stakes = staking.stake_split(p * o, 1.0)
        pay = stakes * o
        exact = pf.threshold_probs_exact(pay, p, THR)
        grid = pf.threshold_probs_batch(pay[None, :], p[None, :], THR)[0]
        worst = max(worst, float(np.abs(exact - grid).max()))
    assert worst < 0.005, f"grid engine drifted {worst:.4f} from exact enumeration"


@pytest.mark.parametrize("n", [8, 12, 16])
@pytest.mark.parametrize("split", ["1/E", "min variance"])
def test_grid_is_accurate_on_the_shapes_the_search_produces(n, split):
    """The wide-spread case, which the narrow one above does not reach.

    The test above ties ``o`` to ``1/p`` with a small edge multiplier, so every
    leg's payout sits in a narrow band and quantisation has little to bite on.
    Real portfolios are not like that: one day's exported set spans odds 1.05 to
    12 against hit probabilities 0.05 to 0.95, and the minimum-variance split
    concentrates most of the stake on a single leg.

    Those are the rows where the grid drifted. At the old ``GRID = 4096``,
    ``p_over_100`` was out by up to 1.4e-02 on the worst real portfolios -- and
    since it is a dominance criterion, rebuilding the frontier at a finer grid
    moved 260 of 2,740 exported portfolios. This pins the shape, not just the
    tolerance, so a future grid change has to face the case that mattered.
    """
    rng = np.random.default_rng(1000 + n)
    worst = 0.0
    for _ in range(6):
        p = rng.uniform(0.05, 0.95, n)
        o = rng.uniform(1.05, 12.0, n)
        stakes = (staking.stake_split(p * o, 1.0) if split == "1/E"
                  else staking.min_variance_alloc(p, o, 1.0))
        pay = stakes * o
        exact = pf.threshold_probs_exact(pay, p, THR)
        grid = pf.threshold_probs_batch(pay[None, :], p[None, :], THR)[0]
        worst = max(worst, float(np.abs(exact - grid).max()))
    assert worst < 0.005, f"grid engine drifted {worst:.4f} on a realistic {n}-leg {split} portfolio"


def test_grid_thresholds_sharpen_as_the_grid_refines():
    """Error is quantisation, so it must fall when the grid is made finer."""
    rng = np.random.default_rng(7)
    p = rng.uniform(0.15, 0.9, 15)
    o = (1.0 / p) * rng.uniform(1.02, 1.25, 15)
    pay = staking.stake_split(p * o, 1.0) * o
    exact = pf.threshold_probs_exact(pay, p, THR)
    errs = [float(np.abs(exact - pf.threshold_probs_batch(pay[None, :], p[None, :], THR, grid=g)[0]).max())
            for g in (256, 1024, 8192)]
    assert errs[0] > errs[-1]
    assert errs[-1] < 0.002


def test_a_skipped_event_changes_nothing():
    """Skips travel as zeros, so they must be a genuine no-op, not a small one."""
    rng = np.random.default_rng(3)
    p = rng.uniform(0.2, 0.85, 6)
    o = (1.0 / p) * rng.uniform(1.02, 1.2, 6)
    pay = staking.stake_split(p * o, 1.0) * o
    padded_p = np.r_[p[:3], 0.0, p[3:], 0.0]
    padded_pay = np.r_[pay[:3], 0.0, pay[3:], 0.0]
    a = pf.threshold_probs_batch(pay[None, :], p[None, :], THR)[0]
    b = pf.threshold_probs_batch(padded_pay[None, :], padded_p[None, :], THR)[0]
    assert a == pytest.approx(b, abs=1e-12)


def test_thresholds_are_monotone_and_bounded():
    rng = np.random.default_rng(11)
    opts = _random_options(rng, 8)
    picks, _ = pf.build_pool(opts, min_legs=1)
    stakes, p, o, rho = pf.stakes_for(picks, opts, pf.SPLIT_GROWTH)
    th = pf.threshold_probs_batch(stakes * o, p, THR)
    assert ((th >= 0) & (th <= 1)).all()
    assert (np.diff(th, axis=1) <= 1e-12).all(), "P(return > t) must fall as t rises"


def test_an_unreachable_threshold_reads_zero():
    """One leg at 1.5x can never return 2x the stake, whatever it does."""
    pay = np.array([[1.5]]); p = np.array([[0.9]])
    assert pf.threshold_probs_batch(pay, p, (2.0,))[0, 0] == 0.0
    assert pf.threshold_probs_batch(pay, p, (1.0,))[0, 0] == pytest.approx(0.9, abs=1e-9)


# --- 2. Closed forms against plain sums ------------------------------------


@pytest.mark.parametrize("split", pf.SPLITS)
def test_metrics_match_the_closed_forms_the_search_ranks_on(split):
    """`metrics` sums; `_search` ranks on `C/W` and `C/W^2`.

    If these ever disagree the search is optimising something the report is not
    measuring, and nothing else in the module would notice.
    """
    rng = np.random.default_rng(5)
    opts = _random_options(rng, 9)
    picks, _ = pf.build_pool(opts, min_legs=1)
    stakes, p, o, rho = pf.stakes_for(picks, opts, split, max_leg_stake=1.0)
    er, var = pf.metrics(stakes, p, o, rho)

    terms = [pf.leg_terms(opts.p[j], opts.o[j]) for j in range(opts.n_events)]
    sums = pf._sum_terms(picks, opts, terms)
    live = sums["w"] > 0
    assert live.any(), "the fixture must produce portfolios that carry some edge"
    assert er[live] == pytest.approx(1.0 + sums["c"][live] / sums["w"][live], rel=1e-9)
    assert var[live] == pytest.approx(sums["c"][live] / sums["w"][live] ** 2, rel=1e-7)


def test_growth_weights_reach_the_capacity_ceiling():
    """Sharpe == sqrt(C), the identity the whole model now rests on.

    Portfolio Sharpe under growth weights equals the root of the sum of squared
    standalone leg Sharpes, which is the maximum any weighting can reach. It is an
    algebraic identity, not a fit, so it should hold to floating point -- and if it
    ever stops, `capacity` has silently become a number that means nothing.
    """
    rng = np.random.default_rng(21)
    opts = _random_options(rng, 9)
    picks, _ = pf.build_pool(opts, min_legs=1)
    stakes, p, o, rho = pf.stakes_for(picks, opts, pf.SPLIT_GROWTH, max_leg_stake=1.0)
    er, var = pf.metrics(stakes, p, o, rho)
    capacity = pf._sum_terms(
        picks, opts, [pf.leg_terms(opts.p[j], opts.o[j]) for j in range(opts.n_events)])["c"]
    live = var > 0
    sharpe = (er[live] - 1.0) / np.sqrt(var[live])
    assert sharpe == pytest.approx(np.sqrt(capacity[live]), rel=1e-9)


@pytest.mark.parametrize("split", pf.SPLITS)
@pytest.mark.parametrize("cap", [1.0, 0.15])
def test_every_portfolio_spends_the_whole_stake(split, cap):
    rng = np.random.default_rng(6)
    opts = _random_options(rng, 7)
    picks, _ = pf.build_pool(opts, min_legs=1)
    stakes, p, _, _ = pf.stakes_for(picks, opts, split, max_leg_stake=cap)
    assert stakes.sum(axis=1) == pytest.approx(1.0)
    # and puts nothing on an event it skipped -- both of that event's slots
    skipped = np.repeat(picks < 0, 2, axis=1)
    assert (stakes[skipped] == 0).all()
    assert (p[skipped] == 0).all()


def test_the_leg_cap_binds_without_ever_losing_stake():
    """`MAX_LEG_STAKE` holds, except where no allocation could satisfy it.

    Six legs cannot each hold under 15%, so the cap relaxes per row to the
    equal-weight floor. Clipping to an unreachable limit instead would leave the
    row staking less than the whole amount -- a different bet from the one scored.
    """
    rng = np.random.default_rng(31)
    opts = _random_options(rng, 9)
    picks, _ = pf.build_pool(opts, min_legs=1)
    stakes, p, _, _ = pf.stakes_for(picks, opts, pf.SPLIT_GROWTH, max_leg_stake=0.15)
    n_props = (p > 0).sum(axis=1)          # propositions held, not events backed
    assert stakes.sum(axis=1) == pytest.approx(1.0)
    row_limit = np.maximum(0.15, 1.0 / np.maximum(n_props, 1))
    assert (stakes.max(axis=1) <= row_limit + 1e-9).all()
    # and it is not a no-op: something in the pool must actually be capped
    assert (n_props >= 7).any() and (stakes.max(axis=1)[n_props >= 7] <= 0.15 + 1e-9).all()


def test_the_growth_split_agrees_with_staking_on_one_portfolio():
    """The portfolio module must not quietly re-derive what `staking` defines."""
    rng = np.random.default_rng(9)
    opts = _random_options(rng, 5)
    picks, _ = pf.build_pool(opts, min_legs=5)
    row = picks[[0]]
    # singles only: `staking`'s primitives know nothing about correlated pairs,
    # and the claim being pinned is that this module does not re-derive them.
    opts = pf.event_options(pf.qualify(_form(
        [0.5, 0.4, 0.6], [2.2, 2.8, 1.9], ["A-1", "A-2", "A-3"])), pairs=False)
    picks, _ = pf.build_pool(opts, min_legs=3)
    row = picks[[0]]
    stakes, p, o, rho = pf.stakes_for(row, opts, pf.SPLIT_GROWTH, max_leg_stake=1.0)
    live = p[0] > 0
    assert stakes[0].sum() == pytest.approx(1.0, rel=1e-12)
    er, var = pf.metrics(stakes, p, o, rho)
    assert er[0] == pytest.approx(staking.expected_return(p[0][live], o[0][live], stakes[0][live]))
    assert var[0] == pytest.approx(staking.variance(p[0][live], o[0][live], stakes[0][live]))


def test_the_retired_splits_refuse_rather_than_guess():
    """`1/E` and `Min variance` are names the ledger still needs, not options."""
    rng = np.random.default_rng(13)
    opts = _random_options(rng, 4)
    picks, _ = pf.build_pool(opts, min_legs=1)
    for split in pf.LEGACY_SPLITS:
        with pytest.raises(ValueError, match="retired"):
            pf.stakes_for(picks, opts, split)


# --- 3. The search against exhaustive enumeration --------------------------


def test_enumeration_covers_exactly_the_legal_portfolios():
    opts = _random_options(np.random.default_rng(21), 4, k_max=3)
    picks, info = pf.build_pool(opts, min_legs=1)
    assert info["mode"] == "exhaustive"
    assert len(picks) == int(np.prod(opts.sizes + 1)) - 1 == int(opts.space())
    assert len(np.unique(picks, axis=0)) == len(picks)
    assert (picks >= 0).sum(axis=1).min() >= 1
    for j in range(opts.n_events):
        assert picks[:, j].max() == opts.sizes[j] - 1


def test_min_legs_all_means_one_from_every_event():
    """The no-skipping switch, and the only setting with a provable answer.

    Spelled as a word because the number it stands for -- how many events
    survived qualifying -- changes every week.
    """
    opts = _random_options(np.random.default_rng(24), 7, k_max=3)
    picks, info = pf.build_pool(opts, min_legs="all")
    assert ((picks >= 0).sum(axis=1) == opts.n_events).all(), "an event was skipped"
    assert info["mode"] == "exhaustive", "no-skipping must enumerate, not search"
    assert len(picks) == int(np.prod(opts.sizes))

    with pytest.raises(ValueError, match="must be an int or 'all'"):
        pf.build_pool(opts, min_legs="every")


def test_leg_count_bounds_are_honoured():
    opts = _random_options(np.random.default_rng(22), 6)
    for lo, hi in [(1, 1), (2, 3), (6, 6)]:
        picks, _ = pf.build_pool(opts, min_legs=lo, max_legs=hi)
        n = (picks >= 0).sum(axis=1)
        assert n.min() >= lo and n.max() <= hi, f"{lo}-{hi} gave {n.min()}-{n.max()}"


def test_searched_pool_finds_the_same_frontier_as_full_enumeration():
    """The search is a cost control, so it must not change the answer.

    Run both paths over one space small enough to enumerate and compare the
    ``(expected return, variance)`` frontier, which is what the dynamic program
    claims to preserve exactly.
    """
    opts = _random_options(np.random.default_rng(31), 7, k_max=3)
    full, info_full = pf.build_pool(opts, min_legs=1, exhaustive_max=10**9)
    found, info_srch = pf.build_pool(opts, min_legs=1, exhaustive_max=0)
    assert info_full["mode"] == "exhaustive" and info_srch["mode"] == "searched"

    def frontier(picks):
        s = pf.score_pool(picks, opts, splits=(pf.SPLIT_GROWTH,))
        keep = pf.undominated_portfolios(s, ("pct_expected_return", "variance"), (+1, -1))
        return np.sort(np.round(s.loc[keep, "pct_expected_return"].to_numpy(), 9))

    assert frontier(found) == pytest.approx(frontier(full), rel=1e-9)


def test_the_default_bucket_count_is_exact_where_1024_was_not():
    """The dial is set from measurement, so pin what the measurement showed.

    On a real 26-event form the searched path finds 87 of 90 frontier portfolios
    at 1024 buckets and all 90 at 4096. Reproduced here on a synthetic space of
    the same shape: whatever else changes, the default must not go back to a
    setting that loses frontier members a finer grid would have found.
    """
    opts = _random_options(np.random.default_rng(101), 12, k_max=4)
    full, _ = pf.build_pool(opts, min_legs=1, exhaustive_max=10**12, pool_max=10**9)
    truth = _frontier_er(full, opts)

    def recall(buckets):
        got, _ = pf.build_pool(opts, min_legs=1, exhaustive_max=0, buckets=buckets,
                               pool_max=10**9)
        found = _frontier_er(got, opts)
        return sum(np.isclose(t, found, rtol=0, atol=1e-12).any() for t in truth)

    assert pf.BUCKETS >= 4096, "the default was lowered below the measured-exact setting"
    assert recall(pf.BUCKETS) >= recall(256), "a finer grid must never find less"
    assert recall(pf.BUCKETS) == len(truth), "the default should reach the whole frontier here"


def test_at_scale_the_search_degrades_gracefully_rather_than_exactly():
    """Bucketing costs coverage, so pin *how much* rather than pretend none.

    On a space large enough for the buckets to bind, the searched frontier is no
    longer identical to the enumerated one. What must still hold is that the best
    portfolio is found exactly, and that anything missed is indistinguishable
    from something that was not -- which is the actual claim the module makes.
    Measured on a real 26-event form: 87 of 90, worst gap 7.4e-05.
    """
    # Sized so both paths can run and the buckets genuinely bind: 1.46e6
    # portfolios over twelve leg counts and 1024 buckets is ~119 candidates per
    # cell against a `KEEP_PER_CELL` of 8, so the search really is discarding
    # most of what it sees rather than trivially keeping everything.
    opts = _random_options(np.random.default_rng(101), 12, k_max=4)
    assert opts.enumeration_cost(1) > 10 * pf.BUCKETS * opts.n_events, "space too small to bind"
    full, _ = pf.build_pool(opts, min_legs=1, exhaustive_max=10**12, pool_max=10**9)
    found, _ = pf.build_pool(opts, min_legs=1, exhaustive_max=0, pool_max=10**9)

    a, b = _frontier_er(full, opts), _frontier_er(found, opts)
    assert b.max() == pytest.approx(a.max(), rel=1e-12), "the best portfolio must be found"
    gaps = [abs(v - b[np.argmin(np.abs(b - v))]) for v in a]
    assert max(gaps) < 1e-3, f"a missed frontier portfolio is {max(gaps):.2e} from anything found"


def test_thinning_does_not_promote_a_dominated_portfolio():
    """A pool cap must not drop the dominator and leave the dominated looking clean.

    This is what `_thin`'s frontier protection is for; without it, thinning
    442,368 real portfolios reported an 88-portfolio frontier where the true one
    has 90 -- some of them dominated by something the cap had removed.
    """
    opts = _random_options(np.random.default_rng(103), 9, k_max=3)
    big, _ = pf.build_pool(opts, min_legs=1)
    small, _ = pf.build_pool(opts, min_legs=1, pool_max=150)

    big_er, big_var = _er_var(big, opts)
    X = np.column_stack([big_er, -big_var])
    for er in _frontier_er(small, opts):
        i = int(np.argmin(np.abs(big_er - er)))
        x = np.array([big_er[i], -big_var[i]])
        ge = np.all(X >= x - 1e-12, axis=1)
        gt = np.any(X > x + 1e-9, axis=1)
        assert not (ge & gt).any(), "thinned pool called a dominated portfolio undominated"


def test_the_pool_cap_is_respected_and_keeps_the_frontier():
    opts = _random_options(np.random.default_rng(41), 8, k_max=3)
    big, _ = pf.build_pool(opts, min_legs=1)
    small, info = pf.build_pool(opts, min_legs=1, pool_max=200)
    assert len(small) <= 200 < len(big)
    assert info["pool"] == len(small)

    # Thinning must not cost the best portfolio available.
    assert _er_var(small, opts)[0].max() == pytest.approx(_er_var(big, opts)[0].max(), rel=1e-9)


# --- Dominance --------------------------------------------------------------


def test_dominance_keeps_only_the_pareto_set():
    df = pd.DataFrame({
        "pct_expected_return": [1.10, 1.05, 1.10, 1.02],
        "variance":            [0.02, 0.01, 0.03, 0.05],
        "p_over_100":          [0.70, 0.68, 0.65, 0.60],
    })
    # row 2 is beaten by row 0 on all three; row 3 by everything.
    assert pf.undominated_portfolios(df).tolist() == [True, True, False, False]


def test_an_exact_tie_on_every_criterion_keeps_both():
    df = pd.DataFrame({"pct_expected_return": [1.1, 1.1], "variance": [0.02, 0.02],
                       "p_over_100": [0.7, 0.7]})
    assert pf.undominated_portfolios(df).tolist() == [True, True]


def test_dominance_matches_brute_force_on_a_real_sized_sample():
    rng = np.random.default_rng(77)
    df = pd.DataFrame({
        "pct_expected_return": rng.uniform(1.0, 1.2, 900),
        "variance": rng.uniform(0.01, 0.05, 900),
        "p_over_100": rng.uniform(0.5, 0.8, 900),
    })
    X = np.column_stack([df["pct_expected_return"], -df["variance"], df["p_over_100"]])
    ge = np.all(X[:, None, :] >= X[None, :, :], axis=2)
    gt = np.any(X[:, None, :] > X[None, :, :], axis=2)
    brute = ~(ge & gt).any(axis=0)
    assert pf.undominated_portfolios(df).tolist() == brute.tolist()


def test_dominance_runs_across_both_splits_at_once():
    """A 1/E row must be comparable with a min-variance row, not siloed from it."""
    scored = pd.DataFrame({
        "split": [pf.SPLIT_GROWTH, pf.SPLIT_GROWTH],
        "pct_expected_return": [1.05, 1.09],
        "variance": [0.04, 0.02],
        "p_over_100": [0.60, 0.75],
    })
    assert pf.undominated_portfolios(scored).tolist() == [False, True]


def test_unknown_criteria_raise_rather_than_silently_ranking_on_nothing():
    df = pd.DataFrame({"pct_expected_return": [1.0], "variance": [0.1], "p_over_100": [0.5]})
    with pytest.raises(ValueError, match="unknown dominance criteria"):
        pf.undominated_portfolios(df, ("pct_expected_return", "sharpe"), (+1, +1))


# --- Qualifying -------------------------------------------------------------


def test_qualify_drops_negative_edge_then_takes_the_p_o_frontier():
    #                     e = 1.20  0.90   1.05         1.10   1.32
    filled = _form(p=[0.60, 0.60, 0.70, 0.50, 0.60],
                   o=[2.00, 1.50, 1.50, 2.20, 2.20],
                   codes=["A", "A", "A", "B", "B"],
                   labels=["a-hi", "a-neg", "a-mid", "b-lo", "b-hi"])
    out = pf.qualify(filled)
    # "a-neg" has e < 1; "a-mid" is beaten by "a-hi" on p? no -- higher p, lower o,
    # so both survive. "b-lo" is beaten by "b-hi" on both p and o.
    assert set(out["label"]) == {"a-hi", "a-mid", "b-hi"}
    assert (out["e"] >= 1.0).all()


def test_an_event_losing_everything_simply_has_no_options():
    filled = _form(p=[0.5, 0.5], o=[2.5, 1.2], codes=["A", "B"])
    opts = pf.event_options(pf.qualify(filled))
    assert opts.codes == ("A",)
    assert opts.n_events == 1


def test_no_portfolio_takes_two_propositions_from_one_event():
    rng = np.random.default_rng(55)
    opts = _random_options(rng, 6, k_max=4)
    picks, _ = pf.build_pool(opts, min_legs=1)
    # One column per event and one pick per column is the structural guarantee;
    # this is what licenses the independence assumption in the variance.
    assert picks.shape[1] == opts.n_events
    assert picks.max() < opts.sizes.max()


# --- End to end -------------------------------------------------------------


def test_search_end_to_end():
    rng = np.random.default_rng(99)
    rows = []
    for j in range(6):
        for i in range(2):
            p = float(rng.uniform(0.3, 0.8))
            rows.append({"sheet_code": f"EV-{j}", "label": f"EV-{j} #{i}", "p": p,
                         "b365": (1.0 / p) * float(rng.uniform(1.02, 1.2)),
                         "home_team": "H", "away_team": "A",
                         "team": "H", "target": "corners", "line": 0.5 + i})
    res = pf.search(pd.DataFrame(rows), min_legs=1)

    s = res["scored"]
    assert len(s) == len(res["picks"])                     # one split, every combo
    assert set(s["split"]) == set(pf.SPLITS)
    assert s["undominated"].any()
    assert s["id"].is_unique
    # Sorted undominated-first, then by probability of profit.
    assert s["undominated"].tolist() == sorted(s["undominated"].tolist(), reverse=True)

    top = s.iloc[0]
    bets = pf.legs(res["picks"], res["options"], int(top["combo"]), top["split"], total=100.0)
    assert len(bets) == top["legs"]
    assert bets["stake"].sum() == pytest.approx(100.0)
    assert bets["event"].is_unique
    assert (bets["e"] >= 1.0).all()


def test_search_on_a_form_where_nothing_qualifies():
    filled = _form(p=[0.4, 0.4], o=[1.5, 1.5], codes=["A", "B"])   # e = 0.6
    res = pf.search(filled)
    assert len(res["picks"]) == 0
    assert res["scored"].empty


# --- the leg floor, set relative to whatever qualifies --------------------


def test_leg_var_resolves_against_the_events_that_actually_qualify():
    """The floor is relative because the number it is relative to is not knowable.

    A weekend of 45 fixtures may qualify 43 events or 37, depending on which
    markets got priced. An absolute floor typed against a guess silently becomes
    "skip nothing" if fewer qualify, and asks for the impossible if fewer still.
    """
    rng = np.random.default_rng(4)
    opts = _random_options(rng, 9)
    assert opts.n_events == 9
    assert opts.min_legs_for(0) == 9        # one bet from every event
    assert opts.min_legs_for(3) == 6
    # an over-large leg_var opens the search rather than asking for no legs
    assert opts.min_legs_for(9) == 1
    assert opts.min_legs_for(50) == 1
    with pytest.raises(ValueError, match="leg_var"):
        opts.min_legs_for(-1)


def test_leg_var_and_min_legs_are_the_same_floor_by_two_names():
    rng = np.random.default_rng(5)
    opts = _random_options(rng, 8)
    for lv in (0, 2, 4):
        a, ia = pf.build_pool(opts, leg_var=lv)
        b, ib = pf.build_pool(opts, min_legs=opts.n_events - lv)
        assert ia["min_legs"] == ib["min_legs"] == opts.n_events - lv
        assert np.array_equal(a, b)
    # `leg_var=0` is `min_legs="all"` by another name, provable path included
    a, ia = pf.build_pool(opts, leg_var=0)
    b, ib = pf.build_pool(opts, min_legs="all")
    assert ia["mode"] == ib["mode"] and np.array_equal(a, b)


def test_the_two_floor_controls_refuse_to_be_given_at_once():
    """They set the same number, so accepting both would let them disagree."""
    rng = np.random.default_rng(6)
    opts = _random_options(rng, 5)
    with pytest.raises(ValueError, match="not both"):
        pf.build_pool(opts, min_legs=3, leg_var=1)


def test_every_portfolio_respects_the_floor_leg_var_asked_for():
    rng = np.random.default_rng(7)
    opts = _random_options(rng, 10)
    for lv in (0, 2, 5):
        picks, info = pf.build_pool(opts, leg_var=lv)
        n_legs = (picks >= 0).sum(axis=1)
        assert n_legs.min() >= opts.n_events - lv
        assert n_legs.max() <= opts.n_events
        assert info["leg_var"] == lv


def test_a_tighter_floor_yields_fewer_portfolios():
    """The floor is the cheapest runtime lever: it cuts what `score_pool` pays for."""
    rng = np.random.default_rng(8)
    opts = _random_options(rng, 12)
    counts = [len(pf.build_pool(opts, leg_var=lv)[0]) for lv in (0, 3, 6, 12)]
    assert counts == sorted(counts), f"more slack must not yield fewer portfolios: {counts}"
    assert counts[0] < counts[-1]
