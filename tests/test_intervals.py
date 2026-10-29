"""Interval selection (`fpp.intervals`): lower bounds, sizing point, track-record width."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from fpp import intervals as iv


def test_level_one_half_is_todays_rule():
    p = np.array([0.2, 0.5, 0.8])
    np.testing.assert_allclose(iv.lower_bound(p, level=0.5, width_a=np.full(3, 0.3)), p)
    np.testing.assert_allclose(iv.lower_bound(p, level=0.5, width_b=np.full(3, 0.1)), p)


def test_lower_bound_falls_with_width_and_level():
    p = np.full(3, 0.6)
    w = np.array([0.05, 0.2, 0.5])
    lo = iv.lower_bound(p, level=0.75, width_a=w)
    assert (np.diff(lo) < 0).all() and (lo < p).all()
    assert (iv.lower_bound(p, level=0.9, width_a=w) < lo).all()
    # A works on the logit scale, B in probability.
    np.testing.assert_allclose(lo, expit(logit(0.6) - iv.z_for(0.75) * w))
    np.testing.assert_allclose(iv.lower_bound(p, level=0.75, width_b=w), 0.6 - iv.z_for(0.75) * w)


def test_exactly_one_width():
    with pytest.raises(ValueError):
        iv.lower_bound([0.5], level=0.6)
    with pytest.raises(ValueError):
        iv.lower_bound([0.5], level=0.6, width_a=[0.1], width_b=[0.1])


def test_sizing_point_runs_from_p_to_the_bound():
    p, lo = np.array([0.6, 0.4]), np.array([0.5, 0.3])
    np.testing.assert_allclose(iv.sizing_probability(p, lo, 0.0), p)
    np.testing.assert_allclose(iv.sizing_probability(p, lo, 1.0), lo)
    np.testing.assert_allclose(iv.sizing_probability(p, lo, 0.5), [0.55, 0.35])


def test_bootstrap_width_is_the_logit_spread_of_the_resamples():
    boot = pd.DataFrame({"p_0": [0.5, 0.5], "p_1": [0.4, 0.5], "p_2": [0.6, 0.5]})
    w = iv.bootstrap_width(boot)
    assert w.iloc[1] == 0.0
    assert w.iloc[0] == pytest.approx(np.std(logit([0.4, 0.6]), ddof=1))


def _slates(err_sd, n_slates=40, n=400, seed=0):
    """Lines above the market whose slate-level error has sd `err_sd`."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_slates):
        p = rng.uniform(0.3, 0.7, n)
        market = p - rng.uniform(0.005, 0.2, n)
        shift = rng.normal(0, err_sd)
        y = rng.random(n) < np.clip(p + shift, 0.01, 0.99)
        rows.append(pd.DataFrame({"run_code": f"R{s:03d}", "p_raw": p, "o": 1 / market,
                                  "y": y.astype(float)}))
    return pd.concat(rows, ignore_index=True)


def test_track_record_width_recovers_injected_error_and_ignores_binomial_noise():
    clean = iv.gap_error_table(_slates(0.0))
    assert np.average(clean["width"], weights=clean["n"]) < 0.03
    noisy = iv.gap_error_table(_slates(0.08, seed=1))
    assert np.average(noisy["width"], weights=noisy["n"]) == pytest.approx(0.08, abs=0.025)


def test_track_record_width_is_never_negative():
    a, b = iv.track_record_width(_slates(0.05, seed=3))
    assert a >= 0 and b >= 0
