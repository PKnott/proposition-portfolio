"""Calibration machinery, checked against streams whose answer is known."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fpp.calibration
from fpp.calibration import (
    AVOID,
    CAUTION,
    MIN_BIN_COUNT,
    TRUST,
    grade,
    _bin_frame,
    _summarise,
    calibration_slope,
    calibration_tables,
    quantile_bins,
    wilson_interval,
)


def _stream(n: int = 20_000, sharpen: float = 1.0, seed: int = 11):
    """Predictions whose miscalibration is set by construction.

    ``sharpen`` > 1 pushes the stated probabilities away from the base rate while
    outcomes keep following the honest ones -- i.e. overconfidence, and the true
    calibration slope is 1 / sharpen.
    """
    rng = np.random.default_rng(seed)
    honest = rng.uniform(0.05, 0.95, n)
    y = rng.binomial(1, honest)
    logit = np.log(honest / (1 - honest)) * sharpen
    return y, 1 / (1 + np.exp(-logit))


# --- Wilson ---------------------------------------------------------------


def test_wilson_stays_inside_the_unit_interval_at_the_extremes():
    """The reason for Wilson over the normal approximation."""
    for k, n in ((0, 40), (40, 40), (1, 200), (199, 200)):
        lo, hi = wilson_interval(k, n)
        assert 0.0 <= lo <= hi <= 1.0


def test_wilson_brackets_the_observed_frequency():
    for k, n in ((5, 50), (25, 50), (45, 50), (300, 1000)):
        lo, hi = wilson_interval(k, n)
        assert lo <= k / n <= hi


def test_wilson_narrows_as_the_sample_grows():
    width = [np.subtract(*reversed(wilson_interval(int(.4 * n), n))) for n in (50, 500, 5000)]
    assert width[0] > width[1] > width[2]


# --- Binning --------------------------------------------------------------


def test_no_bin_falls_below_the_minimum_count():
    rng = np.random.default_rng(3)
    for n in (200, 1000, 9999):
        p = rng.uniform(0, 1, n)
        idx = quantile_bins(p, n_bins=10, min_count=MIN_BIN_COUNT)
        counts = np.bincount(idx)
        assert (counts[counts > 0] >= MIN_BIN_COUNT).all(), counts


def test_bins_are_equal_count_not_equal_width():
    """A clustered stream is exactly what equal-width binning handles badly."""
    rng = np.random.default_rng(5)
    p = np.clip(rng.normal(0.5, 0.04, 5000), 0, 1)  # everything between ~.35 and ~.65
    counts = np.bincount(quantile_bins(p, n_bins=10, min_count=MIN_BIN_COUNT))
    assert (counts > 0).sum() == 10, "expected ten populated bins"
    assert counts.max() / counts.min() < 1.5, "bins should hold similar counts"


def test_bins_are_ordered_by_probability():
    rng = np.random.default_rng(9)
    p = rng.uniform(0, 1, 3000)
    idx = quantile_bins(p)
    means = [p[idx == b].mean() for b in range(idx.max() + 1)]
    assert means == sorted(means)


def test_tiny_samples_collapse_to_one_bin_rather_than_raising():
    idx = quantile_bins(np.random.default_rng(1).uniform(0, 1, 10))
    assert set(idx) == {0}
    assert quantile_bins(np.array([])).size == 0


# --- Slope ----------------------------------------------------------------


def test_honest_predictions_score_a_slope_of_one():
    y, p = _stream(sharpen=1.0)
    slope, intercept, se = calibration_slope(y, p)
    assert slope == pytest.approx(1.0, abs=0.08)
    assert intercept == pytest.approx(0.0, abs=0.08)
    assert abs(slope - 1) < 2 * se, "an honest stream must sit inside two standard errors"


def test_overconfident_predictions_score_a_slope_below_one():
    y, p = _stream(sharpen=2.0)
    slope, _, se = calibration_slope(y, p)
    assert slope == pytest.approx(0.5, abs=0.08), "slope should recover 1 / sharpen"
    assert 1 - slope > 2 * se, "and be distinguishable from one"


def test_underconfident_predictions_score_a_slope_above_one():
    y, p = _stream(sharpen=0.5)
    slope, _, se = calibration_slope(y, p)
    assert slope - 1 > 2 * se


def test_the_standard_error_shrinks_as_the_sample_grows():
    ses = [calibration_slope(*_stream(n=n, sharpen=1.0))[2] for n in (1_000, 10_000, 100_000)]
    assert ses[0] > ses[1] > ses[2]


def test_a_narrow_spread_of_predictions_reports_a_wide_error():
    """Leverage, not sample size, is what makes a slope trustworthy: a market
    whose probabilities barely move gives the regression almost nothing to fit."""
    rng = np.random.default_rng(2)
    n = 20_000
    wide = np.clip(rng.uniform(0.05, 0.95, n), 0.01, 0.99)
    narrow = np.clip(rng.uniform(0.45, 0.55, n), 0.01, 0.99)
    se_wide = calibration_slope(rng.binomial(1, wide), wide)[2]
    se_narrow = calibration_slope(rng.binomial(1, narrow), narrow)[2]
    assert se_narrow > 5 * se_wide


def test_a_single_outcome_class_gives_nan_rather_than_a_fitted_line():
    assert np.isnan(calibration_slope(np.ones(100, dtype=int), np.full(100, 0.9))[0])


# --- Verdicts -------------------------------------------------------------


def _verdicts(y, p):
    idx = quantile_bins(p, min_count=MIN_BIN_COUNT)
    return _bin_frame(np.asarray(y), np.asarray(p), idx)


def test_honest_predictions_are_called_calibrated():
    """A 95% interval flags roughly one band in twenty by chance, so the claim is
    "almost all", not "all" -- asserting the latter would make this test fail on
    a correct implementation about half the time."""
    bins = _verdicts(*_stream(sharpen=1.0))
    off = (bins["verdict"] != "calibrated").sum()
    assert off <= 1, bins[bins["verdict"] != "calibrated"]
    assert bins["gap"].abs().max() < 0.05


def test_overconfident_predictions_are_flagged_in_both_directions():
    """Overconfidence is not a uniform bias -- it over-predicts high bands and
    under-predicts low ones, which is precisely what a single ECE hides."""
    y, p = _stream(sharpen=2.5)
    bins = _verdicts(y, p)
    assert (bins["verdict"] != "calibrated").any()
    assert (bins.iloc[-1]["verdict"] == "over-predicts")
    assert (bins.iloc[0]["verdict"] == "under-predicts")


def test_summary_reports_the_murphy_decomposition_consistently():
    y, p = _stream(sharpen=1.6)
    idx = quantile_bins(p)
    bins = _bin_frame(y, p, idx)
    s = _summarise(y, p, bins)
    assert s["brier"] == pytest.approx(
        s["brier_reliability"] - s["brier_resolution"] + s["brier_uncertainty"], abs=1e-3
    )


def test_summary_grades_honest_and_overconfident_streams_differently():
    good = _summarise(*_stream(sharpen=1.0), _verdicts(*_stream(sharpen=1.0)))
    bad = _summarise(*_stream(sharpen=3.0), _verdicts(*_stream(sharpen=3.0)))
    assert good["verdict"] == TRUST and good["confidence"] == "well-scaled"
    assert bad["confidence"] == "overconfident"
    assert bad["ece"] > good["ece"]


# --- End to end -----------------------------------------------------------


def _paired(n: int = 600, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mu_h, mu_a = rng.uniform(0.7, 2.4, n), rng.uniform(0.5, 2.0, n)
    return pd.DataFrame({
        "fixture_id": np.arange(n),
        "league_key": rng.choice(["Prem", "Liga"], n),
        "season": "2025/2026",
        "mu_home": mu_h, "mu_away": mu_a,
        "actual_home": rng.poisson(mu_h), "actual_away": rng.poisson(mu_a),
    })


@pytest.mark.parametrize("metric", ["team", "total"])
def test_tables_cover_every_segment_and_carry_their_metric(metric):
    summary, bins = calibration_tables(_paired(), "goals", None, metric=metric, min_count=40)

    assert (summary["metric"] == metric).all(), "a number must never lose its metric"
    assert (bins["metric"] == metric).all()
    assert set(summary["segment"]) >= {"all", "league:Prem", "league:Liga"}
    if metric == "team":
        assert {"venue:home", "venue:away"} <= set(summary["segment"])
    else:
        assert not any(s.startswith("venue:") for s in summary["segment"])

    assert summary["verdict"].notna().all()
    assert (bins.groupby(["segment", "line"])["n"].sum() > 0).all()


def test_poisson_draws_scored_by_a_poisson_model_come_back_calibrated():
    """The outcomes are generated from the very distribution being scored, so
    anything other than a clean bill of health is a bug in the machinery."""
    summary, _ = calibration_tables(_paired(4000), "goals", None, metric="team", min_count=100)
    pooled = summary[summary["segment"] == "all"]
    assert (pooled["ece"] < 0.03).all(), pooled[["line", "ece"]]
    assert (pooled["confidence"] != "overconfident").all()


# --- Verdicts -------------------------------------------------------------


def test_grade_trusts_a_clean_market():
    assert grade(ece_value=0.015, slope=1.01, slope_se=0.05, bands_off=0)[0] == TRUST


def test_grade_avoids_a_high_ece_market():
    assert grade(ece_value=0.08, slope=1.0, slope_se=0.05, bands_off=1)[0] == AVOID


def test_a_severely_extreme_slope_condemns_a_market_on_its_own():
    """The corners match-total case: ECE a mild 0.027, slope 0.32.

    When predictions cluster near the base rate the average error stays small
    while the confident calls -- the only ones worth acting on -- are badly
    wrong. ECE alone waves this through, which is why the slope rule exists.
    """
    verdict, confidence = grade(ece_value=0.027, slope=0.32, slope_se=0.18, bands_off=0)
    assert verdict == AVOID
    assert confidence == "overconfident"


def test_a_noisy_slope_estimate_does_not_condemn_a_market():
    """Same slope, but measured so imprecisely it is within two standard errors
    of one. Leverage, not point estimate, decides whether this is a finding."""
    assert grade(ece_value=0.015, slope=0.60, slope_se=0.45, bands_off=0)[0] == TRUST


def test_grade_is_the_single_definition_used_by_the_pipeline():
    """`line_verdicts` must regrade rather than trust a stored label, so an old
    CSV cannot carry superseded thresholds into a new report."""
    y, p = _stream(sharpen=1.0)
    bins = _verdicts(y, p)
    s = _summarise(y, p, bins)
    assert s["verdict"] == grade(s["ece"], s["slope"], s["slope_se"], s["n_bands_off"])[0]


def test_verdict_table_regrades_a_stale_summary():
    summary, bins = calibration_tables(_paired(800), "goals", None, metric="team", min_count=40)
    stale = summary.copy()
    stale["verdict"] = "good"          # a label from an older threshold set
    stale["confidence"] = "nonsense"

    v = fpp.calibration.line_verdicts(stale, bins)
    assert set(v["verdict"]) <= {TRUST, CAUTION, AVOID, "insufficient"}
    assert "good" not in set(v["verdict"])
    assert "nonsense" not in set(v["confidence"])


def test_every_market_gets_a_readable_sentence():
    summary, bins = calibration_tables(_paired(800), "goals", None, metric="team", min_count=40)
    v = fpp.calibration.line_verdicts(summary, bins)
    assert v["reading"].notna().all()
    assert (v["reading"].str.len() > 20).all()
    # "band 7" is not something anyone can act on; the sentence must talk in
    # probabilities or say the market is clean.
    assert v["reading"].str.contains("%|reliable across").all()


def test_reading_names_the_direction_of_a_deliberate_bias():
    y, p = _stream(n=20_000, sharpen=2.5)
    bins = _verdicts(y, p)
    row = _summarise(y, p, bins)
    text = fpp.calibration._reading(row, bins)
    assert "both ends" in text or ("over-predicts" in text and "under-predicts" in text)


def test_rollup_counts_add_up_to_the_lines_examined():
    summary, bins = calibration_tables(_paired(800), "goals", None, metric="team", min_count=40)
    v = fpp.calibration.line_verdicts(summary, bins)
    roll = fpp.calibration.summarise_by_target(v)
    for _, r in roll.iterrows():
        assert r["trust"] + r["caution"] + r["avoid"] <= r["lines"]
