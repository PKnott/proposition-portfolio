"""Guards on the analyses.

Every test builds a stream whose answer is known in advance and checks the
function recovers it -- the same approach `test_calibration.py` takes with
`sharpen`. Real settled data cannot do this job: on real data an analysis that
is subtly wrong still returns a plausible number, and a plausible number is
exactly what this report exists to be trusted about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpp import analysis as A


def stream(p, y, *, o=None, e=None, target="goals", league="Prem", venue="home",
           line=0.5, run_code="R001"):
    """A settled-proposition frame with the columns the analyses read."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    o = np.full(len(p), 2.0) if o is None else np.asarray(o, dtype=float)
    df = pd.DataFrame({
        "run_code": run_code, "run_date": "2026-08-21",
        "p": p, "y": y, "o": o, "target": target, "league": league,
        "league_key": league, "venue": venue, "scope": venue, "line": line,
    })
    df["market_p"] = 1.0 / df["o"]
    df["e"] = df["p"] * df["o"] if e is None else np.asarray(e, dtype=float)
    df["roi"] = df["o"] * df["y"] - 1.0
    return df


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# --- settled --------------------------------------------------------------


def test_settled_drops_the_rows_that_have_no_result(tmp_path):
    from fpp import ledger

    root = tmp_path / "Analysis"
    props = pd.DataFrame({
        "run_code": ["R001", "R001"], "o": [2.0, 2.0], "p": [0.5, 0.5],
        "won": pd.array([True, None], dtype="boolean"),
        "scope": ["home", "home"], "league_key": ["Prem", "Prem"],
    })
    ledger.write_table(props, "propositions", root)
    out = A.settled(root)
    assert len(out) == 1 and out.iloc[0]["y"] == 1.0
    assert out.iloc[0]["market_p"] == pytest.approx(0.5)
    assert out.iloc[0]["roi"] == pytest.approx(1.0)


# --- 1. model against market ----------------------------------------------


def test_the_model_wins_when_it_knows_and_the_market_does_not(rng):
    """Model probabilities equal to the truth; prices that say 50/50 regardless."""
    truth = rng.uniform(0.05, 0.95, 4000)
    y = (rng.uniform(size=4000) < truth).astype(float)
    s = stream(truth, y, o=np.full(4000, 2.0))
    row = A.model_vs_market(s).iloc[0]
    assert row["model_logloss"] < row["market_logloss"]
    assert row["winner"] == "model" and row["model_beats_market"]


def test_the_market_wins_when_the_price_knows_and_the_model_does_not(rng):
    """The mirror image, and the one a single `significant` flag would hide.

    A boolean that reads False both here and when nothing separates the two is
    reporting the same thing for opposite findings.
    """
    truth = rng.uniform(0.05, 0.95, 4000)
    y = (rng.uniform(size=4000) < truth).astype(float)
    s = stream(np.full(4000, 0.5), y, o=1.0 / truth)
    row = A.model_vs_market(s).iloc[0]
    assert row["market_logloss"] < row["model_logloss"]
    assert row["winner"] == "market" and not row["model_beats_market"]


def test_neither_wins_when_they_are_the_same_numbers(rng):
    truth = rng.uniform(0.05, 0.95, 2000)
    y = (rng.uniform(size=2000) < truth).astype(float)
    s = stream(truth, y, o=1.0 / truth)
    row = A.model_vs_market(s).iloc[0]
    assert row["winner"] == "neither"
    assert row["mean_gain"] == pytest.approx(0.0, abs=1e-9)


def test_model_vs_market_reports_a_row_per_market_and_an_all_row(rng):
    a = stream(rng.uniform(size=300), rng.integers(0, 2, 300), target="goals")
    b = stream(rng.uniform(size=300), rng.integers(0, 2, 300), target="corners")
    out = A.model_vs_market(pd.concat([a, b], ignore_index=True))
    assert set(out["target"]) == {"all", "goals", "corners"}
    assert int(out.loc[out["target"] == "all", "n"].iloc[0]) == 600


# --- 2. calibration -------------------------------------------------------


def test_live_calibration_grades_a_well_calibrated_stream_as_trustworthy(rng):
    truth = rng.uniform(0.05, 0.95, 6000)
    y = (rng.uniform(size=6000) < truth).astype(float)
    summary, bins = A.live_calibration(stream(truth, y), pooled=True)
    row = summary[summary["segment"] == "all"].iloc[0]
    assert row["slope"] == pytest.approx(1.0, abs=0.2)
    assert row["ece"] < 0.05
    assert not bins.empty


def test_live_calibration_catches_an_overconfident_stream(rng):
    """Push every probability away from the middle and the slope must fall."""
    truth = rng.uniform(0.05, 0.95, 6000)
    y = (rng.uniform(size=6000) < truth).astype(float)
    logit = np.log(truth / (1 - truth)) * 2.0            # true slope 1/2
    sharp = 1 / (1 + np.exp(-logit))
    summary, _ = A.live_calibration(stream(sharp, y), pooled=True)
    row = summary[summary["segment"] == "all"].iloc[0]
    assert row["slope"] < 0.75
    assert row["verdict"] != "TRUST"


def test_pooling_collapses_the_ladder_to_one_row_per_market(rng):
    parts = [stream(rng.uniform(size=400), rng.integers(0, 2, 400), line=l)
             for l in (0.5, 1.5, 2.5)]
    s = pd.concat(parts, ignore_index=True)
    pooled, _ = A.live_calibration(s, pooled=True)
    per_line, _ = A.live_calibration(s, pooled=False)
    assert set(pooled["line"]) == {"all"}
    assert set(per_line["line"]) == {0.5, 1.5, 2.5}


# --- 3. edge and odds buckets ---------------------------------------------


def test_edge_buckets_recover_a_return_that_was_put_there():
    """Two buckets, one paying 20% and one losing 20%, both exactly."""
    win = stream([0.6] * 100, [1.0] * 60 + [0.0] * 40, o=np.full(100, 2.0),
                 e=np.full(100, 1.15))
    lose = stream([0.4] * 100, [1.0] * 40 + [0.0] * 60, o=np.full(100, 2.0),
                  e=np.full(100, 0.92))
    out = A.edge_buckets(pd.concat([win, lose], ignore_index=True)).set_index("lo")
    assert out.loc[1.10, "flat_roi"] == pytest.approx(0.20)
    assert out.loc[0.90, "flat_roi"] == pytest.approx(-0.20)
    assert out.loc[1.10, "hit_rate"] == pytest.approx(0.60)


def test_a_bucket_is_only_called_profitable_when_its_interval_clears_zero():
    """Six winners out of ten at evens is +20% and means nothing."""
    s = stream([0.5] * 10, [1.0] * 6 + [0.0] * 4, o=np.full(10, 2.0), e=np.full(10, 1.15))
    row = A.edge_buckets(s).iloc[0]
    assert row["flat_roi"] == pytest.approx(0.2)
    assert not row["profitable"]


def test_odds_buckets_split_on_price_not_on_probability():
    short = stream([0.9] * 50, [1.0] * 50, o=np.full(50, 1.1))
    long_ = stream([0.1] * 50, [0.0] * 50, o=np.full(50, 11.0))
    out = A.odds_buckets(pd.concat([short, long_], ignore_index=True))
    assert list(out["lo"]) == [1.0, 10.0]
    assert out.iloc[0]["flat_roi"] == pytest.approx(0.1)
    assert out.iloc[1]["flat_roi"] == pytest.approx(-1.0)


def test_by_market_reports_the_gap_between_predicted_and_realised():
    s = stream([0.5] * 100, [1.0] * 30 + [0.0] * 70, target="corners")
    out = A.by_market(s)
    row = out[(out["target"] == "corners") & (out["league"] == "all")].iloc[0]
    assert row["mean_p"] == pytest.approx(0.5)
    assert row["hit_rate"] == pytest.approx(0.3)
    assert row["gap"] == pytest.approx(-0.2)


# --- 4. the frontier ------------------------------------------------------


def frontier_frame(n=40, slates=("R001", "R002"), seed=0):
    """A frontier where realised return is a known increasing function of `sd_pct`."""
    rng = np.random.default_rng(seed)
    rows = []
    for run in slates:
        sd = np.linspace(0.05, 0.5, n)
        for i, x in enumerate(sd):
            rows.append({
                "run_code": run, "run_date": "2026-08-21", "portfolio_id": i,
                "split": "inverse_e" if i % 2 else "min_variance",
                "legs": 8 + (i % 5), "expected_return_pct": 1.0 + x,
                "sd_pct": x, "variance": x ** 2,
                "p_over_100": 1.0 - x, "p_over_120": x,
                "g_f_protective": 0.2, "g_g_protective": 0.01,
                "g_f_star": 0.9, "g_g_star": 0.05, "g_p0": 0.001,
                "realised_return": x + rng.normal(0, 0.01),
                "settled": True, "n_settled": 8,
            })
    f = pd.DataFrame(rows)
    f["_sharpe"] = (f["expected_return_pct"] - 1.0) / f["sd_pct"]
    return f


def test_each_strategy_picks_the_portfolio_its_rule_names():
    f = frontier_frame(slates=("R001",))
    picks = A.strategy_returns(f)
    all_split = picks[picks["split"] == "all"].set_index("strategy")
    assert all_split.loc["max expected return", "portfolio_id"] == f["expected_return_pct"].idxmax()
    assert all_split.loc["min variance", "portfolio_id"] == f["variance"].idxmin()
    assert all_split.loc["max P(profit)", "portfolio_id"] == f["p_over_100"].idxmax()


def test_strategies_are_run_inside_each_split_as_well_as_across_them():
    """The split is part of the portfolio, so `1/E` against min-variance is a
    question the table has to be able to answer."""
    picks = A.strategy_returns(frontier_frame())
    assert set(picks["split"]) == {"all", "inverse_e", "min_variance"}


def test_strategy_summary_counts_slates_not_portfolios():
    """Two slates of forty portfolios each is two observations, not eighty."""
    summary = A.strategy_summary(A.strategy_returns(frontier_frame()))
    assert set(summary["n_slates"]) == {2}


def test_strategy_summary_hit_rate_is_the_share_of_slates_in_profit():
    picks = pd.DataFrame({
        "split": "all", "strategy": "x", "run_code": ["R001", "R002", "R003", "R004"],
        "realised_return": [0.5, -0.2, 0.1, -0.9], "expected_return_pct": 1.1,
        "percentile": [0.8, 0.3, 0.6, 0.1],
    })
    row = A.strategy_summary(picks).iloc[0]
    assert row["n_slates"] == 4 and row["slates_profitable"] == 2
    assert row["hit_rate"] == pytest.approx(0.5)
    assert row["worst"] == pytest.approx(-0.9) and row["best"] == pytest.approx(0.5)
    assert row["mean_percentile"] == pytest.approx(0.45)
    assert row["slates_above_median"] == 2


def test_attribution_finds_the_characteristic_the_return_was_built_from():
    out = A.frontier_attribution(frontier_frame()).set_index("characteristic")
    assert out.loc["sd_pct", "spearman_within_slate"] > 0.9
    assert out.loc["p_over_100", "spearman_within_slate"] < -0.9
    assert out.loc["sd_pct", "n_slates"] == 2


def test_growth_check_is_the_log_compounding_it_claims_to_be():
    picks = pd.DataFrame({
        "split": "all", "strategy": "x", "run_code": ["R001", "R002"],
        "realised_return": [0.5, -0.5], "g_f_protective": [0.2, 0.2],
        "g_g_protective": [0.01, 0.01],
    })
    row = A.growth_check(picks).iloc[0]
    expected = np.mean([np.log(1 + 0.2 * 0.5), np.log(1 - 0.2 * 0.5)])
    assert row["realised_g"] == pytest.approx(expected)
    assert row["pot_multiple"] == pytest.approx(1.1 * 0.9)
    assert row["gap"] == pytest.approx(expected - 0.01)
    assert not row["ruined"]


def test_a_round_that_empties_the_pot_is_reported_as_ruin_not_as_a_rate():
    """`log(0)` is not a growth rate, and averaging it gives a number that reads
    like one."""
    picks = pd.DataFrame({
        "split": "all", "strategy": "x", "run_code": ["R001", "R002"],
        "realised_return": [-1.0, 0.5], "g_f_protective": [1.0, 1.0],
        "g_g_protective": [0.01, 0.01],
    })
    row = A.growth_check(picks).iloc[0]
    assert row["ruined"] and np.isnan(row["realised_g"]) and row["pot_multiple"] == 0.0


# --- empty input ----------------------------------------------------------


@pytest.mark.parametrize("fn", [A.model_vs_market, A.edge_buckets, A.odds_buckets,
                                A.by_market, A.strategy_returns, A.strategy_summary,
                                A.frontier_attribution, A.growth_check])
def test_nothing_settled_yet_returns_an_empty_frame_rather_than_raising(fn):
    """The first run of 07 happens before any fixture has been played."""
    assert fn(pd.DataFrame()).empty


# --- frontier distribution -------------------------------------------------


def test_the_distribution_reports_the_spread_selection_was_worth():
    f = frontier_frame(slates=("R001",))
    f.loc[0, "realised_return"] = -1.0
    f.loc[1, "realised_return"] = 1.75
    d = A.frontier_distribution(f)
    row = d[d["run_code"] == "R001"].iloc[0]
    assert row["min"] == pytest.approx(-1.0)
    assert row["max"] == pytest.approx(1.75)
    assert row["spread"] == pytest.approx(2.75)
    assert row["share_wiped_out"] == pytest.approx(1 / len(f))


def test_the_histogram_uses_fixed_bands_so_two_slates_line_up():
    a = frontier_frame(slates=("R001",))
    b = frontier_frame(slates=("R002",), seed=7)
    h = A.frontier_histogram(pd.concat([a, b], ignore_index=True))
    bands_a = set(h[h["run_code"] == "R001"]["band"])
    bands_b = set(h[h["run_code"] == "R002"]["band"])
    assert bands_a == bands_b
    for run in ("R001", "R002"):
        assert h[h["run_code"] == run]["n"].sum() == len(a)


def test_percentile_is_the_share_of_that_slate_it_beat():
    f = frontier_frame(slates=("R001",))
    r = f["realised_return"]
    assert A.percentile_of(f, "R001", r.max()) == pytest.approx(1 - 1 / len(f))
    assert A.percentile_of(f, "R001", r.min()) == pytest.approx(0.0)
    assert A.percentile_of(f, "R001", r.median()) == pytest.approx(0.5, abs=0.03)


def test_percentile_is_measured_per_slate_not_pooled():
    """A good slate lifts every return on it, so a pooled rank would say a rule
    picked badly simply because that Saturday was quiet."""
    good = frontier_frame(slates=("R001",))
    poor = frontier_frame(slates=("R002",))
    poor["realised_return"] = poor["realised_return"] - 1.0
    f = pd.concat([good, poor], ignore_index=True)
    best_of_poor = poor["realised_return"].max()
    assert A.percentile_of(f, "R002", best_of_poor) > 0.9      # top of its own slate
    assert A.percentile_of(f, "R001", best_of_poor) == 0.0     # bottom of the other


def test_placed_rank_says_where_the_bet_landed():
    f = frontier_frame(slates=("R001",))
    target = f.iloc[5]
    placed = pd.DataFrame([{"run_code": "R001", "portfolio_id": int(target["portfolio_id"]),
                            "stake": 100.0, "pot": 500.0, "mode": "protective", "f": 0.2}])
    row = A.placed_rank(f, placed).iloc[0]
    assert row["ref"] == f"R001-{int(target['portfolio_id'])}"
    assert row["realised"] == pytest.approx(target["realised_return"])
    assert row["percentile"] == pytest.approx(A.percentile_of(f, "R001", target["realised_return"]))
    assert row["pnl"] == pytest.approx(100.0 * target["realised_return"])


# --- selection rules -------------------------------------------------------


def test_a_quantile_rule_picks_a_portfolio_that_exists():
    """Not the interpolated value -- the answer has to be backable."""
    f = frontier_frame(slates=("R001",))
    rule = A.Rule("expected_return_pct", quantile=0.75)
    pick = rule.pick(f)
    assert pick["portfolio_id"] in set(f["portfolio_id"])
    target = f["expected_return_pct"].quantile(0.75)
    nearest = (f["expected_return_pct"] - target).abs().min()
    assert abs(pick["expected_return_pct"] - target) == pytest.approx(nearest)


def test_sense_picks_the_max_or_the_min():
    f = frontier_frame(slates=("R001",))
    assert A.Rule("sd_pct", +1).pick(f)["sd_pct"] == f["sd_pct"].max()
    assert A.Rule("sd_pct", -1).pick(f)["sd_pct"] == f["sd_pct"].min()


def test_a_rule_on_an_all_null_column_picks_nothing_rather_than_raising():
    f = frontier_frame(slates=("R001",))
    f["median_leg_p"] = np.nan
    assert A.Rule("median_leg_p", +1).pick(f) is None


def test_the_summary_ranks_on_percentile_not_on_return():
    """The whole point of the rewrite: rank leads, return follows."""
    picks = pd.DataFrame({
        "split": "all", "strategy": ["good", "bad"], "run_code": ["R001", "R001"],
        "percentile": [0.9, 0.2], "realised_return": [0.10, 0.30],
        "expected_return_pct": [1.1, 1.1],
    })
    out = A.strategy_summary(picks)
    assert list(out["strategy"]) == ["good", "bad"]       # despite "bad" returning more


# --- deciles ---------------------------------------------------------------


def test_deciles_expose_the_skew_a_correlation_hides():
    """A characteristic that ranks negatively can still own the best outcome.

    Built so the top decile has the worst median and the best maximum -- exactly
    the shape that makes a bare Spearman misleading.
    """
    n = 500
    rng = np.random.default_rng(1)
    x = np.linspace(0, 1, n)
    y = np.where(rng.uniform(size=n) < x * 0.1, 5.0, -x)     # rare huge win, usual loss
    f = pd.DataFrame({"run_code": "R001", "portfolio_id": range(n),
                      "expected_return_pct": x, "realised_return": y, "split": "all"})
    d = A.characteristic_deciles(f, ["expected_return_pct"])
    top = d[d["decile"] == 10].iloc[0]
    bottom = d[d["decile"] == 1].iloc[0]
    assert top["median_return"] < bottom["median_return"]    # ranks badly
    assert top["best"] > bottom["best"]                      # owns the tail
    assert f["expected_return_pct"].corr(f["realised_return"], method="spearman") < 0


# --- directional calibration ----------------------------------------------


def test_calibration_separates_the_safe_direction_from_the_costly_one(rng):
    """Every proposition here is an Over, so under-predicting is error in your
    favour and over-predicting is the one that costs money."""
    truth = rng.uniform(0.1, 0.8, 4000)
    y = (rng.uniform(size=4000) < truth).astype(float)
    under, _ = A.live_calibration(stream(truth - 0.08, y), pooled=True)
    over, _ = A.live_calibration(stream(np.clip(truth + 0.08, 0, 1), y), pooled=True)

    u = under[under["segment"] == "all"].iloc[0]
    o = over[over["segment"] == "all"].iloc[0]
    assert u["mean_gap"] > 0 and u["net_direction"] == "under (safe)"
    assert u["ece_under"] > u["ece_over"]
    assert o["mean_gap"] < 0 and o["net_direction"] == "over (costly)"
    assert o["ece_over"] > o["ece_under"]


def test_the_symmetric_verdict_is_left_alone_for_comparability(rng):
    """`04_Evaluation` computes the symmetric one; it has to keep meaning the
    same thing here or the two reports cannot be read together."""
    truth = rng.uniform(0.1, 0.8, 3000)
    y = (rng.uniform(size=3000) < truth).astype(float)
    su, _ = A.live_calibration(stream(truth - 0.08, y), pooled=True)
    row = su[su["segment"] == "all"].iloc[0]
    assert {"verdict", "verdict_over"} <= set(su.columns)
    assert row["ece"] >= max(row["ece_under"], row["ece_over"])


# --- the figure marks the right slate ---------------------------------------


def two_slate_frames():
    """Two slates with very different medians, and a bet placed on each."""
    a = frontier_frame(slates=("R001",))
    b = frontier_frame(slates=("R002",), seed=3)
    b["realised_return"] = b["realised_return"] + 0.55        # a much better slate
    f = pd.concat([a, b], ignore_index=True)
    placed = pd.DataFrame([
        {"run_code": "R001", "portfolio_id": int(a.iloc[0]["portfolio_id"]),
         "stake": 100.0, "pot": 500.0, "mode": "protective", "f": 0.2},
        {"run_code": "R002", "portfolio_id": int(b.iloc[0]["portfolio_id"]),
         "stake": 100.0, "pot": 500.0, "mode": "protective", "f": 0.2},
    ])
    return f, placed


def test_the_figure_marks_the_bet_from_the_slate_it_is_drawing():
    """The bug this guards: the marker was `placed.iloc[0]`, so a chart of the
    newest slate kept reporting the first bet ever placed -- an old result drawn
    against a new distribution."""
    from fpp.report.figures import frontier_return_figure

    f, placed = two_slate_frames()
    hist, dist, pr = (A.frontier_histogram(f), A.frontier_distribution(f),
                      A.placed_rank(f, placed))
    fig = frontier_return_figure(hist, dist, pr)            # defaults to newest
    txt = " ".join(t.get_text() for t in fig.axes[0].texts) + fig.axes[0].get_title()
    want = pr[pr.run_code == "R002"]["realised"].iloc[0]
    other = pr[pr.run_code == "R001"]["realised"].iloc[0]
    assert f"{want:+.1%}" in txt, txt
    assert f"{other:+.1%}" not in txt, txt
    assert "R002" in txt


def test_the_figure_titles_the_slate_it_is_drawing():
    from fpp.report.figures import frontier_return_figure

    f, placed = two_slate_frames()
    hist, dist = A.frontier_histogram(f), A.frontier_distribution(f)
    for rc in ("R001", "R002"):
        fig = frontier_return_figure(hist, dist, A.placed_rank(f, placed), run_code=rc)
        assert rc in fig.axes[0].get_title()


def test_one_figure_is_written_per_slate(tmp_path):
    f, placed = two_slate_frames()
    frames = {"frontier_histogram": A.frontier_histogram(f),
              "frontier_distribution": A.frontier_distribution(f),
              "placed_rank": A.placed_rank(f, placed)}
    names = {p.name for p in A.write_reports(frames, tmp_path)}
    assert {"frontier_returns_R001.png", "frontier_returns_R002.png",
            "frontier_returns_all.png"} <= names
