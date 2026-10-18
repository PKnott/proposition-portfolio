"""`fpp.study` on synthetic data whose answers are known in advance."""

import numpy as np
import pandas as pd
import pytest

from fpp import study as st


def _slate(n_matches=400, rho=0.5, seed=0, bias=0.0):
    """Two propositions per match, same team and stat, with a Gaussian-copula correlation."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_matches):
        p = rng.uniform(0.2, 0.7, size=2)
        zc = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]])
        from scipy.stats import norm
        won = norm.cdf(zc) < np.clip(p + bias, 0.01, 0.99)
        for k in range(2):
            rows.append({"event": f"m{m}", "run_code": f"R{m % 8:03d}", "team": "A",
                         "target": "shots", "line": 10.5 + k, "p": p[k], "y": float(won[k]),
                         "qualified": True, "market_p": p[k]})
    return pd.DataFrame(rows)


def test_pair_correlation_recovers_a_known_rho():
    s = _slate(n_matches=4000, rho=0.6)
    t = st.rho_table(st.pair_frame(s), "relation")
    # A latent correlation of 0.6 gives a smaller correlation between the binary
    # outcomes, but it must be clearly positive and inside the adjacent class.
    assert list(t["relation"]) == ["same_team_same_stat_adjacent"]
    assert 0.25 < t["rho"].iloc[0] < 0.6
    assert t["lo"].iloc[0] > 0


def test_independent_outcomes_and_the_placebo_sit_at_zero():
    s = _slate(n_matches=4000, rho=0.0, seed=1)
    t = st.rho_table(st.pair_frame(s), "relation")
    assert t["lo"].iloc[0] < 0 < t["hi"].iloc[0]
    pl = st.placebo_correlation(s)
    assert abs(pl["rho"]) < 3 * pl["se"] + 1e-9


def test_relation_classes_match_portfolio_pair_rho():
    got = st.relation([False, True, True, True, True], [True, False, True, True, True],
                      [0, 0, 1, 2, 3])
    assert list(got) == ["opposite_teams", "same_team_diff_stat",
                         "same_team_same_stat_adjacent", "same_team_same_stat_near",
                         "same_team_same_stat_far"]


def test_a_calibrated_stream_needs_no_adjustment():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.05, 0.9, 20_000)
    y = (rng.uniform(size=p.size) < p).astype(float)
    a, b = st.fit_recalibration(y, p)[:2]
    assert abs(a) < 0.06 and abs(b - 1) < 0.06


def test_an_under_predicting_stream_is_shifted_up():
    rng = np.random.default_rng(4)
    p = rng.uniform(0.05, 0.8, 20_000)
    y = (rng.uniform(size=p.size) < np.clip(p + 0.06, 0, 1)).astype(float)
    a, _ = st.fit_recalibration(y, p)[:2]
    assert a > 0.1


def test_ridge_pulls_a_context_back_to_the_global_fit():
    rng = np.random.default_rng(5)
    p = rng.uniform(0.1, 0.8, 4000)
    lv = rng.integers(0, 2, p.size)
    y = (rng.uniform(size=p.size) < np.where(lv == 1, np.clip(p + 0.1, 0, 1), p)).astype(float)
    free = st.fit_recalibration(y, p, lv, ridge=0.0)
    tight = st.fit_recalibration(y, p, lv, ridge=1e6)
    assert np.abs(tight[2:]).max() < 1e-2 < np.abs(free[2:]).max()


def test_loso_never_scores_a_slate_with_a_fit_that_saw_it(monkeypatch):
    s = _slate(n_matches=240, seed=6)
    s["target"] = np.where(s.index % 2, "shots", "sot")
    seen: list[set] = []
    real = st.fit_recalibration

    def spy(y, p, levels=None, **kw):
        seen.append(len(y))
        return real(y, p, levels, **kw)

    monkeypatch.setattr(st, "fit_recalibration", spy)
    cv = st.loso_predictions(s, contexts=("target",))
    n_slate = s.groupby("run_code").size()
    # Every outer fit is on the data minus exactly one slate, never all of it.
    assert max(seen) < len(s)
    assert set(len(s) - n_slate) <= set(seen)
    for c in ("global", "blend", "ctx:target"):
        assert cv[c].notna().all()


def test_grade_keeps_banked_results_and_settles_the_rest():
    props = pd.DataFrame({
        "fixture_date": pd.to_datetime(["2026-10-11", "2026-10-11"]),
        "league_key": ["Prem", "Prem"], "home_team": ["A", "A"], "away_team": ["B", "B"],
        "scope": ["home", "away"], "target": ["corners", "corners"], "line": [4.5, 4.5],
        "won": pd.array([True, pd.NA], dtype="boolean"), "actual": [9.0, np.nan],
    })
    results = pd.DataFrame({
        "league_key": ["Prem"], "date": pd.to_datetime(["2026-10-11"]),
        "home_team": ["A"], "away_team": ["B"], "home_espn": ["A"], "away_espn": ["B"],
        **{f"{side}_{t}": [0.0] for side in ("home", "away") for t in st.TARGETS},
    })
    results["away_corners"] = 6.0
    g = st.grade(props, results)
    assert g["won"].tolist() == [True, True]
    assert g["actual"].tolist() == [9.0, 6.0]


def test_select_mirrors_qualify_edge_and_frontier():
    df = pd.DataFrame({"event": ["m", "m", "m", "n"], "o": [3.0, 2.5, 1.5, 2.0]})
    p = np.array([0.40, 0.30, 0.50, 0.45])
    # e = 1.2, 0.75, 0.75, 0.9: only the first clears 1.0, and it is alone on its match.
    assert st.select(df, p).tolist() == [True, False, False, False]
    # Two that clear, one dominated on both p and o by the other.
    p2 = np.array([0.40, 0.42, 0.70, 0.45])
    df2 = df.assign(o=[2.7, 2.8, 1.5, 2.0])
    assert st.select(df2, p2).tolist() == [False, True, True, False]


def test_slate_growth_never_risks_more_than_the_cap():
    f = np.full(10, 0.1)               # 100% of the pot asked for
    lost = np.full(10, -1.0)
    assert st.slate_growth(f, lost, cap=0.2) == pytest.approx(np.log(0.8))


def test_deflation_is_learnt_without_the_scored_slate():
    rng = np.random.default_rng(7)
    rows = []
    for sl in range(6):
        for m in range(300):
            truth = rng.uniform(0.25, 0.65)
            p = np.clip(truth - 0.05 + rng.normal(0, 0.06), 0.02, 0.95)  # low by 5pp, noisy
            o = 1.0 / np.clip(truth + 0.02, 0.05, 0.95)                    # a book with margin
            rows.append({"run_code": f"R{sl}", "event": f"{sl}-{m}", "p": p, "o": o,
                         "y": float(rng.uniform() < truth), "target": "goals"})
    s = pd.DataFrame(rows)
    per, tune = st.push_deflate_backtest(s, ks=(0.0, 1.0))
    assert len(tune) == 6 and set(per["slate"]) == set(s["run_code"])
    # Once the push has fixed the level, selecting on a noisy p still over-states
    # the selected set, so the deflation learnt at k=1 is positive in every fold.
    assert (tune["d@1"] > 0.01).all()
    summ = st.backtest_summary(per)
    assert {"k=0 push only", "k=1 push+deflate", "tuned"} <= set(summ["strategy"])


def test_replay_slate_settles_the_portfolio_it_picks():
    rows = []
    for m, (home, away) in enumerate([("Arsenal", "Chelsea"), ("Everton", "Fulham"),
                                      ("Brentford", "Burnley")]):
        for k, (target, line, p, o, y) in enumerate([("corners", 4.5, 0.55, 2.2, 1.0),
                                                     ("goals", 1.5, 0.40, 2.9, 0.0)]):
            rows.append({"sheet_code": f"E0-{m + 1:02d}", "label": f"{target.title()} - {home} - Over {line}",
                         "p": p, "o": o, "y": y, "target": target,
                         "fixture_date": pd.Timestamp("2026-10-11"), "league": "Prem",
                         "home_team": home, "away_team": away, "event": f"m{m}"})
    s = pd.DataFrame(rows)
    out = st.replay_slate(s, s["p"].to_numpy(), {"none": np.zeros(len(s)),
                                                 "big": np.full(len(s), 0.2)},
                          leg_var=0, grid=512)
    none, big = out
    assert none["legs"] >= 3 and none["R"] == big["R"]
    # Every corners leg won and every goals leg lost, so R is the corners pay-out.
    assert 0 < none["R"] < 2.2 * 2
    assert none["growth"] == pytest.approx(np.log1p(none["f"] * (none["R"] - 1)))
    # A 20pp deflation kills the edge, so nothing is staked.
    assert big["f"] < none["f"]


def test_model_stamp_is_flat_and_fingerprints_content(tmp_path, monkeypatch):
    from fpp import artifacts

    stamp = artifacts.model_stamp()
    assert set(stamp) == {"model_version", "model_fingerprint", "spec_hash", "scoring_hash",
                          "git_sha", "git_dirty", "shots_shift", "shots_shift_candidate"}
    assert all(not isinstance(v, (dict, list)) for v in stamp.values())
    # Same version, same files -> same fingerprint; change one byte -> different.
    monkeypatch.setattr(artifacts, "ARTIFACTS", tmp_path)
    for t in artifacts.TARGETS:
        for s in artifacts.BEHAVIOUR_STAGES:
            d = tmp_path / t / "v1"; d.mkdir(parents=True, exist_ok=True)
            (d / f"{s}.json").write_text("{}")
    a = artifacts.artifact_fingerprint("v1")
    assert a == artifacts.artifact_fingerprint("v1")
    (tmp_path / artifacts.TARGETS[0] / "v1" / "params.json").write_text('{"x": 1}')
    assert artifacts.artifact_fingerprint("v1") != a


def test_odds_form_round_trips_the_model_stamp(tmp_path):
    from fpp.report.odds_form import form_meta, write_odds_form

    props = pd.DataFrame({
        "sheet_code": ["E0-01"], "date": [pd.Timestamp("2026-10-18")], "league": ["Prem"],
        "home_team": ["Arsenal"], "away_team": ["Chelsea"],
        "label": ["Corners - Arsenal - Over 4.5"], "p": [0.55]})
    stamp = {"model_version": "v9", "model_fingerprint": "abc123", "spec_hash": "s",
             "scoring_hash": "h", "git_sha": "deadbee", "git_dirty": False}
    path = write_odds_form(props, out_path=tmp_path / "odds_input_2026-10-18.xlsx",
                           source="predictions_2026-10-18.xlsx", stamp=stamp)
    meta = form_meta(path)
    assert meta["model"]["model_version"] == "v9"
    assert meta["model"]["model_fingerprint"] == "abc123"
    assert meta["source"] == "predictions_2026-10-18.xlsx" and meta["fixtures"] == 1


def test_rule_search_finds_a_planted_band_and_holdout_scores_out_of_sample():
    """A measure whose 60-80% band holds the winners on every slate is found."""
    rng = np.random.default_rng(11)
    rows = []
    for s in range(6):
        x, z, noise = rng.uniform(size=(3, 400))      # z is unrelated to the outcome
        R = np.where((x > np.quantile(x, 0.6)) & (x <= np.quantile(x, 0.8)), 2.0, 1.0) + 0.1 * noise
        rows.append(pd.DataFrame({"slate": f"S{s}", "x": x, "z": z, "R": R}))
    A = pd.concat(rows, ignore_index=True)
    A["R_pct"] = A.groupby("slate")["R"].rank(pct=True)
    S = st.rule_search(A, measures=("x", "z"))
    mean = S.mean(axis=1)
    singles = mean[mean.index.get_level_values("m2") == ""]
    best = singles.idxmax()
    assert best[0] == "x" and (best[1], best[2]) == (60, 80)
    # Banding on the unrelated measure as well cannot add much.
    assert mean.max() <= singles.max() + 0.05
    H = st.rule_holdout(S[S.notna().all(axis=1)])
    assert len(H) == 6 and (H["held_out_score"] > 0.8).all()
    assert st.describe_rule(best) == "x 60-80%"
