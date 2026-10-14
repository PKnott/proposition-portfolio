"""Triples (`config.TRIPLES_ENABLED`): the rulebook, and that width-3 blocks price exactly."""

import itertools

import numpy as np
import pandas as pd
import pytest

from fpp import portfolio as pf
from fpp.growth import ruin_prob, sweep_prob


def test_block_joint_matches_joint_both_for_pairs():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.1, 0.8, (50, 2))
    r = rng.uniform(-0.2, 0.4, (50, 1))
    J = pf.block_joint(p, r)
    both = pf.joint_both(p[:, 0], p[:, 1], r[:, 0])
    # Where the requested correlation is attainable the two agree exactly. Past
    # the Frechet bounds `joint_both` clamps and `block_joint` clips and
    # renormalises -- different repairs of an impossible request.
    raw = p[:, 0] * p[:, 1] + r[:, 0] * np.sqrt(p[:, 0] * (1 - p[:, 0]) * p[:, 1] * (1 - p[:, 1]))
    ok = (raw >= np.maximum(0, p.sum(axis=1) - 1)) & (raw <= p.min(axis=1))
    assert ok.mean() > 0.9
    np.testing.assert_allclose(J[ok, 3], both[ok], atol=1e-12)
    np.testing.assert_allclose(J.sum(axis=1), 1.0)


def test_block_joint_has_the_requested_marginals_and_correlations():
    p = np.array([[0.3, 0.55, 0.4]])
    r = np.array([[0.17, 0.0, 0.05]])
    J = pf.block_joint(p, r)[0]
    bits = (np.arange(8)[:, None] >> np.arange(3)) & 1
    marg = (J[:, None] * bits).sum(axis=0)
    np.testing.assert_allclose(marg, p[0], atol=1e-12)
    for k, (i, j) in enumerate(pf.BLOCK_PAIRS):
        both = (J * bits[:, i] * bits[:, j]).sum()
        rho = (both - p[0, i] * p[0, j]) / np.sqrt(p[0, i] * (1 - p[0, i]) * p[0, j] * (1 - p[0, j]))
        assert rho == pytest.approx(r[0, k], abs=1e-12)


def test_block_terms_reduce_to_pair_terms():
    rng = np.random.default_rng(1)
    p1, p2 = rng.uniform(0.2, 0.6, 40), rng.uniform(0.2, 0.6, 40)
    o1, o2 = 1.1 / p1, 1.15 / p2
    r = rng.uniform(0, 0.5, 40)
    a = pf.pair_terms(p1, o1, p2, o2, r)
    b = pf.block_terms(np.c_[p1, p2], np.c_[o1, o2], r[:, None])
    np.testing.assert_allclose(b["c"], a["c"], rtol=1e-10)
    np.testing.assert_allclose(b["w"], a["w"], rtol=1e-10)
    # And an empty third slot changes nothing.
    c = pf.block_terms(np.c_[p1, p2, 0 * p1], np.c_[o1, o2, 0 * o1], np.c_[r, 0 * r, 0 * r])
    np.testing.assert_allclose(c["c"], a["c"], rtol=1e-10)


def _event(rows):
    df = pd.DataFrame(rows, columns=["team", "target", "line", "p", "o"])
    return df.assign(sheet_code="E0-01", label=[f"{t} - {tm} - Over {l}" for tm, t, l, *_ in rows])


def test_triples_follow_the_rulebook():
    g = _event([("A", "goals", 1.5, 0.45, 2.5), ("A", "goals", 2.5, 0.30, 3.8),
                ("A", "corners", 4.5, 0.55, 2.0), ("A", "sot", 4.5, 0.40, 2.7),
                ("B", "goals", 1.5, 0.40, 2.7), ("B", "corners", 3.5, 0.50, 2.2)])
    tri = pf._triple_candidates(g, keep=100)
    assert tri
    for a, b, c, *_ in tri:
        assert g.team[a] == g.team[b] != g.team[c]          # two on one team, one on the other
        assert g.target[a] != g.target[b]                   # never same team and stat
    # SOT with goals on the same team is allowed (priced, not banned).
    pairs = {frozenset((g.target[a], g.target[b])) for a, b, *_ in tri}
    assert frozenset(("goals", "sot")) in pairs or frozenset(("corners", "sot")) in pairs


def test_ruin_and_sweep_at_width_three_match_enumeration():
    p = np.array([[0.3, 0.5, 0.4, 0.6, 0.0, 0.0]])
    rho = np.array([[[0.17, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    J = pf.block_joint(p.reshape(1, 2, 3), rho)[0]
    assert ruin_prob(p, rho)[0] == pytest.approx(J[0, 0] * (1 - 0.6))
    assert sweep_prob(p, rho)[0] == pytest.approx(J[0, 7] * 0.6)


def test_return_pmf_width_three_matches_brute_force():
    pay = np.array([[0.5, 0.7, 0.9, 1.1, 0.0, 0.0]])
    p = np.array([[0.3, 0.5, 0.4, 0.6, 0.0, 0.0]])
    rho = np.array([[[0.17, 0.0, 0.05], [0.0, 0.0, 0.0]]])
    dist, step = pf.return_pmf(pay, p, 4096, rho=rho, close_tail=True)
    values = np.arange(4096) * step[0]
    J = pf.block_joint(p.reshape(1, 2, 3), rho)[0]
    # Brute force: block 1's 8 outcomes x block 2's single leg.
    exp_mean = 0.0
    for o1, o2 in itertools.product(range(8), range(2)):
        x = [(o1 >> i) & 1 for i in range(3)]
        prob = J[0, o1] * (0.6 if o2 else 0.4)
        exp_mean += prob * (sum(a * b for a, b in zip(x, pay[0, :3])) + o2 * 1.1)
    assert (dist[0] * values).sum() == pytest.approx(exp_mean, rel=2e-3)
    assert dist.sum() == pytest.approx(1.0)


def test_triples_off_leaves_options_at_width_two():
    q = _event([("A", "goals", 1.5, 0.45, 2.5), ("A", "corners", 4.5, 0.55, 2.0),
                ("B", "goals", 1.5, 0.40, 2.7)])
    assert pf.event_options(q, triples=False).width == 2
    assert pf.event_options(q, triples=True).width == 3


def _triple_slate():
    from fpp.staking import BOOK_COLUMNS

    rows = []
    for m, (h, a) in enumerate([("Arsenal", "Chelsea"), ("Everton", "Fulham")]):
        for team, target, line, p, o in [(h, "Goals", 1.5, 0.45, 2.6), (h, "Corners", 4.5, 0.55, 2.1),
                                         (a, "Goals", 0.5, 0.62, 1.8), (a, "Shots on Target", 3.5, 0.5, 2.3)]:
            rows.append({"sheet_code": f"E0-{m + 1:02d}", "label": f"{target} - {team} - Over {line}",
                         "p": p, "date": pd.Timestamp("2026-10-11"), "league": "Prem",
                         "home_team": h, "away_team": a})
    filled = pd.DataFrame(rows)
    for c in BOOK_COLUMNS:
        filled[c] = np.nan
    filled[BOOK_COLUMNS[0]] = [2.6, 2.1, 1.8, 2.3] * 2
    return filled


def test_search_runs_end_to_end_with_triples():
    filled = _triple_slate()
    res = pf.search(filled, leg_var=0, grid=512, triples=True)
    assert res["options"].width == 3
    sc = res["scored"]
    assert len(sc) and sc["undominated"].any()
    # Stakes still sum to one per portfolio, and legs expand back to propositions.
    stakes, p, o, rho = pf.stakes_for(res["picks"], res["options"], sc["split"].iloc[0])
    np.testing.assert_allclose(stakes.sum(axis=1), 1.0)
    best = sc.iloc[0]
    legs = pf.legs(res["picks"], res["options"], int(best["combo"]), best["split"])
    assert len(legs) == int(best["legs"])


def test_triples_survive_the_edge_book_payload_and_ledger_capture(tmp_path):
    """The two exports downstream of the search, run on a slate that holds a triple."""
    import json

    from fpp import ledger
    from fpp.report.edge_book import write_portfolios_json

    filled = _triple_slate()
    res = pf.search(filled, leg_var=0, grid=512, triples=True)
    assert res["options"].width == 3
    sc = res["scored"]
    # Make sure an exported portfolio actually holds a triple.
    triple_rows = [i for i, c in enumerate(sc["combo"]) if any(
        len(res["options"].members[j][int(k)]) == 3
        for j, k in enumerate(res["picks"][c]) if k >= 0)]
    assert any(sc["undominated"].iloc[i] for i in triple_rows)

    out = write_portfolios_json(res, filled, out_path=tmp_path / "portfolios_test.json",
                                source="odds_input_test.xlsx", run_code="R999")
    payload = json.loads(out.read_text())
    ports = payload["portfolios"]
    assert ports
    for port in ports:
        # One stake per proposition, summing to the whole stake.
        assert len(port["stakes"]) == port["legs"]
        assert abs(sum(port["stakes"]) - 1.0) < 1e-4
    # At least one exported portfolio holds three propositions from one match.
    def most_per_match(port):
        codes = [tok.split("#")[0] for tok in port["picks"].split()] if isinstance(port["picks"], str) \
            else [str(x).split("#")[0] for x in port["picks"]]
        return max(codes.count(c) for c in set(codes))
    assert any(most_per_match(port) == 3 for port in ports)

    slate = ledger.capture(res, filled, source="odds_input_test.xlsx", run_code="R999",
                           root=tmp_path / "Analysis")
    legs = pd.read_parquet(slate.path / "legs.parquet")
    run = json.loads((slate.path / "run.json").read_text())
    assert run["run_code"] == "R999" and run["stamp_origin"] == "capture"
    assert run["model_version"]
    # Every exported portfolio arrives in the ledger with all its legs, and the
    # triples arrive as three legs from one fixture.
    n_legs = legs.groupby("portfolio_id").size()
    assert dict(n_legs) == {p["id"]: p["legs"] for p in ports}
    fixture = legs["prop_key"].str.rsplit("|", n=3).str[0]
    assert legs.assign(fixture=fixture).groupby(["portfolio_id", "fixture"]).size().max() == 3
    np.testing.assert_allclose(legs.groupby("portfolio_id")["stake_frac"].sum(), 1.0, atol=1e-4)
