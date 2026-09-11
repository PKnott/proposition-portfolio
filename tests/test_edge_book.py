"""Guards on the Edge Book export.

The workbook writer's risk was a formula pointing at the wrong cell. This
writer's risk is a **join that does not join**. Two files leave the pipeline from
two different notebooks and are stitched back together in the browser on two
keys -- a proposition id shared between `picks` and `propositions`, and a
proposition *label* shared between a fixture's model lines and the prices
captured against them. Neither is checked by anything at runtime: a drifted label
matches nothing and renders a page that looks complete and quietly shows no
prices at all.

So the tests below build both payloads and then join them, rather than asserting
that keys exist.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from fpp import portfolio as pf
from fpp import staking
from fpp.report import edge_book as eb


# --- fixtures ---------------------------------------------------------------


@pytest.fixture(scope="module")
def filled() -> pd.DataFrame:
    """A synthetic filled odds form: nine events, one to three lines each.

    Seeded, and priced off the model's own probability by a random margin
    straddling 1.0, so some rows clear `E >= 1` and some do not -- which is what
    makes `prices` and `propositions` differ, and therefore what makes the
    negative-edge path reachable at all.
    """
    rng = np.random.default_rng(4)
    rows = []
    n = 0
    for j in range(9):
        for i in range(int(rng.integers(1, 4))):
            p = float(rng.uniform(0.25, 0.85))
            # Every third row is priced *below* the model on purpose rather than
            # by luck of the seed: a margin band straddling 1.0 can draw entirely
            # above it on seventeen samples, and the negative-edge path is exactly
            # what several of these tests are here to reach.
            margin = 0.94 if n % 3 == 2 else float(rng.uniform(1.03, 1.25))
            n += 1
            rows.append({
                "sheet_code": f"EV-{j:02d}",
                "label": staking.proposition_label("corners", f"Team {j}", 0.5 + i),
                "p": p,
                "b365": (1.0 / p) * margin,
                "home_team": f"Home {j}", "away_team": f"Away {j}",
                "team": f"Team {j}", "target": "corners", "line": 0.5 + i,
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def result(filled):
    return pf.search(filled, min_legs=1)


@pytest.fixture(scope="module")
def payload(result, filled) -> dict:
    return eb.portfolios_payload(result, filled, source="test.xlsx")


# --- the id join ------------------------------------------------------------


def test_every_pick_resolves_to_a_proposition(payload):
    """The join the whole Portfolio Book rests on: legs are ids, looked up."""
    known = {p["id"] for p in payload["propositions"]}
    unknown = {pid for pf_ in payload["portfolios"] for pid in pf_["picks"]} - known
    assert not unknown, f"picks reference ids with no proposition: {sorted(unknown)[:5]}"


def test_pick_ids_match_the_label_the_search_writes(result, payload):
    """`{event}#{option}` here is the token `picks_label` already emits.

    Promoting it to an explicit field is only safe while the two agree; if
    `picks_label` ever changed shape, every portfolio would silently lose its
    legs rather than fail.
    """
    opts = result["options"]
    # An option covers one proposition or two, and the token names the
    # *proposition* -- which is what makes it resolve against `propositions`.
    expected = {f"{opts.codes[j]}#{prop}"
                for row in result["picks"] for j, k in enumerate(row) if k >= 0
                for prop in opts.members[j][int(k)]}
    assert {p["id"] for p in payload["propositions"]} >= expected


def test_picks_and_stakes_line_up(payload):
    """One stake per leg, in the same order, summing to the whole stake."""
    for pf_ in payload["portfolios"]:
        # `legs` counts propositions held, which is what both of these list --
        # a portfolio taking a pair from one match has more legs than events.
        assert len(pf_["stakes"]) == len(pf_["picks"]) == pf_["legs"]
        assert pf_["legs"] >= pf_["events"]
        assert sum(pf_["stakes"]) == pytest.approx(1.0, abs=1e-4)


def test_stakes_are_the_ones_the_pipeline_computes(result, payload):
    """Spot-check against `stakes_for` rather than re-deriving the split here."""
    kept = result["scored"][result["scored"]["undominated"]]
    row = kept.iloc[0]
    picks_row = result["picks"][int(row["combo"])]
    stakes, prob, _, _ = pf.stakes_for(picks_row[None, :], result["options"], row["split"])
    # One stake per proposition: `stakes_for` returns two slots per event and a
    # pair fills both, so the live slots are the ones carrying a probability.
    expected = [round(float(v), 6) for v in stakes[0][prob[0] > 0]]
    assert payload["portfolios"][0]["stakes"] == expected


# --- what is and is not exported -------------------------------------------


def test_only_undominated_portfolios_are_exported(result, payload):
    scored = result["scored"]
    assert len(payload["portfolios"]) == int(scored["undominated"].sum())
    assert payload["counts"]["scored"] == len(scored)
    exported = {p["id"] for p in payload["portfolios"]}
    dominated = set(scored.loc[~scored["undominated"], "id"].astype(int))
    assert not exported & dominated


def test_no_leg_rows_are_written(payload):
    """A portfolio's legs are its picks. Anything else is the 29,741-row sheet."""
    assert "legs_rows" not in payload
    for pf_ in payload["portfolios"]:
        assert all(isinstance(x, str) for x in pf_["picks"])


def test_selection_frequency_matches_a_recount(payload):
    n = len(payload["portfolios"])
    counts = Counter(pid for pf_ in payload["portfolios"] for pid in pf_["picks"])
    for prop in payload["propositions"]:
        assert prop["selection_frequency"] == pytest.approx(counts[prop["id"]] / n, abs=5e-5)


def test_unpicked_propositions_are_zero_not_absent(payload):
    """A proposition that qualified and was never picked is a fact, not a gap."""
    picked = {pid for pf_ in payload["portfolios"] for pid in pf_["picks"]}
    for prop in payload["propositions"]:
        assert "selection_frequency" in prop
        if prop["id"] not in picked:
            assert prop["selection_frequency"] == 0.0


# --- the schema the page reads ---------------------------------------------


def test_split_values_are_the_normalised_keys(payload):
    assert {p["split"] for p in payload["portfolios"]} == {"growth"}
    # The two retired splits keep their keys: archived payloads and ledger rows
    # carry them, and a reader that cannot name them cannot read its own history.
    assert set(payload["splits"]) == {"growth", "min_variance", "inverse_e"}


def test_thresholds_travel_with_the_data(payload):
    """The page builds its `P(>X%)` columns from this array.

    Pinned here so that adding a sixth threshold to `staking.THRESHOLDS` fails
    loudly rather than silently dropping a column the app never learned about.
    """
    assert payload["thresholds"] == [int(round(t * 100)) for t in staking.THRESHOLDS]
    for pf_ in payload["portfolios"]:
        for t in payload["thresholds"]:
            assert 0.0 <= pf_[f"p_over_{t}"] <= 1.0


def test_prices_carry_the_negative_edges_propositions_cannot(payload):
    """`propositions` cleared `E >= 1` by construction; `prices` did not.

    Without this array the Match Board can only ever draw a green bar, and the
    style guide's own worked example is an orange one.
    """
    qualified = {(p["event"], p["proposition"]) for p in payload["propositions"]}
    priced = {(r["event"], r["label"]) for r in payload["prices"]}
    assert qualified <= priced
    assert any(r["e"] < 1.0 for r in payload["prices"])


def test_payload_is_json_serialisable(payload):
    """numpy scalars survive `to_dict` and are not JSON types."""
    text = json.dumps(payload, default=eb._json_default)
    assert json.loads(text)["counts"]["exported"] == len(payload["portfolios"])


# --- the label join ---------------------------------------------------------


def test_line_labels_match_the_form_the_prices_were_filled_against():
    """The one string in the system that has to agree across two files.

    `predictions_payload` writes a label per line and `read_filled` reads one
    back off the odds form; the Match Board joins them. Both go through
    `staking.proposition_label`, and this is the test that says so.
    """
    for target, team, line in (("corners", "Coventry City", 2.5),
                               ("sot", "Real Betis", 4.5),
                               ("goals", "Hull", 0.5)):
        label = staking.proposition_label(target, team, line)
        assert label.endswith(f"- {team} - Over {line}")
    assert staking.proposition_label("sot", "Hull", 1.5) == "Shots on Target - Hull - Over 1.5"


def test_propositions_reuses_the_shared_label_builder(monkeypatch):
    """`staking.propositions` must not have its own copy of the format."""
    seen = []
    real = staking.proposition_label

    def spy(target, team, line):
        seen.append((target, team, line))
        return real(target, team, line)

    monkeypatch.setattr(staking, "proposition_label", spy)
    preds = pd.DataFrame([{
        "date": pd.Timestamp("2026-08-21"), "league": "Premier League", "league_key": "Prem",
        "home_team": "Arsenal", "away_team": "Coventry City",
        "goals_home": 1.9, "goals_away": 1.3, "shots_home": 16.1, "shots_away": 11.3,
        "sot_home": 5.8, "sot_away": 3.9, "corners_home": 5.0, "corners_away": 3.9,
    }])
    # The three count families are negative binomial and refuse to build a pmf
    # without a fitted dispersion; any positive number will do to reach the label.
    staking.propositions(preds, {t: {"Prem": 1.5} for t in ("shots", "sot", "corners")})
    assert seen, "propositions() built a label without going through the shared builder"


# --- the bundle -------------------------------------------------------------


def _write_pair(tmp_path, payload, pred_date="2026-08-21", port_date="2026-08-21"):
    pred = tmp_path / f"predictions_{pred_date}.json"
    port = tmp_path / f"portfolios_{port_date}.json"
    pred.write_text(json.dumps({"generated_at": "x", "targets": [], "fixtures": []}))
    port.write_text(json.dumps(payload, default=eb._json_default))
    return pred, port


def test_bundle_inlines_both_payloads_and_the_app(tmp_path, payload):
    pred, port = _write_pair(tmp_path, payload)
    out = eb.write_edge_book(pred, port, out_path=tmp_path / "book.html")
    html = out.read_text(encoding="utf-8")
    assert not any(m in html for m in eb.MARKERS), "a marker survived the build"
    assert '<script id="predictions-data" type="application/json">' in html
    assert '<script id="portfolios-data" type="application/json">' in html
    assert "Edge Book" in html


def test_no_json_block_can_close_its_own_script_tag(tmp_path, payload):
    """`<` is escaped to `\\u003c`, which JSON reads back identically."""
    pred, port = _write_pair(tmp_path, payload)
    html = eb.write_edge_book(pred, port, out_path=tmp_path / "book.html").read_text()
    body = html.split('<script id="portfolios-data" type="application/json">')[1].split("</script>")[0]
    assert "<" not in body
    assert json.loads(body)["counts"]["exported"] == len(payload["portfolios"])


def test_a_mismatched_pair_raises_rather_than_building(tmp_path, payload):
    """Predictions come from `05_Run` and portfolios from `06_Split`.

    Not re-running one of them is a single missed step, and the resulting page
    would open cleanly showing yesterday's fixtures beside today's portfolios.
    """
    pred, port = _write_pair(tmp_path, payload, pred_date="2026-08-20")
    with pytest.raises(ValueError, match="different runs"):
        eb.write_edge_book(pred, port, out_path=tmp_path / "book.html")


def test_latest_json_picks_by_the_date_in_the_name(tmp_path):
    """By name, not mtime: re-exporting an old run touches it without making it current."""
    for d in ("2026-08-19", "2026-08-21", "2026-08-20"):
        (tmp_path / f"portfolios_{d}.json").write_text("{}")
    assert eb.latest_json("portfolios", tmp_path).name == "portfolios_2026-08-21.json"
    assert eb.latest_json("predictions", tmp_path) is None


def test_missing_input_names_the_notebook_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="05_Run"):
        eb.write_edge_book(tmp_path / "nope.json", tmp_path / "also_nope.json")


# --- an empty run -----------------------------------------------------------


def test_nothing_qualifying_still_produces_a_valid_payload():
    """Every price below the model is a normal outcome, not an error."""
    rows = [{"sheet_code": "EV-00", "label": "Corners - Team 0 - Over 0.5",
             "p": 0.5, "b365": 1.5, "home_team": "H", "away_team": "A"}]
    empty = pf.search(pd.DataFrame(rows), min_legs=1)
    out = eb.portfolios_payload(empty, pd.DataFrame(rows))
    assert out["portfolios"] == []
    assert out["counts"]["exported"] == 0
    json.dumps(out, default=eb._json_default)


# --- the projection block ---------------------------------------------------
#
# The Projection Book joins on a third key: every array in a portfolio's
# `projection` is indexed by an axis stored once, at the top of the payload. A
# length that drifts there does not fail -- it silently plots the wrong x for
# every y, on every portfolio at once.


def test_projection_arrays_match_the_hoisted_axes(payload):
    """The join the Projection Book rests on, checked by length rather than trust."""
    axes = payload["projection_axes"]
    for row in payload["portfolios"]:
        pr = row["projection"]
        assert len(pr["g_curve"]) == len(axes["f"])
        assert set(pr["bands"]) == {"suggested"}, "one stake, so one fan"
        band = pr["bands"]["suggested"]
        if band is not None:
            for series in band.values():
                assert len(series) == len(axes["rounds"])


def test_projection_axes_are_absent_when_nothing_carries_one(result, filled):
    """"This run has no Projection Book" must be distinguishable from "this row"."""
    off = eb.portfolios_payload(result, filled, with_projection=False)
    assert "projection_axes" not in off
    assert all("projection" not in r for r in off["portfolios"])


def test_projection_needs_growth(result, filled):
    """`f_star` is the projection's own x-axis, so growth off means projection off."""
    off = eb.portfolios_payload(result, filled, with_growth=False)
    assert all("projection" not in r for r in off["portfolios"])
    assert "projection_axes" not in off


def test_projection_survives_a_json_round_trip(payload):
    """No NaN, no Infinity -- neither is JSON, and both parse as a syntax error."""
    text = json.dumps(payload, default=eb._json_default, allow_nan=False)
    back = json.loads(text)
    assert back["portfolios"][0]["projection"]["g_curve"]


def test_best_case_probability_does_not_round_to_zero(payload):
    """`p_max` is the mirror of `p0` and gets the same significant-figure care.

    A portfolio whose best case reads `0%` next to a positive maximum return is
    the same false statement `_sig` exists to stop at the other end.
    """
    for row in payload["portfolios"]:
        pr = row["projection"]
        if pr["max_return"] and pr["max_return"] > 0:
            assert pr["p_max"] is not None and pr["p_max"] > 0
