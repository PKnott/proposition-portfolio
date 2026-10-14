"""The production shots calibration (`config.SHOTS_LOGIT_SHIFT`): applied once, recorded twice."""

import json

import numpy as np
import pandas as pd
import pytest

from fpp import config, ledger
from fpp.report import markets


def _row(league="Serie"):
    return pd.Series({"league_key": league, "goals_home": 1.4, "goals_away": 1.1,
                      "shots_home": 13.0, "shots_away": 10.5, "sot_home": 4.6, "sot_away": 3.7,
                      "corners_home": 5.2, "corners_away": 4.4})


DISP = {"shots": 12.0, "sot": 20.0, "corners": 15.0}


def test_shift_moves_only_shots_team_lines_and_keeps_the_raw_value():
    mk = markets.fixture_markets(_row("Serie"), DISP)
    c = config.SHOTS_LOGIT_SHIFT["Serie"]
    s = mk["shots"]
    assert s["shift"] == c
    for scope in ("home", "away"):
        for line in s[f"{scope}_lines"]:
            raw, new = s[f"{scope}_over_{line}_raw"], s[f"{scope}_over_{line}"]
            if 0 < raw < 1:
                assert np.log(new / (1 - new)) - np.log(raw / (1 - raw)) == pytest.approx(c)
    # Other targets carry no raw copies and are untouched by the shift.
    assert not any(k.endswith("_raw") for k in mk["sot"]) and "shift" not in mk["corners"]


def test_leagues_without_a_fitted_shift_take_the_pooled_default():
    assert markets.shots_shift("Bund") == config.SHOTS_LOGIT_SHIFT_DEFAULT
    assert markets.shots_shift("Prem") == config.SHOTS_LOGIT_SHIFT["Prem"]


def test_propositions_carry_p_raw():
    from fpp.staking import propositions

    preds = pd.DataFrame([{**_row("Prem").to_dict(), "date": pd.Timestamp("2026-10-18"),
                           "league": "Premier League", "home_team": "Arsenal", "away_team": "Chelsea"}])
    props = propositions(preds, DISP)
    shots, other = props[props["target"] == "shots"], props[props["target"] != "shots"]
    assert (shots["p"] > shots["p_raw"]).all()            # the shift is upward
    assert (other["p"] == other["p_raw"]).all()


def test_ledger_recovers_the_raw_probability_from_the_stamp():
    stamp = json.dumps({"Prem": 0.389, "Serie": 0.728, "*": 0.558})
    raw = np.array([0.30, 0.45, 0.30, 0.50])
    shift = np.array([0.389, 0.728, 0.0, 0.558])
    shifted = 1 / (1 + np.exp(-(np.log(raw / (1 - raw)) + shift)))
    props = pd.DataFrame({"p": shifted, "target": ["shots", "shots", "goals", "shots"],
                          "league_key": ["Prem", "Serie", "Prem", "Bund"]})
    props.loc[2, "p"] = 0.30                                 # goals: never shifted
    np.testing.assert_allclose(ledger.raw_probability(props, stamp), raw)
    # A form stamped before the shift existed: raw is p.
    np.testing.assert_allclose(ledger.raw_probability(props, None), props["p"])
