"""The shots calibration (`config.SHOTS_LOGIT_SHIFT`): computed every run, applied only when live."""

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


def _logit(p):
    return np.log(p / (1 - p))


@pytest.fixture
def live(monkeypatch):
    monkeypatch.setattr(markets, "SHOTS_SHIFT_LIVE", True)
    monkeypatch.setattr(config, "SHOTS_SHIFT_LIVE", True)


def test_shift_is_off_by_default():
    assert config.SHOTS_SHIFT_LIVE is False


def test_shadow_keeps_p_raw_and_records_the_shifted_value():
    mk = markets.fixture_markets(_row("Serie"), DISP)
    c = config.SHOTS_LOGIT_SHIFT["Serie"]
    s = mk["shots"]
    assert s["shift"] == 0.0 and s["candidate_shift"] == c
    for scope in ("home", "away"):
        for line in s[f"{scope}_lines"]:
            raw, new = s[f"{scope}_over_{line}_raw"], s[f"{scope}_over_{line}_shifted"]
            assert s[f"{scope}_over_{line}"] == raw
            if 0 < raw < 1:
                assert _logit(new) - _logit(raw) == pytest.approx(c)
    assert not any(k.endswith(("_raw", "_shifted")) for k in mk["sot"])


def test_live_moves_only_shots_team_lines_and_keeps_the_raw_value(live):
    mk = markets.fixture_markets(_row("Serie"), DISP)
    c = config.SHOTS_LOGIT_SHIFT["Serie"]
    s = mk["shots"]
    assert s["shift"] == c
    for scope in ("home", "away"):
        for line in s[f"{scope}_lines"]:
            raw, new = s[f"{scope}_over_{line}_raw"], s[f"{scope}_over_{line}"]
            assert new == s[f"{scope}_over_{line}_shifted"]
            if 0 < raw < 1:
                assert _logit(new) - _logit(raw) == pytest.approx(c)
    # Other targets carry no raw copies and are untouched by the shift.
    assert not any(k.endswith("_raw") for k in mk["sot"]) and "shift" not in mk["corners"]


def test_leagues_without_a_fitted_shift_take_the_pooled_default():
    assert markets.shots_shift("Bund") == config.SHOTS_LOGIT_SHIFT_DEFAULT
    assert markets.shots_shift("Prem") == config.SHOTS_LOGIT_SHIFT["Prem"]


def _props():
    from fpp.staking import propositions

    preds = pd.DataFrame([{**_row("Prem").to_dict(), "date": pd.Timestamp("2026-10-18"),
                           "league": "Premier League", "home_team": "Arsenal", "away_team": "Chelsea"}])
    return propositions(preds, DISP)


def test_propositions_carry_p_raw_and_p_shifted_in_shadow():
    props = _props()
    shots, other = props[props["target"] == "shots"], props[props["target"] != "shots"]
    assert (shots["p"] == shots["p_raw"]).all()
    assert (shots["p_shifted"] > shots["p_raw"]).all()    # the shift is upward
    assert (other["p"] == other["p_raw"]).all() and (other["p"] == other["p_shifted"]).all()


def test_propositions_use_the_shift_when_live(live):
    shots = _props().query("target == 'shots'")
    assert (shots["p"] > shots["p_raw"]).all() and (shots["p"] == shots["p_shifted"]).all()


def test_stamp_records_applied_and_candidate_shift(live, monkeypatch):
    from fpp import artifacts

    on = artifacts.model_stamp()
    assert json.loads(on["shots_shift"]) == json.loads(on["shots_shift_candidate"])
    monkeypatch.setattr(config, "SHOTS_SHIFT_LIVE", False)
    off = artifacts.model_stamp()
    assert json.loads(off["shots_shift"]) == {}
    assert json.loads(off["shots_shift_candidate"])["Serie"] == config.SHOTS_LOGIT_SHIFT["Serie"]


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


def test_ledger_rebuilds_the_shadow_value_from_the_candidate():
    stamp = json.dumps({"Prem": 0.389, "*": 0.558})
    props = pd.DataFrame({"p_raw": [0.40, 0.40, 0.40], "target": ["shots", "shots", "sot"],
                          "league_key": ["Prem", "Liga", "Prem"]})
    out = ledger.shifted_probability(props, stamp)
    np.testing.assert_allclose(_logit(out[:2]) - _logit(0.40), [0.389, 0.558])
    assert out[2] == 0.40
    # In shadow nothing was applied, so the raw value is `p` itself.
    np.testing.assert_allclose(ledger.raw_probability(props.rename(columns={"p_raw": "p"}), "{}"), 0.40)
