"""European layer: the pieces that must be right by construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpp.europe import ingest, layer, model


# --- calendar -----------------------------------------------------------------

def _calendar(entries):
    return {"leagues": [{"calendar": [{"entries": entries}]}]}


def test_season_dates_skip_qualifying_and_keep_midweek_and_final():
    payload = _calendar([
        {"label": "Qualifying Third Round", "startDate": "2016-07-22T07:00Z", "endDate": "2016-08-07T06:59Z"},
        {"label": "Playoff Round", "startDate": "2016-08-07T07:00Z", "endDate": "2016-09-01T06:59Z"},
        {"label": "Group", "startDate": "2016-09-12T07:00Z", "endDate": "2016-09-19T06:59Z"},
        {"label": "Knockout Round Playoffs", "startDate": "2017-02-13T07:00Z", "endDate": "2017-02-17T06:59Z"},
        {"label": "Final", "startDate": "2017-06-02T07:00Z", "endDate": "2017-06-04T06:59Z"},
    ])
    d = ingest.season_dates(payload, 2016)
    assert not any(x.startswith("201607") or x.startswith("201608") for x in d)
    # Group window 12-19 Sep 2016: Tue 13, Wed 14, Thu 15 only
    assert {x for x in d if x.startswith("201609")} == {"20160913", "20160914", "20160915"}
    assert "20170214" in d  # knockout play-offs are a real round, not a qualifier
    assert {"20170602", "20170603", "20170604"} <= d  # the final window, every day


def test_season_dates_every_day_for_the_2020_tournament():
    payload = _calendar([{"label": "Quarterfinals", "startDate": "2020-08-12T07:00Z",
                          "endDate": "2020-08-16T06:59Z"}])
    assert {"20200814", "20200815"} <= ingest.season_dates(payload, 2019)  # Fri, Sat


# --- team rows ------------------------------------------------------------------

def _clean(rows):
    return pd.DataFrame(rows, columns=["team", "season", "league_key"])


def _stats(**kw):
    base = dict(espn_event_id="1", league_key="UCL", season="2020/2021", stage="group-stage", leg=np.nan,
                date="2020-10-20", home_team="A", away_team="B",
                home_goals=1, away_goals=0, home_shots=10, away_shots=8, home_sot=4, away_sot=3,
                home_corners=5, away_corners=2)
    base.update(kw)
    return pd.DataFrame([base])


def test_team_rows_need_both_clubs_in_a_modelled_league_that_season():
    clean = _clean([("A", "2020/2021", "Prem"), ("B", "2019/2020", "Liga")])  # B not in 2020/21
    assert model.team_rows(_stats(), clean).empty
    clean = _clean([("A", "2020/2021", "Prem"), ("B", "2020/2021", "Liga")])
    r = model.team_rows(_stats(), clean)
    assert len(r) == 2
    home = r[r["is_home"] == 1].iloc[0]
    assert (home["team"], home["team_league"], home["opp_league"]) == ("A", "Prem", "Liga")
    assert (home["shots_for"], home["shots_against"]) == (10, 8)


def test_team_rows_read_the_mixed_season_labels_of_the_clean_table():
    clean = _clean([("A", "2526", "Prem"), ("B", "2025/2026", "Liga")])
    assert len(model.team_rows(_stats(season="2025/2026", date="2025-10-21"), clean)) == 2


# --- as-of priors -----------------------------------------------------------------

def test_asof_reads_the_next_domestic_row_and_the_next_one_at_the_same_venue():
    dom = pd.DataFrame({
        "team": ["A"] * 4,
        "date": pd.to_datetime(["2020-10-17", "2020-10-24", "2020-10-31", "2020-11-07"]),
        "is_home": [1, 0, 1, 0],
        "team_avg_shots_for": [1.0, 2.0, 3.0, 4.0],
        "team_avg_shots_for_v": [10.0, 20.0, 30.0, 40.0],
    })
    rows = pd.DataFrame({"team": ["A", "A"], "date": pd.to_datetime(["2020-10-20", "2020-10-21"]),
                         "is_home": [1, 0]})
    out = model.asof_priors(rows, dom, ["team_avg_shots_for", "team_avg_shots_for_v"])
    assert out["team_avg_shots_for"].tolist() == [2.0, 2.0]       # next row, 24 Oct
    assert out["team_avg_shots_for_v"].tolist() == [30.0, 20.0]   # next home (31 Oct) / away (24 Oct)


def test_asof_is_strict_and_bounded():
    dom = pd.DataFrame({"team": ["A", "A"], "date": pd.to_datetime(["2020-10-20", "2021-06-01"]),
                        "is_home": [1, 1], "team_avg_goals_for": [1.0, 2.0]})
    rows = pd.DataFrame({"team": ["A"], "date": pd.to_datetime(["2020-10-20"]), "is_home": [1]})
    # a same-day domestic row is not "next"; the following one is beyond MAX_LOOKAHEAD
    assert model.asof_priors(rows, dom, ["team_avg_goals_for"]).isna().all().all()


# --- par conversion -----------------------------------------------------------------

def test_par_is_identity_within_a_league_and_neutral_when_unknown():
    rates = {("Prem", 2019, "shots", "for", "all"): 12.0, ("Liga", 2019, "shots", "for", "all"): 10.0}
    assert model.par_ratio(rates, "Prem", "Prem", 2020, "shots", "for", "all") == 1.0
    assert model.par_ratio(rates, "Prem", "Liga", 2020, "shots", "for", "all") == pytest.approx(1.2)
    assert model.par_ratio(rates, "Prem", "Liga", 2021, "shots", "for", "all") == 1.0  # no 2020 rates


def test_lookup_features_maps_opponent_features_to_team_buffers():
    need = model.lookup_features(["team_avg_shots_for", "opp_avg_xg_for_v", "is_home", "team_rest_days"])
    assert need == ["team_avg_shots_for", "team_avg_xg_for_v"]


# --- layer ------------------------------------------------------------------------

def test_stage_groups():
    assert layer.stage_group("league-phase", np.nan) == "league"
    assert layer.stage_group("round-of-16", 1) == "ko1"
    assert layer.stage_group("round-of-16", 2.0) == "ko2"
    assert layer.stage_group("final", np.nan) == "single"
    assert layer.stage_group("quarterfinals", np.nan) == "single"


def test_euroform_is_record_before_update():
    rows = pd.DataFrame({"team": ["A"] * 3, "date": pd.to_datetime(["2020-10-01", "2020-10-15", "2020-11-01"]),
                         "actual": [20.0, 0.0, 0.0], "mu_dom": [10.0, 10.0, 10.0]})
    f = layer.euroform(rows)
    assert f[0] == 0.0                           # nothing before the first match
    assert f[1] == pytest.approx(np.log((20 + 100) / (10 + 100)))   # only the first counts
    assert f[2] == pytest.approx(np.log((20 + 100) / (20 + 100)))


def test_layer_recovers_a_planted_league_gap():
    rng = np.random.default_rng(0)
    n = 6000
    tl = rng.choice(layer.LEAGUE_KEYS, n)
    ol = rng.choice(layer.LEAGUE_KEYS, n)
    mu_dom = rng.uniform(8, 16, n)
    gap = {"Prem": 0.15, "Liga": 0.05, "Bund": 0.0, "Serie": -0.05, "Ligue": -0.15}
    true = mu_dom * np.exp(np.array([gap[a] for a in tl]))
    rows = pd.DataFrame({"team_league": tl, "opp_league": ol, "comp": "UCL", "stage": "league-phase",
                         "leg": np.nan, "mu_dom": mu_dom, "actual": rng.poisson(true).astype(float)})
    fit = layer.fit(rows, "goals", alpha=1e-6)
    c = fit.coefficients()
    est = {lk: c[f"att_{lk}"] for lk in layer.LEAGUE_KEYS}
    centred = {k: v - np.mean(list(est.values())) for k, v in est.items()}
    for lk, g in gap.items():
        assert centred[lk] == pytest.approx(g, abs=0.03)
    # and no spurious defensive gap
    defs = np.array([c[f"def_{lk}"] for lk in layer.LEAGUE_KEYS])
    assert np.ptp(defs) < 0.06
