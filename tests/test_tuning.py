"""Guards on the tuning loop: plateau entry, block acceptance, convergence.

Real CV is 2-6 s a fit, so these drive the machinery with stub scorers. That is
the right trade here: what needs pinning is the *control flow* -- which candidate
a block accepts, whether a round counts as having moved, when the loop stops --
and none of that is about xgboost.

The one behaviour worth naming up front is why blocks defend their incumbent.
Hyperparameters go inert on each other: measured on real data, `reg_alpha=10`
prunes splits hard enough that `min_child_weight` 1 and 30 produce byte-identical
predictions. Over a block of tied scores a plain argmin picks whichever row sorts
first, so the block "moves" on nothing, every pass, and the alternating loop can
never report convergence. A first run of the loop did exactly that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpp import cv
from fpp.build import FeatureTable
from fpp.config import SEARCH
from fpp.search import stages as st


# --- Plateau entry ----------------------------------------------------------


def _surface(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["L", "alpha", "score_mean"])
    df["score_se"] = 0.003
    return df


def test_plateau_entry_takes_the_near_edge_not_the_lowest_point():
    """The zone is flat, so the cheapest point inside it is the useful one."""
    df = _surface([(5, 0.05, 1.4700), (15, 0.05, 1.4600), (25, 0.05, 1.4555),
                   (35, 0.05, 1.4552), (45, 0.05, 1.4550)])
    pick = st._plateau_entry(df, tol=0.001)
    assert int(pick["L"]) == 25, "should enter the plateau, not walk to its floor"


def test_plateau_entry_rejects_a_lucky_dip():
    """A low-L point inside the band with worse scores above it is not a plateau.

    This is the case `best_row(prefer=("L",))` gets wrong: it would take L=9 purely
    for being the smallest thing inside the band, with 13 and 19 worse on both
    sides of it.
    """
    df = _surface([(9, 0.05, 1.4552), (13, 0.05, 1.4620), (19, 0.05, 1.4610),
                   (25, 0.05, 1.4556), (35, 0.05, 1.4553), (45, 0.05, 1.4550)])
    pick = st._plateau_entry(df, tol=0.001)
    assert int(pick["L"]) == 25, f"took the dip at L={int(pick['L'])}"


def test_plateau_entry_is_per_alpha():
    """A tail is only unbroken within one alpha; two alphas are two surfaces."""
    df = _surface([(15, 0.02, 1.4551), (25, 0.02, 1.4700), (35, 0.02, 1.4550),
                   (15, 0.20, 1.4900), (25, 0.20, 1.4553), (35, 0.20, 1.4552)])
    pick = st._plateau_entry(df, tol=0.001)
    # L=15 at alpha 0.02 is in band but L=25 above it is not, so it is not an
    # entry. L=25 at alpha 0.20 is, and is the smallest that qualifies.
    assert (int(pick["L"]), float(pick["alpha"])) == (25, 0.20)


def test_a_surface_with_no_unbroken_tail_falls_back_to_the_best():
    df = _surface([(5, 0.05, 1.4550), (15, 0.05, 1.4900), (25, 0.05, 1.4800)])
    pick = st._plateau_entry(df, tol=0.0001)
    assert int(pick["L"]) == 5 and float(pick["score_mean"]) == 1.4550


def test_plateau_entry_needs_a_score():
    with pytest.raises(ValueError, match="no window candidate"):
        st._plateau_entry(pd.DataFrame({"L": [5], "alpha": [0.05],
                                        "score_mean": [np.nan], "score_se": [0.003]}), tol=0.001)


def test_effective_window_is_capped_by_the_decay():
    """Past ~3/alpha there is no weight left, so a larger L changes nothing."""
    assert st.effective_window(80, 0.0) == 80        # no decay: L is the window
    assert st.effective_window(80, 0.05) == 60       # 3/0.05 = 60
    assert st.effective_window(30, 0.05) == 30       # L binds first
    assert st.effective_window(100, 0.5) == 6


# --- Block acceptance -------------------------------------------------------


def _ft(n=400) -> FeatureTable:
    seasons = ["2019/2020", "2020/2021", "2021/2022", "2022/2023", "2023/2024"]
    meta = pd.DataFrame({
        "fixture_id": np.arange(n) // 2,
        "season": [seasons[i % len(seasons)] for i in range(n)],
        "date": pd.Timestamp("2021-01-01"), "league": "Premier League",
        "league_key": "Prem", "team": "T", "opponent": "O", "is_home": np.arange(n) % 2,
    })
    return FeatureTable(X=pd.DataFrame({"f": np.arange(n, dtype=float)}), meta=meta,
                        y={"goals": np.ones(n)})


def _stub(monkeypatch, fn):
    """Replace the CV call the descent reaches through with `fn(params) -> float`."""
    def fake(ft, target, params, folds, *, features=None, fit_cfg=SEARCH):
        return cv.CVResult(target=target, mean=float(fn(params)), se=0.003,
                           per_fold={"a": 1.0}, n_scored=10,
                           best_iterations=[100], dispersion={})
    monkeypatch.setattr(st, "evaluate", fake)


def test_a_block_whose_parameter_is_inert_keeps_its_incumbent(tmp_path, monkeypatch):
    """The measured failure: tied scores, and argmin flips to whatever sorts first."""
    _stub(monkeypatch, lambda p: 1.0)          # nothing any parameter does matters
    start = {"objective": "count:poisson", "n_estimators": 3000, "learning_rate": 0.05,
             "max_depth": 3, "max_bin": 64, "min_child_weight": 30, "subsample": 0.8,
             "colsample_bytree": 0.8, "reg_alpha": 10, "reg_lambda": 1.0}
    out = st.run_xgb_descent(_ft(), "goals", ["f"], tmp_path / "d.csv",
                             passes=2, start_params=dict(start))
    for k, v in start.items():
        assert out["params"][k] == v, f"{k} moved on tied scores: {v} -> {out['params'][k]}"
    assert out["moved"] is False


def test_a_block_that_genuinely_improves_does_move(tmp_path, monkeypatch):
    _stub(monkeypatch, lambda p: 1.0 - 0.5 * (int(p.get("max_depth", 3)) == 6))
    out = st.run_xgb_descent(_ft(), "goals", ["f"], tmp_path / "d.csv", passes=1,
                             start_params={"objective": "count:poisson", "n_estimators": 3000,
                                           "learning_rate": 0.05, "max_depth": 3, "max_bin": 64})
    assert out["params"]["max_depth"] == 6
    assert out["moved"] is True


def test_an_improvement_inside_the_tolerance_is_not_taken(tmp_path, monkeypatch):
    """Sub-tolerance gains are how a descent wanders without getting anywhere.

    Both axes of the block have to be set for the incumbent to be locatable in a
    2-D grid. With one of them unset there is no incumbent row to compare against,
    the margin is NaN and the winner is taken -- which is the right behaviour for
    an axis `base_params` never set, and is what pass 1 does for most blocks.
    """
    _stub(monkeypatch, lambda p: 1.0 - 0.0001 * (int(p.get("max_depth", 3)) == 6))
    start = {"objective": "count:poisson", "n_estimators": 3000, "learning_rate": 0.05,
             "max_depth": 3, "min_child_weight": 1, "max_bin": 64}
    out = st.run_xgb_descent(_ft(), "goals", ["f"], tmp_path / "d.csv", passes=1, tol=0.001,
                             start_params=dict(start))
    assert out["params"]["max_depth"] == 3, "a 0.0001 gain is inside a 0.001 tolerance"


def test_an_unset_axis_has_no_incumbent_to_defend(tmp_path, monkeypatch):
    """`base_params` sets `max_depth` and not `min_child_weight`, so on the first
    pass that block cannot locate an incumbent row and takes its winner."""
    _stub(monkeypatch, lambda p: 1.0 - 0.0001 * (int(p.get("max_depth", 3)) == 6))
    out = st.run_xgb_descent(_ft(), "goals", ["f"], tmp_path / "d.csv", passes=1, tol=0.001,
                             start_params={"objective": "count:poisson", "n_estimators": 3000,
                                           "learning_rate": 0.05, "max_depth": 3, "max_bin": 64})
    assert out["params"]["max_depth"] == 6


def test_the_descent_stops_once_a_pass_changes_nothing(tmp_path, monkeypatch, capsys):
    calls = []

    def score(p):
        calls.append(dict(p))
        return 1.0 - 0.5 * (int(p.get("max_depth", 3)) == 6)

    _stub(monkeypatch, score)
    st.run_xgb_descent(_ft(), "goals", ["f"], tmp_path / "d.csv", passes=4,
                       start_params={"objective": "count:poisson", "n_estimators": 3000,
                                     "learning_rate": 0.05, "max_depth": 3, "max_bin": 64})
    out = capsys.readouterr().out
    assert "descent converged" in out
    # Pass 1 moves max_depth, pass 2 changes nothing and ends it -- so passes 3
    # and 4 never run, and the checkpoint cannot have four passes' worth of rows.
    assert max(int(c.get("pass", 0)) for c in calls) <= 1


def test_max_bin_is_a_tuned_axis_and_can_leave_the_search_default(tmp_path, monkeypatch):
    """It used to be inherited from the FitConfig and frozen at the probe's 64."""
    _stub(monkeypatch, lambda p: 1.0 - 0.5 * (int(p.get("max_bin", 64)) == 256))
    out = st.run_xgb_descent(_ft(), "goals", ["f"], tmp_path / "d.csv", passes=1,
                             start_params={"objective": "count:poisson", "n_estimators": 3000,
                                           "learning_rate": 0.05, "max_depth": 3,
                                           "max_bin": SEARCH.max_bin})
    assert out["params"]["max_bin"] == 256 != SEARCH.max_bin


def test_the_tree_ceiling_is_a_ceiling_not_a_budget():
    """Early stopping picks the count, so a low rate is judged at the trees it wants."""
    assert SEARCH.n_estimators >= 3000
    assert SEARCH.early_stopping_rounds >= 50
