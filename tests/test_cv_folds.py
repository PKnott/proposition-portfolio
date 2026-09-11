"""Guards on fold construction, the tuning holdout, and checkpoint identity.

All three cover failures that are invisible at the point of use. A fold built in
the wrong order still trains and still scores; a stale checkpoint still returns a
number. Nothing raises, nothing looks wrong, and the answer is from the previous
configuration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpp import cv
from fpp.build import FeatureTable
from fpp.config import (
    PROBE,
    SEARCH,
    FitConfig,
    holdout_season,
    season_sort_key,
)
# Aliased: pytest collects any module-level name starting with `test_`, so a bare
# `from fpp.config import test_season` is picked up as a (returning) test.
from fpp.config import test_season as first_test_season
from fpp.search.stages import fingerprint_for


def _ft(seasons: list[str], per_season: int = 4) -> FeatureTable:
    """A FeatureTable carrying only what fold construction reads."""
    rows = []
    for s in seasons:
        for i in range(per_season):
            rows.append({"fixture_id": len(rows) // 2, "season": s, "date": pd.Timestamp("2020-01-01"),
                         "league": "Premier League", "league_key": "Prem",
                         "team": f"T{i}", "opponent": "X", "is_home": i % 2})
    meta = pd.DataFrame(rows)
    X = pd.DataFrame({"f": np.arange(len(meta), dtype=float)})
    return FeatureTable(X=X, meta=meta, y={"goals": np.ones(len(meta))})


# --- Season ordering --------------------------------------------------------

MIXED = ["2014/2015", "2015/2016", "1920", "2019/2020", "2020/2021", "2021/2022"]


def test_a_lexically_early_label_is_not_treated_as_an_early_season():
    """`'1920'` is Ligue 1's curtailed 2019/20 and sorts before `'2014/2015'`.

    Under string comparison it counted as the oldest season in the table, so it
    could land in the training set for a validation season five years its junior.
    """
    assert "1920" < "2014/2015", "the premise: lexically it really does sort first"
    assert season_sort_key("1920") == 2019
    assert season_sort_key("1920") > season_sort_key("2014/2015")


def test_no_fold_trains_on_a_season_at_or_after_its_validation_season():
    ft = _ft(MIXED)
    year = ft.meta["season"].map(season_sort_key).to_numpy()
    folds = cv.season_folds(ft, first_val="2015/2016")
    assert folds, "no folds built"
    for f in folds:
        val_year = season_sort_key(f.val_season)
        train_years = set(year[f.train])
        assert train_years, f"{f.val_season} trains on nothing"
        assert max(train_years) < val_year, (
            f"{f.val_season} trains on {sorted(y for y in train_years if y >= val_year)}"
        )


def test_two_labels_for_one_real_season_never_train_on_each_other():
    """`'1920'` and `'2019/2020'` are the same season under two source formats."""
    ft = _ft(["2018/2019", "1920", "2019/2020", "2020/2021"])
    year = ft.meta["season"].map(season_sort_key).to_numpy()
    for f in cv.season_folds(ft, first_val="2018/2019"):
        assert not (set(year[f.train]) & set(year[f.val])), (
            f"{f.val_season} trains on its own season under the other label"
        )


def test_the_test_seasons_are_excluded_from_every_fold():
    ft = _ft([*MIXED, first_test_season()])
    year = ft.meta["season"].map(season_sort_key).to_numpy()
    boundary = season_sort_key(first_test_season())
    for f in cv.season_folds(ft, first_val="2015/2016"):
        assert (year[f.train] < boundary).all()
        assert (year[f.val] < boundary).all()


# --- The tuning holdout -----------------------------------------------------


def test_the_holdout_season_is_the_one_before_the_test_boundary():
    assert season_sort_key(holdout_season()) == season_sort_key(first_test_season()) - 1


def test_tuning_folds_stop_before_the_holdout():
    ft = _ft(["2020/2021", "2021/2022", "2022/2023", "2023/2024", holdout_season()])
    tuning = cv.tuning_folds(ft)
    assert tuning, "no tuning folds"
    assert holdout_season() not in [f.val_season for f in tuning]
    assert max(season_sort_key(f.val_season) for f in tuning) < season_sort_key(holdout_season())


def test_the_holdout_fold_is_disjoint_from_every_tuning_fold():
    """It is only a read on overfitting if the tuning never saw it."""
    ft = _ft(["2020/2021", "2021/2022", "2022/2023", "2023/2024", holdout_season()])
    hold = cv.holdout_fold(ft)
    assert hold is not None and hold.val_season == holdout_season()
    for f in cv.tuning_folds(ft):
        assert not (hold.val & f.val).any(), "holdout rows validate in a tuning fold"
        assert not (hold.val & f.train).any(), "holdout rows train in a tuning fold"


def test_a_table_with_no_holdout_season_says_so_rather_than_guessing():
    ft = _ft(["2020/2021", "2021/2022"])
    assert cv.holdout_fold(ft) is None


def test_a_holdout_that_would_leave_nothing_to_tune_on_raises():
    ft = _ft([holdout_season()])
    with pytest.raises(ValueError, match="no tuning folds"):
        cv.tuning_folds(ft)


# --- Checkpoint identity ----------------------------------------------------


def test_the_feature_set_is_part_of_the_checkpoint_identity():
    """The window grid keys points on (L, alpha) alone, which says nothing about
    which features were in the model -- so every row stayed matchable."""
    a = fingerprint_for(["team_avg_goals_for", "is_home"], SEARCH)
    b = fingerprint_for(["team_avg_goals_for", "team_avg_xg_for"], SEARCH)
    assert a != b


def test_ordering_the_same_features_differently_is_the_same_run():
    assert fingerprint_for(["b", "a"], SEARCH) == fingerprint_for(["a", "b"], SEARCH)


def test_the_fold_set_is_part_of_the_checkpoint_identity():
    ft = _ft(["2020/2021", "2021/2022", "2022/2023", "2023/2024", holdout_season()])
    everything = cv.season_folds(ft)
    tuning = cv.tuning_folds(ft)
    assert fingerprint_for(["a"], SEARCH, everything) != fingerprint_for(["a"], SEARCH, tuning)


def test_a_non_tuned_fit_field_changes_identity_but_a_tuned_one_does_not():
    """The tree ceiling and patience change the answer, so they belong in the key.

    The learning rate and `max_bin` are axes the descent *moves*, so folding them
    in would give every candidate its own file and cache nothing at all.
    """
    base = fingerprint_for(["a"], SEARCH)
    assert base != fingerprint_for(["a"], PROBE), "ceiling and patience must count"

    moved_axis = FitConfig(learning_rate=0.2, n_estimators=SEARCH.n_estimators,
                           early_stopping_rounds=SEARCH.early_stopping_rounds,
                           max_bin=256, nthread=SEARCH.nthread)
    assert base == fingerprint_for(["a"], moved_axis), "tuned axes must not count"


def test_the_fingerprint_reaches_the_filename():
    from fpp.paths import search_checkpoint

    plain = search_checkpoint("goals", "window_grid")
    tagged = search_checkpoint("goals", "window_grid", fingerprint="deadbeef")
    assert plain != tagged
    assert "deadbeef" in tagged.name
    assert tagged.name.startswith("goals__window_grid__")
