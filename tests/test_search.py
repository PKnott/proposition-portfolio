"""Guards for the search stages.

The bug these exist for: a feature group that earns no place in the model has an
*empty* winning subset, which round-trips through the checkpoint CSV as NaN and
came back as the literal string ``'nan'`` -- a feature name no column matches.
It reached a fit and crashed there, having first inflated the selected set by 11
phantom entries and printed a misleading log.

The general shape is "pick a winner from candidates, handle the no-winner case",
so the tests below cover both the parsing and the selection side of it.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from fpp.search.runner import append_row, best_row, load_done, run_grid
from fpp.search.stages import (
    _best_per_size,
    _run_choice_pass,
    _run_group_elimination,
    _run_stage_with_retry,
    _subset_from_row,
    _subsets,
)
from fpp.spec import (
    ALL_FEATURE_NAMES,
    CARRY_THROUGH_GROUPS,
    CONTEXT_FEATURES,
    MERGED_GROUP_SOURCES,
    NON_CANDIDATE_STATS,
    STAT_FAMILIES,
    feature_groups,
    merged_groups,
    validate,
)


# --- Parsing a checkpoint cell back into a feature list -------------------


@pytest.mark.parametrize("value", [float("nan"), np.nan, None, pd.NA])
def test_absent_subset_parses_to_empty_list(value) -> None:
    """A group with no winner must come back as an absence, not a placeholder."""
    assert _subset_from_row(value) == []


def test_absent_subset_never_yields_the_string_nan() -> None:
    """The specific regression: str(NaN) == 'nan', which is truthy."""
    assert "nan" not in _subset_from_row(float("nan"))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", []),
        ("a", ["a"]),
        ("a|b", ["a", "b"]),
        ("a||b", ["a", "b"]),  # empty segments dropped
    ],
)
def test_subset_parsing(value: str, expected: list[str]) -> None:
    assert _subset_from_row(value) == expected


def test_empty_subset_round_trips_through_a_real_csv(tmp_path) -> None:
    """End-to-end of the actual failure path: write "", read back, parse."""
    p = tmp_path / "ckpt.csv"
    pd.DataFrame([{"subset": "", "score_mean": 1.0, "score_se": 0.01}]).to_csv(p, index=False)
    row = pd.read_csv(p).iloc[0]
    assert pd.isna(row["subset"]), "precondition: pandas reads an empty cell as NaN"
    assert _subset_from_row(row["subset"]) == []


# --- Appending rows that do not all share a schema ------------------------


def test_append_row_aligns_a_narrower_row_to_the_header(tmp_path) -> None:
    """The stage-4/stage-5 corruption: two key sets, one file, one header.

    Stage 4 writes ``groups``; stage 5 does not. A plain append writes the second
    row's own columns in its own order, so every value from ``groups`` onward
    lands one column to the left -- ``score_mean`` reads back the standard error,
    which is roughly 200x smaller and therefore wins every selection it enters.
    """
    p = tmp_path / "ckpt.csv"
    append_row(p, {"point_id": "a", "stage": "groups", "subset": "x",
                   "groups": "g1", "score_mean": 1.88, "score_se": 0.009})
    append_row(p, {"point_id": "b", "stage": "exhaustive", "subset": "y",
                   "score_mean": 1.89, "score_se": 0.010})

    out = pd.read_csv(p).set_index("point_id")
    assert out.loc["b", "score_mean"] == 1.89, "score_mean must not read score_se"
    assert out.loc["b", "score_se"] == 0.010
    assert pd.isna(out.loc["b", "groups"]), "an absent key is absent, not a shift"
    assert out.loc["a", "groups"] == "g1"


def test_append_row_widens_the_file_for_a_genuinely_new_column(tmp_path) -> None:
    """A new key must not be silently dropped to preserve the header."""
    p = tmp_path / "ckpt.csv"
    append_row(p, {"point_id": "a", "score_mean": 1.0})
    append_row(p, {"point_id": "b", "score_mean": 2.0, "n_features": 3})

    out = pd.read_csv(p).set_index("point_id")
    assert list(out["score_mean"]) == [1.0, 2.0]
    assert out.loc["b", "n_features"] == 3
    assert pd.isna(out.loc["a", "n_features"])


def test_append_row_keeps_point_ids_findable_across_schemas(tmp_path) -> None:
    """`load_done` drives resume, so it must see both row shapes."""
    p = tmp_path / "ckpt.csv"
    append_row(p, {"point_id": "a", "stage": "groups", "groups": "g1", "score_mean": 1.0})
    append_row(p, {"point_id": "b", "stage": "exhaustive", "score_mean": 2.0})
    assert load_done(p) == {"a", "b"}


# --- One checkpoint, several runs -----------------------------------------


def _counting_scorer(useful: set[str]):
    """`_scorer`, plus a call log -- so a test can prove a cache hit or miss."""
    calls: list[list[str]] = []

    def score(feats: list[str]) -> "_FakeResult":
        calls.append(list(feats))
        return _FakeResult(1.0 - 0.1 * len(set(feats) & useful))

    score.calls = calls
    return score


def test_run_grid_returns_only_the_points_asked_about(tmp_path) -> None:
    """A checkpoint outlives the run that wrote it.

    Stage 4's group combinations depend on which groups survived stage 3, so a
    re-run with a different survivor set writes into a file that already holds
    the previous set's rows. Returning the whole file let those rows compete in,
    and win, the current run's selection -- and appear in its printed trial table
    under group names it had already dropped.
    """
    p = tmp_path / "c.csv"
    run_grid([{"stage": "groups", "subset": ["a"]}, {"stage": "groups", "subset": ["b"]}],
             _scorer({"a"}), p, verbose=False)
    out = run_grid([{"stage": "groups", "subset": ["b"]}], _scorer({"a"}), p, verbose=False)

    assert len(out) == 1, f"expected only the requested point, got {len(out)} rows"
    assert out.iloc[0]["subset"] == "b"


def test_a_changed_backdrop_forces_a_fresh_evaluation(tmp_path) -> None:
    """The stage-2/3 gap: the fitted model is `outside + subset`, but only
    `subset` used to reach the key.

    Change `SELECTION_TOL` and stage 1 keeps a different set, so stage 2's
    backdrop changes while its point ids do not -- and the checkpoint served back
    scores measured against a backdrop that no longer exists.
    """
    p = tmp_path / "c.csv"
    groups = {"g": [["team_avg_xg_for"], ["team_avg_npxg_for"]]}

    first = _counting_scorer({"team_avg_goals_for"})
    _run_choice_pass("stage2", groups, lambda gid: ["team_avg_goals_for"],
                     first, p, "goals", tol=0.001)
    assert len(first.calls) == 2, "precondition: both options scored on the first pass"

    second = _counting_scorer({"team_avg_goals_for"})
    _run_choice_pass("stage2", groups, lambda gid: ["opp_avg_goals_for"],
                     second, p, "goals", tol=0.001)
    assert len(second.calls) == 2, (
        "a different backdrop must be a different point, not a cache hit -- "
        f"scored {len(second.calls)} of 2 options"
    )


def test_an_unchanged_backdrop_still_resumes_from_cache(tmp_path) -> None:
    """The other half: checkpointing must survive the fix.

    Invalidating on backdrop is only worth having if an identical re-run is still
    free -- otherwise every interrupted multi-hour search restarts from zero.
    """
    p = tmp_path / "c.csv"
    groups = {"g": [["team_avg_xg_for"], ["team_avg_npxg_for"]]}
    backdrop = lambda gid: ["team_avg_goals_for"]  # noqa: E731

    first = _counting_scorer({"team_avg_goals_for"})
    _run_choice_pass("stage2", groups, backdrop, first, p, "goals", tol=0.001)
    assert len(first.calls) == 2

    second = _counting_scorer({"team_avg_goals_for"})
    _run_choice_pass("stage2", groups, backdrop, second, p, "goals", tol=0.001)
    assert second.calls == [], f"identical re-run should be free, scored {len(second.calls)}"


# --- Choosing a winner ----------------------------------------------------


def test_best_row_picks_the_lowest_score() -> None:
    df = pd.DataFrame({
        "score_mean": [1.0, 0.5, 2.0],
        "score_se": [0.01, 0.01, 0.01],
        "subset": ["a", "b", "c"],
    })
    assert best_row(df, tol=0.0)["subset"] == "b"


def test_best_row_prefers_the_simpler_candidate_within_tolerance() -> None:
    """Candidates inside the tolerance band go to the smaller feature set."""
    df = pd.DataFrame({
        "score_mean": [1.000, 0.999],
        "score_se": [0.10, 0.10],
        "n_features": [1, 5],
        "subset": ["a", "a|b|c|d|e"],
    })
    assert best_row(df, tol=0.05, prefer=("n_features",))["subset"] == "a"


def test_band_is_absolute_and_ignores_fold_noise() -> None:
    """The band is a fixed log-loss epsilon, not a multiple of the fold SE.

    A noisy also-ran must not widen it. Under the old SE-scaled rule the 0.50 SE
    here would have opened the band far enough to admit the 1.40 candidate and --
    with `prefer` -- select it as "simpler".
    """
    df = pd.DataFrame({
        "score_mean": [1.00, 1.40],
        "score_se": [0.001, 0.50],
        "n_features": [5, 1],
        "subset": ["a|b|c|d|e", "z"],
    })
    assert best_row(df, tol=0.05, prefer=("n_features",))["subset"] == "a|b|c|d|e"


def test_tolerance_boundary_is_exactly_tol() -> None:
    """Inside by 0.0005, outside by 0.002 -- whatever the standard errors say.

    Pins the headline change: at the project's default tol of 0.001 a candidate
    half a thousandth worse is eligible and one two thousandths worse is not.
    The wildly different SEs are there to prove they play no part.
    """
    df = pd.DataFrame({
        "score_mean": [1.0000, 1.0005, 1.0020],
        "score_se": [0.0001, 0.5000, 0.5000],
        "n_features": [3, 2, 1],
        "subset": ["a|b|c", "a|b", "a"],
    })
    # 1.0005 is within 1.0000 + 0.001; 1.0020 is not. Smallest eligible wins.
    assert best_row(df, tol=0.001, prefer=("n_features",))["subset"] == "a|b"
    # Widen past 0.002 and the single-feature set becomes reachable.
    assert best_row(df, tol=0.005, prefer=("n_features",))["subset"] == "a"


def test_best_row_raises_clearly_when_nothing_scored() -> None:
    """All-NaN scores previously raised an opaque IndexError from .iloc[0]."""
    df = pd.DataFrame({
        "score_mean": [np.nan, np.nan],
        "score_se": [np.nan, np.nan],
        "subset": ["a", "b"],
    })
    with pytest.raises(ValueError, match="no candidate produced a score"):
        best_row(df)


def test_best_row_raises_on_empty_input() -> None:
    with pytest.raises(ValueError, match="no results"):
        best_row(pd.DataFrame(columns=["score_mean", "score_se"]))


def test_best_row_tolerates_a_missing_se_column_value() -> None:
    """`score_se` is only a tie-break now, so a NaN in it must not break selection."""
    df = pd.DataFrame({
        "score_mean": [1.0, 2.0],
        "score_se": [np.nan, 0.1],
        "subset": ["a", "b"],
    })
    assert best_row(df)["subset"] == "a"


def test_best_row_reports_band_versus_fallback() -> None:
    """The diagnostic that says whether `tol` was wide enough to admit anything."""
    df = pd.DataFrame({
        "score_mean": [1.0, 1.5],
        "score_se": [0.01, 0.01],
        "n_features": [2, 1],
        "subset": ["a|b", "a"],
    })
    _, source = best_row(df, tol=0.6, prefer=("n_features",), anchor=1.0, return_source=True)
    assert source == "band"

    _, source = best_row(df, tol=0.001, prefer=("n_features",), anchor=0.5, return_source=True)
    assert source == "fallback", "nothing within tol of the anchor should report a fallback"


# --- The invariant the crash violated -------------------------------------


def test_group_winners_only_ever_yield_real_feature_names() -> None:
    """Simulates a pass over groups where some earn no place.

    This is the assembly step that produced 26 real features + 11 phantoms.
    """
    groups = {"kept": ["team_avg_goals_for"], "dropped": ["team_avg_xg_for"]}
    rows = pd.DataFrame([
        {"group": "kept", "subset": "team_avg_goals_for", "score_mean": 1.0, "score_se": 0.01},
        {"group": "dropped", "subset": "", "score_mean": 1.0, "score_se": 0.01},
    ])
    # Round-trip through CSV semantics, which is where the empty subset becomes NaN.
    rows["subset"] = rows["subset"].replace("", np.nan)

    winners = {
        gid: _subset_from_row(best_row(rows[rows["group"] == gid], tol=0.001).get("subset"))
        for gid in groups
    }
    selected = [f for gid in groups for f in winners[gid]]

    assert winners["dropped"] == []
    assert selected == ["team_avg_goals_for"]
    assert not set(selected) - set(ALL_FEATURE_NAMES), "selection leaked a non-feature"


# --- The tolerance rule has to actually bite -------------------------------


def test_tol_is_a_no_op_without_a_prefer_column() -> None:
    """Documents the trap: a tolerance band with nothing to sort on does nothing.

    This is why every stage must record `n_features` and pass `prefer` -- the
    band was being computed and discarded, so selection was a plain argmin and
    argmin nearly always means "keep more features".
    """
    df = pd.DataFrame({
        "score_mean": [1.000, 0.999],
        "score_se": [0.10, 0.10],
        "n_features": [1, 5],
        "subset": ["a", "a|b|c|d|e"],
    })
    assert best_row(df, tol=0.05)["subset"] == "a|b|c|d|e"           # argmin
    assert best_row(df, tol=0.05, prefer=("n_features",))["subset"] == "a"  # rule applied


def test_anchor_opens_a_wider_band_than_near_best() -> None:
    """Anchoring to the full-model loss is what lets the size tie-break strip.

    Near-best keeps only what is within noise of this pass's winner; anchored to
    a full model that scored worse, a much smaller subset becomes eligible.

    The 3-feature row is at 1.06 against a band of ``1.00 + 0.05``. It used to be
    at 1.05 -- exactly the edge -- which made the test turn on whether the band
    is inclusive rather than on the rule it means to describe. It is inclusive
    (see `test_the_band_edge_is_inside_the_band`), so that row was eligible and
    near-best correctly returned it. Put it clearly outside instead.
    """
    df = pd.DataFrame({
        "score_mean": [1.00, 1.06, 1.20],
        "score_se": [0.01, 0.01, 0.01],
        "n_features": [6, 3, 1],
        "subset": ["a|b|c|d|e|f", "a|b|c", "a"],
    })
    near_best = best_row(df, tol=0.05, prefer=("n_features",))
    assert near_best["subset"] == "a|b|c|d|e|f", "nothing else is within tol of the best"

    anchored = best_row(df, tol=0.05, prefer=("n_features",), anchor=1.25)
    assert anchored["subset"] == "a", "anchored selection should reach the smallest eligible set"


def test_the_band_edge_is_inside_the_band() -> None:
    """A candidate exactly ``tol`` worse than the best is *within* tolerance.

    Pinned because it is a real decision, not an accident of ``<=``: the greedy
    path's own guard is ``score > anchor + tol -> stop``, and `SELECTION_TOL` is
    documented as "within this much of the reference score". All three have to
    agree, or a candidate is eligible in one place and not in another.
    """
    df = pd.DataFrame({
        "score_mean": [1.00, 1.05],
        "score_se": [0.01, 0.01],
        "n_features": [6, 3],
        "subset": ["a|b|c|d|e|f", "a|b|c"],
    })
    assert best_row(df, tol=0.05, prefer=("n_features",))["subset"] == "a|b|c"


def test_anchor_falls_back_when_nothing_is_within_the_band() -> None:
    df = pd.DataFrame({
        "score_mean": [2.0, 3.0],
        "score_se": [0.01, 0.01],
        "n_features": [1, 2],
        "subset": ["a", "a|b"],
    })
    assert best_row(df, prefer=("n_features",), anchor=0.5)["subset"] == "a"


def test_shap_breaks_remaining_ties() -> None:
    df = pd.DataFrame({
        "score_mean": [1.0, 1.0],
        "score_se": [0.01, 0.01],
        "n_features": [1, 1],
        "subset": ["a", "b"],
    })
    shap = pd.Series({"a": 0.1, "b": 0.9})
    assert best_row(df, prefer=("n_features",), shap=shap)["subset"] == "b"


# --- Group structure ------------------------------------------------------


def test_feature_groups_structure() -> None:
    validate()
    groups = feature_groups()

    assert len(groups) == 10, f"expected 10 stage-1 groups, got {len(groups)}"

    # Eight stat groups of six -- a family's three stats times both venue forms,
    # which is what makes goals argue against xG directly. Pinned as a rule
    # rather than a list of sizes, so that adding a *context* feature moves the
    # runtime without failing a test about stat grouping.
    stat_groups = {k: v for k, v in groups.items() if k not in ("context", "movement")}
    assert len(stat_groups) == 8
    assert all(len(v) == 6 for v in stat_groups.values()), \
        {k: len(v) for k, v in stat_groups.items()}
    assert len(groups["movement"]) == 4
    assert set(groups["context"]) | set(groups["movement"]) == set(CONTEXT_FEATURES)

    grouped = [f for members in groups.values() for f in members]
    assert len(grouped) == len(set(grouped)), "a feature appears in two groups"
    # Venue is folded in, so the groups partition the WHOLE candidate space.
    assert set(grouped) == set(ALL_FEATURE_NAMES)

    # Evaluations in a stage-1 pass; the number is the runtime. It is derived
    # rather than typed because the context group is the one that grows: adding
    # `matches_14d` took it from 5 members to 7, so the pass went from
    # 32 + 16 + 8*64 = 560 to 128 + 16 + 8*64 = 656.
    expected = sum(2 ** len(v) for v in groups.values())
    assert expected == 2 ** len(groups["context"]) + 2 ** len(groups["movement"]) + 8 * 2 ** 6
    assert expected == 656


def test_points_is_not_a_candidate() -> None:
    """Excluded by decision, and the exclusion has to be real, not merely unused."""
    assert "points" in NON_CANDIDATE_STATS
    assert not [f for f in ALL_FEATURE_NAMES if "points" in f]
    assert not [f for members in feature_groups().values() for f in members if "points" in f]


def test_merged_groups_are_built_from_stage1_survivors() -> None:
    """Stage 2 merges end-product with process -- over survivors only.

    Merging the raw groups would be 12 members and 4,096 subsets each; the whole
    point of restricting to survivors is that it keeps the merge affordable.
    """
    survivors = {
        "team_attack": ["team_avg_goals_for"],
        "team_for": ["team_avg_shots_for", "team_avg_sot_for_v"],
        "team_defence": [],                      # lost everything in isolation
        "team_against": ["team_avg_corners_against"],
        "opp_attack": ["opp_avg_xg_for"],
        "opp_for": [],
        "opp_defence": ["opp_avg_goals_against"],
        "opp_against": [],
        "context": ["is_home", "league"],
        "movement": [],
    }
    merged = merged_groups(survivors)

    # end-product and process meet in one group, and only survivors appear
    assert merged["team_attack_merged"] == [
        "team_avg_goals_for", "team_avg_shots_for", "team_avg_sot_for_v"
    ]
    # a group that lost everything contributes nothing
    assert merged["team_defence_merged"] == ["team_avg_corners_against"]
    assert merged["opp_attack_merged"] == ["opp_avg_xg_for"]
    # carry-throughs keep their own identity rather than being folded in
    assert merged["context"] == ["is_home", "league"]
    assert merged["movement"] == []
    # merged names are distinct from stage-1 names -- they are different objects
    assert not set(merged) & (set(survivors) - set(CARRY_THROUGH_GROUPS))


def test_every_stage1_group_reaches_stage_2() -> None:
    """No group's survivors may vanish between stages by simply not being named."""
    groups = set(feature_groups())
    sourced = {s for srcs in MERGED_GROUP_SOURCES.values() for s in srcs}
    assert groups <= sourced | set(CARRY_THROUGH_GROUPS)


def test_trial_rows_are_sorted_best_first_and_named() -> None:
    """The per-subset output has to be readable and free of placeholders."""
    from fpp.search.stages import _trial_rows

    sub = pd.DataFrame({
        "subset": ["a|b", np.nan, "a"],
        "score_mean": [1.20, 1.50, 1.10],
        "score_se": [0.01, 0.01, 0.01],
    })
    rows = _trial_rows(sub)

    assert [s for s, _ in rows] == [1.10, 1.20, 1.50], "trials must be best-first"
    assert rows[0][1] == ["a"] and rows[1][1] == ["a", "b"]
    assert rows[2][1] == [], "the empty subset is an absence, not a placeholder"
    names = [n for _, feats in rows for n in feats]
    assert "nan" not in names and "<NA>" not in names


def test_stat_groups_carry_both_venue_forms() -> None:
    """The venue decision is made inside the group's own search, not afterwards."""
    groups = feature_groups()
    for gid, members in groups.items():
        if gid in ("context", "movement"):
            continue
        venue = [m for m in members if m.endswith("_v")]
        plain = [m for m in members if not m.endswith("_v")]
        assert len(venue) == len(plain), f"{gid} is not an even venue split: {members}"
        for p in plain:
            assert f"{p}_v" in venue, f"{gid}: {p} has no venue form in its own group"


def test_a_whole_family_shares_one_group() -> None:
    """Goals, xG and npxG must fight each other inside one group.

    This is the regression that produced a worse-than-baseline model: splitting a
    family across groups meant each stat argued alone against a backdrop that
    still contained its own near-duplicates, so nothing looked necessary and
    stage 1 discarded too much for the later stages to recover from.
    """
    groups = feature_groups()

    attack = groups["team_attack"]
    for stat in STAT_FAMILIES["end_product"]:          # goals, xg, npxg
        assert f"team_avg_{stat}_for" in attack, f"{stat} is not in team_attack"
        assert f"team_avg_{stat}_for_v" in attack, f"{stat} venue form is not in team_attack"

    process = groups["team_for"]
    for stat in STAT_FAMILIES["process"]:              # shots, sot, corners
        assert f"team_avg_{stat}_for" in process, f"{stat} is not in team_for"

    # ...and the two families stay apart: they measure different things.
    assert not set(attack) & set(process)


def test_league_priors_are_not_candidates() -> None:
    from fpp.spec import LEAGUE_FEATURE_NAMES

    assert LEAGUE_FEATURE_NAMES, "the constant should still be defined for reference"
    assert not set(LEAGUE_FEATURE_NAMES) & set(ALL_FEATURE_NAMES)


# --- Stage 4: group-level elimination -------------------------------------


class _FakeResult:
    """Minimal stand-in for CVResult, so these tests need no model fitting."""

    def __init__(self, mean: float) -> None:
        self.mean = mean
        self.se = 0.001
        self.per_fold = {"f": mean}
        self.n_scored = 200
        self.best_iterations = [10]
        self.dispersion = {}


def _scorer(useful: set[str]):
    """Lower loss the more *useful* features are present; useless ones cost nothing."""

    def score(feats: list[str]) -> _FakeResult:
        return _FakeResult(1.0 - 0.1 * len(set(feats) & useful))

    return score


def _full_model_anchor(bundles: dict[str, list[str]], score) -> float:
    """The loss of keeping every group -- which is what ``anchor`` means.

    Computed rather than typed. These tests used to pass ``anchor=1.0``, which
    for `_scorer` is the score of a model with *no* useful features at all. A
    band of ``no_signal + tol`` admits every subset, including ones carrying
    nothing, and "keep the smallest within tolerance" then correctly strips to a
    single group -- so the stage looked like it was choosing when it was really
    just taking whatever was smallest. It happened to give the right answer
    whenever exactly one group was useful, and the wrong one as soon as two were.
    """
    return score([f for feats in bundles.values() for f in feats]).mean


def test_group_elimination_drops_groups_that_add_nothing(tmp_path) -> None:
    bundles = {
        "keeps": ["team_avg_goals_for"],
        "useless_a": ["team_avg_xg_for"],
        "useless_b": ["team_avg_npxg_for"],
    }
    score = _scorer({"team_avg_goals_for"})
    feats, kept = _run_group_elimination(
        bundles, score, tmp_path / "g.csv", "goals", tol=0.001,
        exhaustive_max=10, anchor=_full_model_anchor(bundles, score),
    )
    assert kept == ["keeps"]
    assert feats == ["team_avg_goals_for"]


def test_group_elimination_returns_empty_for_no_live_groups(tmp_path) -> None:
    feats, kept = _run_group_elimination(
        {"a": [], "b": []}, _scorer(set()), tmp_path / "g.csv", "goals", tol=0.001,
    )
    assert feats == [] and kept == []
    assert "nan" not in feats, "an absent winner must never become a placeholder"


def test_group_elimination_greedy_and_exhaustive_agree(tmp_path) -> None:
    """The greedy path is only a cost control -- it must not change the answer.

    Two groups are useful here, not one, which is what makes this the case that
    catches things: with a single useful group both paths converge on it however
    they tie-break, so the invariant is never actually tested.
    """
    bundles = {
        "a": ["team_avg_goals_for"],
        "b": ["team_avg_xg_for"],
        "c": ["team_avg_npxg_for"],
        "d": ["team_avg_shots_for"],
    }
    useful = {"team_avg_goals_for", "team_avg_shots_for"}
    score = _scorer(useful)
    anchor = _full_model_anchor(bundles, score)

    ex_feats, ex_kept = _run_group_elimination(
        bundles, score, tmp_path / "ex.csv", "goals",
        tol=0.001, exhaustive_max=10, anchor=anchor,
    )
    gr_feats, gr_kept = _run_group_elimination(
        bundles, score, tmp_path / "gr.csv", "goals",
        tol=0.001, exhaustive_max=0, anchor=anchor,
    )
    assert set(ex_kept) == set(gr_kept) == {"a", "d"}
    assert set(ex_feats) == set(gr_feats) == useful


def test_the_two_paths_agree_even_when_every_tie_break_ties(tmp_path) -> None:
    """The failure mode `best_row`'s last sort key exists for.

    Every group here is equally useful, so every same-sized subset ties on size,
    loss and stability at once. With nothing after those the winner was whichever
    row the frame happened to hold first -- and the two paths build their frames
    in different orders, so exhaustive returned one group and greedy returned a
    different one. Both must now name the same group, and it must be the one the
    candidates themselves determine rather than the enumeration.
    """
    bundles = {g: [f] for g, f in (
        ("a", "team_avg_goals_for"), ("b", "team_avg_xg_for"),
        ("c", "team_avg_npxg_for"), ("d", "team_avg_shots_for"),
    )}
    score = _scorer(set())          # nothing helps, so every subset scores 1.0

    kept = []
    for name, ex_max in (("ex", 10), ("gr", 0)):
        _feats, k = _run_group_elimination(
            bundles, score, tmp_path / f"{name}.csv", "goals", tol=0.001,
            exhaustive_max=ex_max, anchor=_full_model_anchor(bundles, score),
            print_trials=False,
        )
        kept.append(k)
    assert kept[0] == kept[1], f"exhaustive {kept[0]} vs greedy {kept[1]}"
    # Decided by the subset string, so it is a property of the data.
    assert kept[0] == ["a"]


def test_group_elimination_never_emits_a_placeholder(tmp_path) -> None:
    """The `groups` column is the new exposure for the 'nan' bug."""
    bundles = {"a": ["team_avg_goals_for"], "b": ["team_avg_xg_for"]}
    score = _scorer({"team_avg_goals_for"})
    feats, kept = _run_group_elimination(
        bundles, score, tmp_path / "g.csv", "goals", tol=0.001,
        exhaustive_max=10, anchor=_full_model_anchor(bundles, score),
    )
    assert "nan" not in kept and "<NA>" not in kept
    assert not set(feats) - set(ALL_FEATURE_NAMES)


# --- Stage 1-3 shared helper ----------------------------------------------


def test_choice_pass_drops_a_group_whose_empty_option_wins(tmp_path) -> None:
    """End-to-end of the original bug through the new shared helper."""
    groups = {"useless": [[], ["team_avg_xg_for"]]}
    winners, _sources = _run_choice_pass(
        "stage1", groups, lambda gid: ["team_avg_goals_for"],
        _scorer({"team_avg_goals_for"}), tmp_path / "c.csv", "goals", tol=0.001,
    )
    assert winners["useless"] == [], f"expected a dropped group, got {winners['useless']}"
    assert "nan" not in winners["useless"]


def test_choice_pass_skips_groups_with_no_options(tmp_path) -> None:
    winners, _sources = _run_choice_pass(
        "stage2", {"gone": []}, lambda gid: [],
        _scorer(set()), tmp_path / "c.csv", "goals", tol=0.001,
    )
    assert winners["gone"] == []


# --- Stage 5: the n=1..k sweep over individual features -------------------


def test_final_sweep_covers_every_size_over_individual_features() -> None:
    """The final stage is a full n=1..k sweep, not a single pass.

    The candidates are individual feature *names* flattened out of the surviving
    groups -- a group that carried two features contributes both separately here,
    which is what lets the sweep find redundancy spanning two different groups.
    """
    stage3 = [
        "team_avg_xg_for",          # from team_quality_for
        "team_avg_npxg_for",        # also team_quality_for -- same group, separate here
        "opp_avg_goals_against",    # from opp_goals_against
        "is_home",                  # from context
    ]
    combos = [list(s) for s in _subsets(stage3) if s]

    assert len(combos) == 2 ** len(stage3) - 1 == 15
    assert sorted({len(c) for c in combos}) == [1, 2, 3, 4], "a size was skipped"
    assert [stage3[0]] in combos, "the n=1 singletons must be scored"
    assert stage3 in combos, "the full set must be scored"
    # Two members of the same group must appear as their own 2-combination.
    assert ["team_avg_xg_for", "team_avg_npxg_for"] in combos


def test_retry_shrinks_the_tolerance_until_the_combined_output_passes(tmp_path, capsys) -> None:
    """The inner tolerance is per-group; the outer check is on the combination.

    A tolerance loose enough to let each group drop its useful member compounds
    into a combined set far worse than any single group's band suggested. The
    retry shrinks the inner dial until the combined result clears the budget.
    """
    members = {"a": ["team_avg_goals_for"], "b": ["team_avg_xg_for"]}
    useful = {"team_avg_goals_for", "team_avg_xg_for"}

    # Budget reachable only when both useful features survive.
    winners, _sources, info = _run_stage_with_retry(
        "stage1", members, lambda gid: [], _scorer(useful), tmp_path / "c.csv",
        "goals", tol=0.5, budget=0.85, budget_label="full-model budget",
        print_trials=False,
    )

    assert info["attempts"] > 1, "a too-loose tolerance must trigger at least one retry"
    assert info["tol"] < 0.5, "the inner tolerance must have shrunk"
    assert not info["fell_back"]
    assert sorted(f for w in winners.values() for f in w) == sorted(useful)

    out = capsys.readouterr().out
    # `report()` prints one line per *distinct* result, not per attempt: a shrink
    # that reproduces the previous set has nothing new to say, and on the way to
    # the floor there are dozens of those. So the failing run appears as
    # "attempts 1-8", not "attempt 1" -- assert on the verdicts and on attempt 1
    # being accounted for, rather than on a format that deliberately collapses.
    assert "FAIL" in out and "PASS" in out, "each distinct result's verdict must be visible"
    assert re.search(r"attempts? 1\b", out), f"attempt 1 must be accounted for:\n{out}"
    verdicts = re.findall(r"attempts? ([\d-]+):.*?(PASS|FAIL)", out)
    assert verdicts[0][1] == "FAIL", "the first reported run is the one that failed"
    assert verdicts[-1][1] == "PASS", "the last reported run is the one that passed"


def test_a_stage_that_can_never_pass_falls_back_to_full_membership(tmp_path, capsys) -> None:
    """The floor: try tol=0, then stop compressing rather than fail the run.

    An unreachable budget must not raise and must not ship a pruned set -- it
    hands back full membership, unchecked, and says so.
    """
    members = {"a": ["team_avg_goals_for"], "b": ["team_avg_xg_for"]}

    winners, _sources, info = _run_stage_with_retry(
        "stage1", members, lambda gid: [], _scorer({"team_avg_goals_for"}),
        tmp_path / "c.csv", "goals", tol=0.5,
        budget=-1.0,                      # no feature set can ever score this low
        budget_label="full-model budget", print_trials=False,
    )

    assert info["fell_back"] is True
    assert info["tol"] == 0.0, "the floor attempt must be tol=0 exactly"
    assert winners == members, "fallback keeps every group's full original membership"

    out = capsys.readouterr().out
    assert "FLOOR REACHED" in out
    assert "tol=0" in out, "the final zero-tolerance attempt must be announced"


def test_every_retry_reculls_from_full_membership(tmp_path) -> None:
    """Never compound a tighter tolerance onto an already-pruned set.

    Culling the previous attempt's survivors would search a space the new
    tolerance was never applied to, and a feature dropped by a loose pass could
    never come back however tight the dial got.
    """
    members = {"a": ["team_avg_goals_for", "team_avg_xg_for"]}

    winners, _sources, info = _run_stage_with_retry(
        "stage1", members, lambda gid: [], _scorer({"team_avg_goals_for", "team_avg_xg_for"}),
        tmp_path / "c.csv", "goals", tol=0.5, budget=0.85,
        budget_label="full-model budget", print_trials=False,
    )

    # Only reachable if the second attempt could still see both members.
    assert info["attempts"] > 1
    assert sorted(winners["a"]) == ["team_avg_goals_for", "team_avg_xg_for"]


def test_stage5_takes_the_outright_best_not_the_smallest_within_tol() -> None:
    """Stage 5 is the one stage that does NOT prefer the smaller candidate.

    Stages 1-4 have already applied that pressure four times, and stage 5 scores
    every combination rather than sampling, so a fifth application only concedes
    score. The numbers here are from a real goals run that kept the 5-feature set
    at 1.461710 over the 6-feature set at 1.460348.

    `best_row(sub, tol=0.0)` with no `prefer` column is how the stage-5 call site
    spells "outright best" -- deliberately the degenerate case that
    `test_tol_is_a_no_op_without_a_prefer_column` pins.
    """
    sub = pd.DataFrame({
        "subset": ["a|b|c|d|e", "a|b|c|d|e|f"],
        "score_mean": [1.461710, 1.460348],
        "score_se": [0.0028, 0.0028],
        "n_features": [5, 6],
    })

    assert best_row(sub, tol=0.0)["subset"] == "a|b|c|d|e|f", "stage 5 takes the best score"
    # Stages 1-4 keep the opposite rule, and it must not have moved.
    assert best_row(sub, tol=0.005, prefer=("n_features",))["subset"] == "a|b|c|d|e"


def test_best_per_size_picks_the_lowest_loss_at_each_size() -> None:
    sub = pd.DataFrame({
        "subset": ["a", "b", "a|b", "a|c", "a|b|c"],
        "score_mean": [1.50, 1.40, 1.20, 1.30, 1.10],
        "score_se": [0.01] * 5,
    })
    rows = _best_per_size(sub)

    assert [n for n, _, _ in rows] == [1, 2, 3], "one row per size, ascending"
    assert rows[0][2] == ["b"] and rows[0][1] == pytest.approx(1.40)
    assert rows[1][2] == ["a", "b"] and rows[1][1] == pytest.approx(1.20)
    assert rows[2][2] == ["a", "b", "c"]


def test_best_per_size_never_reports_a_placeholder() -> None:
    """The printed log is held to the same rule as the selection itself."""
    sub = pd.DataFrame({
        "subset": [np.nan, "a"],
        "score_mean": [2.0, 1.0],
        "score_se": [0.01, 0.01],
    })
    rows = _best_per_size(sub)
    names = [f for _, _, feats in rows for f in feats]
    assert "nan" not in names and "<NA>" not in names
    assert (0, 2.0, []) in rows, "the empty subset should report as size 0 with no names"


def test_best_per_size_handles_an_empty_frame() -> None:
    assert _best_per_size(pd.DataFrame(columns=["subset", "score_mean", "score_se"])) == []
    all_nan = pd.DataFrame({"subset": ["a"], "score_mean": [np.nan], "score_se": [np.nan]})
    assert _best_per_size(all_nan) == []
