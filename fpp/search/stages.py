"""The three search stages: window, feature selection, hyperparameters.

Order matters and mirrors the reference framework: settle the data
representation first ``(L, alpha)``, then which features earn their place, then
the learner's hyperparameters, then re-check that the window did not move.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from ..build import FeatureTable, build_feature_table
from ..config import (
    ALPHA_GRID,
    ANCHOR_FULL_MULT,
    FEATURE_EXHAUSTIVE_MAX,
    GROUP_EXHAUSTIVE_MAX,
    DESCENT_PASSES,
    L_GRID,
    MAX_TUNING_ROUNDS,
    MAX_BIN_BLOCK,
    PAIR_BLOCKS,
    PROBE,
    SEARCH,
    RETRY_SHRINK_PCT,
    SELECTION_TOL,
    STAGE_FULL_MULT,
    TWEEDIE_BLOCK,
    WINDOW_RECHECK_ALPHA_SPAN,
    WINDOW_RECHECK_L_SPAN,
    FitConfig,
    holdout_season,
)
from ..cv import evaluate, holdout_fold, season_folds, tuning_folds
from ..models import base_params
from ..spec import (
    ALL_FEATURE_NAMES,
    CORE_FEATURES,
    feature_groups,
    merged_groups,
    scoring_hash,
    target_spec,
)
from .runner import best_row, run_grid

# --- Stage A: the (L, alpha) window ---------------------------------------


def fingerprint_for(features: list[str], fit_cfg: FitConfig,
                    folds: "list | None" = None, **extra) -> str:
    """Everything a cached window or descent score depends on and `point_id` does not.

    A grid point is identified by the axes it sweeps -- ``(L, alpha)`` for the
    window, the hyperparameter names for the descent. Nothing in either says which
    features were in the model or which seasons formed the folds, so a cached row
    stayed matchable by ``point_id`` across changes to both. Folding those into the
    *filename* makes the old rows unreachable rather than silently reusable, which
    is the same rule `run_feature_selection` already applies to its own stages.

    The fit config contributes only the fields that are **not** themselves tuned:
    the tree ceiling, the early-stopping patience and the thread count. The
    learning rate and ``max_bin`` are axes the descent moves, so including them
    would make every candidate its own file and cache nothing.
    """
    payload = {
        "features": sorted(features),
        "fit": {"n_estimators": fit_cfg.n_estimators,
                "early_stopping_rounds": fit_cfg.early_stopping_rounds},
        "scoring": scoring_hash(),
        **extra,
    }
    if folds is not None:
        payload["folds"] = [f.val_season for f in folds]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]


def _plateau_entry(res: pd.DataFrame, tol: float) -> pd.Series:
    """The **near edge of the good zone**, not its lowest point.

    The window surface is a broad flat optimum. Measured on the selected feature
    set, the spread across the whole grid is 10-12 fold standard errors while its
    interior is under a third of one -- so being *in* the zone is worth everything
    and the exact point inside it is worth almost nothing. Given that, the useful
    choice is the cheapest window that is unambiguously inside, because a smaller
    ``L`` needs less history behind each team.

    Four steps:

    1. ``band = best + tol`` over the whole grid.
    2. A point is **in the zone** if its score is inside the band.
    3. A point is a **plateau entry** if it is in the zone *and every larger ``L``
       at its own alpha is also in the zone*.
    4. Smallest ``L`` among the entries, ties broken on score.

    Step 3 is the part `best_row(prefer=("L",))` cannot express, and it is what
    makes this safe. "Smallest ``L`` within tolerance" would happily take a noisy
    dip that fell inside the band by luck with worse scores either side of it --
    the opposite of solidly inside. Requiring the whole tail above a candidate to
    be in the zone means a spurious dip is rejected by its own neighbours, so what
    comes back is the start of a real plateau.
    """
    r = res.dropna(subset=["score_mean"]).copy()
    if r.empty:
        raise ValueError(f"no window candidate produced a score (of {len(res)} rows)")
    band = float(r["score_mean"].min()) + float(tol)
    r["_in"] = r["score_mean"] <= band

    keep = []
    for _alpha, g in r.groupby("alpha", sort=False):
        g = g.sort_values("L", ascending=False)
        # Walking down from the largest L, a point is an entry only while every
        # L above it has been in the zone too.
        tail_all_in = True
        for idx, row in g.iterrows():
            if not row["_in"]:
                tail_all_in = False
                continue
            if tail_all_in:
                keep.append(idx)
    if not keep:
        # No alpha has an unbroken tail -- the surface is too ragged for the rule
        # to mean anything, so fall back to the outright best and say nothing was
        # gained by asking.
        return r.sort_values(["score_mean", "L"]).iloc[0]
    return r.loc[keep].sort_values(["L", "score_mean"]).iloc[0]


def effective_window(L: int, alpha: float) -> int:
    """Matches carrying ~95% of the decay weight -- what ``L`` *means* in practice.

    Weight falls as ``exp(-alpha * age)``, so past roughly ``3 / alpha`` there is
    almost nothing left to include and a larger ``L`` changes nothing. Worth
    reporting next to the chosen window because "depends on two seasons of data"
    is only true at low alpha: at ``alpha >= 0.05`` the tail beyond ~60 matches is
    already inert, so shrinking ``L`` there is cosmetic. At ``alpha <= 0.02`` it
    is real.
    """
    if alpha <= 0:
        return int(L)
    return int(min(L, math.ceil(3.0 / alpha)))


def run_window_grid(
    df: pd.DataFrame,
    bw,
    target: str,
    checkpoint: Path,
    features: list[str],
    *,
    L_grid: list[int] | None = None,
    alpha_grid: list[float] | None = None,
    params: dict | None = None,
    folds_for=season_folds,
    fit_cfg: FitConfig = SEARCH,
) -> pd.DataFrame:
    """Search the ``(L, alpha)`` grid for one target family.

    ``features`` is required, and that is the change. It used to default to
    `CORE_FEATURES` -- ten hardcoded columns -- on the reasoning that smoothing is
    a question the feature set does not affect. Whether or not that holds, it
    meant the window was chosen against a model nobody ever fits, so this now asks
    the question of the model that actually gets used.

    ``params`` lets a later round search the window at the *tuned* learner rather
    than at defaults, which is what closes the loop the old stability recheck left
    open.

    ``folds_for`` is a **callable**, not a fold list, and it has to be: every
    ``(L, alpha)`` builds its own feature table, and `drop_warmup` removes a
    different number of rows at each ``L`` -- 42,982 at one window against 42,962
    at another. Fold masks are boolean arrays sized to one table, so sharing a
    prepared list across candidates broadcasts against the wrong length. Passing
    the *rule* instead means each candidate derives its own masks and every
    candidate still gets the same seasons.
    """
    feats = list(features)
    points = [{"L": L, "alpha": a} for L in (L_grid or L_GRID) for a in (alpha_grid or ALPHA_GRID)]

    def score(p):
        ft = build_feature_table(df, bw, p["L"], p["alpha"], features=feats)
        return evaluate(ft, target, params or base_params(target, fit_cfg),
                        folds_for(ft), features=feats, fit_cfg=fit_cfg)

    return run_grid(points, score, checkpoint, label=f"{target}/window")


def pick_window(res: pd.DataFrame, target: str, tol: float | None = None,
                verbose: bool = True) -> dict:
    """Plateau entry, plus what taking it cost.

    The cost is printed rather than buried: the gap to the outright best in fold
    SE, and the effective window. Both exist so the choice can be checked instead
    of taken on faith -- if the near edge of the plateau is materially worse than
    its floor, that is visible here.
    """
    tol = SELECTION_TOL[target] if tol is None else float(tol)
    pick = _plateau_entry(res, tol)
    r = res.dropna(subset=["score_mean"])
    best = r.loc[r["score_mean"].idxmin()]
    se = float(pick.get("score_se") or float("nan"))
    gap = float(pick["score_mean"]) - float(best["score_mean"])

    out = {"L": int(pick["L"]), "alpha": float(pick["alpha"]),
           "logloss": float(pick["score_mean"]),
           "best_L": int(best["L"]), "best_alpha": float(best["alpha"]),
           "best_logloss": float(best["score_mean"]),
           "gap_vs_best": gap,
           "gap_se": (gap / se) if se == se and se else float("nan"),
           "effective_window": effective_window(int(pick["L"]), float(pick["alpha"]))}
    if verbose:
        print(f"  [{target}] window -> L={out['L']:>3} alpha={out['alpha']:.2f}  "
              f"ll={out['logloss']:.6f}  (effective window {out['effective_window']} matches)")
        print(f"  [{target}]   outright best was L={out['best_L']:>3} "
              f"alpha={out['best_alpha']:.2f} ll={out['best_logloss']:.6f} -- "
              f"plateau entry costs {gap:+.6f} ({out['gap_se']:+.2f} SE) for "
              f"{out['best_L'] - out['L']} fewer matches of history")
    return out


# --- Stage B: feature selection -------------------------------------------


def shap_importance(ft, target: str, features: list[str], fit_cfg: FitConfig = PROBE) -> pd.Series:
    """Mean |SHAP| per feature, via XGBoost's native TreeSHAP.

    ``pred_contribs`` rather than the ``shap`` package: identical output, one
    fewer dependency, and categorical-safe for the ``league`` column.
    """
    from ..build import scorable_mask

    m = scorable_mask(ft, target)
    X = ft.X.loc[m, features]
    model = xgb.XGBRegressor(**base_params(target, fit_cfg))
    model.fit(X, ft.y[target][m], verbose=False)
    dm = xgb.DMatrix(X, enable_categorical=True)
    contribs = model.get_booster().predict(dm, pred_contribs=True)
    return pd.Series(np.abs(contribs[:, :-1]).mean(axis=0), index=features).sort_values(ascending=False)


def _subsets(items: list[str]):
    for r in range(len(items) + 1):
        yield from combinations(items, r)


def _subset_from_row(value) -> list[str]:
    """Parse a checkpoint ``subset`` cell back into a feature list.

    The *empty* subset -- a group earning no place in the model -- is written as
    an empty string, which ``pd.read_csv`` reads back as a missing value.
    Stringifying that yields a placeholder (``'nan'`` for ``np.nan``, ``'<NA>'``
    for ``pd.NA`` under pandas' Arrow-backed string dtype) which is truthy and
    would be smuggled into the selected set as a feature name no column matches.
    An absent winner has to come back as an absence, not as a placeholder.

    ``pd.isna`` covers every missing flavour, but only ask it about scalars --
    on a list it returns an elementwise array.
    """
    if value is None:
        return []
    if np.ndim(value) == 0 and pd.isna(value):
        return []
    return [f for f in str(value).split("|") if f]


def _n_features_row(p: dict, _res) -> dict:
    """Size column for every checkpointed point.

    Without this the `prefer=("n_features",)` tie-break has no column to sort on,
    the tolerance band is silently discarded, and selection degenerates to plain
    argmin -- which almost always means "keep more features".
    """
    return {"n_features": len(p.get("subset", []))}


def _trial_rows(sub: pd.DataFrame, subset_col: str = "subset") -> list[tuple[float, list[str]]]:
    """``[(score, [names]), ...]`` for every scored trial, best first.

    Names go through ``_subset_from_row``, so an empty subset reads as an absence
    rather than printing a placeholder.
    """
    rows = sub.dropna(subset=["score_mean"])
    out = [
        (float(r["score_mean"]), _subset_from_row(r.get(subset_col)))
        for _, r in rows.iterrows()
    ]
    return sorted(out, key=lambda t: t[0])


def _print_trials(
    sub: pd.DataFrame,
    target: str,
    stage: str,
    gid: str,
    winner: list[str],
    reference: float,
    ref_label: str,
    subset_col: str = "subset",
) -> None:
    """Every trial in one group's search, best first, with the winner marked.

    The point is that a bare ``-> [] (dropped)`` is unfalsifiable by eye: it does
    not say whether the group lost by a hair or by a mile. Printing the whole
    slate makes a close call distinguishable from a landslide without opening a
    checkpoint CSV.

    ``delta`` is against whatever the selection rule actually compared to -- the
    group's own best in stage 1, the full-model anchor from stage 2 on -- so the
    numbers explain the choice rather than just describing it.
    """
    trials = _trial_rows(sub, subset_col)
    if not trials:
        return
    win = sorted(winner)
    print(f"  [{target}] {stage} {gid}: {len(trials)} subsets scored "
          f"(delta vs {ref_label} {reference:.6f})")
    for score, feats in trials:
        mark = "  <- winner" if sorted(feats) == win else ""
        shown = feats if feats else "[]  (drop the whole group)"
        print(f"        n={len(feats):>2}  ll={score:.6f}  delta={score - reference:+.5f}  {shown}{mark}")


def _run_choice_pass(
    stage: str,
    options_by_group: dict[str, list[list[str]]],
    outside_of,
    score_features,
    checkpoint: Path,
    target: str,
    tol: float,
    *,
    anchor: float | None = None,
    shap: pd.Series | None = None,
    print_trials: bool = True,
) -> tuple[dict[str, list[str]], Counter]:
    """One coordinate-descent pass over groups.

    Returns ``({group_id: winning bundle}, {"band": n, "fallback": n})`` -- the
    counts say whether ``tol`` was wide enough to admit anything, which is the
    signal for whether the dial needs loosening for this target.

    ``outside_of(gid)`` supplies the features held fixed while ``gid`` is searched.
    """
    winners: dict[str, list[str]] = {}
    sources: Counter = Counter()

    for gid, options in options_by_group.items():
        if not options:
            winners[gid] = []
            continue
        if len(options) == 1:
            # Nothing to choose -- e.g. a group already emptied by an earlier pass.
            winners[gid] = list(options[0])
            continue

        outside = outside_of(gid)
        # The backdrop is a hidden input to every score in this group's grid: the
        # model fitted is `outside + subset`, but only `subset` reaches the point
        # dict and therefore `point_id`. Stage 2's backdrop depends on stage 1's
        # winners and stage 3's on stage 2's, so changing SELECTION_TOL changes
        # what a cached score was measured against while leaving its key -- and
        # the checkpoint filename fingerprint -- untouched.
        #
        # Fingerprint the backdrop's *content* rather than `tol`: it is the
        # backdrop that determines correctness, so drift from any upstream cause
        # is caught, not just a tolerance change. Stage 1 hashes an empty
        # backdrop to a constant, which needs no special-casing.
        backdrop = hashlib.sha256("|".join(sorted(outside)).encode()).hexdigest()[:8]
        points = [{"group": gid, "subset": list(o), "backdrop": backdrop} for o in options]

        def score(p, _outside=outside):
            return score_features(_outside + list(p["subset"]))

        res = run_grid(points, score, checkpoint, label=f"{target}/{stage}/{gid}",
                       extra_row=_n_features_row, verbose=False)
        sub = res[res["group"] == gid] if "group" in res.columns else res
        if sub.empty:
            winners[gid] = list(options[-1])
            continue

        if tol == 0.0:
            # The floor attempt of a stage retry: outright best subset per group,
            # no size preference and no anchor band. This is the least-compressing
            # setting the culling mechanism can produce -- if the combined result
            # still cannot clear the outer check, nothing tighter will, and the
            # stage gives up rather than shrinking further. Same `tol=0.0` spelling
            # as stage 5's outright-best pick.
            pick, source = best_row(sub, tol=0.0, return_source=True)
        else:
            pick, source = best_row(sub, tol=tol, prefer=("n_features",),
                                    anchor=anchor, shap=shap, return_source=True)
        chosen = _subset_from_row(pick.get("subset"))
        winners[gid] = chosen
        sources[source] += 1

        if print_trials:
            ref = anchor if anchor is not None else float(sub["score_mean"].min())
            _print_trials(sub, target, stage, gid, chosen, ref,
                          "full model" if anchor is not None else "group best")
        note = "" if source == "band" else "   (fallback: nothing within tol)"
        print(f"  [{target}] {stage} {gid:22} -> {chosen if chosen else '[] (dropped)'}{note}")
    return winners, sources


# Below this the shrink has stopped meaning anything, so take the one remaining
# distinct attempt -- exactly zero -- rather than iterating toward it forever.
_TOL_FLOOR = 1e-6


def _run_stage_with_retry(
    stage: str,
    members: dict[str, list[str]],
    outside_of,
    score_features,
    checkpoint: Path,
    target: str,
    tol: float,
    *,
    budget: float,
    budget_label: str,
    anchor: float | None = None,
    shap: pd.Series | None = None,
    print_trials: bool = True,
    shrink: float = RETRY_SHRINK_PCT,
) -> tuple[dict[str, list[str]], Counter, dict]:
    """Cull, check the combined result, shrink the tolerance and cull again.

    Two tiers, deliberately separate:

    * The **inner** tolerance (``tol``) governs each group's own culling and is
      what shrinks here. It is per-group, so small concessions across ten groups
      compound into a combined set far worse than any single group's band
      suggests -- which is precisely why checking only per-group was not enough.
    * The **outer** ``budget`` is fixed and is checked against the *combined*
      output. It never shrinks, so retrying cannot move the bar to meet a bad
      result.

    Every attempt re-culls from **full original membership**, never from the
    previous attempt's survivors. Compounding a tighter tolerance onto an
    already-pruned set would search a space the tolerance was never applied to.

    Returns ``(winners, band_counts, info)``. ``info`` records the tolerance that
    produced the result, the attempt count and whether the stage gave up.
    """
    def cull(t: float):
        winners, sources = _run_choice_pass(
            stage, {g: list(_subsets(m)) for g, m in members.items()},
            outside_of, score_features, checkpoint, target, t,
            anchor=anchor, shap=shap, print_trials=print_trials,
        )
        return winners, sources, tuple(f for gid in members for f in winners[gid])

    # The winners only change at discrete tolerance thresholds, so most shrink
    # steps reproduce the previous attempt's set exactly. The per-group scores are
    # already checkpointed (the backdrop is fixed for the whole loop, so the point
    # ids repeat), but the *combined* score is a fresh CV fit every time -- and on
    # the path down to the floor that is ~60 of them, nearly all identical.
    seen: dict[tuple[str, ...], float] = {}

    def combined_ll(key: tuple[str, ...]) -> float:
        if key not in seen:
            seen[key] = score_features(list(key)).mean if key else float("nan")
        return seen[key]

    def report(first: int, last: int, tol_hi: float, tol_lo: float,
               key: tuple[str, ...], ll: float, ok: bool) -> None:
        """One line per *distinct* result, not per attempt.

        A shrink that reproduces the previous set has nothing new to say, and on
        the way to the floor there are dozens of those. Collapsing them keeps the
        tolerance range visible without burying the attempts that changed
        something.
        """
        span = f"attempt {first}" if first == last else f"attempts {first}-{last}"
        rng = f"{tol_hi:g}" if first == last else f"{tol_hi:g} -> {tol_lo:g}"
        print(f"  [{target}] {stage} {span}: inner tol {rng} -> {len(key)} features, "
              f"combined ll={ll:.6f} vs {budget_label} {budget:.6f}  "
              f"{'PASS' if ok else 'FAIL'}")

    attempt, t = 0, float(tol)
    prev_key: tuple[str, ...] | None = None
    run_first, run_tol_hi = 1, float(tol)     # span of the current identical run

    while True:
        attempt += 1
        winners, sources, key = cull(t)
        ll = combined_ll(key)
        ok = ll == ll and ll <= budget        # NaN never passes

        if key != prev_key:
            if prev_key is not None:          # close off the run that just ended
                report(run_first, attempt - 1, run_tol_hi, prev_tol,
                       prev_key, combined_ll(prev_key), False)
            run_first, run_tol_hi = attempt, t
        prev_key, prev_tol = key, t

        if ok:
            report(run_first, attempt, run_tol_hi, t, key, ll, True)
            return winners, sources, {"tol": t, "attempts": attempt, "fell_back": False}

        if t == 0.0:
            report(run_first, attempt, run_tol_hi, t, key, ll, False)
            break                              # the floor attempt just failed
        t *= (1.0 - shrink)
        if t < _TOL_FLOOR:
            report(run_first, attempt, run_tol_hi, prev_tol, key, ll, False)
            prev_key, run_first, run_tol_hi = None, attempt + 1, 0.0
            t = 0.0
            print(f"  [{target}] {stage} tolerance has shrunk to nothing -- one final "
                  f"attempt at tol=0 (outright best per group, no size preference)")

    # Nothing this stage can select passes, so it stops selecting. Full original
    # membership goes through *unconditionally*: the tol=0 attempt already proved
    # no tighter culling clears the budget, so re-checking this would only fail a
    # run that still has four later stages to compress it.
    winners = {g: list(m) for g, m in members.items()}
    kept = [f for gid in members for f in winners[gid]]
    print(f"  [{target}] {stage} FLOOR REACHED after {attempt} attempts -- no culling "
          f"clears {budget_label} {budget:.6f}. Giving up compression for this stage: "
          f"keeping full membership ({len(kept)} features), passed through unchecked.")
    return winners, Counter(), {"tol": 0.0, "attempts": attempt, "fell_back": True}


def _best_per_size(sub: pd.DataFrame) -> list[tuple[int, float, list[str]]]:
    """Lowest-loss combination at each size, ascending by size.

    Returns ``[(n, logloss, [feature names]), ...]``. Names come out of the
    checkpoint through ``_subset_from_row``, so a missing cell reads as an
    absence rather than printing the literal ``'nan'`` -- the log is held to the
    same rule as the selection itself.
    """
    if sub.empty:
        return []
    rows: list[tuple[int, float, list[str]]] = []
    work = sub.dropna(subset=["score_mean"]).copy()
    if work.empty:
        return []
    work["_feats"] = work["subset"].map(_subset_from_row)
    work["_n"] = work["_feats"].map(len)
    for n, grp in work.groupby("_n", sort=True):
        best = grp.loc[grp["score_mean"].idxmin()]
        rows.append((int(n), float(best["score_mean"]), list(best["_feats"])))
    return rows


def _print_sweep(
    sub: pd.DataFrame, target: str, n_candidates: int, anchor: float, winner: list[str]
) -> None:
    """Show the n=1..k sweep by name, so the result is checkable rather than trusted.

    One line per size rather than every combination: at ``max_exhaustive = 10``
    the full sweep is 1,023 rows per family, which is not something anyone reads.
    Every row is still written to the checkpoint CSV for a full audit.

    ``delta`` is against the full-model anchor because that is what the selection
    rule compares to -- a delta against the sweep's own best would not explain
    why this winner was chosen.
    """
    per_size = _best_per_size(sub)
    if not per_size:
        return
    total = 2 ** n_candidates - 1
    print(f"  [{target}] stage 5 sweep over {n_candidates} individual features "
          f"({total} combinations); best at each size:")
    win = sorted(winner)
    for n, ll, feats in per_size:
        mark = "  <- winner" if sorted(feats) == win else ""
        print(f"      n={n:>2}  ll={ll:.6f}  delta={ll - anchor:+.5f}  {feats}{mark}")


# Tie-break order for "which whole groups earn their place", shared by both
# paths below. Fewest groups first, then fewest features -- and it has to be the
# *same* tuple for both, or the exhaustive and greedy answers can differ on a tie
# for a reason that has nothing to do with the data. The greedy path used
# `("n_features",)` alone, which is harmless inside one drop-one round (every
# candidate there has dropped exactly one group, so `n_groups` is constant) and
# was still one decision written down twice.
_GROUP_PREFER = ("n_groups", "n_features")


def _run_group_elimination(
    bundles: dict[str, list[str]],
    score_features,
    checkpoint: Path,
    target: str,
    tol: float,
    *,
    exhaustive_max: int = GROUP_EXHAUSTIVE_MAX,
    anchor: float | None = None,
    print_trials: bool = True,
) -> tuple[list[str], list[str]]:
    """Decide which whole groups to keep. Returns ``(features, group ids kept)``.

    A surviving group may carry more than one feature, so "does this bundle earn
    its place" is a coarser question than the per-feature check in stage 4 -- and
    dropping a group removes all of its features at once, which is a far more
    efficient lever than removing them one at a time.

    Two paths, one answer. Below ``exhaustive_max`` groups every subset is
    scored; above it, greedy backward elimination drops one group per round. The
    greedy path is a cost control and must not change the result, so both share
    `_GROUP_PREFER` and both inherit `best_row`'s final tie-break on the
    candidate's own identity. Without that last key the two enumerate the same
    candidates in different orders and a tie resolves differently in each.

    ``anchor`` is the reference model's loss, and the band is ``anchor + tol``.
    It has to be a model that actually uses the features -- the full model, or
    whatever reduced set the caller is measuring against. Handing it a
    no-signal score makes the band admit *every* subset, at which point "keep the
    smallest within tolerance" correctly strips to a single group and the stage
    silently stops doing its job.
    """
    live = [g for g, feats in bundles.items() if feats]
    if not live:
        return [], []

    def feats_of(gids) -> list[str]:
        return [f for g in gids for f in bundles[g]]

    if len(live) <= exhaustive_max:
        print(f"  [{target}] stage 4 exhaustive over {len(live)} groups "
              f"({2 ** len(live) - 1} combinations)")
        points = [
            {"stage": "groups", "subset": feats_of(c), "groups": list(c)}
            for r in range(1, len(live) + 1)
            for c in combinations(live, r)
        ]
        res = run_grid(
            points, lambda p: score_features(list(p["subset"])), checkpoint,
            label=f"{target}/stage4", verbose=False,
            extra_row=lambda p, _r: {"n_features": len(p["subset"]), "n_groups": len(p["groups"])},
        )
        sub = res[res["stage"] == "groups"] if "stage" in res.columns else res
        if sub.empty:
            return feats_of(live), list(live)
        pick, source = best_row(sub, tol=tol, prefer=_GROUP_PREFER,
                                anchor=anchor, return_source=True)
        kept = _subset_from_row(pick.get("groups"))
        kept = [g for g in live if g in set(kept)]
        if print_trials:
            ref = anchor if anchor is not None else float(sub["score_mean"].min())
            _print_trials(sub, target, "stage4", "group combinations", kept, ref,
                          "full model" if anchor is not None else "best", subset_col="groups")
        if source == "fallback":
            print(f"  [{target}] stage 4 fallback: no group subset within tol of the full model")
        return feats_of(kept), kept

    # Too many groups to enumerate: greedy backward elimination. Each round tries
    # removing each remaining group and drops the best removal that stays within
    # tolerance, until nothing more can go.
    print(f"  [{target}] stage 4 greedy backward elimination over {len(live)} groups")
    kept = list(live)
    while len(kept) > 1:
        points = [
            {"stage": "greedy", "round": len(kept), "subset": feats_of([g for g in kept if g != drop]),
             "groups": [g for g in kept if g != drop], "dropped": drop}
            for drop in kept
        ]
        res = run_grid(
            points, lambda p: score_features(list(p["subset"])), checkpoint,
            label=f"{target}/stage4-greedy-{len(kept)}", verbose=False,
            extra_row=lambda p, _r: {"n_features": len(p["subset"]), "n_groups": len(p["groups"])},
        )
        sub = (
            res[(res["stage"] == "greedy") & (res["round"] == len(kept))]
            if {"stage", "round"} <= set(res.columns)
            else res.iloc[0:0]
        )
        if sub.empty:
            break
        pick = best_row(sub, tol=tol, prefer=_GROUP_PREFER, anchor=anchor)
        candidate = _subset_from_row(pick.get("groups"))
        if print_trials:
            ref = anchor if anchor is not None else float(sub["score_mean"].min())
            _print_trials(sub, target, "stage4", f"drop-one round of {len(kept)}",
                          candidate, ref, "full model" if anchor is not None else "best",
                          subset_col="groups")
        if len(candidate) >= len(kept):
            break
        limit = (anchor if anchor is not None else float(sub["score_mean"].min())) + tol
        if float(pick["score_mean"]) > limit:
            break
        kept = [g for g in kept if g in set(candidate)]
    return feats_of(kept), kept


def run_feature_selection(
    ft,
    target: str,
    checkpoint: Path,
    *,
    fit_cfg: FitConfig = PROBE,
    tol: float | None = None,
    max_exhaustive: int = FEATURE_EXHAUSTIVE_MAX,
    group_exhaustive_max: int = GROUP_EXHAUSTIVE_MAX,
    fail_on_regression: bool = True,
    print_trials: bool = True,
    shrink: float = RETRY_SHRINK_PCT,
    anchor_mult: float | None = None,
    stage_mult: float | None = None,
) -> dict:
    """Five-stage compression: isolate, merge, re-merge, eliminate, sweep.

    | stage | question | backdrop |
    |---|---|---|
    | 1 | does this group carry signal **on its own**? | none -- isolated |
    | 2 | which of those survivors matter once merged | other merged groups, full |
    | 3 | still, once the rest is reduced | other merged groups' stage-2 winners |
    | 4 | does this whole group earn its place | -- (group-level) |
    | 5 | every combination of the individual features left | -- (global) |

    Stage 1 is deliberately **isolated**. Judging a group against a backdrop of
    everything else meant a group with real signal could lose simply because
    50-odd other features already implied the same thing. With nothing else
    present the only way to lose is to carry nothing standalone, so what survives
    has earned it on its own merits.

    Between stages 1 and 2 the anchor changes. Stage 2 onward merges each side's
    end-product and process groups so goals and shots finally compete directly --
    but only over stage-1 survivors, because merging the raw groups would be 12
    members and 4,096 subsets each.

    Stages 4 and 5 are different granularities: stage 4 keeps or drops whole
    bundles, stage 5 sweeps n=1..k over the individual feature names those bundles
    contributed -- the only place redundancy spanning two groups can be found.

    **Two tolerance tiers.**

    The *inner* tolerance is ``tol`` (default ``SELECTION_TOL[target]``): absolute
    log loss, governing each group's own culling in stages 1-3, keeping the
    smallest candidate within ``tol`` rather than the outright lowest, because
    chasing the absolute best number fits CV noise.

    The *outer* check is fixed and never shrinks. Each stage's combined output is
    scored against the **full model** -- every candidate feature -- and must land
    within ``anchor_mult`` (stage 1) or ``stage_mult`` (stages 2-5) fold standard
    errors of it, each defaulting to this target's entry in
    `config.ANCHOR_FULL_MULT` / `config.STAGE_FULL_MULT`. They are arguments
    rather than module constants because a caller has to be able to raise the bar
    for one family without moving it for the other three -- and because reading
    them off the module meant a notebook could set them, see no effect, and get no
    warning. A stage that fails shrinks its inner tolerance by
    ``shrink`` and culls again **from full membership**; one that cannot pass even
    at ``tol=0`` keeps everything and passes through unchecked.

    The full model, **not** the naive per-league baseline. "Baseline" means
    something specific elsewhere in this codebase -- ``league_mean_baseline`` and
    every comparison in ``evaluate.py`` -- and the two are far apart (goals: a
    full model near 1.4568 against a naive baseline near 1.5212). Gating on the
    naive baseline is a far looser bar and would pass real compression damage.
    The question this check asks is "is the compressed set still as good as using
    everything", so the full model is what it measures.

    And the full model rather than the anchor because the anchor is itself stage
    1's output: gating later stages on it lets an over-pruned stage 1 quietly move
    the bar for everything downstream instead of being caught. ``anchor_logloss``
    stays computed and printed, informational past stage 1.

    Stages 4 and 5 need no retry by construction. Stage 4 always scores "keep
    every surviving group" among its candidates, which is exactly stage 3's
    already-passed output; stage 5 always scores "keep everything stage 4 kept"
    and picks the outright best. Neither can return worse than what passed one
    stage earlier. Their outer checks are reported, not enforced.

    **Raises** when the final set still fails the outer check -- which now means
    even a full-membership fallback could not come within tolerance of the full
    model, and is worth investigating rather than routine.
    ``fail_on_regression=False`` downgrades it to a warning.

    ``print_trials`` prints every subset's score in each group's search, not only
    the winner, so a dropped group is visibly a close call or a landslide.
    """
    tol = SELECTION_TOL[target] if tol is None else float(tol)
    groups = feature_groups()
    folds = season_folds(ft)
    params = base_params(target, fit_cfg)

    all_feats = [f for members in groups.values() for f in members]
    shap = shap_importance(ft, target, all_feats, fit_cfg)

    def score_features(feats: list[str]):
        if not feats:
            return intercept_only()
        return evaluate(ft, target, params, folds, features=feats, fit_cfg=fit_cfg)

    # Stage 1 has no backdrop, so its empty subset means a model with no features
    # at all. Score that as an intercept-only model -- a single constant column,
    # which XGBoost reduces to predicting the mean. That gives "this group carries
    # no standalone signal" a real number to lose to, instead of the empty subset
    # being unscoreable and the group therefore being unable to drop out.
    _ft_const = FeatureTable(
        X=ft.X.assign(_intercept=np.float32(1.0)),
        meta=ft.meta, y=ft.y, opp_idx=ft.opp_idx, L=ft.L, alpha=ft.alpha,
    )

    def intercept_only():
        return evaluate(_ft_const, target, params, folds,
                        features=["_intercept"], fit_cfg=fit_cfg)

    # Two reference scores exist, and they are NOT interchangeable:
    #   full_model_ll -- every candidate feature. Informational only, printed once.
    #   anchor        -- the stage-1 survivor set. The working anchor for stages
    #                    2-5 and for the safety net.
    full = score_features(all_feats)
    full_model_ll, full_se = full.mean, full.se

    # The outer budgets, both measured against the FULL MODEL -- not the naive
    # per-league baseline, which is a different and much looser number. Fixed for
    # the whole run: retrying shrinks the inner tolerance, never these.
    anchor_mult = ANCHOR_FULL_MULT.get(target, 1.0) if anchor_mult is None else anchor_mult
    stage_mult = STAGE_FULL_MULT.get(target, 2.0) if stage_mult is None else stage_mult
    anchor_budget = full_model_ll + anchor_mult * full_se
    stage_budget = full_model_ll + stage_mult * full_se

    print(f"  [{target}] full model ({len(all_feats)} features): logloss "
          f"{full_model_ll:.6f}   [the outer check's reference]")
    print(f"  [{target}] inner tolerance {tol:g} (absolute log loss; fold SE {full_se:.5f}; "
          f"shrinks {shrink:.0%} per retry)")
    print(f"  [{target}] outer check vs FULL MODEL: "
          f"stage 1 budget {anchor_budget:.6f} ({anchor_mult:g}x SE), "
          f"stages 2-5 budget {stage_budget:.6f} ({stage_mult:g}x SE)")

    # Checkpoint names carry a fingerprint of the group structure, the feature
    # window and the scoring definition. A cached score is only valid for the
    # exact backdrop it was measured against, and `point_id` encodes only the
    # group id and its subset -- so without this, changing `feature_groups()`
    # would silently reuse scores from the old structure wherever a group id
    # happened to survive the change (`context` and `movement` keep their names,
    # and would have done exactly that).
    #
    # `scoring_hash()` is the second layer: structure can be identical while the
    # metric underneath has changed (per-team vs match-total, or a revised cap),
    # which changes the number without changing anything the group fingerprint
    # can see.
    fingerprint = hashlib.sha256(
        json.dumps(
            {"groups": {g: sorted(m) for g, m in groups.items()},
             "L": ft.L, "alpha": ft.alpha,
             "scoring": scoring_hash()},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:8]

    base = checkpoint.parent / f"{checkpoint.stem}_{fingerprint}"
    ck1 = base.with_name(f"{base.name}_stage1.csv")
    ck2 = base.with_name(f"{base.name}_stage2.csv")
    ck3 = base.with_name(f"{base.name}_stage3.csv")
    ck4 = base.with_name(f"{base.name}_stage4.csv")
    # Stage 5 gets its own file rather than sharing stage 4's. `append_row` now
    # aligns to the header either way, but two stages with different point keys
    # have no reason to share a checkpoint -- and sharing one is what let their
    # differing column sets collide in the first place.
    ck5 = base.with_name(f"{base.name}_stage5.csv")

    # --- Stage 1: each group ALONE, no backdrop ------------------------------
    # The question here is narrow: does this group carry standalone signal? With
    # a backdrop of everything else, a group with real signal could lose simply
    # because 50-odd other features already implied the same thing. With nothing
    # else present, the only way to lose is to carry nothing on its own.
    winners1, src1, retry1 = _run_stage_with_retry(
        "stage1", groups,
        lambda gid: [],                      # <- isolation: zero backdrop
        score_features, ck1, target, tol,
        budget=anchor_budget, budget_label="full-model budget",
        shap=shap, print_trials=print_trials, shrink=shrink,
    )
    stage1 = [f for gid in groups for f in winners1[gid]]
    dropped1 = [g for g, w in winners1.items() if not w]
    print(f"  [{target}] stage 1 -> {len(stage1)} features  "
          f"[in band {src1['band']}, fallback {src1['fallback']}]"
          + (f"  groups with no standalone signal: {dropped1}" if dropped1 else ""))

    # --- The anchor: the combined stage-1 survivors --------------------------
    # Informational from here on. It used to be the budget every later stage was
    # measured against, which meant an over-pruned stage 1 silently moved the bar
    # for everything downstream instead of being caught.
    if not stage1:
        raise ValueError(
            f"{target}: stage 1 left no features at all, even after retrying down to "
            f"tol=0 and falling back to full membership. That should be unreachable -- "
            f"check the feature table for {target!r}."
        )
    anchor_res = score_features(stage1)
    anchor = anchor_res.mean
    print(f"  [{target}] ANCHOR = stage-1 survivor set ({len(stage1)} features): "
          f"logloss {anchor:.6f}   [informational -- gates nothing]")
    print(f"  [{target}]   vs full model {full_model_ll:.6f}  "
          f"delta {anchor - full_model_ll:+.5f}")

    # --- Stage 2: merged groups, restricted to stage-1 survivors -------------
    # Now goals and shots finally meet inside one group. Merging the raw groups
    # would be 12 members and 4,096 subsets each; restricting to survivors is
    # what makes it affordable.
    merged = merged_groups(winners1)
    print(f"  [{target}] merged groups: "
          + ", ".join(f"{g}({len(v)})" for g, v in merged.items()))

    winners2, src2, retry2 = _run_stage_with_retry(
        "stage2", merged,
        lambda gid: [f for g, ms in merged.items() if g != gid for f in ms],
        score_features, ck2, target, tol,
        budget=stage_budget, budget_label="full-model budget",
        anchor=anchor, shap=shap, print_trials=print_trials, shrink=shrink,
    )
    stage2 = [f for gid in merged for f in winners2[gid]]
    print(f"  [{target}] stage 2 -> {len(stage2)} features  "
          f"[in band {src2['band']}, fallback {src2['fallback']}]")

    # --- Stage 3: re-compress against the stage-2 winners --------------------
    winners3, src3, retry3 = _run_stage_with_retry(
        "stage3", {g: list(winners2[g]) for g in merged},
        lambda gid: [f for g in merged if g != gid for f in winners2[g]],
        score_features, ck3, target, tol,
        budget=stage_budget, budget_label="full-model budget",
        anchor=anchor, shap=shap, print_trials=print_trials, shrink=shrink,
    )
    stage3 = [f for gid in merged for f in winners3[gid]]
    n_venue = sum(1 for f in stage3 if f.endswith("_v"))
    print(f"  [{target}] stage 3 -> {len(stage3)} features ({n_venue} venue-specific)  "
          f"[in band {src3['band']}, fallback {src3['fallback']}]")

    # --- Stage 4: does each whole group earn its place? ----------------------
    stage4, groups_kept = _run_group_elimination(
        winners3, score_features, ck4, target, tol,
        exhaustive_max=group_exhaustive_max, anchor=anchor, print_trials=print_trials,
    )
    print(f"  [{target}] stage 4 -> {len(stage4)} features in {len(groups_kept)} groups")

    # Stages 4 and 5 do not retry: each always has a guaranteed-passing candidate
    # among its own (stage 4 can keep every group, stage 5 can keep everything
    # stage 4 kept), so neither can return worse than what already passed. Their
    # outer check is therefore reported, not enforced -- but stage 4's internal
    # rule is smallest-within-tolerance-of-*best*, not "prefer whatever cleared
    # cleared the budget", so it can in principle pick a smaller, slightly worse
    # option and miss the budget with a better one available. Print it if so.
    stage4_ll = score_features(stage4).mean if stage4 else float("nan")
    if stage4_ll == stage4_ll and stage4_ll > stage_budget:
        print(f"  [{target}] NOTE: stage 4's combined output {stage4_ll:.6f} is over the "
              f"full-model budget {stage_budget:.6f}, despite 'keep every group' being "
              f"among its candidates. Its rule prefers the smaller set within tolerance "
              f"of the best, which can pass over a better-scoring option.")

    # --- Stage 5: exhaustive over the individual features surviving stage 4 ---
    # Not over groups -- that was stage 4. This is the n=1..k sweep across every
    # combination of the specific feature names left, which is the only stage
    # that can see redundancy spanning two different groups.
    final = list(stage4)
    if 0 < len(stage4) <= max_exhaustive:
        points = [{"stage": "exhaustive", "subset": list(s)} for s in _subsets(stage4) if s]
        res = run_grid(points, lambda p: score_features(list(p["subset"])), ck5,
                       label=f"{target}/stage5", verbose=False, extra_row=_n_features_row)
        # Stage 5 owns `ck5`, so every row here is already its own -- but keep the
        # filter: it still guards a file left over from when the two stages shared
        # one. Guard on the column existing: `res.get("stage")` returns None when
        # it does not, and `res[None == "exhaustive"]` is a KeyError rather than
        # an empty frame.
        sub = res[res["stage"] == "exhaustive"] if "stage" in res.columns else res.iloc[0:0]
        if not sub.empty:
            # The outright best score, NOT smallest-within-tolerance. Stages 1-4
            # have already applied that parsimony pressure four times, and this
            # is the one stage that scores every combination rather than
            # sampling, so applying it a fifth time only concedes score for a
            # size reduction the earlier stages have already extracted. A real
            # goals run gave up n=6 ll=1.460348 to keep n=5 ll=1.461710.
            #
            # `tol=0.0` with no `prefer` column collapses the band to the
            # minimum. That is deliberate here -- not the silently-inert rule
            # `best_row`'s docstring warns about -- and it keeps that function's
            # NaN-score and empty-frame guards, which raise named errors.
            pick = best_row(sub, tol=0.0)
            final = _subset_from_row(pick["subset"])
            _print_sweep(sub, target, len(stage4), anchor, final)
            print(f"  [{target}] stage 5 exhaustive over {len(stage4)} individual "
                  f"features -> {len(final)} selected")
    else:
        print(f"  [{target}] stage 5 skipped: {len(stage4)} individual features exceeds "
              f"max_exhaustive={max_exhaustive}; keeping the stage-4 set")

    # A selected feature that is not a real column means a "no winner" case leaked
    # through as a placeholder. Fail loudly here rather than at fit time.
    unknown = set(final) - set(ALL_FEATURE_NAMES)
    if unknown:
        raise ValueError(f"{target}: selection produced unknown feature(s): {sorted(unknown)}")
    if len(set(final)) != len(final):
        raise ValueError(f"{target}: selection produced duplicate features: {final}")

    result = score_features(final)

    # --- Where did the loss go? ---------------------------------------------
    # Score each stage's surviving set so a regression is attributable to the
    # stage that caused it, rather than only visible at the end. Four extra CV
    # evaluations against several hundred -- worth it to know whether stage 1
    # over-pruned or stage 3 dropped a group it should have kept.
    progression = [
        ("full", len(all_feats), full_model_ll),
        ("ANCHOR", len(stage1), anchor),
        ("stage2", len(stage2), score_features(stage2).mean if stage2 else float("nan")),
        ("stage3", len(stage3), score_features(stage3).mean if stage3 else float("nan")),
        ("stage4", len(stage4), stage4_ll),
        ("final", len(final), result.mean),
    ]
    print(f"  [{target}] loss by stage (full model {full_model_ll:.6f}; budget = full + "
          f"{stage_mult:g}x SE = {stage_budget:.6f}; deltas vs full model):")
    for name, n, ll in progression:
        over = ll == ll and ll > stage_budget and name != "full"
        print(f"        {name:7} {n:>3} features  ll={ll:.6f}  "
              f"delta={ll - full_model_ll:+.5f}{'  OVER BUDGET' if over else ''}")

    retries = {"stage1": retry1, "stage2": retry2, "stage3": retry3}
    worked = [f"{s}: {r['attempts']} attempt(s), settled at tol {r['tol']:g}"
              + ("  [FELL BACK to full membership]" if r["fell_back"] else "")
              for s, r in retries.items() if r["attempts"] > 1 or r["fell_back"]]
    if worked:
        print(f"  [{target}] stages that needed a retry:")
        for line in worked:
            print(f"        {line}")

    # Stage 1 defines the anchor, so a check against the anchor could never see
    # stage-1 damage. It is gated on the full model like every other stage now,
    # which is exactly the comparison this note was making by hand.
    stage1_cost = anchor - full_model_ll
    if stage1_cost > 10 * tol:
        print(f"  [{target}] NOTE: stage 1 gave up {stage1_cost:+.5f} against the full "
              f"model ({10 * tol:g} = 10x the starting tol). Everything downstream starts "
              f"from there. Tighten SELECTION_TOL[{target!r}] if that is more than you "
              f"meant to concede.")

    # --- Safety net ----------------------------------------------------------
    # Selection is meant to find a smaller model that is *as good as* the full
    # one. A selected set materially worse than the full model is a failed
    # selection, not a tighter one -- and left unchecked it reaches artifacts/
    # and gets scored, retrained and shipped without anyone noticing. It has
    # already happened once: a `goals` run selected 1.891173 against a 1.874476
    # full model and a 1.8874 baseline, i.e. worse than having no model at all.
    excess = result.mean - stage_budget
    if excess > 0:
        msg = (
            f"{target}: SELECTION FAILED -- the selected {len(final)}-feature set scores "
            f"{result.mean:.6f}, which is {excess:.6f} beyond the budget of "
            f"{stage_budget:.6f} (full model {full_model_ll:.6f} + {stage_mult:g}x "
            f"fold SE {full_se:.5f}). Anchor, for reference, scored {anchor:.6f}.\n"
            f"  Stage losses: "
            + ", ".join(f"{n}={ll:.6f}" for n, _, ll in progression)
            + "\n"
            "  Stages 1-3 each retry at a shrinking tolerance and fall back to full "
            "membership rather than ship a set that cannot clear this, so reaching "
            "here means even keeping everything came in worse than the full model by "
            "more than the budget. Stages 4-5 do not retry -- check the retry lines "
            "above for which stage fell back, and the stage losses for where it went.\n"
            f"  Pass fail_on_regression=False to accept this result anyway."
        )
        if fail_on_regression:
            raise ValueError(msg)
        print("  WARNING: " + msg)

    print(f"  [{target}] selected {len(final)} features, logloss {result.mean:.6f} "
          f"(full model {full_model_ll:.6f}, anchor {anchor:.6f})")
    return {
        "selected": final,
        "provisional": stage4,
        "provisional_stage1": stage1,
        "provisional_stage2": stage2,
        "provisional_stage3": stage3,
        "group_winners_stage1": winners1,
        "group_winners_stage2": winners2,
        "group_winners_stage3": winners3,
        "merged_groups": merged,
        "groups_kept": groups_kept,
        "shap_rank": shap.index.tolist(),
        # The starting tolerance, so a stored artifact stays attributable to the
        # dial that made it. `retries` records what each stage actually settled
        # at, which is the number that produced the result when one retried.
        "tol": tol,
        "retries": retries,
        "band_counts": {"stage1": dict(src1), "stage2": dict(src2), "stage3": dict(src3)},
        # `full_logloss` is what the outer check gates on; `anchor_logloss` is
        # informational past stage 1. The naive per-league baseline is a third,
        # separate thing and is not computed here -- the notebook stores it
        # alongside these under `baseline_logloss`.
        "stage_budget": stage_budget,
        "anchor_logloss": anchor,
        "full_logloss": full_model_ll,
        "full_logloss_se": full_se,
        "logloss": result.mean,
        "logloss_se": result.se,
    }


# --- Stage C: hyperparameter descent --------------------------------------


def run_xgb_descent(
    ft,
    target: str,
    features: list[str],
    checkpoint: Path,
    *,
    fit_cfg: FitConfig = SEARCH,
    tol: float | None = None,
    passes: int = DESCENT_PASSES,
    folds: list | None = None,
    start_params: dict | None = None,
) -> dict:
    """Pairwise coordinate descent over blocks of hyperparameters.

    ``n_estimators`` is *derived* from early stopping rather than tuned, so the
    first block is a learning-rate sweep instead of a trees x rate grid --
    cheaper, and methodologically better than pinning a tree count against a
    rate.

    Note on ``tol``: unlike feature selection, hyperparameter candidates have no
    "size" to prefer when scores tie, so no ``prefer`` column is passed and
    selection is a plain argmin. ``tol`` is accepted for signature consistency but
    **does not bind here** -- said explicitly because the same call in the feature
    stages looked like it was applying a tolerance rule when it was not. What
    replaces it is reporting: each block prints its winner's margin over the
    incumbent in fold SE and flags one that is inside noise.

    ``n_estimators`` is a ceiling, not a budget. `SEARCH` sets it high and early
    stopping decides the count for each candidate, so a learning rate is judged at
    the number of trees it actually wants -- measured, 0.01 wants 994 and 0.05
    wants 213, and the old fixed 400 granted the second and denied the first.

    ``passes`` runs the whole block sequence again from where the last one
    finished; a pass that moves nothing ends it early. One pass, the previous
    default, rarely converges a coordinate descent.
    """
    tol = SELECTION_TOL[target] if tol is None else float(tol)
    folds = folds if folds is not None else season_folds(ft)
    current = dict(start_params) if start_params else base_params(target, fit_cfg)
    blocks = list(PAIR_BLOCKS)
    if target_spec(target).objective == "reg:tweedie":
        blocks.append(TWEEDIE_BLOCK)
    # `max_bin` last: it is a resolution knob, and asking it after the shape of
    # the model is settled costs three evaluations instead of multiplying every
    # earlier block by three.
    blocks.append(MAX_BIN_BLOCK)

    moved_any = False
    for p in range(passes):
        moved_this_pass = False
        for block_name, axes in blocks:
            grids = [(k, vs) for k, vs in axes]
            points = []
            if len(grids) == 1:
                k, vs = grids[0]
                points = [{"block": block_name, "pass": p, k: v} for v in vs]
            else:
                (k1, v1s), (k2, v2s) = grids
                points = [{"block": block_name, "pass": p, k1: a, k2: b} for a in v1s for b in v2s]

            def score(pt, _cur=current):
                trial = dict(_cur)
                trial.update({k: v for k, v in pt.items() if k not in ("block", "pass")})
                return evaluate(ft, target, trial, folds, features=features, fit_cfg=fit_cfg)

            res = run_grid(points, score, checkpoint,
                           label=f"{target}/xgb/{block_name}/p{p}", verbose=False)
            sub = res[(res.get("block") == block_name) & (res.get("pass") == p)]
            if sub.empty:
                continue
            pick = best_row(sub, tol=tol)
            before = {k: current.get(k) for k, _ in grids}

            # How much does the block's winner actually beat the value already in
            # `current`? Located by matching the incumbent's own row in the grid,
            # so the comparison is like for like on the same folds.
            keys = [k for k, _ in grids if k in sub.columns and before.get(k) is not None]
            incumbent = sub
            for k in keys:
                incumbent = incumbent[np.isclose(incumbent[k].astype(float), float(before[k]))]
            margin = (float(incumbent["score_mean"].iloc[0]) - float(pick["score_mean"])
                      if len(keys) == len(grids) and not incumbent.empty else float("nan"))

            # **Keep the incumbent unless the winner genuinely beats it.** A plain
            # argmin looks right and is not: hyperparameters go inert on each
            # other -- measured, `reg_alpha=10` prunes splits hard enough that
            # `min_child_weight` 1 and 30 give byte-identical predictions -- and
            # over a block of tied scores argmin picks whichever row sorts first.
            # The block then "moves" on nothing, every pass, so the descent churns
            # and the alternating loop can never report convergence. That is
            # exactly what a first run of this loop did.
            #
            # NaN margin means the incumbent is not in this block's grid (the first
            # pass, for an axis `base_params` does not set), and then there is
            # nothing to defend, so the winner is taken.
            accept = not (margin == margin) or margin > tol
            if accept:
                for k, _ in grids:
                    if k in sub.columns and not pd.isna(pick.get(k)):
                        v = pick[k]
                        current[k] = (int(v) if float(v).is_integer() and k != "learning_rate"
                                      else float(v))
            after = {k: current[k] for k, _ in grids}
            moved = after != before
            moved_this_pass |= moved
            moved_any |= moved

            se = float(pick.get("score_se") or float("nan"))
            verdict = ""
            if margin == margin:
                verdict = f"  margin {margin:+.6f}"
                if se == se and se:
                    verdict += f" ({margin / se:+.2f} SE)"
                if not accept:
                    verdict += f"  [<= tol {tol:g}, kept {before}]"
            print(f"  [{target}] p{p} {block_name:26} -> {after}  "
                  f"ll={pick['score_mean']:.6f}{verdict}")

        if not moved_this_pass:
            print(f"  [{target}] pass {p + 1} changed nothing -- descent converged")
            break

    final = evaluate(ft, target, current, folds, features=features, fit_cfg=fit_cfg)
    # Early stopping decides the tree count per candidate, so report where it
    # landed: at the ceiling means the rate never got to finish, and at exactly
    # the patience means the rate lost to its own stopping rule rather than to
    # the data.
    iters = final.best_iterations or []
    if iters:
        med = int(np.median(iters))
        ceiling = int(current.get("n_estimators", fit_cfg.n_estimators))
        note = "  <- AT THE CEILING, raise SEARCH_TREE_CEILING" if med >= 0.95 * ceiling else ""
        print(f"  [{target}] settled lr={current.get('learning_rate')} "
              f"max_bin={current.get('max_bin')}, median stop {med}/{ceiling} trees"
              f" (patience {fit_cfg.early_stopping_rounds}){note}")
    return {"params": current, "logloss": final.mean, "logloss_se": final.se,
            "best_iterations": final.best_iterations, "dispersion": final.dispersion,
            "moved": moved_any}


# --- Stage D: the alternating tuning loop ---------------------------------


def run_tuning(
    df: pd.DataFrame,
    bw,
    target: str,
    features: list[str],
    *,
    fit_cfg: FitConfig = SEARCH,
    rounds: int = MAX_TUNING_ROUNDS,
    passes: int = DESCENT_PASSES,
    tol: float | None = None,
    checkpoint_dir=None,
) -> dict:
    """Alternate window and hyperparameters until they stop moving each other.

    The three stages this replaces were run once each, in order, and the third one
    was a dead end: the stability recheck updated the window when the winner moved
    and then froze hyperparameters that had been tuned at the *old* window. The
    two settings interact, so a single pass in a fixed order cannot settle them --
    which is what the recheck was detecting and had no way to act on.

    ::

        round 1   window over the full grid, at default parameters
                  descent from that window
        round 2   window over a neighbourhood of the incumbent, at ROUND-1 params
                  descent from that window, starting where round 1 finished
        ...       until a round moves neither, or `rounds` is reached

    Convergence is the stopping rule and the round cap is the safety net, so a run
    that will not settle shows up as "hit the cap" rather than as a number.

    Every fit here uses `cv.tuning_folds` -- the most recent development season is
    withheld. Nothing in this function is allowed to see it; `holdout_score` is
    what looks, once, afterwards.
    """
    from ..paths import search_checkpoint

    tol = SELECTION_TOL[target] if tol is None else float(tol)
    feats = list(features)

    def ckpt(stage: str, **extra) -> Path:
        fp = fingerprint_for(feats, fit_cfg, folds_ref, target=target, **extra)
        base = search_checkpoint(target, stage, fingerprint=fp)
        return Path(checkpoint_dir) / base.name if checkpoint_dir else base

    # One reference table, only to name the fold *seasons* for the fingerprint and
    # the log line. The masks themselves cannot be reused -- `drop_warmup` removes
    # a different number of rows at each L -- so every candidate derives its own
    # from `tuning_folds`, which keys on the season boundary rather than on shape.
    ref = build_feature_table(df, bw, L_GRID[len(L_GRID) // 2], ALPHA_GRID[0], features=feats)
    folds_ref = tuning_folds(ref)
    print(f"  [{target}] tuning on {len(folds_ref)} folds "
          f"({folds_ref[0].val_season} .. {folds_ref[-1].val_season}); "
          f"{holdout_season()} withheld")

    window: dict | None = None
    params: dict | None = None
    history: list[dict] = []

    for rnd in range(1, rounds + 1):
        print(f"\n  [{target}] --- round {rnd} ---")

        if window is None:
            L_grid, alpha_grid = None, None          # full grid, first time only
        else:
            L_grid = [L for L in L_GRID if abs(L - window["L"]) <= WINDOW_RECHECK_L_SPAN]
            alpha_grid = [a for a in ALPHA_GRID
                          if abs(a - window["alpha"]) <= WINDOW_RECHECK_ALPHA_SPAN]

        res = run_window_grid(df, bw, target, ckpt("window_grid", rnd=rnd),
                              feats, L_grid=L_grid, alpha_grid=alpha_grid,
                              params=params, folds_for=tuning_folds, fit_cfg=fit_cfg)
        new_window = pick_window(res, target, tol=tol)
        window_moved = window is None or (new_window["L"], new_window["alpha"]) != (
            window["L"], window["alpha"])
        if window is not None and not window_moved:
            print(f"  [{target}] window unchanged at the tuned learner")
        window = new_window

        ft = build_feature_table(df, bw, window["L"], window["alpha"], features=feats)
        tuned = run_xgb_descent(ft, target, feats, ckpt("xgb_descent", rnd=rnd,
                                                       L=window["L"], alpha=window["alpha"]),
                                fit_cfg=fit_cfg, tol=tol, passes=passes,
                                folds=tuning_folds(ft), start_params=params)
        params_moved = params is not None and tuned["params"] != params
        params = tuned["params"]

        history.append({"round": rnd, "window": {k: window[k] for k in ("L", "alpha")},
                        "logloss": tuned["logloss"], "window_moved": window_moved,
                        "params_moved": params_moved or rnd == 1})
        print(f"  [{target}] round {rnd}: L={window['L']} alpha={window['alpha']:.2f} "
              f"ll={tuned['logloss']:.6f}  "
              f"[window {'moved' if window_moved else 'held'}, "
              f"params {'moved' if params_moved or rnd == 1 else 'held'}]")

        if rnd > 1 and not window_moved and not params_moved:
            print(f"  [{target}] converged after {rnd} rounds")
            break
    else:
        print(f"  [{target}] hit the {rounds}-round cap without converging -- "
              f"the window and the learner are still moving each other")

    return {"window": window, "params": params, "logloss": tuned["logloss"],
            "logloss_se": tuned["logloss_se"], "best_iterations": tuned["best_iterations"],
            "dispersion": tuned["dispersion"], "rounds": history,
            "converged": len(history) < rounds or (
                len(history) > 1 and not history[-1]["window_moved"]
                and not history[-1]["params_moved"]),
            "folds": [f.val_season for f in folds_ref]}


def holdout_score(df, bw, target: str, features: list[str], window: dict, params: dict,
                  *, fit_cfg: FitConfig = SEARCH) -> dict:
    """Score the settled configuration once against the withheld season.

    This is the only read on whether a tuning run bought its improvement with
    signal or with dials, and it is available without spending the test season --
    which by design is looked at once ever. A holdout materially worse than the
    tuning CV means the extra freedom went into the folds it could see.
    """
    ft = build_feature_table(df, bw, int(window["L"]), float(window["alpha"]), features=features)
    fold = holdout_fold(ft)
    if fold is None:
        return {"season": None, "logloss": float("nan"), "gap": float("nan")}
    res = evaluate(ft, target, params, [fold], features=features, fit_cfg=fit_cfg)
    return {"season": fold.val_season, "logloss": res.mean, "n_scored": res.n_scored}


__all__ = [
    "run_window_grid", "pick_window", "effective_window", "fingerprint_for",
    "run_tuning", "holdout_score", "shap_importance", "run_feature_selection",
    "run_xgb_descent",
    # Exported for the regression tests, which pin the "an absent winner is an
    # absence, never a placeholder" invariant through every stage.
    "_subset_from_row", "_run_choice_pass", "_run_group_elimination",
    "_subsets", "_best_per_size",
]
