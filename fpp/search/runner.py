"""Resumable, append-only grid runner.

Every search stage writes one row per evaluated point and re-reads the file to
skip work already done, so a run interrupted at any point resumes where it
stopped. That matters far more now than it did for the reference framework:
tuning happens four times over, once per target family.

One correctness detail that is the classic failure mode of resumable CSV grids:
float keys are formatted through ``_fmt`` before becoming part of the
already-done lookup. Without it ``0.05`` round-trips through CSV inconsistently,
nothing ever matches "already done", and the entire grid silently recomputes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd


def _fmt(v: Any) -> str:
    """Stable string form of a grid key. Floats go through %.6g, always."""
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return "|".join(_fmt(x) for x in v)
    if isinstance(v, dict):
        return json.dumps({k: _fmt(x) for k, x in sorted(v.items())}, sort_keys=True)
    return str(v)


def point_id(key: dict) -> str:
    return "&".join(f"{k}={_fmt(v)}" for k, v in sorted(key.items()))


def load_done(path: Path) -> set[str]:
    """Point ids already evaluated in this checkpoint file."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        prev = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return set()
    return set(prev["point_id"].astype(str)) if "point_id" in prev.columns else set()


def append_row(path: Path, row: dict) -> None:
    """Append one result, aligned to the file's existing header.

    Rows written to one checkpoint do not all carry the same keys: stage 5's
    points have no ``groups``, the greedy path adds ``round`` and ``dropped``,
    and each xgb-descent block names its own axes. ``to_csv(mode="a")`` writes a
    row's *own* columns in its *own* order, so without this alignment a
    differently-keyed row lands positionally under the first row's header and
    every column past the first difference is shifted one to the left --
    ``score_mean`` silently reads ``score_se``, and selection follows it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not (path.exists() and path.stat().st_size > 0):
        pd.DataFrame([row]).to_csv(path, index=False)
        return

    header = list(pd.read_csv(path, nrows=0).columns)
    extra = [c for c in row if c not in header]
    if extra:
        # A genuinely new column: widen the file rather than drop the value.
        # Rare (once per new key), so the full rewrite is not a hot path.
        old = pd.read_csv(path)
        pd.concat([old, pd.DataFrame([row])], ignore_index=True) \
          .reindex(columns=header + extra).to_csv(path, index=False)
        return

    # `reindex` puts every value under its own name and fills absent keys with
    # NaN in the right position -- the alignment the append is missing.
    pd.DataFrame([row]).reindex(columns=header).to_csv(
        path, mode="a", header=False, index=False)


def read_results(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def run_grid(
    points: Sequence[dict],
    score: Callable[[dict], Any],
    checkpoint: Path,
    *,
    label: str = "grid",
    extra_row: Callable[[dict, Any], dict] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Evaluate every point not already in ``checkpoint``; return the rows for
    ``points``.

    Deliberately **not** every row in the file. Several stages write to one
    checkpoint across invocations with different candidate sets -- stage 4's
    group combinations depend on which groups survived stage 3 -- and a caller
    filtering only on ``stage`` or ``group`` cannot tell this run's rows from an
    earlier run's. Returning the whole file let a previous run's rows compete in,
    and win, the current run's selection.

    ``score(point)`` must return a ``CVResult``-like object exposing ``mean``,
    ``se``, ``per_fold`` and ``best_iterations``.
    """
    wanted = {point_id(p) for p in points}
    done = load_done(checkpoint) & wanted   # cached rows *for these points*
    todo = [p for p in points if point_id(p) not in done]
    if verbose:
        print(f"[{label}] {len(points)} points, {len(done)} cached, {len(todo)} to run")

    for i, p in enumerate(todo, 1):
        res = score(p)
        row = {
            "point_id": point_id(p),
            **{k: _fmt(v) if isinstance(v, (dict, list, tuple)) else v for k, v in p.items()},
            "score_mean": getattr(res, "mean", float("nan")),
            "score_se": getattr(res, "se", float("nan")),
            "n_scored": getattr(res, "n_scored", 0),
            "per_fold_json": json.dumps(getattr(res, "per_fold", {})),
            "best_iter_median": (
                int(pd.Series(res.best_iterations).median())
                if getattr(res, "best_iterations", None) else None
            ),
            "dispersion_json": json.dumps(getattr(res, "dispersion", {})),
        }
        if extra_row:
            row.update(extra_row(p, res))
        append_row(checkpoint, row)
        if verbose:
            print(f"  ({i}/{len(todo)}) {point_id(p)} -> {row['score_mean']:.6f} (+/-{row['score_se']:.6f})")

    out = read_results(checkpoint)
    if not out.empty and "point_id" in out.columns:
        out = out[out["point_id"].astype(str).isin(wanted)]
    return out


def best_row(
    results: pd.DataFrame,
    tol: float = 0.001,
    prefer: Iterable[str] = (),
    *,
    anchor: float | None = None,
    shap: "pd.Series | None" = None,
    subset_col: str = "subset",
    return_source: bool = False,
) -> "pd.Series | tuple[pd.Series, str]":
    """Pick a winner: the *simplest* candidate whose score is within tolerance.

    ``tol`` is an **absolute** log-loss slack, not a multiple of the fold standard
    error. Scaling by noise made the band ~10x wider than the reference
    framework's fixed 0.001 and let almost anything look redundant.

    Two rules, matching the reference:

    * ``anchor is None`` -- near-best. Band is ``best + tol``. Used by the first
      compression pass: "which members are worth keeping, relative to the best
      this group can manage".
    * ``anchor`` given -- near-anchor. Band is ``anchor + tol``, the anchor being
      the *full-model* loss. Later candidates are already reduced models, so
      comparing them to the full model is what lets the size tie-break strip.
      If nothing falls inside, fall back to the rows tied at the minimum, so a
      stage can always name a winner.

    ``prefer`` names columns to minimise as tie-breakers (typically
    ``n_features``). **Passing it matters**: with no ``prefer`` column the sort
    collapses to ``score_mean`` ascending, the band is discarded, and the
    "smallest within tolerance" rule is silently inert.

    ``shap`` is an optional per-feature importance series used as the tie-break
    after size, score and stability.

    The **last** key is the candidate's own identity, and it is not decoration.
    Size, loss, stability and SHAP can all tie -- routinely, on the synthetic
    scorers the tests use, and whenever a band is wide enough to admit
    equally-good candidates -- and with nothing after them the winner was decided
    by the order the rows happened to arrive in. That made two callers asking the
    same question of the same candidates able to get different answers:
    `_run_group_elimination` enumerates group subsets in one order when it can
    afford to and reaches them by backward elimination when it cannot, and the
    greedy path is documented as a cost control that "must not change the
    answer". It could. Sorting on the subset string last makes the choice a
    property of the candidates rather than of how they were generated.

    ``return_source`` also returns ``"band"`` or ``"fallback"`` -- the signal that
    says whether ``tol`` was wide enough to admit anything at all.
    """
    if results.empty:
        raise ValueError("no results to choose from")

    # The `.empty` check above is not sufficient: rows exist but every score can
    # be NaN (e.g. every fold came out empty). Without this the band ends up
    # empty and `.iloc[0]` raises an opaque IndexError instead of saying so.
    r = results.dropna(subset=["score_mean"]).copy()
    if r.empty:
        raise ValueError(
            f"no candidate produced a score -- all {len(results)} rows have NaN score_mean"
        )

    best = float(r["score_mean"].min())
    source = "band"

    if anchor is None:
        band = r[r["score_mean"] <= best + tol]
    else:
        band = r[r["score_mean"] <= float(anchor) + tol]
        if band.empty:
            band = r[r["score_mean"] == best]
            source = "fallback"

    cols = [c for c in prefer if c in band.columns]
    sort_cols = [*cols, "score_mean", "score_se"]

    if shap is not None and subset_col in band.columns:
        weight = band[subset_col].map(
            lambda v: -float(shap.reindex(_split_subset(v)).fillna(0.0).sum())
        )
        band = band.assign(_shap_rank=weight)
        sort_cols.append("_shap_rank")

    # Last resort, so a full tie resolves the same way whoever asks. `subset` is
    # already a stable "a|b|c" string by the time it reaches a checkpoint
    # (`runner._fmt`), so this is an ordinary lexicographic sort; the empty
    # subset reads back as NaN and pandas puts it last, which costs nothing
    # because it can only tie with another empty one.
    if subset_col in band.columns:
        sort_cols.append(subset_col)

    pick = band.sort_values(sort_cols).iloc[0]
    return (pick, source) if return_source else pick


def _split_subset(value) -> list[str]:
    """Local mirror of ``stages._subset_from_row`` -- kept here to avoid a cycle."""
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    return [f for f in str(value).split("|") if f]


__all__ = ["point_id", "load_done", "append_row", "read_results", "run_grid", "best_row"]
