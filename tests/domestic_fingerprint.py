"""A fingerprint of everything the league model computes, for the Euro guard rail.

The European work on the `Euro-Expansion` branch is allowed to *read* the league
model and nothing more: league -> UEFA is a one-way street. This pins that down by
measurement rather than by promise. It hashes the cleaned team-match table and,
per target, the feature matrix and targets the frozen artifact's window and
feature selection produce -- priors, seeds, promoted multipliers and shrinkage all
land in that matrix, so a change to any of them moves the hash.

Only completed seasons are hashed, so the live season filling in does not move the
fingerprint. Rebuilding the clean table with genuinely new history (a reconcile
fix, a backfill) does move it, legitimately: re-capture on `main` with

    Advanced_Football_Project/bin/python -m tests.domestic_fingerprint

and commit the JSON alongside the data change, never from a branch that touches
European code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

SNAPSHOT = Path(__file__).with_name("domestic_fingerprint.json")


def _hash(df: pd.DataFrame) -> str:
    df = df.copy()
    for c in df.columns:
        if isinstance(df[c].dtype, pd.CategoricalDtype) or df[c].dtype == object:
            df[c] = df[c].astype(str)
    num = df.select_dtypes("number").columns
    df[num] = df[num].round(6)
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


def fingerprint() -> dict:
    from fpp.artifacts import RunContext
    from fpp.build import build_feature_table
    from fpp.clean import load_clean_table
    from fpp.config import previous_season, season_sort_key
    from fpp.priors import BufferWindows
    from fpp.spec import TARGETS

    df = load_clean_table()
    hist = df[df["season"].map(season_sort_key) <= season_sort_key(previous_season())]
    hist = hist.reset_index(drop=True)

    ctx = RunContext.load()
    out = {"artifact_version": ctx.version, "clean_rows": len(hist), "clean": _hash(hist), "targets": {}}
    for t in TARGETS:
        w = ctx.windows[t]
        bw = BufferWindows(hist, L_max=max(50, int(w["L"])))
        ft = build_feature_table(hist, bw, int(w["L"]), float(w["alpha"]), features=ctx.features[t])
        out["targets"][t] = {"rows": len(ft), "X": _hash(ft.X), "y": _hash(pd.DataFrame(ft.y))}
    return out


if __name__ == "__main__":
    SNAPSHOT.write_text(json.dumps(fingerprint(), indent=2) + "\n")
    print(f"wrote {SNAPSHOT}")
