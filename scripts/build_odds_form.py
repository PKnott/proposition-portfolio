"""Build the odds capture form from an existing predictions workbook.

`05_Run` writes the form automatically, but doing so costs a full retrain first.
Every predicted rate the form needs is already recorded in the last workbook's
metadata block, so this recovers them and writes the form directly -- the same
trick, and the same `read_preds`, as `rebuild_workbook_from_output.py`.

    python scripts/build_odds_form.py [Outputs/predictions_<date>.xlsx]

Regenerates presentation from stored predictions; it cannot produce new ones.
"""

from __future__ import annotations

import sys
from pathlib import Path

import fpp
from fpp import staking
from fpp.report.odds_form import write_odds_form
from rebuild_workbook_from_output import read_preds


def _current_predictions() -> Path:
    """The predictions workbook in the `Outputs` root.

    The layout guarantees there is exactly one, which is why this can be a
    default at all -- the old hardcoded date silently stopped resolving the day
    it was archived.
    """
    found = sorted(fpp.paths.OUTPUTS.glob("predictions_*.xlsx"))
    if not found:
        sys.exit("No predictions workbook in Outputs/ -- run 05_Run first, "
                 "or pass one from Outputs/Archive/predictions/.")
    return found[-1]


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else _current_predictions()
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    preds = read_preds(src)
    ctx = fpp.RunContext.load()
    props = staking.propositions(preds, ctx.dispersion)
    qual = staking.qualifying(props)

    n_books = len(staking.BOOK_COLUMNS)
    print(f"recovered {len(preds)} fixtures from {src}")
    print(f"{len(props)} propositions -> {len(qual)} at P >= {staking.MIN_MODEL_P:.2f} "
          f"({len(qual) / len(props):.1%}, {len(qual) / len(preds):.1f} per match, "
          f"{len(qual) * n_books:,} cells across {n_books} books)")
    write_odds_form(qual, out_path=dst, source=src.name)


if __name__ == "__main__":
    main()
