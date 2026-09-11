"""Execute a notebook's code cells as a plain script.

The venv has no nbclient/nbconvert, and adding them just to re-run an evaluation
would be a dependency the project does not otherwise need. The cells are ordinary
Python; the only notebook-isms are `display` and `get_ipython`, which are shimmed
below. Stored notebook outputs are *not* refreshed -- run it in Jupyter for that.

    python scripts/run_notebook_cells.py Notebooks/04_Evaluation.ipynb
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are saved by the cells, not shown


def _display(*objs) -> None:
    for o in objs:
        try:
            import pandas as pd

            if isinstance(o, pd.DataFrame):
                print(o.to_string(max_rows=60))
                continue
            styler = getattr(o, "data", None)
            if styler is not None and isinstance(styler, pd.DataFrame):
                print(styler.to_string(max_rows=60))
                continue
        except Exception:
            pass
        print(o)


def main() -> int:
    path = Path(sys.argv[1])
    nb = json.loads(path.read_text())
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]

    import matplotlib.pyplot as plt

    env = {"__name__": "__main__", "display": _display, "get_ipython": lambda: None}
    for i, cell in enumerate(cells, 1):
        code = "".join(cell["source"])
        if not code.strip():
            continue
        print(f"\n{'=' * 78}\n[cell {i}/{len(cells)}]\n{'=' * 78}")
        try:
            exec(compile(code, f"<cell {i}>", "exec"), env)
        except Exception:
            traceback.print_exc()
            return 1
        plt.close("all")
    print("\nAll cells completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
