"""The European run: the league pipeline's own writers, pointed at `Outputs/Euro/`.

Nothing here prices, searches or settles anything itself. `05b_Euro_Run` and
`06b_Euro_Split` call the same `report`, `staking`, `portfolio` and `ledger`
functions `05_Run` and `06_Split` do, and this module supplies the three things
that differ:

* **where files go** -- `Outputs/Euro/`, so the domestic writers' housekeeping
  (`paths.publish` archives only the `Outputs` root; `_write_json` prunes only
  its own folder) never touches a European file, and vice versa;
* **the ledger store** -- `Outputs/Euro/Analysis/`, with its own `E001`, `E002`
  run codes, so a European slate is never absorbed into the league ledger or
  confused with an `R` run;
* **results** -- European scoreboards, refetched for the dates the European
  ledger is still waiting on and passed to `ledger.settle` as data.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import pandas as pd

from .. import ledger
from ..ingest.retry import ESPN_SITE_API, get_json
from ..report import write_odds_form, write_predictions_json, write_workbook
from ..report.edge_book import latest_json, write_edge_book, write_portfolios_json
from ..report.odds_form import latest_form
from . import ingest
from .config import EURO_ANALYSIS, EURO_CACHE, EURO_COMPS, EURO_MATCH_STATS, EURO_OUTPUTS

_RUN_CODE = re.compile(r"^E(\d+)$")


def _today() -> str:
    return dt.date.today().isoformat()


# --- 05b: predictions and the odds form ---------------------------------------------


def write_outputs(preds: pd.DataFrame, qualified: pd.DataFrame, baselines: pd.DataFrame,
                  dispersion: dict, *, workbook: bool = True, outputs: Path | None = None) -> dict[str, Path]:
    """The workbook, the Match Board payload and the odds form, all under `Outputs/Euro/`."""
    outputs = Path(outputs) if outputs else EURO_OUTPUTS
    d = _today()
    out: dict[str, Path] = {}
    if workbook:
        out["workbook"] = write_workbook(preds, baselines, outputs / f"predictions_{d}.xlsx",
                                         dispersion=dispersion)
    out["predictions_json"] = write_predictions_json(preds, baselines, outputs / "Data" / f"predictions_{d}.json",
                                                     dispersion=dispersion)
    out["odds_form"] = write_odds_form(qualified, outputs / f"odds_input_{d}.xlsx",
                                       source=(out.get("workbook") or out["predictions_json"]).name)
    return out


# --- 06b: the European ledger and Edge Book --------------------------------------------


def latest_euro_form(outputs: Path | None = None) -> Path | None:
    return latest_form(Path(outputs) if outputs else EURO_OUTPUTS)


def next_run_code(root: Path | None = None) -> str:
    """``E001``, ``E002`` ... over absorbed runs and pending slates in the European store."""
    root = root or EURO_ANALYSIS
    used = set()
    runs = ledger.read_table("runs", root)
    if not runs.empty:
        used |= set(runs["run_code"].astype(str))
    used |= {p.name.split("_")[1] for p in ledger.pending_slates(root)}
    n = max((int(m.group(1)) for c in used if (m := _RUN_CODE.match(c))), default=0)
    return f"E{n + 1:03d}"


def capture(result: dict, filled: pd.DataFrame, *, source: str | None = None,
            root: Path | None = None):
    """`ledger.capture` into the European store, under an ``E`` run code."""
    root = Path(root) if root else EURO_ANALYSIS
    return ledger.capture(result, filled, source=source, run_code=next_run_code(root), root=root)


def write_book(result: dict, filled: pd.DataFrame, *, source: str | None, run_code: str,
               outputs: Path | None = None) -> Path:
    """The portfolio payload and the Edge Book, from today's European pair only."""
    outputs = Path(outputs) if outputs else EURO_OUTPUTS
    d = _today()
    port = write_portfolios_json(result, filled, outputs / "Data" / f"portfolios_{d}.json",
                                 source=source, run_code=run_code)
    pred = latest_json("predictions", outputs / "Data")
    return write_edge_book(pred, port, outputs / f"edge_book_{d}.html")


# --- 07: settle --------------------------------------------------------------------------


def refresh_results(root: Path | None = None) -> pd.DataFrame:
    """Refetch the European scoreboards the ledger is waiting on, rebuild, return results.

    Only the dates with an unsettled proposition are refetched, and they are
    always refetched: a scoreboard cached before kick-off has fixtures and no
    statistics, and would otherwise be read as the answer.
    """
    root = root or EURO_ANALYSIS
    props = ledger.read_table("propositions", root)
    pending = props[props["won"].isna()] if not props.empty else props
    days = sorted({d for x in pd.to_datetime(pending.get("fixture_date", pd.Series(dtype="datetime64[ns]")))
                   for d in (x - pd.Timedelta(days=1), x, x + pd.Timedelta(days=1))})
    EURO_CACHE.mkdir(parents=True, exist_ok=True)
    for comp in sorted(set(pending.get("league_key", []))):
        slug = EURO_COMPS[comp].espn_slug
        for d in days:
            payload = get_json(f"{ESPN_SITE_API}/{slug}/scoreboard?dates={d:%Y%m%d}")
            (EURO_CACHE / f"Schedule_{slug}_{d:%Y%m%d}.json").write_text(json.dumps(payload))
    ingest.build_match_stats(save=True)
    return ledger.load_results(EURO_MATCH_STATS)


def settle(root: Path | None = None) -> dict:
    """Absorb pending European slates, fetch their results, and settle."""
    root = root or EURO_ANALYSIS
    ledger.absorb(root)
    return ledger.settle(root, results=refresh_results(root))


__all__ = ["write_outputs", "latest_euro_form", "next_run_code", "capture", "write_book",
           "refresh_results", "settle"]
