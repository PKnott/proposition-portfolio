"""The Edge Book export: two JSON payloads, and the page that carries them.

Why this exists
---------------
`portfolio_excel` writes a table you filter *in Excel*, which cost it a `Query`
sheet whose predicate had to be authored twice -- once as a copied-down `Match`
formula for any version of Excel, once as a `FILTER()` block for 365 -- and a
5.7 MB file to hold a table nobody scrolls. A filter cockpit is a thing browsers
do well and spreadsheets do badly. So the same numbers come out here as JSON and
the filtering moves to a page.

Two files, because two notebooks own them
----------------------------------------
`predictions.json` is written by `05_Run` and `portfolios.json` by `06_Split`,
and that division is forced rather than stylistic. `06_Split` reads the filled
odds form *and nothing else* -- see `odds_form`'s docstring for why -- so it has
no access to `preds`, the clean table, or the dispersion parameters, and could
only recover a scoreline matrix by retraining or by joining strings back to a
workbook. Both are exactly what the self-contained form exists to avoid.

What is dropped on the way out
------------------------------
* **Dominated portfolios.** The workbook writes 20,000 rows carrying an
  `Undominated` flag; on a real run 4,940 of 200,000 scored rows are undominated
  and the app never shows the others. Filtering before export, not after load, is
  most of the size difference.
* **Leg rows.** The workbook's `Legs` sheet repeats every leg of every portfolio
  -- 29,741 rows for 2,000 portfolios. A portfolio's legs are its `picks`, looked
  up against `propositions`, so the array of ids is the whole of it.

What is added
-------------
* **`selection_frequency`** per proposition -- the share of exported portfolios
  containing it. One numpy pass here, or a scan of five thousand pick lists per
  keystroke in the browser.
* **`prices`** -- every proposition any book quoted, not just the ones that
  survived `E >= 1` and `(P, O)` dominance. The Match Board draws an edge bar
  wherever a market price exists, and a *negative* bar is information: it says
  the model looked and the price was not there. `propositions` alone cannot show
  that, because everything in it cleared the bar by construction.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections import Counter
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (
    GROWTH_BAND_QUANTILES,
    GROWTH_DRAWDOWN_D,
    GROWTH_DRAWDOWN_GRID_D,
    GROWTH_DRAWDOWN_GRID_P,
    GROWTH_DRAWDOWN_P,
    GROWTH_ROUNDS,
    MAX_LEG_STAKE,
    PESSIMISM_B,
    SLATE_TAU,
)
from ..config import EXPORT_MAX
from ..portfolio import SPLIT_EVEN, SPLIT_GROWTH, SPLIT_MINVAR
from ..spec import TARGETS
from ..staking import THRESHOLDS, add_edge, best_price, proposition_label
from .excel import sheet_codes
from .markets import (
    fixture_markets,
    ladder_lines,
    league_baselines,
    matrix_labels,
    safe_odds,
)

# The reported thresholds, as the integer percentages the app labels columns
# with. Derived from `staking.THRESHOLDS` -- the same derivation
# `portfolio_excel.THRESHOLD_KEYS` makes -- and carried *in the payload* so the
# page builds its `P(>X%)` columns from data. Adding a sixth threshold is then a
# line in `staking.py` and nothing else.
THRESHOLD_PCTS: tuple[int, ...] = tuple(int(round(t * 100)) for t in THRESHOLDS)
THRESHOLD_KEYS: tuple[str, ...] = tuple(f"p_over_{p}" for p in THRESHOLD_PCTS)

# Stake splits, as a stable key and the label to draw. The display strings are
# what `portfolio.SPLITS` holds and what `scored["split"]` is filled with; a
# payload keyed on "1/E" would make every consumer quote a string with a slash
# in it, and would change under any future rewording of the label.
#
# The two retired splits keep their keys. Runs scored under them are in the
# ledger and in archived payloads, and a reader that cannot name them cannot
# read its own history.
SPLIT_KEYS: dict[str, str] = {
    SPLIT_GROWTH: "growth", SPLIT_MINVAR: "min_variance", SPLIT_EVEN: "inverse_e"}

# What the page prints for each key. Not the inverse of `SPLIT_KEYS`: the sheets
# label the even split "1/E" and the design labels it "1/E split", and this is
# the presentation layer, so the design wins here without renaming a value the
# workbook and `scored["split"]` both carry.
SPLIT_LABELS: dict[str, str] = {
    "growth": "Growth", "min_variance": "Min variance", "inverse_e": "1/E split"}

APP_DIR = Path(__file__).resolve().parent.parent.parent / "app" / "edge-book"

# The three points `index.html` hands over to the bundler. Asserted present
# rather than silently skipped: a template that lost its data marker would build
# cleanly and open to an empty page.
MARKERS = ("<!--EDGE_BOOK_CSS-->", "<!--EDGE_BOOK_DATA-->", "<!--EDGE_BOOK_JS-->")

_DATED = re.compile(r"_(\d{4}-\d{2}-\d{2})\.json$")


# --- Rounding ---------------------------------------------------------------

# Probabilities are written to six decimal places and prices to three. Full
# float64 repr costs roughly a third of the file for digits no one reads and no
# arithmetic here depends on -- the dominance comparisons that *do* need exact
# equality all happened in Python, before this point.
def _p(x: float | None) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 6)


def _o(x: float | None) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)


def _sig(x: float | None, digits: int = 4) -> float | None:
    """Round to significant figures, not decimal places.

    For a quantity that spans orders of magnitude. `_p`'s six decimals are right
    for a probability that lives in [0.01, 1], and wrong for one that does not:
    ruin probability runs from 2.7e-09 to 0.63 across a single day's portfolios,
    and at six decimals the bottom 405 of 2,740 collapse to exactly ``0.0``.

    Storing zero there is not a rounding nicety, it is a false statement. Every
    portfolio can return nothing -- that is why `growth.optimal_fraction` caps
    ``f*`` below 1 -- so a row reading ``P(nothing) = 0`` next to ``f* = 0.94``
    contradicts the reason its own stake was capped.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    x = float(x)
    return 0.0 if x == 0 else float(f"%.{digits}g" % x)


def _arr(a, digits: int = 6) -> list | None:
    """A numpy row as a JSON-safe list, with every non-finite entry as ``null``.

    ``None`` in, ``None`` out, so a band that could not be computed stays absent
    rather than becoming an empty line the page would draw.

    Non-finite means ``null``, never a substitute value. `growth.growth_rate`
    returns ``-inf`` for a stake that is terminal and its docstring is explicit
    that a clipped logarithm "would make ruin look merely unattractive instead of
    terminal"; a finite stand-in invented here is a point someone plots. A
    ``null`` makes the client break the line, which is the honest shape.
    """
    if a is None:
        return None
    return [None if not np.isfinite(v) else round(float(v), digits) for v in np.asarray(a)]


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (dt.date, dt.datetime, pd.Timestamp)):
        return obj.isoformat()
    raise TypeError(f"{type(obj).__name__} is not JSON serialisable")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


# --- predictions.json -------------------------------------------------------


def _mean_from_survival(surv: np.ndarray | None) -> float | None:
    """``E[X]`` from ``surv[k] = P(X >= k)``, which is ``sum(surv[1:])``.

    Read off the baseline object the colours already use rather than recomputed
    from the clean table, because `league_baselines` restricts to the last two
    seasons and a second derivation would have to repeat that choice to agree
    with it. The final bucket is a cap, so this is a hair low on a heavy tail --
    acceptable for a figure shown as context, and wrong in a direction that
    understates rather than flatters.
    """
    return None if surv is None else round(float(np.asarray(surv)[1:].sum()), 3)


def _line_record(target: str, team: str, scope: str, line: float, mk: dict, base) -> dict:
    p = float(mk[target][f"{scope}_over_{line}"])
    return {
        "line": float(line),
        "label": proposition_label(target, team, line),
        "p": _p(p),
        "odds": _o(safe_odds(p)),
        "league_avg": _p(base.over_rate(target, scope, line)) if base is not None else None,
    }


def predictions_payload(preds: pd.DataFrame, team_matches: pd.DataFrame,
                        dispersion: dict | None = None,
                        generated_at: str | None = None) -> dict:
    """Every fixture's model output, keyed by the sheet code the pipeline uses.

    Computes nothing of its own: the markets come from `fixture_markets`, the
    ladders from `ladder_lines` and the comparison rates from `league_baselines`,
    which are the same three calls the fixture sheets and the odds form make. A
    fourth derivation of "which lines is this fixture priced at" is how the files
    stop lining up.
    """
    bases = league_baselines(team_matches)
    codes = sheet_codes(preds)
    axis = matrix_labels()

    fixtures = []
    for code, (_, row) in zip(codes, preds.iterrows()):
        mk = fixture_markets(row, dispersion)
        base = bases.get(row["league_key"])
        home, away = str(row["home_team"]), str(row["away_team"])
        g = mk["goals"]

        lines: dict[str, dict[str, list]] = {"home": {}, "away": {}}
        for target in TARGETS:
            home_lines, away_lines = ladder_lines(target, mk[target])
            for scope, team, ladder in (("home", home, home_lines), ("away", away, away_lines)):
                lines[scope][target] = [
                    _line_record(target, team, scope, line, mk, base) for line in ladder
                ]

        fixtures.append({
            "event": code,
            "date": pd.Timestamp(row["date"]).date().isoformat(),
            "league": str(row["league"]),
            "league_key": str(row["league_key"]),
            "home": home,
            "away": away,
            "exp_goals": _o(float(row["goals_home"]) + float(row["goals_away"])),
            "projections": {
                scope: {t: _o(float(row[f"{t}_{scope}"])) for t in TARGETS}
                for scope in ("home", "away")
            },
            "scoreline_matrix": {
                "rows": [f"{home} {lab}" for lab in axis],
                "cols": [f"{away} {lab}" for lab in axis],
                "values": [[_p(v) for v in r] for r in g["matrix"]],
            },
            "match_markets": {
                "home_win": _p(g["p_home"]),
                "draw": _p(g["p_draw"]),
                "away_win": _p(g["p_away"]),
                "btts_yes": _p(g["p_btts_yes"]),
                "btts_no": _p(g["p_btts_no"]),
            },
            "lines": lines,
            "league_context": None if base is None else {
                "seasons": len(base.seasons),
                "matches": int(base.n_matches),
                "avg_goals": _mean_from_survival(base.surv.get(("goals", "match"))),
                "rates": {
                    "home_win": _p(base.p_home),
                    "draw": _p(base.p_draw),
                    "away_win": _p(base.p_away),
                    "btts_yes": _p(base.p_btts_yes),
                    "btts_no": _p(base.p_btts_no),
                },
            },
        })

    return {
        "generated_at": generated_at or _now(),
        "targets": list(TARGETS),
        "fixtures": fixtures,
    }


# --- portfolios.json --------------------------------------------------------


def _selection_frequency(picks_lists: list[list[str]], ids: list[str]) -> dict[str, float]:
    """Share of exported portfolios containing each id.

    Every id gets an entry, including the ones no exported portfolio reached: a
    proposition that qualified and was then never picked is a fact worth seeing
    on the Match Board, and a missing key would render as blank rather than 0%.
    """
    n = len(picks_lists)
    if n == 0:
        return {pid: 0.0 for pid in ids}
    counts = Counter(chain.from_iterable(picks_lists))
    return {pid: round(counts.get(pid, 0) / n, 4) for pid in ids}


def _stake_fractions(kept: pd.DataFrame, picks: np.ndarray, opts) -> list[list[float]]:
    """Each exported portfolio's per-leg stakes, as fractions of the total.

    Computed here rather than in the browser, deliberately. The split is a closed
    form and a JavaScript copy of it would be small, correct on the day it was
    written, and completely untested. That is the shape of every "second
    derivation that happens to agree today" this pipeline has already been bitten
    by. `stakes_for` is the one definition, `tests/test_staking.py` pins it, and
    the page just multiplies by whatever is in the stake box.

    **One stake per proposition, not per event.** `stakes_for` returns two slots
    per event, the second filled only where the option is a pair, so a portfolio
    holding a pair emits two stakes for that match. The order matches `picks`:
    events in index order, and within an event the pair's first leg then its
    second -- which is the order `legs` expands them in.
    """
    from ..portfolio import stakes_for

    if not len(kept):
        return []
    out: list[list[float] | None] = [None] * len(kept)
    positions = {label: i for i, label in enumerate(kept.index)}
    for split, sub in kept.groupby("split", sort=False):
        combos = sub["combo"].to_numpy()
        rows = picks[combos]
        stakes, prob, _, _ = stakes_for(rows, opts, str(split))
        # A slot is a real bet when it carries a probability; that covers skipped
        # events and the empty second slot of a single in one test.
        for label, st, pr in zip(sub.index, stakes, prob):
            out[positions[label]] = [round(float(v), 6) for v in st[pr > 0]]
    assert all(v is not None for v in out), "a portfolio was left without stakes"
    return out  # type: ignore[return-value]


def _projection_axes() -> dict:
    """The stake and round axes every projection block is indexed on.

    Taken from `growth` rather than restated, so the page plots against the same
    grid `optimal_fraction` searched and the same rounds `wealth_bands` sampled.
    """
    from ..growth import band_rounds, f_curve

    return {"f": [round(float(v), 2) for v in f_curve()],
            "rounds": [int(v) for v in band_rounds()]}


def _band_rows(b) -> dict | None:
    """A ``(quantiles, rounds)`` band array as ``{"p5": [...], ...}``.

    ``None`` where the fan could not be drawn -- `growth.wealth_bands` returns it
    for a stake that is non-finite or that wipes the bankroll out on a reachable
    round. The key is absent rather than empty so the page can say so.

    Four decimals because these are *log* wealth: the whole exported range is
    roughly -5 to 20, so four decimals is more resolution than a chart can show,
    and the multiple the client exponentiates to inherits its precision from the
    pot rather than from here.
    """
    if b is None:
        return None
    return {f"p{q}": _arr(row, 4) for q, row in zip(GROWTH_BAND_QUANTILES, np.asarray(b))}


def _growth_blocks(kept: pd.DataFrame, picks: np.ndarray, opts, *,
                   projection: bool = False) -> tuple[list[dict], list[dict | None]]:
    """Each exported portfolio's growth block, and its projection block if on.

    Grouped by split and ordered back into `kept`'s order, the same way
    `_stake_fractions` does it and for the same reason: `stakes_for` answers a
    whole split at once, and re-deriving per portfolio would be both slower and a
    second copy of the allocation rule.

    Both blocks come out of **one** `growth_metrics` call rather than two,
    because the projection reads that call's `sample_rounds` draw. A second call
    would redraw it, and the fan would then describe a different 4,000 futures
    from the `drawdown.p` printed beside it -- two answers to one question, which
    is the thing this module keeps designing out.

    Runs on the undominated set only -- dominance is already settled by this
    point and neither block feeds back into it.
    """
    from ..growth import growth_metrics
    from ..portfolio import stakes_for

    if not len(kept):
        return [], []
    out: list[dict | None] = [None] * len(kept)
    proj: list[dict | None] = [None] * len(kept)
    positions = {label: i for i, label in enumerate(kept.index)}
    for split, sub in kept.groupby("split", sort=False):
        rows = picks[sub["combo"].to_numpy()]
        stakes, prob, odds, corr = stakes_for(rows, opts, str(split))
        g = growth_metrics(stakes * odds, prob, rho=corr, projection=projection)
        for label, rec in zip(sub.index, g.to_dict("records")):
            i = positions[label]
            out[i] = {
                # spans ~9 orders of magnitude in one day's set -- see `_sig`
                "p0": _sig(rec["p0"]),
                "f_suggested": _p(rec["f_suggested"]),
                "g_suggested": _p(rec["g_suggested"]),
                # What the one number is made of: the drawdown constraint, then
                # each model-risk dial's multiplier. Shipped so the page can say
                # *why* the stake is what it is instead of asserting it.
                "f_drawdown": _p(rec["f_drawdown"]),
                "edge_factor": _p(rec["edge_factor"]),
                "var_factor": _p(rec["var_factor"]),
                "k_leverage": _p(rec["k_leverage"]),
                "breakeven_shift": _p(rec["breakeven_shift"]),
                "drawdown": {"d": _p(rec["drawdown_d"]), "p": _p(rec["drawdown_p"])},
                # The stake at every tolerance in the grid, so the page can offer
                # the risk setting as a choice rather than stating one answer.
                # The published stake above is this dict's default cell -- every
                # cell carries the same model-risk haircuts, so the two can never
                # read differently. `edge_factor` and `var_factor` above are kept
                # as the audit trail for why a cell reads what it reads; the page
                # does no arithmetic with them.
                "drawdown_grid": {k[3:]: _p(v) for k, v in rec.items()
                                  if isinstance(k, str) and k.startswith("dd_")},
            }
            if not projection:
                continue
            # `f_grid` and `band_rounds` are the same two axes for every
            # portfolio, so they are hoisted into `projection_axes` on the
            # payload instead of repeated here -- 120 numbers a row across
            # thousands of rows is most of what this block would otherwise weigh.
            proj[i] = {
                "g_curve": _arr(rec["g_curve"]),
                "bands": {"suggested": _band_rows(rec["bands_suggested"])},
                "hist": _arr(rec["hist"]),
                "hist_step": _p(rec["hist_step"]),
                "max_return": _p(rec["max_return"]),
                # the mirror of `p0`, and small for the same reason -- see `_sig`
                "p_max": _sig(rec["p_max"]),
            }
    assert all(v is not None for v in out), "a portfolio was left without a growth block"
    return out, proj  # type: ignore[return-value]


def portfolios_payload(result: dict, filled: pd.DataFrame | None = None, *,
                       source: str | None = None,
                       generated_at: str | None = None,
                       with_growth: bool = True,
                       with_projection: bool = True,
                       run_code: str | None = None) -> dict:
    """A `portfolio.search` result, undominated only, with legs as id references.

    `run_code` is the ledger's name for this run. Portfolio ids restart at 1 every
    run, so "portfolio 770" stops identifying anything the moment there are two;
    carrying the code lets the page print `R007-770`, which is what
    `ledger.record_bet` takes back.
    """
    scored = result["scored"]
    opts = result["options"]
    info = dict(result.get("info") or {})

    props = opts.props if len(getattr(opts, "props", [])) else pd.DataFrame()
    prop_ids: list[str] = []
    prop_rows: list[dict] = []
    for rec in props.to_dict("records"):
        # The same token `picks_label` writes, so `picks` resolves against this
        # table by construction rather than by both encoding the rule the same way.
        pid = f"{rec['sheet_code']}#{int(rec['option'])}"
        prop_ids.append(pid)
        prop_rows.append({
            "id": pid,
            "event": str(rec["sheet_code"]),
            "fixture": rec.get("fixture") or f"{rec.get('home_team', '')} vs {rec.get('away_team', '')}",
            "proposition": str(rec["label"]),
            "p": _p(rec["p"]),
            "odds": _o(rec["o"]),
            "book": rec.get("book"),
            "e": _p(rec["e"]),
        })

    kept = scored[scored["undominated"]] if len(scored) else scored
    # Bound what reaches the page. Usually a no-op; see `portfolio.thin_export`
    # for the slate shape it is not.
    if len(kept) > EXPORT_MAX:
        from ..portfolio import thin_export
        kept = kept[thin_export(kept)]
    picks_lists = [str(s).split() for s in kept["picks"]] if len(kept) else []
    freq = _selection_frequency(picks_lists, prop_ids)
    for rec in prop_rows:
        rec["selection_frequency"] = freq[rec["id"]]

    stake_lists = _stake_fractions(kept, result["picks"], opts)
    # The projection rides on the growth block's own call -- it has no meaning
    # without `f_star` and `f_protective`, and asking for it with growth off
    # would silently produce neither.
    projecting = with_growth and with_projection
    growth_blocks, projection_blocks = (
        _growth_blocks(kept, result["picks"], opts, projection=projecting) if with_growth
        else ([None] * len(kept), [None] * len(kept)))
    portfolios = [
        {
            "id": int(rec["id"]),
            "split": SPLIT_KEYS[rec["split"]],
            # `legs` is propositions held, `events` is matches backed. They differ
            # wherever an option is a pair, and conflating them was the bug that
            # would have turned a leg-count control into a lid on propositions.
            "legs": int(rec["legs"]),
            "events": int(rec.get("events", rec["legs"])),
            "expected_return_pct": _p(rec["pct_expected_return"]),
            "sd_pct": _p(rec["pct_sd"]),
            "variance": round(float(rec["variance"]), 8),
            # `capacity` is the leg set's C -- what it is worth weighted ideally,
            # and the one channel leg count enters the model through. The rest say
            # how much of it this row's stakes actually reach and how concentrated
            # they are; `.get` because payloads predate them.
            "capacity": _p(rec.get("capacity")),
            "capacity_used": _p(rec.get("capacity_used")),
            "max_leg_stake": _p(rec.get("max_leg_stake")),
            "legs_at_cap": int(rec.get("legs_at_cap") or 0),
            "n_eff": _p(rec.get("n_eff")),
            "median_leg_p": _p(rec.get("median_leg_p")),
            **{k: _p(rec[k]) for k in THRESHOLD_KEYS},
            "picks": picks,
            "stakes": stakes,
            **({"growth": growth} if growth else {}),
            **({"projection": projection} if projection else {}),
        }
        for rec, picks, stakes, growth, projection in zip(
            kept.to_dict("records") if len(kept) else [],
            picks_lists, stake_lists, growth_blocks, projection_blocks)
    ]

    prices = []
    if filled is not None and len(filled):
        priced = add_edge(best_price(filled))
        prices = [
            {
                "event": str(rec["sheet_code"]),
                "label": str(rec["label"]),
                "p": _p(rec["p"]),
                "odds": _o(rec["o"]),
                "book": rec.get("book"),
                "e": _p(rec["e"]),
            }
            for rec in priced.to_dict("records")
        ]

    return {
        "generated_at": generated_at or _now(),
        "run_code": run_code,
        "source_odds_file": source,
        "min_legs": info.get("min_legs"),
        "max_legs": info.get("max_legs"),
        "thresholds": list(THRESHOLD_PCTS),
        "splits": SPLIT_LABELS,
        # The settings that decide the stake, shipped so the page can state them
        # rather than present the number as an oracle. Every one is a choice: the
        # suggested stake means "the largest fraction keeping P(a `drawdown_d`
        # fall within `horizon_rounds` rounds) under `drawdown_p`", shrunk for
        # model risk and capped. Read them and the number stops being magic.
        "risk": {
            "grid_d": list(GROWTH_DRAWDOWN_GRID_D),
            "grid_p": list(GROWTH_DRAWDOWN_GRID_P),
            "drawdown_d": GROWTH_DRAWDOWN_D,
            "drawdown_p": GROWTH_DRAWDOWN_P,
            "horizon_rounds": GROWTH_ROUNDS,
            "pessimism_b": PESSIMISM_B,
            "slate_tau": SLATE_TAU,
            "max_leg_stake": MAX_LEG_STAKE,
        },
        # The two axes every projection block is indexed on. Written once
        # because they are the same arrays for every portfolio, and absent
        # entirely when nothing carries a projection -- which is how the page
        # tells "this run has no Projection Book" from "this portfolio does not".
        **({"projection_axes": _projection_axes()} if projecting and portfolios else {}),
        "search": info,
        "counts": {
            "scored": int(len(scored)),
            "exported": len(portfolios),
            "events": int(getattr(opts, "n_events", 0)),
            "propositions": len(prop_rows),
            "priced": len(prices),
        },
        "propositions": prop_rows,
        "prices": prices,
        "portfolios": portfolios,
    }


# --- Writing ----------------------------------------------------------------


def _write_json(payload: dict, out_path: Path, prefix: str | None = None) -> Path:
    """Write the payload, and drop any earlier dated copy of the same kind.

    No history here on purpose: this file rebuilds from the workbook or the form
    in seconds, and the page it fed is archived whole. A dated pile of them is
    just a pile.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, default=_json_default), encoding="utf-8")
    if prefix:
        for old in out_path.parent.glob(f"{prefix}_*.json"):
            if old.resolve() != out_path.resolve():
                old.unlink()
    return out_path


def write_predictions_json(preds: pd.DataFrame, team_matches: pd.DataFrame,
                           out_path: Path | None = None,
                           dispersion: dict | None = None) -> Path:
    """`Outputs/Data/predictions_YYYY-MM-DD.json` -- the Match Board's whole input."""
    from ..paths import OUTPUTS_DATA

    payload = predictions_payload(preds, team_matches, dispersion)
    out_path = Path(out_path) if out_path else OUTPUTS_DATA / f"predictions_{dt.date.today().isoformat()}.json"
    _write_json(payload, out_path, "predictions")
    print(f"Wrote {len(payload['fixtures'])} fixtures -> {out_path}")
    return out_path


def write_portfolios_json(result: dict, filled: pd.DataFrame | None = None,
                          out_path: Path | None = None,
                          source: str | None = None,
                          run_code: str | None = None) -> Path:
    """`Outputs/Data/portfolios_YYYY-MM-DD.json` -- the Portfolio Book's whole input."""
    from ..paths import OUTPUTS_DATA

    payload = portfolios_payload(result, filled, source=source, run_code=run_code)
    out_path = Path(out_path) if out_path else OUTPUTS_DATA / f"portfolios_{dt.date.today().isoformat()}.json"
    _write_json(payload, out_path, "portfolios")
    c = payload["counts"]
    print(f"Wrote {c['exported']:,} undominated portfolios of {c['scored']:,} scored, "
          f"{c['propositions']} propositions ({c['priced']} priced) -> {out_path}")
    return out_path


# --- The bundle -------------------------------------------------------------


def latest_json(prefix: str, outputs: Path | None = None) -> Path | None:
    """The newest ``{prefix}_YYYY-MM-DD.json`` in `Outputs`, by the date it names.

    By the *name*, not the mtime: re-exporting an old run to pick up a schema
    change touches the file without making it current, and this is what decides
    which pair the bundle is built from.
    """
    from ..paths import OUTPUTS_DATA

    root = Path(outputs) if outputs else OUTPUTS_DATA
    dated = [(m.group(1), p) for p in root.glob(f"{prefix}_*.json")
             if (m := _DATED.search(p.name))]
    return max(dated)[1] if dated else None


def _date_of(path: Path) -> str | None:
    m = _DATED.search(Path(path).name)
    return m.group(1) if m else None


def write_edge_book(predictions: Path | None = None, portfolios: Path | None = None,
                    out_path: Path | None = None, app_dir: Path | None = None) -> Path:
    """Fuse the app and one day's data into a single file you can double-click.

    A page opened from `file://` has an opaque origin and cannot `fetch` a
    sibling JSON file -- Chrome refuses it -- so a "static app plus two data
    files" layout only works behind a server. Inlining sidesteps that entirely:
    the app source stays three editable files in `app/edge-book/`, and this is
    the build step.

    The two payloads come from different notebooks on different runs, so a
    mismatched pair is reachable by simply not re-running one of them. That is
    the same failure `check_form_is_current` exists to catch, and it is caught
    the same way -- loudly, before anything is written.
    """
    from ..paths import OUTPUTS, publish

    app_dir = Path(app_dir) if app_dir else APP_DIR
    predictions = Path(predictions) if predictions else latest_json("predictions")
    portfolios = Path(portfolios) if portfolios else latest_json("portfolios")
    for name, path in (("predictions", predictions), ("portfolios", portfolios)):
        if path is None or not Path(path).exists():
            raise FileNotFoundError(
                f"no {name} JSON found -- run "
                f"{'05_Run' if name == 'predictions' else '06_Split'} first")

    d_pred, d_port = _date_of(predictions), _date_of(portfolios)
    if d_pred and d_port and d_pred != d_port:
        raise ValueError(
            f"predictions ({d_pred}) and portfolios ({d_port}) are from different "
            f"runs; re-run the notebook that is behind, or pass both paths "
            f"explicitly to build the pair deliberately")

    template = (app_dir / "index.html").read_text(encoding="utf-8")
    css = (app_dir / "edge-book.css").read_text(encoding="utf-8")
    js = (app_dir / "edge-book.js").read_text(encoding="utf-8")
    missing = [m for m in MARKERS if m not in template]
    if missing:
        raise ValueError(f"{app_dir / 'index.html'} is missing {', '.join(missing)}")
    for name, src in (("edge-book.css", css), ("edge-book.js", js)):
        if "</script" in src.lower() or "</style" in src.lower():
            raise ValueError(f"{name} contains a closing tag that would end its own block")

    data = "\n".join(
        f'<script id="{key}" type="application/json">{_embed(Path(path))}</script>'
        for key, path in (("predictions-data", predictions), ("portfolios-data", portfolios))
    )
    html = (template
            .replace(MARKERS[0], f"<style>\n{css}\n</style>")
            .replace(MARKERS[1], data)
            .replace(MARKERS[2], f"<script>\n{js}\n</script>"))

    stamp = d_port or d_pred or dt.date.today().isoformat()
    out_path = Path(out_path) if out_path else OUTPUTS / f"edge_book_{stamp}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    publish(out_path, "edge_book")
    print(f"Wrote {out_path.stat().st_size / 1e6:.1f} MB -> {out_path}")
    return out_path


def _embed(path: Path) -> str:
    """A JSON file's text, safe to sit inside a `<script>` element.

    `<` is escaped to its `\\u003c` form, which JSON treats as the same string and
    an HTML parser cannot read as the start of a tag. Without it a team named in
    the data could in principle close the block early -- unlikely, and the sort
    of unlikely that only shows up in production.
    """
    return path.read_text(encoding="utf-8").replace("<", "\\u003c")


__all__ = [
    "predictions_payload", "portfolios_payload",
    "write_predictions_json", "write_portfolios_json", "write_edge_book",
    "latest_json", "THRESHOLD_PCTS", "THRESHOLD_KEYS", "SPLIT_KEYS", "APP_DIR",
]
