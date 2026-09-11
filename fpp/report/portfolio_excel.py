"""The portfolio workbook: what survived, every mix worth considering, and a calculator.

The old split workbook ended in an answer -- one shortlist, one set of stakes.
This one ends in a *choice*, because at portfolio level there is no single best
outcome: raising expected return raises variance, and the mix with the highest
probability of profit is rarely the mix with the highest return. So the centre of
this file is a table you filter rather than a list you read.

Filtering, twice over
---------------------
The `Query` sheet holds one input cell per criterion and drives the results two
different ways, deliberately:

* a **`Match` column** on `Portfolios`, recomputed by formula from those cells,
  which every version of Excel can filter on; and
* a **`FILTER()` block** on `Query` itself, which spills live in Excel 365.

Neither is a fallback for the other -- they are the same predicate, written twice
for two audiences, and a test pins them to return the same rows.

Live where it matters
---------------------
As on the old calculator, the stake column and the cash figures are formulas keyed
to `B1`, so typing a new total re-costs the sheet without rerunning anything. That
is affordable because every stake is a fixed *fraction* of the total, so the whole
return distribution scales linearly with it: every percentage on the sheet is the
same number whatever `B1` says, and only cash moves.

`B2` takes a portfolio id and re-drives the legs the same way. The calculator is
therefore not a recommendation -- it is whichever row of the explorer you point it
at.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from ..portfolio import DOMINANCE_CRITERIA, SPLITS, legs as portfolio_legs
from ..staking import THRESHOLDS
from .style import (
    BORDER,
    CENTER,
    FILL_CARD,
    FILL_HEADER,
    FILL_MAX,
    FILL_TITLE,
    FONT_H,
    FONT_TITLE,
    LEFT,
    NUM3,
    ODDS,
    PCT,
    header,
    plain_cell,
    table_header,
)

MONEY = "#,##0.00"
VAR_FMT = "0.00000"

# Excel will not accept a post-2007 function written by a non-Excel tool under its
# plain name. It stores them in a "future functions" namespace, and a bare
# `FILTER(` in the XML makes the whole worksheet part invalid -- Excel opens with
# "We found a problem with some content" and offers to repair, which is what a
# first run of this writer did.
#
# `FILTER` and `SORT` additionally sit in the worksheet sub-namespace `_xlws`,
# because those names already existed elsewhere in the format. Everything else
# added since 2007 takes the plain `_xlfn.` prefix.
#
# `IFNA` is not here because it is not used: `IFERROR` is a 2007 function, needs
# no prefix, and guards exactly the same lookup failure.
FUTURE_FUNCTIONS: dict[str, str] = {
    "FILTER": "_xlfn._xlws.FILTER",
    "SORT": "_xlfn._xlws.SORT",
}


def future(name: str) -> str:
    """The namespaced spelling of a function Excel considers "future".

    Call this rather than typing the prefix, so adding one more dynamic-array
    function is a line in `FUTURE_FUNCTIONS` and not another corrupt workbook.
    `tests/test_portfolio_excel.py` scans every generated formula for bare names.
    """
    return FUTURE_FUNCTIONS.get(name, name)

# Rows written to `Portfolios`. Excel copes with far more, but the file grows and
# nobody scrolls a hundred thousand rows -- the undominated set is written first
# and is what the sheet is actually for, so the cap only ever truncates the
# also-rans. Named so the Contents sheet can say when it bit.
MAX_ROWS = 20_000

# Portfolios whose individual legs are written out, undominated first. The
# calculator can only cost a portfolio whose legs are here, so this wants to
# cover the whole undominated set -- 989 of them on a real run, at a dozen legs
# each. The cap exists so a pathological run cannot write a million rows.
MAX_LEG_PORTFOLIOS = 2_000

THRESHOLD_KEYS = tuple(f"p_over_{int(round(t * 100))}" for t in THRESHOLDS)

# (column key, heading, number format). One definition: the writer lays the sheet
# out from this, the query predicate finds its columns through it, and the tests
# read it rather than hardcoding letters.
COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("id", "ID", "0"),
    ("split", "Split", None),
    ("legs", "Legs", "0"),
    ("pct_expected_return", "Expected return %", PCT),
    ("pct_sd", "SD %", PCT),
    ("variance", "Variance", VAR_FMT),
    *((k, f"P(>{int(round(t * 100))}%)", PCT) for k, t in zip(THRESHOLD_KEYS, THRESHOLDS)),
    ("undominated", "Undominated", None),
    ("picks", "Picks", None),
)
COL_KEYS = tuple(c[0] for c in COLUMNS)
MATCH_COL = len(COLUMNS) + 1  # `Match` sits one past the data

HEADER_ROW = 3
FIRST_DATA_ROW = 4

# Query inputs: (label, column key, comparison, default). ">=" keeps rows at or
# above the cell, "<=" keeps rows at or below it. A blank cell means "no
# constraint", which is why every test is wrapped in an ISBLANK.
def _threshold_query_row(key: str, t: float) -> tuple[str, str, str, object]:
    """One query row per threshold, built from `THRESHOLDS` rather than listed.

    The list used to name all five by hand, so trimming the tuple left rows
    pointing at columns that no longer existed -- a formula referencing a missing
    key, which Excel reports as a broken name rather than as the edit it was.
    """
    pct = int(round(t * 100))
    label = f"Min P(>{pct}%)" + ("  (profitability)" if pct == 100 else "")
    return (label, key, ">=", 0.70 if pct == 100 else None)


QUERY_ROWS: tuple[tuple[str, str, str, object], ...] = (
    ("Min expected return %", "pct_expected_return", ">=", 1.02),
    ("Max SD %", "pct_sd", "<=", None),
    ("Max variance", "variance", "<=", None),
    *(_threshold_query_row(k, t) for k, t in zip(THRESHOLD_KEYS, THRESHOLDS)),
    ("Min legs", "legs", ">=", None),
    ("Max legs", "legs", "<=", None),
)
QUERY_FIRST_ROW = 4

# The four non-numeric inputs sit directly under the numeric block. **Derived,
# not typed**: they were hardcoded at B15-B18 while the block actually ended at
# B13, so the Match formula tested the "Undominated only" cell for the split and
# the "Sort by" cell for undominated -- a filter that looked plausible and
# filtered on the wrong thing. Adding a row to QUERY_ROWS now moves them.
_EXTRA_FIRST_ROW = QUERY_FIRST_ROW + len(QUERY_ROWS)
SPLIT_CELL = f"B{_EXTRA_FIRST_ROW}"       # "" for either split
UNDOM_CELL = f"B{_EXTRA_FIRST_ROW + 1}"   # TRUE to restrict to the undominated set
SORT_CELL = f"B{_EXTRA_FIRST_ROW + 2}"
DESC_CELL = f"B{_EXTRA_FIRST_ROW + 3}"
RESULT_ROW = _EXTRA_FIRST_ROW + 7


def _col_letter(key: str) -> str:
    return get_column_letter(COL_KEYS.index(key) + 1)


def _abs(cell: str) -> str:
    """``"B15"`` -> ``"$B$15"``. Query cells are referenced from copied-down
    formulas, so they have to be absolute or the predicate walks down with them."""
    col = "".join(ch for ch in cell if ch.isalpha())
    row = "".join(ch for ch in cell if ch.isdigit())
    return f"${col}${row}"


def _title(ws, text: str, span: int) -> None:
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=span)
    c = ws.cell(1, 1)
    c.value = text
    c.fill = FILL_TITLE
    c.font = FONT_TITLE
    c.alignment = LEFT


# --- Propositions -----------------------------------------------------------


def _write_propositions(ws, qualified: pd.DataFrame) -> None:
    _title(ws, f"{len(qualified)} propositions survived, across "
               f"{qualified['sheet_code'].nunique()} events", span=8)
    ws.cell(2, 1).value = ("Every price that beats the model (E >= 1) and that no other "
                           "proposition in the same event beats on both P and O. "
                           "A portfolio takes at most one row per event.")
    ws.cell(2, 1).font = Font(italic=True)

    table_header(ws, HEADER_ROW, 1,
                 ["Event", "Fixture", "Proposition", "P", "O", "Book", "E"])
    for i, rec in enumerate(qualified.to_dict("records")):
        r = FIRST_DATA_ROW + i
        plain_cell(ws, r, 1, rec["sheet_code"], fill=FILL_CARD, bold=True)
        fixture = rec.get("fixture") or f"{rec.get('home_team', '')} vs {rec.get('away_team', '')}"
        plain_cell(ws, r, 2, fixture).alignment = LEFT
        plain_cell(ws, r, 3, rec["label"]).alignment = LEFT
        plain_cell(ws, r, 4, float(rec["p"]), PCT)
        plain_cell(ws, r, 5, float(rec["o"]), ODDS)
        plain_cell(ws, r, 6, rec.get("book"))
        plain_cell(ws, r, 7, float(rec["e"]), NUM3)

    ws.auto_filter.ref = f"A{HEADER_ROW}:G{FIRST_DATA_ROW + len(qualified) - 1}"
    for col, w in zip("ABCDEFG", (12, 30, 44, 11, 10, 22, 10)):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = f"A{FIRST_DATA_ROW}"


# --- Portfolios -------------------------------------------------------------


def _match_formula(row: int) -> str:
    """``AND(...)`` over every query cell, skipping the blank ones.

    Written against one data row and copied down, so the column references are
    relative and the query references absolute -- which is what lets Excel apply
    the same predicate to twenty thousand rows from one authored formula.
    """
    tests = []
    for i, (_, key, cmp, _default) in enumerate(QUERY_ROWS):
        q = f"Query!$B${QUERY_FIRST_ROW + i}"
        col = f"{_col_letter(key)}{row}"
        tests.append(f"OR(ISBLANK({q}),{col}{cmp}{q})")
    tests.append(f'OR(Query!{_abs(SPLIT_CELL)}="",'
                 f"{_col_letter('split')}{row}=Query!{_abs(SPLIT_CELL)})")
    tests.append(f"OR(NOT(Query!{_abs(UNDOM_CELL)}),{_col_letter('undominated')}{row})")
    return "=AND(" + ",".join(tests) + ")"


def _write_portfolios(ws, scored: pd.DataFrame) -> int:
    """Returns the last data row written."""
    n_undom = int(scored["undominated"].sum())
    _title(ws, f"{len(scored):,} portfolios scored, {n_undom:,} undominated",
           span=MATCH_COL)
    ws.cell(2, 1).value = (
        "One row per portfolio per stake split. Undominated means nothing else here "
        f"beats it on all of {', '.join(DOMINANCE_CRITERIA)}. "
        "Filter on Match (driven by the Query sheet), or use the column filters directly."
    )
    ws.cell(2, 1).font = Font(italic=True)

    table_header(ws, HEADER_ROW, 1, [c[1] for c in COLUMNS] + ["Match"])
    for i, rec in enumerate(scored.to_dict("records")):
        r = FIRST_DATA_ROW + i
        for j, (key, _label, fmt) in enumerate(COLUMNS):
            v = rec[key]
            if isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = float(v)
            elif isinstance(v, (np.bool_,)):
                v = bool(v)
            cell = plain_cell(ws, r, 1 + j, v, fmt)
            if key in ("picks",):
                cell.alignment = LEFT
            if key == "p_over_100" and rec["undominated"]:
                cell.fill = FILL_MAX
        plain_cell(ws, r, MATCH_COL, _match_formula(r))

    last = FIRST_DATA_ROW + len(scored) - 1
    ws.auto_filter.ref = f"A{HEADER_ROW}:{get_column_letter(MATCH_COL)}{last}"
    for j, (key, label, _fmt) in enumerate(COLUMNS):
        ws.column_dimensions[get_column_letter(1 + j)].width = 60 if key == "picks" else max(12, len(label) + 3)
    ws.column_dimensions[get_column_letter(MATCH_COL)].width = 10
    ws.freeze_panes = f"D{FIRST_DATA_ROW}"
    return last


# --- Query ------------------------------------------------------------------


def _write_query(ws, last_row: int, n_rows: int) -> None:
    _title(ws, "Query - type your criteria, read the answers below", span=6)
    ws.cell(2, 1).value = (
        "Leave a cell blank to drop that constraint. The same test drives the Match "
        "column on Portfolios (filter on it there, in any version of Excel) and the "
        "live results below (Excel 365 only -- older versions show #NAME?)."
    )
    ws.cell(2, 1).font = Font(italic=True)

    for i, (label, _key, cmp, default) in enumerate(QUERY_ROWS):
        r = QUERY_FIRST_ROW + i
        a = plain_cell(ws, r, 1, f"{label}  ({cmp})", fill=FILL_CARD, bold=True)
        a.alignment = LEFT
        b = plain_cell(ws, r, 2, default, PCT if "%" in label else None)
        b.fill = FILL_MAX

    r = _EXTRA_FIRST_ROW
    for label, cell, default, note in (
        ("Split", SPLIT_CELL, "", f"blank = either; otherwise one of {', '.join(SPLITS)}"),
        ("Undominated only", UNDOM_CELL, True, "TRUE hides everything that something else beats outright"),
        ("Sort by", SORT_CELL, "p_over_100", f"one of: {', '.join(COL_KEYS[2:-2])}"),
        ("Highest first", DESC_CELL, True, "FALSE to sort ascending"),
    ):
        assert cell == f"B{r}", f"{label} cell drifted: constant says {cell}, writing to B{r}"
        a = plain_cell(ws, r, 1, label, fill=FILL_CARD, bold=True)
        a.alignment = LEFT
        plain_cell(ws, r, 2, default).fill = FILL_MAX
        ws.cell(r, 3).value = note
        ws.cell(r, 3).font = Font(italic=True)
        r += 1

    dv = DataValidation(type="list", formula1='"' + ",".join(("", *SPLITS)) + '"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(SPLIT_CELL)
    dv2 = DataValidation(type="list", formula1='"' + ",".join(COL_KEYS[2:-2]) + '"', allow_blank=True)
    ws.add_data_validation(dv2)
    dv2.add(SORT_CELL)

    header(ws, RESULT_ROW - 1, 1, len(COLUMNS), "Matching portfolios")
    table_header(ws, RESULT_ROW, 1, [c[1] for c in COLUMNS])

    data = f"Portfolios!$A${FIRST_DATA_ROW}:${_col_letter('picks')}${last_row}"
    match = f"Portfolios!${get_column_letter(MATCH_COL)}${FIRST_DATA_ROW}:${get_column_letter(MATCH_COL)}${last_row}"
    # The sort key is chosen by name, so it is looked up rather than hardcoded:
    # MATCH finds which position the name sits at in the column-key list, and SORT
    # is handed that index. Changing COLUMNS therefore moves the formula with it.
    key_names = "{" + ";".join(f'"{k}"' for k in COL_KEYS) + "}"
    sort_idx = f"MATCH({SORT_CELL},{key_names},0)"
    order = f"IF({DESC_CELL},-1,1)"
    ws.cell(RESULT_ROW + 1, 1).value = (
        f'=IFERROR({future("SORT")}({future("FILTER")}({data},{match},'
        f'"no portfolio matches"),{sort_idx},{order}),"no portfolio matches")'
    )

    ws.cell(RESULT_ROW - 2, 1).value = (
        f"{n_rows:,} portfolios are on the Portfolios sheet; the block below spills as many "
        "as match. In older Excel, filter Portfolios on Match = TRUE instead."
    )
    ws.cell(RESULT_ROW - 2, 1).font = Font(italic=True)

    ws.column_dimensions["A"].width = 34
    ws.column_dimensions["B"].width = 16
    ws.column_dimensions["C"].width = 60
    for j in range(3, len(COLUMNS)):
        ws.column_dimensions[get_column_letter(1 + j)].width = 14


# --- Legs and the calculator ------------------------------------------------


def _leg_rows(scored: pd.DataFrame, picks: np.ndarray, options, cap: int) -> pd.DataFrame:
    """Individual bets for the portfolios worth writing out, undominated first."""
    order = scored.sort_values(["undominated", "p_over_100"], ascending=[False, False])
    frames = []
    for rec in order.head(cap).to_dict("records"):
        df = portfolio_legs(picks, options, int(rec["combo"]), rec["split"], total=1.0)
        df.insert(0, "id", int(rec["id"]))
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["id", "event", "fixture", "label", "p", "o", "book", "e", "stake"])


def _write_legs(ws, rows: pd.DataFrame) -> int:
    _title(ws, f"Legs for {rows['id'].nunique() if len(rows) else 0} portfolios "
               f"({len(rows):,} bets)", span=9)
    ws.cell(2, 1).value = ("Stake is a fraction of the total. The Stake Calculator reads this "
                           "sheet by ID, so a portfolio missing from here cannot be costed there.")
    ws.cell(2, 1).font = Font(italic=True)

    table_header(ws, HEADER_ROW, 1,
                 ["ID", "Event", "Fixture", "Proposition", "P", "O", "Book", "E", "Stake fraction"])
    for i, rec in enumerate(rows.to_dict("records")):
        r = FIRST_DATA_ROW + i
        plain_cell(ws, r, 1, int(rec["id"]), "0", fill=FILL_CARD, bold=True)
        plain_cell(ws, r, 2, rec["event"])
        plain_cell(ws, r, 3, rec["fixture"]).alignment = LEFT
        plain_cell(ws, r, 4, rec["label"]).alignment = LEFT
        plain_cell(ws, r, 5, float(rec["p"]), PCT)
        plain_cell(ws, r, 6, float(rec["o"]), ODDS)
        plain_cell(ws, r, 7, rec.get("book"))
        plain_cell(ws, r, 8, float(rec["e"]), NUM3)
        plain_cell(ws, r, 9, float(rec["stake"]), "0.0000")
    last = FIRST_DATA_ROW + len(rows) - 1
    ws.auto_filter.ref = f"A{HEADER_ROW}:I{max(last, HEADER_ROW)}"
    for col, w in zip("ABCDEFGHI", (8, 12, 28, 44, 11, 10, 20, 10, 14)):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = f"A{FIRST_DATA_ROW}"
    return last


def _write_calculator(ws, scored: pd.DataFrame, last_leg_row: int,
                      stake: float, default_id: int) -> None:
    ws.cell(1, 1).value = "Stake:"
    ws.cell(1, 1).font = FONT_H
    s_cell = plain_cell(ws, 1, 2, float(stake), MONEY)
    s_cell.fill = FILL_MAX
    ws.cell(1, 4).value = "Type any total here -- stakes and cash recalculate; percentages do not move."
    ws.cell(1, 4).font = Font(italic=True)

    ws.cell(2, 1).value = "Portfolio ID:"
    ws.cell(2, 1).font = FONT_H
    id_cell = plain_cell(ws, 2, 2, int(default_id), "0")
    id_cell.fill = FILL_MAX
    ws.cell(2, 4).value = ("Any ID from the Legs sheet. Everything below re-reads from it, so this "
                           "sheet is whichever portfolio you point it at -- not a recommendation.")
    ws.cell(2, 4).font = Font(italic=True)

    legs_id = f"Legs!$A${FIRST_DATA_ROW}:$A${max(last_leg_row, FIRST_DATA_ROW)}"
    legs_all = f"Legs!$B${FIRST_DATA_ROW}:$I${max(last_leg_row, FIRST_DATA_ROW)}"
    header(ws, 4, 1, 8, "The bets")
    ws.cell(5, 1).value = (f'=IFERROR({future("FILTER")}({legs_all},{legs_id}=$B$2),'
                           f'"no portfolio with that ID on the Legs sheet")')
    ws.cell(6, 1).value = ("Spills: Event | Fixture | Proposition | P | O | Book | E | Stake fraction. "
                           "Multiply the last column by B1 for cash.")
    ws.cell(6, 1).font = Font(italic=True)

    # The summary is looked up from the scored table rather than recomputed, so the
    # calculator and the explorer cannot print two different numbers for one
    # portfolio. Every figure here is a percentage and so is independent of B1.
    r = 8
    header(ws, r, 1, 3, "Outcome distribution")
    table_header(ws, r + 1, 1, ["", "% of stake", "Cash"])
    r += 2
    ids = _range_for(scored, "id")
    for label, key, fmt in (
        ("Expected return", "pct_expected_return", PCT),
        ("Standard deviation", "pct_sd", PCT),
        ("Legs", "legs", "0"),
    ):
        plain_cell(ws, r, 1, label, fill=FILL_CARD, bold=True).alignment = LEFT
        plain_cell(ws, r, 2, f'=IFERROR(INDEX({_range_for(scored, key)},MATCH($B$2,{ids},0)),"")', fmt)
        if key != "legs":
            plain_cell(ws, r, 3, f"=IFERROR(B{r}*$B$1,\"\")", MONEY)
        r += 1
    plain_cell(ws, r, 1, "Variance", fill=FILL_CARD, bold=True).alignment = LEFT
    plain_cell(ws, r, 2, f'=IFERROR(INDEX({_range_for(scored, "variance")},MATCH($B$2,{ids},0)),"")', VAR_FMT)
    plain_cell(ws, r, 3, f"=IFERROR(B{r}*$B$1^2,\"\")", MONEY)
    ws.cell(r, 4).value = "as a fraction of S^2 -- scales with the square, not the total"
    ws.cell(r, 4).font = Font(italic=True)
    r += 2

    header(ws, r, 1, 3, "Probability the return clears")
    r += 1
    for key, t in zip(THRESHOLD_KEYS, THRESHOLDS):
        label = f"{t:.0%} of stake" + ("   (% Profitability)" if t == 1.00 else "")
        plain_cell(ws, r, 1, label, fill=FILL_CARD, bold=True).alignment = LEFT
        c = plain_cell(ws, r, 2, f'=IFERROR(INDEX({_range_for(scored, key)},MATCH($B$2,{ids},0)),"")', PCT)
        if t == 1.00:
            c.fill = FILL_MAX
            c.font = FONT_H
        r += 1

    for col, w in zip("ABCDEFGH", (34, 16, 16, 44, 12, 12, 20, 12)):
        ws.column_dimensions[col].width = w


def _range_for(scored: pd.DataFrame, key: str) -> str:
    last = FIRST_DATA_ROW + len(scored) - 1
    col = _col_letter(key)
    return f"Portfolios!${col}${FIRST_DATA_ROW}:${col}${max(last, FIRST_DATA_ROW)}"


# --- Contents ---------------------------------------------------------------


def _write_contents(ws, qualified: pd.DataFrame, scored: pd.DataFrame,
                    info: dict, written: int, n_legs: int, source: str | None) -> None:
    _title(ws, f"Portfolio search - {int(scored['undominated'].sum()):,} undominated "
               f"of {len(scored):,} scored", span=4)
    r = 3
    for label, value in (
        ("Odds source", source or "(not recorded)"),
        ("Generated", dt.date.today().isoformat()),
        ("Propositions surviving E >= 1 and (P, O) dominance", len(qualified)),
        ("Events with at least one survivor", info.get("n_events", 0)),
        ("Portfolios in the search space", f"{info.get('space', 0):,.4g}"),
        ("How the space was covered", info.get("mode", "?")),
        ("Portfolios reached", f"{info.get('found', 0):,}"),
        ("Portfolios scored (2 splits each)", f"{len(scored):,}"),
        ("Rows written to Portfolios", f"{written:,}"),
        ("Portfolios costable in the calculator", f"{n_legs:,}"),
        ("Leg counts allowed", f"{info.get('min_legs', '?')} to {info.get('max_legs', '?')}"),
    ):
        a = ws.cell(r, 1)
        a.value = label
        a.font = FONT_H
        a.alignment = LEFT
        plain_cell(ws, r, 2, value)
        r += 1

    if info.get("mode") == "searched":
        ws.cell(r + 1, 1).value = (
            "The space was too large to enumerate, so these are the portfolios the search "
            "reached rather than all of them. The (expected return, variance) frontier is "
            "preserved exactly; measured coverage loss elsewhere is a handful of portfolios "
            "indistinguishable from their neighbours."
        )
        ws.cell(r + 1, 1).font = Font(italic=True)
        r += 2
    if written < len(scored):
        ws.cell(r + 1, 1).value = (
            f"Portfolios is capped at {MAX_ROWS:,} rows. Every undominated portfolio is written "
            "first, so what the cap dropped is all dominated."
        )
        ws.cell(r + 1, 1).font = Font(italic=True)
        r += 2

    r += 1
    for name in ("Propositions", "Portfolios", "Query", "Legs", "Stake Calculator"):
        c = ws.cell(r, 1)
        c.value = name
        c.hyperlink = f"#'{name}'!A1"
        c.font = Font(color="0563C1", underline="single", bold=True)
        r += 1

    ws.column_dimensions["A"].width = 52
    ws.column_dimensions["B"].width = 26


# --- Entry point ------------------------------------------------------------


def write_portfolio_workbook(result: dict, out_path: Path | None = None,
                             stake: float = 100.0, source: str | None = None,
                             max_rows: int = MAX_ROWS,
                             max_leg_portfolios: int = MAX_LEG_PORTFOLIOS) -> Path:
    """Write the whole workbook from a `portfolio.search` result. Computes nothing."""
    from ..paths import OUTPUTS, publish

    out_path = Path(out_path) if out_path else OUTPUTS / f"portfolios_{dt.date.today().isoformat()}.xlsx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scored = result["scored"]
    if scored.empty:
        with pd.ExcelWriter(out_path, engine="openpyxl") as w:
            pd.DataFrame({"Message": [
                "No portfolio was built. Either nothing was priced, or every price "
                "left an edge below 1.00."]}).to_excel(w, sheet_name="Info", index=False)
        publish(out_path, "portfolios")
        print(f"Nothing survived -- wrote placeholder to {out_path}")
        return out_path

    # Undominated first, so a row cap can only ever drop dominated portfolios.
    ordered = scored.sort_values(["undominated", "p_over_100"], ascending=[False, False])
    written = ordered.head(max_rows).reset_index(drop=True)
    leg_rows = _leg_rows(written, result["picks"], result["options"], max_leg_portfolios)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        book = writer.book
        for name in ("Contents", "Propositions", "Portfolios", "Query", "Legs", "Stake Calculator"):
            pd.DataFrame().to_excel(writer, sheet_name=name, index=False)

        n_costable = int(leg_rows["id"].nunique()) if len(leg_rows) else 0
        _write_contents(book["Contents"], result["qualified"], scored,
                        result["info"], len(written), n_costable, source)
        _write_propositions(book["Propositions"], result["qualified"])
        last = _write_portfolios(book["Portfolios"], written)
        _write_query(book["Query"], last, len(written))
        last_leg = _write_legs(book["Legs"], leg_rows)
        default_id = int(leg_rows["id"].iloc[0]) if len(leg_rows) else int(written["id"].iloc[0])
        _write_calculator(book["Stake Calculator"], written, last_leg, stake, default_id)

    publish(out_path, "portfolios")
    print(f"Wrote {len(written):,} portfolios ({int(written['undominated'].sum()):,} undominated), "
          f"legs for {leg_rows['id'].nunique() if len(leg_rows) else 0} -> {out_path}")
    return out_path


__all__ = ["write_portfolio_workbook", "COLUMNS", "COL_KEYS", "QUERY_ROWS",
           "MAX_ROWS", "MAX_LEG_PORTFOLIOS"]
