"""The odds capture workbook: a blank form for hand-entering bookmaker prices.

`05_Run` writes this next to the predictions workbook, with the same sheet names,
so the two line up one to one. Each fixture sheet lists every proposition the
fixture is priced at -- `staking.MIN_MODEL_P` is zero, so that is the whole
ladder -- and leaves one blank column for a price from each book in
`staking.BOOKS`.

Why the model's probability is written into the form
----------------------------------------------------
Column B carries `P`, which looks redundant next to a predictions workbook that
already holds it. It is what makes the filled form **self-contained**: `06_Split`
reads this one file and nothing else. The alternative -- a bare form, with `P`
recovered afterwards by matching a label like
``"Shots on Target - Atletico Madrid - Over 1.5"`` back to a row of the
predictions workbook -- puts a string join on the critical path, where a renamed
team or a re-run that shifts a dynamic ladder breaks the pairing silently and
mis-prices everything downstream.

The precision survives the trip. openpyxl stores the full float64 in the sheet
XML and `number_format` only changes what is drawn on screen, so a cell shown as
`90.00%` still reads back as 0.899757476307095 -- which matters because the
dominance rule in `staking.undominated` compares probabilities exactly.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
from openpyxl.styles import Font, Protection
from openpyxl.worksheet.datavalidation import DataValidation

from ..staking import BOOKS, NOT_OFFERED
from .style import BORDER, CENTER, FILL_CARD, FILL_HEADER, FILL_TITLE, FONT_H, FONT_TITLE, LEFT, ODDS, PCT, plain_cell

BOOK_COLUMNS = tuple(BOOKS)
BOOK_HEADERS = tuple(BOOKS.values())

HEADER_ROW = 2
FIRST_DATA_ROW = 3

# Column A is the proposition and B the model's P, so the books start at C and the
# sheet is two columns wider than the number of books. Derived rather than typed
# so that adding a book to `staking.BOOKS` widens the title merge, the validated
# range and the column widths together -- they drifted apart the last time this
# was three literals.
FIRST_BOOK_COL = 3
LAST_COL = 2 + len(BOOK_COLUMNS)
_COL_LETTERS = "".join(chr(ord("A") + i) for i in range(LAST_COL))
LAST_COL_LETTER = _COL_LETTERS[-1]

# Decimal odds are a multiplier on the stake and so are always above 1.00 -- a
# "1.85" typed as "0.85" or "185" is the kind of slip that silently poisons every
# edge downstream, and it is far cheaper to refuse it at entry than to reason
# backwards from a strange final list.
#
# Zero is the deliberate exception: books do not price every line, and typing 0 is
# quicker than tabbing past a cell. It means the same as blank. The gap between
# them -- anything in (0, 1.00] -- stays refused, because that range is only ever
# reachable by mistake.
_MIN_DECIMAL_PRICE = 1.0


def _fixture_index(props: pd.DataFrame) -> pd.DataFrame:
    cols = ["sheet_code", "date", "league", "home_team", "away_team"]
    idx = props[cols].drop_duplicates("sheet_code").reset_index(drop=True)
    counts = props.groupby("sheet_code", sort=False).size().rename("n_props")
    return idx.join(counts, on="sheet_code")


def _write_contents(ws, index: pd.DataFrame, source: str | None) -> None:
    ws.merge_cells("A1:F1")
    t = ws.cell(1, 1)
    t.value = (f"Odds entry - {len(index)} fixtures, "
               f"{int(index['n_props'].sum())} propositions to price")
    t.fill = FILL_TITLE
    t.font = FONT_TITLE
    t.alignment = LEFT

    ws.cell(2, 1).value = "Predictions source"
    ws.cell(2, 1).font = FONT_H
    ws.cell(2, 2).value = source or "(not recorded)"
    ws.cell(3, 1).value = "Generated"
    ws.cell(3, 1).font = FONT_H
    ws.cell(3, 2).value = dt.date.today().isoformat()

    head = 5
    for j, h in enumerate(["Sheet", "Date", "League", "Home", "Away", "To price"]):
        c = ws.cell(head, 1 + j)
        c.value = h
        c.fill = FILL_HEADER
        c.font = FONT_H
        c.alignment = CENTER
        c.border = BORDER

    for i, row in enumerate(index.to_dict("records")):
        r = head + 1 + i
        link = ws.cell(r, 1)
        link.value = row["sheet_code"]
        link.hyperlink = f"#'{row['sheet_code']}'!A1"
        link.font = Font(color="0563C1", underline="single")
        link.alignment = CENTER
        link.border = BORDER
        for j, v in enumerate([
            str(pd.Timestamp(row["date"]).date()), row["league"],
            row["home_team"], row["away_team"], int(row["n_props"]),
        ]):
            plain_cell(ws, r, 2 + j, v)

    for col, w in zip("ABCDEF", (12, 12, 18, 24, 24, 10)):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "A6"


def _write_fixture(ws, code: str, rows: pd.DataFrame) -> None:
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=LAST_COL)
    t = ws.cell(1, 1)
    r0 = rows.iloc[0]
    t.value = (f"{r0['home_team']} vs {r0['away_team']}  -  "
               f"{pd.Timestamp(r0['date']).date()}  -  {r0['league']}  ({code})")
    t.fill = FILL_TITLE
    t.font = FONT_TITLE
    t.alignment = LEFT

    for j, h in enumerate(["Proposition", "Model P", *BOOK_HEADERS]):
        c = ws.cell(HEADER_ROW, 1 + j)
        c.value = h
        c.fill = FILL_HEADER
        c.font = FONT_H
        c.alignment = CENTER
        c.border = BORDER

    for i, rec in enumerate(rows.to_dict("records")):
        r = FIRST_DATA_ROW + i
        lab = plain_cell(ws, r, 1, rec["label"], fill=FILL_CARD, bold=True)
        lab.alignment = LEFT
        # Full float64 goes in; PCT only changes how it is drawn.
        ref = plain_cell(ws, r, 2, float(rec["p"]), PCT, fill=FILL_CARD)
        ref.protection = Protection(locked=True)
        for j in range(len(BOOK_COLUMNS)):
            cell = plain_cell(ws, r, FIRST_BOOK_COL + j, None, ODDS)
            cell.protection = Protection(locked=False)

    last = FIRST_DATA_ROW + len(rows) - 1
    first = f"C{FIRST_DATA_ROW}"
    dv = DataValidation(
        # Excel applies a custom formula relatively across the range, so writing it
        # against the top-left cell covers every cell in it.
        type="custom", formula1=f"=OR({first}={NOT_OFFERED:g},{first}>{_MIN_DECIMAL_PRICE:g})",
        allow_blank=True, showErrorMessage=True,
        errorTitle="Not a decimal price",
        error=("Decimal odds are always above 1.00. Enter 1.85, not 0.85 or 185.\n\n"
               "Leave blank or enter 0 if the book is not pricing this line."),
    )
    ws.add_data_validation(dv)
    dv.add(f"C{FIRST_DATA_ROW}:{LAST_COL_LETTER}{last}")

    # A guard rail, not security: no password, so it is one click to lift. It
    # exists so the reference column is not edited by accident while tabbing
    # through a few hundred price cells.
    ws.protection.sheet = True
    ws.protection.formatColumns = False
    ws.protection.formatRows = False
    ws.protection.sort = False

    for col, w in zip(_COL_LETTERS, (46, 12) + (12,) * len(BOOK_COLUMNS)):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "A3"


def write_odds_form(props: pd.DataFrame, out_path: Path | None = None,
                    source: str | None = None) -> Path:
    """Write the blank capture workbook. ``props`` is already filtered and sorted.

    Filtering lives in `staking.qualifying` rather than here, so the threshold is
    one named constant in one place and this function stays a renderer.
    """
    from ..paths import OUTPUTS, publish

    out_path = Path(out_path) if out_path else OUTPUTS / f"odds_input_{dt.date.today().isoformat()}.xlsx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if props.empty:
        with pd.ExcelWriter(out_path, engine="openpyxl") as w:
            pd.DataFrame({"Message": ["No propositions cleared the model threshold."]}).to_excel(
                w, sheet_name="Info", index=False)
        publish(out_path, "odds")
        print(f"No qualifying propositions -- wrote placeholder to {out_path}")
        return out_path

    index = _fixture_index(props)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pd.DataFrame().to_excel(writer, sheet_name="Contents", index=False)
        _write_contents(writer.book["Contents"], index, source)
        for code, rows in props.groupby("sheet_code", sort=False):
            pd.DataFrame().to_excel(writer, sheet_name=code, index=False)
            _write_fixture(writer.book[code], code, rows)

    publish(out_path, "odds")
    print(f"Wrote {len(props)} propositions across {len(index)} fixtures -> {out_path}")
    return out_path


def form_meta(path: Path) -> dict:
    """What a filled form says about its own origin: source, date, size.

    `06_Split` reads the form and nothing else -- deliberately, so a renamed team
    or a shifted ladder cannot mis-pair a probability. The cost of that isolation
    is that the form is also the *only* thing that can say which predictions it
    came from, so it has to be asked.
    """
    import openpyxl

    wb = openpyxl.load_workbook(Path(path), read_only=True)
    cs = wb["Contents"] if "Contents" in wb.sheetnames else None
    fixtures = [s for s in wb.sheetnames if s not in ("Contents", "Info")]
    return {
        "path": Path(path),
        "source": (cs.cell(2, 2).value if cs else None),
        "generated": (str(cs.cell(3, 2).value)[:10] if cs and cs.cell(3, 2).value else None),
        "fixtures": len(fixtures),
    }


def latest_form(outputs: Path | None = None) -> Path | None:
    """The newest ``odds_input_<date>.xlsx`` on disk, by the date in its name.

    Sorted on the date rather than on mtime: filling a form rewrites it, so an
    older form topped up this afternoon has a newer mtime than a form generated
    this morning, and mtime would pick the wrong one.
    """
    from ..paths import OUTPUTS

    outputs = Path(outputs) if outputs else OUTPUTS
    forms = sorted(outputs.glob("odds_input_*.xlsx"),
                   key=lambda p: p.stem.replace("odds_input_", ""))
    return forms[-1] if forms else None


def check_form_is_current(path: Path, *, strict: bool = True) -> dict:
    """Refuse a form that is not the newest one on disk.

    This exists because it happened. `06_Split`'s config cell carried a hardcoded
    ``odds_input_2026-08-19.xlsx`` and was run two days later against a freshly
    retuned model; it consumed the old form without a murmur and produced a
    workbook that was internally consistent, plausible, and entirely stale --
    fixtures that were no longer on, priced by a model that no longer existed.
    Nothing downstream could have noticed, because the form carries its own
    probabilities and there is nothing else to disagree with them.

    A stale form is not a warning-level event. It silently answers a question
    about last week, so ``strict`` raises by default.
    """
    meta = form_meta(path)
    newest = latest_form(Path(path).parent)
    meta["is_newest"] = newest is None or Path(path).resolve() == newest.resolve()
    meta["newest"] = newest

    if not meta["is_newest"]:
        msg = (f"{Path(path).name} is not the newest odds form -- {newest.name} exists. "
               f"It was generated {meta['generated']} from {meta['source']}, so its "
               f"probabilities are that model's, not the one frozen now. "
               f"Set ODDS_FILE deliberately if you really mean to re-price an old form.")
        if strict:
            raise ValueError(msg)
        print(f"  WARNING: {msg}")
    return meta


def read_filled(path: Path) -> pd.DataFrame:
    """Read a filled form back: ``sheet_code, label, p`` + one column per book + fixture info.

    A 0 reads back as "not offered", exactly like a blank cell.

    Anything else at or below 1.00 raises rather than being dropped. The form's
    validation refuses those at entry, but validation does not apply to a pasted
    block, and a price of 0.85 would otherwise sail through as a negative edge and
    simply vanish at the `e >= 1` step -- indistinguishable from a proposition
    that was correctly priced and correctly rejected.
    """
    import openpyxl

    wb = openpyxl.load_workbook(Path(path), data_only=True)
    meta = {}
    if "Contents" in wb.sheetnames:
        cs = wb["Contents"]
        for r in range(6, cs.max_row + 1):
            code = cs.cell(r, 1).value
            if code:
                meta[str(code)] = {
                    "date": cs.cell(r, 2).value, "league": cs.cell(r, 3).value,
                    "home_team": cs.cell(r, 4).value, "away_team": cs.cell(r, 5).value,
                }

    rows, blank_sheets, bad = [], [], []
    for code in [s for s in wb.sheetnames if s not in ("Contents", "Info")]:
        ws = wb[code]
        priced = 0
        for r in range(FIRST_DATA_ROW, ws.max_row + 1):
            label = ws.cell(r, 1).value
            if not label:
                continue
            prices = []
            for j in range(len(BOOK_COLUMNS)):
                v = ws.cell(r, FIRST_BOOK_COL + j).value
                v = float(v) if isinstance(v, (int, float)) else None
                if v is not None and v == NOT_OFFERED:
                    v = None  # "not pricing this line" -- identical to a blank
                elif v is not None and v <= _MIN_DECIMAL_PRICE:
                    bad.append(f"{code}!{_COL_LETTERS[FIRST_BOOK_COL - 1 + j]}{r} = {v}")
                prices.append(v)
            priced += any(v is not None for v in prices)
            rows.append({
                "sheet_code": code, "label": str(label),
                "p": float(ws.cell(r, 2).value),
                **dict(zip(BOOK_COLUMNS, prices)),
                **meta.get(code, {}),
            })
        if priced == 0:
            blank_sheets.append(code)

    if bad:
        raise ValueError(
            "decimal odds must be above 1.00 (use 0 or blank for a line the book "
            "is not pricing); found "
            + ", ".join(bad[:10]) + (f" (+{len(bad) - 10} more)" if len(bad) > 10 else "")
        )
    if blank_sheets:
        print(f"note: {len(blank_sheets)} fixture(s) left entirely unpriced and will be skipped: "
              + ", ".join(blank_sheets[:8]) + (" ..." if len(blank_sheets) > 8 else ""))

    out = pd.DataFrame(rows)
    # Force the price columns to float. Unpriced cells arrive as None, which would
    # otherwise leave an object-dtype column the moment a sheet is entirely
    # unpriced -- and object columns do not do arithmetic the way the rest of the
    # pipeline assumes. One coercion here keeps "missing" a real NaN throughout.
    for col in BOOK_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    meta = form_meta(path)
    print(f"Read {len(out)} propositions from {Path(path).name} "
          f"(generated {meta['generated']} from {meta['source']}); "
          f"{int(out[list(BOOK_COLUMNS)].notna().any(axis=1).sum())} priced")
    return out


__all__ = ["write_odds_form", "read_filled", "form_meta", "latest_form",
           "check_form_is_current", "BOOK_COLUMNS", "BOOK_HEADERS"]
