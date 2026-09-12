"""From model probabilities to a stake: edge, dominance, allocation.

The predictions workbook prices every proposition against the model alone. This
module is the step after that -- it takes a bookmaker's price for the same
proposition and asks whether the price is wrong in our favour, then decides how
much to put on the ones that are.

This module holds the per-proposition primitives. Choosing *which* propositions
to back together, and in what mix, is a portfolio question and lives in
`fpp.portfolio`.

Naming, deliberately strict
---------------------------
The fixture sheets already carry a column headed "Odds" holding ``1 / P`` -- the
model's own fair price, not a bookmaker's. Reusing that word here would make it
genuinely ambiguous which of the two any given variable held, so nothing in this
module is called `odds`:

===  =====================================================================
`p`  model probability, straight from the predictions workbook
`o`  bookmaker price in decimal, best of the two books quoted
`e`  edge, ``p * o``. Above 1 the price pays more than the model says it should
`s`  stake on one proposition
`S`  total stake across a portfolio
`F`  ``sum(1 / e)`` over a portfolio -- the normaliser for the stake split
===  =====================================================================

There is deliberately no `m`. A rank score of ``p * e`` used to order the
survivors and pick one per match; it was dropped because ``p * e = o * p**2``
squares the model probability and so ranks a near-certainty at a thin price
above a genuine edge at a fat one. Nothing replaces it at proposition level --
ordering propositions against each other is the wrong question, and
`fpp.portfolio` asks the right one instead.

Independence
------------
The outcome distribution treats propositions as independent. That is safe *by
construction* rather than by assumption: a portfolio admits at most one
proposition per match, and two propositions from different matches are genuinely
unrelated. If that one-per-match rule is ever relaxed, same-match propositions
(a team's goals line and its own corners line) are strongly correlated and the
variance and threshold figures here would understate the real spread. The rule
and the assumption stand or fall together, so they are documented together.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .spec import STAT_BY_KEY, STATS, TARGETS

# Floor on the model's own probability for a proposition to reach the odds form.
#
# Zero, deliberately. It was 0.50 back when the form was typed in by hand and the
# only defensible filter was "do not pay for a price you would never back". The
# form is now filled from Oddschecker by the `fill-odds` skill, so a row costs a
# lookup rather than a keystroke, and the filter was buying nothing while cutting
# out the whole low-probability half of every ladder -- exactly where a book's
# margin is widest and an edge, when there is one, is largest. A proposition the
# model rates at 0.20 against a price of 6.0 is a 1.20 edge, and under the old
# floor it was never even asked about.
#
# The real filter is `positive_edge` (`E >= 1`), which is a statement about the
# price rather than about the model, and it comes after the form is priced.
#
# Kept as a named dial rather than deleted: `qualifying` still takes it, so
# raising it is one number if the fill rate ever makes breadth expensive again.
MIN_MODEL_P = 0.0

# 2**20 paths is ~1M rows at ~21MB as int8 -- comfortable. Above it, sample.
#
# It binds routinely now. `portfolio` builds portfolios of up to one leg per
# event, and twenty-six events is 67 million paths for a single portfolio, so
# exact enumeration is the oracle rather than the engine -- see
# `threshold_probs` and `portfolio.threshold_probs_batch`.
MAX_EXACT_N = 20

# Floor and ceiling on `p` wherever a formula divides by `p * (1 - p)`. Small
# enough not to move any realistic price, large enough that a 0.999 proposition
# cannot swallow a whole allocation. See `min_variance_alloc`.
P_CLAMP = 1e-6
MONTE_CARLO_DRAWS = 2_000_000

# Return multiples the outcome distribution is reported at. Symmetric about a
# break-even 1.00 so the table answers "how bad" and "how good" with the same
# resolution.
#
# Trimmed from five to three. 0.80 and 1.20 were dropped because they earned no
# decisions: every column here is a filter axis in the Edge Book and a dominance
# candidate, and the two extremes moved together with 0.90 and 1.10 closely
# enough to add width without adding information. Three rungs keep the shape --
# a loss, break-even, a win -- with a rail you can actually read.
THRESHOLDS = (0.90, 1.00, 1.10)

# The books quoted on the form, column key -> the name a human reads. One
# definition: the form writes these headers, the form reader parses them back, and
# `best_price` reports which of them won. Adding a book is a line here.
#
# Virgin Bet was dropped on 2026-09-13. Its feed on the comparison page this is
# filled from had been frozen since 1 September -- every `VE` price on a match
# carried one identical timestamp eleven days stale, while the live books
# re-quoted hourly -- and the site had already removed it from its own bookmaker
# catalogue, so nothing a visitor sees was ever showing those prices. Stale
# prices are worse than absent ones here: `best_price` takes the row max, so a
# frozen quote wins rows it should not and invents the edge on them. On the
# 2026-09-12 slate it was the sole best price on 23 of 762 priced rows, leading
# by a median of 5% and by as much as 20%.
#
# Betfair Exchange replaces it, and is deliberately the *first* key here. It had
# been dropped once on coverage -- 3 of 122 sampled propositions, none of them
# shots or corners -- and that is still the expectation: it will price the goals
# markets and little else. It earns its column on a different axis. An exchange
# does not restrict or close a winning account, so where it matches the best
# price it is the one to take, and this order is what says so: it is the column
# order on the form, the order `best_price` names joint winners in, and the head
# of the Edge Book's house order.
#
# The keys are the column names the whole pipeline uses. Nothing downstream of
# `best_price` sees a book column at all, so this set can change without
# touching `06_Split`.
BOOKS: dict[str, str] = {
    "betfair": "Betfair Exchange",
    "b365": "Bet365",
    "paddypower": "Paddy Power",
    "tenbet": "10bet",
    "boylesports": "BoyleSports",
    "betmgm": "BetMGM",
}
BOOK_COLUMNS: tuple[str, ...] = tuple(BOOKS)

# A cell left blank, or explicitly zeroed, means the book is not pricing that line.
NOT_OFFERED = 0.0


# --- Building the proposition table ---------------------------------------


def proposition_label(target: str, team: str, line: float) -> str:
    """``"Corners - Coventry City - Over 2.5"`` -- a proposition's name, once.

    This string is the join key between every file the pipeline writes: the
    odds form is keyed on it, `read_filled` reads it back, and the Edge Book
    export matches a fixture's model lines to the prices captured against them.
    Nothing carries a numeric id across those boundaries, so a second copy of
    this format that drifted by a space would not fail -- it would silently
    match nothing, and a fixture would quietly show no prices at all.
    """
    return f"{STAT_BY_KEY[target].display} - {team} - Over {line}"


# The inverse of `proposition_label`, and it lives beside it deliberately: the two
# are a pair, and a parser that drifted from its builder would not fail, it would
# match nothing. `ledger` re-exports this under its old name.
_TARGET_BY_DISPLAY: dict[str, str] = {s.display: s.key for s in STATS}
_LABEL = re.compile(r"^(?P<head>.+) - Over (?P<line>-?\d+(?:\.\d+)?)$")


def parse_label(label: str) -> tuple[str, str, float]:
    """``"Corners - Parma - Over 1.5"`` -> ``("corners", "Parma", 1.5)``.

    Split from the right on `" - Over "` and only then on the first `" - "`: a
    club whose name contains a hyphen-space is a real possibility, and none of
    the four display names does.
    """
    m = _LABEL.match(str(label).strip())
    if not m:
        raise ValueError(f"cannot parse proposition label {label!r}")
    head, line = m.group("head"), float(m.group("line"))
    display, _, team = head.partition(" - ")
    if not team:
        raise ValueError(f"cannot parse proposition label {label!r}")
    if display not in _TARGET_BY_DISPLAY:
        raise ValueError(f"unknown market {display!r} in {label!r}")
    return _TARGET_BY_DISPLAY[display], team, line


def label_parts(labels: pd.Series) -> pd.DataFrame:
    """``(target, team, line)`` per label, as columns, ``NA`` where unparseable.

    The tolerant, vectorised companion to `parse_label`. What a proposition is
    *about* is only recoverable from its name -- the odds form carries no separate
    columns for it -- and pairing two propositions needs to know whether they share
    a team or a market.

    Tolerant rather than strict because a label this cannot read is a proposition
    that simply cannot be paired, which is a normal outcome and not an error. The
    strict version stays for `ledger`, where an unreadable label *is* a bug: it
    means a result cannot be joined back to the bet that produced it.
    """
    out = {"target": [], "team": [], "line": []}
    for raw in labels:
        try:
            target, team, line = parse_label(raw)
        except (ValueError, TypeError):
            target, team, line = pd.NA, pd.NA, np.nan
        out["target"].append(target)
        out["team"].append(team)
        out["line"].append(line)
    return pd.DataFrame(out, index=labels.index)


def propositions(preds: pd.DataFrame, dispersion: dict | None = None) -> pd.DataFrame:
    """Every priced proposition across every fixture, one per row.

    Built by calling the same `fixture_markets` the workbook writer calls and
    taking its lines from the same `ladder_lines`, so these rows *are* the rows
    the fixture sheets show -- 50 per match -- rather than a second derivation
    that happens to agree today.
    """
    # All three imported here rather than at module scope, to keep `fpp.report`
    # off the import cycle: `report.odds_form` imports `BOOKS` from this module,
    # so a top-level `from .report... import` makes which of the two loads first
    # decide whether the package imports at all. It held together only while
    # `fpp/__init__` happened to list `report` before `staking`, and adding
    # `fpp.portfolio` -- which imports this module and sorts before `report` --
    # was enough to break it.
    from .report.excel import sheet_codes
    from .report.markets import fixture_markets, ladder_lines

    codes = sheet_codes(preds)
    rows = []
    for code, (_, row) in zip(codes, preds.iterrows()):
        mk = fixture_markets(row, dispersion)
        home, away = row["home_team"], row["away_team"]
        for target in TARGETS:
            home_lines, away_lines = ladder_lines(target, mk[target])
            for scope, team, lines in (("home", home, home_lines), ("away", away, away_lines)):
                for line in lines:
                    rows.append({
                        "sheet_code": code,
                        "date": row["date"],
                        "league": row["league"],
                        "home_team": home,
                        "away_team": away,
                        "fixture": f"{home} vs {away}",
                        "target": target,
                        "team": team,
                        "scope": scope,
                        "line": float(line),
                        "label": proposition_label(target, team, line),
                        "p": float(mk[target][f"{scope}_over_{line}"]),
                    })
    return pd.DataFrame(rows)


def qualifying(props: pd.DataFrame, min_p: float = MIN_MODEL_P) -> pd.DataFrame:
    """The propositions worth asking for a price on, in the order you type them.

    Sorted **fixture, then target, then team, then line ascending** -- the shape
    of a bookmaker's own page. Probability order reads better on screen but is
    miserable to enter from: it scatters one team's ladder down the sheet, so
    every price means finding the row again. Grouping a team's whole ladder
    together and running it upward turns entry into typing down a column.

    Target order comes from `spec.TARGETS` rather than a literal, so it stays the
    order the fixture sheets use; team order is home then away, matching those
    sheets' left and right columns. Fixtures keep the predictions workbook's own
    order -- the two files are meant to be read side by side with the same tab in
    the same place, and sorting by sheet code would silently reorder them.

    At the default `min_p` of zero every ladder arrives whole, so a team's block
    runs from its lowest line to its highest with nothing missing. Raise `min_p`
    and the ladders become **discontiguous** -- the filter takes lines off the
    top, so a team can have shots Over 10.5 through 12.5 and nothing above. The
    order stays predictable either way; only the holes are new.
    """
    out = props[props["p"] >= min_p].copy()
    fixture = {code: i for i, code in enumerate(props["sheet_code"].drop_duplicates())}
    target = {t: i for i, t in enumerate(TARGETS)}
    out["_fixture"] = out["sheet_code"].map(fixture)
    out["_target"] = out["target"].map(target)
    out["_side"] = (out["scope"] == "away").astype(int)
    out = out.sort_values(["_fixture", "_target", "_side", "line"])
    return out.drop(columns=["_fixture", "_target", "_side"]).reset_index(drop=True)


# --- Price, edge, dominance -----------------------------------------------


def best_price(df: pd.DataFrame, books: tuple[str, ...] = BOOK_COLUMNS) -> pd.DataFrame:
    """``o`` = the best price quoted, and ``book`` = who quoted it.

    An unpriced cell -- blank, or an explicit 0 -- means the book is not offering
    that line. Those are skipped in the max rather than treated as a number, so
    only a proposition unpriced by *every* book disappears.

    The winning book travels with the price because the output is a list of bets
    to go and place: an edge is worthless if you have to come back here to find
    out where it was available. On an exact tie every winner pays the same, so
    none of them is "the" source and the row names all of them -- "Bet365 / Sky
    Bet" rather than a bare "Both", which stopped being answerable the moment a
    third book joined the form.
    """
    out = df.copy()
    # A form filled before a book was added to `BOOKS` simply has no column for
    # it. Taking the max over the books that are actually there beats a bare
    # KeyError, but it must say so: silently pricing off four books when you
    # think it is six is the kind of wrong that looks right.
    present = [b for b in books if b in out.columns]
    if not present:
        raise ValueError(f"none of the book columns {list(books)} are in this frame; "
                         f"it has {list(out.columns)}")
    if len(present) < len(books):
        missing = [BOOKS.get(b, b) for b in books if b not in present]
        print(f"note: no column for {', '.join(missing)} -- pricing off "
              f"{len(present)} of {len(books)} books")
    prices = out[present]
    out["o"] = prices.max(axis=1, skipna=True)
    is_best = prices.eq(out["o"], axis=0) & prices.notna()
    out["book"] = [
        " / ".join(BOOKS.get(b, b) for b in row.index[row]) if row.any() else None
        for _, row in is_best.iterrows()
    ]
    return out[out["o"].notna()].reset_index(drop=True)


def add_edge(df: pd.DataFrame) -> pd.DataFrame:
    """``e = p * o`` -- the only derived quantity a single proposition gets.

    The companion ``m = p * e`` is gone. See the module docstring: it is
    ``o * p**2``, so it rewards confidence twice and price once, and it was
    deciding which proposition each match contributed.
    """
    out = df.copy()
    out["e"] = out["p"] * out["o"]
    return out


def positive_edge(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only prices that pay more than the model says they should.

    A match can lose every proposition here. That is a normal outcome, not an
    error: it contributes nothing downstream and gets no sheet.
    """
    return df[df["e"] >= 1.0].reset_index(drop=True)


def undominated(df: pd.DataFrame, axes: tuple[str, str] = ("p", "o")) -> pd.DataFrame:
    """The Pareto frontier on ``axes``, by default ``(p, o)``.

    A proposition is dominated when another beats or matches it on *both* axes
    and strictly beats it on at least one. Two propositions tied exactly on both
    dominate neither -- so both survive, rather than an arbitrary one being
    dropped by comparison order.

    Why ``(p, o)`` and not ``(p, e)``
    ---------------------------------
    ``e`` is ``p * o``, so a ``(p, e)`` frontier compares one axis against a
    product that already contains it: a proposition can lose on "edge" purely
    because its probability is lower, having already lost on probability. The
    honest pair is the two things that actually vary independently of each other
    -- what the model thinks, and what the book will pay. On those, "worse on
    both" is a statement no reallocation of stake can rescue.

    Expect it to prune far less. ``p`` and ``o`` are close to inverses of each
    other by construction, so they correlate strongly *negatively* -- measured at
    -0.87 on a real filled form -- and mutual dominance is correspondingly rare.
    On that form ``(p, e)`` cut 57 propositions to 42 and ``(p, o)`` cut them to
    53. That is the rule working, not failing: the survivors are genuinely
    incomparable, and choosing between them is the portfolio search's job.

    The comparison runs on raw float64. `p` arrives with a dozen significant
    digits, so propositions that look identical at the two decimal places a sheet
    displays are almost never actually tied; rounding first would invent ties
    that the rule then has to treat as real.
    """
    if df.empty:
        return df.reset_index(drop=True)
    x = df[axes[0]].to_numpy(dtype=float)
    y = df[axes[1]].to_numpy(dtype=float)
    ge = (x[:, None] >= x[None, :]) & (y[:, None] >= y[None, :])
    gt = (x[:, None] > x[None, :]) | (y[:, None] > y[None, :])
    dominated = (ge & gt).any(axis=0)  # column j dominated by some row i
    return df[~dominated].reset_index(drop=True)


# --- Stakes ----------------------------------------------------------------


def stake_split(e: np.ndarray, total: float = 1.0) -> np.ndarray:
    """``s_i = S / (e_i * F)`` with ``F = sum(1 / e)``.

    Equalises every proposition's *expected* contribution: ``s_i * e_i`` is the
    same for all `i`, which is why the expected return collapses to the closed
    form ``n * S / F`` and needs no enumeration.
    """
    inv = 1.0 / np.asarray(e, dtype=float)
    return float(total) * inv / inv.sum()


def expected_return(p, o, stakes) -> float:
    """``sum(s_i * o_i * p_i)``. Exact, no enumeration."""
    return float(np.sum(np.asarray(stakes) * np.asarray(o) * np.asarray(p)))


def variance(p, o, stakes) -> float:
    """``sum(s_i^2 * o_i^2 * p_i * (1 - p_i))``.

    Outcomes are independent, so the covariance terms vanish and the whole
    variance is this one sum -- which is also what gives the minimum-variance
    allocation a closed form rather than needing an optimiser.
    """
    p = np.asarray(p, dtype=float)
    o = np.asarray(o, dtype=float)
    s = np.asarray(stakes, dtype=float)
    return float(np.sum(s**2 * o**2 * p * (1.0 - p)))


def outcome_paths(p, o, stakes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every ``2**n`` hit/miss combination: (bits, path probability, return)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n > MAX_EXACT_N:
        raise ValueError(f"{n} propositions is {2**n:.3g} paths; above MAX_EXACT_N={MAX_EXACT_N}")
    bits = ((np.arange(2**n, dtype=np.int64)[:, None] >> np.arange(n)) & 1).astype(np.int8)
    probs = np.prod(np.where(bits == 1, p, 1.0 - p), axis=1)
    returns = bits @ (np.asarray(stakes, dtype=float) * np.asarray(o, dtype=float))
    return bits, probs, returns


def threshold_probs(p, o, stakes, total: float = 1.0,
                    thresholds=THRESHOLDS, rng_seed: int = 0) -> dict[float, float]:
    """``P(return > t * S)`` for each `t`, exact where it can be.

    Exact enumeration up to `MAX_EXACT_N`; a fixed-seed Monte Carlo above it, so
    a list long enough to need sampling still produces the same numbers twice.

    **This is the oracle, not the production engine.** It costs ``2**n`` paths,
    which is why `MAX_EXACT_N` exists, and a portfolio search scores hundreds of
    thousands of portfolios of twenty-odd legs each -- so production uses
    `portfolio.threshold_probs_batch`, a grid convolution. This function is what
    that one is *pinned against* in `tests/test_portfolio.py`, and it is worth
    keeping exactly because it is the slow, obviously-correct version. Do not
    wire it back into a reporting path: two engines answering the same question
    to different precision would put two different numbers for one portfolio on
    two sheets of the same workbook.
    """
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return {t: 0.0 for t in thresholds}
    pay = np.asarray(stakes, dtype=float) * np.asarray(o, dtype=float)
    if n <= MAX_EXACT_N:
        _, probs, returns = outcome_paths(p, o, stakes)
        return {t: float(probs[returns > t * total].sum()) for t in thresholds}
    rng = np.random.default_rng(rng_seed)
    hits = rng.random((MONTE_CARLO_DRAWS, n)) < p
    returns = hits @ pay
    return {t: float((returns > t * total).mean()) for t in thresholds}


def summarise(p, o, stakes, total: float = 1.0, thresholds=THRESHOLDS) -> dict:
    """Everything the calculator reports, for one allocation.

    Percentages are the honest unit here: every stake is a fixed fraction of `S`,
    so the return distribution scales linearly with it. `pct_*` therefore do not
    move when `S` changes -- only the cash figures do, by a plain multiply.
    """
    er = expected_return(p, o, stakes)
    var = variance(p, o, stakes)
    tp = threshold_probs(p, o, stakes, total, thresholds)
    return {
        "n": int(len(np.asarray(p))),
        "expected_return": er,
        "pct_expected_return": er / total if total else float("nan"),
        "variance": var,
        "sd": float(np.sqrt(var)),
        "pct_sd": float(np.sqrt(var)) / total if total else float("nan"),
        "thresholds": tp,
        "profitability": tp.get(1.00, float("nan")),
        "estimated": bool(len(np.asarray(p)) > MAX_EXACT_N),
    }


# --- Allocation ------------------------------------------------------------
#
# Two ways to divide one total across a chosen set of propositions, both closed
# form: `stake_split` above equalises expected contribution, `min_variance_alloc`
# below minimises spread. They are the two the portfolio search scores.
#
# Three others used to live here -- max expected return, and two
# `differential_evolution` searches for the allocation maximising
# `P(return > t)`. They are gone, and the reason is worth recording. They all
# optimised the *stakes* over a set of propositions that had already been chosen,
# which is the small half of the problem: the spread of outcomes available from
# reallocating a fixed list is far narrower than the spread available from
# picking a different list. `fpp.portfolio` searches both together, so a
# stake-only optimiser is now strictly dominated by the thing that replaced it --
# and the threshold searches were the slowest code in the pipeline besides.
#
# (`P(return > t)` is **piecewise constant** in the stakes, so those searches had
# to be derivative-free: handed a zero gradient, `scipy.optimize.minimize`
# returns its starting point and reports `success=True`. That trap is worth
# remembering if anyone reaches for a stake optimiser again.)


def min_variance_alloc(p, o, total: float = 1.0) -> np.ndarray:
    """``s_i ~ 1 / (o_i^2 * p_i * (1 - p_i))``.

    Independence makes the variance a sum of squares with no cross terms, so
    minimising it over the simplex is a positive-definite quadratic with an
    interior solution in closed form -- unique, interior, and genuinely
    diversified, which is why it is one of the two splits the portfolio search
    scores.

    ``p`` is clamped off both endpoints. The weight is the reciprocal of a
    proposition's own variance contribution, and that variance goes to zero as
    ``p`` approaches 0 or 1, so the weight diverges: a near-certainty would take
    the entire stake and every other weight would round to nothing. This did not
    arise while the odds form was filtered at ``p >= 0.50`` and the top of a
    ladder was never priced; with `MIN_MODEL_P` at zero, ``p`` of 0.99 is an
    ordinary row. The clamp caps one proposition's share rather than changing any
    answer in the range where the formula is well behaved.
    """
    p = np.clip(np.asarray(p, dtype=float), P_CLAMP, 1.0 - P_CLAMP)
    o = np.asarray(o, dtype=float)
    w = 1.0 / (o**2 * p * (1.0 - p))
    return float(total) * w / w.sum()
