"""European competitions and where their data lives.

A registry of its own rather than rows in `config.LEAGUES`, for the reason the
`config.LowerLeague` comment gives and one more: every consumer of `LEAGUES`
treats it as "leagues we pull Understat for", and the league model must not see
these at all. The slugs are likewise kept out of `config.SLUG_TO_KEY`, which is
what `ingest.scoreboard.parse_cache` filters the domestic cache on.

Data lives under its own roots for the same reason: a European scoreboard file in
`SOCCERDATA_CACHE` would be one more thing for the domestic pipeline to step
around, and one mistake from leaking into it.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..paths import ARTIFACTS, INPUTS, OUTPUTS


@dataclass(frozen=True)
class EuroComp:
    key: str          # our key, and the `league_key` a European fixture carries
    espn_slug: str
    prefix: str       # sheet / ledger code, the counterpart of `League.prefix`
    display: str
    first_season: int  # first season (start year) the competition existed


EURO_COMPS: dict[str, EuroComp] = {
    "UCL":  EuroComp("UCL",  "uefa.champions",   "UCL",  "Champions League",  2014),
    "UEL":  EuroComp("UEL",  "uefa.europa",      "UEL",  "Europa League",     2014),
    "UECL": EuroComp("UECL", "uefa.europa.conf", "UECL", "Conference League", 2021),
}
EURO_SLUG_TO_KEY = {c.espn_slug: c.key for c in EURO_COMPS.values()}

# --- Paths ------------------------------------------------------------------
EURO_INPUTS = INPUTS / "Euro"
EURO_CACHE = EURO_INPUTS / "scoreboard_cache"
EURO_MATCH_STATS = EURO_INPUTS / "euro_match_stats.csv"
EURO_ARTIFACTS = ARTIFACTS / "euro"
EURO_OUTPUTS = OUTPUTS / "Euro"
