"""European competitions, priced from the league model. An experiment.

League -> UEFA is a one-way street. Everything here may *read* the league model --
its priors, its frozen artifacts, its clean table -- and nothing in the league
model may read anything here. `tests/test_euro_isolation.py` enforces both halves:
no module outside this package imports it, and the league model's feature tables
hash to the fingerprint captured on `main`. Deleting this package leaves the
league model exactly as it was.

Scope: Champions League, Europa League and Conference League fixtures in which
**both** clubs play in one of the five modelled leagues that season. Those are
the only clubs with domestic priors, and the only fixtures priced.
"""
