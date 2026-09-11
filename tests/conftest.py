"""Shared fixtures.

`clean_table` is the only one that reaches outside the repo, and it skips rather
than fails when the cache is not there. The tests that use it assert facts about
the real dataset -- that the Premier League promotes three clubs a season, that a
feature is not constant -- which is exactly the class of bug a synthetic fixture
cannot catch, because a synthetic fixture is built by the same understanding that
wrote the code. So they are worth having, and they are worth skipping cleanly on
a machine that has never run `01_Cleaning`.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def clean_table():
    """The cached team-match table, or a skip if no build has happened here."""
    from fpp.clean import load_clean_table

    try:
        return load_clean_table()
    except FileNotFoundError:
        pytest.skip("no cached clean table -- run fpp.clean.build_clean_table() first")
