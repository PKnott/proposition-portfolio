"""Pooled multi-market football prediction pipeline.

Four independent target families -- goals, shots, shots on target, corners --
over one shared walk-forward feature pipeline, pooled across five leagues.

Typical use::

    import fpp
    tm  = fpp.clean.build_clean_table()
    ctx = fpp.RunContext.load()
    models = fpp.predict.train_production(tm, ctx)
    fixtures = fpp.predict.load_upcoming_fixtures("2026-05-20", "2026-05-24")
    preds = fpp.predict.score_fixtures(tm, fixtures, models)
    fpp.report.write_workbook(preds, tm)
"""

from __future__ import annotations

import os

# The python.org 3.13 build ships without a populated OpenSSL cert store, so
# stdlib HTTPS fails until "Install Certificates.command" is run (which needs
# admin rights). requests uses certifi regardless; pointing the stdlib at the
# same bundle makes urllib work too, with no privileged install.
try:  # pragma: no cover
    import certifi as _certifi

    os.environ.setdefault("SSL_CERT_FILE", _certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", _certifi.where())
except Exception:  # pragma: no cover
    pass

from . import (
    analysis,
    artifacts,
    build,
    calibration,
    clean,
    config,
    cv,
    evaluate,
    ingest,
    ledger,
    metrics,
    models,
    paths,
    portfolio,
    predict,
    priors,
    reconcile,
    report,
    search,
    spec,
    staking,
)
from .artifacts import RunContext
from .build import FeatureTable, build_feature_table
from .config import LEAGUES
from .priors import BufferWindows
from .spec import TARGETS

__version__ = "0.1.0"


def __getattr__(name: str):
    """Keep ``fpp.TEST_SEASON`` working, resolved live.

    It is derived from the current season now, so binding it here at import time
    would reintroduce exactly the staleness the accessors exist to remove.
    """
    if name == "TEST_SEASON":
        return config.test_season()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "artifacts", "build", "calibration", "clean", "config", "cv", "evaluate",
    "ingest", "metrics", "models", "paths", "portfolio", "predict", "priors", "reconcile",
    "report", "search", "spec", "staking",
    "RunContext", "FeatureTable", "BufferWindows", "build_feature_table",
    "LEAGUES", "TARGETS", "TEST_SEASON", "__version__",
]
