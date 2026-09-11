"""Network plumbing shared by every scraping path.

``with_retry`` is lifted from the original Data Pull notebook (cell 4) and kept
deliberately identical in behaviour: linear backoff, re-raise on final failure.
"""

from __future__ import annotations

import time
from typing import Any, Callable, TypeVar

import requests

MAX_RETRIES = 3
RETRY_DELAY = 15  # seconds, multiplied by attempt number

T = TypeVar("T")

# The host the original notebook used, `site.api.espn.com`, now returns 403 for
# direct requests. `site.web.api.espn.com` serves the same payloads and works.
ESPN_SITE_API = "https://site.web.api.espn.com/apis/site/v2/sports/soccer"
ESPN_CORE_API = "https://sports.core.api.espn.com/v2/sports/soccer/leagues"

# ESPN rejects requests without a browser-ish User-Agent.
ESPN_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-GB,en;q=0.9",
    "Referer": "https://www.espn.com/",
}


def with_retry(
    fn: Callable[..., T],
    *args: Any,
    retries: int = MAX_RETRIES,
    backoff: int = RETRY_DELAY,
    **kwargs: Any,
) -> T:
    """Call ``fn(*args, **kwargs)``, retrying with linear backoff on any exception."""
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 -- deliberately broad, this is a retry wrapper
            if attempt == retries:
                print(f"  All {retries} attempts failed.")
                raise
            wait = backoff * attempt
            print(f"  Attempt {attempt}/{retries} failed ({type(e).__name__}: {e}). Retrying in {wait}s...")
            time.sleep(wait)
    raise AssertionError("unreachable")


_session: requests.Session | None = None


def session() -> requests.Session:
    """A single pooled session -- the old code opened a new connection per call."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update(ESPN_HEADERS)
    return _session


def get_json(url: str, timeout: int = 25) -> dict:
    """GET + parse JSON, with retry. Raises on non-2xx."""

    def _fetch() -> dict:
        r = session().get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()

    return with_retry(_fetch)


__all__ = ["with_retry", "session", "get_json", "ESPN_SITE_API", "ESPN_CORE_API", "ESPN_HEADERS"]
