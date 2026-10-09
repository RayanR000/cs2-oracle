"""A requests.Session that retries transient failures.

The daily dumps are live-only: a feed missed today cannot be fetched tomorrow,
so one dropped connection or one 503 costs that source-day permanently. Every
collector GET goes through this session instead of a bare requests.Session.

Retried: connection errors, read timeouts, 429 and 5xx, on GET only, up to
four attempts with backoff 0s, 2s, 4s and Retry-After honoured. Not retried:
any other 4xx (a Cloudflare 403 or a 406 is not transient), and a 200 with a bad
payload, which the callers' own shape checks still refuse. After the last
attempt the final response is returned, not raised, so each caller's existing
status handling and error messages are unchanged.
"""

from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

RETRY_TOTAL = 3
RETRY_BACKOFF_FACTOR = 1.0
RETRY_STATUSES = (429, 500, 502, 503, 504)


def retry_policy() -> Retry:
    return Retry(
        total=RETRY_TOTAL,
        connect=RETRY_TOTAL,
        read=RETRY_TOTAL,
        status=RETRY_TOTAL,
        backoff_factor=RETRY_BACKOFF_FACTOR,
        status_forcelist=RETRY_STATUSES,
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )


def retrying_session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_policy())
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session
