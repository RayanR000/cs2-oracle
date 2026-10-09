# 2026-10-09 — Collector GETs retry transient failures

## Why

The daily dumps are live-only. A feed that fails today cannot be fetched tomorrow, and a
failed feed is skipped while the run stays green. Before this change, one dropped connection,
read timeout or 503 cost that source-day permanently. The CSGOTrader aggregator, the supply
depth feeds and the Skinport sales-volume call all used a bare `requests.Session`.

## What the logs show

This is insurance, not a fix for an observed loss:

- The last 200 Aggregator runs (back to 2026-06-09) all concluded `success`.
- Across the last 40 run logs there is no `Failed to fetch`, timeout, connection error,
  429 or 5xx. The only recurring feed failure was bitskins' payload-shape refusal, which a
  retry would not have helped; that feed was dropped in #88.
- The performance review's "the 2026-08-23 run died on one `OperationalError`" has no matching
  failed run. It most likely falls in the window when the workflows were disabled. No DB retry
  is added: there is no evidence for one, and `pool_pre_ping=True` already covers stale
  connections.

## Change

`collectors/http.py::retrying_session()` mounts an `HTTPAdapter` with urllib3 `Retry`:

- up to 4 attempts, backoff 0 s / 2 s / 4 s, `Retry-After` honoured;
- GET only, on connection errors, read timeouts, 429 and 500/502/503/504;
- any other 4xx is not retried (a Cloudflare 403 or a 406 is a block, not an outage);
- `raise_on_status=False`, so after the last attempt the caller gets the final response and its
  existing status handling and error messages are unchanged.

Used by `CSGOTraderAggregator`, `supply_depth` (`_get_json`, the per-feed and ladder sessions)
and `sales_volume.fetch_sales_history`. Injected sessions (tests) are untouched.

Worst case per request adds ~6 s of backoff plus three more timeouts. The supply ladder is the
slow step at ~7 min; the job's limit is 60 min.

## Tests

`tests/test_collector_http_retry.py` runs a local HTTP server and covers: 5xx retried to
success, 429 retried, 403 not retried, exhausted retries returning the last response, and the
aggregator's session carrying the policy. With the adapter unmounted, 4 of the 5 fail (the 403
case passes either way, as it should).
