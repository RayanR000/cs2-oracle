"""The collectors' session retries transient failures and nothing else."""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from collectors import http as collector_http
from collectors.http import RETRY_TOTAL, retrying_session


@pytest.fixture
def server(monkeypatch):
    """A local server answering each GET from a per-test script of status codes."""
    monkeypatch.setattr(collector_http, "RETRY_BACKOFF_FACTOR", 0)
    state = {"script": [], "hits": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            code = state["script"][min(state["hits"], len(state["script"]) - 1)]
            state["hits"] += 1
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok": true}')

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/", state
    httpd.shutdown()


def test_transient_5xx_is_retried_until_success(server):
    url, state = server
    state["script"] = [503, 502, 200]
    with retrying_session() as s:
        resp = s.get(url, timeout=5)
    assert resp.status_code == 200
    assert state["hits"] == 3


def test_429_is_retried(server):
    url, state = server
    state["script"] = [429, 200]
    with retrying_session() as s:
        assert s.get(url, timeout=5).status_code == 200
    assert state["hits"] == 2


def test_other_4xx_is_not_retried(server):
    """A Cloudflare 403 or a 406 is a block, not an outage."""
    url, state = server
    state["script"] = [403, 200]
    with retrying_session() as s:
        assert s.get(url, timeout=5).status_code == 403
    assert state["hits"] == 1


def test_exhausted_retries_return_the_last_response(server):
    """Callers keep their own status handling, so the final 503 comes back
    as a response rather than a urllib3 MaxRetryError."""
    url, state = server
    state["script"] = [503]
    with retrying_session() as s:
        assert s.get(url, timeout=5).status_code == 503
    assert state["hits"] == RETRY_TOTAL + 1


def test_collectors_use_the_retrying_session():
    from collectors.csgotrader_aggregator import CSGOTraderAggregator

    with CSGOTraderAggregator() as agg:
        adapter = agg.session.get_adapter("https://prices.csgotrader.app/")
        assert adapter.max_retries.total == RETRY_TOTAL
