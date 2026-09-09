"""Tests for the Reddit structured-event detector.

The classifier is pure: no creds, no network. Future-dated posts must never
enter a past panel (leakage guard).
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from collectors.reddit_events import (  # noqa: E402
    classify_narratives,
    collect,
    posts_to_events,
)

DAY = date(2026, 9, 8)
AT = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
TS = datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc).timestamp()


def _post(title="Whale sweeping the Kilowatt Case supply", **kw):
    base = {"title": title, "body": "", "created_utc": TS, "score": 100,
            "num_comments": 20, "url": "https://example.com/p",
            "item_slug": "kilowatt-case", "family": "case"}
    base.update(kw)
    return base


class TestClassifier:
    def test_whale_and_supply_tags(self):
        tags = classify_narratives("Whale buying up the whole supply, sold out everywhere")
        assert "whale_buying" in tags
        assert "supply_disappearance" in tags

    def test_tradeup_and_valve_tags(self):
        assert "tradeup_path" in classify_narratives("New trade-up path found")
        assert "valve_update" in classify_narratives("Valve update changes case drops")

    def test_no_match_is_no_event(self):
        assert classify_narratives("Nice screenshot from tonight's match") == []


class TestPanel:
    def test_posts_reduce_to_events_with_first_seen(self):
        rows = posts_to_events([_post(), _post(score=10, num_comments=2)], DAY, AT)
        assert len(rows) >= 1
        whale = rows[rows["narrative"] == "whale_buying"].iloc[0]
        assert whale["post_count"] == 2
        assert whale["item_slug"] == "kilowatt-case"

    def test_future_posts_are_dropped(self):
        future = _post(created_utc=datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc).timestamp())
        rows = posts_to_events([future], DAY, AT)
        assert rows.empty

    def test_reversed_sentiment_is_negated(self):
        rows = posts_to_events([_post(sentiment=0.6)], DAY, AT)
        row = rows.iloc[0]
        assert row["sentiment_mean"] == 0.6
        assert row["sentiment_reversed"] == -0.6


class TestCollectWithoutCreds:
    def test_skips_loudly_not_zero_row_success(self, tmp_path, monkeypatch):
        monkeypatch.delenv("REDDIT_BEARER_TOKEN", raising=False)
        monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
        monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
        summary = collect(archive_dir=tmp_path, snapshot_day=DAY)
        assert summary["status"] == "skipped"
        assert "reason" in summary
