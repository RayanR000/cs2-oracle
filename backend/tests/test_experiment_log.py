"""The experiment log stays queryable: schema, enum, and resolvable links.

`docs/experiment_log.csv` is the append-only answer to "has this been tried":
one row per arm with its verdict and the note that measured it. Without this
test the log rots — a malformed row, a dead link, or a creative new verdict
spelling silently un-answers the question. See
docs/changelog/2026-09-15-experiment-log.md.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LOG = REPO_ROOT / "docs" / "experiment_log.csv"

COLUMNS = ["date", "name", "hypothesis", "metric", "estimate", "ci_lo", "ci_hi", "mde", "n", "verdict", "link"]
VERDICTS = {"shipped", "refuted", "void", "measured", "inconclusive"}


def _rows():
    with LOG.open(newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == COLUMNS, f"schema drift: {reader.fieldnames}"
        return list(reader)


def test_log_schema_and_verdict_enum():
    rows = _rows()
    assert len(rows) >= 50, f"backfill lost rows: {len(rows)}"
    seen = set()
    for r in rows:
        date.fromisoformat(r["date"])  # raises on non-YYYY-MM-DD
        assert r["verdict"] in VERDICTS, f"{r['name']}: verdict {r['verdict']!r} not in {sorted(VERDICTS)}"
        for col in ("name", "hypothesis", "metric", "verdict", "link"):
            assert r[col].strip(), f"{r['name']}: empty {col}"
        key = (r["date"], r["name"])
        assert key not in seen, f"duplicate row: {key}"
        seen.add(key)


def test_every_row_links_to_a_repo_file():
    for r in _rows():
        target = (REPO_ROOT / r["link"]).resolve()
        assert str(target).startswith(str(REPO_ROOT)), f"{r['name']}: link escapes repo: {r['link']}"
        assert target.is_file(), f"{r['name']}: dead link: {r['link']}"


def test_dead_ideas_are_present():
    """The rows this log exists for: re-proposing any of these is the failure.
    Keyed by arm name so a rename still fails loudly here rather than silently
    dropping the guard."""
    names = {r["name"] for r in _rows()}
    for dead in (
        "market-relative-labels",
        "smoothed-anchor-label",
        "learned-band-scale",
        "qhat-bagging",
        "mondrian-conformal",
        "social-features",
        "volume-in-scale",
        "rank-transform-transfer",
    ):
        assert dead in names, f"dead idea {dead!r} missing from the log"
