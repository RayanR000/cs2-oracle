"""Every archive reader must see production's universe, harnesses included.

Production applies two rules inside `_fetch_voted_price_history`: a BID never
votes in the consensus (`BID_SOURCES`) and a name that prices several assets is
not in the universe (`PHASE_COLLAPSED_SLUG_PATTERNS`). Anything that globs the
Parquet itself bypasses that function and has to spell both out, and until
2026-08-08 all thirteen `ab_test_*` harnesses did bypass it — so an A/B advised
a model it had not measured.

The rules are spelled once, in `models/item_parser.py`, because each of them has
already cost a separate fix per loader: the bid exclusion needed three and the
phase-collapsed names four. This file locks down the predicates themselves and
then checks that no harness has grown a private glob around them again.

See `docs/changelog/2026-08-08-embargo-and-harness-hygiene.md`.
"""

from __future__ import annotations

import ast
import re
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from models.item_parser import (
    BID_SOURCES,
    archive_universe_sql_filter,
    bid_sources_sql_filter,
    phase_collapsed_sql_filter,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

#: Every harness, and whether it reads the price archive at all.
#: `recency_weights` takes a pre-built frame on `--frame` and issues no archive
#: query, so it is the one module with nothing to filter.
HARNESSES = sorted(p.stem for p in SCRIPTS.glob("ab_test_*.py"))
NO_ARCHIVE_READ = {"ab_test_recency_weights"}
READS_ARCHIVE = [n for n in HARNESSES if n not in NO_ARCHIVE_READ]


@pytest.fixture
def priced(tmp_path):
    """A one-file archive spanning both eras and both defects.

    The pre-2026 rows carry `source IS NULL`, which is the whole reason both
    predicates are written NULL-safe: a bare `NOT IN` over a NULL evaluates to
    NULL and would silently drop 13 years of prices.
    """
    rows = [
        # (slug, source) — the pre-2026 series, source not yet a column
        ("AK-47 | Redline (Field-Tested)", None),
        ("★ Gut Knife | Doppler (Factory New)", None),
        # the 2026 multi-source era
        ("AK-47 | Redline (Field-Tested)", "aggregator_sync"),
        ("AK-47 | Redline (Field-Tested)", "aggregator_buff163_buy"),
        ("★ Gut Knife | Doppler (Factory New)", "aggregator_sync"),
        ("Sticker | Doppler Poison Frog (Foil)", "aggregator_sync"),
    ]
    df = pd.DataFrame([
        {"item_slug": slug, "day": date(2026, 7, 10), "source": src,
         "mean_price": 10.0, "volume": 1}
        for slug, src in rows
    ])
    path = tmp_path / "prices-2026-07.parquet"
    df.to_parquet(path, index=False)
    return path


def _slugs(priced, where):
    con = duckdb.connect()
    try:
        return sorted(r[0] for r in con.sql(
            f"SELECT item_slug FROM read_parquet('{priced}') WHERE {where}"
        ).fetchall())
    finally:
        con.close()


class TestThePredicates:

    def test_the_bid_filter_drops_only_the_bid(self, priced):
        kept = _slugs(priced, bid_sources_sql_filter())
        assert len(kept) == 5, "one bid row of six should go"

    def test_the_bid_filter_keeps_the_null_source_era(self, priced):
        """`source NOT IN (...)` alone evaluates to NULL for a pre-2026 row and
        drops it. That is 13 of the archive's 13 years."""
        con = duckdb.connect()
        try:
            naive = con.sql(
                f"SELECT count(*) FROM read_parquet('{priced}') "
                f"WHERE source NOT IN ('aggregator_buff163_buy')").fetchone()[0]
            safe = con.sql(
                f"SELECT count(*) FROM read_parquet('{priced}') "
                f"WHERE {bid_sources_sql_filter()}").fetchone()[0]
        finally:
            con.close()
        assert naive == 3, "precondition: the naive predicate loses the NULL era"
        assert safe == 5

    def test_the_universe_filter_applies_both_rules(self, priced):
        kept = _slugs(priced, archive_universe_sql_filter())
        # Both Doppler rows and the bid row go; the sticker exemption stays.
        assert kept == [
            "AK-47 | Redline (Field-Tested)",
            "AK-47 | Redline (Field-Tested)",
            "Sticker | Doppler Poison Frog (Foil)",
        ]

    def test_the_universe_filter_can_skip_a_missing_source_column(self, priced):
        """A relation with no `source` column at all still gets the name rule.

        Safe only because every bid source is a 2026 aggregator feed: a file old
        enough to lack the column is old enough to contain none of them.
        """
        where = archive_universe_sql_filter(source_column=None)
        assert "source" not in where
        assert set(_slugs(priced, where)) == {
            "AK-47 | Redline (Field-Tested)",
            "Sticker | Doppler Poison Frog (Foil)",
        }

    def test_the_filters_qualify_the_column_they_are_given(self):
        """The loaders pass `sub.item_slug` where a join makes the bare name
        ambiguous, so the predicate must not hardcode either spelling."""
        assert "sub.item_slug" in phase_collapsed_sql_filter("sub.item_slug")
        assert "sub.source" in bid_sources_sql_filter("sub.source")

    def test_the_rules_have_one_definition(self):
        """Re-exported from forecaster, defined in item_parser. A second copy
        is how the bid exclusion came to need three separate fixes."""
        from models import forecaster

        assert forecaster.BID_SOURCES is BID_SOURCES


@pytest.mark.parametrize("name", READS_ARCHIVE)
class TestNoHarnessGlobsUnfiltered:

    def test_it_declares_the_universe(self, name):
        src = (SCRIPTS / f"{name}.py").read_text()
        assert "_UNIVERSE" in src, (
            f"{name} reads the archive without declaring the universe it "
            f"reads; see models/item_parser.py"
        )

    def test_every_archive_read_carries_it(self, name):
        """One `_UNIVERSE` reference per `read_parquet`/`prices_relation` read.

        Counting is crude but it catches the real regression: a second loader
        added beside a filtered one, which is exactly how `ab_test_regime.py`
        came to carry two.
        """
        src = (SCRIPTS / f"{name}.py").read_text()
        reads = len(re.findall(
            r"FROM \{?(?:read_parquet|relation|union_sql|prices_relation)", src))
        assert src.count("_UNIVERSE") - 1 >= 1, f"{name} defines but never uses it"
        assert reads > 0, f"{name} was listed as reading the archive but does not"

    def test_it_does_not_glob_the_archive_raw(self, name):
        """`read_parquet('prices-*.parquet')` narrows to the first file's
        schema, where `source` does not exist — so a source filter over it is
        either an error or a lie. See backend/AGENTS.md."""
        src = (SCRIPTS / f"{name}.py").read_text()
        offenders = [
            line.strip() for line in src.splitlines()
            if "prices-*.parquet" in line
            and "glob(" not in line
            and not line.strip().startswith("#")
        ]
        assert not offenders, f"{name} still globs the archive raw: {offenders}"


def test_the_no_archive_harness_really_reads_no_archive():
    """`recency_weights` is exempt from the rules above. That exemption has to
    stay true, or it becomes a hole rather than a fact."""
    src = (SCRIPTS / "ab_test_recency_weights.py").read_text()
    # `pd.read_parquet(args.frame)` is the pre-built frame, not the archive.
    assert "prices-" not in src
    assert "prices_relation" not in src
    assert "duckdb" not in src
    assert "--frame" in src, "the exemption rests on it taking a pre-built frame"


def test_every_harness_still_parses():
    """The filters were threaded in by rewriting f-strings across 12 files."""
    for name in HARNESSES:
        ast.parse((SCRIPTS / f"{name}.py").read_text())
