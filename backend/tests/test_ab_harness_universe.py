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
    historical_fallback_sql_filter,
    phase_collapsed_sql_filter,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

#: Every harness, and whether it reads the price archive at all.
#:
#: Two modules issue no archive query and so have nothing to filter:
#: `recency_weights` takes a pre-built frame on `--frame`, and `frozen_runs`
#: delegates every read to `walkforward_backtest.run_walkforward`, which
#: already routes through `archive_universe_sql_filter`.
#:
#: Adding a name here is a claim that the module contains no archive read.
#: `test_the_exempt_harnesses_really_do_not_read_the_archive` enforces it, so
#: an exemption cannot be used to smuggle an unfiltered glob past the checks
#: below.
HARNESSES = sorted(p.stem for p in SCRIPTS.glob("ab_test_*.py"))
NO_ARCHIVE_READ = {"ab_test_recency_weights", "ab_test_frozen_runs"}
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
        # a re-stamped stale price: fresh `day`, prefixed source
        ("AK-47 | Redline (Field-Tested)", "historical_fallback:aggregator_sync"),
    ]
    df = pd.DataFrame(
        [
            {"item_slug": slug, "day": date(2026, 7, 10), "source": src, "mean_price": 10.0, "volume": 1}
            for slug, src in rows
        ]
    )
    path = tmp_path / "prices-2026-07.parquet"
    df.to_parquet(path, index=False)
    return path


def _slugs(priced, where):
    con = duckdb.connect()
    try:
        return sorted(r[0] for r in con.sql(f"SELECT item_slug FROM read_parquet('{priced}') WHERE {where}").fetchall())
    finally:
        con.close()


class TestThePredicates:
    def test_the_bid_filter_drops_only_the_bid(self, priced):
        kept = _slugs(priced, bid_sources_sql_filter())
        assert len(kept) == 6, "one bid row of seven should go"

    def test_the_fallback_filter_drops_only_the_restamp(self, priced):
        kept = _slugs(priced, historical_fallback_sql_filter())
        assert len(kept) == 6, "one historical_fallback row of seven should go"
        assert all("historical_fallback" not in s for s in kept)

    def test_the_fallback_filter_keeps_the_null_source_era(self, priced):
        """`NOT LIKE 'historical_fallback:%'` alone is NULL for a pre-2026 row and
        would drop it; the predicate is NULL-safe."""
        con = duckdb.connect()
        try:
            naive = con.sql(
                f"SELECT count(*) FROM read_parquet('{priced}') WHERE source NOT LIKE 'historical_fallback:%'"
            ).fetchone()[0]
            safe = con.sql(
                f"SELECT count(*) FROM read_parquet('{priced}') WHERE {historical_fallback_sql_filter()}"
            ).fetchone()[0]
        finally:
            con.close()
        assert naive == 4, "precondition: the naive predicate loses the NULL era"
        assert safe == 6

    def test_the_bid_filter_keeps_the_null_source_era(self, priced):
        """`source NOT IN (...)` alone evaluates to NULL for a pre-2026 row and
        drops it. That is 13 of the archive's 13 years."""
        con = duckdb.connect()
        try:
            naive = con.sql(
                f"SELECT count(*) FROM read_parquet('{priced}') WHERE source NOT IN ('aggregator_buff163_buy')"
            ).fetchone()[0]
            safe = con.sql(
                f"SELECT count(*) FROM read_parquet('{priced}') WHERE {bid_sources_sql_filter()}"
            ).fetchone()[0]
        finally:
            con.close()
        assert naive == 4, "precondition: the naive predicate loses the NULL era"
        assert safe == 6

    def test_the_universe_filter_applies_all_rules(self, priced):
        kept = _slugs(priced, archive_universe_sql_filter())
        # Both Doppler rows, the bid row and the historical_fallback re-stamp go;
        # the sticker exemption stays.
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


@pytest.mark.parametrize("name", sorted(NO_ARCHIVE_READ))
def test_the_exempt_harnesses_really_do_not_read_the_archive(name):
    """An exemption must be earned, not asserted.

    NO_ARCHIVE_READ turns off every universe check for a module, so without
    this the set is a hole: adding a name to it is all anyone would need to do
    to land an unfiltered glob. A module claiming the exemption must contain no
    archive read of any kind — if it grows one, it stops being exempt and the
    checks below apply to it again.
    """
    src = (SCRIPTS / f"{name}.py").read_text()
    reads = [
        line.strip()
        for line in src.splitlines()
        # `prices_relation` is the archive reader outright. A bare
        # `read_parquet(` is DuckDB's, inside a SQL string, and is an archive
        # glob; `pd.read_parquet(` is pandas opening a file the CALLER named,
        # which is how `recency_weights` takes its `--frame` and is not an
        # archive read at all.
        if re.search(r"prices_relation\s*\(", line) or re.search(r"(?<!pd\.)(?<!pandas\.)\bread_parquet\s*\(", line)
        if not line.strip().startswith("#")
    ]
    assert not reads, (
        f"{name} is in NO_ARCHIVE_READ but reads the archive: {reads}. Remove the exemption and declare _UNIVERSE."
    )


@pytest.mark.parametrize("name", READS_ARCHIVE)
class TestNoHarnessGlobsUnfiltered:
    def test_it_declares_the_universe(self, name):
        src = (SCRIPTS / f"{name}.py").read_text()
        assert "_UNIVERSE" in src, (
            f"{name} reads the archive without declaring the universe it reads; see models/item_parser.py"
        )

    def test_every_archive_read_carries_it(self, name):
        """One `_UNIVERSE` reference per `read_parquet`/`prices_relation` read.

        Counting is crude but it catches the real regression: a second loader
        added beside a filtered one, which is exactly how `ab_test_regime.py`
        came to carry two.
        """
        src = (SCRIPTS / f"{name}.py").read_text()
        reads = len(re.findall(r"FROM \{?(?:read_parquet|relation|union_sql|prices_relation)", src))
        assert src.count("_UNIVERSE") - 1 >= 1, f"{name} defines but never uses it"
        assert reads > 0, f"{name} was listed as reading the archive but does not"

    def test_it_does_not_glob_the_archive_raw(self, name):
        """`read_parquet('prices-*.parquet')` narrows to the first file's
        schema, where `source` does not exist — so a source filter over it is
        either an error or a lie. See backend/AGENTS.md."""
        src = (SCRIPTS / f"{name}.py").read_text()
        offenders = [
            line.strip()
            for line in src.splitlines()
            if "prices-*.parquet" in line and "glob(" not in line and not line.strip().startswith("#")
        ]
        assert not offenders, f"{name} still globs the archive raw: {offenders}"


#: Harnesses whose union builder filters rows on `source`, and the value each
#: one asks for. Every one of them sniffed `"source" in cols` to decide whether
#: to apply that filter, which was the same question as "is this a 2026 file"
#: only until the archive was migrated.
SOURCE_FILTERED = {
    "ab_test_csfloat_basis": "aggregator_sync",
    "ab_test_item_metadata": "aggregator_sync",
    "ab_test_training_breadth": "aggregator_sync",
    "ab_test_price_primitives": "aggregator_sync",
    "ab_test_volume_features": "aggregator_sync",
    "ab_test_feature_contribution": "STEAMCOMMUNITY",
    "ab_test_direction_labels": "STEAMCOMMUNITY",
    "ab_test_q50_sampling": "STEAMCOMMUNITY",
}


@pytest.fixture
def migrated_archive(tmp_path):
    """A **migrated** two-file archive: the shape that broke every harness.

    `normalize_price_schema.py` materialises `source` on the pre-2026 files as
    a typed all-NULL column. That is the correct migration — `source IS NULL`
    is what "the pre-2026 series" means, per `db/archive.py::prices_relation`.
    What it breaks is any caller that decided *whether* to filter on `source`
    by asking whether the column exists. Before the migration those two
    questions had the same answer; after it, the pre-2026 file takes the
    `source = '...'` branch, matches nothing, and 13 years of prices vanish.
    """
    old = pd.DataFrame(
        [
            {
                "item_slug": "AK-47 | Redline (Field-Tested)",
                "day": date(2025, 6, 1),
                "source": None,
                "mean_price": 10.0,
                "volume": 1,
            },
        ]
    )
    old.to_parquet(tmp_path / "prices-2025.parquet", index=False)
    new = pd.DataFrame(
        [
            {
                "item_slug": "AK-47 | Redline (Field-Tested)",
                "day": date(2026, 7, 10),
                "source": "aggregator_sync",
                "mean_price": 11.0,
                "volume": 1,
            },
        ]
    )
    new.to_parquet(tmp_path / "prices-2026-07.parquet", index=False)
    return tmp_path


class TestTheMigratedArchiveStillYieldsThePre2026Series:
    """The 2026-08-08 archive migration silently emptied eight harnesses.

    All three that were re-run that day failed on an empty universe — one with
    `IndexError: list index out of range`, one with `IN ()`, one on a stale
    cache fingerprint. None of them said "zero rows", which is why this is
    pinned behaviourally and not just by reading the source.
    """

    @pytest.mark.parametrize("name", ["ab_test_csfloat_basis", "ab_test_item_metadata", "ab_test_training_breadth"])
    def test_the_union_returns_pre_2026_rows(self, name, migrated_archive, monkeypatch):
        import importlib

        mod = importlib.import_module(f"scripts.{name}")
        monkeypatch.setattr(mod, "ARCHIVE_DIR", migrated_archive)
        con = duckdb.connect()
        try:
            n_old = con.sql(
                f"SELECT count(*) FROM ({mod._archive_union_sql(con)}) WHERE day < DATE '2026-01-01'"
            ).fetchone()[0]
        finally:
            con.close()
        assert n_old == 1, (
            f"{name} dropped the pre-2026 series from a migrated archive: its "
            f"union filters `source` on a file whose source column is all NULL"
        )


@pytest.mark.parametrize("name", sorted(SOURCE_FILTERED))
def test_a_source_filter_is_null_safe(name):
    """Guards the whole family, including the five not re-run on 2026-08-08.

    A bare `source = 'x'` is only ever correct for the 2026 era. Written
    NULL-safe it is correct in both, which removes the need to decide per file
    — and it is the rule `prices_relation` already documents.
    """
    src = (SCRIPTS / f"{name}.py").read_text()
    for line in src.splitlines():
        # Matches the comparison itself, not the `WHERE` in front of it: after
        # the fix the clause reads `WHERE (source IS NULL OR source = '...')`,
        # and anchoring on `WHERE source` would make this pass by not matching.
        if re.search(r"source = '", line) and not line.strip().startswith("#"):
            assert "source IS NULL" in line, (
                f"{name} filters `source` without a NULL-safe clause, so a "
                f"migrated pre-2026 file contributes nothing: {line.strip()}"
            )


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
