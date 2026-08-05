"""Nested Python values must not be handed to DuckDB's type inference.

``append_table`` used to pass a pandas column of raw Python dicts straight to
DuckDB, which infers that column's SQL type FROM THE BATCH'S CONTENTS. Measured
2026-08-05, the same code path inferred three different types for
``prediction_accuracy.metrics``:

* ``STRUCT`` when every dict in the batch had identical keys and value types,
* ``MAP(VARCHAR, DOUBLE)`` when a nested dict's key set differed between rows
  (``mape_by_tier`` carries only the tiers a cohort actually has),
* ``VARCHAR`` holding a Python ``repr`` when it could reconcile neither.

The on-disk file holds a STRUCT with a FROZEN field list, so the ``UNION ALL``
in ``_append_parquet`` has to cast today's inferred type into that struct and
raises whenever they disagree. That data-dependence is why the 2026-08-05 00:07
run passed and the 10:32 run failed on identical code, one reporting
``Could not convert string 'None' to DOUBLE`` and the next
``Type VARCHAR ... can't be cast to ... STRUCT(...)``.

``_append_parquet``'s existing drift handling (``union_cols`` / ``_project``)
cannot help: it NULLs out whole columns a side lacks, and ``metrics`` is ONE
column — drift *inside* its struct is invisible to it.

Three ops tables carried a nested column when this was written —
``prediction_accuracy.metrics``, ``collection_runs.source_breakdown`` and
``accuracy_alerts.details`` — so this is a property of the store, not of one
table, and the fix lives in the store.
"""
from __future__ import annotations

import json

import duckdb
import pandas as pd

from db.parquet import _append_parquet, replace_rows


def _frozen_struct_file(path):
    """A file whose ``metrics`` column is a frozen STRUCT, as production's was."""
    pd.DataFrame([
        {"prediction_type": "forecast", "horizon_days": 7, "price_tier": 1,
         "metrics": {"mae": 1.0, "mape_by_tier": {"tier_1": 8.0}}},
    ]).to_parquet(path, index=False)
    assert "STRUCT" in _column_type(path, "metrics"), "fixture is not a STRUCT"


def _column_type(path, column):
    con = duckdb.connect()
    try:
        return next(
            t for c, t, *_ in
            con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()
            if c == column
        )
    finally:
        con.close()


def _metrics_of(path, horizon):
    out = pd.read_parquet(path)
    row = out[out.horizon_days == horizon].iloc[0]
    return json.loads(row["metrics"])


KEYS = ["prediction_type", "horizon_days", "price_tier"]


class TestNestedColumnsBecomeJson:
    def test_a_metrics_key_the_file_lacks_does_not_raise(self, tmp_path):
        """(a) of the regression the triage doc asked for: a brand-new key.

        Commit d7ffaae added four fields to score_cohort and the next run died.
        """
        path = tmp_path / "prediction_accuracy.parquet"
        _frozen_struct_file(path)

        _append_parquet(path, pd.DataFrame([
            {"prediction_type": "forecast", "horizon_days": 3, "price_tier": 1,
             "metrics": {"mae": 2.0, "mape_by_tier": {"tier_1": 9.0},
                         "directional_accuracy_moved": 51.2,
                         "n_unchanged": 7, "date_coverage_sufficient": False}},
        ]), KEYS)

        assert _metrics_of(path, 3)["directional_accuracy_moved"] == 51.2
        assert _metrics_of(path, 3)["date_coverage_sufficient"] is False

    def test_a_none_in_a_numeric_metrics_field_does_not_raise(self, tmp_path):
        """(b) of the regression: ``None`` where the struct says DOUBLE.

        skill_vs_baseline is None when baseline_mae == 0, and the carry-forward
        split is None for an empty partition. Both are legitimate, and both are
        data-dependent — which is what made the failure intermittent.
        """
        path = tmp_path / "prediction_accuracy.parquet"
        _frozen_struct_file(path)

        _append_parquet(path, pd.DataFrame([
            {"prediction_type": "forecast", "horizon_days": 3, "price_tier": 1,
             "metrics": {"mae": 2.0, "skill_vs_baseline": None,
                         "directional_accuracy_moved": None}},
        ]), KEYS)

        m = _metrics_of(path, 3)
        assert m["skill_vs_baseline"] is None
        assert m["directional_accuracy_moved"] is None

    def test_rows_in_one_batch_may_carry_different_nested_key_sets(self, tmp_path):
        """score_by_tier emits exactly this: mape_by_tier differs per cohort.

        The tier-4 row has only tier_4, the all-tiers row has tier_0..4. This is
        the batch shape that made DuckDB fall back to MAP, then to VARCHAR.
        """
        path = tmp_path / "prediction_accuracy.parquet"
        _append_parquet(path, pd.DataFrame([
            {"prediction_type": "forecast", "horizon_days": 7, "price_tier": 4,
             "metrics": {"mae": 1.0, "mape_by_tier": {"tier_4": 3.0}}},
            {"prediction_type": "forecast", "horizon_days": 7, "price_tier": None,
             "metrics": {"mae": 2.0, "mape_by_tier": {"tier_0": 1.0, "tier_4": 3.0}}},
        ]), KEYS)

        out = pd.read_parquet(path)
        by_tier = {
            (None if pd.isna(r.price_tier) else int(r.price_tier)):
                json.loads(r.metrics)["mape_by_tier"]
            for r in out.itertuples()
        }
        assert by_tier[4] == {"tier_4": 3.0}
        assert by_tier[None] == {"tier_0": 1.0, "tier_4": 3.0}

    def test_the_column_lands_as_a_stable_scalar_type(self, tmp_path):
        """One type forever, so no future field can reopen this class."""
        path = tmp_path / "prediction_accuracy.parquet"
        _append_parquet(path, pd.DataFrame([
            {"prediction_type": "forecast", "horizon_days": 7, "price_tier": 1,
             "metrics": {"mae": 1.0}},
        ]), KEYS)
        assert _column_type(path, "metrics") == "VARCHAR"

        # A second write with a wholly different field set keeps that type.
        _append_parquet(path, pd.DataFrame([
            {"prediction_type": "forecast", "horizon_days": 3, "price_tier": 1,
             "metrics": {"something_new": [1, 2, 3]}},
        ]), KEYS)
        assert _column_type(path, "metrics") == "VARCHAR"
        assert _metrics_of(path, 3)["something_new"] == [1, 2, 3]

    def test_a_list_valued_column_is_also_carried(self, tmp_path):
        """accuracy_alerts.details holds STRUCT(window_accuracies DOUBLE[])."""
        path = tmp_path / "accuracy_alerts.parquet"
        _append_parquet(path, pd.DataFrame([
            {"id": 1, "details": {"window_accuracies": [33.7, 61.5]}},
        ]), ["id"])
        out = pd.read_parquet(path)
        assert json.loads(out.iloc[0]["details"]) == {"window_accuracies": [33.7, 61.5]}

    def test_a_genuinely_null_nested_value_stays_null(self, tmp_path):
        """source_breakdown and details are both nullable columns."""
        path = tmp_path / "collection_runs.parquet"
        _append_parquet(path, pd.DataFrame([
            {"id": 1, "source_breakdown": {"aggregator": 5}},
            {"id": 2, "source_breakdown": None},
        ]), ["id"])
        out = pd.read_parquet(path).sort_values("id")
        assert json.loads(out.iloc[0]["source_breakdown"]) == {"aggregator": 5}
        assert out.iloc[1]["source_breakdown"] is None


class TestNullValuedDedupKeys:
    """A NULL in a dedup key is a key VALUE, and must match a NULL.

    ``prediction_accuracy`` uses NULL price_tier as a discriminator meaning "the
    all-tiers aggregate" — the row every accuracy endpoint serves by default —
    and horizon_days / model_version are nullable too. The anti-join compared
    keys with ``=``, and ``NULL = NULL`` is NULL rather than TRUE, so those rows
    never matched their replacement and accumulated a duplicate per run.

    This was latent until the mirror gained ``price_tier``: while the column was
    absent from the file it was dropped from ``usable_keys`` altogether, so
    dedup ran on the four non-NULL keys and happened to be correct. The DB side
    was never affected — ``filter_by(price_tier=None)`` emits ``IS NULL`` — so
    this is another way for the two stores to disagree.
    """

    def test_a_null_key_row_is_replaced_not_duplicated(self, tmp_path):
        path = tmp_path / "prediction_accuracy.parquet"
        row = {"prediction_type": "forecast", "horizon_days": 7,
               "price_tier": None, "metrics": {"mae": 1.0}}

        _append_parquet(path, pd.DataFrame([row]), KEYS)
        _append_parquet(path, pd.DataFrame([dict(row, metrics={"mae": 2.0})]), KEYS)

        out = pd.read_parquet(path)
        assert len(out) == 1, (
            f"the all-tiers row duplicated instead of being replaced:\n{out}"
        )
        assert json.loads(out.iloc[0]["metrics"])["mae"] == 2.0

    def test_a_null_key_still_does_not_collide_with_a_non_null_one(self, tmp_path):
        """The aggregate and the real tiers stay distinct rows."""
        path = tmp_path / "prediction_accuracy.parquet"
        _append_parquet(path, pd.DataFrame([
            {"prediction_type": "forecast", "horizon_days": 7, "price_tier": None,
             "metrics": {"mae": 1.0}},
            {"prediction_type": "forecast", "horizon_days": 7, "price_tier": 1,
             "metrics": {"mae": 2.0}},
        ]), KEYS)
        out = pd.read_parquet(path)
        assert len(out) == 2, "the aggregate and tier 1 collided"

    def test_the_whole_tier_fanout_is_stable_across_runs(self, tmp_path):
        """score_by_tier emits 7 rows per (horizon, model); a re-run must too."""
        path = tmp_path / "prediction_accuracy.parquet"
        rows = pd.DataFrame([
            {"prediction_type": "forecast", "horizon_days": 7, "price_tier": t,
             "metrics": {"mae": 1.0}}
            for t in (0, 1, 2, 3, 4, -1, None)
        ])
        _append_parquet(path, rows, KEYS)
        _append_parquet(path, rows, KEYS)
        assert len(pd.read_parquet(path)) == 7


class TestExistingStructFilesAreMigratedInPlace:
    """The served mirror is gitignored, so no operator can be relied on to run a
    migration script before the next unattended daily run. The append converts
    the existing side itself, in the same single rewrite."""

    def test_pre_existing_struct_rows_survive_as_json(self, tmp_path):
        path = tmp_path / "prediction_accuracy.parquet"
        _frozen_struct_file(path)

        _append_parquet(path, pd.DataFrame([
            {"prediction_type": "forecast", "horizon_days": 3, "price_tier": 1,
             "metrics": {"mae": 2.0, "brand_new": 1.0}},
        ]), KEYS)

        # The row that was already on disk is still there, and still readable.
        assert _metrics_of(path, 7) == {"mae": 1.0, "mape_by_tier": {"tier_1": 8.0}}
        assert _column_type(path, "metrics") == "VARCHAR"

    def test_migrated_inner_fields_stay_queryable(self, tmp_path):
        """Queryability was the cost of choosing JSON; DuckDB still has it."""
        path = tmp_path / "prediction_accuracy.parquet"
        _frozen_struct_file(path)
        _append_parquet(path, pd.DataFrame([
            {"prediction_type": "forecast", "horizon_days": 3, "price_tier": 1,
             "metrics": {"mae": 2.0}},
        ]), KEYS)

        con = duckdb.connect()
        try:
            got = con.execute(
                f"SELECT horizon_days, CAST(metrics->>'$.mae' AS DOUBLE) AS mae "
                f"FROM read_parquet('{path}') ORDER BY horizon_days"
            ).fetchall()
        finally:
            con.close()
        assert got == [(3, 2.0), (7, 1.0)]

    def test_replace_rows_migrates_the_same_way(self, tmp_path, monkeypatch):
        """replace_rows has the identical exposure and must not diverge.

        It is the write path --reresolve takes, and it raises rather than
        blanking a column the incoming rows lack, so a nested column it could
        not reconcile would fail the backfill instead of the daily run.
        """
        monkeypatch.setattr("db.parquet.OPS_DIR", tmp_path)
        path = tmp_path / "prediction_accuracy.parquet"
        _frozen_struct_file(path)

        replace_rows(
            "prediction_accuracy", "horizon_days", [7],
            [{"prediction_type": "forecast", "horizon_days": 7, "price_tier": 1,
              "metrics": {"mae": 5.0, "brand_new": 2.0}}],
        )

        assert _column_type(path, "metrics") == "VARCHAR"
        assert _metrics_of(path, 7) == {"mae": 5.0, "brand_new": 2.0}
