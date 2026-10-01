# Composition-Break Calendar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Void training labels that span a market-wide source-composition switch (catching
2026-01-01 and 2026-04-16, which the item-count detector misses on the train universe). Build, gated off, a rule that NaNs lookback
features crossing such a switch. Remove the dead `SHRINK_K_GBM` / vol-rank plumbing in the same
retrain PR.

**Architecture:**
- The DuckDB vote and its pandas reference each emit a canonical `source_set` string per
  item-day.
- `fetch_price_history` derives a break calendar (the item-count rule ∪ a new composition rule)
  from the voted frame before `engineer_features` drops the column, and stores it on the
  forecaster.
- `prepare_targets` voids labels against that calendar.
- Behind `BREAK_AWARE_LOOKBACKS=1`, `build_training_data` and `_prepare_predict_features` mask
  features by a declared per-feature lookback window.

**Tech Stack:** Python 3.13 locally / 3.11 in CI, pandas, DuckDB, LightGBM, pytest. All commands
run from `backend/` through `venv/bin/python`. The repo's convention, `uv run` with
`PYTHONPATH=.`, also works.

**Spec:** `docs/specs/2026-09-30-composition-break-calendar-design.md`

## Global Constraints

- `COMPOSITION_BREAK_FRACTION = 0.90`; minimum paired cross-section `MIN_DEGENERATE_CROSS_SECTION`
  (25, existing constant).
- `VOTED_CACHE_VERSION` goes 11 → 12.
- NULL source is spelled `<null>` inside `source_set`. Members are sorted and joined with `|`.
- `BREAK_AWARE_LOOKBACKS` defaults **off**. Serving follows the artifact's
  `meta["break_aware_lookbacks"]`; a missing key reads as off.
- Never remove an env feature flag without a retrain in the same PR. This PR is merged to trigger
  the Mon 2026-10-05 `mode=full` retrain.
- A label rule is never A/B-scored on DA alone (invariant 4). Part B is not switched on in this PR.
- No Claude attribution in commits.
- **`backend/.env` points at production.** Never run a script from `backend/` that opens a DB
  session. Every test here uses `db_session=None` or `MagicMock()`.

## Review Focus

1. **Category dtype across DuckDB chunks.** `pd.concat` of categoricals with different
   categories silently becomes `object`, about 1 GB on the 12.4M-row train read, which is the
   OOM the chunking exists to prevent. Task 1 pins the dtype after concat.
2. **Engineered-cache hit in predict.** No fetch runs, so `self.break_dates` is empty. The mask
   must fall back to the artifact's training calendar, not silently mask nothing (Task 4).
3. **Harnesses that call `prepare_targets` on a hand-built frame** (`scripts/archive/*`, many
   tests). They must keep the item-count rule and log a WARNING, not crash on a missing attribute
   (Task 3).
4. **Old artifacts.** A `meta.json` with `shrink_k_gbm` / `vol_rank_gbm` / `vol_rank_norm`, and
   stale `shrink_k_*` / `vol_rank_*` booster files on disk, must load and predict (Task 5).
5. **The 2025/2026 year boundary is a real break and must fire.** Every pre-2026 row is
   `<null>`-sourced; on 2026-01-01 none is, and the median paired item moves −4.15% (1,029 items
   ≥$1). The item-count rule catches it only on the full universe, and the 09-28 train-universe
   retrain did not log it. Within pre-2026 history, `<null>` → `<null>` must never fire (Task 2).

---

### Task 1: The vote emits `source_set`

**Files:**
- Modify: `backend/models/forecaster.py`
  - `_multi_source_voting_sql`, around :2677
  - `_apply_multi_source_voting`, around :2525
  - `_fetch_voted_price_history` chunk loop, around :2505
  - `VOTED_CACHE_VERSION` comment block, around :1049-1068
- Test: `backend/tests/test_sql_voting.py`

**Interfaces:**
- Produces: the voted frame gains a column `source_set` (pandas `category`, values like
  `"aggregator_buff163|aggregator_youpin"` or `"<null>"`) on the DuckDB, DB and pandas paths.
  The legacy branch with no `source` column does not emit it.
- Produces: module constant `NULL_SOURCE_SET_LABEL = "<null>"` in `models/forecaster.py`.

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_sql_voting.py`:

```python
def _canon_sets(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["item_id", "date", "source_set"]].copy()
    out["source_set"] = out["source_set"].astype(str)
    return out.sort_values(["item_id", "date"]).reset_index(drop=True)


@pytest.mark.parametrize("seed", range(8))
def test_source_set_matches_between_sql_and_pandas(seed):
    df = _random_rows(seed)
    expected = _canon_sets(ItemForecaster._apply_multi_source_voting(df.copy()))
    actual = _canon_sets(_sql_vote(df))
    pd.testing.assert_frame_equal(actual, expected)


def test_source_set_is_the_kept_sources_sorted_and_deduplicated():
    day = pd.Timestamp("2026-04-16").date()
    df = pd.DataFrame(
        [
            ("a", day, 10.0, 1.0, "aggregator_youpin"),
            ("a", day, 10.1, 1.0, "aggregator_buff163"),
            ("a", day, 10.2, 1.0, "aggregator_buff163"),  # duplicate source
            ("a", day, 10.3, 1.0, sorted(CONDITIONAL_STEAM_SOURCES)[0]),  # stands down: 2 other asks
            ("a", day, 9.0, 1.0, sorted(BID_SOURCES)[0]),  # bids never vote
            ("b", day, 5.0, 1.0, None),
        ],
        columns=["item_id", "date", "price", "volume", "source"],
    )
    for out in (ItemForecaster._apply_multi_source_voting(df.copy()), _sql_vote(df)):
        sets = dict(zip(out["item_id"], out["source_set"].astype(str)))
        assert sets == {"a": "aggregator_buff163|aggregator_youpin", "b": "<null>"}
        assert (out["source_set"].astype(str).str.count(r"\|") + 1 == out["n_ask_sources"]).all()


def test_pandas_vote_emits_category_dtype():
    out = ItemForecaster._apply_multi_source_voting(_random_rows(0))
    assert isinstance(out["source_set"].dtype, pd.CategoricalDtype)
```

- [ ] **Step 2: Run them and confirm they fail.**
  Run: `venv/bin/python -m pytest tests/test_sql_voting.py -q -k source_set`
  Expected: FAIL with `KeyError: 'source_set'`.

- [ ] **Step 3: Add the column to the SQL vote.** In `_multi_source_voting_sql`, change the `stats`
  CTE and the final SELECT:

```sql
            stats AS (
                SELECT item_id, date, COUNT(*) AS n, median(price) AS med,
                       stddev_pop(price) AS sd,
                       COUNT(DISTINCT COALESCE(source, '__null__')) AS n_ask_sources,
                       string_agg(DISTINCT COALESCE(source, '{NULL_SOURCE_SET_LABEL}'), '|'
                                  ORDER BY COALESCE(source, '{NULL_SOURCE_SET_LABEL}')) AS source_set,
                       COALESCE(SUM(volume), 0.0) AS volume
                FROM kept GROUP BY item_id, date
            )
            SELECT k.item_id, k.date,
                   median(k.price) FILTER (
                       WHERE s.n < 3 OR NOT (isfinite(s.sd) AND s.sd > 0)
                          OR abs(k.price - s.med) <= 2.0 * s.sd * (1 + {VOTE_TIE_RTOL})
                   ) AS price,
                   any_value(s.volume) AS volume,
                   any_value(s.n_ask_sources)::BIGINT AS n_ask_sources,
                   any_value(s.source_set) AS source_set
            FROM kept k JOIN stats s USING (item_id, date)
            GROUP BY k.item_id, k.date
```

  Add the module constant next to the other vote constants at the top of `models/forecaster.py`:

```python
#: How a NULL source is spelled inside `source_set`. The pre-2026 series is
#: NULL-sourced throughout, so every pre-2026 item-day carries this one value
#: and can never register as a composition change against itself.
NULL_SOURCE_SET_LABEL = "<null>"
```

- [ ] **Step 4: Add the column to the pandas reference.** In `_apply_multi_source_voting`:
  - Return an empty frame with `source_set` in its columns.
  - In the `multi_groups.empty` branch, set
    `out["source_set"] = out_source.fillna(NULL_SOURCE_SET_LABEL)` (capture
    `out_source = df["source"]` before the `drop`).
  - In the single-source fast path, aggregate `source_set=("source", "first")`, then `fillna`.
  - In `vote`, add:

```python
            source_set = "|".join(sorted(set(group["source"].fillna(NULL_SOURCE_SET_LABEL))))
```

  and return it in both `pd.Series` dicts. After the final concat:

```python
        result["source_set"] = result["source_set"].fillna(NULL_SOURCE_SET_LABEL).astype("category")
```

  Every empty-frame constructor in the method gains `"source_set"` in its `columns=` list.

- [ ] **Step 5: Keep the category through the chunked DuckDB read.** In `_fetch_voted_price_history`,
  replace the concat:

```python
                part["source_set"] = part["source_set"].astype("category")
                parts.append(part)
            # Categoricals with different categories concat to `object`: about
            # 1 GB of strings on the 12.4M-row train read, which is the memory
            # this chunking exists to save. Align the categories first.
            cats = sorted(set().union(*(p["source_set"].cat.categories for p in parts)))
            for p in parts:
                p["source_set"] = p["source_set"].cat.set_categories(cats)
            df = pd.concat(parts, ignore_index=True)
```

  Add a test in `tests/test_sql_voting.py` that calls `pd.concat` the way the code does on two
  frames with disjoint categories and asserts the result is categorical:

```python
def test_aligned_categories_survive_concat():
    a = pd.DataFrame({"source_set": pd.Categorical(["x"])})
    b = pd.DataFrame({"source_set": pd.Categorical(["y"])})
    cats = sorted(set().union(*(p["source_set"].cat.categories for p in (a, b))))
    for p in (a, b):
        p["source_set"] = p["source_set"].cat.set_categories(cats)
    assert isinstance(pd.concat([a, b], ignore_index=True)["source_set"].dtype, pd.CategoricalDtype)
```

- [ ] **Step 6: Bump the cache.** Set `VOTED_CACHE_VERSION = 12` and append to its comment block:

```python
    # v12: source_set on the voted frame -- the composition-break detector's
    # input. A v11 frame lacks the column, so a stale cache would silently
    # disable the detector and fall back to the item-count rule alone.
```

- [ ] **Step 7: Run the voting suite.**
  Run: `venv/bin/python -m pytest tests/test_sql_voting.py tests/test_forecaster.py -q -k "vot or source_set"`
  Expected: all PASS, including the existing `test_sql_vote_matches_pandas_reference`. `_canon`
  selects explicit columns, so the new column cannot break it.

- [ ] **Step 8: Commit.**

```bash
git add backend/models/forecaster.py backend/tests/test_sql_voting.py
git commit -m "feat: the vote emits each item-day's source set (voted cache v12)"
```

---

### Task 2: Composition-break detector

**Files:**
- Modify: `backend/models/forecaster.py`, next to `_collection_shift_dates`, around :5184
- Create: `backend/tests/test_composition_breaks.py`

**Interfaces:**
- Consumes: the `source_set` column from Task 1.
- Produces: `ItemForecaster.COMPOSITION_BREAK_FRACTION: float = 0.90` and
  `ItemForecaster._composition_break_dates(df: pd.DataFrame) -> frozenset[datetime.date]`
  (classmethod). Returns `frozenset()` when `source_set` is absent.

- [ ] **Step 1: Write the failing tests.** Create `tests/test_composition_breaks.py`:

```python
"""_composition_break_dates: a market-wide switch in which sources vote.

The item-count rule (_collection_shift_dates) cannot see 2026-04-16: every
train-universe item switched from buff163/csfloat/youpin to steam_17mafo with
the item count flat (+0.02%). See
docs/specs/2026-09-30-composition-break-calendar-design.md.
"""

from datetime import date, timedelta

import pandas as pd
from models.forecaster import ItemForecaster

D0 = date(2026, 4, 14)


def _frame(n_items: int, sets_by_day: list[list[str]]) -> pd.DataFrame:
    rows = []
    for k, sets in enumerate(sets_by_day):
        for i in range(n_items):
            rows.append({"item_id": f"i{i}", "date": D0 + timedelta(days=k), "price": 1.0, "source_set": sets[i % len(sets)]})
    df = pd.DataFrame(rows)
    df["source_set"] = df["source_set"].astype("category")
    return df


def test_full_switch_fires_on_the_switch_day_only():
    df = _frame(100, [["a|b"], ["a|b"], ["c"], ["c"]])
    assert ItemForecaster._composition_break_dates(df) == frozenset({D0 + timedelta(days=2)})


def test_ordinary_churn_does_not_fire():
    # ~56% of items change set: the highest ordinary day measured on the train universe.
    day1 = ["a|b"] * 100
    day2 = ["a"] * 56 + ["a|b"] * 44
    df = pd.concat([_frame(100, [day1]), _frame(100, [day1, day2]).iloc[100:]])
    assert ItemForecaster._composition_break_dates(df) == frozenset()


def test_ninety_percent_is_the_boundary():
    day1 = ["a"] * 100
    at = ["b"] * 90 + ["a"] * 10
    below = ["b"] * 89 + ["a"] * 11
    assert ItemForecaster._composition_break_dates(_frame(100, [day1, at])) == frozenset({D0 + timedelta(days=1)})
    assert ItemForecaster._composition_break_dates(_frame(100, [day1, below])) == frozenset()


def test_small_cross_section_never_fires():
    assert ItemForecaster._composition_break_dates(_frame(24, [["a"], ["b"]])) == frozenset()


def test_null_sourced_history_is_one_composition():
    df = _frame(100, [["<null>"], ["<null>"], ["<null>"]])
    assert ItemForecaster._composition_break_dates(df) == frozenset()


def test_new_items_do_not_count_as_changed():
    # Day 2 adds 1,000 items that did not exist on day 1; only the 100 paired items are judged.
    df = pd.concat([_frame(100, [["a"], ["a"]]), _frame(1100, [["a"], ["b"]]).iloc[1100:].assign(item_id=lambda d: "new" + d["item_id"])])
    assert ItemForecaster._composition_break_dates(df) == frozenset()


def test_the_null_to_labelled_year_boundary_fires():
    df = _frame(100, [["<null>"], ["<null>"], ["aggregator_sync"]])
    assert ItemForecaster._composition_break_dates(df) == frozenset({D0 + timedelta(days=2)})


def test_missing_column_is_empty_not_an_error():
    df = _frame(100, [["a"], ["b"]]).drop(columns=["source_set"])
    assert ItemForecaster._composition_break_dates(df) == frozenset()


def test_a_price_crash_with_a_stable_source_set_does_not_fire():
    df = _frame(100, [["a|b"], ["a|b"]])
    df.loc[df["date"] == D0 + timedelta(days=1), "price"] = 0.5
    assert ItemForecaster._composition_break_dates(df) == frozenset()
```

- [ ] **Step 2: Run them and confirm they fail.**
  Run: `venv/bin/python -m pytest tests/test_composition_breaks.py -q`
  Expected: FAIL with `AttributeError: ... '_composition_break_dates'`.

- [ ] **Step 3: Implement.** Below `_collection_shift_dates`:

```python
    #: Share of items, present on both d-1 and d, whose voting source set
    #: changed. Break days measured 0.9989-1.0000 (03-22, 04-16, 07-09/10/11);
    #: the highest ordinary day 0.56 on the train universe (Jan-Feb, where
    #: aggregator_sync steps in and out). 2026-09-30 probe, spec §Problem.
    COMPOSITION_BREAK_FRACTION = 0.90

    @classmethod
    def _composition_break_dates(cls, df: pd.DataFrame) -> frozenset:
        """Dates on which the whole cross-section's source set switched.

        The item-count rule misses a switch that keeps the count flat, which
        2026-04-16 did (+0.02% items, 100% of sets changed, median return
        -1.92%). Like that rule, this reads the universe, never prices, so a
        real crash cannot fire it. Only items observed on both days are judged;
        a new item is not a changed one.
        """
        if df.empty or not {"item_id", "date", "source_set"} <= set(df.columns):
            return frozenset()
        d = df[["item_id", "date", "source_set"]].drop_duplicates(["item_id", "date"]).copy()
        d["date"] = pd.to_datetime(d["date"])
        d["source_set"] = d["source_set"].astype("category")
        prev = d.copy()
        prev["date"] = prev["date"] + pd.to_timedelta(1, unit="D")
        m = d.merge(prev, on=["item_id", "date"], suffixes=("", "_prev"))
        if m.empty:
            return frozenset()
        m["changed"] = m["source_set"].cat.codes != m["source_set_prev"].cat.codes
        agg = m.groupby("date")["changed"].agg(["mean", "size"])
        hits = agg[(agg["mean"] >= cls.COMPOSITION_BREAK_FRACTION) & (agg["size"] >= cls.MIN_DEGENERATE_CROSS_SECTION)]
        return frozenset(ts.date() for ts in hits.index)
```

  The codes comparison is valid because both columns come from one categorical, so they share
  categories. Use `astype("category")` on `d` **before** copying it to `prev`.

- [ ] **Step 4: Run the tests.**
  Run: `venv/bin/python -m pytest tests/test_composition_breaks.py -q`
  Expected: 9 passed.

- [ ] **Step 5: Commit.**

```bash
git add backend/models/forecaster.py backend/tests/test_composition_breaks.py
git commit -m "feat: detect market-wide source-composition breaks"
```

---

### Task 3: The calendar reaches the label span rule and the artifact

**Files:**
- Modify: `backend/models/forecaster.py`
  - `__init__`, around :1227
  - `fetch_price_history`, :2308-2364
  - `prepare_targets`, :5290-5400
  - `save_models` meta dict, around :11733
  - `load_models`, around :11790
- Test: `backend/tests/test_composition_breaks.py`

**Interfaces:**
- Consumes: `_composition_break_dates` (Task 2), `_collection_shift_dates` (existing).
- Produces:
  - `self.break_dates: frozenset[date] | None`, where `None` means no fetch has run.
  - `self._record_break_dates(df) -> None`.
  - `meta["break_dates"]` and `meta["composition_break_dates"]` (sorted ISO strings).
  - `self._artifact_break_dates: frozenset[date]`, read from meta, empty when absent.
  - `label_voiding["composition_break_dates"]` and `label_voiding["break_dates"]`.

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_composition_breaks.py`:

```python
import json
import logging
from unittest.mock import MagicMock

import numpy as np


def _priced(n_items: int, days: int, switch_day: int | None) -> pd.DataFrame:
    rows = []
    for k in range(days):
        s = "c" if switch_day is not None and k >= switch_day else "a|b"
        for i in range(n_items):
            rows.append({"item_id": f"i{i}", "date": D0 + timedelta(days=k), "price": 10.0 + i + 0.01 * k, "volume": 1.0, "source_set": s})
    df = pd.DataFrame(rows)
    df["source_set"] = df["source_set"].astype("category")
    return df


def _fc(tmp_path):
    return ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))


def test_record_break_dates_is_the_union(tmp_path):
    fc = _fc(tmp_path)
    fc._record_break_dates(_priced(100, 6, switch_day=3))
    assert fc.break_dates == frozenset({D0 + timedelta(days=3)})


def test_label_spanning_a_composition_break_is_voided(tmp_path):
    fc = _fc(tmp_path)
    df = _priced(100, 10, switch_day=5)
    fc._record_break_dates(df)
    out = fc.prepare_targets(df.drop(columns=["source_set"]), horizon=3)
    anchor = pd.to_datetime(out["date"])
    b = pd.Timestamp(D0 + timedelta(days=5))
    spans = (anchor < b) & (b <= anchor + pd.Timedelta(days=3))
    assert spans.any()
    assert out.loc[spans, "target_return_3d"].isna().all()
    assert out.loc[~spans & out["target_3d"].notna(), "target_return_3d"].notna().all()
    assert fc.label_voiding["composition_break_dates"] == [str(D0 + timedelta(days=5))]


def test_without_a_fetch_prepare_targets_falls_back_and_warns(tmp_path, caplog):
    fc = _fc(tmp_path)
    assert fc.break_dates is None
    with caplog.at_level(logging.WARNING, logger="models.forecaster"):
        fc.prepare_targets(_priced(100, 10, switch_day=5).drop(columns=["source_set"]), horizon=3)
    assert "item-count rule only" in caplog.text


def test_break_dates_round_trip_through_meta_json(tmp_path):
    f = _fc(tmp_path)
    f.break_dates = frozenset({date(2026, 4, 16)})
    f.composition_break_dates = frozenset({date(2026, 4, 16)})
    f.feature_cols = ["a"]
    f.feature_medians = pd.Series({"a": 0.0})
    f.conformal_calibration = {h: 1.0 for h in f.HORIZONS}
    f.save_models()
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["break_dates"] == ["2026-04-16"]
    g = _fc(tmp_path)
    g.load_models()
    assert g._artifact_break_dates == frozenset({date(2026, 4, 16)})


def test_artifact_without_break_dates_loads_empty(tmp_path):
    f = _fc(tmp_path)
    f.feature_cols = ["a"]
    f.feature_medians = pd.Series({"a": 0.0})
    f.conformal_calibration = {h: 1.0 for h in f.HORIZONS}
    f.save_models()
    meta = json.loads((tmp_path / "meta.json").read_text())
    meta.pop("break_dates", None)
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    g = _fc(tmp_path)
    g.load_models()
    assert g._artifact_break_dates == frozenset()
```

- [ ] **Step 2: Run them and confirm they fail.**
  Run: `venv/bin/python -m pytest tests/test_composition_breaks.py -q`
  Expected: the five new tests FAIL (`_record_break_dates` missing).

- [ ] **Step 3: State on the instance.** In `__init__`, next to `self.label_voiding`:

```python
        # The break calendar for the frame this process fetched: the item-count
        # rule's cutovers plus the composition rule's switches, computed in
        # fetch_price_history on the VOTED frame -- engineer_features drops
        # source_set, so prepare_targets cannot compute it itself. None means
        # no fetch ran (a harness building its own frame).
        self.break_dates: frozenset | None = None
        self.composition_break_dates: frozenset = frozenset()
        self.collection_shift_dates: frozenset = frozenset()
        self._artifact_break_dates: frozenset = frozenset()
```

- [ ] **Step 4: Record in `fetch_price_history`.** Add the method:

```python
    def _record_break_dates(self, df: pd.DataFrame) -> None:
        """Store this frame's break calendar; see `self.break_dates`."""
        self.composition_break_dates = self._composition_break_dates(df)
        self.collection_shift_dates = self._collection_shift_dates(df)
        self.break_dates = self.collection_shift_dates | self.composition_break_dates
        logger.info(
            f"  Break calendar: {sorted(d.isoformat() for d in self.break_dates)} "
            f"(composition: {sorted(d.isoformat() for d in self.composition_break_dates)})"
        )
```

  Then call `self._record_break_dates(...)` on every return path of `fetch_price_history` that
  returns a voted frame: the cache hit (`cached`), the fresh Parquet frame (`df`) and the DB
  frame (`df`). Do not call it on the empty-DB early return.

- [ ] **Step 5: Use it in `prepare_targets`.** Replace `shifts = self._collection_shift_dates(df)`
  with:

```python
        if self.break_dates is None:
            shifts = self._collection_shift_dates(df)
            count_shifts = shifts
            logger.warning(
                    "  No break calendar on this forecaster (fetch_price_history did not "
                    "run); voiding on the item-count rule only -- composition switches "
                    "such as 2026-04-16 are NOT voided on this frame."
                )
        else:
            shifts = self.break_dates
            count_shifts = self.collection_shift_dates
```

  The warning's indentation follows `if`; its argument string is unchanged from above.

  In the `self.label_voiding = {...}` dict add:

```python
            "composition_break_dates": sorted(d.isoformat() for d in self.composition_break_dates),
            "break_dates": sorted(d.isoformat() for d in shifts),
```

  Set the existing `"collection_shift_dates"` entry to `sorted(d.isoformat() for d in count_shifts)`,
  so it keeps its meaning (the item-count rule alone). In the two log lines, change "collector
  cutover(s)" to "break day(s)".

- [ ] **Step 6: Persist.** In the `save_models` meta dict, next to `"label_voiding"`:

```python
            "break_dates": sorted(d.isoformat() for d in (self.break_dates or frozenset())),
            "composition_break_dates": sorted(d.isoformat() for d in self.composition_break_dates),
```

  In `load_models`, next to `_artifact_feature_native_nan`:

```python
        self._artifact_break_dates = frozenset(date.fromisoformat(s) for s in meta.get("break_dates", []))
```

  Check that `date` is imported at the top of the module; add `from datetime import date` if not.

- [ ] **Step 7: Run the tests and the existing label suites.**
  Run: `venv/bin/python -m pytest tests/test_composition_breaks.py tests/ -q -k "composition or label or void or target or cutover or snapshot"`
  Expected: all PASS. Existing tests that call `prepare_targets` without a fetch now log the
  WARNING and keep their old behaviour.

- [ ] **Step 8: Commit.**

```bash
git add backend/models/forecaster.py backend/tests/test_composition_breaks.py
git commit -m "feat: void labels across composition breaks; persist the break calendar"
```

---

### Task 4: Break-aware lookbacks (`BREAK_AWARE_LOOKBACKS`, default off)

**Files:**
- Create: `backend/models/lookback_windows.py`
- Modify: `backend/models/forecaster.py`
  - flag + accessor next to `feature_native_nan_enabled`, around :3290
  - `build_training_data`, after `self._base_feature_cols = list(self.feature_cols)`, around :5808
  - `_prepare_predict_features`, after the engineered-cache `if/else`, around :9955
  - meta write and read
- Create: `backend/tests/test_break_aware_lookbacks.py`

**Interfaces:**
- Consumes: `self.break_dates` and `self._artifact_break_dates` (Task 3).
- Produces:
  - `models.lookback_windows.FEATURE_LOOKBACK_DAYS: dict[str, int]`
  - `models.lookback_windows.mask_break_lookbacks(df, cols, breaks) -> int`, which mutates `df`
    and returns the number of cells set to NaN
  - `ItemForecaster.break_aware_lookbacks_enabled() -> bool` (static)
  - `ItemForecaster._break_aware_lookbacks_served() -> bool`
  - `meta["break_aware_lookbacks"]: bool`

- [ ] **Step 1: Write the failing tests.** Create `tests/test_break_aware_lookbacks.py`:

```python
"""BREAK_AWARE_LOOKBACKS: a feature whose lookback spans a source break is NaN."""

import datetime as dt
import inspect
import tempfile

import numpy as np
import pandas as pd
import pytest
from models.forecaster import ItemForecaster, _feature_group
from models.lookback_windows import FEATURE_LOOKBACK_DAYS, mask_break_lookbacks

from tests._source import method_closure_source

B = dt.date(2026, 7, 9)


def _rows(days):
    return pd.DataFrame(
        {
            "date": [B + dt.timedelta(days=k) for k in days],
            "return_1d": 1.0,
            "return_7d": 1.0,
            "price_dist_ma200": 1.0,
            "price_tier": 2,
        }
    )


def test_window_spanning_a_break_is_nan_and_others_are_untouched():
    df = _rows([-1, 0, 1, 6, 7, 199, 200])
    n = mask_break_lookbacks(df, ["return_1d", "return_7d", "price_dist_ma200", "price_tier"], frozenset({B}))
    by_day = df.set_index(df["date"].map(lambda d: (d - B).days))
    assert by_day.loc[[-1, 1], "return_1d"].notna().all() and np.isnan(by_day.loc[0, "return_1d"])
    assert by_day.loc[[0, 1, 6], "return_7d"].isna().all() and by_day.loc[[-1, 7], "return_7d"].notna().all()
    assert by_day.loc[[0, 199], "price_dist_ma200"].isna().all() and by_day.loc[[-1, 200], "price_dist_ma200"].notna().all()
    assert by_day["price_tier"].notna().all()  # window 0: a level, not a lookback
    assert n == 1 + 3 + 5


def test_no_breaks_is_a_no_op():
    df = _rows([0, 1])
    assert mask_break_lookbacks(df, ["return_7d"], frozenset()) == 0
    assert df["return_7d"].notna().all()


def test_unknown_column_raises():
    with pytest.raises(KeyError, match="FEATURE_LOOKBACK_DAYS"):
        mask_break_lookbacks(_rows([0]).assign(new_feat=1.0), ["new_feat"], frozenset({B}))


def test_table_covers_every_price_technical_engineer_features_emits():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(3):
        p = 10.0
        for k in range(400):
            p *= 1 + rng.normal(0, 0.02)
            ts = pd.Timestamp("2025-01-01") + pd.Timedelta(days=k)
            rows.append({"item_id": f"i{i}", "date": ts.date(), "timestamp": ts, "price": p, "volume": 5.0})
    fc = ItemForecaster(db_session=None, model_dir=tempfile.mkdtemp())
    out = fc.engineer_features(pd.DataFrame(rows), pd.DataFrame(columns=["id", "type", "timestamp", "description"]), skip_unused_groups=True)
    emitted = {c for c in out.columns if _feature_group(c) == "price_technicals"}
    assert emitted <= set(FEATURE_LOOKBACK_DAYS), sorted(emitted - set(FEATURE_LOOKBACK_DAYS))


def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("BREAK_AWARE_LOOKBACKS", raising=False)
    assert ItemForecaster.break_aware_lookbacks_enabled() is False


def test_serving_follows_the_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("BREAK_AWARE_LOOKBACKS", "1")
    fc = ItemForecaster(db_session=None, model_dir=str(tmp_path))
    fc._artifact_break_aware_lookbacks = False
    assert fc._break_aware_lookbacks_served() is False


def test_train_and_predict_both_apply_the_mask():
    assert "_apply_break_mask" in method_closure_source(ItemForecaster, "build_training_data")
    assert "_apply_break_mask" in method_closure_source(ItemForecaster, "_prepare_predict_features")


def test_predict_on_a_cache_hit_uses_the_artifact_calendar(tmp_path, monkeypatch):
    fc = ItemForecaster(db_session=None, model_dir=str(tmp_path))
    fc._artifact_break_aware_lookbacks = True
    fc._artifact_break_dates = frozenset({B})
    fc.break_dates = None  # cache hit: no fetch ran
    df = _rows([0, 1])
    fc._apply_break_mask(df, ["return_7d"], served=True)
    assert df["return_7d"].isna().all()
```

- [ ] **Step 2: Run them and confirm they fail.**
  Run: `venv/bin/python -m pytest tests/test_break_aware_lookbacks.py -q`
  Expected: FAIL with `ModuleNotFoundError: models.lookback_windows`.

- [ ] **Step 3: Create `models/lookback_windows.py`.**

```python
"""Calendar lookback, in days, of every price-technical feature.

A feature on row date d reads prices from (d - W, d]. If a source-composition
break b falls in that span, the feature compares two price bases and measures
the switch rather than the market; `mask_break_lookbacks` sets it to NaN.
W = 0 marks a level or a count that reads no earlier price.

EWM features have infinite memory; they get an effective window of 3 x span
(about 95% of the weight). MACD: line 3 x 26 = 78; the signal line adds
3 x 9 = 27 on top. Windows assume one row per calendar day, which the 2026
archive (the only era with breaks) satisfies.

`tests/test_break_aware_lookbacks.py` fails if engineer_features emits a
price-technical column that is not listed here.
"""

from __future__ import annotations

import pandas as pd

_N = (1, 3, 7, 14, 30, 60, 90, 120, 180)

FEATURE_LOOKBACK_DAYS: dict[str, int] = {
    **{f"price_lag_{n}d": n for n in _N},
    **{f"return_{n}d": n for n in _N},
    **{f"price_{s}_{n}d": n for s in ("mean", "std", "min", "max") for n in (7, 14, 20, 30, 60)},
    **{f"price_cv_{n}d": n for n in (7, 14, 20, 30, 60)},
    "price_zscore_30d": 30,
    "vol_regime_60_30": 60,
    "trend_divergence_30_60": 60,
    "price_accel_7d": 14,
    "price_log": 0,
    "price_tier": 0,
    "log_return_1d": 1,
    "log_return_7d": 7,
    "autocorr_1d": 2,
    "autocorr_7d": 14,
    **{f"bb_{s}": 20 for s in ("upper", "lower", "pct_b", "width")},
    "rsi_14": 14,
    "rsi_missing": 14,
    "macd_line": 78,
    "macd_line_rel": 78,
    "macd_signal": 105,
    "macd_histogram": 105,
    "macd_histogram_rel": 105,
    "macd_missing": 0,
    "price_mean_100d": 100,
    "price_dist_ma100": 100,
    "price_mean_200d": 200,
    "price_dist_ma200": 200,
    "trend_up_fraction_30d": 30,
}


def mask_break_lookbacks(df: pd.DataFrame, cols: list[str], breaks: frozenset) -> int:
    """NaN each `cols` cell whose lookback (d - W, d] contains a break. Returns the count."""
    if not breaks:
        return 0
    unknown = [c for c in cols if c not in FEATURE_LOOKBACK_DAYS]
    if unknown:
        raise KeyError(f"no FEATURE_LOOKBACK_DAYS entry for {unknown}; declare its window")
    day = pd.to_datetime(df["date"])
    b = pd.to_datetime(sorted(breaks))
    n = 0
    for c in cols:
        w = FEATURE_LOOKBACK_DAYS[c]
        if w == 0 or c not in df.columns:
            continue
        hit = pd.Series(False, index=df.index)
        for bk in b:
            hit |= (day >= bk) & (day < bk + pd.Timedelta(days=w))
        n += int((hit & df[c].notna()).sum())
        df.loc[hit, c] = float("nan")
    return n
```

  Check the boundary against the tests: `d − W < b ≤ d` is the same as `b ≤ d < b + W`. For
  `return_1d` (W=1) only `d = b` is masked; for `return_7d`, days b..b+6 are masked.

- [ ] **Step 4: Flag, accessor and mask helper in `forecaster.py`.** Next to
  `feature_native_nan_enabled`:

```python
    @staticmethod
    def break_aware_lookbacks_enabled() -> bool:
        """NaN a feature whose lookback spans a source-composition break.

        Off by default and unmeasured: with it on, return_90d+ and the 100/200d
        averages are NaN for every 2026 row after 2026-03-22 until about
        2027-01. Switched on only after the paired band-quality A/B in
        docs/specs/2026-09-30-composition-break-calendar-design.md (Part B gate).
        """
        return os.environ.get("BREAK_AWARE_LOOKBACKS") == "1"

    def _break_aware_lookbacks_served(self) -> bool:
        if self._artifact_break_aware_lookbacks is not None:
            return self._artifact_break_aware_lookbacks
        return self.break_aware_lookbacks_enabled()

    def _apply_break_mask(self, df: pd.DataFrame, cols: list[str], *, served: bool) -> None:
        """Apply mask_break_lookbacks when the arm is on (train: env; serve: artifact)."""
        on = self._break_aware_lookbacks_served() if served else self.break_aware_lookbacks_enabled()
        if not on:
            return
        # A predict on an engineered-cache hit never fetched, so it has no
        # calendar of its own; the artifact's training calendar is the fallback.
        breaks = (self.break_dates or frozenset()) | (self._artifact_break_dates if served else frozenset())
        # Every allowlisted feature is price_technicals today; the coverage test
        # guarantees each one has a declared window.
        n = mask_break_lookbacks(df, [c for c in cols if _feature_group(c) == "price_technicals"], breaks)
        logger.info(f"  BREAK_AWARE_LOOKBACKS: {n:,} feature cells NaN'd across {len(breaks)} break day(s)")
```

  Add
  `from models.lookback_windows import FEATURE_LOOKBACK_DAYS, mask_break_lookbacks` to the
  imports. In `__init__`: `self._artifact_break_aware_lookbacks: bool | None = None`.

- [ ] **Step 5: Hooks.**
  - In `build_training_data`, directly after `self._base_feature_cols = list(self.feature_cols)`:

```python
        # Before the cross-sectional rank transform, so a masked cell is ranked
        # as missing rather than ranked and then blanked.
        self._apply_break_mask(df, self.feature_cols, served=False)
```

  - In `_prepare_predict_features`, directly after the engineered-cache `if df is not None: … else: …`
    block ends (before the `tier_lead_col` check). Placing it after the cache write keeps the
    cache flag-independent:

```python
        self._apply_break_mask(df, self.feature_cols, served=True)
```

  - Meta write, next to `"feature_native_nan"` at :11621:
    `"break_aware_lookbacks": self.break_aware_lookbacks_enabled(),`
  - Meta read, next to `_artifact_feature_native_nan`:
    `self._artifact_break_aware_lookbacks = meta.get("break_aware_lookbacks")`

- [ ] **Step 6: Run the tests.**
  Run: `venv/bin/python -m pytest tests/test_break_aware_lookbacks.py -q`
  Expected: 8 passed.

- [ ] **Step 7: Commit.**

```bash
git add backend/models/lookback_windows.py backend/models/forecaster.py backend/tests/test_break_aware_lookbacks.py
git commit -m "feat: BREAK_AWARE_LOOKBACKS arm (default off) masks features spanning a source break"
```

---

### Task 5: Remove the `SHRINK_K_GBM` / vol-rank plumbing (next-steps item 16)

**Files:**
- Modify: `backend/models/forecaster.py`
  - attrs at :1127-1128 and :1171-1174
  - `vol_rank_gbm_enabled` :1689, `shrink_k_gbm_enabled` :1734, `_shrink_k_gbm_served` :1750,
    `_vol_rank_gbm_served` :1755
  - `_fit_vol_rank_model` :7947 through `_vol_rank_feature_frame`
  - `_compute_per_item_optimal_k` :9151 through `_build_climatology_table_adaptive`
  - the branch at :9373 and the branch at :9416
  - save at :11494-11518, meta at :11655-11657, load at :11862-11864 and :12046-12071
- Modify: `.github/workflows/price-forecast.yml:267-279` (delete the `SHRINK_K_GBM` key and its comment)
- Modify: `backend/scripts/archive/anomaly_band_modulator_ab.py` (gains `matched_width`)
- Modify: `backend/tests/test_anomaly_band_modulator_ab.py:144,160` (import from there)
- Delete: `backend/scripts/archive/shrink_k_vol_rank_ab.py`, `backend/scripts/archive/shrink_k_stability.py`,
  `backend/tests/test_shrink_k_vol_rank_ab.py`
- Test: `backend/tests/test_composition_breaks.py` (old-artifact load)

**Interfaces:**
- Produces: none. After this task, nothing in the repo names `shrink_k_gbm`, `vol_rank`,
  `SHRINK_K_GBM` or `VOLATILITY_RANK_GBM` except the stale-file cleanup in `save_models` and the
  dated docs.

- [ ] **Step 1: Write the failing test.** Append to `tests/test_composition_breaks.py`:

```python
def test_an_artifact_with_the_retired_shrink_k_and_vol_rank_keys_still_loads(tmp_path):
    f = _fc(tmp_path)
    f.feature_cols = ["a"]
    f.feature_medians = pd.Series({"a": 0.0})
    f.conformal_calibration = {h: 1.0 for h in f.HORIZONS}
    f.save_models()
    meta = json.loads((tmp_path / "meta.json").read_text())
    meta.update({"shrink_k_gbm": True, "vol_rank_gbm": True, "vol_rank_norm": {"3": 1.2}})
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "shrink_k_3d.txt").write_text("not a booster")
    (tmp_path / "vol_rank_3d_e0.txt").write_text("not a booster")
    g = _fc(tmp_path)
    g.load_models()  # must not raise, must not try to parse the stale files
    assert not hasattr(g, "shrink_k_models") and not hasattr(g, "vol_rank_models")


def test_save_removes_stale_retired_booster_files(tmp_path):
    (tmp_path / "shrink_k_7d.txt").write_text("x")
    (tmp_path / "vol_rank_7d_e1.txt").write_text("x")
    f = _fc(tmp_path)
    f.feature_cols = ["a"]
    f.feature_medians = pd.Series({"a": 0.0})
    f.conformal_calibration = {h: 1.0 for h in f.HORIZONS}
    f.save_models()
    assert not list(tmp_path.glob("shrink_k_*.txt")) and not list(tmp_path.glob("vol_rank_*.txt"))
```

- [ ] **Step 2: Run them and confirm they fail.**
  Run: `venv/bin/python -m pytest tests/test_composition_breaks.py -q -k "retired"`
  Expected: FAIL (`shrink_k_models` exists; the stale files survive the save).

- [ ] **Step 3: Delete the plumbing.**
  - Remove the four attrs, the four flag/accessor methods, the eight helper methods, the
    `_build_climatology_table_adaptive` branch (keep only the `else` body, dedented), the vol-rank
    multiplier block in `_climatology_scale_for_rows`, the three meta keys, the load lines and
    the load loops.
  - Replace the two save loops with:

```python
        # SHRINK_K_GBM and VOLATILITY_RANK_GBM were refuted (2026-09-10) and
        # removed (2026-10-05 retrain PR). Clear any boosters an older artifact
        # left behind so nothing on disk suggests they serve.
        for stale in [*Path(self.model_dir).glob("shrink_k_*d.txt"), *Path(self.model_dir).glob("vol_rank_*d_e*.txt")]:
            stale.unlink()
```

    Check that `Path` is imported in the module; it is used elsewhere, so confirm with
    `grep -n "^from pathlib" models/forecaster.py`.
  - Copy `matched_width` and its docstring verbatim from `shrink_k_vol_rank_ab.py:91-105` into
    `scripts/archive/anomaly_band_modulator_ab.py`, which already defines
    `TARGET_COVERAGE = 0.80`. Change the two test imports to
    `from scripts.archive.anomaly_band_modulator_ab import matched_width`.
  - Delete the three files listed above, and the `SHRINK_K_GBM` key and its comment block from
    `price-forecast.yml`.

- [ ] **Step 4: Prove nothing references them.**
  Run: `cd .. && git grep -nE "shrink_k_gbm|SHRINK_K_GBM|VOLATILITY_RANK_GBM|vol_rank|shrink_k_models|predict_shrink_k|_build_climatology_table_adaptive" -- backend .github ':!docs'`
  Expected: only the stale-file cleanup lines in `save_models`.
  `climatology_shrink_k` (the flat K=320, `test_climatology_shrink_k_persisted.py`) is a
  **different** thing and stays.

- [ ] **Step 5: Run the affected suites.**
  Run: `venv/bin/python -m pytest tests/test_composition_breaks.py tests/test_anomaly_band_modulator_ab.py tests/test_climatology_shrink_k_persisted.py tests/test_served_recalibration_wiring.py -q`
  Expected: all PASS.

- [ ] **Step 6: Commit.**

```bash
git add -A backend .github/workflows/price-forecast.yml
git commit -m "chore: remove the refuted SHRINK_K_GBM / vol-rank plumbing (next-steps 16)"
```

---

### Task 6: Full suite, real-archive reproduction, docs, PR

**Files:**
- Create: `docs/changelog/2026-09-30-composition-break-calendar.md`
- Modify:
  - `docs/research/2026-09-28-next-steps.md` (§6 composition-break bullet, item 16, calendar row for 10-05)
  - `.claude/rules/labels-and-embargo.md` (one bullet)

- [ ] **Step 1: Full suite.**
  Run: `venv/bin/python -m pytest tests -q -m "not slow"`
  Expected: 0 failures. Then `ruff check` on every changed `.py` file.
  Expected: `All checks passed!`

- [ ] **Step 2: Reproduce on the real archive** for the PR description. Fetch `prices-2026-0[1-9].parquet`
  and the pre-2026 yearly files from `RayanR000/cs2-oracle-data` into a scratch dir with
  `gh api -H "Accept: application/vnd.github.raw" repos/RayanR000/cs2-oracle-data/contents/price-archive/<file>`.
  Then run this from `backend/` with `PYTHONPATH=.`. It opens no DB session:

```python
import sys, tempfile, duckdb
from pathlib import Path
from db.archive import prices_relation
from models.item_parser import archive_universe_sql_filter
from models.forecaster import ItemForecaster
arch = Path(sys.argv[1])
fc = ItemForecaster(db_session=None, model_dir=tempfile.mkdtemp())
fc.archive_dir = arch
df = fc._fetch_voted_price_history(days_back=400, backfilled_only=True,
                                   backfilled_slugs=fc._archive_universe_slugs(exclude_iflow=True))
print(sorted(map(str, fc._composition_break_dates(df))))
print(sorted(map(str, fc._collection_shift_dates(df))))
```

  Expected: composition `['2026-01-01', '2026-03-22', '2026-04-16', '2026-07-09', '2026-07-10', '2026-07-11']`;
  item-count `['2026-03-22', '2026-07-09', '2026-07-10', '2026-07-11', '2026-07-12']`, as the 09-28
  retrain logged. The two new days are 01-01 and 04-16. `days_back` must reach 2025-12-01, and
  the scratch dir must hold `prices-2025.parquet` or the year boundary cannot appear.

- [ ] **Step 3: Docs.**
  - The changelog records:
    - the probe numbers (spec §Problem);
    - what Part A changes (04-16 voided) and what it doesn't (features, bands, serving);
    - that Part B ships off;
    - item 16's removal;
    - the open per-item-churn question.
  - The rule bullet: "Labels void across **two** detectors: `_collection_shift_dates`
    (item count) ∪ `_composition_break_dates` (≥90% of paired items change source set). The
    calendar is computed in `fetch_price_history` because `engineer_features` drops
    `source_set`; `prepare_targets` without a fetch WARNs and uses the item-count rule only."
  - Mark next-steps item 16 and the §6 bullet done, pointing at the changelog.
  - Add no `experiment_log.csv` row until the Part B A/B reads.

- [ ] **Step 4: Commit, push, open the PR.** The PR body carries the Step 2 output and the
  post-merge check from the spec ("Verification after merge"). **Do not merge before the user
  says so.** Merging triggers the retrain on the next scheduled run.

```bash
git add docs .claude/rules/labels-and-embargo.md
git commit -m "docs: composition-break calendar changelog, rule and next-steps"
git push -u origin HEAD
gh pr create --title "feat: composition-break calendar + remove SHRINK_K_GBM plumbing (10-05 retrain)" --body-file /tmp/pr-body.md
```
