# Deterministic Forecast Backtest Implementation Plan

> # ✅ EXECUTED AND CLOSED (2026-08-01)
>
> All 9 tasks landed. Evidence: the `backend/backtest/` package
> (`price_resolution.py`, `scoring.py`, `resolution_gate.py`), migration
> `0019_freeze_forecast_outcome_actuals.py` (`528b764`), commits `ab2bcd9`, `dd115e2`, and the
> determinism test at `backend/tests/test_backtest_scoring.py:1204`. Ledger:
> `.superpowers/sdd/2026-08-01-deterministic-backtest/progress.md`.
>
> **Outcome and the prod backfill: `docs/changelog/2026-08-01-deterministic-backtest.md`** —
> 60,737 outcomes rewritten, 0.4% unresolvable.
>
> The boxes below are ticked retroactively. Nothing here is outstanding.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the live forecast accuracy metric reproducible by scoring both legs of `actual_ret` with one shared price estimator, freezing resolved actuals, and reporting per price tier.

**Architecture:** Split `backtest_accuracy.py`'s single resolve-score-store loop into a new `backend/backtest/` package with an impure resolution stage (reads the Parquet archive, insert-only) and a pure scoring stage (frozen rows → metrics, no I/O). The determinism guarantee lives at that boundary: once resolution has run, the reported number is a pure function of stored data.

**Tech Stack:** Python 3.11, pandas, DuckDB (archive reads), SQLAlchemy + Alembic, pytest. No new dependencies.

**Spec:** `docs/specs/2026-08-01-deterministic-backtest-design.md`

## Global Constraints

- **Both legs of `actual_ret` must call the identical estimator function.** This is the whole point of the change; any code path that computes one leg differently from the other is a defect.
- **Estimator:** median of the last `3` observed voted daily prices at or before the anchor date. Row-based lookback (mirrors `tail(3)` at `forecaster.py:3513`), not a calendar window.
- **Staleness cap:** `MAX_WINDOW_SPAN_DAYS = 7`. Matches `FALLBACK_MAX_AGE_DAYS = 7` in `collectors/pipeline.py` — the codebase gets one staleness convention, not two.
- **Voting:** always via `ItemForecaster._apply_multi_source_voting`. Never re-implement it.
- **`FLAT_TOLERANCE = 0.005`** — unchanged from the current `backtest_accuracy.py:36`.
- **Price tiers** come from the existing `_price_tier` boundaries: `0` is `<$1`, `1` is `$1–5`, `2` is `$5–20`, `3` is `$20–100`, `4` is `>=$100`.
- **Resolution is insert-only.** A `forecast_id` that already has a row is never re-resolved unless `--reresolve` is passed.
- **`item_forecasts.current_price` is never read for scoring** after Task 4. It stays in the schema and stays written by the forecaster; the backtest just stops depending on it.
- **No new dependencies.** DuckDB, pandas, numpy, SQLAlchemy are already in `backend/requirements.txt`.
- Run tests from `backend/`: `python -m pytest tests/ -q`.

## File Structure

| File | Responsibility |
|---|---|
| `backend/backtest/__init__.py` | Package marker. Empty. |
| `backend/backtest/price_resolution.py` | The shared estimator + archive loading. `smoothed_prices()` is pure over an already-voted frame; `load_voted_prices()` is the impure DuckDB read. |
| `backend/backtest/scoring.py` | Pure scoring. `direction_from_return`, `price_tier`, `score_cohort`, `score_by_tier`. No I/O, no clock, no archive. |
| `backend/scripts/backtest_accuracy.py` | CLI entry and orchestration only. Loses its metric math to `scoring.py` and its price loading to `price_resolution.py`. |
| `backend/database.py:262-295` | `ForecastOutcome` gains `base_price`, `resolved_at`. `PredictionAccuracy` gains `price_tier`. |
| `backend/migrations/versions/0019_freeze_forecast_outcome_actuals.py` | The schema migration. |
| `backend/tests/test_backtest_resolution.py` | Estimator symmetry, staleness cap, window semantics. |
| `backend/tests/test_backtest_scoring.py` | Purity, tier partition, determinism under archive revision, freeze invariance. |

---

### Task 1: Schema — freeze columns and the tier dimension

**Files:**
- Create: `backend/migrations/versions/0019_freeze_forecast_outcome_actuals.py`
- Modify: `backend/database.py:262-295` (`ForecastOutcome`), `backend/database.py:230-260` (`PredictionAccuracy`)
- Test: `backend/tests/test_backtest_scoring.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ForecastOutcome.base_price: float | None`, `ForecastOutcome.resolved_at: datetime | None`, `PredictionAccuracy.price_tier: int | None`. Every later task reads these names.

- [x] **Step 1: Write the failing test**

Create `backend/tests/test_backtest_scoring.py`:

```python
from __future__ import annotations

from datetime import date, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database import Base, ForecastOutcome, PredictionAccuracy


@pytest.fixture()
def session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine)()
    engine.dispose()


def test_forecast_outcome_has_freeze_columns(session):
    row = ForecastOutcome(
        forecast_id=1,
        item_id=1,
        forecast_date=date(2026, 7, 1),
        horizon_days=3,
        target_date=date(2026, 7, 4),
        current_price=1.0,
        base_price=1.05,
        predicted_price_mid=1.1,
        actual_price=1.2,
        direction_correct=1,
        abs_error=0.1,
        resolved_at=datetime(2026, 7, 4, 9, 0, 0),
    )
    session.add(row)
    session.commit()

    stored = session.query(ForecastOutcome).one()
    assert stored.base_price == 1.05
    assert stored.resolved_at == datetime(2026, 7, 4, 9, 0, 0)


def test_prediction_accuracy_has_price_tier(session):
    row = PredictionAccuracy(
        prediction_type="forecast",
        evaluation_date=date(2026, 8, 1),
        horizon_days=3,
        model_version="lgbm-v3-regime",
        price_tier=1,
        sample_count=10,
        metrics={"directional_accuracy": 55.0},
        created_at=datetime(2026, 8, 1, 9, 0, 0),
    )
    session.add(row)
    session.commit()

    assert session.query(PredictionAccuracy).one().price_tier == 1
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_backtest_scoring.py -v`
Expected: FAIL with `TypeError: 'base_price' is an invalid keyword argument for ForecastOutcome`

- [x] **Step 3: Add the ORM columns**

In `backend/database.py`, inside `class ForecastOutcome`, after the `actual_price` column (line 281):

```python
    # The forecast-time leg of actual_ret, resolved by the backtest with the
    # same estimator as actual_price. Distinct from current_price, which is
    # whatever the serving run happened to write and is no longer scored on.
    base_price = Column(Float, nullable=True)
```

And after the `evaluated_at` column (line 289):

```python
    # Set once, when the outcome is first resolved. base_price and
    # actual_price are frozen from that moment; only --reresolve moves them.
    resolved_at = Column(DateTime, nullable=True)
```

In `class PredictionAccuracy`, after the `horizon_days` column (line 248):

```python
    # Price tier of the cohort (0 = <$1 ... 4 = >=$100), or NULL for the
    # all-tiers aggregate row.
    price_tier = Column(Integer, nullable=True)
```

- [x] **Step 4: Write the migration**

Create `backend/migrations/versions/0019_freeze_forecast_outcome_actuals.py`:

```python
"""Add base_price/resolved_at to forecast_outcomes, price_tier to prediction_accuracy.

Revision ID: 0019_freeze_forecast_outcome_actuals
Revises: 0018_add_social_mentions
Create Date: 2026-08-01
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0019_freeze_forecast_outcome_actuals"
down_revision = "0018_add_social_mentions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    outcome_cols = {c["name"] for c in inspector.get_columns("forecast_outcomes")}
    if "base_price" not in outcome_cols:
        op.add_column("forecast_outcomes", sa.Column("base_price", sa.Float(), nullable=True))
    if "resolved_at" not in outcome_cols:
        op.add_column("forecast_outcomes", sa.Column("resolved_at", sa.DateTime(), nullable=True))

    accuracy_cols = {c["name"] for c in inspector.get_columns("prediction_accuracy")}
    if "price_tier" not in accuracy_cols:
        op.add_column("prediction_accuracy", sa.Column("price_tier", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("prediction_accuracy", "price_tier")
    op.drop_column("forecast_outcomes", "resolved_at")
    op.drop_column("forecast_outcomes", "base_price")
```

- [x] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_backtest_scoring.py -v`
Expected: PASS (2 passed)

- [x] **Step 6: Commit**

```bash
git add backend/database.py backend/migrations/versions/0019_freeze_forecast_outcome_actuals.py backend/tests/test_backtest_scoring.py
git commit -m "feat: schema for frozen backtest actuals and tiered accuracy"
```

---

### Task 2: The shared estimator

This is the fix. Everything else is plumbing around it.

**Files:**
- Create: `backend/backtest/__init__.py`, `backend/backtest/price_resolution.py`
- Test: `backend/tests/test_backtest_resolution.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `SMOOTH_WINDOW: int = 3`
  - `MAX_WINDOW_SPAN_DAYS: int = 7`
  - `smoothed_prices(voted: pd.DataFrame, anchors: set[tuple[str, date]], window: int = SMOOTH_WINDOW, max_span_days: int = MAX_WINDOW_SPAN_DAYS) -> dict[tuple[str, date], float]`
    - `voted` has columns `item_id` (slug string), `date` (`datetime.date`), `price` (float), already voted and one row per item-day.
    - Returns only resolvable anchors. An anchor absent from the result means "unresolvable", which callers must treat as a dropped forecast.

- [x] **Step 1: Write the failing tests**

Create `backend/tests/test_backtest_resolution.py`:

```python
from __future__ import annotations

from datetime import date

import pandas as pd

from backtest.price_resolution import (
    MAX_WINDOW_SPAN_DAYS,
    SMOOTH_WINDOW,
    smoothed_prices,
)


def _frame(rows):
    return pd.DataFrame(rows, columns=["item_id", "date", "price"])


def test_flat_series_gives_exactly_the_flat_price():
    """The symmetry property: a flat series must resolve to its own value on
    both legs, so actual_ret is exactly 0.0. The old code could not do this."""
    rows = [("ak", date(2026, 7, d), 2.0) for d in range(1, 11)]
    out = smoothed_prices(_frame(rows), {("ak", date(2026, 7, 4)), ("ak", date(2026, 7, 10))})

    base = out[("ak", date(2026, 7, 4))]
    actual = out[("ak", date(2026, 7, 10))]
    assert base == 2.0
    assert actual == 2.0
    assert (actual - base) / base == 0.0


def test_uses_median_of_last_three_observations_at_or_before_anchor():
    rows = [
        ("ak", date(2026, 7, 1), 1.0),
        ("ak", date(2026, 7, 2), 10.0),   # spike
        ("ak", date(2026, 7, 3), 2.0),
        ("ak", date(2026, 7, 4), 3.0),    # after the anchor — must be ignored
    ]
    out = smoothed_prices(_frame(rows), {("ak", date(2026, 7, 3))})
    # median(1.0, 10.0, 2.0) == 2.0 — the spike is filtered, 07-04 excluded
    assert out[("ak", date(2026, 7, 3))] == 2.0


def test_lookback_is_row_based_not_calendar_based():
    """Observations on 07-01, 07-05, 07-09 are the last 3 rows even though
    they span 8 calendar days — but that exceeds the staleness cap."""
    rows = [
        ("ak", date(2026, 7, 1), 1.0),
        ("ak", date(2026, 7, 5), 2.0),
        ("ak", date(2026, 7, 9), 3.0),
    ]
    out = smoothed_prices(_frame(rows), {("ak", date(2026, 7, 9))})
    assert ("ak", date(2026, 7, 9)) not in out  # span 8d > 7d cap


def test_staleness_cap_rejects_scattered_observations():
    rows = [
        ("ak", date(2026, 5, 1), 1.0),
        ("ak", date(2026, 6, 1), 2.0),
        ("ak", date(2026, 7, 1), 3.0),
    ]
    out = smoothed_prices(_frame(rows), {("ak", date(2026, 7, 1))})
    assert out == {}


def test_fewer_than_three_observations_resolve_within_the_span():
    rows = [
        ("ak", date(2026, 7, 8), 4.0),
        ("ak", date(2026, 7, 9), 6.0),
    ]
    out = smoothed_prices(_frame(rows), {("ak", date(2026, 7, 9))})
    assert out[("ak", date(2026, 7, 9))] == 5.0  # median(4.0, 6.0)


def test_anchor_before_any_observation_is_unresolvable():
    rows = [("ak", date(2026, 7, 9), 4.0)]
    out = smoothed_prices(_frame(rows), {("ak", date(2026, 7, 1))})
    assert out == {}


def test_items_are_independent():
    rows = [
        ("ak", date(2026, 7, 1), 1.0),
        ("ak", date(2026, 7, 2), 1.0),
        ("m4", date(2026, 7, 1), 50.0),
        ("m4", date(2026, 7, 2), 50.0),
    ]
    out = smoothed_prices(_frame(rows), {("ak", date(2026, 7, 2)), ("m4", date(2026, 7, 2))})
    assert out[("ak", date(2026, 7, 2))] == 1.0
    assert out[("m4", date(2026, 7, 2))] == 50.0


def test_constants_match_the_codebase_staleness_convention():
    from collectors.pipeline import FALLBACK_MAX_AGE_DAYS

    assert SMOOTH_WINDOW == 3
    assert MAX_WINDOW_SPAN_DAYS == FALLBACK_MAX_AGE_DAYS
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_backtest_resolution.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtest'`

- [x] **Step 3: Create the package and implement the estimator**

Create `backend/backtest/__init__.py` as an empty file.

Create `backend/backtest/price_resolution.py`:

```python
"""Shared price estimator for the forecast backtest.

Both legs of ``actual_ret`` — the forecast-date base and the target-date
actual — go through :func:`smoothed_prices`. That is the entire determinism
guarantee: the same function, the same window, the same source on both sides.

Before 2026-08-01 the base leg was ``item_forecasts.current_price`` (a
3-observation median written at serving time) and the actual leg was a raw
single-day voted price read fresh from the archive on every run. Differencing
two different estimators against a 0.5% flat band is what let the same 5,512
forecasts score 61.76% one day and 33.74% the next.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

# Mirrors ItemForecaster.predict()'s tail(3) median (forecaster.py:3513).
SMOOTH_WINDOW = 3

# The 3 observations must lie within this many calendar days of each other.
# Matches collectors.pipeline.FALLBACK_MAX_AGE_DAYS so the codebase has one
# staleness convention. Measured cost: ~0.74% of item-days.
MAX_WINDOW_SPAN_DAYS = 7


def smoothed_prices(
    voted: pd.DataFrame,
    anchors: set[tuple[str, date]],
    window: int = SMOOTH_WINDOW,
    max_span_days: int = MAX_WINDOW_SPAN_DAYS,
) -> dict[tuple[str, date], float]:
    """Median of the last ``window`` observed prices at or before each anchor.

    ``voted`` must already be voted to one row per item-day, with columns
    ``item_id`` (slug), ``date``, ``price``.

    Anchors that cannot be resolved — no observation at or before the anchor,
    or observations spanning more than ``max_span_days`` — are omitted from the
    result. Callers must treat a missing key as a dropped forecast rather than
    substituting a fallback, which would reintroduce the asymmetry this
    function exists to remove.
    """
    if voted.empty or not anchors:
        return {}

    by_item: dict[str, list[tuple[date, float]]] = {}
    for slug, group in voted.groupby("item_id", sort=False):
        ordered = group.sort_values("date")
        by_item[slug] = list(zip(ordered["date"], ordered["price"]))

    resolved: dict[tuple[str, date], float] = {}
    for slug, anchor in anchors:
        observations = by_item.get(slug)
        if not observations:
            continue

        # Last `window` observations at or before the anchor.
        selected = [(d, p) for d, p in observations if d <= anchor][-window:]
        if not selected:
            continue

        span = (selected[-1][0] - selected[0][0]).days
        if span > max_span_days:
            continue

        prices = sorted(p for _, p in selected)
        mid = len(prices) // 2
        if len(prices) % 2:
            resolved[(slug, anchor)] = float(prices[mid])
        else:
            resolved[(slug, anchor)] = float((prices[mid - 1] + prices[mid]) / 2)

    return resolved
```

- [x] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_backtest_resolution.py -v`
Expected: PASS (8 passed)

- [x] **Step 5: Mutation-check the staleness cap**

Temporarily change `MAX_WINDOW_SPAN_DAYS` to `9999` and re-run.
Expected: `test_lookback_is_row_based_not_calendar_based` and `test_staleness_cap_rejects_scattered_observations` both FAIL.
Revert to `7` and confirm they pass again. A cap whose tests still pass when the cap is disabled is not testing the cap.

- [x] **Step 6: Commit**

```bash
git add backend/backtest/ backend/tests/test_backtest_resolution.py
git commit -m "feat: shared 3-observation price estimator for both backtest legs"
```

---

### Task 3: Archive loading for both legs

**Files:**
- Modify: `backend/backtest/price_resolution.py`
- Test: `backend/tests/test_backtest_resolution.py`

**Interfaces:**
- Consumes: `smoothed_prices` from Task 2.
- Produces: `load_voted_prices(archive_dir: Path, slugs: list[str], min_date: date, max_date: date) -> pd.DataFrame` returning columns `item_id`, `date`, `price`. Raises `FileNotFoundError` when `archive_dir` does not exist.

- [x] **Step 1: Write the failing test**

Append to `backend/tests/test_backtest_resolution.py`:

```python
from pathlib import Path

import pytest

from backtest.price_resolution import load_voted_prices


def test_missing_archive_raises_rather_than_returning_empty(tmp_path):
    """A green run with zero actuals is the failure shape 324cfff removed.
    Resolution must fail loudly instead."""
    with pytest.raises(FileNotFoundError, match="price archive"):
        load_voted_prices(tmp_path / "nope", ["ak"], date(2026, 7, 1), date(2026, 7, 9))


def test_loads_and_votes_multi_source_rows(tmp_path):
    archive = tmp_path / "price-archive"
    archive.mkdir()
    pd.DataFrame(
        {
            "item_slug": ["ak", "ak", "ak"],
            "day": pd.to_datetime([date(2026, 7, 1)] * 3),
            "mean_price": [2.0, 2.1, 90.0],  # 90.0 is the outlier source
            "volume": [10, 10, 10],
            "source": ["a", "b", "c"],
        }
    ).to_parquet(archive / "prices-2026.parquet")

    out = load_voted_prices(archive, ["ak"], date(2026, 7, 1), date(2026, 7, 1))

    assert list(out.columns) == ["item_id", "date", "price"]
    assert len(out) == 1  # one row per item-day after voting
    assert out.iloc[0]["price"] < 10.0  # the 90.0 source was voted out


def test_window_dates_before_the_range_are_loaded(tmp_path):
    """Resolving an anchor needs the observations *before* it, so the loader
    must reach back past min_date by the staleness cap."""
    archive = tmp_path / "price-archive"
    archive.mkdir()
    days = pd.to_datetime([date(2026, 6, 28), date(2026, 6, 29), date(2026, 7, 1)])
    pd.DataFrame(
        {
            "item_slug": ["ak"] * 3,
            "day": days,
            "mean_price": [1.0, 1.0, 1.0],
            "volume": [1, 1, 1],
            "source": ["a", "a", "a"],
        }
    ).to_parquet(archive / "prices-2026.parquet")

    out = load_voted_prices(archive, ["ak"], date(2026, 7, 1), date(2026, 7, 1))
    assert len(out) == 3
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_backtest_resolution.py -k load -v`
Expected: FAIL with `ImportError: cannot import name 'load_voted_prices'`

- [x] **Step 3: Implement the loader**

Append to `backend/backtest/price_resolution.py`:

```python
def load_voted_prices(
    archive_dir: Path,
    slugs: list[str],
    min_date: date,
    max_date: date,
    max_span_days: int = MAX_WINDOW_SPAN_DAYS,
) -> pd.DataFrame:
    """Load voted daily prices from the Parquet archive.

    Reaches back ``max_span_days`` before ``min_date``: resolving an anchor
    needs the observations preceding it, not just the anchor's own day.

    Raises FileNotFoundError when the archive is absent. The previous
    behaviour — warn and return {} — produced a green run that evaluated zero
    forecasts, which is exactly the silent-success shape commit 324cfff was
    written to eliminate.
    """
    import duckdb
    from models.forecaster import ItemForecaster

    archive_dir = Path(archive_dir)
    if not archive_dir.exists():
        raise FileNotFoundError(f"price archive not found at {archive_dir}")

    pq_files = sorted(str(p) for p in archive_dir.glob("prices-*.parquet"))
    if not pq_files:
        raise FileNotFoundError(f"price archive at {archive_dir} contains no prices-*.parquet")

    if not slugs:
        return pd.DataFrame(columns=["item_id", "date", "price"])

    lookback_start = min_date - pd.Timedelta(days=max_span_days)

    con = duckdb.connect()
    try:
        selects = []
        for pqf in pq_files:
            cols = {r[0] for r in con.sql(f"DESCRIBE SELECT * FROM read_parquet('{pqf}')").fetchall()}
            source_expr = "source" if "source" in cols else "NULL::VARCHAR AS source"
            selects.append(
                f"SELECT item_slug, CAST(day AS DATE) AS day, mean_price AS price, "
                f"{source_expr}, volume FROM read_parquet('{pqf}')"
            )
        union_sql = " UNION ALL BY NAME ".join(selects)

        con.register("wanted_slugs", pd.DataFrame({"item_slug": slugs}))
        rows = con.sql(
            f"""
            SELECT s.item_slug, s.day, s.price, s.source, s.volume
            FROM ({union_sql}) s
            JOIN wanted_slugs w ON w.item_slug = s.item_slug
            WHERE s.day BETWEEN DATE '{lookback_start.date()}' AND DATE '{max_date}'
            """
        ).fetchall()
    finally:
        con.close()

    if not rows:
        return pd.DataFrame(columns=["item_id", "date", "price"])

    df = pd.DataFrame(rows, columns=["item_id", "timestamp", "price", "source", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])

    df = ItemForecaster._apply_multi_source_voting(df)

    return df[["item_id", "date", "price"]].reset_index(drop=True)
```

Note the `JOIN` against a registered frame rather than the current
f-string `IN (...)` list at `backtest_accuracy.py:160-167` — that built a
literal containing every slug, with manual quote-escaping.

- [x] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_backtest_resolution.py -v`
Expected: PASS (11 passed)

- [x] **Step 5: Commit**

```bash
git add backend/backtest/price_resolution.py backend/tests/test_backtest_resolution.py
git commit -m "feat: archive loader for backtest price resolution"
```

---

### Task 4: Extract the pure scorer

Moves the metric math out of the resolve loop with **no metric changes** — same numbers, new location — so that Task 6's tiering has something testable to build on.

**Files:**
- Create: `backend/backtest/scoring.py`
- Modify: `backend/scripts/backtest_accuracy.py:36-60` (delete `FLAT_TOLERANCE`, `_direction_from_return`, `_price_tier`, re-import them), `backend/scripts/backtest_accuracy.py:379-500` (delete the metric block, call `score_cohort`)
- Test: `backend/tests/test_backtest_scoring.py`

**Interfaces:**
- Consumes: nothing from Tasks 2–3.
- Produces:
  - `FLAT_TOLERANCE: float = 0.005`
  - `direction_from_return(ret: float) -> str` — `"up"` / `"down"` / `"flat"`
  - `price_tier(price: float) -> int` — `0`–`4`
  - `score_cohort(records: list[dict]) -> tuple[dict, int]` — returns `(metrics, sample_count)`. Each record must carry keys: `abs_error`, `pct_error`, `sq_error`, `direction_correct`, `predicted_direction`, `actual_direction`, `in_interval`, `confidence`, `base_price`, `actual_price`, `price_tier`, `item_id`.

**One deliberate non-change:** `pct_error` stays `abs(mid - actual) / base * 100`, i.e. divided by the *base* leg rather than by `actual`. That is unusual for a MAPE, but it is what the current code does (`backtest_accuracy.py:340`, using `current`). Changing it would move MAPE for reasons unrelated to determinism. Flag it in the changelog as a known oddity; do not fix it here.

- [x] **Step 1: Write the failing test**

Append to `backend/tests/test_backtest_scoring.py`:

```python
from backtest.scoring import (
    FLAT_TOLERANCE,
    direction_from_return,
    price_tier,
    score_cohort,
)


def _record(**overrides):
    base = {
        "abs_error": 0.10,
        "pct_error": 10.0,
        "sq_error": 0.01,
        "direction_correct": 1,
        "predicted_direction": "up",
        "actual_direction": "up",
        "in_interval": 1,
        "confidence": "high",
        "base_price": 1.00,
        "actual_price": 1.10,
        "price_tier": 1,
        "item_id": 1,
    }
    base.update(overrides)
    return base


def test_direction_from_return_respects_flat_tolerance():
    assert direction_from_return(FLAT_TOLERANCE * 2) == "up"
    assert direction_from_return(-FLAT_TOLERANCE * 2) == "down"
    assert direction_from_return(0.0) == "flat"
    assert direction_from_return(FLAT_TOLERANCE) == "flat"  # boundary is inclusive-flat


def test_price_tier_boundaries():
    assert price_tier(0.99) == 0
    assert price_tier(1.0) == 1
    assert price_tier(5.0) == 2
    assert price_tier(20.0) == 3
    assert price_tier(100.0) == 4


def test_score_cohort_is_pure_and_repeatable():
    records = [_record(item_id=i) for i in range(20)]
    first, n_first = score_cohort(records)
    second, n_second = score_cohort(records)
    assert first == second
    assert n_first == n_second == 20


def test_score_cohort_does_not_mutate_its_input():
    records = [_record(item_id=i) for i in range(20)]
    snapshot = [dict(r) for r in records]
    score_cohort(records)
    assert records == snapshot


def test_score_cohort_computes_directional_accuracy():
    records = [_record(direction_correct=1) for _ in range(6)]
    records += [_record(direction_correct=0, predicted_direction="down") for _ in range(4)]
    metrics, n = score_cohort(records)
    assert n == 10
    assert metrics["directional_accuracy"] == 60.0


def test_score_cohort_uses_base_price_for_the_persistence_baseline():
    """baseline_mae is |base - actual|, the error a predict-no-change model
    would make. It must read base_price, not the retired current_price."""
    records = [_record(base_price=1.00, actual_price=1.50) for _ in range(10)]
    metrics, _ = score_cohort(records)
    assert metrics["baseline_mae"] == 0.5
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_backtest_scoring.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtest.scoring'`

- [x] **Step 3: Create the pure scorer**

Create `backend/backtest/scoring.py`. Move the bodies verbatim from
`backtest_accuracy.py` — lines 36-40 (constants), 43-48
(`_direction_from_return`), 51-60 (`_price_tier`), 63-76 (`_bootstrap_ci`),
and 379-489 (the metric block) — with `r["current_price"]` renamed to
`r["base_price"]` at the `baseline_mae` line:

```python
"""Pure scoring for the forecast backtest.

No I/O, no archive access, no clock. Given frozen outcome records, produces
metrics. Keeping this pure is what makes the reported accuracy reproducible:
after resolution has run, the number is a function of stored data only.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

FLAT_TOLERANCE = 0.005
N_BOOTSTRAP = 1000
BOOTSTRAP_CI = 95
BOOTSTRAP_RNG_SEED = 42
CONFIDENCE_TARGET_ACCURACY = 80.0


def direction_from_return(ret: float) -> str:
    if ret > FLAT_TOLERANCE:
        return "up"
    if ret < -FLAT_TOLERANCE:
        return "down"
    return "flat"


def price_tier(price: float) -> int:
    if price >= 100:
        return 4
    if price >= 20:
        return 3
    if price >= 5:
        return 2
    if price >= 1:
        return 1
    return 0


def bootstrap_ci(values, n_resamples=N_BOOTSTRAP, ci=BOOTSTRAP_CI):
    if len(values) < 10:
        return None, None
    rng = np.random.default_rng(BOOTSTRAP_RNG_SEED)
    stats = np.empty(n_resamples)
    n = len(values)
    arr = np.array(values)
    for i in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        stats[i] = np.mean(sample)
    alpha = (100 - ci) / 2
    return (
        round(float(np.percentile(stats, alpha)), 4),
        round(float(np.percentile(stats, 100 - alpha)), 4),
    )


def score_cohort(records: list[dict]) -> tuple[dict, int]:
    """Metrics for one (horizon, model_version, tier) cohort.

    Returns (metrics, sample_count). Does not mutate ``records``.
    """
    n = len(records)
    if n == 0:
        return {}, 0

    mae = sum(r["abs_error"] for r in records) / n
    rmse = math.sqrt(sum(r["sq_error"] for r in records) / n)
    mape = sum(r["pct_error"] for r in records) / n

    directional_accuracy = sum(r["direction_correct"] for r in records) / n * 100

    interval_records = [r for r in records if r["in_interval"] is not None]
    interval_total = len(interval_records)
    interval_hits = sum(r["in_interval"] for r in interval_records)
    interval_coverage = (interval_hits / interval_total * 100) if interval_total else 0

    total_actual = sum(r["actual_price"] for r in records)
    wmape = (sum(r["abs_error"] for r in records) / total_actual * 100) if total_actual > 0 else 0

    tier_errors = defaultdict(list)
    for r in records:
        tier_errors[r["price_tier"]].append(r["pct_error"])
    mape_by_tier = {
        f"tier_{t}": round(sum(errs) / len(errs), 2)
        for t, errs in sorted(tier_errors.items())
    }

    baseline_hits = sum(1 for r in records if r["actual_direction"] == "flat")
    baseline_directional_accuracy = baseline_hits / n * 100
    baseline_mae = sum(abs(r["base_price"] - r["actual_price"]) for r in records) / n

    high_conf = [r for r in records if r["confidence"] == "high"]
    low_conf = [r for r in records if r["confidence"] == "low"]
    high_dir_acc = sum(r["direction_correct"] for r in high_conf) / len(high_conf) * 100 if high_conf else 0
    low_dir_acc = sum(r["direction_correct"] for r in low_conf) / len(low_conf) * 100 if low_conf else 0

    high_interval = [r for r in high_conf if r["in_interval"] is not None]
    high_int_cov = (
        round(sum(r["in_interval"] for r in high_interval) / len(high_interval) * 100, 2)
        if high_interval else 0
    )

    dir_ci_lower, dir_ci_upper = bootstrap_ci([r["direction_correct"] for r in records])
    mae_ci_lower, mae_ci_upper = bootstrap_ci([r["abs_error"] for r in records])

    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 2),
        "wmape": round(wmape, 2),
        "mape_by_tier": mape_by_tier,
        "directional_accuracy": round(directional_accuracy, 2),
        "interval_coverage": round(interval_coverage, 2),
        "baseline_directional_accuracy": round(baseline_directional_accuracy, 2),
        "improvement_over_baseline_pp": round(directional_accuracy - baseline_directional_accuracy, 2),
        "baseline_mae": round(baseline_mae, 4),
        "skill_vs_baseline": round(mae / baseline_mae, 4) if baseline_mae > 0 else None,
        "conf_gap_pp": round(high_dir_acc - low_dir_acc, 2),
        "conf_high_interval_cov": high_int_cov,
        "conf_calibration_error": round(abs(high_dir_acc - CONFIDENCE_TARGET_ACCURACY), 2),
        "directional_accuracy_ci_lower": dir_ci_lower,
        "directional_accuracy_ci_upper": dir_ci_upper,
        "mae_ci_lower": mae_ci_lower,
        "mae_ci_upper": mae_ci_upper,
    }
    return metrics, n
```

- [x] **Step 4: Rewire `backtest_accuracy.py` to call it**

Delete lines 36-40, 43-48, 51-60, 63-76 from `backend/scripts/backtest_accuracy.py` and add to its imports:

```python
from backtest.scoring import (
    FLAT_TOLERANCE,
    bootstrap_ci,
    direction_from_return,
    price_tier,
    score_cohort,
)
```

Replace the metric block at lines 379-500 with:

```python
        metrics, n = score_cohort(records)
        if n == 0:
            logger.info(f"  [{horizon}d / {model_version}] No valid comparisons")
            continue

        results.append({
            "prediction_type": "forecast",
            "evaluation_date": today,
            "horizon_days": horizon,
            "model_version": model_version,
            "evaluation_window_days": None,
            "sample_count": n,
            "metrics": metrics,
            "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        })
```

Update the two remaining call sites of the renamed helpers: `_direction_from_return(actual_ret)` → `direction_from_return(actual_ret)` (line 328) and `_price_tier(current)` → `price_tier(current)` (line 336).

- [x] **Step 5: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: PASS. Previously 235 tests passed; expect 235 + the new ones.

- [x] **Step 6: Commit**

```bash
git add backend/backtest/scoring.py backend/scripts/backtest_accuracy.py backend/tests/test_backtest_scoring.py
git commit -m "refactor: extract pure scorer from backtest resolve loop"
```

---

### Task 5: Resolve both legs with the shared estimator

The behavioural change. After this task the metric is symmetric.

**Files:**
- Modify: `backend/scripts/backtest_accuracy.py:110-193` (delete `_load_actual_prices`), `:294-373` (the resolve loop)
- Test: `backend/tests/test_backtest_scoring.py`

**Interfaces:**
- Consumes: `load_voted_prices`, `smoothed_prices` (Tasks 2–3); `direction_from_return`, `price_tier` (Task 4).
- Produces: outcome dicts now carrying `base_price`, and `resolved_at`.

- [x] **Step 1: Write the failing test**

Append to `backend/tests/test_backtest_scoring.py`:

```python
from datetime import timedelta

from backtest.price_resolution import smoothed_prices


def test_both_legs_use_the_same_estimator_so_a_flat_market_scores_flat():
    """The end-to-end symmetry property. Under the old code the base leg was a
    3-observation median and the actual leg a single-day price, so a perfectly
    flat market could still produce non-flat direction labels."""
    import pandas as pd

    days = [date(2026, 7, 1) + timedelta(days=i) for i in range(12)]
    voted = pd.DataFrame(
        {"item_id": ["ak"] * 12, "date": days, "price": [3.0] * 12}
    )

    forecast_date, target_date = date(2026, 7, 5), date(2026, 7, 8)
    prices = smoothed_prices(voted, {("ak", forecast_date), ("ak", target_date)})

    base = prices[("ak", forecast_date)]
    actual = prices[("ak", target_date)]
    assert direction_from_return((actual - base) / base) == "flat"


def test_resolution_drops_rather_than_falling_back_when_a_leg_is_unresolvable():
    """Substituting a fallback for a missing leg would reintroduce exactly the
    asymmetry this change removes."""
    import pandas as pd

    voted = pd.DataFrame(
        {
            "item_id": ["ak", "ak"],
            "date": [date(2026, 5, 1), date(2026, 7, 8)],
            "price": [3.0, 3.0],
        }
    )
    prices = smoothed_prices(voted, {("ak", date(2026, 7, 8))})
    # Only 2 observations, spanning 68 days — beyond the cap, so unresolvable.
    assert ("ak", date(2026, 7, 8)) not in prices
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_backtest_scoring.py -k legs -v`
Expected: FAIL — `test_both_legs_use_the_same_estimator_so_a_flat_market_scores_flat` errors on the `direction_from_return` import if Task 4's import line was not added.

- [x] **Step 3: Delete `_load_actual_prices` and rewrite the resolve loop**

Delete `backend/scripts/backtest_accuracy.py:110-193` entirely.

In `backtest_forecasts`, replace the per-group body (lines 294-373) with:

```python
    archive_dir = Path(__file__).parent.parent.parent / "price-archive"

    slug_rows = db.execute(text("SELECT id, item_id FROM items")).fetchall()
    id_to_slug = {r.id: r.item_id for r in slug_rows}

    results = []
    all_outcomes = []
    n_unresolvable = 0
    n_considered = 0

    for (horizon, model_version), forecasts in sorted(groups.items()):
        anchors = set()
        slugs = set()
        for f in forecasts:
            slug = id_to_slug.get(f.item_id)
            if slug is None:
                continue
            f_date = f.forecast_date if isinstance(f.forecast_date, date) else date.fromisoformat(str(f.forecast_date)[:10])
            slugs.add(slug)
            anchors.add((slug, f_date))
            anchors.add((slug, f_date + timedelta(days=horizon)))

        if not anchors:
            logger.info(f"  [{horizon}d / {model_version}] No slug mappings")
            continue

        anchor_dates = [a[1] for a in anchors]
        voted = load_voted_prices(archive_dir, sorted(slugs), min(anchor_dates), max(anchor_dates))
        prices = smoothed_prices(voted, anchors)
        logger.info(f"  Resolved {len(prices):,} of {len(anchors):,} anchors")

        records = []
        for f in forecasts:
            n_considered += 1
            slug = id_to_slug.get(f.item_id)
            f_date = f.forecast_date if isinstance(f.forecast_date, date) else date.fromisoformat(str(f.forecast_date)[:10])
            target_date = f_date + timedelta(days=horizon)

            base = prices.get((slug, f_date))
            actual = prices.get((slug, target_date))
            if base is None or actual is None or base <= 0 or actual <= 0:
                n_unresolvable += 1
                continue

            mid, low, high = f.price_mid, f.price_low, f.price_high
            if mid is None:
                n_unresolvable += 1
                continue
            if min_price > 0 and base < min_price:
                continue

            abs_error = abs(mid - actual)
            actual_ret = (actual - base) / base
            predicted_direction = f.direction or "flat"
            actual_direction = direction_from_return(actual_ret)
            direction_correct = 1 if predicted_direction == actual_direction else 0
            in_interval = None if (low is None or high is None) else (1 if low <= actual <= high else 0)
            pct_error = abs(abs_error / base) * 100

            records.append({
                "abs_error": abs_error,
                "pct_error": pct_error,
                "sq_error": (mid - actual) ** 2,
                "direction_correct": direction_correct,
                "predicted_direction": predicted_direction,
                "actual_direction": actual_direction,
                "in_interval": in_interval,
                "confidence": f.confidence or "low",
                "base_price": base,
                "actual_price": actual,
                "price_tier": price_tier(base),
                "item_id": f.item_id,
            })

            all_outcomes.append({
                "forecast_id": f.id,
                "item_id": f.item_id,
                "forecast_date": f_date,
                "horizon_days": horizon,
                "target_date": target_date,
                "current_price": f.current_price,
                "base_price": base,
                "predicted_price_low": low,
                "predicted_price_mid": mid,
                "predicted_price_high": high,
                "actual_price": actual,
                "direction_predicted": predicted_direction,
                "direction_actual": actual_direction,
                "direction_correct": direction_correct,
                "in_interval": in_interval,
                "abs_error": round(abs_error, 4),
                "pct_error": pct_error,
                "model_version": model_version,
            })
```

Add `from backtest.price_resolution import load_voted_prices, smoothed_prices` to the imports.

`current_price` is still *written* to the outcome row for reference, but nothing reads it for scoring.

- [x] **Step 4: Add the unresolvable-rate gate**

After the group loop in `backtest_forecasts`, before `_upsert_accuracy`:

```python
    if n_considered:
        unresolvable_pct = n_unresolvable / n_considered * 100
        logger.info(f"  Unresolvable: {n_unresolvable:,}/{n_considered:,} ({unresolvable_pct:.1f}%)")
        if unresolvable_pct > MAX_UNRESOLVABLE_PCT:
            raise RuntimeError(
                f"{unresolvable_pct:.1f}% of mature forecasts could not be resolved "
                f"(cap {MAX_UNRESOLVABLE_PCT}%). Silent cohort shrinkage is how this "
                f"metric moved unnoticed before — refusing to report a number."
            )
```

With `MAX_UNRESOLVABLE_PCT = 10.0` alongside the other module constants.

- [x] **Step 5: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: PASS

- [x] **Step 6: Commit**

```bash
git add backend/scripts/backtest_accuracy.py backend/tests/test_backtest_scoring.py
git commit -m "fix: score both legs of actual_ret with one shared estimator"
```

---

### Task 6: Freeze resolved actuals

**Files:**
- Modify: `backend/scripts/backtest_accuracy.py:200-240` (`_store_forecast_outcomes`), and the `main()` argument parsing at `:380-386`
- Test: `backend/tests/test_backtest_scoring.py`

**Interfaces:**
- Consumes: outcome dicts from Task 5.
- Produces: `_store_forecast_outcomes(db, outcomes, reresolve: bool = False) -> int` returning the number of rows actually written. CLI flags `--rescore` and `--reresolve`.

- [x] **Step 1: Write the failing test**

Append to `backend/tests/test_backtest_scoring.py`:

```python
import scripts.backtest_accuracy as bt


def test_resolution_is_insert_only(session):
    outcome = {
        "forecast_id": 7,
        "item_id": 1,
        "forecast_date": date(2026, 7, 1),
        "horizon_days": 3,
        "target_date": date(2026, 7, 4),
        "current_price": 1.0,
        "base_price": 1.0,
        "predicted_price_low": 0.9,
        "predicted_price_mid": 1.1,
        "predicted_price_high": 1.3,
        "actual_price": 1.2,
        "direction_predicted": "up",
        "direction_actual": "up",
        "direction_correct": 1,
        "in_interval": 1,
        "abs_error": 0.1,
        "pct_error": 10.0,
        "model_version": "lgbm-v3-regime",
    }
    assert bt._store_forecast_outcomes(session, [dict(outcome)]) == 1

    # The archive is revised: the same forecast now resolves to a different
    # actual. Freezing means the stored row does not move.
    revised = dict(outcome, actual_price=99.0, direction_actual="down", direction_correct=0)
    assert bt._store_forecast_outcomes(session, [revised]) == 0

    stored = session.query(ForecastOutcome).filter_by(forecast_id=7).one()
    assert stored.actual_price == 1.2
    assert stored.direction_correct == 1
    assert stored.resolved_at is not None


def test_reresolve_overrides_the_freeze(session):
    outcome = {
        "forecast_id": 8,
        "item_id": 1,
        "forecast_date": date(2026, 7, 1),
        "horizon_days": 3,
        "target_date": date(2026, 7, 4),
        "current_price": 1.0,
        "base_price": 1.0,
        "predicted_price_mid": 1.1,
        "actual_price": 1.2,
        "direction_predicted": "up",
        "direction_actual": "up",
        "direction_correct": 1,
        "in_interval": 1,
        "abs_error": 0.1,
        "pct_error": 10.0,
        "model_version": "lgbm-v3-regime",
    }
    bt._store_forecast_outcomes(session, [dict(outcome)])
    revised = dict(outcome, actual_price=99.0)
    assert bt._store_forecast_outcomes(session, [revised], reresolve=True) == 1

    assert session.query(ForecastOutcome).filter_by(forecast_id=8).one().actual_price == 99.0
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_backtest_scoring.py -k resolve -v`
Expected: FAIL — the current implementation deletes and re-inserts, so the second call returns the full count and `actual_price` becomes `99.0`.

- [x] **Step 3: Make resolution insert-only**

Replace `_store_forecast_outcomes` (lines 200-240) with:

```python
def _store_forecast_outcomes(db, outcomes, reresolve: bool = False) -> int:
    """Persist per-forecast outcomes. Insert-only unless *reresolve*.

    Resolved actuals are frozen: a forecast_id that already has a row keeps
    its base_price and actual_price forever. Re-running the backtest after the
    archive gains source rows for an old target date must not rewrite history
    — that silent rewriting is why the same 5,512 forecasts scored 61.76% on
    07-18 and 33.74% on 07-19. Metrics stay derived and are recomputed every
    run, so a *scoring* fix still lands without --reresolve.
    """
    if not outcomes:
        return 0
    from database import ForecastOutcome

    all_fids = [o["forecast_id"] for o in outcomes]
    existing_ids = set()
    for i in range(0, len(all_fids), 900):
        batch = all_fids[i:i + 900]
        rows = db.query(ForecastOutcome.forecast_id).filter(
            ForecastOutcome.forecast_id.in_(batch)
        ).all()
        existing_ids.update(r[0] for r in rows)

    if reresolve:
        stale = list(existing_ids)
        for i in range(0, len(stale), 900):
            batch = stale[i:i + 900]
            db.query(ForecastOutcome).filter(
                ForecastOutcome.forecast_id.in_(batch)
            ).delete(synchronize_session=False)
        to_write = outcomes
    else:
        to_write = [o for o in outcomes if o["forecast_id"] not in existing_ids]

    if not to_write:
        db.commit()
        logger.info(f"  All {len(outcomes):,} outcomes already resolved (frozen)")
        return 0

    resolved_at = datetime.now(timezone.utc).replace(tzinfo=None)
    for o in to_write:
        o["evaluated_at"] = resolved_at
        o["resolved_at"] = resolved_at

    db.bulk_insert_mappings(ForecastOutcome, to_write)
    db.commit()

    from db.parquet import append_table
    append_table("forecast_outcomes", to_write, ["forecast_id"])

    logger.info(
        f"  Resolved {len(to_write):,} new outcomes "
        f"({len(outcomes) - len(to_write):,} already frozen)"
    )
    return len(to_write)
```

- [x] **Step 4: Add the CLI flags**

In `main()` at `backend/scripts/backtest_accuracy.py:380`, alongside the existing arg parsing:

```python
    reresolve = "--reresolve" in args
    rescore = "--rescore" in args
```

Thread `reresolve` through `backtest_forecasts(db, today=today, min_price=min_price, reresolve=reresolve)` into the `_store_forecast_outcomes` call. When `rescore` is set, skip resolution entirely and score from the stored `ForecastOutcome` rows — add to `backtest_forecasts`:

```python
def _records_from_frozen_outcomes(db, min_price=0):
    """Rebuild scoring records from stored outcomes, without the archive.

    Grouped identically to the resolve path so score_by_tier sees the same
    shape either way. `confidence` is not on ForecastOutcome, so it is joined
    back from item_forecasts.
    """
    from database import ForecastOutcome

    rows = db.execute(text("""
        SELECT o.forecast_id, o.item_id, o.horizon_days, o.model_version,
               o.base_price, o.actual_price, o.predicted_price_low,
               o.predicted_price_mid, o.predicted_price_high,
               o.direction_predicted, o.direction_actual, o.direction_correct,
               o.in_interval, f.confidence
        FROM forecast_outcomes o
        LEFT JOIN item_forecasts f ON f.id = o.forecast_id
        WHERE o.base_price IS NOT NULL AND o.base_price > 0
    """)).fetchall()

    groups = defaultdict(list)
    for r in rows:
        if min_price > 0 and r.base_price < min_price:
            continue
        abs_error = abs(r.predicted_price_mid - r.actual_price)
        groups[(r.horizon_days, r.model_version or "unknown")].append({
            "abs_error": abs_error,
            "pct_error": abs(abs_error / r.base_price) * 100,
            "sq_error": (r.predicted_price_mid - r.actual_price) ** 2,
            "direction_correct": r.direction_correct,
            "predicted_direction": r.direction_predicted,
            "actual_direction": r.direction_actual,
            "in_interval": r.in_interval,
            "confidence": r.confidence or "low",
            "base_price": r.base_price,
            "actual_price": r.actual_price,
            "price_tier": price_tier(r.base_price),
            "item_id": r.item_id,
        })
    return groups
```

Then at the top of `backtest_forecasts`, before the archive is touched:

```python
    if rescore:
        logger.info("  --rescore: scoring from frozen outcomes, archive not read")
        groups = _records_from_frozen_outcomes(db, min_price=min_price)
        results = []
        for (horizon, model_version), records in sorted(groups.items()):
            for tier, metrics, n in score_by_tier(records):
                results.append({
                    "prediction_type": "forecast",
                    "evaluation_date": today,
                    "horizon_days": horizon,
                    "model_version": model_version,
                    "price_tier": tier,
                    "evaluation_window_days": None,
                    "sample_count": n,
                    "metrics": metrics,
                    "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
                })
        _upsert_accuracy(db, results)
        return results
```

Update the module docstring usage block (lines 9-11) to list all three invocations:

```
    python scripts/backtest_accuracy.py                  # resolve new + score
    python scripts/backtest_accuracy.py --rescore        # score frozen only
    python scripts/backtest_accuracy.py --reresolve      # re-read the archive
```

- [x] **Step 5: Run tests**

Run: `python -m pytest tests/test_backtest_scoring.py -v`
Expected: PASS

- [x] **Step 6: Mutation-check the freeze**

Temporarily force `to_write = outcomes` unconditionally and re-run.
Expected: `test_resolution_is_insert_only` FAILS with `actual_price == 99.0`.
Revert and confirm it passes.

- [x] **Step 7: Commit**

```bash
git add backend/scripts/backtest_accuracy.py backend/tests/test_backtest_scoring.py
git commit -m "fix: freeze resolved backtest actuals; add --rescore/--reresolve"
```

---

### Task 7: Tiered metrics and headline reporting

**Files:**
- Modify: `backend/backtest/scoring.py` (add `score_by_tier`), `backend/scripts/backtest_accuracy.py` (`_upsert_accuracy` key, the log line)
- Test: `backend/tests/test_backtest_scoring.py`

**Interfaces:**
- Consumes: `score_cohort` (Task 4).
- Produces: `score_by_tier(records: list[dict]) -> list[tuple[int | None, dict, int]]` — one entry per tier present plus a final `(None, metrics, n)` all-tiers row. `HEADLINE_MIN_TIER: int = 1`.

- [x] **Step 1: Write the failing test**

Append to `backend/tests/test_backtest_scoring.py`:

```python
from backtest.scoring import HEADLINE_MIN_TIER, score_by_tier


def test_tier_rows_partition_the_all_row():
    records = [_record(price_tier=0, item_id=i) for i in range(30)]
    records += [_record(price_tier=1, item_id=100 + i) for i in range(20)]

    scored = score_by_tier(records)
    per_tier = {tier: n for tier, _, n in scored if tier is not None}
    all_rows = [(m, n) for tier, m, n in scored if tier is None]

    assert per_tier == {0: 30, 1: 20}
    assert len(all_rows) == 1
    assert all_rows[0][1] == 50
    assert sum(per_tier.values()) == all_rows[0][1]


def test_empty_tiers_are_omitted_not_zero_filled():
    records = [_record(price_tier=4, item_id=i) for i in range(12)]
    tiers = {tier for tier, _, _ in score_by_tier(records) if tier is not None}
    assert tiers == {4}


def test_headline_tier_is_one_dollar_and_up():
    assert HEADLINE_MIN_TIER == 1
    assert price_tier(0.99) < HEADLINE_MIN_TIER
    assert price_tier(1.00) >= HEADLINE_MIN_TIER
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_backtest_scoring.py -k tier -v`
Expected: FAIL with `ImportError: cannot import name 'score_by_tier'`

- [x] **Step 3: Implement tiered scoring**

Append to `backend/backtest/scoring.py`:

```python
# Tiers at or above this are aggregated into the headline figure. Tier 0
# (<$1) is 72% of the evaluated universe and one cent there is a 20% move,
# so its up/flat/down label is dominated by tick quantisation. It is
# reported separately rather than filtered out — "the model is worse on
# penny items" is a real question the tier rows keep answerable.
HEADLINE_MIN_TIER = 1


def score_by_tier(records: list[dict]) -> list[tuple[int | None, dict, int]]:
    """Score per price tier plus an all-tiers aggregate.

    Returns [(tier, metrics, n), ..., (None, metrics, n)]. Tiers with no
    records are omitted rather than emitted as zeros.
    """
    by_tier: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_tier[r["price_tier"]].append(r)

    out: list[tuple[int | None, dict, int]] = []
    for tier in sorted(by_tier):
        metrics, n = score_cohort(by_tier[tier])
        if n:
            out.append((tier, metrics, n))

    metrics, n = score_cohort(records)
    if n:
        out.append((None, metrics, n))
    return out


def headline_records(records: list[dict]) -> list[dict]:
    """The >=$1 subset used for the headline log line."""
    return [r for r in records if r["price_tier"] >= HEADLINE_MIN_TIER]
```

- [x] **Step 4: Emit tier rows and the split log line**

In `backtest_forecasts`, replace the single `results.append(...)` with:

```python
        for tier, metrics, n in score_by_tier(records):
            results.append({
                "prediction_type": "forecast",
                "evaluation_date": today,
                "horizon_days": horizon,
                "model_version": model_version,
                "price_tier": tier,
                "evaluation_window_days": None,
                "sample_count": n,
                "metrics": metrics,
                "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
            })
```

Replace the log line at `:505-513` with a headline/penny split:

```python
        head = headline_records(records)
        penny = [r for r in records if r["price_tier"] < HEADLINE_MIN_TIER]
        head_metrics, head_n = score_cohort(head)
        penny_metrics, penny_n = score_cohort(penny)

        if head_n:
            logger.info(
                f"  [{horizon}d / {model_version}] >=$1: {head_n:,} samples — "
                f"MAE=${head_metrics['mae']:.2f} MAPE={head_metrics['mape']:.1f}% "
                f"DirAcc={head_metrics['directional_accuracy']:.1f}% "
                f"[CI: {head_metrics['directional_accuracy_ci_lower'] * 100:.1f}–"
                f"{head_metrics['directional_accuracy_ci_upper'] * 100:.1f}] "
                f"IntCov={head_metrics['interval_coverage']:.1f}% "
                f"ConfGap={head_metrics['conf_gap_pp']:.1f}pp "
                f"Skill={head_metrics['skill_vs_baseline']}"
            )
        if penny_n:
            logger.info(
                f"  [{horizon}d / {model_version}] <$1: {penny_n:,} samples — "
                f"DirAcc={penny_metrics['directional_accuracy']:.1f}% (tick-dominated)"
            )
```

Note the `* 100` on the CI bounds. `bootstrap_ci` runs over 0/1 values so it
returns fractions, which the old line printed unscaled next to a percentage —
producing `DirAcc=55.0% [CI: 0.5–0.6]`.

- [x] **Step 5: Extend the upsert key**

In `_upsert_accuracy` (line 79), add `price_tier` to the filter dict so tier rows do not overwrite each other:

```python
        filters["price_tier"] = row.get("price_tier")
```

and add `price_tier` to the `append_table` dedup keys at line 107:

```python
        append_table("prediction_accuracy", rows,
                     ["prediction_type", "evaluation_date", "horizon_days",
                      "model_version", "price_tier"])
```

- [x] **Step 6: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: PASS

- [x] **Step 7: Commit**

```bash
git add backend/backtest/scoring.py backend/scripts/backtest_accuracy.py backend/tests/test_backtest_scoring.py
git commit -m "feat: per-tier accuracy rows with a >=\$1 headline"
```

---

### Task 8: Determinism regression test

The test that would have caught the original bug. Written last because it needs the whole pipeline.

**Files:**
- Test: `backend/tests/test_backtest_scoring.py`

**Interfaces:**
- Consumes: everything from Tasks 2–7.
- Produces: nothing.

- [x] **Step 1: Write the test**

```python
def test_metrics_are_stable_when_the_archive_gains_a_source_row(tmp_path):
    """The 61.76 -> 33.74 -> 61.54 regression, in miniature.

    An archive revision for an already-resolved target date must not move the
    reported metric. Freezing guarantees it; the shared estimator makes the
    first resolution trustworthy in the first place.
    """
    import pandas as pd
    from backtest.price_resolution import load_voted_prices, smoothed_prices

    archive = tmp_path / "price-archive"
    archive.mkdir()
    days = [date(2026, 7, 1) + timedelta(days=i) for i in range(12)]

    def write(extra_rows):
        rows = {
            "item_slug": ["ak"] * 12,
            "day": pd.to_datetime(days),
            "mean_price": [3.0] * 12,
            "volume": [5] * 12,
            "source": ["a"] * 12,
        }
        frame = pd.DataFrame(rows)
        if extra_rows is not None:
            frame = pd.concat([frame, extra_rows], ignore_index=True)
        frame.to_parquet(archive / "prices-2026.parquet")

    anchors = {("ak", date(2026, 7, 5)), ("ak", date(2026, 7, 8))}

    write(None)
    before = smoothed_prices(
        load_voted_prices(archive, ["ak"], date(2026, 7, 1), date(2026, 7, 12)), anchors
    )

    # A second source appears for an already-resolved day, well off consensus.
    write(pd.DataFrame({
        "item_slug": ["ak"],
        "day": pd.to_datetime([date(2026, 7, 8)]),
        "mean_price": [75.0],
        "volume": [5],
        "source": ["b"],
    }))
    after = smoothed_prices(
        load_voted_prices(archive, ["ak"], date(2026, 7, 1), date(2026, 7, 12)), anchors
    )

    assert direction_from_return(
        (before[("ak", date(2026, 7, 8))] - before[("ak", date(2026, 7, 5))])
        / before[("ak", date(2026, 7, 5))]
    ) == "flat"

    # Voting rejects the outlier source; even unfrozen, the estimator holds.
    assert after[("ak", date(2026, 7, 8))] == before[("ak", date(2026, 7, 8))]
```

- [x] **Step 2: Run it**

Run: `python -m pytest tests/test_backtest_scoring.py -k stable -v`
Expected: PASS

- [x] **Step 3: Mutation-check it**

Temporarily change `smoothed_prices` to return the single anchor-day price instead of the window median (`selected = selected[-1:]`).
Expected: the test FAILS — the outlier source moves the 07-08 price.
Revert and confirm PASS.

- [x] **Step 4: Commit**

```bash
git add backend/tests/test_backtest_scoring.py
git commit -m "test: archive revision must not move reported accuracy"
```

---

### Task 9: Historical backfill and documentation

**Files:**
- Create: `docs/changelog/2026-08-01-deterministic-backtest.md`
- Modify: `backend/AGENTS.md`, `docs/README.md` (changelog entry count), `docs/operations.md`

**Interfaces:**
- Consumes: the `--reresolve` flag (Task 6).
- Produces: nothing.

- [x] **Step 1: Re-resolve the historical outcomes**

The 65,642 existing outcomes carry an `actual_price` from the old estimator and no `base_price`. Run once:

```bash
cd backend && python scripts/backtest_accuracy.py --type forecast --reresolve 2>&1 | tee /tmp/reresolve.log
```

- [x] **Step 2: Verify the backfill against prod rows, not a green exit**

Per `collectors-fail-silently`, a zero exit is not evidence. Confirm every outcome now has a `base_price` and that the headline moved for the expected reason:

```bash
python3 -c "
import duckdb
d='../price-archive/ops/forecast_outcomes.parquet'
print(duckdb.sql(f'''select count(*) n, count(base_price) with_base,
  count(resolved_at) resolved from read_parquet(\"{d}\")''').df().to_string())
"
```
Expected: `with_base == resolved == n`.

- [x] **Step 3: Record the before/after**

Capture the 3d/7d/14d/30d directional accuracy per model version before and after, and put both tables in the changelog. The historical series *will* move — that is the fix landing, and the entry must say so plainly rather than presenting the new numbers as if they were always there.

- [x] **Step 4: Write the changelog**

Create `docs/changelog/2026-08-01-deterministic-backtest.md` covering: the 61.76/33.74/61.54/57.91 evidence, the two-estimator root cause with file:line references, the shared estimator and its staleness cap, the freeze semantics and the two escape hatches, the tier split with the <$0.50 tick-noise table, the `pct_error`-divided-by-base oddity left deliberately unfixed (Task 4), and the before/after accuracy tables from Step 3.

- [x] **Step 5: Update the docs**

- `backend/AGENTS.md` — add a gotcha: the backtest resolves both legs through `backtest.price_resolution.smoothed_prices`; `item_forecasts.current_price` is written but never scored on; resolved outcomes are frozen and `--reresolve` is the only thing that moves them.
- `docs/operations.md` — document `--rescore` / `--reresolve` and the unresolvable-rate gate.
- `docs/README.md` — bump the changelog entry count.

- [x] **Step 6: Run the full suite and commit**

```bash
cd backend && python -m pytest tests/ -q
git add docs/ backend/AGENTS.md
git commit -m "docs: changelog and operations notes for the deterministic backtest"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §1 Architecture: pure scorer over frozen inputs | 4 (extract), 5 (resolve), 6 (freeze) |
| §2 The estimator | 2 (estimator), 3 (loader) |
| §3 Freezing the actuals, not the metrics | 6, with the backfill in 9 |
| §4 Tiered metrics | 7 |
| §5 Staleness cap | 2 |
| §5 Missing archive → hard fail | 3 |
| §5 Unresolvable-rate gate | 5 Step 4 |
| §6 Test 1 determinism | 8 |
| §6 Test 2 estimator symmetry | 2, 5 |
| §6 Test 3 freeze invariance | 6 |
| §6 Test 4 tier partition | 7 |
| §6 Test 5 staleness cap | 2 |
| §6 Test 6 purity | 4 |
| §6 Mutation checks | 2 Step 5, 6 Step 6, 8 Step 3 |

No gaps.

**Type consistency:** `smoothed_prices` and `load_voted_prices` keep the same signatures across Tasks 2, 3, 5, 8. `score_cohort` returns `(metrics, n)` in Tasks 4, 7. Records carry `base_price` (never `current_price`) from Task 5 onward, matching what `score_cohort` reads in Task 4. `price_tier` is the function; `price_tier` is also the record key and the DB column — same meaning throughout.

**Known scope boundary:** this plan makes accuracy *measurable*. It does not change the drift threshold, regime-model persistence, or conformal calibration — those are the two follow-on items and get their own plans once the metric can be trusted.
