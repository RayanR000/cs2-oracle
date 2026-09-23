# Served Forecast Surface Implementation Plan

> # ✅ EXECUTED AND CLOSED (2026-08-03)
>
> All 7 tasks landed. Evidence: `backend/api/serving_policy.py` (`acbbff1`,
> `MIN_SERVED_PRICE_USD = 1.0`), `backend/scripts/check_forecast_freshness.py`, the
> "Verify forecasts were persisted" step at `price-forecast.yml:206`, and the removal of the
> `sub=` confidence badge from `frontend/app/items/[id]/page.tsx`.
>
> **Outcome: `docs/changelog/2026-08-03-served-forecast-surface.md`.**
>
> The boxes below are ticked retroactively. Nothing here is outstanding.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the API from selecting and ranking forecasts by an anti-predictive confidence flag, restrict ranked surfaces to items whose direction labels are not tick-quantization artifacts, and make the daily forecast run fail loudly when it persists nothing.

**Architecture:** One new module holds the single serving threshold and the SQLAlchemy clause that applies it, so four call sites cannot drift. The in-Python selection logic in `opportunities.py` is extracted into pure functions that take plain objects, because this repo has no API test fixtures and no route test can run against SQLite (the endpoints use PostgreSQL `DISTINCT ON`). A new script asserts forecast freshness in both stores the API reads.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy 2.x, pytest, DuckDB (Parquet reads), Next.js 15 / React (frontend), GitHub Actions.

## Global Constraints

- **No changes to `backend/models/forecaster.py`, to features, or to training.** The accuracy roadmap in `docs/research/accuracy-opportunities.md` is closed and stays closed; this is serving-layer work only.
- **`MIN_SERVED_PRICE_USD = 1.0`**, chosen to equal the lower bound of `HEADLINE_MIN_TIER = 1` in `backend/backtest/scoring.py`. Never inline the literal — import the constant.
- **Do not delete `_calibrate_confidence`, the `confidence` column, or `conf_gap_pp` scoring.** The flag must keep being computed and scored so it can be diagnosed later; only its *use as a filter or sort key* is removed.
- **Do not remove `confidence` from any Pydantic response schema.** `TrendAnalysisOut.confidence` and `PredictionOut.confidence` are consumed by `frontend/app/items/[id]/page.tsx`; dropping a field is a breaking API change and is out of scope.
- **Run all tests from the `backend/` directory** using `python3 -m pytest`, which is what puts `backend/` on `sys.path`. Tests import as `from api.serving_policy import ...`, not `from backend.api...`. Locally the interpreter is `python3` — bare `python` is not on PATH — and pytest 9.0.2 resolves from system Python; a `backend/venv/` also exists if you prefer it. Inside `price-forecast.yml` the interpreter is `python`, provided by `actions/setup-python`; do not change that.
- **Price floor applies to ranked/selection surfaces only, never to per-item detail endpoints.** A user who opens a $0.30 item must still see its forecast; the floor governs what the product *chooses* to show, not what it shows on request.
- Tests follow the house style in `backend/tests/test_accuracy_tier_mirror.py`: module docstring stating the regression, `from __future__ import annotations`, plain builder helpers, classes grouping related cases.

---

### Task 1: Serving policy module

**Files:**
- Create: `backend/api/serving_policy.py`
- Test: `backend/tests/test_serving_policy.py`

**Interfaces:**
- Consumes: `price_tier`, `HEADLINE_MIN_TIER` from `backend.backtest.scoring` (import path `backtest.scoring`).
- Produces:
  - `MIN_SERVED_PRICE_USD: float` — the floor, `1.0`.
  - `price_floor_clause(column) -> ColumnElement` — SQLAlchemy `column >= MIN_SERVED_PRICE_USD`.
  - `meets_price_floor(price: float | None) -> bool` — the in-Python equivalent; `None` is False.

- [x] **Step 1: Write the failing test**

Create `backend/tests/test_serving_policy.py`:

```python
"""The served universe and the measured universe must stay the same population.

The headline accuracy number is computed over price_tier >= HEADLINE_MIN_TIER
(scoring.py), while the API surfaces had no price floor at all — 84% of the
forecasts they ranked were sub-$1 items where one cent is a 20% move. The
number quoted and the list displayed described different populations. These
tests pin the floor to the headline tier so the two cannot drift apart again.
"""
from __future__ import annotations

from sqlalchemy.dialects import postgresql

from api.serving_policy import (
    MIN_SERVED_PRICE_USD,
    meets_price_floor,
    price_floor_clause,
)
from backtest.scoring import HEADLINE_MIN_TIER, price_tier
from database import ItemForecast


def _sql(clause) -> str:
    return str(
        clause.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )


class TestFloorMatchesHeadlineTier:
    def test_floor_is_the_lower_bound_of_the_headline_tier(self):
        """If either constant moves alone, this fails — that is the point."""
        assert price_tier(MIN_SERVED_PRICE_USD) == HEADLINE_MIN_TIER

    def test_just_below_the_floor_is_outside_the_headline_tier(self):
        assert price_tier(MIN_SERVED_PRICE_USD - 0.01) < HEADLINE_MIN_TIER


class TestPriceFloorClause:
    def test_clause_compiles_to_a_greater_or_equal_comparison(self):
        sql = _sql(price_floor_clause(ItemForecast.current_price))
        assert "item_forecasts.current_price >= 1.0" in sql


class TestMeetsPriceFloor:
    def test_at_the_floor_passes(self):
        assert meets_price_floor(1.0) is True

    def test_below_the_floor_fails(self):
        assert meets_price_floor(0.99) is False

    def test_none_fails_rather_than_raising(self):
        assert meets_price_floor(None) is False

    def test_zero_fails(self):
        assert meets_price_floor(0.0) is False
```

- [x] **Step 2: Run test to verify it fails**

```bash
cd backend && python3 -m pytest tests/test_serving_policy.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'api.serving_policy'`.

- [x] **Step 3: Write minimal implementation**

Create `backend/api/serving_policy.py`:

```python
"""What the product is willing to put in front of a user.

The one threshold here exists because sub-$1 items are ~72% of the forecast
universe and one cent there is a 20% move, so their up/flat/down label is
dominated by tick quantisation rather than by anything the model knows
(``backtest/scoring.py:209-213``). Ranking those items by percentage move —
which every opportunities surface did — promotes rounding artifacts to the top
of the list.

The floor is deliberately equal to the lower bound of ``HEADLINE_MIN_TIER``, so
the population the product shows is the population the headline accuracy figure
describes. ``tests/test_serving_policy.py`` fails if the two ever diverge.

It is a convention, not a derivation: the sharp break in the tier evidence is
nearer $0.50 (27.6% actual-flat below it against ~1% above). Matching the
headline is what earns the number its meaning.
"""
from __future__ import annotations

from sqlalchemy.sql.elements import ColumnElement

MIN_SERVED_PRICE_USD = 1.0


def price_floor_clause(column) -> ColumnElement:
    """SQL-side floor, for query filters."""
    return column >= MIN_SERVED_PRICE_USD


def meets_price_floor(price: float | None) -> bool:
    """Python-side floor, for rows already in memory."""
    if price is None:
        return False
    return price >= MIN_SERVED_PRICE_USD
```

- [x] **Step 4: Run test to verify it passes**

```bash
cd backend && python3 -m pytest tests/test_serving_policy.py -v
```

Expected: PASS, 7 tests.

- [x] **Step 5: Commit**

```bash
git add backend/api/serving_policy.py backend/tests/test_serving_policy.py
git commit -m "feat: add serving price floor pinned to the headline accuracy tier"
```

---

### Task 2: Remove the confidence gate from opportunities and apply the floor

**Files:**
- Modify: `backend/api/routes/opportunities.py` (whole file; 246 lines)
- Test: `backend/tests/test_opportunity_selection.py`

**Interfaces:**
- Consumes: `MIN_SERVED_PRICE_USD`, `meets_price_floor`, `price_floor_clause` from Task 1.
- Produces:
  - `opportunity_type_for(direction: str | None) -> str` — returns `"undervalued"` / `"overheated"` / `"momentum"`.
  - `select_opportunities(forecasts, items_map, type_filter, limit) -> list[OpportunityOut]` — pure; takes already-fetched rows.

- [x] **Step 1: Write the failing test**

Create `backend/tests/test_opportunity_selection.py`:

```python
"""The opportunities surface must not select on confidence, and must not show
tick-quantised items.

Directional accuracy split by served confidence, computed within each forecast
date so both groups faced the same market, was negative in all nine
date x horizon x model cells with n_high >= 30 — high confidence scored 4-31%
against low confidence 25-55%. The surface gated on ``confidence == "high"`` in
four queries and ranked by it, so it promoted the worst forecasts first. It also
had no price floor: the newest date offered 1,680 "undervalued" candidates with
a median price of $0.50.

See docs/specs/2026-08-03-served-forecast-surface-design.md.
"""
from __future__ import annotations

from types import SimpleNamespace

from api.routes.opportunities import opportunity_type_for, select_opportunities


def _forecast(item_id=1, direction="up", confidence="low", current=10.0, mid=11.0):
    return SimpleNamespace(
        item_id=item_id,
        direction=direction,
        confidence=confidence,
        current_price=current,
        price_mid=mid,
    )


def _item(item_id=1, name="AK-47 | Redline"):
    return SimpleNamespace(id=item_id, name=name)


def _items_map(*items):
    return {i.id: i for i in items}


class TestOpportunityType:
    def test_up_is_undervalued_without_high_confidence(self):
        assert opportunity_type_for("up") == "undervalued"

    def test_down_is_overheated_without_high_confidence(self):
        assert opportunity_type_for("down") == "overheated"

    def test_flat_is_momentum(self):
        assert opportunity_type_for("flat") == "momentum"

    def test_none_is_momentum(self):
        assert opportunity_type_for(None) == "momentum"


class TestSelectOpportunities:
    def test_low_confidence_item_is_eligible(self):
        """The regression: low confidence was excluded from undervalued."""
        results = select_opportunities(
            [_forecast(confidence="low")], _items_map(_item()), None, 10
        )
        assert [r.opportunity_type for r in results] == ["undervalued"]

    def test_sub_dollar_item_is_dropped(self):
        results = select_opportunities(
            [_forecast(current=0.03, mid=0.04)], _items_map(_item()), None, 10
        )
        assert results == []

    def test_item_exactly_at_the_floor_is_kept(self):
        results = select_opportunities(
            [_forecast(current=1.0, mid=1.2)], _items_map(_item()), None, 10
        )
        assert len(results) == 1

    def test_ranks_by_absolute_predicted_return_not_confidence(self):
        big_low_conf = _forecast(item_id=1, confidence="low", current=10.0, mid=14.0)
        small_high_conf = _forecast(item_id=2, confidence="high", current=10.0, mid=10.5)
        results = select_opportunities(
            [small_high_conf, big_low_conf],
            _items_map(_item(1), _item(2, "Glock")),
            None,
            10,
        )
        assert [r.item_id for r in results] == [1, 2]

    def test_type_filter_selects_one_bucket(self):
        results = select_opportunities(
            [_forecast(item_id=1, direction="up"), _forecast(item_id=2, direction="down", mid=9.0)],
            _items_map(_item(1), _item(2, "Glock")),
            "overheated",
            10,
        )
        assert [r.item_id for r in results] == [2]

    def test_limit_is_applied_after_sorting(self):
        forecasts = [
            _forecast(item_id=1, current=10.0, mid=10.5),
            _forecast(item_id=2, current=10.0, mid=14.0),
        ]
        results = select_opportunities(
            forecasts, _items_map(_item(1), _item(2, "Glock")), None, 1
        )
        assert [r.item_id for r in results] == [2]

    def test_missing_item_row_is_skipped(self):
        results = select_opportunities([_forecast(item_id=99)], {}, None, 10)
        assert results == []

    def test_none_direction_is_skipped(self):
        results = select_opportunities(
            [_forecast(direction=None)], _items_map(_item()), None, 10
        )
        assert results == []


class TestNoConfidenceGateRemains:
    def test_module_source_never_filters_on_confidence(self):
        """A source-level guard, because the four gates were in SQLAlchemy
        filters that cannot be exercised without a PostgreSQL connection —
        these endpoints use DISTINCT ON, which SQLite cannot compile."""
        from pathlib import Path

        import api.routes.opportunities as mod

        source = Path(mod.__file__).read_text()
        assert "ItemForecast.confidence" not in source
        assert "high confidence" not in source
```

- [x] **Step 2: Run test to verify it fails**

```bash
cd backend && python3 -m pytest tests/test_opportunity_selection.py -v
```

Expected: FAIL — `ImportError: cannot import name 'opportunity_type_for'`.

- [x] **Step 3: Write the implementation**

In `backend/api/routes/opportunities.py`, add to the imports after line 8 (`from api.schemas import OpportunityOut`):

```python
from api.serving_policy import meets_price_floor, price_floor_clause
```

Replace `_reason_for_type` (lines 30-36) — the strings asserted a confidence level that no longer gates anything and was never true:

```python
def _reason_for_type(opp_type: str) -> str:
    if opp_type == "undervalued":
        return "ML forecast predicts upward movement over the next 7 days."
    if opp_type == "overheated":
        return "ML forecast predicts downward movement over the next 7 days."
    return "ML forecast shows strong predicted price movement."
```

In `_latest_forecasts` (lines 45-66), add the floor to the outer `.filter(...)` so it reads:

```python
        .filter(
            ItemForecast.horizon_days == horizon_days,
            price_floor_clause(ItemForecast.current_price),
        )
```

Replace the whole of `_build_opportunities` (lines 81-108) with a thin fetch plus two pure functions:

```python
def _build_opportunities(db: Session, type: Optional[str], limit: int):
    forecasts = _latest_forecasts(db)
    item_ids = [f.item_id for f in forecasts if f.direction is not None]
    items_map = _load_items(item_ids, db)
    return select_opportunities(forecasts, items_map, type, limit)


def opportunity_type_for(direction: Optional[str]) -> str:
    """Direction alone decides the label.

    This previously required ``confidence == "high"`` for the directional
    labels, which meant the two headline buckets were populated exclusively by
    the anti-predictive subset. Confidence no longer participates.
    """
    if direction == "up":
        return "undervalued"
    if direction == "down":
        return "overheated"
    return "momentum"


def select_opportunities(forecasts, items_map, type_filter, limit):
    """Pure selection over already-fetched rows, ranked by |predicted return|."""
    results = []
    for f in forecasts:
        if f.direction is None:
            continue
        if not meets_price_floor(f.current_price):
            continue
        item = items_map.get(f.item_id)
        if not item:
            continue
        opp_type = opportunity_type_for(f.direction)
        if type_filter and opp_type != type_filter:
            continue
        results.append(_build_opportunity(item, f, opp_type))

    results.sort(key=lambda x: abs(x.opportunity_score), reverse=True)
    return results[:limit]
```

Note the dead `predicted_return` local in the old loop is gone — `_build_opportunity` computes it. The `meets_price_floor` call is redundant with the SQL floor in `_latest_forecasts` and kept deliberately: it is what makes the function testable without a database, and it defends the other callers.

In `/undervalued` (lines 111-153): delete `ItemForecast.confidence == "high",` from **both** the `subq` filter and the outer filter, and in the outer filter replace `ItemForecast.current_price > 0,` with `price_floor_clause(ItemForecast.current_price),`.

In `/overheated` (lines 156-198): make the identical two edits — delete `ItemForecast.confidence == "high",` from the `subq` filter and the outer filter, and replace `ItemForecast.current_price > 0,` with `price_floor_clause(ItemForecast.current_price),`.

In `/momentum` (lines 201-246): there is no confidence gate to remove; replace `ItemForecast.current_price > 0,` with `price_floor_clause(ItemForecast.current_price),`.

- [x] **Step 4: Run tests to verify they pass**

```bash
cd backend && python3 -m pytest tests/test_opportunity_selection.py tests/test_serving_policy.py -v
```

Expected: PASS, 20 tests. Then confirm the module still imports cleanly:

```bash
cd backend && python3 -c "import api.routes.opportunities; print('ok')"
```

Expected: `ok`.

- [x] **Step 5: Commit**

```bash
git add backend/api/routes/opportunities.py backend/tests/test_opportunity_selection.py
git commit -m "fix: stop gating opportunities on the anti-predictive confidence flag"
```

---

### Task 3: Stop ranking the trending list by confidence

**Files:**
- Modify: `backend/api/routes/items.py:93-127` (`_build_trending`)
- Test: `backend/tests/test_trending_ranking.py`

**Interfaces:**
- Consumes: `price_floor_clause` from Task 1.
- Produces: nothing consumed by later tasks.

- [x] **Step 1: Write the failing test**

Create `backend/tests/test_trending_ranking.py`:

```python
"""The trending list must not rank by confidence.

``_build_trending`` ordered by ``desc(confidence_order)`` first, so the
anti-predictive high-confidence subset occupied the top of the list before
predicted return was considered at all. The ``"medium"`` tier it ranked between
high and low has never been emitted once — 0 rows in the entire forecast
history.

See docs/specs/2026-08-03-served-forecast-surface-design.md.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import api.routes.items as items_mod


class TestTrendingDoesNotRankByConfidence:
    def test_confidence_order_is_gone(self):
        source = inspect.getsource(items_mod._build_trending)
        assert "confidence_order" not in source

    def test_dead_medium_tier_is_gone(self):
        source = Path(items_mod.__file__).read_text()
        assert '"medium"' not in source

    def test_trending_applies_the_price_floor(self):
        source = inspect.getsource(items_mod._build_trending)
        assert "price_floor_clause" in source
```

- [x] **Step 2: Run test to verify it fails**

```bash
cd backend && python3 -m pytest tests/test_trending_ranking.py -v
```

Expected: FAIL on all three — `confidence_order` and `"medium"` are present, `price_floor_clause` is not.

- [x] **Step 3: Write the implementation**

In `backend/api/routes/items.py`, add to the import block after line 16 (`from api.cache import get_or_build`):

```python
from api.serving_policy import price_floor_clause
```

In `_build_trending`, change the local import on line 94 from `from sqlalchemy import case` to nothing — delete that line, since `case` is only used by `confidence_order`. Keep `from datetime import date` on line 95.

Add the floor to the `subq` filter (line 108) so it reads:

```python
        .filter(
            ItemForecast.forecast_date == today,
            ItemForecast.horizon_days == 7,
            price_floor_clause(ItemForecast.current_price),
        )
```

Delete the `confidence_order` block (lines 114-118) entirely, and change the `order_by` on line 124 to drop it:

```python
        .order_by(desc(subq.c.price_mid / func.nullif(subq.c.current_price, 0)))
```

Leave `subq.c.confidence` in the `db.query(...)` select list on line 121 and the `Item.icon_url` / `backfilled_item_clause()` filter untouched. The floor sits on the joined forecast, not on `Item`, so a sub-$1 item still appears in the list — it simply carries no forecast annotation. That is intentional: this is a list of items, and only the forecast attached to it is subject to the floor.

- [x] **Step 4: Run tests to verify they pass**

```bash
cd backend && python3 -m pytest tests/test_trending_ranking.py -v && python3 -c "import api.routes.items; print('ok')"
```

Expected: PASS, 3 tests, then `ok`.

- [x] **Step 5: Commit**

```bash
git add backend/api/routes/items.py backend/tests/test_trending_ranking.py
git commit -m "fix: rank trending by predicted return instead of confidence"
```

---

### Task 4: Stop asserting a confidence level in user-facing copy

**Files:**
- Modify: `backend/api/routes/items.py:508-513` (`_build_trend_explanation`) and its two call sites at `items.py:367` and `items.py:449`
- Test: `backend/tests/test_trend_explanation_copy.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `_build_trend_explanation(direction: str, current_price) -> str` — the `confidence` parameter is removed.

- [x] **Step 1: Write the failing test**

Create `backend/tests/test_trend_explanation_copy.py`:

```python
"""Explanation copy must not state a confidence level.

"Confidence is high" was printed for forecasts realizing 4-31% directional
accuracy against 25-55% for the forecasts labelled low. The claim is not merely
unhelpful, it is backwards, so the sentence is removed rather than reworded.
The ``confidence`` field stays on the response schema — the frontend consumes
it and removing a field is a breaking change — it is simply no longer narrated.

See docs/specs/2026-08-03-served-forecast-surface-design.md.
"""
from __future__ import annotations

import inspect

import pytest

from api.routes.items import _build_trend_explanation


class TestExplanationCopy:
    @pytest.mark.parametrize("direction", ["bullish", "bearish", "neutral"])
    def test_no_direction_mentions_confidence(self, direction):
        text = _build_trend_explanation(direction, 12.50)
        assert "onfidence" not in text

    def test_bullish_still_describes_the_direction(self):
        assert "upward" in _build_trend_explanation("bullish", 12.50)

    def test_bearish_still_describes_the_direction(self):
        assert "downward" in _build_trend_explanation("bearish", 12.50)

    def test_neutral_still_describes_the_direction(self):
        assert "stable" in _build_trend_explanation("neutral", 12.50)

    def test_signature_no_longer_takes_confidence(self):
        params = list(inspect.signature(_build_trend_explanation).parameters)
        assert "confidence" not in params
```

- [x] **Step 2: Run test to verify it fails**

```bash
cd backend && python3 -m pytest tests/test_trend_explanation_copy.py -v
```

Expected: FAIL — `TypeError: _build_trend_explanation() missing 1 required positional argument: 'current_price'`, because the current signature takes three arguments.

- [x] **Step 3: Write the implementation**

Replace `_build_trend_explanation` (`items.py:508-513`):

```python
def _build_trend_explanation(direction: str, current_price) -> str:
    if direction == "bullish":
        return "ML forecast predicts upward movement over the next 7 days."
    elif direction == "bearish":
        return "ML forecast predicts downward movement over the next 7 days."
    return "ML forecast predicts a stable price over the next 7 days."
```

Update the call site in `_trends_parquet` (`items.py:367`):

```python
    explanation = _build_trend_explanation(trend_dir, current_price)
```

Update the call site in the DB trends path (`items.py:449`):

```python
    explanation = _build_trend_explanation(trend_dir, current_price)
```

Leave the `confidence = ...` assignments on lines 359 and 447 in place — both still populate `TrendAnalysisOut.confidence`, which stays on the schema.

- [x] **Step 4: Run tests to verify they pass**

```bash
cd backend && python3 -m pytest tests/test_trend_explanation_copy.py -v && grep -n "_build_trend_explanation" api/routes/items.py
```

Expected: PASS, 7 tests. The `grep` must show exactly three lines — the definition and two call sites — each with two arguments.

- [x] **Step 5: Commit**

```bash
git add backend/api/routes/items.py backend/tests/test_trend_explanation_copy.py
git commit -m "fix: stop narrating a confidence level in trend explanations"
```

---

### Task 5: Remove the confidence badge from the item page

**Files:**
- Modify: `frontend/app/items/[id]/page.tsx:426`

**Interfaces:**
- Consumes: nothing. The `TrendResponse.confidence` type on line 47 and the `confidence` local on line 287 stay — the API still returns the field.

**Scope note:** the design spec lists only backend files. This task is included because the spec's stated goal is the accuracy of what the product *shows*, and a page rendering "Confidence high" on a forecast realizing ~25% accuracy defeats that goal regardless of what the backend copy says. Flag it if you would rather ship backend-only.

- [x] **Step 1: Find the badge and confirm the current state**

```bash
cd frontend && grep -n "Confidence" app/items/\[id\]/page.tsx
```

Expected: two matches. Line 426 — `sub={\`Confidence ${confidence}\`}` — is the forecast badge and the one to remove. Line 623 — `Confidence: {imp.confidence_score.toFixed(2)}` — is an event-impact correlation score, unrelated to forecast confidence; **leave it alone.**

- [x] **Step 2: Remove the sub-label**

On line 426, delete the whole `sub={...}` prop from that `MetricCard`, leaving its other props untouched. Do not delete the `confidence` local on line 287 or the type on line 47.

- [x] **Step 3: Verify nothing else renders it and the app still builds**

```bash
cd frontend && grep -rn "Confidence \${" app/ ; npm run build
```

Expected: the `grep` returns nothing (exit 1 is fine), and the build completes without type errors. If `npm run build` is not runnable in this environment, run `npx tsc --noEmit` instead and expect no new errors.

- [x] **Step 4: Commit**

```bash
git add "frontend/app/items/[id]/page.tsx"
git commit -m "fix: stop displaying the anti-predictive confidence badge"
```

---

### Task 6: Fail the forecast run when it persists nothing

**Files:**
- Create: `backend/scripts/check_forecast_freshness.py`
- Create: `backend/tests/test_forecast_freshness.py`
- Modify: `.github/workflows/price-forecast.yml` (new step after "Run ML price forecasting", around line 130)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `newest_forecast_date(dates: list[date | None]) -> date | None` and `freshness_verdict(db_newest, parquet_newest, expected) -> tuple[bool, str]` — the pure decision, returning `(ok, message)`.

- [x] **Step 1: Write the failing test**

Create `backend/tests/test_forecast_freshness.py`:

```python
"""A green forecast run is not evidence that forecasts exist.

``item_forecasts`` holds 5 distinct forecast dates across the eight months from
2025-12-01 — 2026-07-17, 07-18, 07-19 and 07-29, four of them manual local
runs. Nothing failed; nothing checked. The same silent-success shape emptied the
supply scraper and reddit collector (docs/changelog/
2026-07-31-accuracy-work-closed.md).

Both stores are asserted because the API reads Parquet first with a DB
fallback, so a DB-only write still serves nothing.

See docs/specs/2026-08-03-served-forecast-surface-design.md.
"""
from __future__ import annotations

from datetime import date

from scripts.check_forecast_freshness import freshness_verdict, newest_forecast_date

TODAY = date(2026, 8, 3)
YESTERDAY = date(2026, 8, 2)


class TestNewestForecastDate:
    def test_picks_the_maximum(self):
        assert newest_forecast_date([YESTERDAY, TODAY]) == TODAY

    def test_ignores_none_entries(self):
        assert newest_forecast_date([None, YESTERDAY]) == YESTERDAY

    def test_empty_is_none(self):
        assert newest_forecast_date([]) is None

    def test_all_none_is_none(self):
        assert newest_forecast_date([None, None]) is None


class TestFreshnessVerdict:
    def test_both_stores_current_passes(self):
        ok, msg = freshness_verdict(TODAY, TODAY, TODAY)
        assert ok is True
        assert "2026-08-03" in msg

    def test_db_stale_fails(self):
        ok, msg = freshness_verdict(YESTERDAY, TODAY, TODAY)
        assert ok is False
        assert "item_forecasts" in msg

    def test_parquet_stale_fails_even_when_db_is_current(self):
        """The regression that matters: the API reads Parquet first."""
        ok, msg = freshness_verdict(TODAY, YESTERDAY, TODAY)
        assert ok is False
        assert "Parquet" in msg

    def test_empty_db_fails(self):
        ok, msg = freshness_verdict(None, TODAY, TODAY)
        assert ok is False

    def test_empty_parquet_fails(self):
        ok, msg = freshness_verdict(TODAY, None, TODAY)
        assert ok is False

    def test_a_date_ahead_of_expected_passes(self):
        """A forecast dated tomorrow is odd but it is not staleness."""
        ok, _ = freshness_verdict(date(2026, 8, 4), date(2026, 8, 4), TODAY)
        assert ok is True
```

- [x] **Step 2: Run test to verify it fails**

```bash
cd backend && python3 -m pytest tests/test_forecast_freshness.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.check_forecast_freshness'`.

- [x] **Step 3: Write the implementation**

Create `backend/scripts/check_forecast_freshness.py`:

```python
#!/usr/bin/env python3
"""Fail when a forecast run did not persist today's forecasts.

Green CI is not evidence of collection — see the collector audit in
docs/changelog/2026-07-31-accuracy-work-closed.md, where three scheduled jobs
reported success while storing zero rows. ``item_forecasts`` accumulated only 5
distinct forecast dates in eight months for the same reason: nothing asserted
the output.

Both stores are checked because ``api/routes`` reads the Parquet mirror first
and falls back to the DB, so a DB row without a mirror row serves nothing.

Usage:
    python scripts/check_forecast_freshness.py
    python scripts/check_forecast_freshness.py --expected-date 2026-08-03
"""

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import func

from database import SessionLocal, ItemForecast
from db.parquet import ParquetQuery

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def newest_forecast_date(dates):
    """Maximum non-null date, or None."""
    present = [d for d in dates if d is not None]
    if not present:
        return None
    return max(present)


def freshness_verdict(db_newest, parquet_newest, expected):
    """Decide whether both stores carry a forecast for *expected* or later."""
    problems = []
    if db_newest is None:
        problems.append("item_forecasts (DB) holds no forecasts at all")
    elif db_newest < expected:
        problems.append(
            f"item_forecasts (DB) newest forecast_date is {db_newest}, expected {expected}"
        )
    if parquet_newest is None:
        problems.append("the item_forecasts Parquet mirror holds no forecasts at all")
    elif parquet_newest < expected:
        problems.append(
            f"the item_forecasts Parquet mirror newest forecast_date is "
            f"{parquet_newest}, expected {expected}"
        )
    if problems:
        return False, "; ".join(problems)
    return True, f"Both stores carry forecasts for {expected} (DB {db_newest}, Parquet {parquet_newest})."


def _as_date(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)[:10]).date()


def _db_newest():
    db = SessionLocal()
    try:
        return _as_date(db.query(func.max(ItemForecast.forecast_date)).scalar())
    finally:
        db.close()


def _parquet_newest():
    with ParquetQuery("item_forecasts") as q:
        return _as_date(q.scalar("SELECT max(forecast_date) FROM item_forecasts"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-date", default=None, help="ISO date; defaults to today (UTC).")
    args = parser.parse_args()

    expected = (
        datetime.fromisoformat(args.expected_date).date()
        if args.expected_date
        else datetime.utcnow().date()
    )

    ok, message = freshness_verdict(_db_newest(), _parquet_newest(), expected)
    if ok:
        logger.info(message)
        return 0
    logger.error(message)
    logger.error(
        "The forecast run reported success without persisting forecasts. The API "
        "serves the newest available forecast, so this degrades silently rather "
        "than erroring."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [x] **Step 4: Run tests to verify they pass**

```bash
cd backend && python3 -m pytest tests/test_forecast_freshness.py -v
```

Expected: PASS, 10 tests.

- [x] **Step 5: Add the workflow step**

In `.github/workflows/price-forecast.yml`, insert immediately after the "Run ML price forecasting" step (which ends around line 130) and before "Check for boosters to save":

```yaml
      # A run that persists nothing must not report success. item_forecasts
      # accumulated 5 forecast dates in eight months because nothing asserted
      # the output — the same shape as the collectors audited in
      # docs/changelog/2026-07-31-accuracy-work-closed.md. Skipped for
      # train-only, which is not expected to write forecasts.
      - name: Verify forecasts were persisted
        if: always() && steps.mode.outputs.mode != 'train-only'
        working-directory: ./backend
        env:
          DATABASE_URL: ${{ secrets.SUPABASE_DATABASE_URL }}
          ENVIRONMENT: production
        run: python scripts/check_forecast_freshness.py
```

- [x] **Step 6: Verify the workflow parses and the script runs locally**

```bash
python3 -c "import yaml,sys; d=yaml.safe_load(open('.github/workflows/price-forecast.yml')); \
names=[s['name'] for s in d['jobs']['forecast']['steps']]; print(names); \
assert 'Verify forecasts were persisted' in names"
cd backend && python3 scripts/check_forecast_freshness.py --expected-date 2026-07-29; echo "exit=$?"
```

Expected: the step name is listed. The local run against the checked-in Parquet exits 0 for `2026-07-29` (its newest date) — confirming the Parquet leg reads correctly. The DB leg depends on your local SQLite and may report the DB as empty, which correctly exits 1; if so, re-run with `--expected-date` matching your local data or accept the Parquet-leg confirmation.

- [x] **Step 7: Commit**

```bash
git add backend/scripts/check_forecast_freshness.py backend/tests/test_forecast_freshness.py .github/workflows/price-forecast.yml
git commit -m "feat: fail the forecast run when it persists no forecasts"
```

---

### Task 7: Record the change

**Files:**
- Create: `docs/changelog/2026-08-03-served-forecast-surface.md`
- Modify: `docs/specs/2026-08-03-served-forecast-surface-design.md` (status line only)

- [x] **Step 1: Write the changelog**

Create `docs/changelog/2026-08-03-served-forecast-surface.md` covering, with the numbers from the spec:

- **What changed:** the four `opportunities.py` confidence gates and the `items.py` confidence sort are gone; a `MIN_SERVED_PRICE_USD = 1.0` floor now applies to ranked surfaces; explanation copy and the item-page badge no longer state a confidence level; `price-forecast.yml` fails when it persists nothing.
- **Why:** the nine-cell within-date table showing high-confidence accuracy of 4-31% against low-confidence 25-55%, and the 84%-sub-$1 composition of the ranked surfaces against a headline computed over ≥$1 only.
- **Coverage effect:** 7d on 2026-07-29, `/undervalued`'s candidate pool goes 1,680 → 893 and `/overheated` → 509; 1,402 of 8,691 forecasts clear the floor.
- **What was deliberately not done:** no replacement confidence signal, no model/feature/training change, `_calibrate_confidence` and the `confidence` column retained and still scored.
- **Still open:** the four follow-ups from the spec, all gated on reaching `MIN_FORECAST_DATES = 20`.

- [x] **Step 2: Flip the spec status**

In `docs/specs/2026-08-03-served-forecast-surface-design.md`, change the status line from `**Status:** Approved, not yet implemented` to `**Status:** Implemented 2026-08-03`.

- [x] **Step 3: Run the whole suite**

```bash
cd backend && python3 -m pytest tests/ -q
```

Expected: the four new test files pass and no previously-passing test breaks. Record the actual pass/fail counts in the changelog rather than asserting success — if anything fails, report it and stop.

- [x] **Step 4: Commit**

```bash
git add docs/changelog/2026-08-03-served-forecast-surface.md docs/specs/2026-08-03-served-forecast-surface-design.md
git commit -m "docs: record the served forecast surface changes"
```

---

## Verification

After all tasks, from the repository root:

```bash
cd backend && python3 -m pytest tests/ -q
grep -rn "confidence" api/routes/opportunities.py || echo "no confidence references remain"
grep -rn "ItemForecast.confidence" api/routes/ || echo "no confidence filters remain"
python3 -c "import api.routes.items, api.routes.opportunities; print('routes import cleanly')"
```

The first `grep` should report nothing in `opportunities.py`. The second should report nothing across `api/routes/` except the `subq.c.confidence` select-list entry in `items.py`, which populates a response field and does not filter.

## Spec coverage

| Spec section | Task |
|---|---|
| Component 1 — price floor, `serving_policy.py`, tier-coupling test | 1, and applied in 2, 3 |
| Component 2 — remove confidence gates and sort, fix copy | 2, 3, 4, 5 |
| Component 3 — forecast freshness | 6 |
| Testing section | 1, 2, 3, 4, 6 |
| Follow-ups gated on Component 3 | Not implemented, recorded in Task 7 |

Two deviations from the spec, both deliberate:

1. **The spec said "the `items.py` forecast surfaces"** for the floor. This plan applies it to the trending *list* only, not to `/items/{id}/trends` or the prediction detail endpoints — flooring a detail page would blank the forecast for any item a user opens below $1, which is not what the floor is for.
2. **Task 5 touches the frontend**, which the spec's scope line does not list. Rationale is in the task.
