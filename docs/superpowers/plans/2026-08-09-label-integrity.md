# Label Integrity Implementation Plan

> # ✅ COMPLETE — all four tasks landed and merged, 2026-08-09/10
>
> Branch `label-integrity`, off `training-cost` @ `c311194`, merged to `main` at `58d681e`
> (pushed). Ledger: `.superpowers/sdd/2026-08-09-label-integrity/progress.md`.
>
> | Task | Status | Commits |
> |---|---|---|
> | 1 — label-voiding audit | ✅ complete, review clean | `3d582ab`, `5e7dc60` |
> | 2 — `n_ask_sources` | ✅ complete, review clean | `48fd352`, `4f828fd` |
> | 3 — composition test | ✅ instrument + 16 tests, write-up committed, re-run post-exclusion | `80c3e07`, `f833882`, `cddcf76`, `3c62a49`, `6f90ce0` |
> | 4 — trailing-window exclusion | ✅ complete; cache key moved v4 → **v6** in the same commit | `873148b`, `ac713cc`, `09e945e` |
>
> **The goal was met: the gate returned, and it lifted.** The reversal survives source-composition
> control (2026 set basis, post-exclusion: stable **+0.0838** vs unconditional **+0.0838**, 167–172
> dates), so the label is a tradeable return and Track C accuracy work is unblocked. Full result:
> `docs/research/2026-08-09-composition-stability.md`.
>
> ⚠️ **One thing this branch leaves owing: a retrain.** Task 4 changed the consensus, so every
> label from 2026-03 onward moved. The shipped artifact was trained before `873148b` and no
> workflow has run on the merged tree. Dispatch `price-forecast.yml` with **`mode=train-only`**
> (*not* `mode=full` — the artifact is a day old, so `full` skips the age gate's retrain branch
> and `FORCE_RETRAIN` is not a dispatch input). It will vote **cold**, since no `voted-v6-` cache
> key exists yet.
> **No stored A/B verdict is citable against a post-merge number.**
>
> **Task 1's open question is closed by measurement.** On the backfilled-only frame the detectors
> fire on cutovers **2026-03-22, 07-09, 07-10, 07-11, 07-12** and snapshots **2026-07-16, 07-22**.
> **2026-07-11 IS present**, so no special case is needed; 07-12 is present too and had never been
> documented. ⚠️ New finding: `_collection_shift_dates` *also* fires on 2026-04-16, 07-14 and 07-15
> on the **full-universe** frame. "Which dates are void" is a function of the universe handed to
> the detector — previously unrecorded.
>
> **Task 3 changed the question, and the plan's expected outcome was wrong.** Step 2's baseline
> **does not reproduce**, for a definitional reason: "composition stable" was computed with a NULL
> source never equal to itself, and every pre-2026 row has `source IS NULL` (9,417,947 item-days).
> The published "stable" cell therefore holds **no pre-2026 data**, and its 775 "changed" dates are
> ~716 pre-2026 dates classified as changed for want of a label. **The 0.1676 → 0.1006 fall is a
> 2013-2025 → 2026 regime difference, not composition control.** Within 2026, powered at 181–185
> dates: **stable +0.1027, changed +0.0967, unconditional +0.1023.** So the plan's "likely outcome
> is `underpowered`" note at the foot of this file is **superseded** — the cells came back powered.
>
> Caveat the write-up must carry: `n_ask_sources` is a count, not a set, so a swap at constant
> count reads stable; the instrument **under-detects** composition change, biasing toward the
> answer it found. Agreed basis — primary = 2026 era on **set**-based composition, secondary = full
> archive on the count basis with its assumption stated.
>
> **Four pre-flight rulings this plan's text does not reflect:**
> 1. `_apply_multi_source_voting` is a `@staticmethod` (called unbound at
>    `backtest/price_resolution.py:246`), so Task 4 Step 4's `self.BID_SOURCES` **cannot work**.
>    Keep it static; reference constants by module/class name.
> 2. `BID_SOURCES` lives in `models/item_parser.py:34`, not on the class. Define
>    `TRAILING_WINDOW_SOURCES` beside it and expose both as class-level aliases so Task 4's tests
>    pass verbatim.
> 3. ~~⚠️ **The CI cache key is `voted-v4-`, not `voted-v5-`.** Task 2 bumped the constant to 5
>    without moving the key. **Task 4 must move it v4 → v6.**~~ **Done in `873148b`** — key went
>    `voted-v4-` → `voted-v6-` alongside the constant, no v5 key left behind.
> 4. The working tree carries unrelated frontend work. `git add` only the paths a task names —
>    never `git add -A`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Answer whether the price signal survives source-composition control, and remove the two remaining basis errors from the consensus vote — so that every accuracy item downstream is either unblocked or honestly declared blocked on calendar time.

**Architecture:** Four tasks. Task 1 persists the label-voiding sets that already exist but are never written down. Task 2 adds `n_ask_sources` to the voted frame. Task 3 runs the powered-up composition test as a script whose output is committed. Task 4 removes the Steam trailing-window feeds from voting and bumps the cache version.

**Tech Stack:** Python 3.13 local / 3.11 CI, DuckDB, pandas, pytest. Run from `backend/` through `venv/bin/python`.

**Spec:** `docs/superpowers/specs/2026-08-09-label-integrity-design.md`

## Global Constraints

- Run tests as `venv/bin/python -m pytest tests/... -q` from `backend/`. **Never bare `pytest`.**
- **`backend/.env` points at production Supabase.** Every step here is read-only against the archive or a test. Do not add a step that writes to the DB.
- **Read prices through `db/archive.py::prices_relation`, never a raw glob.** A bare `read_parquet('prices-*.parquet')` silently returns the first file's schema and drops columns without erroring.
- **Any loader that globs the archive must apply `archive_universe_sql_filter()`** — bid sources, phase-collapsed names and phantom duplicate keys, all NULL-safe. A bare `NOT IN` drops 13 years of prices.
- **Never pass a bare horizon to a purge.** The embargo is `horizon + 13` via `models/forecaster.py::embargo_days`.
- **Never quote a directional accuracy on its own.** Beside `constant_call_accuracy` and `realised_down_rate`, with Pesaran–Timmermann as the headline.
- **Do not change the logic of `_snapshot_dates` or `_collection_shift_dates`.** Cutovers are detected from universe size and never from prices, so the detector cannot mask a real crash. `test_a_price_crash_is_never_flagged` must keep passing untouched.
- **Do not winsorise large daily returns.** 2025-10-22 is the only dated event where item attributes dominated the market factor.
- Commit after every task.

---

### Task 1: Persist the label-voiding audit

**Files:**
- Modify: `backend/models/forecaster.py:2999` — `prepare_targets` records what it voided
- Modify: `backend/models/forecaster.py` — `train()` writes `label_voiding` into `meta.json`
- Test: `backend/tests/test_label_voiding_audit.py` (create)

**Interfaces:**
- Consumes: `_snapshot_dates(df) -> frozenset` (`:2951`), `_collection_shift_dates(df) -> frozenset` (`:2975`)
- Produces: `ItemForecaster.label_voiding: dict` with keys `snapshot_dates`, `collection_shift_dates`, `voided_labels_by_horizon`, `frame_date_range`

- [x] **Step 1: Write the failing tests**

Create `backend/tests/test_label_voiding_audit.py`:

```python
"""The label path must say which dates it voided.

`_collection_shift_dates` fires 12 times in 4,735 days and the list of dates has
never been written down anywhere -- it exists only as a count in a code comment.
The measured cost of that: a 2026-08-09 audit re-reported the already-handled
2026-07-09/10 cutover as a new, undocumented consensus break, and put a no-op
task at the top of a roadmap.

These tests pin the audit trail. They do NOT touch either detector's logic --
the universe-size rule is what makes it impossible for the detector to mask a
real crash.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from models.forecaster import ItemForecaster


def _f(tmp_path):
    return ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))


def _frame_with_a_cutover(n_items=60, n_dates=40, cutover_at=20):
    """The item universe halves on one day -- a collector cutover by the
    universe-size rule, with no price move at all."""
    rows = []
    start = pd.Timestamp("2026-01-01")
    for d in range(n_dates):
        live = n_items if d < cutover_at else n_items // 3
        for item in range(live):
            rows.append({
                "item_id": f"item-{item}",
                "date": start + pd.Timedelta(days=d),
                "price": 10.0 + item * 0.01,
            })
    return pd.DataFrame(rows)


def test_audit_is_empty_before_prepare_targets(tmp_path):
    assert _f(tmp_path).label_voiding == {}


def test_audit_records_the_shift_date(tmp_path):
    f = _f(tmp_path)
    f.prepare_targets(_frame_with_a_cutover(), horizon=3)
    shifts = f.label_voiding["collection_shift_dates"]
    assert "2026-01-21" in shifts, f"cutover day not recorded: {shifts}"


def test_audit_records_the_frame_window(tmp_path):
    """Absence from the list must be distinguishable from 'outside the window'.
    The training window is days_back=1460, so the 2013 and 2016 cutovers fall
    outside it on most runs."""
    f = _f(tmp_path)
    f.prepare_targets(_frame_with_a_cutover(), horizon=3)
    lo, hi = f.label_voiding["frame_date_range"]
    assert lo == "2026-01-01"
    assert hi == "2026-02-09"


def test_audit_counts_voided_labels_per_horizon(tmp_path):
    f = _f(tmp_path)
    f.prepare_targets(_frame_with_a_cutover(), horizon=3)
    f.prepare_targets(_frame_with_a_cutover(), horizon=7)
    counts = f.label_voiding["voided_labels_by_horizon"]
    assert set(counts) == {3, 7}
    assert counts[7] >= counts[3], (
        "a longer horizon spans the cutover from more anchor dates, so it "
        "cannot void fewer labels")


def test_dates_are_sorted_iso_strings(tmp_path):
    """meta.json is JSON; a frozenset of datetime.date is not serialisable, and
    an unsorted list is not diffable across runs."""
    f = _f(tmp_path)
    f.prepare_targets(_frame_with_a_cutover(), horizon=3)
    for key in ("snapshot_dates", "collection_shift_dates"):
        got = f.label_voiding[key]
        assert isinstance(got, list)
        assert all(isinstance(d, str) for d in got)
        assert got == sorted(got)
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_label_voiding_audit.py -q`
Expected: FAIL — `AttributeError: 'ItemForecaster' object has no attribute 'label_voiding'`

- [x] **Step 3: Initialise the audit dict**

In `ItemForecaster.__init__`, beside the other per-run state:

```python
        # What the label path voided, recorded so a reader can tell a handled
        # cutover from an unhandled one. `_collection_shift_dates` fires 12
        # times in 4,735 days and that list has never been written down, which
        # is how a 2026-08-09 audit re-reported the handled 2026-07-09/10
        # cutover as a new finding.
        self.label_voiding: dict = {}
```

- [x] **Step 4: Record in `prepare_targets`**

In `prepare_targets` (`:2999`), after `snapshots` and `shifts` are computed and after the `bad`
mask is applied, accumulate rather than overwrite — `prepare_targets` is called once per horizon:

```python
        voided = int(df[f"target_return_{horizon}d"].isna().sum() - pre_void_na)
        counts = self.label_voiding.get("voided_labels_by_horizon", {})
        counts[horizon] = voided
        self.label_voiding = {
            "snapshot_dates": sorted(d.isoformat() for d in snapshots),
            "collection_shift_dates": sorted(d.isoformat() for d in shifts),
            "voided_labels_by_horizon": counts,
            "frame_date_range": [
                pd.to_datetime(df["date"]).min().date().isoformat(),
                pd.to_datetime(df["date"]).max().date().isoformat(),
            ],
        }
        logger.info(
            f"  Label voiding (h={horizon}): {voided:,} labels voided; "
            f"{len(shifts)} collector cutovers, {len(snapshots)} snapshot days; "
            f"cutovers: {sorted(d.isoformat() for d in shifts)}"
        )
```

Capture `pre_void_na` immediately before the `bad` mask is applied so the count measures the
voiding rather than pre-existing NaNs.

- [x] **Step 5: Write it into `meta.json`**

In `save_models()`, add `"label_voiding": self.label_voiding` to the metadata dict.

- [x] **Step 6: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_label_voiding_audit.py -q`
Expected: PASS, 5 tests

- [x] **Step 7: Verify the detectors are untouched**

Run: `venv/bin/python -m pytest tests/test_degenerate_label_dates.py -q`
Expected: PASS, including `test_a_price_crash_is_never_flagged`. If that test fails, a detector's
logic was changed and the diff must be reverted — the universe-size rule is what makes the
detector unable to delete a real market event.

- [x] **Step 8: Print the real fired-date list and answer the 2026-07-11 question**

```bash
cd backend
venv/bin/python - <<'PY'
from unittest.mock import MagicMock
from models.forecaster import ItemForecaster
f = ItemForecaster(db_session=MagicMock(), model_dir="/tmp/lv")
df = f.fetch_price_history(days_back=1460, backfilled_only=True)
print("snapshots:", sorted(d.isoformat() for d in f._snapshot_dates(df)))
print("cutovers:", sorted(d.isoformat() for d in f._collection_shift_dates(df)))
PY
```

Record the output. **The expected members are 2026-03-22 and 2026-07-09/10.** Note explicitly
whether **2026-07-11** is present — that is the one genuinely open question from the review, and
this command is the measurement that closes it. If it is absent, do not add a special case; open
it as a finding and size it before acting, because the universe-size rule is what keeps the
detector honest.

- [x] **Step 9: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_label_voiding_audit.py
git commit -m "feat: record which dates the label path voided, and why"
```

---

### Task 2: Add `n_ask_sources` to the voted frame

**Files:**
- Modify: `backend/models/forecaster.py:1206-1260` — `_apply_multi_source_voting`
- Modify: `backend/models/forecaster.py` — `VOTED_CACHE_VERSION` 4 → 5
- Test: `backend/tests/test_n_ask_sources.py` (create)

**Interfaces:**
- Consumes: `BID_SOURCES`, the voted frame's `source` column
- Produces: voted frame gains `n_ask_sources: int`, never NULL

- [x] **Step 1: Write the failing tests**

Create `backend/tests/test_n_ask_sources.py`:

```python
"""How many ask sources voted on each item-day.

The composition partition in docs/research/2026-08-08-model-review.md §5 -- which
takes the reversal signal from rank IC 0.168 to 0.101 and to ~0 on the cleanest
subset -- is currently recomputed ad hoc in a scratchpad. It needs to be a column
so the partition is auditable, and so a future basis-change detector has
something to key on that is not a price move.

It is also the best candidate cause of the harness's undiagnosed run-to-run
non-reproducibility (mean_diff_pp -0.1581 -> -0.0026 on identical commands).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from models.forecaster import ItemForecaster


def _f(tmp_path):
    return ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))


def _rows(*specs):
    return pd.DataFrame([
        {"item_id": i, "date": pd.Timestamp(d), "price": p, "source": s}
        for i, d, p, s in specs
    ])


def test_single_source_day_counts_one(tmp_path):
    out = _f(tmp_path)._apply_multi_source_voting(
        _rows(("a", "2026-07-01", 10.0, "aggregator_sync")))
    assert out["n_ask_sources"].iloc[0] == 1


def test_three_sources_count_three(tmp_path):
    out = _f(tmp_path)._apply_multi_source_voting(_rows(
        ("a", "2026-07-11", 10.0, "aggregator_buff163"),
        ("a", "2026-07-11", 10.2, "aggregator_youpin"),
        ("a", "2026-07-11", 9.9, "aggregator_csfloat"),
    ))
    assert out["n_ask_sources"].iloc[0] == 3


def test_bid_does_not_count_as_an_ask_source(tmp_path):
    """aggregator_buff163_buy is the wrong side of the book and was removed
    from voting 2026-08-07. It must not inflate the composition count either."""
    out = _f(tmp_path)._apply_multi_source_voting(_rows(
        ("a", "2026-07-11", 10.0, "aggregator_buff163"),
        ("a", "2026-07-11", 5.8, "aggregator_buff163_buy"),
    ))
    assert out["n_ask_sources"].iloc[0] == 1


def test_null_source_counts_one_not_zero(tmp_path):
    """Every pre-2026 row has source IS NULL -- 9.4M rows, 4,523 days. A NULL
    source is one backfill source, not the absence of one."""
    out = _f(tmp_path)._apply_multi_source_voting(
        _rows(("a", "2020-01-01", 10.0, None)))
    assert out["n_ask_sources"].iloc[0] == 1


def test_column_is_never_null(tmp_path):
    out = _f(tmp_path)._apply_multi_source_voting(_rows(
        ("a", "2020-01-01", 10.0, None),
        ("b", "2026-07-11", 3.0, "aggregator_skinport"),
    ))
    assert out["n_ask_sources"].notna().all()
    assert out["n_ask_sources"].dtype.kind in "iu"


def test_cache_version_bumped():
    assert ItemForecaster.VOTED_CACHE_VERSION >= 5, (
        "the voted frame gained a column; a stale cache would train the next "
        "model on a frame without it")
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_n_ask_sources.py -q`
Expected: FAIL — `KeyError: 'n_ask_sources'`, and the version assert fails at 4

- [x] **Step 3: Compute the count on both voting paths**

`_apply_multi_source_voting` splits single-source item-days into a vectorised `groupby().agg`
(`:1206-1217`) and routes only multi-source days through `groupby().apply(vote)`. **Both paths
need the column**, and it must be computed after the bid exclusion so a bid cannot inflate it.

On the vectorised path the count is 1 by construction. On the `apply` path, add
`n_ask_sources = int(group["source"].fillna("__null__").nunique())` inside `vote`.

- [x] **Step 4: Bump the cache version**

```python
    VOTED_CACHE_VERSION = 5   # v5: n_ask_sources on the voted frame
```

- [x] **Step 5: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_n_ask_sources.py -q`
Expected: PASS, 6 tests

- [x] **Step 6: Run the universe and voting tests**

Run: `venv/bin/python -m pytest tests/test_phase_collapsed_universe.py tests/test_forecaster.py -q`
Expected: PASS

- [x] **Step 7: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_n_ask_sources.py
git commit -m "feat: record how many ask sources voted on each item-day"
```

---

### Task 3: Run the powered-up composition test

**Files:**
- Create: `backend/scripts/measure_composition_stability.py`
- Create: `docs/research/2026-08-09-composition-stability.md` (the committed result)

**Interfaces:**
- Consumes: `db/archive.py::prices_relation`, `models/item_parser.py::archive_universe_sql_filter`, `_snapshot_dates`, `_collection_shift_dates`, `n_ask_sources` from Task 2

- [x] **Step 1: Write the script**

`backend/scripts/measure_composition_stability.py` — rank IC of `−r_t` against the forward
`--horizon`-day return on the voted daily series, ≥$1, partitioned by composition stability and
source count.

Requirements, each of which is a way the answer goes wrong if omitted:

```python
MIN_DATES_TO_REPORT = 30   # below this a cell reads `underpowered`, never a number
```

- Read through `prices_relation` with `archive_universe_sql_filter()` applied. Never a raw glob.
- Exclude every date in `_collection_shift_dates` and `_snapshot_dates`. The script reads the
  archive directly rather than through `prepare_targets`, so it must apply the exclusion itself.
- Partition on: composition stable across `t−1 … t+h`; composition changed; stable & single
  source; stable & `n_ask_sources >= 3`.
- Report per cell: `n_dates`, `n_rows`, `rank_ic`, `t`, and `verdict ∈ {measured, underpowered}`.
- Never pool across price tiers. `--min-price` defaults to 1.0.
- `--horizon` defaults to 3 to reproduce the §5 baseline; accept 7/14/30.

- [x] **Step 2: Reproduce the published baseline first**

```bash
cd backend
venv/bin/python scripts/measure_composition_stability.py --horizon 3 --from 2024-01-01
```

Expected, from `docs/research/2026-08-08-model-review.md` §5: all rows ≈ **+0.1676 on 932 dates**;
composition-stable ≈ **+0.1006 on 188**; stable & three sources ≈ **+0.0044 on 28**.

**If the baseline does not reproduce, stop.** A new number from an instrument that cannot
reproduce the old one is not evidence. The most likely causes are the date exclusion (the old
measurement may not have applied it) and the universe filter.

- [x] **Step 3: Run the powered-up version**

```bash
venv/bin/python scripts/measure_composition_stability.py --horizon 3 --from 2013-08-14
venv/bin/python scripts/measure_composition_stability.py --horizon 7 --from 2013-08-14
```

- [x] **Step 4: Write up the result, including a null result** — ✅ **COMPLETE** (`f833882`, extended by `cddcf76`, `3c62a49`, `6f90ce0`). `docs/changelog/2026-08-09-composition-stability-refutes-quoting-artifact.md` written; `2026-08-08-model-review.md` §5 corrected in place (rows struck, numbers kept because they reproduce exactly under the old rule); `docs/research/2026-08-09-composition-stability.md` is 378 lines carrying both bases, the three-way stable/changed/gapped partition, the post-`873148b` re-run and the reproduction commands. The `<!--RESULTS-->` placeholder is gone

Create `docs/research/2026-08-09-composition-stability.md`. State plainly which of the three
outcomes occurred:

| Outcome | What it means | What happens to Track C |
|---|---|---|
| Signal survives composition control at a reportable date count | the label is a tradeable return | **unblocked** |
| Signal vanishes when composition is held stable | the label is a quoting artifact | Track C is **dead**, and that is the headline result of the project |
| The clean cells stay under `MIN_DATES_TO_REPORT` | **unresolvable until more multi-source days accumulate** | Track C is **blocked on calendar time, not effort** — record it and stop |

The third is the likely one and it is a legitimate answer. The multi-source era is **24 days
deep**: `csgotrader`/`skinport`/`csmoney`/`steam_*`/`buff163_buy` span 2026-07-11 → 08-07 only,
`buff163`/`youpin`/`csfloat` have 49 days present of 139, and 2026-04-16 → 07-10 is `17mafo`
alone. Do not work around it by lowering `MIN_DATES_TO_REPORT`.

Quote the 28-date cell **only ever with its date count.**

- [x] **Step 5: Commit** — landed as `80c3e07`, **script and tests only, deliberately no doc**

```bash
git add backend/scripts/measure_composition_stability.py docs/research/2026-08-09-composition-stability.md
git commit -m "feat: measure the reversal signal against source-composition stability"
```

---

### Task 4: Stop the Steam trailing-window feeds voting

**Files:**
- Modify: `backend/models/forecaster.py` — add `TRAILING_WINDOW_SOURCES`, exclude in `_apply_multi_source_voting`
- Modify: `backend/models/forecaster.py` — `VOTED_CACHE_VERSION` 5 → 6
- Modify: `.github/workflows/price-forecast.yml` — cache key `voted-v5-` → `voted-v6-` if Track A Task 6 already landed
- Test: `backend/tests/test_trailing_window_sources.py` (create)
- Create: `docs/changelog/2026-08-09-trailing-window-sources-excluded.md`

**Interfaces:**
- Produces: `ItemForecaster.TRAILING_WINDOW_SOURCES: frozenset`

- [x] **Step 1: Write the failing tests**

Create `backend/tests/test_trailing_window_sources.py`:

```python
"""Steam's trailing-window means must not vote against point-in-time asks.

aggregator_steam_7d/30d/90d are MA(7)/MA(30)/MA(90) of the SALE price
(collectors/csgotrader_aggregator.py:308-312). Same class of basis error as
aggregator_buff163_buy, removed 2026-08-07: excluding them costs 670 item-days
of 3,093,793 but moves the voted median on 17.13% of 2026 >=$1 item-days, median
-7.16%, flipping 5.75% of return directions.

This is NOT a staleness fix -- it clears 2.30pp of the 20.25pp >=$1 stale rate,
because aggregator_sync and aggregator_steam_17mafo are last_24h FALLING BACK to
the same windows. There is no point-in-time Steam price in this archive at all.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from models.forecaster import ItemForecaster


def _f(tmp_path):
    return ItemForecaster(db_session=MagicMock(), model_dir=str(tmp_path))


def _rows(*specs):
    return pd.DataFrame([
        {"item_id": i, "date": pd.Timestamp(d), "price": p, "source": s}
        for i, d, p, s in specs
    ])


def test_the_three_windows_are_named():
    assert ItemForecaster.TRAILING_WINDOW_SOURCES == frozenset({
        "aggregator_steam_7d", "aggregator_steam_30d", "aggregator_steam_90d"})


def test_they_are_separate_from_bid_sources():
    """A bid is the wrong side of the book; an MA is the wrong time basis.
    Overloading BID_SOURCES would make the next reader think these are bids."""
    assert not (ItemForecaster.TRAILING_WINDOW_SOURCES
                & set(ItemForecaster.BID_SOURCES))


def test_trailing_windows_do_not_move_the_median(tmp_path):
    """Two asks and three MA legs: the vote must be the asks' median alone."""
    out = _f(tmp_path)._apply_multi_source_voting(_rows(
        ("a", "2026-07-11", 10.0, "aggregator_buff163"),
        ("a", "2026-07-11", 10.4, "aggregator_csfloat"),
        ("a", "2026-07-11", 7.0, "aggregator_steam_7d"),
        ("a", "2026-07-11", 6.5, "aggregator_steam_30d"),
        ("a", "2026-07-11", 6.0, "aggregator_steam_90d"),
    ))
    assert out["price"].iloc[0] == 10.2


def test_a_trailing_only_item_day_is_dropped_not_zeroed(tmp_path):
    """670 item-days of 3,093,793 have no other source. They must vanish, not
    become a 0 or a NaN price."""
    out = _f(tmp_path)._apply_multi_source_voting(
        _rows(("a", "2026-07-11", 7.0, "aggregator_steam_7d")))
    assert len(out) == 0


def test_exclusion_is_null_safe(tmp_path):
    """Invariant 2: a bare NOT IN against a NULL source drops 13 years of
    prices. Every pre-2026 row has source IS NULL."""
    out = _f(tmp_path)._apply_multi_source_voting(_rows(
        ("a", "2020-01-01", 10.0, None),
        ("b", "2020-01-01", 20.0, None),
    ))
    assert len(out) == 2


def test_n_ask_sources_excludes_trailing_windows(tmp_path):
    out = _f(tmp_path)._apply_multi_source_voting(_rows(
        ("a", "2026-07-11", 10.0, "aggregator_buff163"),
        ("a", "2026-07-11", 7.0, "aggregator_steam_7d"),
    ))
    assert out["n_ask_sources"].iloc[0] == 1


def test_cache_version_bumped():
    assert ItemForecaster.VOTED_CACHE_VERSION >= 6
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_trailing_window_sources.py -q`
Expected: FAIL — `TRAILING_WINDOW_SOURCES` does not exist

- [x] **Step 3: Add the constant**

Beside `BID_SOURCES`:

```python
    # Steam's trailing-window MEAN SALE price -- MA(7)/MA(30)/MA(90) -- which
    # must not vote on equal terms against point-in-time asks
    # (collectors/csgotrader_aggregator.py:308-312). Same class of basis error
    # as BID_SOURCES, kept separate because it is a different error: a bid is
    # the wrong side of the book, an MA is the wrong time basis.
    #
    # Measured: excluding them costs 670 item-days of 3,093,793 on the >=$1
    # cohort, but moves the voted median on 17.13% of 2026 >=$1 item-days
    # (median -7.16%) and flips 5.75% of consecutive-day return directions.
    #
    # NOT a staleness fix: clears 2.30pp of the 20.25pp >=$1 stale rate, because
    # aggregator_sync and aggregator_steam_17mafo are `last_24h` FALLING BACK to
    # these same windows on exactly the illiquid items. There is no
    # point-in-time Steam price in this archive at all.
    #
    # Do NOT also drop aggregator_sync: that deletes 2026-01 and 2026-02 in full
    # for the >=$1 cohort (52,048 item-days) to buy a further 1.4pp.
    TRAILING_WINDOW_SOURCES = frozenset({
        "aggregator_steam_7d", "aggregator_steam_30d", "aggregator_steam_90d",
    })
```

- [x] **Step 4: Apply the exclusion, NULL-safely**

In `_apply_multi_source_voting`, alongside the `BID_SOURCES` drop and **before** the
`n_ask_sources` count so a trailing window cannot inflate it:

```python
        excluded = set(self.BID_SOURCES) | set(self.TRAILING_WINDOW_SOURCES)
        # NULL-safe by construction: `.isin` is False for NaN, so a NULL source
        # (every pre-2026 row, 9.4M of them) is kept. Invariant 2.
        df = df[~df["source"].isin(excluded)]
```

- [x] **Step 5: Bump the cache version and the workflow key**

```python
    VOTED_CACHE_VERSION = 6   # v6: Steam trailing-window sources out of voting
```

If Track A Task 6 has landed, move the `price-forecast.yml` cache key from `voted-v5-` to
`voted-v6-` **in this same commit**. Otherwise CI restores a frame voted under the old rule.

- [x] **Step 6: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_trailing_window_sources.py tests/test_n_ask_sources.py -q`
Expected: PASS, 13 tests

- [x] **Step 7: Measure the coverage cost against the real archive**

```bash
cd backend
venv/bin/python - <<'PY'
from unittest.mock import MagicMock
from models.forecaster import ItemForecaster
f = ItemForecaster(db_session=MagicMock(), model_dir="/tmp/tw")
df = f.fetch_price_history(days_back=1460, backfilled_only=True)
print("voted item-days:", len(df))
PY
```

Expected: **≈670 fewer ≥$1 item-days than the pre-change frame, out of 3,093,793.** If the loss is
materially larger, the exclusion is matching more than the three intended sources — check it is not
prefix-matching `aggregator_steam_17mafo`, which must be kept (it is the only feed for
2026-04-16 → 07-10).

- [x] **Step 8: Write the changelog entry**

Create `docs/changelog/2026-08-09-trailing-window-sources-excluded.md`. Record the measured
displacement (17.13% of item-days, median −7.16%, 5.75% direction flips), the coverage cost, the
cache-version bump, and — prominently — that **every A/B and every label from 2026-03 onward sits
downstream of this**, so no stored A/B verdict predating it is citable. State explicitly that it
is not a staleness fix and that `aggregator_steam_17mafo` remains unaudited and is subsumed here.

- [x] **Step 9: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_trailing_window_sources.py \
        .github/workflows/price-forecast.yml docs/changelog/
git commit -m "fix: stop Steam's trailing-window means voting against point-in-time asks"
```

---

## Self-review notes

- **Task 2 and Task 4 both bump `VOTED_CACHE_VERSION`** (4 → 5 → 6). If they land in one session,
  a single bump to 6 is fine; if they land separately, both bumps are required, and Track A Task
  6's workflow cache key must follow whichever lands last.
- **Task 4's exclusion must run before Task 2's count.** Step 4 states the ordering; the
  `test_n_ask_sources_excludes_trailing_windows` case is the guard.
- **Task 3 depends on Task 1** (importable voided-date sets) and **Task 2** (`n_ask_sources`).
  Running it before Task 4 is correct — the baseline must reproduce on the *current* consensus, or
  the 6c change confounds the comparison. Re-run it after Task 4 as a separate reading.
- **No task changes a detector's logic.** `test_a_price_crash_is_never_flagged` is the guard, and
  Task 1 Step 7 checks it explicitly.
- ~~**The likely outcome of Task 3 is `underpowered`.**~~ **Wrong — superseded 2026-08-09.** The
  2026 cells came back **powered at 181–185 dates**, and the signal did not move under composition
  control (stable +0.1027 vs changed +0.0967). What was underpowered was the *published* 28-date
  cell, which turns out to have been a pre-2026-vs-2026 split rather than a composition split. The
  plan's framing — that a recorded null is a completed task, not a failure — still holds; it just
  is not the outcome that occurred.
