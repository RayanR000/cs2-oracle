# C2 lambdarank serving-transfer read — Implementation Plan

> # ❌ EXECUTED AND REFUTED (2026-08-14) — not pending work
>
> The driver and scoring core were built (`ccf7ad0`, `c2277e9`, `b2c7d0c`, `5586bbf`) and the
> read ran. Serving transfer **fails at 3 of 4 horizons and h=14 is degenerate**, so lambdarank
> was not adopted at serving. Outcome:
> `docs/changelog/2026-08-14-lambdarank-serving-transfer-measured.md` and
> `docs/changelog/2026-08-13-lambdarank-clears-the-diagnostic-bar.md` (the offline half that did
> pass). Do not re-execute the tasks below.
>
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether the CV lambdarank vs-q50 rank-IC edge (+0.026–0.065, folds ending 2026-02-09) survives to the 2026-04-18→06-08 served window on the tied cohort — read-only, writing nothing.

**Architecture:** A standalone diagnostic `scripts/replay_lambdarank.py` builds the engineered+targeted master frame per horizon (reusing `fetch_price_history`→`engineer_features`→`prepare_targets`), retrains a matched q50+lambdarank pair at the 14-day production cadence, and scores each frozen served-window anchor date's tied cross-section against replay-resolved outcomes. Three rulers: within-date rank IC (primary, edge lr−q50), decile long-short spread, and a Pesaran-Timmermann test on the ranker's within-date directional call. Reuses `replay_serving.py`'s referee and `_lambdarank_fold_scores`; adds one production sibling method and one pure metrics module.

**Tech Stack:** Python 3.13 (local) / 3.11 (CI), LightGBM, DuckDB, pandas, numpy, pytest. Run through `backend/venv/bin/python` from `backend/`.

## Global Constraints

- Run everything from `backend/` through `venv/bin/python`. **`.env` there points at production** — this script reads the archive and trains locally; it must never write the DB. It opens a `SessionLocal()` only for the `is_backfilled` flag + events metadata (same as any forecaster read).
- **Tied cohort is the verdict.** `p[d]/S[d] == 1` rows only, via `replay_serving._tied_mask`. Pooled is descriptive, never quoted as the result.
- **Embargo is `horizon + 13`**, from `ItemForecaster.embargo_days(horizon)`. Never pass a bare horizon.
- **Never quote DA alone** (backend/AGENTS.md invariant 4). The ranker emits an ordinal score — no MAE/DA/coverage for it; only rank IC, the decile spread, and the PT test.
- **No look-ahead:** each booster trains only on rows whose label resolves at or before its retrain-point cutoff; outcomes are resolved with `after=anchor`.
- Anchor set is **frozen and logged before any rank IC is read**. `ALLOW_DIRTY_ANCHOR` is not set.
- Keep each dispatch **under 30 minutes**; shard by horizon (`--horizon 3|7|14|30`).
- Reuse `replay_serving.py`'s referee helpers by import; do **not** fork them (they are single-source by design).

---

### Task 1: `_fold_q50_scores` — matched q50 sibling of the ranker trainer

**Files:**
- Modify: `backend/models/forecaster.py` (add method next to `_lambdarank_fold_scores`, ~line 8575)
- Test: `backend/tests/test_lambdarank_diagnostic.py` (add a `TestFoldQ50Scores` class)

**Interfaces:**
- Consumes: `self.feature_cols`, `self.QUANTILES`, `self.MAX_BIN`, `self.BOOSTING_TYPE`, `self._boost_rounds`, `self._compute_sample_weights`, `self._naive_offset` (all existing).
- Produces: `ItemForecaster._fold_q50_scores(self, train_df, val_df, horizon: int, per_quantile_params: dict) -> np.ndarray` — raw q50 predictions on `val_df` rows in `val_df` order, matched to `_lambdarank_fold_scores` (same rows, fillna, fixed rounds, tree HP from `per_quantile_params[0.5]`). Asserts sample-weights and N1-offset are inert so the "matched contrast" claim holds.

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_lambdarank_diagnostic.py`:

```python
class TestFoldQ50Scores:
    """The q50 twin of the ranker trainer: same rows/HP, quantile objective."""

    def _forecaster(self, feature_cols):
        fc = ItemForecaster.__new__(ItemForecaster)
        fc.feature_cols = feature_cols
        return fc

    def _frame(self, dates, feats, rng):
        n = len(dates)
        data = {"date": dates, "target_return_7d": rng.normal(size=n),
                "price": rng.uniform(1, 100, size=n)}
        for f in feats:
            data[f] = rng.normal(size=n)
        return pd.DataFrame(data)

    def test_q50_scores_align_to_val_rows(self):
        rng = np.random.default_rng(3)
        feats = ["f0", "f1", "f2"]
        train_df = self._frame(
            np.repeat(pd.to_datetime(["2026-01-01", "2026-01-02"]), 60),
            feats, rng)
        val_df = self._frame(
            np.repeat(pd.to_datetime(["2026-01-10", "2026-01-11"]), 30),
            feats, rng)
        fc = self._forecaster(feats)
        scores = fc._fold_q50_scores(train_df, val_df, 7, {0.5: {}})
        assert len(scores) == len(val_df)
        assert np.isfinite(np.asarray(scores)).all()

    def test_q50_learns_a_monotone_signal(self):
        # A frame where the target is f0 + noise: q50 rank must correlate with f0.
        rng = np.random.default_rng(4)
        n = 400
        dates = np.repeat(pd.to_datetime(["2026-01-01", "2026-01-02"]), n // 2)
        f0 = rng.normal(size=n)
        train_df = pd.DataFrame({"date": dates, "f0": f0,
                                 "price": rng.uniform(1, 100, size=n),
                                 "target_return_7d": f0 + 0.1 * rng.normal(size=n)})
        val_df = train_df.copy()
        fc = self._forecaster(["f0"])
        scores = np.asarray(fc._fold_q50_scores(train_df, val_df, 7, {0.5: {}}))
        # Spearman of scores vs f0 should be strongly positive.
        assert pd.Series(scores).corr(pd.Series(f0), method="spearman") > 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_lambdarank_diagnostic.py::TestFoldQ50Scores -q`
Expected: FAIL with `AttributeError: 'ItemForecaster' object has no attribute '_fold_q50_scores'`.

- [ ] **Step 3: Write minimal implementation**

Add to `backend/models/forecaster.py` immediately after `_lambdarank_fold_scores` (after its `return model.predict(X_val)`):

```python
    def _fold_q50_scores(self, train_df, val_df, horizon,
                         per_quantile_params):
        """The q50 twin of `_lambdarank_fold_scores`: same rows, same tree HP,
        `objective="quantile"` at alpha 0.5.

        The serving-transfer read (scripts/replay_lambdarank.py) needs a q50
        baseline trained on IDENTICAL rows/HP/embargo as the ranker, so the
        rank-IC delta between them isolates the objective — the "transferable
        number" the diagnostic changelog defined. Production's own fold q50
        (`_cv_evaluate_horizon`) carries sample weights and the N1 offset; both
        are inert under the shipped config (SAMPLE_WEIGHT_HALFLIFE_DAYS=0.0,
        NAIVE_INIT_SCORE off), so this plain fit reproduces it. The assert makes
        that assumption loud rather than silent: if either is ever turned on,
        this matched contrast stops being matched and must be revisited.
        """
        tcol = f"target_return_{horizon}d"
        # Guard the "matched to production q50" claim (see docstring).
        assert self._naive_offset(train_df) is None, (
            "_fold_q50_scores assumes the N1 offset is off; it is on")
        w = self._compute_sample_weights(train_df, horizon)
        assert w is None or np.allclose(np.asarray(w), np.asarray(w)[0]), (
            "_fold_q50_scores assumes uniform sample weights; they are not")
        feat = train_df[self.feature_cols].replace([np.inf, -np.inf], np.nan)
        med = feat.median()
        X_tr = feat.fillna(med)
        q50 = per_quantile_params.get(0.5, {})
        params = {
            "objective": "quantile",
            "alpha": 0.5,
            "metric": "quantile",
            "boosting_type": self.BOOSTING_TYPE,
            "max_bin": self.MAX_BIN,
            "num_leaves": q50.get("num_leaves", 31),
            "learning_rate": q50.get("learning_rate", 0.03),
            "max_depth": q50.get("max_depth", 5),
            "min_data_in_leaf": q50.get("min_data_in_leaf", 15),
            "lambda_l1": q50.get("lambda_l1", 0.5),
            "lambda_l2": q50.get("lambda_l2", 0.5),
            "feature_fraction": q50.get("feature_fraction", 0.7),
            "verbosity": -1,
            "n_jobs": -1,
            "random_state": 42,
        }
        ds = lgb.Dataset(
            X_tr, label=train_df[tcol].to_numpy(dtype=float),
            params={"max_bin": self.MAX_BIN, "feature_pre_filter": False})
        model = lgb.train(
            params, ds,
            num_boost_round=self._boost_rounds(horizon, cv=True),
            callbacks=[lgb.log_evaluation(0)])
        X_val = val_df[self.feature_cols].replace(
            [np.inf, -np.inf], np.nan).fillna(med)
        return model.predict(X_val)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_lambdarank_diagnostic.py -q`
Expected: PASS (existing lambdarank tests + the two new ones).

- [ ] **Step 5: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_lambdarank_diagnostic.py
git commit -m "feat: _fold_q50_scores, the matched q50 twin of the ranker trainer"
```

---

### Task 2: `backtest/longshort.py` — decile spread + directional records (pure)

**Files:**
- Create: `backend/backtest/longshort.py`
- Test: `backend/tests/test_longshort.py`

**Interfaces:**
- Consumes: nothing project-specific (pure numpy/pandas).
- Produces:
  - `decile_longshort_by_date(score, realised, dates, mask=None, decile=0.1, min_rows=20) -> dict` — returns `{"per_date": {date: spread}, "mean": float|None, "n_dates": int}` where each date's spread = mean realised return of the top-`decile` by `score` minus the bottom-`decile`, on masked rows.
  - `net_of_cost(spread: float, roundtrip: float) -> float` — `spread - roundtrip` (the fee+spread haircut applied to a long-short; long and short each pay, so `roundtrip` is the summed cost passed by the caller).
  - `direction_records(score, realised, dates, mask=None, min_rows=20) -> list[dict]` — one record per masked row in the shape `pesaran_timmermann` consumes: `predicted_direction`/`actual_direction` are the within-date **above/below-median** sign of `score` / `realised`, `direction_correct` their equality, `forecast_date` the ISO date string.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_longshort.py`:

```python
import numpy as np
import pandas as pd

from backtest.longshort import (
    decile_longshort_by_date, net_of_cost, direction_records)


def _one_date(n, rng):
    d = np.repeat(pd.to_datetime(["2026-05-01"]), n)
    return d, rng.normal(size=n), rng.normal(size=n)


def test_decile_spread_positive_when_score_matches_realised():
    rng = np.random.default_rng(0)
    n = 200
    dates = np.repeat(pd.to_datetime(["2026-05-01", "2026-05-02"]), n // 2)
    score = rng.normal(size=n)
    realised = score + 0.1 * rng.normal(size=n)  # top-score → top-realised
    out = decile_longshort_by_date(score, realised, dates)
    assert out["n_dates"] == 2
    assert out["mean"] > 0  # top decile outperforms bottom


def test_decile_spread_zero_mean_when_score_is_noise():
    rng = np.random.default_rng(1)
    n = 2000
    dates = np.repeat(pd.to_datetime(["2026-05-01"]), n)
    score = rng.normal(size=n)
    realised = rng.normal(size=n)  # independent
    out = decile_longshort_by_date(score, realised, dates)
    assert abs(out["mean"]) < 0.2  # no systematic spread


def test_mask_restricts_to_tied_rows():
    rng = np.random.default_rng(2)
    n = 100
    dates = np.repeat(pd.to_datetime(["2026-05-01"]), n)
    score = rng.normal(size=n)
    realised = rng.normal(size=n)
    mask = np.zeros(n, dtype=bool)
    mask[:10] = True  # too few rows after masking → no date read
    out = decile_longshort_by_date(score, realised, dates, mask=mask,
                                   min_rows=20)
    assert out["n_dates"] == 0
    assert out["mean"] is None


def test_net_of_cost_subtracts_roundtrip():
    assert net_of_cost(0.30, 0.20) == 0.10


def test_direction_records_shape_and_correctness():
    # Two rows above median score → predicted "up"; realised follows.
    score = np.array([0.0, 1.0, 2.0, 3.0] * 6)  # 24 rows, one date
    realised = score.copy()
    dates = np.repeat(pd.to_datetime(["2026-05-01"]), len(score))
    recs = direction_records(score, realised, dates, min_rows=20)
    assert len(recs) == len(score)
    assert set(recs[0]) == {"predicted_direction", "actual_direction",
                            "direction_correct", "forecast_date"}
    # score == realised → every call correct
    assert all(r["direction_correct"] for r in recs)
    assert recs[0]["forecast_date"] == "2026-05-01"


def test_direction_records_skips_thin_dates():
    score = np.arange(10, dtype=float)
    realised = score.copy()
    dates = np.repeat(pd.to_datetime(["2026-05-01"]), 10)
    assert direction_records(score, realised, dates, min_rows=20) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_longshort.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'backtest.longshort'`.

- [ ] **Step 3: Write minimal implementation**

Create `backend/backtest/longshort.py`:

```python
"""Cross-sectional long-short read-outs for the C2 serving-transfer diagnostic.

Pure functions over (score, realised, dates). The ranker emits an ordinal
score with no return scale, so its only honest read-outs are the ORDERING it
implies: the within-date rank IC (in `forecaster._within_date_rank_ic`), the
top-minus-bottom decile spread here, and a within-date above/below-median
directional call for the Pesaran-Timmermann test. No level metric belongs here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _grouped(score, realised, dates, mask):
    p = np.asarray(score, dtype=float)
    a = np.asarray(realised, dtype=float)
    d = pd.to_datetime(pd.Series(dates).to_numpy())
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        p, a, d = p[m], a[m], d[m]
    fin = np.isfinite(p) & np.isfinite(a)
    return pd.DataFrame({"d": d[fin], "p": p[fin], "a": a[fin]})


def decile_longshort_by_date(score, realised, dates, mask=None,
                             decile: float = 0.1, min_rows: int = 20) -> dict:
    """Per date: mean realised of the top-`decile` by score minus the bottom.

    A date with fewer than `min_rows` scorable rows contributes nothing (the
    same floor `_within_date_rank_ic` uses), so the two reads describe one
    calendar. `mean` is None when no date qualifies.
    """
    frame = _grouped(score, realised, dates, mask)
    per_date: dict = {}
    for day, g in frame.groupby("d"):
        if len(g) < min_rows:
            continue
        k = max(1, int(round(len(g) * decile)))
        order = g.sort_values("p")
        bottom = order["a"].iloc[:k].mean()
        top = order["a"].iloc[-k:].mean()
        per_date[str(pd.Timestamp(day).date())] = float(top - bottom)
    mean = float(np.mean(list(per_date.values()))) if per_date else None
    return {"per_date": per_date, "mean": mean, "n_dates": len(per_date)}


def net_of_cost(spread: float, roundtrip: float) -> float:
    """The decile spread after the microstructure haircut.

    `roundtrip` is the SUMMED cost the long and short legs pay (15% Steam fee +
    tier spread, doubled), supplied by the caller — this function does not know
    the venue. A relative edge is only actionable if it clears this; but it is a
    tradeability conditioner, not the pass/fail bar.
    """
    return spread - roundtrip


def direction_records(score, realised, dates, mask=None,
                      min_rows: int = 20) -> list[dict]:
    """Within-date above/below-median directional call, for `pesaran_timmermann`.

    Maps the ranker's ordinal score to a served direction the only way that
    respects its scale-freedom: above its own within-date median → "up". The
    actual leg is the same split on realised return. This is the ranker's
    directional skill on the cross-section, which is what invariant 4's PT test
    scores. Thin dates (< `min_rows`) are dropped, matching the rank-IC floor.
    """
    frame = _grouped(score, realised, dates, mask)
    out: list[dict] = []
    for day, g in frame.groupby("d"):
        if len(g) < min_rows:
            continue
        pmed, amed = g["p"].median(), g["a"].median()
        fd = str(pd.Timestamp(day).date())
        for p, a in zip(g["p"], g["a"]):
            pdir = "up" if p > pmed else "down"
            adir = "up" if a > amed else "down"
            out.append({"predicted_direction": pdir, "actual_direction": adir,
                        "direction_correct": pdir == adir, "forecast_date": fd})
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_longshort.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/backtest/longshort.py backend/tests/test_longshort.py
git commit -m "feat: long-short decile spread + directional records for C2"
```

---

### Task 3: `_anchor_metrics` — score one anchor date (the pure scoring core)

**Files:**
- Create: `backend/scripts/replay_lambdarank.py` (start it with imports + this one function)
- Test: `backend/tests/test_replay_lambdarank.py`

**Interfaces:**
- Consumes: `replay_serving._resolve`, `replay_serving._tied_mask` (referee, imported); `ItemForecaster._within_date_rank_ic_detail`; `backtest.longshort.decile_longshort_by_date`, `direction_records`; `backtest.directional_test.pesaran_timmermann`.
- Produces: `_anchor_metrics(anchor, horizon, val_df, lr_scores, q50_scores, naive_scores, outcomes, floor, tied) -> dict|None` — resolves the outcome at `anchor+horizon` (smoothed, `after=anchor`), restricts to `current>=floor` AND the tied mask, and returns per-anchor `{n_tied, lr_ic, q50_ic, naive_ic, ls_spread, pt_records}` or `None` if fewer than `MIN_TIED_ROWS` tied rows resolve. `val_df` has columns `item_id`, `current`, `date`. `pt_records` is a list to accumulate for a cross-date PT test.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_replay_lambdarank.py`:

```python
import numpy as np
import pandas as pd
import pytest

from scripts.replay_lambdarank import _anchor_metrics, MIN_TIED_ROWS


def _synthetic_anchor(n, signal, rng):
    """One anchor date with `n` items; `signal` scales how well score predicts
    the realised outcome. Returns (val_df, lr, q50, naive, outcomes, tied)."""
    anchor = pd.Timestamp("2026-05-01")
    item_id = np.arange(n)
    current = rng.uniform(2, 50, size=n)  # all >= $1 floor
    true_rank = rng.normal(size=n)
    lr = true_rank + 0.05 * rng.normal(size=n)
    q50 = 0.3 * true_rank + rng.normal(size=n)  # weaker orderer
    naive = rng.normal(size=n)
    realised = current * (1.0 + signal * true_rank * 0.01)
    val_df = pd.DataFrame({"item_id": item_id, "current": current,
                           "date": anchor})
    # Outcome rows land inside the h=7 resolve window (anchor, anchor+7].
    out_day = anchor + pd.Timedelta(days=7)
    outcomes = pd.DataFrame({"item_id": item_id,
                             "day": np.repeat(out_day, n),
                             "price": realised})
    tied = pd.Series(True, index=item_id)  # everyone tied in the synthetic
    tied.index.name = "item_id"
    return val_df, lr, q50, naive, outcomes, tied


def test_returns_none_below_min_tied(monkeypatch):
    rng = np.random.default_rng(0)
    val_df, lr, q50, naive, outcomes, tied = _synthetic_anchor(
        MIN_TIED_ROWS - 1, signal=1.0, rng=rng)
    out = _anchor_metrics(pd.Timestamp("2026-05-01"), 7, val_df, lr, q50,
                          naive, outcomes, floor=1.0, tied=tied)
    assert out is None


def test_ranker_beats_q50_on_a_planted_signal():
    rng = np.random.default_rng(1)
    val_df, lr, q50, naive, outcomes, tied = _synthetic_anchor(
        300, signal=1.0, rng=rng)
    out = _anchor_metrics(pd.Timestamp("2026-05-01"), 7, val_df, lr, q50,
                          naive, outcomes, floor=1.0, tied=tied)
    assert out is not None
    assert out["n_tied"] == 300
    assert out["lr_ic"] > out["q50_ic"]   # lr orders the planted signal better
    assert out["ls_spread"] > 0           # top decile outperforms bottom
    assert len(out["pt_records"]) == 300
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_replay_lambdarank.py -q`
Expected: FAIL with `ModuleNotFoundError`/`ImportError` (module/function not yet defined).

- [ ] **Step 3: Write minimal implementation**

Create `backend/scripts/replay_lambdarank.py` with the imports and this function (the driver `main()` is Task 4):

```python
"""C2 lambdarank serving-transfer read — does the CV vs-q50 edge reach serving?

Read-only. Retrains a matched q50 + lambdarank pair at the 14-day production
cadence, scores each frozen served-window anchor date's TIED cross-section
against replay-resolved outcomes, and reports three rulers: within-date rank IC
(primary, edge lr-q50), the decile long-short spread, and a Pesaran-Timmermann
test on the ranker's within-date directional call. Writes nothing.

    venv/bin/python -m scripts.replay_lambdarank --horizon 7

Design: docs/superpowers/specs/2026-08-14-lambdarank-serving-transfer-design.md
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.directional_test import pesaran_timmermann          # noqa: E402
from backtest.longshort import (                                  # noqa: E402
    decile_longshort_by_date, direction_records, net_of_cost)
from models.forecaster import ItemForecaster                      # noqa: E402
from scripts.replay_serving import _resolve, _tied_mask           # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Fewer tied items than this on an anchor and the within-date correlations are
# noise — the same floor `_within_date_rank_ic_detail` applies per date.
MIN_TIED_ROWS = 20
# Window start / end, and the retrain cadence. Frozen: the served window every
# stored A/B of the last fortnight is read on.
SERVED_WINDOW = (date(2026, 4, 18), date(2026, 6, 8))
RETRAIN_CADENCE_DAYS = 14
# Summed long+short microstructure cost for the net-of-cost decile read: 15%
# Steam fee each way + a mid-tier ~5% spread each way. Tradeability conditioner
# only; NOT a pass/fail bar.
ROUNDTRIP_COST = 2 * (0.15 + 0.05)


def _anchor_metrics(anchor, horizon, val_df, lr_scores, q50_scores,
                    naive_scores, outcomes, floor, tied) -> Optional[dict]:
    """Score one anchor date on its TIED, served (`current >= floor`) cohort.

    Returns per-anchor rank ICs (lr / q50 / naive), the decile long-short
    spread, and the ranker's per-row PT records for cross-date pooling — or
    None when fewer than MIN_TIED_ROWS tied rows resolve. The outcome is the
    smoothed trailing median at `anchor + horizon`, strictly after the anchor
    (`_resolve(..., after=anchor)`), so a row's outcome never reaches back over
    its own anchor quote.
    """
    target = anchor.date() if isinstance(anchor, pd.Timestamp) else anchor
    realised = _resolve(outcomes, target + timedelta(days=int(horizon)),
                        after=target)
    frame = val_df.copy()
    frame["lr"] = np.asarray(lr_scores, dtype=float)
    frame["q50"] = np.asarray(q50_scores, dtype=float)
    frame["naive"] = np.asarray(naive_scores, dtype=float)
    frame = frame.merge(realised, on="item_id", how="inner")
    frame = frame[frame["current"] >= floor]
    is_tied = frame["item_id"].map(tied).eq(True).to_numpy()
    frame = frame[is_tied]
    frame = frame[np.isfinite(frame["realised"]) & (frame["current"] > 0)]
    if len(frame) < MIN_TIED_ROWS:
        return None

    ret = (frame["realised"] / frame["current"] - 1.0).to_numpy()
    dates = frame["date"].to_numpy()
    # One anchor is one date, so min_rows is the tied-count floor above and the
    # detail reader returns a single-date IC (or None if degenerate).
    def _ic(pred):
        return ItemForecaster._within_date_rank_ic_detail(
            pred, ret, dates, min_rows=MIN_TIED_ROWS)[0]

    ls = decile_longshort_by_date(frame["lr"].to_numpy(), ret, dates,
                                  min_rows=MIN_TIED_ROWS)
    return {
        "anchor": str(target),
        "n_tied": int(len(frame)),
        "lr_ic": _ic(frame["lr"].to_numpy()),
        "q50_ic": _ic(frame["q50"].to_numpy()),
        "naive_ic": _ic(frame["naive"].to_numpy()),
        "ls_spread": ls["mean"],
        "pt_records": direction_records(frame["lr"].to_numpy(), ret, dates,
                                        min_rows=MIN_TIED_ROWS),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_replay_lambdarank.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/replay_lambdarank.py backend/tests/test_replay_lambdarank.py
git commit -m "feat: _anchor_metrics — per-anchor tied-cohort scoring core for C2"
```

---

### Task 4: The walk-forward driver — master frame, folds, anchor freeze, `main()`

**Files:**
- Modify: `backend/scripts/replay_lambdarank.py` (add the driver below `_anchor_metrics`)
- Test: `backend/tests/test_replay_lambdarank.py` (add `TestFrozenAnchors` and a guarded smoke)

**Interfaces:**
- Consumes: `ItemForecaster` (`fetch_price_history`, `engineer_features`, `prepare_targets`, `embargo_days`, `_lambdarank_fold_scores`, `_fold_q50_scores`, `feature_cols`, `_artifact_min_median_price` fallback `MIN_SERVED_PRICE_USD`); `replay_serving._feed_profile`, `audit_anchor_feed`, `cutovers_from_counts`, `cutovers_in_outcome_window`, `_outcomes`; `_anchor_metrics` (Task 3); `pesaran_timmermann`.
- Produces: `frozen_anchors(fc, horizon, window=SERVED_WINDOW) -> list[date]` (audit-passing, cutover-free, sorted); `_retrain_points(window, cadence) -> list[date]`; `main() -> int` (sharded by `--horizon`).

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_replay_lambdarank.py`:

```python
from datetime import date
from scripts.replay_lambdarank import _retrain_points, SERVED_WINDOW


class TestRetrainPoints:
    def test_biweekly_points_span_the_window(self):
        pts = _retrain_points(SERVED_WINDOW, cadence=14)
        assert pts[0] == date(2026, 4, 18)
        assert all((pts[i + 1] - pts[i]).days == 14
                   for i in range(len(pts) - 1))
        assert pts[-1] <= date(2026, 6, 8)

    def test_each_point_precedes_the_window_end(self):
        pts = _retrain_points(SERVED_WINDOW, cadence=14)
        assert pts[-1] < SERVED_WINDOW[1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_replay_lambdarank.py::TestRetrainPoints -q`
Expected: FAIL with `ImportError: cannot import name '_retrain_points'`.

- [ ] **Step 3: Write minimal implementation**

Append to `backend/scripts/replay_lambdarank.py`:

```python
def _retrain_points(window, cadence: int = RETRAIN_CADENCE_DAYS) -> list:
    """Retrain-point dates at `cadence` days from the window start, each
    strictly before the window end (a point on the last day scores nothing)."""
    lo, hi = window
    pts, d = [], lo
    while d < hi:
        pts.append(d)
        d = d + timedelta(days=cadence)
    return pts


def frozen_anchors(fc, horizon, window=SERVED_WINDOW) -> list:
    """Every date in `window` that passes the feed audit AND has no collector
    cutover inside its `horizon` outcome window. Enumerated and LOGGED before
    any rank IC is read — the pre-registration the spec requires.
    """
    lo, hi = window
    surviving = []
    d = lo
    while d <= hi:
        profile = _feed_profile(d)
        ok, _ = audit_anchor_feed(d, profile)
        if ok:
            span = _feed_profile(d, window=horizon)
            cutovers = cutovers_from_counts(
                span.set_index(pd.to_datetime(span["day"]).dt.date)["items"])
            if not cutovers_in_outcome_window(d, [horizon], cutovers):
                surviving.append(d)
        d = d + timedelta(days=1)
    logger.info("FROZEN anchor set (h=%s): %d dates — %s", horizon,
                len(surviving), ", ".join(a.isoformat() for a in surviving))
    return surviving


def _master_frame(fc, horizon, cutoff):
    """Engineered + targeted frame bounded at `cutoff`, for TRAINING one fold.

    REPLAY_ANCHOR bounds the archive read at the cutoff (no look-ahead), then
    `engineer_features` (causal lags) and `prepare_targets` (forward label +
    ANCHOR_TIED_COL) build the frame the fold trainers consume. Rows without a
    resolved `target_return_{h}d` fall out of `prepare_targets`, so the train
    set is automatically `date <= cutoff - horizon`; the embargo below removes
    the rest of the label's 13-day support.
    """
    os.environ["REPLAY_ANCHOR"] = cutoff.isoformat()
    try:
        price_df = fc.fetch_price_history(days_back=3650, backfilled_only=True)
        feat = fc.engineer_features(price_df)
        tdf = fc.prepare_targets(feat, horizon)
    finally:
        os.environ.pop("REPLAY_ANCHOR", None)
    embargo = fc.embargo_days(horizon)
    keep = pd.to_datetime(tdf["date"]) <= (
        pd.Timestamp(cutoff) - pd.Timedelta(days=embargo))
    return tdf[keep].reset_index(drop=True), feat


def _val_frame(feat, anchor):
    """The served cross-section AS OF `anchor`: the engineered rows on that day.

    Taken from `engineer_features` output (causal), not `prepare_targets`, so an
    item with no resolved forward label at the anchor is still scored — its
    outcome is resolved separately from the full archive.
    """
    day = pd.Timestamp(anchor)
    v = feat[pd.to_datetime(feat["date"]) == day].copy()
    return v.reset_index(drop=True)


def main() -> int:
    if "--horizon" not in sys.argv:
        logger.error("Pass --horizon 3|7|14|30 (shard by horizon for the cap).")
        return 2
    horizon = int(sys.argv[sys.argv.index("--horizon") + 1])

    from database import SessionLocal
    from api.serving_policy import MIN_SERVED_PRICE_USD
    db = SessionLocal()
    try:
        fc = ItemForecaster(db_session=db, prune_failed_groups=False)
        anchors = frozen_anchors(fc, horizon)
        if not anchors:
            logger.error("No clean anchors in %s at h=%s.", SERVED_WINDOW,
                         horizon)
            return 1
        points = _retrain_points(SERVED_WINDOW)
        # Outcomes over the whole window+horizon, resolved once from the full
        # (unbounded) archive — REPLAY_ANCHOR is cleared inside `_outcomes`.
        outcomes = _outcomes(fc, SERVED_WINDOW[1], [horizon])
        floor = fc._artifact_min_median_price or MIN_SERVED_PRICE_USD
        per_quantile_params = {0.5: {}}  # tuned HP not needed for the sign bar

        rows = []
        all_pt = []
        for i, cutoff in enumerate(points):
            block_hi = (points[i + 1] if i + 1 < len(points)
                        else SERVED_WINDOW[1] + timedelta(days=1))
            block = [a for a in anchors if cutoff <= a < block_hi]
            if not block:
                continue
            train_df, feat = _master_frame(fc, horizon, cutoff)
            logger.info("retrain @ %s: %d train rows, %d anchors in block",
                        cutoff, len(train_df), len(block))
            for anchor in block:
                val_df = _val_frame(feat, anchor)
                if len(val_df) < MIN_TIED_ROWS:
                    continue
                lr = fc._lambdarank_fold_scores(train_df, val_df, horizon,
                                                per_quantile_params)
                q50 = fc._fold_q50_scores(train_df, val_df, horizon,
                                          per_quantile_params)
                naive = (-val_df["return_1d"].to_numpy(dtype=float)
                         if "return_1d" in val_df.columns
                         else np.zeros(len(val_df)))
                tied = _tied_mask(outcomes, anchor)
                m = _anchor_metrics(pd.Timestamp(anchor), horizon, val_df, lr,
                                    q50, naive, outcomes, floor, tied)
                if m is not None:
                    rows.append(m)
                    all_pt.extend(m["pt_records"])

        if not rows:
            logger.error("No anchor produced >= %d tied resolved rows.",
                         MIN_TIED_ROWS)
            return 1

        def _mean(key):
            vals = [r[key] for r in rows if r[key] is not None]
            return float(np.mean(vals)) if vals else None

        lr_ic, q50_ic, naive_ic = _mean("lr_ic"), _mean("q50_ic"), _mean("naive_ic")
        edge_q50 = (None if lr_ic is None or q50_ic is None
                    else lr_ic - q50_ic)
        edge_naive = (None if lr_ic is None or naive_ic is None
                      else lr_ic - naive_ic)
        ls = _mean("ls_spread")
        pt = pesaran_timmermann(all_pt, min_dates=len(rows))

        print(f"\nC2 SERVING-TRANSFER @ h={horizon}   "
              f"(tied cohort, {len(rows)} anchor-dates, floor >= ${floor:g})")
        print(f"  lr rank IC      {lr_ic}")
        print(f"  q50 rank IC     {q50_ic}")
        print(f"  naive rank IC   {naive_ic}")
        print(f"  EDGE vs q50     {edge_q50}   <-- primary bar: > 0 confirms")
        print(f"  edge vs naive   {edge_naive}")
        print(f"  decile L/S      gross {ls}   "
              f"net {None if ls is None else net_of_cost(ls, ROUNDTRIP_COST)}")
        print(f"  PT (lr call)    excess_pp={pt['pt_excess_pp']} "
              f"t={pt['pt_t_stat']} verdict={pt['pt_verdict']} "
              f"n_dates={pt['pt_n_dates']}")
        print("\nEDGE vs q50 is the pre-registered pass/fail. Decile net and "
              "PT are descriptive; DA/MAE are undefined for a ranker.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_replay_lambdarank.py -q`
Expected: PASS (Task 3 tests + `TestRetrainPoints`).

- [ ] **Step 5: Byte-compile and a real dry sanity check**

Run: `venv/bin/python -c "import ast; ast.parse(open('scripts/replay_lambdarank.py').read())"`
Expected: no output (parses).

Run: `venv/bin/python -m scripts.replay_lambdarank --horizon 7 2>&1 | head -40`
Expected: it logs the FROZEN anchor set line and `retrain @ …` lines, and prints the `C2 SERVING-TRANSFER` block (or a clean "No clean anchors"/"No anchor produced" error if the local archive lacks the window — the local `price-archive/` runs behind the durable one). If it tracebacks, fix before committing.

- [ ] **Step 6: Commit**

```bash
git add backend/scripts/replay_lambdarank.py backend/tests/test_replay_lambdarank.py
git commit -m "feat: C2 serving-transfer walk-forward driver (replay_lambdarank)"
```

---

### Task 5: Run the four horizon shards and record the result

**Files:**
- Create: `docs/changelog/2026-08-14-lambdarank-serving-transfer-measured.md`

**Interfaces:** none (measurement + write-up).

- [ ] **Step 1: Verify the archive covers the served window**

Run: `venv/bin/python -c "from scripts.replay_serving import _feed_profile; import datetime as dt; print(_feed_profile(dt.date(2026,6,8)))"`
Expected: non-empty profile around 2026-06-08. If the local archive is behind, note it and run against the durable `cs2-oracle-data` archive per backend/AGENTS.md before quoting numbers.

- [ ] **Step 2: Run each horizon shard (separately, under the 30-min cap)**

Run, one at a time:
```bash
venv/bin/python -m scripts.replay_lambdarank --horizon 3  | tee /tmp/c2_h3.txt
venv/bin/python -m scripts.replay_lambdarank --horizon 7  | tee /tmp/c2_h7.txt
venv/bin/python -m scripts.replay_lambdarank --horizon 14 | tee /tmp/c2_h14.txt
venv/bin/python -m scripts.replay_lambdarank --horizon 30 | tee /tmp/c2_h30.txt
```
Expected: each prints the FROZEN anchor set, then the `C2 SERVING-TRANSFER` block. Record `EDGE vs q50`, `edge vs naive`, decile gross/net, and the PT line per horizon.

- [ ] **Step 3: Write the changelog against the pre-registered bars**

Create `docs/changelog/2026-08-14-lambdarank-serving-transfer-measured.md` stating, per horizon: the frozen anchor count, the vs-q50 edge (the pass/fail), the vs-naive edge, decile spread gross and net-of-cost, and the PT verdict. Read it against the spec's bars: a horizon whose vs-q50 edge stays > 0 transferred; ≤ 0 did not. Cross-reference the CV numbers (+0.0648 / +0.0614 / +0.0391 / +0.0297) and state whether the edge attenuated, held, or vanished. Note the tradeability read (net decile) as a conditioner, not a bar. If the local archive ran behind the durable one, say which archive produced the numbers.

- [ ] **Step 4: Commit**

```bash
git add docs/changelog/2026-08-14-lambdarank-serving-transfer-measured.md
git commit -m "docs: C2 lambdarank serving-transfer result"
```

---

## Self-Review

**Spec coverage:**
- Read-only, no serving/artifact/DB write → Tasks 3–4 train locally and only print; `main()` opens a session for reads. ✓
- Objective-transfer on served months, tied cohort → `_anchor_metrics` masks to tied; `frozen_anchors` restricts to the window. ✓
- Reuse referee (`_resolve`, `_tied_mask`, `_outcomes`, audit, cutover) → imported in Tasks 3–4, not forked. ✓
- Matched q50 vs lr contrast → Task 1 sibling trained on identical rows/HP, with the weights/offset asserts. ✓
- Biweekly retrain / daily score → `_retrain_points` + the block loop. ✓
- Three rulers (rank IC edge, decile spread, PT) → `_within_date_rank_ic_detail`, `decile_longshort_by_date`, `pesaran_timmermann(direction_records)`. ✓
- Frozen-before-read anchor set → `frozen_anchors` logs before scoring. ✓
- Tradeability net read, non-gating → `net_of_cost`, printed beside gross, labelled descriptive. ✓
- 30-min cap via horizon shards → `--horizon` required. ✓
- Void conditions: no level metric for the ranker (none computed); within-date relevance (inherited from `_lambdarank_labels`); tied verdict; no look-ahead (`REPLAY_ANCHOR` bound + embargo filter + `after=anchor`). ✓

**Placeholder scan:** every step has runnable code/commands; Task 5 write-up is content-specified (per-horizon bars to record), not "write tests for the above". ✓

**Type consistency:** `_anchor_metrics` returns keys `lr_ic/q50_ic/naive_ic/ls_spread/pt_records/n_tied/anchor`, consumed by `main()`'s `_mean` and PT pooling identically. `decile_longshort_by_date` returns `mean` (used as `ls_spread`). `_retrain_points`/`frozen_anchors` return `list[date]`; `_anchor_metrics` takes a `pd.Timestamp` anchor (constructed at the call site). `_master_frame` returns `(train_df, feat)`; `_val_frame(feat, anchor)` consumes `feat`. ✓

**Deviations from the spec, flagged:** (1) `_fold_q50_scores` is a new sibling, not an extraction from `_cv_evaluate_horizon` — lower risk to the load-bearing loop, byte-equivalent under shipped config, guarded by asserts. (2) Val features come from the reused CV master-frame builder rather than a bespoke per-anchor `predict()`-style build; the outcome leg still uses the replay referee, so the served-basis numerator (the ~0.025 rank-IC axis the tied cohort does not remove) is preserved. Both keep the pre-registered measurement intact.
