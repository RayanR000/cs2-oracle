---
paths:
  - "backend/models/{forecaster,staleness}.py"
  - "backend/scripts/{forecast_prices,walkforward_backtest}.py"
---

# Labels, staleness, and the embargo

- **A label may not touch a frozen price run, on either leg.** `models/staleness.py`
  counts consecutive bit-identical prices per item; `prepare_targets` voids the label of any
  row whose anchor day *or* target day sits on a run longer than
  `LABEL_MAX_STALE_RUN_DAYS` (0 — set it to `None` to disable, which is the harness's control
  arm). This is the per-item companion to `_snapshot_dates`, which voids a day where the whole
  cross-section repeats. **A gap wider than `MAX_WINDOW_SPAN_DAYS` (7) breaks a run** rather
  than continuing it — nothing was observed across a collection outage to be frozen.
  **Two numbers not to misread.** The "0–1.8% stale at ≥$1" figure in the research review is
  measured on **resolved, 3-day-smoothed anchors**; the raw voted series the label path sees is
  **12–27%**. Different quantities — neither sizes the other. And the rate is a **2026 feed
  property**, 0.5–0.8% through 2025 against 6–33% across 2026, because no Steam-derived series
  in this archive is a point observation (`aggregator_steam_7d/30d/90d` are trailing-window
  means outright; `aggregator_sync` and `aggregator_steam_17mafo` are `last_24h` falling back
  to them, which fires on exactly the illiquid items). So the rule voids ~30% of 2026 ≥$1
  labels and ~0.6% of pre-2024 ones, and it takes **13.7–15.9% of *non-zero* ≥$1 labels** with
  it — it can be net-harmful, which is why `scripts/ab_test_frozen_runs.py` verifies it on
  paired interval **width**, not on a point estimate. See
  `docs/superpowers/specs/2026-08-08-frozen-price-runs-design.md`.
- **The label's denominator is the RAW anchor quote, and that is a measured defect.**
  `prepare_targets` divides by the price observed on the anchor day; `predict` quotes
  against `_smoothed_anchor_prices`' span-bounded median. The same raw quote drives
  `return_1d` and every level feature, so one noisy observation deflates the label and
  inflates the feature together. The 2026-08-11 serving replay attributed **+0.1398 of the
  +0.1464** CV↔serving rank IC gap to this axis alone (outcome leg +0.0251, serving
  transforms none), confirmed in CI at four non-overlapping anchors in **16 cells of 16**.
  `LABEL_SMOOTHED_ANCHOR=1` puts both legs on the served median via
  `_rolling_anchor_prices`; off by default, unmeasured for accuracy. **It is a LABEL change,
  so a CV rank IC / DA / `naive_rank_ic` under it is scored against a different target and
  must never be differenced against a control's** — read it through
  `scripts/replay_serving.py`, which builds its own labels from the archive. `meta.json`
  carries `label_smoothed_anchor`. See
  `docs/changelog/2026-08-11-label-smoothed-anchor.md`.
  ⚠️ **Measured 2026-08-11 and it is NOT the fix — leave it off.** Against a control at four
  anchors it swings pooled served rank IC to **+0.17–0.31, 4/4 anchors**, and every point of
  that is `p[d]/S[d]` — the anchor deviation — entering the label as a factor readable at the
  anchor from `return_1d`. On the **tied** cohort, where the factor is identically 1, it is
  **−0.033 / −0.017 / −0.020 / +0.008**. Both bases are contaminated by `p/S` with opposite
  signs, so **rank arms on the tied subset**, which is the only cohort where neither operates.
  `docs/changelog/2026-08-11-smoothed-anchor-label-measured.md`.
  ✅ **The CONFORMAL residual moved to the served denominator on 2026-08-12, and that is a
  different change.** `calibration_target_col(h)` → `target_return_{h}d_cal` is emitted by
  `prepare_targets` alongside the label, always on `_rolling_anchor_prices`, winsorized and
  voided on the same rows; `_conformal_records(..., residual_actual_ret=)` measures
  `residual_pct` from it while `actual_ret` still feeds `hit` / `change_pct`. The label is
  **untouched**, so none of the refutation above applies — `q_hat` is post-hoc and cannot hand
  the model a factor. Fitting it on the label had inflated it by `p[d]/S[d]` and the served
  band over-covered at **87.2 / 91.8 / 90.6 / 89.0%** against 80%. `meta.json` carries
  `conformal_basis`; the fallback WARNs. Conditional coverage (58.2–99.2% per date) is
  **not** fixed. `docs/changelog/2026-08-12-conformal-basis-follows-serving.md`.
- **The wedge `p[d]/S[d]` is also attackable in the SERVING basis, and that arm is
  `SERVE_OUTLIER_GATED_ANCHOR=1`.** `predict` detects items whose latest quote deviates >10%
  from their local median, logs `using smoothed price` — and then substitutes the median for
  **every** item, so the served `current_price` is a number no venue quoted. The flag gates
  the substitution on the deviation test the warning already describes
  (`_serving_base_price`, threshold `ItemForecaster.ANCHOR_OUTLIER_TOLERANCE`, boundary
  exclusive). Off by default. ⚠️ **Measured 2026-08-11: it PASSES on dollar error — 14 of 16
  cells above a cohort-composition placebo, tied cohort 0.00 in all 16 — and it buys NO
  forecasting skill.** The model's edge over its own no-change quote moves at 10 of 16 (coin
  flip), and `pinnedIC`'s +0.06–0.40 is the `p/S` free factor again, this time in the
  prediction: pinning the denominator does not make rank IC arm-comparable when the arm moves
  the served mid's level. Still off by default — the open blocker is the backtest resolver,
  which shares `SMOOTH_WINDOW`. `docs/changelog/2026-08-11-serving-anchor-freshness-measured.md`.
  Three things to know before reading it:
  it is **serving-only**, so it touches no CV number and leaves **no trace in `meta.json`** —
  the log line and the `model-diagnostics.yml` arm heading are the only record; the deviation
  mask is **arm-invariant** by construction, so both dispatches size the same cohort; and it
  **must be read on `replay_serving.py`'s DOLLAR ERROR table, deviating row** — rank IC divides
  both legs by the served quote this arm moves, and on the tied cohort both arms serve an
  identical price. `docs/superpowers/plans/2026-08-11-serving-anchor-freshness.md`.
- **The embargo is `horizon + 13`, not `horizon`.** `models/forecaster.py::embargo_days`
  derives the 13 at call time from `LAG_TOLERANCE_DAYS` (3) + `SMOOTH_WINDOW` (3) +
  `MAX_WINDOW_SPAN_DAYS` (7): the label at `d + horizon` is a **resolved anchor**, not a
  point observation, so its support runs 13 days past its nominal date and a bare-`horizon`
  purge left that carry inside the validation window. Never pass a bare horizon to
  `_compute_cv_splits(purge_days=…)` or re-derive the band locally. At h=30 the embargo (43d)
  **exceeds `VALIDATION_WINDOW_DAYS`** — that is the correct cost, not a bug. It also tracks
  the env-overridable `FALLBACK_MAX_AGE_DAYS`, so fold geometry is not a constant.
- **`walkforward_backtest.py` embargoes by default** since 2026-08-08, and writes
  **`model_version = "lgbm-v4-embargoed"`** when it does. That bump is the point: appending
  purged rows to the un-purged `lgbm-v3-clustered` series would surface as a model regression
  on the dashboard trend and in `backtest-triage`, with nothing stored to say otherwise. The
  persisted `metrics` also carry `purge` and `embargo_days`, so a row is self-describing.
  `--no-purge` reproduces the old split — and the old version string — for a like-for-like
  read against a pre-flip run, and must never be published from. The discontinuity was
  **measured on 2026-08-08** by running both arms and pairing them: the un-purged split inflates
  DA by **+10.15pp at 30d** and **+5.44pp at 14d** (both intervals exclude zero); h=3 and h=7 are
  unresolved, not clean. See `docs/changelog/2026-08-08-embargo-discontinuity-measured.md`.
- **`walkforward_backtest.py` does NOT use `fetch_price_history`.** Its own
  `_load_all_prices` skips multi-source voting, the `historical_fallback:` source filter,
  the dead-item filter, and `backfilled_only`. It does apply the universe rules, in SQL,
  through `archive_universe_sql_filter`. Lags are *not* corrupted — `engineer_features`
  collapses to one row per item-day — but it collapses the archive's 1.37× duplicate
  item-days with a plain **mean**, where production serves an **outlier-voted median**
  (sources >2σ from the median are rejected). The fresh-model gate therefore scores a
  different price consensus over a different item universe than production trains on.
  Not directly comparable to production DA.
