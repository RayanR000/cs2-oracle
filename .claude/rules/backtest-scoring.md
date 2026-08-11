---
paths:
  - "backend/backtest/**"
  - "backend/scripts/{backtest_accuracy,backtest_walkforward_report}.py"
---

# Scoring and resolving the backtest

- **The backtest resolves BOTH legs of `actual_ret` through
  `backtest/price_resolution.py::resolve_anchors`.** Using `item_forecasts.current_price` as
  the base leg is the bug that let one cohort score 61.76% and 33.74% on different days, and
  that rule is absolute: an actual return whose two legs come from different estimators is not
  a measurement. **What `current_price` IS scored on, since 2026-08-11, is the PREDICTION —
  both legs of the outcome stay on the resolver.** Two metrics: `actionable`'s `r_hat`, and
  `in_interval`, whose band is rebased by `base_price / current_price` before the predicate
  (`_derive_verdict`, `quote=` is keyword-only and required so a new path cannot omit it).
  `interval_coverage` is therefore the **calibrated** figure, the basis `q_hat` was fitted in
  and the one `scripts/replay_serving.py` reports; `interval_coverage_dollar_basis` is the
  **published** one, and their gap is the anchor wedge. Quote them together, and read
  `interval_n_served_basis` / `interval_n_fallback_basis` first — legacy rows and
  `walkforward_records` have no separate quote, so the two figures coincide on them by
  construction. `abs_error` / `pct_error` deliberately do **not** rebase: a dollar error is
  basis-free. `docs/changelog/2026-08-11-in-interval-basis.md`.
  For `r_hat`, it divides by the price the forecast was quoted from,
  because `predicted_mid` was built as `current_price × (1 + r̂)`. Dividing it by the resolved
  base instead recovers `r̂` plus the wedge between the two bases: median **13.74%** at
  h ∈ {14,30} against a 7.2–37.5% bar, disagreeing on 85% of rows, which selected **1,141**
  rows where the served basis selects **21**. `actionable_n_served_basis` /
  `actionable_n_fallback_basis` report which denominator each cohort used.
  `docs/changelog/2026-08-11-actionable-selection-is-the-base-wedge.md`. Resolved outcomes are **frozen**: `base_price` / `actual_price`
  / `resolved_at` are final, and `--reresolve` is the only thing that can move them
  (`--rescore` recomputes verdicts from the frozen actuals without reading the archive).
- **Backtest maturity is bounded by archive coverage, not `date.today()`.** A forecast is
  evaluable only once `prices-*.parquet` covers its target date; the cutoff is
  `min(today, archive_max_day())`. The archive always lags the calendar, and admitting
  that lag window puts guaranteed misses into the cohort — it tripped the 10%
  unresolvable gate at 38.5% and reported nothing.
- **Never quote a directional accuracy on its own.** The published headline is a
  Pesaran–Timmermann test (`backtest/directional_test.py`), computed per forecast date with a
  Newey–West t-stat over dates and a `|t| > 3.0` hurdle; `score_cohort` stores it as `pt_*` and
  `scripts/backtest_accuracy.py::_headline_line` logs it. DA is quotable only beside
  `constant_call_accuracy` and `realised_down_rate` — an always-down call scored **29.4% on
  2025-12-01 and 76.9% on 2026-07-17** at 7d, so a fixed hit rate is skill on one date and
  incompetence on the next. Note a **constant call has per-date excess identically zero**, so it
  resolves as `degenerate`, never as skill. `baseline_directional_accuracy` is the always-*flat*
  call, not the constant-call baseline, despite the name. See
  `docs/changelog/2026-08-07-pesaran-timmermann-headline.md`.
- **`constant_call_accuracy` is hindsight-selected, so never difference model DA against it.**
  `constant_call_baseline` counts the *outcomes* and returns whichever fixed call would have won,
  and `_cv_evaluate_horizon` re-picks it **per fold** — the implied direction is `up` on 4 of 8
  folds at 30d, so the average of the per-fold maxima (68.99%) exceeds any runnable strategy by
  20.5pp. **The runnable fixed call is always-down, and its accuracy *is* `realised_down_rate`**
  by construction. Against that, the served classifier is +3.5 / +0.1 / −1.3 / +4.4pp at
  3/7/14/30d, not the −4 to −16pp that `edge_vs_constant_call*` reports. Keep
  `constant_call_accuracy` — it upper-bounds what a direction-free call could extract, which is
  the right diagnostic for how much of a horizon's DA is base rate — but the comparison term is
  `realised_down_rate` and the test is PT. See
  `docs/changelog/2026-08-10-constant-call-is-hindsight-picked.md`.
- **`price_tier` has six bands, and tier 4 changed meaning.** The cut at $1000 landed
  2026-08-07 because the bid–ask spread is 10.8% at $50–500 against 5.2% at $1000+ — the two
  most different liquidity populations in the market. **A stored row with `price_tier == 4`
  written before that date means `≥$100`, not `$100–1000`.** `score_by_tier` also emits the
  `FLOOR_SWEEP` sentinels `-1`/`-2`/`-3` for the `≥$1`/`≥$5`/`≥$20` headline sweep; only `-1`
  (`HEADLINE_TIER`) is the published headline, and `price_tier = NULL` is pooled across those
  liquidity populations and is **not a quotable number**. Adding a floor that is not a
  `price_tier` cut silently rounds down — `floor_records` filters in tier space.
- **`base_stale_run_days` is a frozen observation, and NULL means unknown.** Written by
  `resolve_outcomes` beside `base_price`, absent from `_REFRESH_VERDICTS_SQL`, moved only by
  `--reresolve` — `backtest/scoring.py` is pure and `--rescore` never opens the archive, so
  the run length cannot be derived at score time. Every row resolved before 2026-08-08 carries
  NULL and **must never be read as 0**; `score_by_staleness` buckets those as `unknown`.
  The axis is **four fixed bands, not quartiles** (`fresh` / `repeat_1` / `run_2_6` /
  `run_7_plus`): ~80% of the ≥$1 cohort sits at zero, so data-driven quartiles collapse to one
  populated bucket and would still be reported as four.
- **`actionable_*` is populated only at h ∈ {14, 30}, and its stored series breaks at
  2026-08-11.** Rows scored before that date divided `r_hat` by `base_price`; from that date it
  is the served quote (above). The published cohort collapses from 1,141 to ~21 at h ∈ {14,30}
  and the hit rate from 55.6% to 38.1% — **that is the correction, not a regression**, and it
  is what `2026-08-07-next-steps.md` step 3 predicted: the model's predicted |return| is median
  0.95% against a median threshold of 23.1%, so almost nothing it says implies a trade. Do not
  difference an actionable number across that date. `backtest/actionable.py` conditions
  DA on `|r̂| > round trip + tier spread`, so it answers "does this call imply a trade at
  all" — at CSFloat that bar is 7.2% at tier 5 and 23.1% at tier 1. Read
  **`actionable_scope`** first: `out_of_scope` (wrong horizon), `no_prediction_leg` (records
  without `predicted_mid`) and an `actionable_n` of 0 are three different states, and only
  the last one is a result. It uses the **raw sign**, not `direction_from_return`, so a
  carried-forward price (`r_act == 0`) scores as a miss. `SPREAD_BY_TIER` in
  `backtest/friction.py` is a **nearest-band approximation** — the spread was measured at
  bands that are not the tier cuts — so never cite it as measured per tier.
