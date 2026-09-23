# Market-relative direction labels — design

**Date:** 2026-08-06
**Status:** **REFUTED on measurement 2026-08-06.** Pre-registered rule 1 (the
stage-1 kill switch) fired. `relative_accuracy_ge1` came in at 36.7 / 32.7 /
34.6 / 39.0 at 3/7/14/30d against a majority-class baseline of 38.8 / 42.7 /
46.4 / 51.7 — below a constant call at every horizon, against a rule requiring
> 51%. Once the market factor is removed the price-technical feature set
predicts nothing about which item moves which way. Result and caveats:
`docs/changelog/2026-08-06-market-relative-labels-refuted.md`. The default stays
off; the code is kept as the instrument that produced the measurement.

Two things this document got wrong, recorded rather than edited away:

1. It defined `relative_accuracy_ge1` as `sign(ê)` vs `sign(e)` but the
   implementation used 3-class agreement, whose chance baseline is the
   majority-class share, not the 50% the `<= 51%` threshold assumed. A
   `relative_majority_ge1` baseline was added before the arm ran — the fix made
   the bar harder, not easier — but the pre-registration did not catch it.
2. The flat-band section below asserts residuals are *less* dispersed than raw
   returns. The opposite is true here: a large share of raw returns are exactly
   zero (penny items, stale prices), and subtracting a nonzero `m` moves them
   off zero, so demeaning *spreads* the distribution. Matched bands came out at
   ±1.668 / ±2.572 / ±3.609 / ±4.746% against a fixed ±0.5%.

**Scope:** an off-to-the-side, default-off experiment. Production behaviour stays
byte-identical.

## The hypothesis

Decompose an item's realized forward return into a market factor and an
idiosyncratic residual:

```
r[i, d, h]  =  m[d, h]  +  e[i, d, h]
```

`m` is the cross-sectional common move over the window `d → d+h`; `e` is what is
specific to the item.

The directional classifier is trained on `r`, whose variance is dominated by `m`.
Item-level price technicals carry approximately no information about future `m`
— it is driven by Valve events, case releases and Steam-wide flows, none of which
are in the 36-column feature set. So the classifier cannot reduce that term and
instead settles on a near-constant tilt plus noise. The evidence that this is
what actually happened:

* It predicts "down" on **57–87% of rows regardless of date**
  (`2026-08-03-accuracy-is-clustered-by-forecast-date.md`).
* Its accuracy therefore tracks the market, not the item: 33.4% on a rising date,
  63.7% on a falling one, at 7d.
* A constant always-down call beats it on every stored forecast date.
* Per-fold CV `classifier_accuracy_ge1` at 30d spans **20.2% to 96.3%** — the
  fold's market direction, not the model.
* Served-cohort `classifier_accuracy_ge1` is 49.9 / 49.0 / 51.1 / 53.4 at
  3/7/14/30d (`models/saved_models/meta.json`, trained 2026-08-06 05:17 UTC).

**Hypothesis:** training the classifier on `e` instead of `r` removes an
unlearnable term from the loss, letting the model fit item-specific structure it
currently cannot afford to. Rebuilding the absolute call as `m̂ + ê` — with `m̂`
estimated from history only — should then beat the current absolute model.

## Why this is not one of the already-refuted variants

| Prior result | Why it does not cover this |
|---|---|
| Cross-sectional **features** dropped (`FEATURE_GROUP_ALLOWLIST = ["price_technicals"]`, 2026-07-24; `HORIZON_EXCLUDED_GROUPS` at 14d/30d, 2026-07-19) | Those add columns while leaving the loss dominated by `m`. This changes the target. |
| Served-cohort **weighting** refuted, ~0.1pp (`2026-08-06-served-cohort-weighting-refuted.md`) | That reallocates loss across rows. The cohort mix is not the constraint; the label's variance decomposition is a different claim. |
| Six price primitives shelved (2026-07-31) | Feature-side, scored on the pooled metric, same structural issue. |
| Vol-scaled direction labels refuted (2026-07-27) | Rescales the flat band. It does not re-center the label. |

The changelog for the served-cohort refutation closes with market-date
domination listed under **Still open**, as "the larger unexplained term in served
accuracy". This spec addresses that term.

## Stage 1 — the market factor and the demeaned label

### Chain-linked index, not a median of h-day returns

The factor is built as a daily equal-weighted index and then differenced, rather
than as a per-date median of `target_return_{h}d`:

```
For each date d:
    pairs   = items with a price on both d-1 and d, and price[d-1] >= $1
    ret[d]  = median over pairs of log(price[d] / price[d-1])
    valid   = len(pairs) >= MIN_INDEX_ITEMS
I[d] = exp(cumsum(ret))          # index level, I[first] = 1.0
m[d, h] = 100 * (I[d+h] / I[d] - 1)
```

Three reasons for this shape over a direct median of h-day returns:

1. **Composition.** Items enter and leave the archive. A chain-linked index
   handles entry/exit at daily granularity; a median over h-day returns silently
   restricts to items surviving the whole window, which biases toward
   long-history cheap items.
2. **Cost.** One `groupby.shift` over the full frame yields all four horizons.
   Four `merge_asof` passes over 5.8M rows would not.
3. **`m̂` falls out of the same object.** Stage 2 needs a trailing daily market
   return series; the index already is one.

`>= $1` on `d-1` because that is the served cohort (`MIN_SERVED_PRICE_USD = 1.0`
in `api/serving_policy.py`, equal to `HEADLINE_MIN_TIER`). Below $1 a one-cent
tick on a $0.03 item is a 33% move, so the penny tier's median is rounding noise.

### The factor is computed on the full frame, before subsampling

This is load-bearing and easy to get wrong. `_stratified_item_subsample` cuts the
frame to ~99 items to hit the 100,000-row budget; at `>= $1` that leaves roughly
20 items per date, well under any usable cross-sectional median. The index must
therefore be computed over the **full pre-subsample price history** (~5,378
items) and joined onto the subsampled frame by date. It is a per-date aggregate —
about 1,460 rows — so carrying it across the subsample costs nothing.

`MIN_INDEX_ITEMS = 30`. A date whose index return is invalid makes any `m[d, h]`
whose window spans it NaN.

### NaN `m` must not drop rows

Dropping rows with a NaN `m` would change `n_train` and `n_val` in the treatment
arm only, breaking the row-identical pairing the entire protocol rests on. So a
row with no valid `m` is demeaned by **zero** — `e = r`, i.e. it falls back to
the control's label for that row — and is counted in `market_factor_coverage`.

The fallback is measured rather than hidden: if coverage is low, the read is
diluted toward the control and the fold is reported as such. Coverage is expected
to be near-total — the index needs 30 items priced `>= $1` on a given day and the
median-`>= $1` cohort holds 925 items (served-cohort spec, §Not in scope) — so
this should be a guard that never fires, not a routine path. A run whose coverage
falls below 95% on any fold is void, for the same reason a broken pairing is.

### Calendar gaps

The archive is missing whole days (`aggregator-archive-day-gaps`). `I[d+h]` is
resolved by as-of lookup to the nearest available date `>= d+h` within
`INDEX_TOLERANCE_DAYS = 3`, matching the `LAG_TOLERANCE_DAYS = 3` convention
already used for lag features. Beyond tolerance, `m` is NaN.

### The label

```
e[i, d, h] = target_return_{h}d[i, d] - m[d, h]
```

Applied to the **directional classifier only**. The quantile models,
`_compute_sample_weights` and the split-conformal band are untouched, so they
cannot confound the read — the same isolation discipline the served-cohort
experiment used.

### Flat-band recalibration

`_direction_classes` buckets into down/flat/up on a `±DIRECTION_FLAT_TOLERANCE_PCT`
(0.5) band. Residuals have smaller dispersion than raw returns, so reusing 0.5
unchanged would inflate the flat class, push the classifier toward predicting
flat, and produce a **fake negative** that looks like "relabelling hurt".

The residual arm's band is therefore set per horizon so its **flat-class share
matches the control arm's**: take the control's flat share `f_h` on the same
rows, and set the residual band to the `f_h`-quantile of `|e|`. Class balance is
then held constant across arms and the only difference is where the label is
centered.

This is a controlled confounder, not a tuned hyperparameter — the target share is
read off the control, never optimized against the outcome.

## Stage 2 — a leak-free market forecast

At forecast date `d`, the most recent **resolved** h-day window ended at `d`,
having started at `d-h`. So the visible history of realized `m` is
`{m[d', h] : d' <= d - h}`. Equivalently, and more simply: only daily index
returns up to and including `d`.

**Pre-registered estimator**, fixed before any arm runs:

```
m̂[d, h] = 100 * (exp(h * mean(ret[d-179 : d])) - 1)
```

the trailing 180-day mean daily index log-return, scaled to the horizon. This is
the honest analogue of the always-down constant that currently beats the model:
a slow-moving unconditional drift, estimated from data available at `d`.

Two alternates are computed and reported as **diagnostics only**, with no bearing
on the gate:

* `trailing_k_median` — median of the last 4 non-overlapping resolved `m[·, h]`.
* `past_h_momentum` — the realized index return over `d-h → d`.

Reporting them guards against the estimator choice being the reason for a
negative result, without letting a post-hoc pick inflate a positive one.

### Leakage is the primary risk in this design

The realized `m[d, h]` is built from other items' **future** prices. Using it at
serve or eval time would produce a large fake gain. Two defences:

1. `forecast_market_factor` takes the daily return series and an as-of date and
   is structurally incapable of reading past it — the series is truncated before
   any statistic is computed.
2. A test asserts that for a frame where all data after `d` is replaced with
   NaN, `forecast_market_factor(d)` returns an identical value.

## Reconstruction and the served call

```
r̂[i, d, h] = m̂[d, h] + ê[i, d, h]
direction  = band(r̂)          # existing DIRECTION_FLAT_TOLERANCE_PCT yardstick
```

Scored against the same fixed ±0.5% actual-class yardstick the control uses, so
`classifier_accuracy_ge1` stays directly comparable to today's 49.9 / 49.0 /
51.1 / 53.4.

The classifier emits a 3-class label, not a magnitude. `ê` is taken as the
class's signed representative: the residual arm's per-horizon band edge for
up/down, 0 for flat. This is coarse and it is a real limitation — it means stage 2
can only shift the decision boundary, not rank within a class. It is accepted for
this measurement because the alternative (a residual quantile model) doubles the
build to test a hypothesis that stage 1 may kill outright.

## Where the code lives

| File | Change |
|---|---|
| `backend/models/market_factor.py` | **new.** `build_market_index`, `market_factor_for_horizon`, `forecast_market_factor`. Pure functions over DataFrames; no `ItemForecaster` internals, no DB, no I/O. |
| `backend/models/forecaster.py` | `market_relative_labels: bool = False` on `__init__`; the market index computed once in `build_training_data` before subsampling; the two `_fit_direction_classifier` call sites demean `y` when the flag is on; two new reported CV metrics. |
| `backend/scripts/forecast_prices.py` | `TRAIN_MARKET_RELATIVE_LABELS` env knob, `DEFAULT_MARKET_RELATIVE_LABELS = False`. |
| `backend/scripts/ab_test_market_relative_labels.py` | **new.** Driver, not a harness — runs two cold `--train-only` retrains via `FORECAST_MODEL_DIR` and prints the paired table. |
| `backend/tests/test_market_relative_labels.py` | **new.** |

`forecaster.py` is already 5,633 lines. The factor logic goes in its own module
so it can be understood and tested without loading the forecaster.

### Why not a standalone harness

`scripts/ab_test_*.py` mostly build their own walk-forward loops. That pattern
produced `2026-08-06-volume-ab-and-harness-defects.md`: the harness scored a
cohort 92% of which production never serves, making ~31pp of its reported
directional accuracy free hits — a defect that "calls every prior null measured
through it into question". Both arms here
run through the real `_cv_evaluate_horizon` on the real retrain path; the script
only sets env vars, invokes the two runs and diffs their `meta.json`.

## Measurement protocol

Two cold `--train-only` retrains into scratch directories via
`FORECAST_MODEL_DIR`. The deployed artifact is not touched.

* **Control:** `TRAIN_MARKET_RELATIVE_LABELS=0` — provably the current model.
* **Treatment:** `TRAIN_MARKET_RELATIVE_LABELS=1`.

Both arms see the same items, the same folds and the same `tuned_params`; only
the classifier's label vector differs. This is genuinely paired — unlike the
subsample variant the served-cohort spec declined, which changes the item
universe and is unpaired by construction.

**Pairing is verified before any metric is read:** identical fold ids, `n_train`,
`n_val` and date bounds per horizon, and identical `tuned_params`. If they
differ, the run is void.

### New per-fold metrics — reported, never gated

* `relative_accuracy_ge1` — `sign(ê)` vs `sign(e)` on `>= $1` validation rows.
  The stage-1 kill switch.
* `classifier_accuracy_ge1` — recomputed on `sign(m̂ + ê)`. The headline.
* `market_factor_coverage` — share of validation rows with a non-NaN `m`. A low
  value invalidates the fold rather than the hypothesis.

### Pre-registered decision rule

Fixed now, before anything runs.

1. **Kill at stage 1.** If `relative_accuracy_ge1` <= 51% at *every* horizon,
   there is no idiosyncratic directional signal to recover. Stop. Do not
   interpret stage 2, and do not tune the `m̂` estimator to rescue it.
2. **Recommend adoption** only on a paired mean difference in
   `classifier_accuracy_ge1` of **> +2pp at two or more horizons**, and **not
   worse than −1pp at any**.
3. **Within ±1pp** → market-date domination is not addressable by relabelling.
   Record the result, default stays off, close it.

The +2pp bar is set by the project's own noise floor: the A/B design cannot
resolve sub-1pp effects (`ab-harness-noise-floor`), and the served-cohort run
demonstrated that one nominally significant horizon out of four is unremarkable
(7d at t=2.39 offset by 30d at t=−2.13).

Adoption is a **recommendation for a follow-up change**, not part of this work.
Nothing ships to production here regardless of outcome.

## Testing

`backend/tests/test_market_relative_labels.py`:

* **Leakage** — `forecast_market_factor(d)` is unchanged when all data after `d`
  is nulled. The single most important test in the file.
* **Control-arm byte-identity** — `market_relative_labels=False` produces the
  same label vector, same weights, same booster as before the change.
* **Index construction** — known synthetic prices give a hand-computed index;
  entry/exit mid-window handled; `< MIN_INDEX_ITEMS` days marked invalid.
* **`>= $1` restriction** — a penny item swinging 50% does not move the index.
* **Calendar gaps** — a 2-day gap resolves within tolerance, a 5-day gap yields
  NaN.
* **Flat-share matching** — the residual arm's band reproduces the control's flat
  share to within a row.
* **Degenerate dates** — snapshot days (`SNAPSHOT_DAY_FLAT_FRACTION`) and dates
  where every item is flat produce `m = 0`, not NaN or a crash.
* **Reconstruction** — `band(m̂ + ê)` with `m̂ = 0` reduces exactly to the
  residual arm's own call.

* **NaN-`m` fallback** — a row with no valid `m` gets `e = r` and is counted as
  uncovered; row counts are unchanged.

Regression: `backend/tests/` in full, with the count recorded in the changelog
rather than asserted here. `scripts/test_social_signal.py` fails to import on a
missing `thefuzz` under a bare `pytest -q` — pre-existing, unrelated, the deleted
social collector.

## Explicitly not in scope

* **No production change.** `DEFAULT_MARKET_RELATIVE_LABELS = False`.
* **No residual quantile model.** The classifier alone carries the served
  direction; adding a residual q50 doubles the build before stage 1 has reported.
* **No serving-path change.** `predict()` is untouched. If the gate clears, the
  serving reconstruction is a separate spec.
* **No `m̂` estimator tuning.** One pre-registered estimator, two diagnostics.
  Tuning after seeing the outcome is how a null result becomes a false positive.
* **No re-litigation of the shelved feature groups.** Cross-sectional features
  stay out of the allowlist in both arms.

## Related

* `docs/changelog/2026-08-03-accuracy-is-clustered-by-forecast-date.md` — the
  finding this spec acts on.
* `docs/changelog/2026-08-06-served-cohort-weighting-refuted.md` — the paired
  cold-retrain methodology reused here, and the "Still open" item this addresses.
* `docs/changelog/2026-08-06-volume-ab-and-harness-defects.md` — why the arms run
  through the live path rather than a bespoke harness.
* `docs/specs/2026-08-05-cv-cohort-parity-design.md` — origin of
  `classifier_accuracy_ge1`.
