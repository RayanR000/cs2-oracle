# An instrument for market-relative direction labels — built, not yet measured

**Date:** 2026-08-06
**Spec:** `docs/superpowers/specs/2026-08-06-market-relative-labels-design.md`
**Plan:** `docs/superpowers/plans/2026-08-06-market-relative-labels.md`
**Commits:** `444268f` (new files); the `forecaster.py` / `forecast_prices.py` edits are
still uncommitted — see "Commit provenance" below.

**Measured later the same day — the hypothesis was REFUTED.** See
`docs/changelog/2026-08-06-market-relative-labels-refuted.md`: pre-registered rule 1 fired,
`relative_accuracy_ge1` came in at 36.7 / 32.7 / 34.6 / 39.0 against a majority-class
baseline of 38.8 / 42.7 / 46.4 / 51.7, and the default stays off. Read that entry for the
result; this one remains the record of **what was built and why**, written before any number
existed.

Everything below was written pre-measurement and is left unedited, including the two
assumptions the run contradicted (the flat-band dispersion direction, and the `<= 51%`
threshold's implicit 50% chance baseline).

## The hypothesis: the label's variance is dominated by a term the features cannot see

Decompose an item's realized forward return:

```
r[i, d, h]  =  m[d, h]  +  e[i, d, h]
```

`m` is the cross-sectional common move over `d → d+h`; `e` is item-specific. The
directional classifier is trained on `r`. `m` is driven by Valve events, case releases and
Steam-wide flows — none of which appear in the 36-column price-technicals feature set — so
the model cannot reduce that term and appears instead to settle on a near-constant tilt.

The evidence that this is what happened, all from prior entries and the deployed artifact:

* It predicts "down" on **57–87% of rows regardless of date**
  (`2026-08-03-accuracy-is-clustered-by-forecast-date.md`).
* Its accuracy tracks the market rather than the item: **33.4% on a rising date vs 63.7%
  on a falling one, at 7d** (same entry).
* Per-fold CV `classifier_accuracy_ge1` at 30d spans **20.2% to 96.3%**
  (`2026-08-06-served-cohort-weighting-refuted.md`, both arms).
* **A constant always-down call beats it on every stored forecast date.**
* Deployed served accuracy — `classifier_accuracy_ge1` from
  `models/saved_models/meta.json`, trained 2026-08-06 05:17 UTC — is **49.9 / 49.0 / 51.1
  / 53.4** at 3/7/14/30d. This is the ≥$1 cohort (`MIN_SERVED_PRICE_USD = 1.0`), not the
  pooled all-tiers figure.

The hypothesis to be tested is that training the classifier on `e` instead of `r` removes
an unlearnable term from the loss, and that rebuilding the absolute call as `m̂ + ê`, with
`m̂` estimated from history only, then beats the current absolute model. Whether that is
true is exactly what has not been measured.

## Why this is not one of the already-refuted variants

| Prior result | Why it does not cover this |
|---|---|
| Cross-sectional **features** dropped (`FEATURE_GROUP_ALLOWLIST = ["price_technicals"]`, 2026-07-24; `HORIZON_EXCLUDED_GROUPS` at 14d/30d, 2026-07-19) | Adds columns while leaving the loss dominated by `m`. |
| Served-cohort **weighting** refuted at ~0.1pp (`2026-08-06-served-cohort-weighting-refuted.md`) | Reallocates loss across rows. Cohort mix, not label centering. |
| Vol-scaled direction labels refuted (2026-07-27) | Rescales the flat band. Does not re-center the label. |

This changes **where the label is centered**. The served-cohort entry closes with
market-date domination listed under "Still open", as "the larger unexplained term in
served accuracy"; this instrument addresses that item.

## What was built, all default-off

Production behaviour is unchanged: `DEFAULT_MARKET_RELATIVE_LABELS = False`, and every new
path is guarded on the flag.

**`backend/models/market_factor.py`** — new, pure functions over DataFrames, no DB and no
`ItemForecaster` internals:

* `build_market_index` — chain-linked equal-weighted daily index over the ≥$1 cohort: the
  median of paired day-over-day log returns, cumulated. `MIN_INDEX_ITEMS = 30`,
  `MARKET_MIN_PRICE_USD = 1.0` applied to the *prior* day's price. Chain-linked rather than
  a median of h-day returns because items enter and leave the archive, because one
  `groupby.shift` yields all four horizons, and because the daily return series is what the
  forecast estimator needs anyway.
* `market_factor_for_horizon` — the realized `m`, resolved as-of within
  `INDEX_TOLERANCE_DAYS = 3` (the archive is missing whole days). NaN when the window's end
  date is out of tolerance or the window spans a day whose cross-section was too thin.
* `forecast_market_factor` — the **pre-registered** estimator: the trailing 180-day mean
  daily index log-return, compounded to the horizon. The honest analogue of the always-down
  constant that currently beats the model.
* `forecast_market_factor_diagnostics` — `trailing_k_median` and `past_h_momentum`,
  reported only and **explicitly excluded from the gate**. They exist so the estimator
  choice cannot be blamed for a negative result; they are not eligible to rescue a positive
  one.

**`backend/models/forecaster.py`**:

* `market_relative_labels: bool = False` on the constructor; `self.market_index` and
  `self.direction_bands`.
* The index is built in `build_training_data` **before** `_stratified_item_subsample`. This
  is load-bearing: the subsample cuts to ~99 items, which leaves roughly 20 priced ≥$1 per
  date — far too thin for a cross-sectional median. The index is a per-date aggregate
  (~1,460 rows), so carrying it past the subsample costs nothing.
* `market_factor_{h}d` columns joined onto the frame by date, and **excluded from
  `feature_cols` by `_select_feature_cols`**. The factor is built from other items' future
  prices; it is a label input and never a feature. A test pins this.
* Helpers `_demean_returns`, `_matched_flat_band`, `_residual_point_estimate`; a
  `flat_band` parameter on `_fit_direction_classifier`.
* New reported per-fold metrics `relative_accuracy_ge1` and `market_factor_coverage`, and
  the aggregates `mean_relative_acc_ge1` / `mean_market_factor_coverage`. Reported, never
  gated.

**`backend/scripts/forecast_prices.py`** — `TRAIN_MARKET_RELATIVE_LABELS` env knob,
`DEFAULT_MARKET_RELATIVE_LABELS = False`.

**`backend/scripts/ab_test_market_relative_labels.py`** — the driver.

## Two design decisions that the diff does not explain

**A NaN `m` demeans by zero rather than dropping the row.** Dropping would change
`n_train` and `n_val` in the treatment arm only, which breaks the row-identical pairing the
entire protocol rests on. So a row with no valid factor falls back to the control's own
label (`e = r`). The fallback is measured rather than hidden: `market_factor_coverage` is
reported per fold, and **a run whose coverage falls below 95% on any fold is void**, for
the same reason a broken pairing is.

**The residual arm's flat band is set so its flat-class share matches the control's, per
horizon.** Residuals are less dispersed than raw returns, so reusing the fixed
±`DIRECTION_FLAT_TOLERANCE_PCT` (0.5%) band would inflate the flat class, push the
classifier toward predicting flat, and read as "relabelling hurt" when the real cause is a
class-balance artifact. The target share is read off the control arm and **never optimized
against the outcome** — it is a controlled confounder, not a tuned hyperparameter. The
*actual*-class yardstick stays at a fixed ±0.5% in both arms, so
`classifier_accuracy_ge1` remains directly comparable to today's 49.9 / 49.0 / 51.1 / 53.4.

## The driver is a driver, not a harness

`ab_test_market_relative_labels.py` sets env vars and invokes two real cold `--train-only`
retrains via `FORECAST_MODEL_DIR`, then diffs their `meta.json`. Both arms therefore run
through the live `_cv_evaluate_horizon` on the real retrain path.

That distinction is deliberate. Most `scripts/ab_test_*.py` build their own walk-forward
loops, and `2026-08-06-volume-ab-and-harness-defects.md` measured what that pattern costs:
the harness scored a cohort 92% of which production never serves, making ~31pp of its
reported directional accuracy free hits — a defect that "calls every prior null measured
through it into question". Running both arms through the live path is the direct response
to that finding.

The driver verifies pairing **before reading any metric** — identical fold ids, `n_train`,
`n_val`, date bounds and `tuned_params` per horizon — and raises rather than reporting if
they differ. The coverage check runs in the same block, also before any accuracy number is
touched.

## The pre-registered decision rule

Copied verbatim from the spec (`§ Pre-registered decision rule`), which was written before
any arm ran, so it cannot drift:

> 1. **Kill at stage 1.** If `relative_accuracy_ge1` <= 51% at *every* horizon, there is no
>    idiosyncratic directional signal to recover. Stop. Do not interpret stage 2, and do
>    not tune the `m̂` estimator to rescue it.
> 2. **Recommend adoption** only on a paired mean difference in `classifier_accuracy_ge1`
>    of **> +2pp at two or more horizons**, and **not worse than −1pp at any**.
> 3. **Within ±1pp** → market-date domination is not addressable by relabelling. Record the
>    result, default stays off, close it.

The +2pp bar comes from the project's own noise floor — the A/B design cannot resolve
sub-1pp effects (`ab-harness-noise-floor`) — and from the served-cohort run, which showed
that one nominally significant horizon out of four is unremarkable (7d at t=2.39 offset by
30d at t=−2.13). These will be CV gate numbers from the retrain's own expanding-window
folds, not production DA.

Adoption is a recommendation for a follow-up change. Nothing ships from this instrument
regardless of outcome.

## A known limitation of the reconstruction

The classifier emits a class, not a magnitude. `ê` is therefore taken as the residual arm's
per-horizon band edge (signed for up/down, 0 for flat), which means `m̂` can **shift the
decision boundary but cannot rank within a class**. That is coarse, and it bounds what a
positive stage-2 read would mean. A residual quantile model is out of scope until stage 1
reports: building it now doubles the work to test a hypothesis rule 1 may kill outright.

## What was deliberately not done

* **No production change.** `DEFAULT_MARKET_RELATIVE_LABELS = False`; `predict()` is
  untouched. If the gate ever clears, the serving reconstruction is a separate spec.
* **No `m̂` estimator tuning.** One pre-registered estimator, two diagnostics that are
  excluded from the rule. Picking the best of three after seeing the outcome is how a null
  result becomes a false positive.
* **No residual quantile model** — see the limitation above.
* **No change to the quantile models, `_compute_sample_weights`, or the split-conformal
  band.** Only the directional classifier's label moves, so nothing else can confound the
  read. Same isolation discipline the served-cohort experiment used.
* **No re-litigation of the shelved feature groups.** Cross-sectional features stay out of
  the allowlist in both arms.
* **The arms themselves were not run.** That is the measurement, and it is the next step,
  not this entry.

## Verification

* `backend/tests/test_market_factor.py` — **21 tests**.
* `backend/tests/test_market_relative_labels.py` — **30 tests** (21 functions, 11 of them
  from two `parametrize` sets over the env knob).
* Full `backend/tests/` suite: **948 passed**.

**The leakage tests are the load-bearing ones.** The realized `m` is built from other
items' future prices, so an estimator that reads past its as-of date would produce a large
fake gain that looks exactly like the hypothesis being true. The tests assert that nulling
or poisoning all data after the as-of date leaves `forecast_market_factor` and every
diagnostic estimator's output unchanged. The other structural test is that
`market_factor_{h}d` never reaches `feature_cols`.

A bare `pytest -q` from `backend/` additionally collects `scripts/test_social_signal.py`,
which fails to import on a missing `thefuzz` — pre-existing and unrelated, that is the
deleted social collector.

## How to run it

From `backend/`:

```
python scripts/ab_test_market_relative_labels.py --out /tmp/mrl
```

Each arm writes into its own subdirectory of `--out` via `FORECAST_MODEL_DIR`, so the
deployed artifact in `backend/models/saved_models/` is never touched. `--report-only`
re-reads existing `meta.json` files without retraining.

## Commit provenance

`backend/models/market_factor.py`, `backend/scripts/ab_test_market_relative_labels.py`,
`backend/tests/test_market_factor.py` and `backend/tests/test_market_relative_labels.py`
landed in **`444268f`**.

The `backend/models/forecaster.py` and `backend/scripts/forecast_prices.py` changes were
**deliberately left uncommitted**: they sit in the same files as unrelated in-flight work
(the scale-free-features change, `2026-08-06-scale-free-features-and-fabricated-labels.md`)
and splitting them cleanly was not attempted. **They must be committed before the arms can
run from a clean checkout**, and whichever tree the arms do run from should be recorded
with the result — the served-cohort entry hit exactly this problem and had to attribute its
absolute levels to "the arms' tree" rather than to a named artifact.

## Still open

* ~~The measurement itself.~~ **Closed 2026-08-06** by
  `2026-08-06-market-relative-labels-refuted.md`. The treatment arm ran; rule 1 fired.
  Market-factor coverage came in at 100.0% of validation rows at every horizon, so the
  expectation recorded below was correct.
* Market-factor coverage in practice is unmeasured. The spec expects it to be near-total
  (the index needs 30 items priced ≥$1 on a day, and the median-≥$1 cohort holds 925
  items), but that is a prediction, not a reading.

## Related

* `docs/superpowers/specs/2026-08-06-market-relative-labels-design.md` — the full design and
  the pre-registered rule.
* `docs/superpowers/plans/2026-08-06-market-relative-labels.md` — the implementation plan.
* `docs/changelog/2026-08-03-accuracy-is-clustered-by-forecast-date.md` — the finding this
  instrument acts on.
* `docs/changelog/2026-08-06-served-cohort-weighting-refuted.md` — the paired cold-retrain
  methodology reused here, and the "Still open" item this addresses.
* `docs/changelog/2026-08-06-volume-ab-and-harness-defects.md` — why the arms run through
  the live path instead of a standalone harness.
* `docs/changelog/2026-08-01-deterministic-backtest.md` — the related but distinct finding
  that the *live backtest* was nondeterministic; its scope note is also where
  `scripts/ab_test_*.py` and `scripts/backtest_accuracy.py` are separated as artifacts.
