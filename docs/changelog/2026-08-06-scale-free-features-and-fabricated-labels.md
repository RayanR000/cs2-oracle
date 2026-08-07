# Most of the model's gain was denominated in dollars against a percentage target

**Date:** 2026-08-06
**Change:** three correctness fixes in `backend/models/forecaster.py` — the dollar-scale
columns leave the feature set for scale-free forms, the production train/val split is
purged, and labels are no longer built across days the collector fabricated.
`MODEL_ARTIFACT_VERSION` 4 → 5.
**Commits:** none yet — this landed in the working tree on branch
`docs/refresh-2026-08-05`, alongside two other uncommitted changes noted under
*Provenance* below.

## What this does NOT establish

**There is no accuracy claim here.** All three fixes ship on mechanism, and the paired-arm
measurement was **not run**. Nothing below should be read as "accuracy improved".

The reasons it was not run are the standing ones:

* Production DA remains unmeasurable. Every ≥$1 cohort spans 1–2 distinct forecast dates
  against `MIN_FORECAST_DATES = 20`, so all four horizons report NO HEADLINE
  (`2026-08-03-accuracy-is-clustered-by-forecast-date.md`).
* `scripts/walkforward_backtest.py` diverges from production in four ways **at once**: raw
  classifier argmax vs the four serving transforms; interior rows vs median-filled serving
  rows; a `backfilled_only=False` first-N item slice vs the 5,542 `is_backfilled` items;
  and a plain-mean price consensus vs the outlier-voted one. Its numbers are gate numbers
  and are not comparable to production DA.

The available instrument is the **paired cold-retrain harness** validated in
`2026-08-06-served-cohort-weighting-refuted.md`: two arms via `FORECAST_MODEL_DIR` with
`--train-only`, scored per fold on `classifier_accuracy_ge1`, paired sd 1.3–2.6pp over
9 folds. **Running it against these changes is the open next step**, and until it runs the
only defensible statement is that the previous behaviour was wrong for a reason that can
be written down.

One piece of supporting context, offered as context and not as evidence: the served-cohort
**row** reweighting was refuted at ~0.1pp on 2026-08-06 (same entry). That is consistent
with the defect being feature-side — reweighting rows toward ≥$1 cannot help if the
features cannot represent that range in the first place.

## 1. Dollar-denominated features carried 55.6–86.6% of total gain

The target is `target_return_{h}d`, a **percentage**. Read off the shipped 2026-08-06
artifact's `meta.json` `feature_importance` (`model_artifact_version` 3, 36 columns,
trained 2026-08-06 05:17 UTC), the dollar-denominated columns carried:

| horizon | share of total gain |
|---|---|
| 3d | **55.6%** |
| 7d | **70.2%** |
| 14d | **77.5%** |
| 30d | **86.6%** |

The share rises with horizon, in step with the served-cohort accuracy gap on that same
artifact: `mean_classifier_acc` is **67.2–68.6%** pooled against `mean_classifier_acc_ge1`
of **49.0–53.4%**.

This was visible three days earlier and read past.
`2026-08-06-serving-down-skew-refuted.md` recorded that after the volume shelve "no volume
feature appears in any horizon's top five; those are now `price_std_*`,
`trend_up_fraction_30d` and the MACD legs" — four of those five names are in dollars.

A dollar feature against a percent target cannot be a price signal — it can only encode
which item this is. The scale mismatch is not marginal: the training median price is
**$0.086** and the median `price_std_60d` **$0.0127**, against served items reaching
**$639** and **$72.20**. With `MAX_BIN = 63`, nearly everything above ~$1 lands in one
saturated terminal bin, so the tree learns a per-tier base rate over penny items and has
no resolution left where production actually serves.

`models/conformal.py:48` had already reached this conclusion for the band: `sigma =
price_std_60d / price`, with the docstring "in dollars, so it is divided by price to make
the scale return-space and comparable across price tiers". **The band got the scale-free
form; the model did not.**

### What changed

`_DOLLAR_SCALE_FEATURES` (37 names) joins `SHELVED_FEATURES`, taking it from 19 names to
56. Shelved, **not deleted** — every one is still computed, because the conformal band
reads `price_std_60d`, `_apply_market_aggregates` reads `price_std_30d`, and the z-score,
Bollinger, MA-distance, support/resistance and log-return features are all derived from
the means, mins, maxes and lags. Shelved: all `price_std_*`, `price_mean_*`, `price_min_*`,
`price_max_*`, `price_lag_*`, `price_log`, `macd_line`, `macd_signal`, `macd_histogram`,
`bb_upper`, `bb_lower`.

New scale-free columns: `price_cv_{7,14,20,30,60}d` (= `price_std_{w}d / price`),
`macd_line_rel` and `macd_histogram_rel` (MACD is a difference of price EMAs, so it too is
in dollars). The rest of the shelved set already had scale-free derivatives in the feature
list — `return_{n}d`/`log_return_{n}d` for the lags, `price_zscore_30d` and
`price_dist_ma{100,200}` for the means, `distance_to_support`/`resistance` and
`high_low_range_30d` for the min/max, `bb_pct_b`/`bb_width` for the bands.

`price_tier` is **deliberately kept**. A bounded categorical (0–4) is the honest way to
express price level; an unbounded dollar axis a tree extrapolates from is not. It is the
single documented exemption in `SCALE_DEPENDENT_BY_DESIGN`.

The set is listed exhaustively rather than by prefix for the reason the volume shelf
documented: the >0.95 correlation prune is data-dependent, so a column pruned today can
survive tomorrow.

**Verified by a training smoke run:** the feature set goes **36 → 32 columns**, and
dollar-scale gain is **0.0% at every horizon**, with `price_cv_*` absorbing the gain
`price_std_*` had held. The gain redistribution is the only post-change number here, and
it is a composition figure, not an accuracy figure.

`MODEL_ARTIFACT_VERSION` 4 → 5, because a v4 booster's splits are thresholds *in dollars*
and are meaningless against v5 features. This must not load across the bump.

## 2. The production train/val split was leaking its labels; CV never was

`_compute_cv_splits(..., purge_days=horizon)` has always purged. The production split in
`_train_horizon_inline` did not.

A row dated `d` is labelled with the price at `d + horizon` — `prepare_targets` merges on
exactly that date. So every training row in `[split_date - horizon, split_date)` carried a
label drawn from inside the validation window. At `horizon == VALIDATION_WINDOW_DAYS ==
30`, that is the **entire** window's worth of future prices.

That frame is not a bystander. It produces the `dval` early stopping stops on, the set
Optuna scores every trial against, and the directional classifier's stopping set — so the
shipped tree counts, the shipped hyperparameters and the classifier's stopping points were
all selected against partly-seen labels.

`_purge_overlapping_train_rows` drops the band rather than moving it: those rows belong to
neither side. The positional fallback split (`tdf.iloc[:split_idx]`) leaked identically and
is purged the same way, keyed on `val_set`'s first date.

The size of the effect on the shipped model is **not measured** — see *What this does NOT
establish*. What is measured is the band width: `test_purge_band_is_exactly_horizon_days`
pins 3/7/14/30 rows dropped per horizon on a one-item daily frame, and
`test_at_30d_the_whole_validation_window_would_otherwise_leak` asserts the 30-row leak
exists before the purge and is gone after.

## 3. Two archive days are copies of the day before, and twelve are collector cutovers

Both measured over all **4,735** archive days. Winsorization at ±500% cannot catch either
one: a −31.6% cutover return and a 0% re-published return are both well inside the clip and
survived as confident, completely wrong labels.

### Re-published snapshots — `_snapshot_dates`, `SNAPSHOT_DAY_FLAT_FRACTION = 0.99`

**2026-07-16 and 2026-07-22 are byte copies of the preceding day** across the whole
~40k-item cross-section — 100.00% of items carry an identical price. Exactly those two days
fire archive-wide; the **next-highest day sits at 69.01%**, so the 0.99 threshold lives in a
wide empty gap and is not a tuned parameter. `MIN_DEGENERATE_CROSS_SECTION = 25` keeps a
quiet day on a handful of items from qualifying.

**2026-07-16 is the day before 2026-07-17, the only genuine production forecast date of the
current model lineage.**

Treated as bad **endpoints only**. A copied day shifts no level, so it is harmless mid-window;
the rule voids a label whose anchor *or* whose target lands on one.

### Collector cutovers — `_collection_shift_dates`, `COLLECTION_SHIFT_FRACTION = 0.20`

The archive is a stitch of source regimes, not a continuously-collected panel. Mean market
return reads **−31.6% on 2026-03-22** and **+17.4% then −17.8% on 2026-07-09/10**, against
**±0.5%** on a normal day. Those are basis changes between source sets, not price moves.

Detection is from the **item universe size**, deliberately and never from prices: prices
moving cannot change how many items a collector returns, so this detector **can never mask
a real crash**. That property is pinned by `test_a_price_crash_is_never_flagged`, which
drops the market 60% and asserts neither detector fires.

It fires **12 times in 4,735 days (0.25%)**: 4 in 2013 at archive startup, 1 in 2016, 7 in
2026.

A cutover corrupts any label whose window **spans** it — the anchor is quoted on the old
source basis and the target on the new — so the whole horizon-wide anchor band before a
cutover is voided, not just the cutover day itself.

`prepare_targets` logs the voided count per horizon (`Voided N {h}d targets spanning …`).
The count on a production frame is **not recorded here**; no production retrain has run
against this code yet.

## 4. Sample-uniqueness weighting was investigated and declined

Measured on 300 deep-cohort items / **656,522** real item-days:

| horizon | mean uniqueness | sd | CV | p05 | p95 |
|---|---|---|---|---|---|
| 7d | 0.12549 | 0.0069 | 5.5% | 1/8 | 1/8 |
| 30d | 0.03258 | 0.0035 | 10.9% | 1/31 | 1/31 |

**p05 and p95 are both exactly `1/(h+1)`.** For >90% of rows the weight is an identical
constant, which the existing mean-normalisation divides straight back out.

López de Prado's construction earns its keep on **variable-width** triple-barrier labels.
These labels are fixed-width on a near-dense daily grid, so the concentration is a property
of the grid, not of the data.

The real concern the technique addresses — overstated effective sample size, since 100,000
rows at h=30 carry roughly 3,300 independent labels' worth of information — is a
**reporting** problem, not a training-weight problem. The training-side half of it is what
change 2 fixes. No code was added.

## Verification

Three new test files, **39 tests**; the full `backend/tests/` suite is **840 passed**.

* `tests/test_scale_free_features.py` (18) — the load-bearing test is
  `test_model_features_are_invariant_to_price_scale`: multiply every price by 100 and
  assert every served feature is unchanged. It asserts the **property**, not a name list,
  so a dollar-scale feature added later fails it without anyone remembering to update
  `_DOLLAR_SCALE_FEATURES`. `test_the_test_can_fail` guards against it passing vacuously
  by checking a raw dollar column *is* detected as scale-dependent by the same comparison.
  `test_cv_60d_matches_what_conformal_computes_for_sigma` pins `price_cv_60d` to
  `conformal.sigma_from_columns` so the feature and the band cannot drift into two
  definitions of one quantity. `test_no_dollar_column_survives_the_real_selection_and_prune`
  runs the real selection + prune + allowlist path, because shelving a column changes which
  *other* columns the >0.95 prune keeps — the failure mode a name-list assertion
  structurally cannot see.
* `tests/test_purged_production_split.py` (9) — band width per horizon, the 30d
  whole-window case, and the three degenerate frames.
* `tests/test_degenerate_label_dates.py` (12) — both detectors, the endpoint-vs-span
  distinction (a snapshot day inside a window is asserted harmless), clean data keeping
  every label, and `test_a_price_crash_is_never_flagged`.

Two existing tests were changed, both **reclassification rather than weakening**:

* `tests/test_volume_features_shelved.py` swapped its witness column `price_std_60d` →
  `price_cv_60d`. The witness existed to prove volume shelving does not over-reach into the
  price technicals; it was itself shelved here for an unrelated reason, so the scale-free
  replacement keeps the assertion testing what it was written to test.
* `tests/test_predict_tail_truncation.py` added `macd_line_rel` / `macd_histogram_rel` to
  `EWM_FAMILY`. They are the same `ewm()` quantities divided by price, so they inherit the
  non-finite-memory property exactly — and they are now the members of that family that
  actually reach a booster, so the `EWM_RTOL = 1e-4` bound matters more than it did.

`test_build_training_data_feature_count`'s lower bound moved **30 → 20**. That is a sanity
range, not a pin: the shelving is a net removal, since one `price_cv` column replaces one
`price_std` column while the raw means, mins, maxes and lags leave with no new replacement.
`test_scale_free_features.py` is what asserts the composition.

No production run has executed this code. There is no prod evidence in this entry.

## What was deliberately not done

* **No accuracy measurement.** Restated because it is the most important omission here.
  The paired cold-retrain harness exists and was not run; see the top section for why the
  cheaper gates do not substitute.
* **The dollar columns were not deleted.** They are shelved from the *feature set* only.
  The conformal band, `_apply_market_aggregates` and every derived technical still read
  them, and deleting them would break the band.
* **`price_tier` was not shelved with them.** It is dollar-*derived* but bounded, and it is
  the only remaining way to tell the model how expensive an item is. It is also the column
  the served-cohort weighting reads, which is why that code passes it explicitly rather than
  off `X_train`.
* **No sample-uniqueness weighting was implemented**, on the measurement in section 4 —
  a weight that is a constant for >90% of rows and is then mean-normalised is a no-op with
  a maintenance cost.
* **The snapshot rule was not extended to mid-window days.** A copied day shifts no level,
  so voiding labels that merely span one would discard real data for no gain. The cutover
  rule *is* span-based, because a basis change does shift the level.
* **The cutover detector was not keyed on returns.** A return-based trigger would delete
  genuine market crashes; the universe-size trigger cannot, by construction.

## Provenance

This entry describes an uncommitted working tree that also carries two other changes,
neither of which is claimed here:

* The **served-cohort classifier weighting**, built and refuted, documented in
  `2026-08-06-served-cohort-weighting-refuted.md` (`DEFAULT_SERVED_COHORT_SHARE = None`,
  production behaviour unchanged).
* The **as-of lag lookup** (`LAG_TOLERANCE_DAYS = 3`, `MODEL_ARTIFACT_VERSION` 3 → 4),
  which is the fix for the calendar-gap median-fill left open in
  `2026-08-06-serving-down-skew-refuted.md`. **It has no changelog entry yet and needs
  one.** The v4 → v5 bump recorded above sits on top of it.

Because all three changes share a tree, the smoke-run feature count (36 → 32) is a property
of that tree, not of the scale-free change in isolation.

## Still open

* **Run the paired cold-retrain harness** on these changes. This is the next step, and it
  is the only thing that can turn "wrong for a documented reason" into "measurably better".
* Which of the three fixes, if any, moves `classifier_accuracy_ge1` is entirely unknown.
  They landed together and would need separate arms to attribute.
* The market-date domination of DA (`2026-08-03-accuracy-is-clustered-by-forecast-date.md`)
  is untouched by all of this and remains the larger unexplained term.
* The as-of lag change needs its own entry.

## Related

* `2026-08-06-served-cohort-weighting-refuted.md` — the **row**-level version of the
  train/serve mismatch, refuted at ~0.1pp; the harness and the `FORECAST_MODEL_DIR`
  override this work would use.
* `2026-08-06-volume-features-shelved.md` — the previous column-level train/serve fix, and
  the source of the prune-interaction failure mode both test suites now guard against.
* `2026-08-03-accuracy-is-clustered-by-forecast-date.md` — why production DA is not
  available as a gate.
* `2026-07-31-price-primitives-decision-scale.md` — the measurement floor that makes an
  unpaired comparison useless at this effect size.

## Docs updated in the same pass

`docs/architecture/model.md`:

* `MODEL_ARTIFACT_VERSION` 3 → 5, with what v4 and v5 each mark.
* `SHELVED_FEATURES` 17 → **56**, split into its three groups. The old "17 / eleven volume"
  count was already stale — the volume shelf is thirteen names, not eleven.
* Feature-count trail 47 → 36 → 32, and the note that shelved columns are still computed.
* Price technicals: `price_cv_*`, `macd_*_rel`, the scale-free property and the test that
  pins it; and the as-of lag lookup (`LAG_TOLERANCE_DAYS = 3`), which belongs to the
  undocumented v4 change but left the previous paragraph describing code that no longer
  exists.
* New **Label hygiene** subsection under Training Pipeline.
* Purged production train/val split under Validation and calibration.
* Known limitations: the `+13.4pp` serving down-skew bullet replaced with the refuted
  like-for-like figures, sourced from `2026-08-06-serving-down-skew-refuted.md` — it was
  stale before this pass.
* Accuracy table refreshed to the 2026-08-06 artifact with the ≥$1 column alongside pooled.
* File reference line counts and the three new test files.

`docs/architecture/model-optimization.md`: the **Features** row, which carried the same
47 → 36 trail and the same stale "11 volume features".
