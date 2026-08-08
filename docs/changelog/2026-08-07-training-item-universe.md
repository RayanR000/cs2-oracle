# Production trains on 99 items and serves 8,691 — the item draw is worth ±3pp, and a $1 price floor buys +3.50pp at 30d

**Date:** 2026-08-07
**Change:** `backend/models/forecaster.py` gains `_filter_by_median_price` and a
`min_median_price` parameter on `build_training_data` and `train`;
`backend/scripts/forecast_prices.py` gains `_train_min_median_price()` reading
**`TRAIN_MIN_MEDIAN_PRICE`**, defaulted `None`. Production is byte-identical until the
variable is set. Also: `fold_id` threaded through three A/B harnesses so they keep working
against the fold-clustered `paired_mde`.
**Bears on:** `2026-08-06-data-acquisition-ranking.md` (two numbers corrected, below),
`2026-08-06-breadth-beats-depth-item-age-does-not.md` (its CIs were date-clustered),
`2026-08-06-served-cohort-weighting-refuted.md` (this is the "more ≥$1 items" test it left
open).

The question was "am I training on an optimal number and set of items?". The answer has
four parts: the position was mis-recorded, the item draw dominates every effect this
project chases, item *type* and *weapon identity* are dead axes, and a $1 median-price
floor is worth shipping — for measurability first and accuracy second.

## The position was 99 items, and two documented numbers hid it

`DEFAULT_TRAIN_FEATURE_ROWS = 100_000` (`backend/scripts/forecast_prices.py:45`) buys a
**99-item, ~115,700-row** subsample of the 5,542-item pool. `price-archive/ops/item_forecasts.parquet`
carries **8,691 distinct items** on its latest date. The model learns from 1.8% of the
items it scores.

Two things obscured that:

* **`docs/changelog/2026-08-06-data-acquisition-ranking.md`, § "The constraint that bounds
  all of it", named the binding budget as "`TRAIN_FEATURE_ROWS`, 700K".** 700K is `TRAIN_HORIZON_MAX_ROWS` — the *per-horizon* cap
  applied **after** feature engineering, which `forecast_prices.py` itself documents as
  never binding at the feature budget. The real position is 5× smaller than recorded, and
  the same entry's breadth reading (§ below) was anchored on the wrong number as a result.
* **The cost objection in `models/forecaster.py::train` is priced as a daily cost.** It
  reads "at 700_000 (646 items) training costs 468.7s against 104.6s at 100_000 (99
  items) — more than the 462s the pre-rewrite 40-model grid cost".
  `.github/workflows/price-forecast.yml:75–79` sets `mode=full` only when `date +%u` is
  `1`; every other day is predict-only. A budget increase is paid **once a week**, not
  daily. That does not make the comparison false, but it changes what it is a comparison
  to.

## The item draw is the largest effect in the system

`_stratified_item_subsample` (`models/forecaster.py:2994`) takes `seed: int = 42`, hard-coded
at both call sites. Eight full production-config retrains were run varying **only** that
seed, into isolated `model_dir`s.

`mean_classifier_acc_ge1` across the eight draws, percentage points:

| h | mean | sd | range |
|---|---|---|---|
| 3d | 49.59 | 1.82 | 46.2 – 52.0 (5.80) |
| 7d | 48.55 | 3.05 | 43.5 – 51.9 (8.40) |
| 14d | 49.09 | 1.54 | 46.4 – 50.6 (4.20) |
| 30d | 50.88 | 2.77 | 47.7 – 56.1 (8.40) |

All-tiers `mean_classifier_acc` moves comparably, sd **2.36–2.54pp**. The control rules out
general nondeterminism: **seed 42 run twice differed by ≤0.2pp**. The draws are also
near-disjoint — Jaccard **0.010–0.026** against seed 42, i.e. **2 to 5 shared items out of
99** — so these are eight almost independent universes, not eight perturbations of one.
Served-cohort (≥$1) items per draw: **17–23, mean 20.1**.

Two consequences, and they point in opposite directions:

* **Paired tests at a fixed seed are not confounded.** Both arms see the same 99 items, so
  the served-cohort weighting refutation (`2026-08-06-served-cohort-weighting-refuted.md`)
  and the paired ByMykel retrains are unaffected by this.
* **Any conclusion drawn from a single retrain's absolute level is.** Specifically the
  "the train/serve gap is closed at 3 of 4 horizons" reading carried by the 2026-08-06
  retrain: across these eight seeds the **7d residual spans −5.9 to +2.5pp and changes
  sign**, and the **30d residual spans +1.0 to +9.4pp**. The qualitative claim — 30d is the
  bad horizon — survives every draw. The magnitudes do not survive any of them.

The sweep also produced a second reading on the model's own instrument. Across the 32
horizon-runs, the in-model permutation test on `price_technicals` — the only allowlisted
group — reported **7 WARN**, three of them with a **negative** shuffled delta (−1.83,
−0.12, −0.10pp), meaning shuffling the only feature group the model has *improved*
accuracy. At 99 items the fit is not reliably distinguishable from its own permutation.

## Item type and specific weapon are dead axes

Measured directly on `price-archive/ops/forecast_outcomes.parquet`
(`model_version='lgbm-v3'`, `base_price >= 1`) joined to `weapon_type` — already-scored
production outcomes, no retrain involved.

**By type** (pistol / rifle / smg / sticker / sniper / case), χ² p = **0.66 / 0.36 / 0.19 /
0.77** at 3/7/14/30d. Nothing at any horizon.

**By specific weapon** (16 weapons with n ≥ 100) χ² p = **0.001**, DA ranging 31.0–52.9%.
That looks like signal and is not:

* The model predicts "down" on **73.4%** of rows against an actual down-rate of **50.0%**.
* **corr(DA, actual-down-share) = 0.70** across the 16 weapons — the spread is measuring
  which weapons happened to fall, not which the model understands.
* Against the always-down constant it competes with, the model wins on **2 of 16** weapons,
  mean edge **−4.5pp**. Best: MP7 +7.6, Dual Berettas +3.8. Everything else is negative,
  down to MP9 at −13.6.
* Desert Eagle (50.0%) and StatTrak™ Desert Eagle (31.0%) are **the same weapon, 19pp
  apart**, which is the cleanest evidence that the variation is not tracking weapon
  identity.

This is `no-idiosyncratic-signal-in-features` and `da-is-dominated-by-market-date`
reappearing one level down. There is no item-type or weapon-family stratification worth
building.

## The composition problem, quantified

Training pool over the production window (1460 days, `is_backfilled`, deduplicated):
**5,542 items / 6,077,412 item-days / 1,096.6 rows per item**. **82.56% of item-days are
sub-$1.** (`2026-08-06-served-cohort-weighting-refuted.md` recorded 81.96% on a 5,378-item
/ 5,832,742-item-day snapshot to 2026-08-04; this is a re-measurement on a larger snapshot,
not a contradiction.)

By type the pool is **32.5% sticker + 11.5% graffiti**, median price **$0.03** each. The
≥$1 cohort is **24.7% rifle, 22.0% pistol, 0% graffiti**. The subsample stratifies on
*rarity*, so it inherits this mix wholesale.

The number that decides the design: the **≥$1 cohort is 926 items / 993,464 item-days /
1,072.9 rows per item** — essentially the pool's own density. **A price floor costs no
history**, and a budget above ~1.0M covers the entire served cohort with **no subsample at
all**.

## The change: a floor on the universe, applied before the budget is spent

`ItemForecaster._filter_by_median_price(price_df, min_median_price)` keeps whole item
histories whose **median** `price` clears the floor. Threaded through `build_training_data`
and `train` as `min_median_price`, default `None`.

Placement is the entire point. It runs **after** `_filter_dead_items` and **before**
`_stratified_item_subsample`. Filtering after the subsample would spend the row budget on
the pool and then discard four fifths of what it bought; filtering before makes the budget
buy served-cohort breadth. `test_floor_is_applied_before_the_subsample` asserts the source
order for exactly this reason.

Median rather than mean, and over the whole window: a single print must not promote a penny
item (`test_uses_the_median_not_the_mean`), and it is the same statistic the 926-item cohort
was sized on.

`scripts/forecast_prices.py::_train_min_median_price()` reads `TRAIN_MIN_MEDIAN_PRICE`,
returning `None` for unset, unparseable, or non-positive values. It is env-configured
because the script parses argv as a plain set, matching `TRAIN_FEATURE_ROWS` and
`TRAIN_SERVED_COHORT_SHARE`.

## The full ≥$1 run: no subsample, 4/4 permutation PASS, and the first served-cohort baselines

`TRAIN_MIN_MEDIAN_PRICE=1.0`, `TRAIN_FEATURE_ROWS=1_200_000`, and `max_rows` raised to 3M.
The `max_rows` change is not cosmetic: at the default 700K the per-horizon cap **would have
bound here** (958,289 train rows at 3d) **and only here**, which would have confounded
breadth against a random row downsample in the treatment arm alone.

Result: **926 of 5,542 items, 993,464 rows, no subsampling, 538s (9.0 min)** against
~95–140s for the 99-item baseline runs.

`mean_classifier_acc_ge1`: **50.30 / 49.70 / 51.30 / 58.60** at 3/7/14/30d. The **permutation
test passes 4/4 at p = 0.0000**, against 7 WARN in 32 horizon-runs at the 99-item baseline.

Because the universe *is* the served cohort, all-tiers and ≥$1 metrics coincide, so the CV
baselines are on the served cohort for the first time in this project:

| baseline | 3d | 7d | 14d | 30d |
|---|---|---|---|---|
| persistence | 10.70 | 7.20 | 5.10 | 3.30 |
| momentum | 36.60 | 38.90 | 40.00 | 48.80 |
| **`edge_vs_best_baseline`** | **+13.60** | **+11.00** | **+11.60** | **+9.80** |

`offline-da-inflated-by-stale-prices` recorded that "whether the model beats momentum on
≥$1 is unmeasured". **It does, at every horizon.** And persistence collapsing from the
pooled 41–45% to **3–11%** confirms that the old persistence baseline was living on
carry-forward staleness in penny items — it was never a baseline for the served product.

**One caveat that limits all of it:** the constant always-down call that beats the model on
every stored production date (`da-is-dominated-by-market-date`) is **not among these
baselines**. Clearing momentum and persistence is not clearing the thing that actually
beats this model.

## The paired rerun, which halved the headline

The 58.60 vs 50.88 comparison at 30d is **not paired**. The baseline's `acc_ge1` is computed
over the ~20 ≥$1 items that happen to fall inside a 99-item draw; the treatment's is
computed over 926. Different evaluation cohorts, so most of the gap could be cohort.

A paired harness was built inheriting `ab_test_training_breadth.py`'s metric and fold
schedule: DA(strict, ≥$1) with flat-actual rows dropped, 21-day validation window, 60-day
stride, **150 held-out ≥$1 items evaluated on identical rows in every arm**, 25–26 folds,
`paired_da_difference` with `cluster_key="fold_id"`.

Arms: `prod_a` — 99 items at production's *measured* composition (17 ≥$1 + 82 sub-$1);
`prod_b` — a **disjoint** second draw of the same shape, i.e. the placebo; `ge1_budgeted` —
728 ≥$1 items capped at `prod_a`'s row budget; `ge1_full` — the same 728 items uncapped.

Paired difference vs `prod_a`, held-out items, 95% CI, percentage points:

| h | prod_b (placebo) | ge1_budgeted | ge1_full |
|---|---|---|---|
| 14d | −1.60 [−4.62, +0.92] | +2.29 [−0.87, +5.59] | +1.12 [−2.09, +4.03] |
| 30d | +2.09 [−1.43, +5.96] | **+5.72 [+2.91, +9.02]** | **+3.50 [+1.56, +5.98]** |

MDE 2.21–3.69pp. `prod_a` absolute DA: 53.06 (14d), 52.21 (30d).

Three readings:

1. **The placebo is null at both horizons, but it moves ±2pp from redrawing items alone.**
   That is the seed-variance result reproduced on a second, independent instrument.
2. **30d survives pairing at +3.50pp — so roughly half of the +7.72pp unpaired gap was the
   evaluation-cohort artifact.** 14d does not survive; its CI covers zero in both treatment
   arms.
3. **More rows hurt.** `ge1_full` carries 728K rows/fold against `ge1_budgeted`'s 71K and
   loses at *both* horizons. That replicates the breadth harness's `wide_unbudgeted`
   finding: what pays is item diversity per row, not row count.

**`ge1_budgeted` — the better arm — cannot be expressed in production today.**
`_stratified_item_subsample` selects whole item histories, so a 110K budget under the floor
buys ~93 items at full depth, not 728 items at ~98 rows each. Realising it needs per-item
row sampling, which is a further change and was not built. **The shippable configuration is
`ge1_full`**, and +3.50pp at 30d is what shipping it is worth on this instrument.

## The fold-clustering fix, and what it invalidates

`backend/backtest/paired_mde.py` carries an uncommitted 2026-08-07 change making
`cluster_key` explicit and defaulting it to `fold_id`, on the grounds that clustering the
bootstrap on `forecast_date` under-disperses the interval — every date inside one fold's
validation window is scored by the same fitted model. Its docstring names the CSFloat
basis, ByMykel metadata and training-breadth A/Bs as results to re-derive, and it
deliberately **raises** rather than silently falling back to dates.

Three harnesses called `paired_da_difference` without a `cluster_key` and emitted no
`fold_id`, so they would now raise `ValueError`. Fixed here by threading `fold_id` through
record construction and passing `cluster_key="fold_id"`:
`scripts/ab_test_training_breadth.py`, `scripts/ab_test_item_metadata.py`,
`scripts/ab_test_csfloat_basis.py`. `scripts/ab_test_volume_features.py` does not call the
function. `scripts/compute_mde.py` needs no change — its records come from
`backtest/walkforward_records.py::fold_records`, which now emits `fold_id`.

**The CIs recorded in those three experiments were date-clustered and are too narrow.** On
a comparable design at a comparable fold count (25–26), fold-clustering produced an MDE of
**2.21–3.69pp** where the breadth run reported **0.32–0.78pp** — a factor of roughly 5.
Every sub-1pp finding in those entries should be treated as unresolved until re-derived,
including breadth's headline **"+1.18pp at 14d [+0.58, +1.83]"**. **They were not re-run
here and nothing below claims they were.** The direction of the error is the dangerous one:
it overstates significance.

## Recommendation

Ship `TRAIN_MIN_MEDIAN_PRICE=1.0` with `TRAIN_FEATURE_ROWS` ≥ 1.0M. Ranked by how well each
reason is supported:

1. **Measurability.** No subsample means the item draw is not a random variable any more —
   the sd 1.5–3.1pp this entry measures goes to zero by construction, and the model's own
   permutation test goes 7-WARN-of-32 → 4/4 PASS at p = 0.0000. This is the strongest
   reason and it is not an accuracy claim.
2. **30d accuracy, +3.50pp paired** [+1.56, +5.98] on held-out ≥$1 items.
3. **Null at 3d, 7d and 14d.** Nothing is claimed there.

Cost: **+7 minutes on a weekly retrain** (538s vs 95–140s), paid on Mondays only.

**The consequence to carry:** with the floor on, the model never sees sub-$1 data, while
`predict()` still writes forecasts for all 8,691 items and `MIN_SERVED_PRICE_USD = 1.0`
filters them at serve time. Sub-$1 forecasts would then be pure extrapolation. Nothing
displays them today, but nothing in the code enforces that either.

## What was deliberately not done

* **The floor was not turned on.** `TRAIN_MIN_MEDIAN_PRICE` is unset and
  `min_median_price` defaults to `None`, so the daily and Monday paths are byte-identical
  to before this change. The evidence above is CV and held-out-item CV; none of it is
  production DA, and the decision to enable belongs to whoever accepts that gap.
* **Per-item row sampling was not built**, so the best-measured arm (`ge1_budgeted`,
  +5.72pp at 30d) cannot be run in production. Building it changes what
  `_stratified_item_subsample` means for every existing result, which is not a change to
  make in the same pass as the measurement that motivates it.
* **The three date-clustered A/Bs were not re-derived.** The harnesses now emit `fold_id`
  and would produce correct intervals on a re-run; the re-runs cost full sweeps and were
  not done. This entry records the invalidation, not the correction.
* **`seed` was not exposed as a knob and multi-seed averaging was not adopted.** Averaging
  8 draws would cost 8× the retrain to buy a number the price floor makes unnecessary.
* **No item-type or weapon stratification was built.** § above is the reason: both axes
  measured dead, and the one that looked alive was tracking the market's direction per
  weapon rather than the model's skill.
* **3d and 7d were not run in the paired harness.** The unpaired sweep put both inside a
  point of the baseline, and the harness's MDE (2.21–3.69pp) cannot resolve that. Only 14d
  and 30d had a candidate effect large enough to be worth the folds.
* **`max_rows` was raised to 3M for the treatment run only, and was not changed in the
  code.** `TRAIN_HORIZON_MAX_ROWS` is still 700K. Anyone enabling the floor at a ≥1.0M
  feature budget must raise it too, or the per-horizon cap silently becomes the binding
  constraint at 3d.

## Verification

* `backend/tests/test_training_item_coverage.py` — two new classes. `TestMedianPriceFloor`
  (7 cases): median-not-mean promotion, whole histories survive the filter, no-op when
  every item clears, both signatures expose `min_median_price` defaulted `None`, `train`
  forwards it, and the source-order assertion that the floor precedes the subsample.
  `TestTrainMinMedianPriceEnv` (6 cases, 4 of them one `parametrize`): unset, honoured,
  and `""` / `"dollars"` / `"0"` / `"-1"` all disabling the floor. **13 new cases**,
  bringing the file to **25**.
* `backend/tests/test_paired_mde.py` — 6 new cases covering the fold-clustering default,
  fold-clustered intervals being wider than date-clustered ones on the same records, the
  pairing grain being unchanged, and a missing or `None` `fold_id` raising rather than
  falling back.
* **232 pass across the five affected test files** — `test_training_item_coverage.py`,
  `test_cv_cohort_parity.py`, `test_served_cohort_weighting.py`,
  `test_drift_retrain_guard.py` and `test_forecaster.py` — re-run after the harness
  cluster-key fixes landed, not carried over from the earlier run.

## Provenance

Every measurement harness in this entry was **scratchpad-only**. Nothing was written to
`price-archive/`, to `models/saved_models/`, or to the database; the seed sweep and the
full ≥$1 run used isolated `model_dir`s via `FORECAST_MODEL_DIR`. The findings on item type
and weapon read `price-archive/ops/forecast_outcomes.parquet` and wrote nothing. The only
changes to production code are the price-floor knob (defaulted off) and the three harness
cluster-key fixes. All of it is uncommitted in the working tree at the time of writing.

The 8-seed sweep, the full ≥$1 run and the paired harness are the four numbers that carry
this entry's conclusions and each was run once at its stated configuration.

## Still open

* **Whether the floor moves production DA.** All the evidence here is offline. The 30d
  +3.50pp is a held-out-item CV number on a 728-item universe, not a production figure, and
  offline gate numbers from `walkforward_backtest.py` are not comparable to production DA
  either.
* **Per-item row sampling**, which is what separates the shippable arm (+3.50pp) from the
  better one (+5.72pp).
* **Re-deriving the CSFloat, ByMykel and breadth CIs** under fold clustering. Until then
  the sub-1pp results in those entries are unsupported, not refuted.
* **The always-down constant** remains unbeaten and remains outside every baseline this
  entry reports.
* **The sub-$1 serving surface** if the floor is enabled: 8,691 items forecast, ≥$1 served,
  and no code path enforcing that the untrained tier stays unshown.

## Next step: the floor should probably be $0.50, not $1.00

Raised after the fact, by the question "if I only look at items over a dollar, how do I
catch a case under a dollar rising over it?" That is a real gap, and checking it showed the
$1.00 value is not where the evidence puts the boundary.

**$1.00 is inherited, not derived.** `backend/AGENTS.md` already states that
`MIN_SERVED_PRICE_USD = 1.0` "is a convention, not a derivation" — it was pinned to
`HEADLINE_MIN_TIER` so the shown population matches the headline population. This entry's
`TRAIN_MIN_MEDIAN_PRICE=1.0` inherited that number for cohort-matching, not because $1 is
where sub-dollar data stops being usable.

**Measured where it actually stops being usable** (scored `forecast_outcomes`,
`model_version='lgbm-v3'`, bucketed by `base_price`; "stale" = `actual_price == base_price`):

| band | n | stale % | 1¢ as % of price | DA |
|---|---|---|---|---|
| <$0.05 | 15,057 | **71.9** | 33.3 | 45.8 |
| $0.05–0.10 | 3,752 | 16.2 | 12.8 | 40.6 |
| $0.10–0.25 | 4,997 | 7.9 | 6.2 | 45.2 |
| $0.25–0.50 | 3,794 | 4.6 | 2.8 | 47.4 |
| $0.50–1.00 | 3,712 | **2.6** | 1.4 | 43.8 |
| $1–2 | 2,190 | 1.4 | 0.8 | 45.9 |
| ≥$5 | 2,630 | 0.2 | 0.1 | 47.0 |

The degradation cliff is at **~$0.05**, not $1.00. The $0.50–1.00 band is 2.6% stale
against $1–2's 1.4% — the same order of magnitude. A $1 floor is discarding clean data.

**And the discarded band is disproportionately cases.** Median case price is **$0.92**, so
the type straddles the cutoff: of 158 cases in the pool, 15 are <$0.25, 33 are $0.25–0.50,
**44 are $0.50–1.00**, 11 are $1–2 and 55 are ≥$2. A $1 floor trains on 66 of 158 cases; a
$0.50 floor trains on 110.

| floor | items | item-days | cases |
|---|---|---|---|
| $1.00 | 926 | 993,464 | 66 |
| **$0.50** | **1,395** | **1,521,732** | **110** |
| $0.25 | 1,924 | 2,118,761 | 146 |
| $0.10 | 2,660 | 2,938,587 | 158 |

$0.50 costs ~1.5× the rows (projected ~15 min against the measured 9 min at $1.00, not
measured). $0.25 doubles again for 36 more cases and readmits a 4.6%-stale band.

**The proposed next step, in order:**

1. Re-run the paired arm from § *The paired rerun* at a $0.50 floor, unchanged otherwise —
   same 150 held-out items, same folds, same `cluster_key="fold_id"`. The question is
   whether the 30d +3.50pp survives a 1.5× wider universe. ~15 min.
2. Only if it does, set `TRAIN_MIN_MEDIAN_PRICE=0.50` rather than `1.00`.

**Two things that must not be conflated with this, and are the reason it is filed as a next
step rather than a fix.**

* **The training floor and the serving floor are different filters.** Lowering
  `TRAIN_MIN_MEDIAN_PRICE` changes nothing a user can see: the $0.80 case's forecast is
  already written and already hidden by `MIN_SERVED_PRICE_USD`. Actually surfacing a
  sub-$1 riser is a three-part change — training floor, serving floor, and either moving
  `HEADLINE_MIN_TIER` with them or deliberately unpinning the two.
  `backend/tests/test_serving_policy.py` fails if they diverge, by design.
* **Coverage is not detection.** DA is 40.6–47.4% in *every* price band above, and the
  model loses to a constant always-down call on 14 of 16 weapons (§ *Item type and specific
  weapon are dead axes*). There is no band where the model works and the floor is hiding
  it. Lowering the floor buys the ability to score these items, not the ability to call
  them.

**A reframe worth recording separately:** "will this cross $1" is a **threshold-crossing**
question, not a direction question. The model already emits a q50 level with a
conformal band, so items whose predicted band crosses $1 could be screened with no model
change at all — and unlike direction, the band carries information even where the point
forecast does not. That is a serving/screening feature, unmeasured, and it still needs the
serving floor lowered to be visible.

## Related

* `2026-08-06-served-cohort-weighting-refuted.md` — closed with "whether *more* ≥$1
  training items would help is untested". This is that test, and it separates the two
  cleanly: reweighting the loss did nothing (~0.1pp pooled), changing the universe did
  something at 30d.
* `2026-08-06-breadth-beats-depth-item-age-does-not.md` — the source of the fold schedule,
  the strict metric and the 150 held-out items this entry's paired harness inherits; also
  the entry whose CIs are now known to be too narrow.
* `2026-08-06-data-acquisition-ranking.md` — amended in this pass (§ below).
* `2026-08-03-accuracy-is-clustered-by-forecast-date.md` — the phenomenon the weapon-level
  χ² result turns out to be a third reading of.
* `2026-08-05-cv-cohort-parity-design.md` — where `classifier_accuracy_ge1` came from,
  without which none of the ≥$1 figures here could be stated.

## Docs touched

* `docs/architecture/model.md` — § Training row budget gains the `min_median_price` row, the
  seed-variance warning, and the ≥$1-cohort sizing; the "trains on 1.8% of the pool and
  serves 5,542" bullet corrected to **8,691 items forecast** on the latest date.
* `docs/architecture/model-optimization.md` — the coverage-cost table gains the
  floor-plus-1.2M row (926 items, 538s), and the "~4.5× cost" argument is qualified as a
  **weekly** cost.
* `backend/AGENTS.md` — the "Training subsamples the pool" gotcha now carries the
  99-items-vs-8,691-served figure, the hardcoded `seed=42` and its sd 1.5–3.1pp, the
  `TRAIN_MIN_MEDIAN_PRICE` knob, and the `max_rows`-binds-at-~1M trap. (Edited after the
  changelog pass, which was scoped to `docs/`.)
* `docs/changelog/2026-08-06-data-acquisition-ranking.md` — two factual corrections, marked
  inline as 2026-08-07 amendments and otherwise unchanged:
  1. The binding training budget is **`TRAIN_FEATURE_ROWS`, 100K**, not 700K. 700K is
     `TRAIN_HORIZON_MAX_ROWS`, the non-binding per-horizon cap.
  2. Production sits at **99 items**, not ~521 items / ~1,343 rows per item, so the
     interpolation placing it "between the 150- and 350-item arms" is wrong — it is
     **below** the 150-item arm. The note also records that the breadth CIs that argument
     rests on were date-clustered.
