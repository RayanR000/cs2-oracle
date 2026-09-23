# The $1 training floor's look-ahead is real and worth −0.004pp; the +3.50pp does not reproduce

**Date:** 2026-08-08
**Plan:** `docs/plans/2026-08-08-per-fold-price-filter.md` — **stage 1 only**
**Scope:** research instrument. Production is untouched; `build_training_data` does not call
anything added here.
**Branch:** `research/per-fold-price-filter`, uncommitted at time of writing —
`backend/models/forecaster.py` (+35), `backend/tests/test_training_item_coverage.py` (+74),
new `backend/scripts/ab_test_train_universe.py` (693 lines)
**Suite:** 1,576 pass, 0 failures

The plan asked one question: is `TRAIN_MIN_MEDIAN_PRICE`'s **+3.50pp [+1.56, +5.98] at 30d**
(`2026-08-07-training-item-universe.md`, `ge1_full` vs `prod_a`) real, or is it survivorship?

The answer is neither. **The survivorship hypothesis is refuted** — the look-ahead is large
but carries no accuracy. **And the +3.50pp still does not reproduce**: on the current
instrument the same contrast reads **+1.642pp [−0.809, +4.505], null**, for reasons that have
nothing to do with the filter. Step 7's accuracy justification is withdrawn; its
measurability justification stands untouched.

## The look-ahead is real: 876 items used where 319 were knowable

`_filter_by_median_price` takes the median over the whole 2013–2026 frame, once, before any
split exists. Measured on this archive at h=30:

| fold | selection cutoff | knowable per-fold | used by `full_sample` |
|---|---|---:|---:|
| earliest | 2021-12-18 | 319 | 876 |
| last | 2026-01-26 | 817 | 876 |

So the earliest fold trains on 594 items it could not have named. The leak also runs the
other way: **141 items enter some per-fold universe but never the full-sample one** — items
that were expensive and collapsed to pennies, rejected by the full-sample median on prices
postdating every fold that wanted them.

`_fold_median_price_items(price_df, min_median_price, cutoff)` is the look-ahead-free
counterpart, a pure static helper beside `_filter_by_median_price`. Three properties are
load-bearing and each has a test:

- **`cutoff` must be `val_start − embargo_days(horizon)`, never `val_start`.** The selection
  statistic is itself a function of prices, so computed to the boundary it reads the embargo
  window — the same leak in a smaller form (invariant 3).
- **`date < cutoff`, strictly.**
- **An empty pre-cutoff window returns an empty set, not everything.** A "keep all" fallback
  would silently restore full-sample behaviour on exactly the early folds where the leak is
  largest.

## The leak carries no accuracy

Removing it, at equal universe size, moves DA by **−0.004pp [−0.664, +0.851] at 30d** and
**−0.362pp [−1.135, +0.318] at 14d**. Both null, both intervals narrow enough to be
informative rather than merely inconclusive — the matched placebo puts this contrast's noise
floor at ±0.9pp, not ±4pp.

At 30d the three ≥$1 arms land on **55.25% to two decimals** — leaked, unleaked, and
downsampled-to-match. Whatever the floor is doing, selecting it on future prices is not part
of it.

## The measurement

Instrument: DA(strict, ≥$1) with flat-actual rows dropped, 21-day validation window, 60-day
stride, **150 held-out ≥$1 items scored on identical rows in every arm**, `paired_arm_contrasts`
clustered on `fold_id`. Frame: 1,622 items / 4,151,088 price rows / 32 features after shelving
+ the `price_technicals` allowlist + a 0.95 correlation prune. **Train side filtered only** —
`val_set` is never touched, because the metric is already a ≥$1 cohort and filtering val would
move the evaluation population alongside the treatment. That confound is what halved the
original unpaired 58.60-vs-50.88 headline. The market factor is computed once on the pooled
frame, so it is identically wrong in every arm and differences out; that is a pairing device,
not a claim that the pooled factor is correct.

`full_sample`'s 726 items/fold is the 876-item universe minus the 150 held out.

**h=30d, 25 folds, n = 46,599 scored:**

| arm | items/fold | rows/fold | DA |
|---|---:|---:|---:|
| `prod_pool` (99 items, production's measured 17 ≥$1 + 82 sub-$1) | 99.0 | 162,736 | 53.61% |
| `prod_pool_b` (disjoint second draw — **placebo**) | 99.0 | 155,981 | 52.90% |
| `full_sample` (= `ge1_full`, the config step 7 would ship) | 726.0 | 698,640 | **55.25%** |
| `per_fold` | 412.6 | 658,803 | **55.25%** |
| `full_sample_matched` | 412.6 | 421,720 | **55.25%** |
| `full_sample_matched_b` (**placebo**) | 412.6 | 424,921 | 55.04% |

Placebos, read first: `prod_pool_b` −0.708pp [−4.170, +2.192] **null**;
`full_sample_matched_b` −0.208pp [−0.882, +0.442] **null**.

vs `prod_pool`: `full_sample` **+1.642pp [−0.809, +4.505] null**; `per_fold` +1.635pp
[−0.600, +4.089] null; `full_sample_matched` +1.639pp [−0.821, +4.416] null.

The leak alone, `per_fold` vs `full_sample_matched`: **−0.004pp [−0.664, +0.851] null.**

**h=14d, 26 folds, n = 47,467 scored:** `prod_pool` 52.74%, `prod_pool_b` 52.15%,
`full_sample` 53.92%, `per_fold` 53.71%, `full_sample_matched` 54.07%,
`full_sample_matched_b` 53.92%.

Placebos: `prod_pool_b` −0.598pp [−4.066, +1.809] null; `full_sample_matched_b` −0.154pp
[−0.776, +0.531] null. vs `prod_pool`: `full_sample` +1.180pp [−1.186, +3.780] null;
`per_fold` +0.963pp [−1.163, +3.094] null. The leak alone: −0.362pp [−1.135, +0.318] null.

### The constant call on the same rows

Invariant 4: no DA above is quotable alone. Computed on the same scored eval rows.

| h | realised down-rate | best constant call | per-fold down-rate min / median / max |
|---:|---:|---:|---|
| 14 | 49.75% | 50.25% | 7.5% / 48.0% / 76.8% |
| 30 | 52.80% | 52.80% | 22.7% / 47.3% / 86.6% |

At h=30 the ≥$1 arms clear the constant call by **2.45pp** and `prod_pool` by **0.81pp**.
This is an offline harness on 150 held-out ≥$1 items — **not comparable to production DA and
not a gate number either.** The per-fold spread (22.7% to 86.6% at h=30) is the date effect
this project has recorded repeatedly; it is the reason the constant call is printed beside
every figure here.

## The +3.50pp does not reproduce, and the filter is not why

`full_sample` here *is* `ge1_full` and `prod_pool` *is* `prod_a`. The same contrast that
stored +3.50pp [+1.56, +5.98] reads +1.642pp [−0.809, +4.505]. Three things changed
underneath it since 2026-08-07, and they are confounded with one another:

1. **The train-side embargo the original lacked** (below). The unpurged split was later
   measured to inflate DA by **+10.15pp at 30d** (`2026-08-08-embargo-discontinuity-measured.md`),
   and arms differing in item set are asymmetrically exposed — a wider arm sees more distinct
   items inside the overlapping band.
2. **Step 6's frozen-price-run voiding** — **48,338 30d targets voided on this frame**
   (`2026-08-08-frozen-price-runs-dropped-from-labels.md`).
3. **The universe rules** — the bid source excluded, the phase-collapsed names dropped.

This run cannot separate them, and it was not designed to: the plan's own preconditions 1 and
2 say a contrast against the stored number is invalid regardless of the filter. **The stored
+3.50pp is not recoverable in isolation and should not be cited again.**

## The instrument that produced +3.50pp carried three defects

`ab_universe_paired.py` existed only in a session scratchpad and was never in the repo. It is
promoted here as `backend/scripts/ab_test_train_universe.py`, with all three fixed:

1. **`WHERE source = 'aggregator_sync'`, not NULL-safe.** This is the exact bug that emptied
   eight harnesses on the archive migration (`2026-08-08-migrated-archive-emptied-eight-harnesses.md`).
   **Re-run today as written, it returns an empty universe.** The promoted harness reads
   through `db/archive.py::prices_relation` (invariant 1) and applies
   `archive_universe_sql_filter` (invariant 2), with the universe predicate hashed into the
   frame-cache key because that key fingerprints `forecaster.py`, which cannot see
   `item_parser.py`.
2. **Neither universe rule** — no bid-source exclusion, no phase-collapsed names.
3. **No train-side embargo at all.** `ab_test_training_breadth.py` was purged on 2026-08-07;
   this one never was, so **the stored +3.50pp is an un-purged number.** The promoted harness
   purges the train side with production's own `_purge_overlapping_train_rows`.

The full-sample `COUNT(DISTINCT day) >= 180` clause is survivorship on the same axis as the
median and moves inside the cutoff with it in `per_fold`. Fixing only the median would have
produced a confident null.

## Two design points where the plan was wrong

- **The plan named `ab_test_training_breadth.py` as the instrument.** That harness's arms are
  narrow/mid/wide breadth; it did not produce the +3.50pp. The number came from the
  scratchpad harness above.
- **The plan's third arm was "`per_fold`, downsampled to `full_sample`'s item count".** The
  inequality runs the other way — 319 knowable against 876 used — so it is the **full-sample**
  arm that gets thinned. The arms are `full_sample_matched` and `_b`, `full_sample`
  downsampled per fold to `per_fold`'s realised item count at two seeds.
- **The plan's Task 1.4 requires reading a placebo first, but its arm table contains none.**
  Two were added: `prod_pool_b`, a disjoint second 99-item draw, and `full_sample_matched_b`,
  a second matching seed. They measure two different noise floors and both were read before
  any treatment number.

## What this leaves of step 7

**The accuracy justification is gone.** +1.642pp with an interval covering zero is not a
result to ship on, and `prod_pool_b` re-measures the item-draw noise floor at **±3–4pp** on
this instrument, which is why a +1.6pp effect cannot resolve here at all.

**The measurability justification is untouched and was always the stronger one.** The floor
removes item-draw variance *by construction* — the 99-item subsample's seed alone moves
`acc_ge1` by sd **1.5–3.1pp** — and that is a determinism argument, not an accuracy claim. It
does not depend on any number in this entry.

**Stage 2 is not needed for the correctness of this measurement.** The leak is not
load-bearing, so an anchored per-fold filter buys no accuracy. If the knob ever ships, the
anchored form is still the better one for the reasons the plan gives (B1 and B2 both
disappear), but that is a cleanliness argument.

## Deliberately not done

- **`build_training_data` does not call `_fold_median_price_items`.** A fold-varying universe
  would move the market factor and the row budget alongside the treatment — the plan's B1 and
  B2. The docstring says research-only and names the plan.
- **Stage 2 was not started.** No anchored filter, no knob change, no retrain.
- **The market factor was not recomputed per arm or per fold.** Real, out of scope, and it
  affects every harness rather than this one.
- **`ab_test_training_breadth.py` was not modified.** It was not the instrument, and its own
  full-sample clauses are a separate defect on a separate result.
- **No attempt to decompose the +3.50pp → +1.642pp move** into embargo / step 6 / universe.
  It would need three more paired runs against label sets that no longer exist, and the plan
  pre-declared the contrast invalid.
- **h=3 and h=7 were not run.** The original effect was h=30-only; 14d and 30d are where the
  claim lives.
- **The harness does not emit the constant call.** The baseline table above was computed from
  the stored records after the fact. Every other `ab_test_*` harness has the same gap; fixing
  it store-wide is not this change.

## Verification

- **1,576 pass, 0 failures** on the full `tests/` suite.
- New `TestFoldMedianPriceItems` in `backend/tests/test_training_item_coverage.py`, 5 cases:
  a late riser is excluded; an early clearer that later collapsed is included; **the two
  filters demonstrably disagree on the same frame** (without this the fixture would not be
  testing the leak); an empty pre-cutoff window yields nothing; `date < cutoff` is strict.
- The harness asserts a non-empty universe and fails loudly if the frame is empty, naming the
  archive-migration failure mode rather than reporting a data statement.
- Both placebos read null before any treatment contrast was looked at.

## Still open

- **The +3.50pp cannot be re-derived in isolation** and is retired rather than corrected.
  `2026-08-07-training-item-universe.md` has not been edited; this entry is the correction.
- **Whether the floor is worth anything at all is unresolved, not refuted.** +1.6pp sits
  inside a ±3–4pp noise floor at both horizons. Resolving it needs a design that removes the
  item draw, not another run of this one.
- **The ±3–4pp `prod_pool_b` floor is the same order as `2026-08-07-training-item-universe.md`'s
  2.21–3.69pp MDE**, reproduced on a third instrument. Nothing item-level below ~4pp is
  measurable by this family of harnesses.
- **R19 (multiplicity deflation) still gates step 7** and is unaffected by this run.
- **The work is uncommitted** on `research/per-fold-price-filter`.

## Related

- `docs/changelog/2026-08-07-training-item-universe.md` — the +3.50pp this retires
- `docs/changelog/2026-08-08-embargo-discontinuity-measured.md` — the +10.15pp at 30d that
  sizes defect 3
- `docs/changelog/2026-08-08-migrated-archive-emptied-eight-harnesses.md` — the NULL-safety
  bug the scratchpad harness carried
- `docs/changelog/2026-08-08-frozen-price-runs-dropped-from-labels.md` — step 6, and its
  "Relationship to step 7" section
- `docs/changelog/2026-08-08-phase-collapsed-names-dropped.md`,
  `docs/changelog/2026-08-07-bid-source-excluded-from-voting.md` — the two universe rules
- `docs/changelog/2026-08-07-paired-mde-fold-clustering.md` — the fold-clustered bootstrap
  behind every interval here

## Docs touched

This entry, and `docs/research/2026-08-07-next-steps.md` step 7 — marked re-derived, accuracy
claim withdrawn, still **NOT STARTED** as a shipping decision. Nothing under
`docs/architecture/` moved.
