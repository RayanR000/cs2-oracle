# Market-relative labels are refuted — with the market factor removed, the classifier lands below a constant call

**Date:** 2026-08-06
**Spec:** `docs/superpowers/specs/2026-08-06-market-relative-labels-design.md`
(pre-registered in `bc550e0`, before any arm ran)
**Instrument:** `docs/changelog/2026-08-06-market-relative-labels-instrument.md`
**Commits:** `444268f` (`market_factor.py`, the driver, the tests), `9a5564b` (the
instrument entry); the `forecaster.py` / `forecast_prices.py` edits are still uncommitted
— see the instrument entry's "Commit provenance".

**Pre-registered rule 1 fired.** `relative_accuracy_ge1` — the model's ability to call the
*idiosyncratic* direction once the market factor is subtracted — came in at **36.7 / 32.7 /
34.6 / 39.0** at 3/7/14/30d, against a majority-class baseline of **38.8 / 42.7 / 46.4 /
51.7** on the same rows. The rule required > 51% at at least one horizon to continue. Not
only is every horizon far below that bar, the model is **below a constant call at all
four**. The hypothesis is refuted, `DEFAULT_MARKET_RELATIVE_LABELS` stays `False`, and
production is unchanged.

These are CV gate numbers from the retrain's own expanding-window folds. They are not
production DA and are not comparable to it.

## Only one arm was run, deliberately

The design called for two paired cold retrains. **Only the treatment arm was run.**

Rule 1 makes the pairing moot: it is a single-arm test by construction — it asks whether
the treatment model can call the residual direction at all, and the control arm produces no
`relative_accuracy_ge1` to compare against, because the control has no residual labels.
Once rule 1 fires, the spec says explicitly: "Stop. Do not interpret stage 2." Running the
control would have spent ~3 minutes producing a paired `classifier_accuracy_ge1` table that
the pre-registration forbids reading. That is the whole point of a pre-registered kill
switch, so it was honoured rather than routed around.

The consequence is that **every stage-2 number below is unpaired** and is labelled as such
where it appears.

## Run provenance

`--train-only` with `FORECAST_MODEL_DIR` pointed at a scratch directory; the deployed
artifact in `backend/models/saved_models/` was never written.

* `TRAIN_MARKET_RELATIVE_LABELS=1`, 2026-08-06 17:58:41 → 18:01:49 local, ~3 min wall clock.
* **99 of 5,378 items**, **115,763 of 5,832,742 rows** (budget `TRAIN_FEATURE_ROWS=100,000`),
  1460-day window. Same subsample shape as the served-cohort arms.
* **33 features** after the `FEATURE_GROUP_ALLOWLIST = ["price_technicals"]` cut, from 118
  engineered.
* Market index: **1,455 dates, 1,451 valid (99.7%)**. The 4 invalid days are the known
  archive day gaps (`aggregator-archive-day-gaps`), not a construction failure.

**Market-factor coverage was not the limiting factor.** The spec voids a run whose coverage
falls below 95% on any fold; coverage was **100.0% of validation rows at every horizon**.
Row-level coverage across the whole frame, before the CV split, was 99.3 / 99.0 / 98.4 /
97.1% at 3/7/14/30d — the shortfall is the tail of rows whose `d+h` runs past the end of
the archive, and those rows are not in any validation fold. The NaN-`m` fallback path
(`e = r`) therefore never diluted the read.

## Stage 1: the kill

`relative_accuracy_ge1` is 3-class agreement between the classifier's call on the demeaned
label and the demeaned actual, over ≥$1 validation rows. `relative_majority_ge1` is the
best constant call on those same actuals — the chance baseline.

| horizon | folds | relative acc | majority baseline | edge |
|---|---|---|---|---|
| 3d | 9 | 36.7% | 38.8% | **−2.1** |
| 7d | 8 | 32.7% | 42.7% | **−10.0** |
| 14d | 8 | 34.6% | 46.4% | **−11.8** |
| 30d | 8 | 39.0% | 51.7% | **−12.7** |

Per fold, `relative_acc` / `majority`:

* 3d: 25.9/38.6, 35.1/38.9, 35.6/41.1, 37.7/43.7, 36.4/36.8, 36.5/40.1, 41.9/38.1,
  40.5/36.2, 40.8/35.5
* 7d: 28.9/35.7, 30.0/41.1, 32.9/45.6, 29.3/55.7, 29.2/37.9, 28.4/46.4, 40.6/40.7,
  42.3/38.1
* 14d: 37.5/37.1, 33.8/41.6, 31.0/53.0, 33.3/60.8, 24.4/40.0, 32.0/53.7, 42.9/43.2,
  42.2/41.6
* 30d: 56.6/56.4, 47.3/47.3, 30.3/56.3, 19.7/69.8, 23.6/37.5, 37.8/45.1, 52.8/55.3,
  43.7/45.6

Three folds beat their baseline at 3d and one at 30d; the rest lose, several by 20pp or
more. There is no horizon and no sustained fold run where removing the market factor leaves
something the classifier can call.

## A metric defect fixed before any data was seen

The spec defines `relative_accuracy_ge1` as "`sign(ê)` vs `sign(e)` on ≥$1 validation
rows". The implementation computes **3-class agreement** (down/flat/up), not sign
agreement. Those are different metrics with different chance baselines: sign agreement has
a ~50% floor, which is what the `<= 51%` threshold in rule 1 assumed; 3-class agreement's
floor is the **majority-class share**, which the matched flat band puts in the 35–52% range
here. Read against 50%, the metric is uninterpretable.

`relative_majority_ge1` was added to `_cv_evaluate_horizon` to report that baseline
alongside. **This was done before the arm ran and before any result existed**, and it moved
the bar in the harder direction — 51% flat was in several folds a *lower* bar than the
majority share. It does not rescue the reading either way, since the measured values are 12
to 19 points under the threshold as written.

This is a spec/implementation mismatch that the pre-registration did not catch. Recording
it plainly matters more than the fact that it happened to be harmless: the same class of
error, discovered after a positive result, would have been unfixable without contaminating
the read.

## Stage 2, reported but not a paired measurement

Under the pre-registration this should not be interpreted at all. It is recorded because it
corroborates the kill and because leaving it out would look like selective reporting.

Reconstructed absolute `classifier_accuracy_ge1` from `band(m̂ + ê)`, scored on the same
fixed ±0.5% actual-class yardstick:

| horizon | treatment (this run) | deployed `meta.json` |
|---|---|---|
| 3d | 48.3 | 49.9 |
| 7d | 45.7 | 49.0 |
| 14d | 47.8 | 51.1 |
| 30d | 43.5 | 53.4 |

**This is not a valid paired comparison and must not be cited as one.** No control arm was
run, and the right-hand column is a different artifact on a different tree: `meta.json`
holds `model_artifact_version` 3 trained 2026-08-06 05:17 UTC on **36** features, before the
in-flight scale-free-features change (`2026-08-06-scale-free-features-and-fabricated-labels.md`),
while this run had **33**. The gap conflates the relabelling with at least one unrelated
feature-set change. It corroborates the kill; it does not measure it.

Pooled `classifier_accuracy` collapsed further, to **55.2 / 30.4 / 31.4 / 31.5%** at
3/7/14/30d against the deployed artifact's 67.3 / 67.2 / 67.8 / 68.6. That collapse is
mechanical, and the spec named
it in advance as a limitation of the reconstruction: `ê` is the residual arm's signed band
edge, so `m̂ + ê` almost never lands inside the fixed ±0.5% flat band. Nearly nothing gets
called flat, while ~40% of actuals *are* flat. The pooled number is measuring the coarseness
of the reconstruction, not the labels.

## An assumption the spec got backwards

The spec justified the matched flat band by asserting that "residuals have smaller
dispersion than raw returns", so reusing the fixed ±0.5% band would **inflate** the flat
class. The opposite happened. Matched bands, per horizon:

| horizon | control flat share | matched residual band |
|---|---|---|
| 3d | 43.7% | ±1.668% |
| 7d | 41.7% | ±2.572% |
| 14d | 40.4% | ±3.609% |
| 30d | 39.6% | ±4.746% |

To hold the flat share constant, the residual band had to open to **3.3–9.5× the fixed
±0.5%**. Demeaning *spread* the label distribution.

The cause is the atom at zero. A large share of raw forward returns are **exactly** zero —
`2026-08-06-volume-ab-and-harness-defects.md` measured **41.0% of 3d forward returns as
exactly zero** across the frame (64.0% for sub-$0.10 items, 0.07% for >$5): penny items and
stale prices that do not tick. Subtracting a nonzero `m` moves every one of those off zero.
Demeaning destroys the point mass that the narrow fixed band was mostly catching, and the
band has to widen to recover the same flat share.

The matched-band mechanism still did its job — class balance held across the arms, which is
what it was for. But the flat class in the treatment arm now contains **different rows**
than the control's: the control's flat class is dominated by non-ticking items, the
treatment's by items whose move happened to track the market. That is a real property of
this data the design did not anticipate, and it is worth knowing before anyone re-derives
the same band.

## What this actually says

Once the market factor is removed, the price-technicals feature set (36 columns in
production, 33 here) predicts **nothing** about which item moves which way. Everything the
directional classifier was doing was riding the common factor.

That is a mechanism for `2026-08-03-accuracy-is-clustered-by-forecast-date.md`, which found
accuracy tracking the forecast date rather than the item — 33.4% on a rising date vs 63.7%
on a falling one at 7d — and for the constant always-down call beating the model on every
stored forecast date. If the only learnable content is the market's direction, a model that
guesses the market's direction and a constant tilt are the same strategy, and the constant
one has less variance.

It also offers a retroactive explanation for the null streak. A dozen `scripts/ab_test_*.py`
experiments measured variations on a signal that is not present at the item level. That does
not retroactively validate any of them — several have their own defects
(`2026-08-06-volume-ab-and-harness-defects.md`) — but it stops the pattern of nulls looking
like a series of unlucky feature choices.

## Caveats that bound the conclusion

1. **A mover-weighting artifact inflates the negative edge.** The classifier trains with
   `DIRECTION_MOVER_WEIGHT_MAP` = 3× on movers, so it systematically under-predicts the flat
   class — while the majority baseline **is** flat at every horizon here. Some of the −2 to
   −13pp edge is that artifact, not the model being worse than a coin. There is no
   control-arm majority number to net it out, because the control was not run. It does not
   rescue the read: 33–39% absolute is nowhere near the ≥51% rule 1 required, and a class
   weight cannot manufacture a 12pp gap.
2. **Scope.** 99 items, 115,763 rows, the ≥$1 cohort within that subsample, on this feature
   set at this training budget. This establishes "no idiosyncratic directional signal in
   *these* features at *this* budget". It does not establish that no feature set could find
   one — supply depth, listing counts and float/wear data are all outside the frame
   (`2026-08-06-free-bulk-supply-depth-feeds-exist.md`).
3. **Single arm.** Stage 2 is unpaired, as stated above.

## Decision

* `DEFAULT_MARKET_RELATIVE_LABELS` stays **`False`**. Production is unchanged; `predict()`
  was never touched.
* The code is **kept, defaulted off**, on the same reasoning that retained the served-cohort
  instrument (`2026-08-06-served-cohort-weighting-refuted.md`) and
  `scripts/ab_test_direction_labels.py`: it is the instrument that produced this
  measurement, and the next person to doubt the finding should re-run it rather than rebuild
  it.

## What was deliberately not done

* **The control arm was not run.** See above — rule 1 makes it moot, and running it would
  produce numbers the pre-registration says not to interpret.
* **No residual quantile model.** It was already out of scope until stage 1 reported; stage
  1 reported a kill, so it stays out. A finer `ê` would sharpen a reconstruction of a signal
  that is not there.
* **No `m̂` estimator tuning.** Rule 1 forbids it by name: "do not tune the `m̂` estimator to
  rescue it." The two diagnostic estimators (`trailing_k_median`, `past_h_momentum`) exist
  precisely so the pre-registered estimator cannot be blamed for the negative — and they
  were never eligible to rescue it.
* **The spec was annotated, not rewritten.** Its Status block now records the refutation and
  the two things the document got wrong, with the original text left in place below.

## Verification

No new tests: the instrument's tests (`backend/tests/test_market_factor.py`, 21;
`backend/tests/test_market_relative_labels.py`, 30) were written and run before the arm, and
the full `backend/tests/` suite stood at **948 passed** at that point. This entry records a
run, not a code change.

The load-bearing check for a *negative* result is the leakage suite, and it cuts the other
way from usual: leakage would have produced a fake **positive**. Its absence is not what
makes this null credible. What bounds this null is the coverage reading (100.0% on every
validation fold, so the fallback path never fired) and the pairing-independent nature of
stage 1 — `relative_accuracy_ge1` is scored within the treatment arm against its own
baseline, so no cross-arm comparison can have gone wrong.

## Still open

**The structural finding is larger than this experiment.** With 8–9 effectively independent
CV folds and a noise floor around 1pp (`ab-harness-noise-floor`), this project cannot
resolve a genuine directional-accuracy improvement of the size any realistic feature change
would produce. The served-cohort run needed a ±1pp band to declare "no effect" and got four
diffs inside it with mixed signs; the +2pp adoption bar here was set from that. Every
directional-accuracy experiment in this repo is underpowered by construction, and the fix is
a power calculation before the next one, not another arm.

**Both previously-unsourced figures are now closed from the run log.** The treatment arm's
3d pooled `classifier_accuracy` is **55.2%**, folded into the table above. The feature count
is **33**, from `Feature allowlist ['price_technicals']: 118 -> 33 features` in the run log
and `len(feature_cols) == 33` in the arm's `meta.json`; the scale-free entry's 36 → 32 is a
different run under a different configuration and does not describe this one.

## Related

* `2026-08-06-market-relative-labels-instrument.md` — what was built, the two design
  decisions the diff does not explain, and the pre-registered rule as it stood before the
  run.
* `docs/superpowers/specs/2026-08-06-market-relative-labels-design.md` — the design and the
  rule, now carrying the refutation in its Status block.
* `2026-08-03-accuracy-is-clustered-by-forecast-date.md` — the finding this experiment acted
  on, and which this result now supplies a mechanism for.
* `2026-08-06-served-cohort-weighting-refuted.md` — the paired cold-retrain methodology,
  the "Still open" item this addressed, and the precedent for keeping a refuted instrument.
* `2026-08-06-volume-ab-and-harness-defects.md` — the source of the exactly-zero return
  measurement, and why both arms were designed to run through the live retrain path.
