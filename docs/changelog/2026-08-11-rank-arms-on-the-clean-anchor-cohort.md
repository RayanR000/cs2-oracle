# CV reports rank IC on the clean-anchor cohort, 2026-08-11

The metric every arm in this project was ranked on is contaminated, and the size of the
contamination is larger than every effect being chased. This makes the uncontaminated
number available in the vehicle that reads it.

## Why

`prepare_targets` divides the label by the raw quote at *d*, which is also what `return_1d`
and every level feature are built from. `predict` divides by the local median. The 2026-08-11
replay crossed the two axes and attributed **+0.1398 of the +0.1464** CV↔serving rank IC gap
to that denominator alone, confirmed in CI on a fresh artifact at four non-overlapping anchors
in **16 cells of 16** (`2026-08-11-the-gap-is-the-anchor-denominator.md`,
`2026-08-11-clean-anchor-confirmed-in-ci.md`).

Moving the label to the served denominator was the obvious fix and it was **measured and
refuted**: `LABEL_SMOOTHED_ANCHOR=1` swings pooled served rank IC to +0.17–0.31 at 4/4
anchors, and all of it is `p[d]/S[d]` re-entering as a factor readable at the anchor. On the
tied cohort it is −0.033 / −0.017 / −0.020 / +0.008
(`2026-08-11-smoothed-anchor-label-measured.md`).

Both bases carry the wedge, with opposite signs. The **tied** cohort — rows where `p[d]`
already equals `S[d]` — is the only place where neither operates. That is the cohort an arm
has to be ranked on, and until now nothing in CV could report it.

## What shipped

- `ItemForecaster._anchor_is_tied` and the `anchor_is_tied` column, added by
  `prepare_targets` and travelling with the frame. Same `isclose(rtol=0, atol=1e-9)` as
  `replay_serving._tied_mask`, and a test holds the two against each other on one panel —
  every published tied/deviating number came from that function, so two definitions printing
  one word is the failure mode worth a test.
- **Arm-invariant by construction.** It reads the price series and never
  `label_smoothed_anchor_enabled()` or `outlier_gated_anchor_enabled()`; a mask that followed
  either would score the control and the arm on different populations.
- Per fold: `rank_ic_tied`, `naive_rank_ic_tied`, `n_tied`, `rank_ic_tied_dates`,
  `rank_ic_dates`. In `cv_results` / `meta.json`: `mean_rank_ic_tied`,
  `mean_naive_rank_ic_tied`, **`rank_ic_edge_vs_naive_tied`**, plus `tied_rows` / `tied_dates`.
- `_summarise_rank_ic` extracted so the block is testable without a training run.
- The pooled keys are **unchanged**. They are the series every historical `meta.json` holds
  and the trust warning reads; redefining them would turn the stored trend into a comparison
  of two quantities. The tied pair sits beside them, and the log marks which to read.

`rank_ic_tied` is `None`, never the pooled value, when the mask is absent — every `ab_test_*`
frame predates the column. Falling back would publish the contaminated number under the clean
key, which is worse than no number. Same rule `classifier_accuracy_ge1` follows.

`rank_ic_tied_dates` is not decoration: the `min_rows=20` bar bites harder on a third of the
cross-section, and one 2026-08-11 anchor had 26 tied items of 669. Without the count the two
columns can describe different calendars while looking paired.

## A gap this found

`model-diagnostics.yml` filters the run log into the step summary with one `grep -E` whose
alternative was `rank IC`. The headline line reads `rank_ic=`, so **the cross-sectional pair
only reached the summary when it was a warning** — absent from every run where the edge was
positive, i.e. exactly when an arm had worked. `Cross-sectional` is now matched explicitly,
and a test pins the emitted line against the workflow's own pattern.

## Cost

`_anchor_is_tied` is 0.29s on a 1,018,600-row production-shaped frame, once per horizon —
~1.2s of a ~900s retrain. Tied share on iid noise is 33.4%, which is the 1/3 the three-point
median predicts and consistent with the 27–56% measured on the archive.

## What this does not do

It does not re-rank anything. Every stored A/B verdict — C1's +0.0556/+0.0561/+0.0371/+0.0316,
the `-return_1d` bar, the naive-`init_score` read — was measured on the pooled basis and is
still measured on the pooled basis. This makes the next read honest; it does not retroactively
fix the last one. Re-reading C1 on the tied cohort is one dispatch and is the obvious first
use.
