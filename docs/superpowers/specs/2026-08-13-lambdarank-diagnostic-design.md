# C2 lambdarank — within-date ranking diagnostic (design)

**Date:** 2026-08-13.
**Item:** C2 in `docs/research/2026-08-10-next-steps.md` — the `lambdarank` half (the rank-IC
*metric* already shipped as `8be48c5`; the objective is unbuilt).
**Scope decision:** diagnostic-first. This measures whether a within-date ranking objective
extracts more ordering signal than the pointwise q50. It does **not** change serving. A pass
opens the serving question as a separate, later pre-registration; it does not license it.
**Status of the field it sits in:** the two obvious accuracy levers are done and negative — C1
(`xs_rank`, +0.05 CV rank IC but closed 2026-08-13 because the gain is the anchor denominator,
not real forward signal) and N1 (`NAIVE_INIT_SCORE`, misses the directional classifier). C2 is
the last unbuilt *modeling* lever. Temper expectations: the source literature's own portfolio
results show a rank-IC gain that did not convert to performance.

## Why this experiment at all

Production trains quantile boosters (q10/q50/q90) per horizon with `objective="quantile"`; the
q50 sign is the served direction. On the shipped rank-IC diagnostic the q50 currently **orders
worse than the naive `−return_1d`**: `mean_rank_ic 0.1307` vs `naive 0.1656`, edge **−0.035**.
The served question is within-date ordering, but the loss is pointwise. A ranking objective is
the one untried thing that optimizes the metric the product is actually judged on. If within-date
ranking has no more signal than the q50, C2 closes and the ranking-vs-pointwise question is
settled cheaply.

## Architecture

A gated flag `LAMBDARANK=1`, off by default, read only in the CV diagnostic path (the same path
that already computes `mean_rank_ic` / `mean_naive_rank_ic` / `rank_ic_edge_vs_naive`). When set,
for each horizon and each CV fold:

1. Train **one** LightGBM booster with `objective="lambdarank"`, in addition to — never replacing
   — the existing quantile boosters. Its query `group` is the date's cross-section: rows are
   sorted by (fold, date) and the group vector is the per-date row counts.
2. Score the ranker's raw output through the **existing** `_within_date_rank_ic(pred, y_val,
   val_dates)` (forecaster.py:3926), so the number is directly comparable to the q50's and to
   `naive_rank_ic` on the same folds.
3. Report the ranker's `mean_rank_ic` and its `rank_ic_edge_vs_naive` under its own keys in
   `meta.json` / the CV summary, beside the unchanged q50 figures.

No serving path is touched. `model-diagnostics.yml` cannot promote an artifact, so the arm is
read-only by construction. One dispatch per horizon-shard reads it.

### Components and their boundaries

- **Label builder** (`_lambdarank_relevance(y, dates)` — new, pure): forward return → graded
  integer relevance, bucketed **within each date** via `qcut` into K levels. Input: forward
  returns + date keys for one fold. Output: an integer relevance array and the group-size vector.
  No LightGBM dependency; unit-testable in isolation.
- **Ranker trainer**: wraps `lgb.train` with the ranking params below. Depends on the label
  builder's group vector. Emits a booster whose `.predict` gives per-row scores.
- **Rank-IC reader**: the existing `_within_date_rank_ic` / `_summarise_rank_ic`, unchanged. The
  ranker's scores are just another `pred` argument.

The seam that matters: the label builder and the rank-IC reader never touch LightGBM, so the
relevance construction and the scoring can each be tested without training anything.

## Label construction and ranking params

- **Relevance:** forward return `qcut` into **K = 8** graded bins **per date** (ties → lowest
  bin, so an all-flat date collapses to one relevance level and contributes nothing, which is
  correct). Within-date bucketing makes relevance the within-date rank the objective optimizes;
  a global bucketing would re-import the market factor as relevance.
- **`lambdarank_truncation_level`:** set to the max group size in the fold (cover the whole
  cross-section). The default 30 optimizes only the top-30 NDCG over ~900 items, which is not the
  full-ordering quantity rank IC measures.
- **`label_gain`:** linear (`[0,1,2,…,K-1]`). The exponential default (`2^label − 1`) blows up at
  K=8 and would let the top bin dominate the loss.
- **`lambdarank_norm`:** default (True) — documented as a knob, not tuned in the first read.
- **HP otherwise:** held at these fixed defaults across arms for comparability (num_leaves,
  depth, learning_rate mirrored from the q50 params where they apply). The magnitude is therefore
  provisional; a positive result must be re-confirmed with `FORCE_HP_SEARCH=1` before its size is
  quoted, the same caveat the tier-lead and xs_rank instruments carry.

## The trap this design is built around

Within-date rank IC is contaminated by the anchor-deviation factor `p[d]/S[d]` — the raw anchor
quote `prepare_targets` divides the label by. That factor is readable within a date and is
precisely what made `xs_rank` show +0.05 CV rank IC that then failed to serve
(`changelog/2026-08-13-cohort-geometry-is-not-c1s-gap.md`) and what made `LABEL_SMOOTHED_ANCHOR`'s
entire gain vanish on the tied cohort (`changelog/2026-08-11-smoothed-anchor-label-measured.md`).
A ranker will learn to order on `p/S` and post a fake rank-IC win.

**Therefore the primary read is on the tied cohort** — item-days whose anchor quote equals the
local median, where `p/S ≡ 1` and the factor cannot operate — following the established rule
"rank arms on the tied subset." The pooled cohort is reported only as descriptive, never as the
verdict.

## The bar — pre-registered, fixed before any number

Both conditions are required, both on the **tied cohort**:

1. **Beats the naive baseline:** lambdarank `rank_ic_edge_vs_naive ≥ 0`. The q50 fails this today
   at −0.035, so clearing it is a real, non-trivial event.
2. **Beats what is already served:** lambdarank `mean_rank_ic` > the q50's `mean_rank_ic` on the
   same folds. A ranker that ties the q50 buys nothing.

Read per horizon. A horizon that clears both is a candidate for the (separate) serving step; a
horizon that clears neither is closed. One dispatch per horizon-shard; comparability rests on the
documented bit-reproducibility of the rank-IC diagnostic across runs at fixed HP and folds.

## Predictions, before the run

Pre-registered point prediction: **the tied-cohort edge is null or negative at 3/4 horizons.**
The reasoning: every within-date signal measured in this archive has either been the market
factor or the `p/S` factor, both of which the tied-cohort read removes; once removed, "no
idiosyncratic signal in the features" has held at every prior test. The horizon most likely to
clear is 14d, where the tier-lead column already lands top-5 — but that is a weak prior, not a
bar. If any horizon clears both conditions on the tied cohort, that is the first genuinely new
forecasting result since the archive audit, and it earns the serving pre-registration.

## Cost

One lambdarank booster per horizon ≈ one quantile model in training time. All four horizons
plausibly fit a single dispatch under the 30-minute cap; if not, shard by horizon (the flag reads
the same regardless of how many horizons the dispatch runs). No retrain of production, no serving
write, no `--rescore`.

## Void conditions

- HP or fold geometry differs between the lambdarank arm and the q50 baseline it is read against.
- The naive baseline is recomputed on a different label basis than the arm.
- A **level** metric (DA, MAE, coverage) is quoted for the ranker — its output is an ordinal
  score with no return scale, so those are undefined; only rank IC is meaningful.
- The **pooled** cohort is read as the verdict instead of the tied cohort.
- Relevance is bucketed globally rather than per-date (re-imports the market factor).

## What a pass and a null each mean

**Pass (some horizon clears both bars on the tied cohort):** within-date ranking extracts real
ordering signal the pointwise q50 misses. The next step is a separate pre-registration for how a
rank score becomes a served direction and band — it is not licensed here, and building that
plumbing before this read would be building for a likely null.

**Null:** the pointwise-vs-ranking question is settled against ranking, C2 joins C1 and N1 as a
measured dead end, and the honest remaining accuracy lever is N2's exogenous market-factor leg.
One cheap dispatch bought that closure.
