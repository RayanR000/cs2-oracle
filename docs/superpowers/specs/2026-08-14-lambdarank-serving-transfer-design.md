# C2 lambdarank — serving-transfer read (design)

**Date:** 2026-08-14.
**Item:** the serving-transfer pre-registration deferred by
`docs/superpowers/specs/2026-08-13-lambdarank-diagnostic-design.md` and earned by
`docs/changelog/2026-08-13-lambdarank-clears-the-diagnostic-bar.md`. It is the top-priority
"change the target" move in `docs/research/2026-08-14-what-moves-skin-prices-web-reconsideration.md`
(re-test plan item 1) and item 5 of `docs/research/2026-08-13-next-steps.md`.
**Scope:** read-only diagnostic. Trains q50 and lambdarank boosters to historical cutoffs and
scores them against outcomes the archive already holds. **Writes nothing** — no serving path, no
artifact, no database, no `q_hat`. A pass licenses a *further* serving pre-registration; it does not
license shipping the ranker.

## The question

The CV diagnostic (`2026-08-13-lambdarank-clears-the-diagnostic-bar.md`) showed the lambdarank arm
beats both the naive `−return_1d` baseline and the q50's own within-date ordering at all four
horizons on the tied cohort — **vs-q50 rank IC edge +0.0648 / +0.0614 / +0.0391 / +0.0297**,
HP-confirmed. But those folds validate on data ending **2026-02-09**, while production serves the
**2026-04-18 → 2026-06-08** window. The two legs share no dates
(`docs/changelog/2026-08-13-cv-folds-are-not-time-aligned-with-serving.md`). This experiment asks
one thing: **does the vs-q50 rank IC edge survive to the served months, on the served
cross-section?** It is the confound that C1 was never closed against — all four of C1's closure
arguments were about *which rows*; this is about *which months*.

## What this is NOT

- Not a product-transfer read. It does **not** wire a rank score into `predict()`, map it to a
  served direction/band, or push it through the serving transforms (recenter / tier-bias / prior-day
  blend / conformal). That is the *next* pre-registration if this passes. Here both objectives are
  read as raw within-date scores, on the same rows, so the delta isolates the objective.
- Not a re-run of the CV diagnostic. It reuses the diagnostic's fold trainer but places the
  validation cross-sections in the served window and adds two portfolio metrics the CV path does not
  produce.
- Not a production fold-grid change. It does **not** implement next-steps item 2 (anchoring the CV
  grid to the frame end), which would move production's `q_hat` and PT sample as a side effect. This
  read is isolated in its own script and leaves production's grid untouched.

## Architecture

A standalone script `scripts/replay_lambdarank.py`, run through the venv from `backend/`, read-only.

### Reused, not reimplemented

- **Referee** (import from `scripts/replay_serving.py`): `_outcomes`, `_resolve` (trailing smoothed
  outcome on the served basis, `after=anchor`), `_tied_mask`, `_pinned_anchor`, `audit_anchor_feed`,
  `_feed_profile`, `cutovers_from_counts`, `cutovers_in_outcome_window`. These are single-source by
  deliberate design (each docstring warns against a second copy drifting); this script must not fork
  them.
- **Ranker trainer** (`ItemForecaster._lambdarank_fold_scores`, forecaster.py:8523): already trains
  one lambdarank booster on an arbitrary `train_df`, query-grouped by date, and returns raw scores on
  an arbitrary `val_df` in row order.
- **Feature engineering / universe / embargo:** the forecaster's own loaders and
  `engineer_features` with the production allowlist (`price_technicals`, 33 cols) and
  `skip_unused_groups=True`, the `archive_universe_sql_filter()` universe, and
  `embargo_days(horizon)` = `horizon + 13`.

### New code

1. **`_fold_q50_scores(train_df, val_df, horizon, per_quantile_params)`** — extracted twin of
   `_lambdarank_fold_scores`, factored out of the existing CV fold loop so the q50 raw scores come
   from the *same* code the production diagnostic uses. Both boosters then train on identical rows,
   HP, and embargo — the controlled contrast the CV changelog's "transferable number" rests on. The
   existing fold loop is refactored to call it, so there is one q50 fold trainer, not two.
2. **Walk-forward driver.** See cadence below.
3. **Two portfolio metrics.** Decile long-short spread and the long-short PT test (below).

### Walk-forward cadence — retrain biweekly, score daily

Retraining a q50+lambdarank pair at every served date is ~400 boosters and exceeds the 30-minute
cap. Production retrains on a ~14-day age gate and serves daily from the standing artifact, so a
faithful replay does the same:

- **Retrain points:** `2026-04-18`, `2026-05-02`, `2026-05-16`, `2026-05-30` (14-day cadence from the
  window start). At each, train the q50 and lambdarank boosters to that date's cutoff (features
  bounded at the cutoff, `embargo_days(horizon)` purge — no look-ahead in either booster).
- **Scoring:** each retrain point scores every frozen anchor date in its forward 14-day block
  (`[point, next_point)`), clipped to the window end `2026-06-08`.
- **Training rows:** bounded by `CV_MAX_TRAIN_ROWS` (300,000), as the CV diagnostic's folds already
  are, so one retrain ≈ one fold's cost.
- **Sharding:** `--horizons 3` etc., one horizon per invocation like `replay_serving`, to stay under
  the cap. Four shards cover 3/7/14/30d.

This yields ~40–50 within-date cross-sections (one per frozen anchor date) per horizon — the
statistical unit for rank IC and PT.

## Cohort and anchor set — pre-registered

- **Cohort: the tied subset only** — item-days whose raw anchor quote equals the local smoothed
  median (`_tied_mask`), within the served ≥$1 universe (`_artifact_min_median_price`, else
  `MIN_SERVED_PRICE_USD`). This is non-negotiable: the pooled within-date rank IC is contaminated by
  the anchor-deviation factor `p[d]/S[d]` (0.03–0.24), which a ranker orders on happily and which
  made `xs_rank`/C1 a serving mirage and `LABEL_SMOOTHED_ANCHOR`'s gain vanish on the tied cohort.
  On the tied cohort `p/S ≡ 1`, so raw and smoothed anchor denominators coincide and the read is
  basis-invariant. Pooled is computed and printed as descriptive only — **never** the verdict.
- **Anchor set:** every date in `[2026-04-18, 2026-06-08]` that (a) passes `audit_anchor_feed` and
  (b) has no `cutovers_in_outcome_window` for the horizon being scored. The surviving list is
  enumerated and **frozen and logged before any rank IC is read**. `ALLOW_DIRTY_ANCHOR=1` is not set
  — a dirty anchor is dropped, not diagnosed here. (The two known contaminants, the 2026-03-22
  consensus break and the 2026-07-09 feed substitution, fall outside the window; the audit is still
  run, not assumed.)

## Metrics and bars — pre-registered, fixed before any number

All on the **tied cohort**, per horizon, averaged over the frozen anchor dates.

1. **Primary — Rank IC edge vs q50.** Mean over anchor dates of
   `within_date_rank_ic(lr_score) − within_date_rank_ic(q50_score)`, both scored against the realised
   forward return via `_within_date_rank_ic`. **Serving transfer confirms at a horizon iff this edge
   > 0.** CV showed +0.026–0.065 at 4/4; the honest expectation is attenuation, possibly to null at
   30d (the smallest CV edge).
2. **Secondary — Rank IC edge vs naive.** Same, `lr − naive(−return_1d)`; reported, not a gate.
3. **Decile long-short spread.** Per anchor date, sort the tied cross-section by lr score, take mean
   realised return of the top decile minus the bottom decile; report the mean spread and its
   distribution across dates. Reported **gross and net** of the 15% Steam fee + tier spread (see
   caveat). Descriptive.
4. **Long-short PT test.** Pesaran–Timmermann on the per-date long-short return series (is the sign of
   the decile spread predictable). This is the web doc's third ruler and the only significance test
   here that respects `backend/AGENTS.md` invariant 4. Descriptive.

The verdict is read **per horizon**. A horizon whose vs-q50 edge stays > 0 is a candidate for the
serving pre-registration; a horizon that goes to zero or negative did not transfer and is reported as
such.

### The tradeability caveat — conditions interpretation, does not gate

A relative-value edge must clear the 15% Steam fee and the 35%→5% tier spread to be *actionable*
(`what-moves-skin-prices` caveats). The decile spread is therefore reported net of a fee/spread
haircut alongside gross. But "is there signal" and "is the signal bigger than costs" are different
questions: the net figure informs whether a passing horizon is worth a serving build, it does **not**
decide whether the edge survived. The bar is the gross vs-q50 rank IC edge.

## Void conditions

The read is void, not merely weak, if any hold:

- The **pooled** cohort is quoted as the verdict instead of the tied cohort.
- Relevance is bucketed **globally** rather than within-date (re-imports the market factor as
  relevance) — inherited from `_lambdarank_labels`, which already buckets within-date; a
  reimplementation must not regress it.
- A **level** metric (DA, MAE, coverage) is quoted for the ranker — its output is an ordinal score
  with no return scale.
- The q50 and lambdarank boosters are trained on **different** rows, HP, or embargo — the contrast
  then measures something other than the objective.
- A **dirty** anchor (failing the feed audit or spanning a cutover) is scored as a measurement.
- Look-ahead: either booster is trained on data past its retrain-point cutoff, or the outcome window
  reaches back across the anchor (`_resolve(..., after=anchor)` guards this; do not remove it).

## Cost

Per horizon shard: 4 retrain points × 2 boosters on ≤300K rows ≈ 8 fold-scale trains, plus feature
engineering once per retrain point. This is comparable to a fraction of one CV diagnostic run and
fits the 30-minute cap per shard. No production retrain, no serving write, no `--rescore`. Four
shards (one per horizon) cover the experiment.

## What a pass and a null each mean

**Pass (vs-q50 edge > 0 on the tied served-window cohort at ≥1 horizon):** the within-date ranking
edge is not a pre-serving-months artifact — it reaches the cohort and the months production actually
serves. That earns the *next* pre-registration: how a rank score becomes a served direction and band,
and whether the edge survives the trip through `predict()`'s transforms. It does **not** license
shipping the ranker, and building that plumbing before this read would be building for a possible
null.

**Null (edge ≤ 0 at every horizon):** the CV vs-q50 edge did not transfer to the served window. C2
joins C1 and N1 as a measured dead end for serving, the "optimize the wrong target" thesis loses its
one piece of transferable internal evidence, and the honest remaining lever is N2's exogenous
market-factor leg. One bounded set of dispatches bought that closure.
</content>
</invoke>
