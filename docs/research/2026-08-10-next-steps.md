# Next steps after the 2026-08-10 model audit

**Source:** the audit recorded in `docs/changelog/2026-08-10-constant-call-is-hindsight-picked.md`
and `docs/changelog/2026-08-10-band-and-confidence-are-miscalibrated.md`.
**Supersedes ordering in:** `docs/research/2026-08-09-next-steps.md`. Every item carried forward
keeps its old label (`C1`, `C2`, `C5`, `O2`, `G2`, `D2`–`D5`) so it can be traced. That document
remains the reference for the *content* of those items; this one re-ranks them and adds two tracks.

**Ranked by accuracy gain per minute of work**, which is the stated objective: fast training, fast
A/B, maximum accuracy.

---

## What the audit changed about the priorities

Three things, and the first is the reason for the re-rank.

**1. The model is not badly beaten by anything runnable on directional accuracy.** The published
"edge vs constant call" of −4.08 / −9.65 / −16.01pp at 7/14/30d is measured against a baseline that
picks its direction with hindsight, per fold, and switches sides on 4 of 8 folds at 30d. Against the
runnable always-down call the served classifier is **+3.5 / +0.1 / −1.3 / +4.4pp** — a wash to
modest positive, inside per-fold noise. Full derivation in the companion changelog.

*Consequence:* work aimed at closing a 16pp DA gap is aimed at a gap that does not exist. Deprioritised.

> ✅ **Not contradicted by realised outcomes (2026-08-11), and the figures were corrected
> 2026-08-12.** `changelog/2026-08-11-the-da-gap-is-the-market-direction-of-five-dates.md`.
> `2026-08-11-clean-anchor-gate-measured-on-realised-outcomes.md` §3 measured the served
> classifier **11–30pp below always-down** on the ≥$1 `lgbm-v3*` panel, which read as a
> contradiction of this item. Decomposed per (horizon, date) on the corrected panel,
> **52% / 82% / 103%** of that at
> 3/7/14d is the realised direction of the 3–8 anchor dates and the within-date term is
> **−2.26 / +0.34 / +4.98pp** (mirror-era: 56/78/89% and −1.19 / +0.99 / −1.08). ⚠️ **The CV-basis
> leg — +2.70 / +0.17 / +0.42pp against the +3.5 / +0.1 / −1.3 quoted here — was NOT recomputed and
> is unverified, so "the two measurements agree" is an open claim.** What holds is that composition
> is the first-order term, so the pooled deficit is not evidence against this item, which therefore
> stands. Per-date sd of the excess is 14.96–24.39pp.
> **New constraint on everything below:** never read `DA − realised_down_rate` on fewer than ~50
> forecast dates — report the within-date term. And **never read the local working copy of
> `price-archive/ops/forecast_outcomes.parquet` for a panel figure**: it holds 14,668 of the 23,073
> ≥$1 scored rows and its gaps are selected on "the verdict changed", which is what cost nine
> published figures on 2026-08-11. **Neither Parquet copy is the full panel** — the durable archive is
> fresh and cell-complete but only 10 dates deep, so **the publish leg is not broken**; query prod
> Postgres read-only. The one defect the read surfaced is **diagnosed**:
> 2026-07-19 called `flat` on 64.0% of items at h=3 against a **23.59%** realised flat rate (18.3%
> hit rate — a bad call, not a guaranteed miss), because `f71ffb4` shipped a global ±0.5% dead band
> in `predict()` 42 minutes before that run and no directional classifier existed yet. Two open
> items, both small: exclude 07-19 as a distinct **serving config** (`served_identity()` cannot see
> the direction rule, so it will not separate it), and put a WARNING plus a flat-call count on
> `predict()`'s no-classifier fallback, which still reapplies the band. **Also a correction to the
> record: 2026-07-19 is the production daily run and 2026-07-18's `-global-only` rows are an
> ablation arm that overwrote it** — the reverse of what `da-is-dominated-by-market-date` states.

**2. The one real, runnable baseline the model loses to is `−return_1d`, on rank IC, at every
horizon** (−0.0159 / −0.0371 / −0.0433 / −0.0091). This uses no hindsight and survived the
2026-08-09 re-vote. It is the only baseline gap that is both measured and legitimate.

*Consequence:* Track N below, and it is now the top priority.

**3. Two serving-path calibration defects and one structural measurement blocker were found**, none
of which is an accuracy problem in the modelling sense. They are cheap and they gate what can
honestly be published.

*Consequence:* Track F.

**What did not change.** The PT verdict is `skill` at all four horizons on the served signal
(t = 13.18 / 8.00 / 5.40 / 4.31). The model has real within-date signal. Nothing here is an argument
that it is a null model — the argument is that a free one-liner currently has more of the same
signal.

---

## Track N — close the naive-baseline gap (new, highest priority)

### N1. Fit the q50 on top of `−return_1d` via `init_score`

> ## ⚠️ MEASURED 2026-08-10, and this ranking is now wrong.
> `docs/changelog/2026-08-10-instrument-panel-first-read.md`. N1's rank-IC edge is
> **+0.0090 / −0.0115 / +0.0131 / −0.0167** at 3/7/14/30d — still negative at 7d and 30d, so the
> offset does **not** floor the model at the baseline. Worse for the framing here: N1's served
> figures are byte-identical to the control's, because the offset reaches the q50 boosters and
> not the directional classifier that production serves.
> **`C1` below is the arm that cleared the bar**, at all four horizons (+0.0556 / +0.0561 /
> +0.0371 / +0.0316), and it raises served PT excess by 45–96% as well. Track N is no longer the
> top priority; `C1` is.

> **Instrument shipped 2026-08-10, gated off (`NAIVE_INIT_SCORE=1`).** See
> `docs/changelog/2026-08-10-naive-init-score-instrument.md` — it covers all five `Dataset`
> seams and every predict path, and adds a `naive_init_score` arm to `model-diagnostics.yml`.
> What remains is the read: one dispatch per horizon against a control on the same commit.
> Two things the changelog corrects about the plan below: the floor is **empirical, not
> algebraic** (a boosted model can fit its way back below the offset), and a CV rank-IC gain
> does **not** transfer to the served mid until **F1** lands, because `predict` recentres the
> mid on the classifier's call afterwards.

- **Do:** pass `init_score = −return_1d` (in the same units as the target) on the training
  `lgb.Dataset`, and add it back at predict time. The model then fits the *residual* to the naive
  predictor instead of competing with it.
- **Why:** the measured gap is −0.0091 to −0.0433 rank IC at all four horizons. `init_score` makes
  "loses to the one-liner" structurally impossible rather than empirically true: the worst case is
  the model learns nothing and reproduces the baseline. This is the standard boosting-from-an-offset
  pattern and LightGBM supports it natively.
- **Why this and not the swap:** `docs/research/2026-08-10-training-cost-levers.md` §"The structural
  option" records replacing the served mid with the naive predictor, correctly flagging it as a
  product decision rather than a cost tweak. `init_score` gets the same floor without giving up the
  q50, and keeps the conformal band's level.
- **Read:** `mean_rank_ic` against `mean_naive_rank_ic` on the same folds. The bar is
  `rank_ic_edge_vs_naive ≥ 0` — currently negative at all four.
- ⚠️ **Two things to get right.** (a) The offset must be applied on the predict path too, or the
  served mid silently loses the baseline; assert it in a test rather than trusting the symmetry.
  (b) `tuned_params` were selected against the un-offset target, so hold HP fixed for the first
  read and confirm any positive with `FORCE_HP_SEARCH=1` before believing its size — the same
  caveat the tier-lead instrument carries.
- **Vehicle:** `model-diagnostics.yml`, one arm per horizon. It reports rank IC and served
  classifier accuracy and cannot promote its artifact.
- **Cost:** ~10 lines plus a test; one dispatch (~10 min wall clock) to read.

### N2. Split the problem into a market layer and a within-date rank layer

> ## ⚠️ FIRST READ DONE 2026-08-11 — the own-history leg is NULL.
> `docs/changelog/2026-08-11-market-factor-is-not-forecastable-from-its-own-history.md`.
> The pre-registered `forecast_market_factor` (trailing 180d drift) loses to a **flat-zero
> forecast** on MAE at all four horizons, correlates **negatively** at all four (−0.093 to
> −0.162), and its DA is at or below the realised down-rate at all four. It is negative on
> 75–78% of dates while the factor is up-majority — an anti-signal, not a weak one.
> **What remains open is the exogenous leg only** (FX, events, player counts — that ingest
> exists and was not used). The decomposition is not refuted; it currently buys nothing over
> a zero market term. The 4-of-4 negative correlation is a mean-reversion signature and is
> **post-hoc** — pursuing it needs its own pre-registration.
>
> ❌ **It was pre-registered, tested, and it does NOT replicate.**
> `changelog/2026-08-11-mean-reversion-does-not-replicate.md`. On 2014-2023 — a disjoint
> decade, ~2,000 non-overlapping windows — the correlations are −0.038 / +0.008 / +0.038 /
> +0.016 (primary 1 of 4, FAIL) and the contrarian call loses to the constant call by 12-22pp
> at every horizon (secondary 0 of 4, FAIL). **N2's own-history leg is closed in both
> directions.** Only the exogenous leg remains.

- **Do:** forecast the market factor as a single per-date series (one number per day, not ~900), and
  keep the existing cross-sectional model for relative position. Served item return = market
  forecast + relative component.
- **Why the decomposition is the right shape here:** three independent measurements in this repo say
  the cross-section is dominated by one common factor —
  `da-is-dominated-by-the-market-date`, `no-idiosyncratic-signal-in-features`, and the audit finding
  that the hindsight constant call scores 68.99% at 30d precisely *because* it re-picks that
  factor's sign each fold. The factor is where the predictable variance is; the model currently has
  no term for it.
- **Why it is measurable and the current work is not:** the item-level A/B floor is **2.21–3.69pp**
  fold-clustered (`docs/research/accuracy-opportunities.md`, Measurement Floor) against candidate
  effects of 0.5–1.5pp — those experiments cannot return a decision. The date-level series is a few
  thousand observations of one quantity; the exogenous ingest built for it already exists
  (`docs/changelog/2026-08-06-date-level-exogenous-ingest.md`). Experiments at this layer are
  cheap to run and cheap to power.
- **Cost:** small. One series, seconds to fit, no change to the item pipeline for the first read.
- ⚠️ **Read the prior refutation before starting, and note it is a different change.**
  `docs/changelog/2026-08-06-market-relative-labels-refuted.md` demeaned the **label** and made
  things worse; its own diagnosis is that demeaning spread the label distribution 3.3–9.5× while a
  pointwise quantile loss fought the residual. N2 leaves the label alone and predicts the two
  components separately. The mechanism that killed the earlier attempt does not apply — but that is
  an argument, not a result, and it should be stated as such in whatever this produces.
- ⚠️ **The market factor may be unforecastable.** That is a legitimate outcome and it is worth
  knowing: it would mean the constant call's apparent strength is unreachable in principle, which
  closes the question rather than leaving it open. Budget one experiment, not a programme.

### C2 (carried forward). `lambdarank` within date, scored by rank IC

- **Unchanged from `2026-08-09-next-steps.md`, and its cautions still bind** — the truncation level
  defaults to 30 over a ~900-item cross-section, `label_gain` is exponential, `lambdarank_norm`
  matters, and a ranker emits a score rather than a level so it **adds** four boosters rather than
  replacing them. The paper evidence supports a rank-IC gain and argues *against* expecting a
  performance win.
- **Why it moves up:** the audit sharpened the diagnosis it rests on. Rank IC 0.10–0.18 alongside
  DA at or below the runnable base rate is the signature of a model that orders well and calibrates
  badly, and the loss is pointwise while the served question is within-date ordering.
- **Do N1 first.** It is a tenth of the work against the same metric and the same bar, and if it
  closes the gap the case for adding four boosters weakens considerably.

---

## Track F — fixes that gate what can be published (new)

None of these is an accuracy improvement. All three are cheap, and all three block an honest claim.

### F1. Calibrate `q_hat` around the mid that is actually served

> ## ✅ SHIPPED 2026-08-11 — and inert on the daily path by design.
> `docs/changelog/2026-08-11-conformal-centre-follows-serving.md`.
> `_conformal_records` now measures the residual against `_recenter_on_direction`'s own
> output, both callers pass a direction call, and `meta.json` carries `conformal_centre`
> per horizon. Sized on a held-out split: **59.8% → 79.1%** served coverage against an
> 80% target, with the never-served q50-centred band at 79.5%.
> **And the coverage claim did not replicate — the decision is CLOSED, change nothing.**
> Three arms at two anchors (`31529688893` / `31532517480` / `31532543867`) agree within
> **1.1pp** in all 8 cells: served centre, q50 centre, and recentring-off are
> indistinguishable. The displacement is bounded by `2*|mid|` and predicted `|return|` is
> median 0.95% / p90 9.01% against half-widths of ±10–31% (`q_hat` 95.5–312.7). The 59.8%
> figure came from a synthetic generator whose mid was *larger* than its half-width — the
> inverse of production. **Do not pay the 932s and do not drop the recentring.**
> F1's deliverable is coherence plus disclosure (`conformal_centre`, the WARNING,
> `BAND COVERAGE`), at zero cost.
> **The real finding is a new question, and it outranks what is left here:** the replay
> reads ~81% pooled coverage on the band production reports at `IntCov` 34.6–61.8%. Two
> numbers that far apart for one band put the fault in the **backtest's comparison** — the
> suspect is `in_interval` testing an archive-resolved actual against a band built on the
> served `current_price`, a wedge of median 5.70% / p90 37.82% against half-widths of
> 10–31%. The "both legs are dollars" ruling in
> `2026-08-11-actionable-selection-is-the-base-wedge.md` is about units, not basis.
> **C5 is unblocked, and it should be re-scoped: recalibrating a band whose reported
> coverage is measured on the wrong basis will chase the wrong residual.**
>
> ✅ **That basis is FIXED, 2026-08-11 — `changelog/2026-08-11-in-interval-basis.md`.**
> `_derive_verdict` rebases the band by `base / quote` before the predicate (`quote` is
> keyword-only and required, so no new path can omit it), and both figures are now reported:
> `interval_coverage` (calibrated, comparable to the replay) beside
> `interval_coverage_dollar_basis` (the published dollars, which is what 34.6–61.8% was).
> Neither leg of the outcome moved, and `abs_error` / `pct_error` are deliberately not rebased
> — a dollar error is basis-free. **C5 is now genuinely unblocked**: it can be scoped against a
> coverage number that means what it says. What remains is a read — one `--rescore` to see
> where production's `IntCov` lands against the replay's ~81%, which is a prediction until run.

- **Do:** apply `_recenter_on_direction` inside the conformal records path before
  `_calibrate_conformal`, so the calibration centre and the serving centre are the same.
- **Why:** `q_hat` is fitted on residuals to the **q50** mid, then `predict` moves the centre three
  times — prior-day blend, tier bias, and the directional recentring that can translate the mid by
  twice its magnitude or pin it to exactly zero. Half-widths are preserved, which is what breaks
  coverage. Production reports `IntCov` 34.6–61.8% on the `lgbm-v3` cohort against an 80% target.
- **Cost:** no extra model fit. Needs classifier predictions in CV, which
  `CV_DIAGNOSTIC_CLASSIFIER=1` already supplies in `model-diagnostics.yml`.
- **Relationship to C5** (split conformal + ACI, carried forward as R18): F1 is prior and much
  smaller. C5 changes the calibration *scheme*; F1 fixes which centre the current scheme calibrates
  to. Do F1 first — C5 built on the wrong centre inherits the same defect.
- **Add the test that would have caught it:** pooled served coverage on a held-out window against
  `NOMINAL_COVERAGE`. The existing check is unfalsifiable — `q_hat` measured on the OOF it was fitted
  on is ≥80% by construction (`2026-08-10-served-classifier-scored.md` §4).

### F2. Calibrate or withdraw the served `confidence` label

> ## ✅ SHIPPED 2026-08-12 as the withdrawal.
> `docs/changelog/2026-08-12-served-confidence-withdrawn.md`. `confidence` is gone from
> `PredictionOut` and `TrendAnalysisOut`; the column, the writer and `conf_gap_pp` stay so
> the withdrawal stays falsifiable.
> **One correction to the reasoning below.** The evidence is not that the tag inverts.
> `conf_gap_pp` is pooled across forecast dates (`backtest/scoring.py:416-419`), so the
> −4.6 / −4.8 / −2.0 / −38.7pp it prints carries the same market-composition term this
> document elsewhere forbids reading — and its h=30 figure is **one item**. Recomputed
> within-date on the ≥$1 `lgbm-v3%` panel, on the 11 cells with `n_high >= 30`, the gap runs
> **−8.26 to +4.65pp with mixed sign** (7 of 11 negative): no measurable separation either
> way. What decides it is the level, not the gap — the `high` cohort is right **29.0–45.8%**
> in every powered cell, against `CONFIDENCE_TARGET_ACCURACY = 80.0`. A tag reading "high" on
> sub-coin-flip calls is false regardless of the ordering.

- **Do:** either calibrate the classifier-probability cut against realised hits and store it under
  its own key, or stop publishing `confidence` until it is calibrated.
- **Why:** the served label is a bare `>= 0.5` cut on a 3-class argmax
  (`DIRECTION_CONFIDENCE_HIGH`), never validated. The `confidence_thresholds` that *were* fitted
  describe `_compute_confidence`, which only runs on the no-classifier fallback — so `high_accuracy`
  in `meta.json` and `conf_gap_pp` in the backtest are measuring two different things and have been
  read as one. Separately, the fitted path shows `range_pct` carries no directional information at
  all: it never reaches its 80% target, falls back to maximising coverage, and lands at 39.9 / 41.8 /
  42.6 / 47.1% against a q50 mean of 39.9 / 42.0 / 43.1 / 48.4%.
- **Cost:** small. Withdrawal is a one-line product change and is the right interim state — an
  uncalibrated confidence tag is worse than none, because consumers weight on it.

### F3. Stop encoding configuration in `model_version`

> ## ✅ SHIPPED 2026-08-11.
> `docs/changelog/2026-08-11-model-version-is-not-a-config.md`.
> `served_identity()` keys the cohort on the artifact, the writer puts the config in the
> Parquet mirror and **observes** it (`SKIP_REGIMES=1` had been labelling itself `-regime`),
> and `config_dates` / the `pooling …` prefix disclose every merge. No migration and no
> production write — legacy rows canonicalise at read. On the ops mirror the panel goes
> **5 → 8 / 4 → 7 / 2 → 4 / 1 → 1** forecast dates at 3/7/14/30d.
> **It does not produce a headline**, and the diagnosis above was half right: the code half
> is fixed, and ~12 more daily runs at h=3 are the calendar half. Two things it corrects —
> the suffix never separated a row (`item_forecasts` is unique on
> `(item_id, forecast_date, horizon_days)`, so the second config **overwrote** the first),
> and it does **not** unblock the bias fit, which already pooled via `LIKE 'lgbm-v3%'`.

- **Do:** separate the served-model identity from its configuration, so a config flag does not fork
  the scoring panel. Alternatively, score across versions where the artifact is materially the same.
- **Why:** `ops/item_forecasts.parquet` holds **6 distinct forecast dates in total**, split
  `lgbm-v3-regime` (3) / `lgbm-v3` (2) / `lgbm-v3-global-only` (1). `score_cohort` keys on
  `model_version`, so run `31409508960` scored 110,615 frozen outcomes and returned
  `NO HEADLINE (insufficient_dates)` at every horizon. `global-only` and `regime` are the same
  artifact with `SKIP_REGIMES` flipped.
- **Why waiting does not fix it:** `docs/README.md` calls the 20-date failure "a calendar problem,
  not a code problem". Half of it is a code problem. Twenty daily runs yield twenty dates only if
  nothing about the config moves for twenty days — and `SKIP_REGIMES=1` landed 2026-08-10.
- **Cost:** small, and it is the binding constraint on ever publishing a live accuracy figure.

---

## Deprioritised by this audit

- **Anything scoped as "close the constant-call gap".** The gap is against an oracle. This includes
  the framing in `2026-08-10-tier-lead-and-rank-transform-instruments.md`'s "Follows" line; the
  instruments themselves are unaffected and still worth measuring on rank IC.
- **The 30d-horizon question**, but only in priority, not in substance. It remains the only horizon
  whose `feature_validation` fails (`price_technicals` drop 0.18pp, p = 0.40) while its served PT
  reads t = 4.31, and it is ~40% of runtime. The tension is real and unresolved; it is simply
  cheaper to resolve after N1, which changes what the 30d model is fitting.
- **Further retrain-cost work.** The warm arm64 retrain measured **996.6s training / 17m48s job**
  (run `31407938154`), inside the 30-minute cap. `docs/research/2026-08-10-training-cost-levers.md`
  budgets against 1884–2306s runs that predate the warm cache, `SKIP_REGIMES=1` and arm64; its
  remaining levers are real but no longer urgent. **The velocity problem is the experiment loop, not
  the retrain** — see below.

> ## ⚠️ 2026-08-11: the metric itself is the bottleneck, not the harness power
>
> The serving replay attributes the CV↔serving gap to **one axis**: `prepare_targets` divides by
> the single raw quote at the anchor, which the features are also built from. Swapping only that
> denominator recovers 95% of the gap (mean Δ rank IC +0.1398 of +0.1464); the serving transforms
> account for none of it. Every arm in this document was ranked on CV rank IC, and so was the
> `-return_1d` bar they were held to. `changelog/2026-08-11-the-gap-is-the-anchor-denominator.md`
> and `changelog/2026-08-11-serving-transforms-are-not-the-gap.md`.
>
> **Replicated across 10 anchors** (`changelog/2026-08-11-clean-anchor-signal-replicates.md`): the
> gap is zero on items whose anchor quote equals its local median (sign counts 4–6 of 10) and
> positive at 10 of 10 where it deviates, at every horizon. The same run found the project's first
> measured served signal — rank IC **+0.12 to +0.16** on that clean third of the cohort, positive
> at 8–10 anchors of 10 — against **−0.2750 at 10 of 10** on the deviating two-thirds at h=3.
>
> Two consequences ahead of everything below: **rebuild the CV label's denominator** before it
> ranks another arm, and **a serving gate on anchor cleanliness** is now evidence-backed.
>
> ✅ **The first is built, gated, and MEASURED — and it is not the fix.**
> `LABEL_SMOOTHED_ANCHOR=1` (`changelog/2026-08-11-label-smoothed-anchor.md`), read against a
> control on `4d0c8bc` at four non-overlapping anchors (runs `31459789143` / `31461219973`,
> `changelog/2026-08-11-smoothed-anchor-label-measured.md`). Pooled served rank IC swings to
> +0.17–0.31 at 4/4 anchors — and all of it is `p[d]/S[d]`, the anchor deviation, entering the
> label as a factor the model can read at the anchor. On the **tied** cohort, where that factor
> is exactly 1, the arm is −0.033 / −0.017 / −0.020 / +0.008. Keep the flag off.
>
> **Two consequences.** Rank arms on the **tied subset** — both label bases are contaminated by
> `p/S`, with opposite signs, and it is the only cohort where neither operates. And the serving
> gate below is retired as an accuracy lever: the tied/deviating split measures how far the
> quoted `current_price` sits from the last observation, not those items' prices. The open
> question it raises is a **serving-basis** one — quote against a fresher anchor — not a label one.
>
> The section below is still true about statistical power. It is now the second problem.

## The measurement problem is now the bottleneck

Worth stating plainly, because it bounds everything in Track N and C:

- The item-level A/B floor is **2.21–3.69pp** fold-clustered; every candidate effect in this repo is
  0.5–1.5pp. Those experiments **cannot return a decision** regardless of how they come out.
- **No `ab_test_*` harness reports rank IC** — `grep -l rank_ic scripts/ab_test_*.py` returns
  nothing — and rank IC is the metric every item in Track N is read on. `model-diagnostics.yml` is
  the only vehicle that reports it.
- The one thing working in your favour: rank IC reproduced **exactly** across two runs at identical
  cached HP and folds (`2026-08-10-served-classifier-scored.md` §1). That bit-reproducibility is the
  control a paired harness would otherwise have to supply, and it is why one dispatch per arm is
  readable — provided the arm does not perturb the item draw.

**Do not build more `ab_test_*` harnesses.** Invest in `model-diagnostics.yml` and in the date-level
layer (N2), where the floor is low enough that an experiment can conclude. This is the same
conclusion `accuracy-opportunities.md` reached — *"the shortest path to any further accuracy work is
a measurement problem, not a feature problem"* — and the audit does not overturn it.

## Carried forward unchanged

Content in `docs/research/2026-08-09-next-steps.md`; re-ranked below N and F.

| Item | State |
|---|---|
| `C1` | ❌ **CLOSED 2026-08-11 — re-read on the clean cohort against a pre-registered bar and it failed the serving leg.** CV passed 4/4 (arm edge +0.1368/+0.1312/+0.1253/+0.0943, Δ vs control +0.0792/+0.0801/+0.0641/+0.0509), serving tied-cohort deltas positive at only 2/4, 2/4, **0/4**, 4/4 anchors — and 30d alone was ruled insufficient in advance. `changelog/2026-08-11-c1-fails-the-clean-cohort-read.md`. Superseded detail below. |
| `C1` (old) | ⚠️ **CV-positive, and it does not appear at serving.** CV edge +0.0556 / +0.0561 / +0.0371 / +0.0316 on cached HP (re-confirmed 2026-08-11 with its own control: Δ vs control +0.0686 / +0.0883 / +0.0755 / +0.0483). But the serving replay of the same two artifacts is **worse in 6 of 8 cells**, mean Δ −0.0176 / −0.0478 / −0.0763 / +0.0447. Two anchors, no interval — not a refutation, and not shippable on the CV number either. `changelog/2026-08-11-rank-transform-does-not-transfer-to-serving.md`. |
| `C3` | Residual reversal as a feature. Not started. Note the overlap with N1 — both are `−return_1d` derivatives. |
| `C4` | Re-derive the feature allowlist with `cross_sectional` restored. Compute the MDE first; it may not clear the floor. |
| `C5` | ⚠️ **2026-08-12: the coverage half is now DIAGNOSED, and it was never calendar-blocked.** The remedy is not a calibration set, not a scheme and not time-varying — it is `sigma`'s **exponent**. Level-matched to 80% marginal, every date-conditional scheme (60-day trailing `q_hat`, three date-level volatility states) is **worse than pooled**, with shuffled placebos at ≤0.11pp: there is no date-level information to condition on, the same wall N2's own-history leg hit. Coverage by `sigma` decile ramps **62→95%** (h=3) to **58→98%** (h=30), monotone in all ten deciles at 4/4 horizons, and a fitted **β = 0.408 / 0.401 / 0.363 / 0.327** cuts the stratum error **8.62/8.02/8.00/9.73pp → 0.57/1.20/1.82/1.01pp**. `sigma` spans 11× across deciles while the `\|residual\|` it normalises spans 2.3–2.6×. **ACI's forecast-date shortage is irrelevant** — this axis is cross-sectional and ~500K rows already exist offline. ⚠️ **Not implementable yet:** two reads of the exponent disagree 3× (0.33–0.41 here vs the 0.798/0.692/1.034/1.150 measured on `sigma` implied as `half_pct / q_hat` from ~20K prod rows), and the deciding statistic was post-hoc after the pre-registered read went **void on its placebo**. Next is **one report-only dispatch** on real OOF residuals. `docs/changelog/2026-08-12-the-band-is-tilted-in-sigma.md`, instrument `backend/scripts/measure_conditional_qhat.py`. ✅ **That dispatch RAN (`31619383780`) and confirms it:** elasticity **0.429 / 0.369 / 0.350 / 0.313** on ~157K real OOF records, within **0.014–0.032** of the model-free prediction, deciles **62→93 / 60→95 / 57→96 / 52→97**; the 0.798/0.692/1.034/1.150 read is refuted at 4/4 and hardest at 14d/30d, which it had called fine. ⚠️ **Held out, the remedy only reaches 3d/7d** — conditional error **−76% / −62%** against **−26% / −12%** at 14d/30d, where one global exponent leaves the profile un-flat. So: implement per-horizon β for 3d/7d, and treat 14d/30d as a separate decision (shrunk or non-parametric scale). ⚠️ **And it does not explain the marginal over-coverage** — level-matching removes that quantity before the tilt is measured, so 87.2/91.8/90.6/89.0% vs 80% is still unattributed after six causes. `docs/changelog/2026-08-12-sigma-tilt-confirmed-on-oof-residuals.md`. ✅ **2026-08-12, later the same day: the marginal half is now PARTLY ATTRIBUTED, and to the same exponent.** Served `sigma` runs **1.28–1.29×** the calibration median — **measured directly**, not implied as `half_pct / q_hat` — and pushing the measured coverage-vs-`sigma` curve through that mix buys **+4.0 to +4.9pp**, i.e. **36–68%** of the excess (`A` = +0.68/+0.40/+0.44/+0.51). All pre-registered legs pass: validity MAE **0.89–1.43pp**, per-date footprint corr **+0.50 to +0.61** (R² 0.25–0.38 of the per-date coverage spread), and the **`β = 1` placebo at 0.0000pp**, which proves the channel is the tilt and nothing else. ⚠️ **The `0.93×` at 30d is refuted** — `sigma` has no horizon term, so it rested on production's single 30d forecast date; the "predicts the opposite at 30d" objection is withdrawn. ⚠️ At **7d the remainder is unreachable** from this channel (`k* = 2.45×` beats every date in two years). The residual 32–64% is **named by identity** — the residual law at given `sigma`, measurable on resolved served outcomes. **This is a second, independent argument for shipping `β`, and it applies at all four horizons** rather than only the 3d/7d the conditional read supports, because marginal invariance needs average flatness rather than per-fold stability. `docs/changelog/2026-08-12-marginal-over-coverage-is-half-the-sigma-mix.md`, instrument `backend/scripts/attribute_marginal_coverage.py`.  ✅ **The remaining design question is CLOSED the same day** — `changelog/2026-08-12-the-sigma-scale-is-one-exponent-per-horizon.md`. Walk-forward, 507–588 dates, production's 14-day refit cadence, four arms: **shrinkage is a no-op** (β's departure from 1 is 6–14× its standard error; λ = 0.996–0.999; the shrunk arm reproduces the plain one to two decimals) and **flexibility buys nothing** (per-decile `q_hat` worse at 3/4; a binned non-parametric scale better only inside the noise and worse on marginal coverage at 4/4). **Ship one fitted exponent per horizon.** Held out it cuts the level-matched tilt **−89% / −94% / −84% / −73%** and narrows bands to **0.87 / 0.86 / 0.84 / 0.77×**, i.e. **it reaches all four horizons**, which contradicts the −26% / −12% at 14d/30d above (43 independent refits vs one held-out CV fold; model-free panel vs real OOF residuals — the dispatch settles it, so stop repeating "only 3d/7d" as settled). ⚠️ The pinned selection rule **misfired** and picked production at 4/4 by gating on a marginal coverage that drifts 5–10pp between periods for every arm; the comparison is labelled post-hoc. ⚠️ Tilt and width wins are stable across both periods, the **marginal win is not** — the exponent arms cover **74–77%** on the earlier period, so this fixes the tilt and leaves the level open in both directions. Implementation designed, not built: `docs/superpowers/specs/2026-08-12-sigma-exponent-design.md`. ✅ **PAIRED-READ 2026-08-12 (`31629626929` control / `31629638834` arm, commit `5d1bc04`) — and `SIGMA_EXPONENT` stays OFF.** β reproduces at **0.4241 / 0.3672 / 0.3370 / 0.3266** (within 0.024 of the OOF read), the (β, `q_hat`) pair is never split, the audit lines are identical across arms, and σ backs out of the widths at **0.0687 / 0.0678** from both ends — the implementation is coherent. ❌ **The 0.87/0.86/0.84/0.77× width bar FAILS on the basis it was quoted on**: the calibration set moves only **0.966 / 0.957 / 0.947 / 0.965×**, while the *served* band narrows **0.794 / 0.768 / 0.750 / 0.761×**. Mechanical — the ratio is `(q_b/q_1)·σ^(β−1)`, decreasing in σ, and served σ runs above the calibration median. **A width or coverage bar on this flag must name calibration-set or served before the dispatch; the two support opposite verdicts on the same number.** ⚠️ **The cost is the level**: served marginal coverage falls **−8.07 / −8.26 / −7.88 / −2.19pp** paired over four anchors (t = −3.91 / −5.81 / −3.80 / −1.33), which helps at 7d/14d/30d and is a **regression at 3d**, where the control was already at 77.80% and the arm overshoots to 69.72%. Misses grow on both sides, so it is width, not centre. ⚠️ **`fold_beta` hits the `[0.2, 1.0]` clamp** on 2–3 early folds at 7d/14d/30d, contradicting "no window of this archive has produced" one, and β drifts **0.20–0.31 (folds 1–4) → 0.33–0.61 (folds 5–9)**, so the pooled fit sits below what a 14-day refit would land on. ⚠️ **The held-out leg replicates 3d/7d-only for the SECOND time** (−78% / −67% / −22% / −35%), so the walk-forward −84%/−73% at 14d/30d loses the dispute. **Ship β only together with a re-level against the served σ mix.** `docs/changelog/2026-08-12-sigma-exponent-paired-read.md`. ❌ **AND IT FAILS ITS OWN JOB AT SERVING (`31643819235` / `31643829741`, commit `0d9f426`) — the flag stays off for a second, independent reason.** The tilt IS real where the band is served — the control's coverage ramps **+15.66 / +9.94 / +8.10 / +9.13pp** across σ quintiles, positive at **15 of 16** anchor-horizon cells, though a third to a half the size of the OOF ramp. But β **flips the sign rather than flattening it**: the arm ramps **−9.50 / −13.57 / −14.54 / −2.48**, negative at **14 of 16** cells, significant at 4/4 (t = −6.50 / −8.49 / −5.52 / −3.27), and **larger in magnitude than the control at 7d and 14d**. Only 30d improves. Read the **ramp**, not `sigma_tilt_pp` — the ramp is level-free and the tilt statistic is inflated by the arm's ~8pp marginal drop. ⚠️ **The exponent is fitted on the wrong population, and this REINSTATES a refuted read.** Interpolated, the served-flattening exponent is **~0.64 / 0.73 / 0.76 / 0.47** — roughly **double** the fitted 0.42/0.37/0.34/0.33 and far closer to the prod-implied **0.798 / 0.692 / 1.034 / 1.150** that was published as "refuted at 4/4". Those were two different populations; the served one governs. ⚠️ **Second basis confusion on this feature in one day** — the first cost two cancelled dispatches. ⚠️ **The horizon story inverts too**: OOF says works at 3d/7d, fails at 14d/30d; serving says overcorrects at 3d/7d/14d, roughly right at 30d. **Next is a different instrument — fit β on resolved served outcomes, which needs a forecast-date panel rather than a CV fold.** `docs/changelog/2026-08-12-served-sigma-profile.md` ⚠️ **A LEARNED scale was then built and measured (`31649561391` / `31649571169`, commit `5b39760`), and it also stays off.** One small booster per fold on `log|residual|` from the item's features, on records that already exist — **26.6s, 2.6% of training**. CQR was costed and **does not fit**: the OOF conformal CV is 63.5% of training and needs p10/p90 out of fold, putting a retrain at **40.9–46.3 min** against a 30-minute cap. The learned scale beats `SIGMA_EXPONENT` at 7d/14d, loses at 30d, narrows bands 0.78–0.88×. ⚠️ **Do not quote its 14d ramp of +0.65 as flat** — per anchor it is −4.09 / −15.74 / −8.95 / **+31.39**, i.e. cancellation. `2026-07-09` reverses the arm's ramp at **4 of 4** horizons; on the other three anchors the arm **overcorrects harder than the exponent** at 3d (−22.69) and 7d (−23.26), and 30d (+0.75) is the genuine win. 🔑 **Three scales tried, three different calibration→serving displacements** (`sigma` over-covers 77.9/85.0/84.8/87.2, `σ**β` under-covers 69.7/76.7/76.9/84.9, learned under-covers 72.1/78.7/76.6/85.7) — all calibrated to exactly 80% on their own records. **The width variable is not the lever; a fourth one is not the next experiment.** The live thread is why one date reverses all four horizons. `docs/changelog/2026-08-12-learned-band-scale-measured.md`. 🔑 **THAT THREAD IS ANSWERED, and 2026-07-09 is not a market date at all.** The archive's usual feed (`aggregator_steam_17mafo`, ~26,170 items, median $2.97-3.00) is **absent** on 07-09 and `aggregator_sync` (5,502 items) stands in, quoting the served ≥$1 cohort at **1.287× (p10 1.046, p90 1.414, 85.3% of items up >10%)** and reverting **0.773×** the next day, while 07-08→07-10 is **1.000×**. It is the only such substitution day in 2026. Everything follows: tied share **2.5%** against 27-56% (flagged as an outlier twice, never diagnosed), the highest down-rate of the four anchors at 3/4 horizons, the highest DA (63-70%, calling down into an arithmetic decline), and the learned scale **widening** the band there (halfw 15.77/18.42/26.23/29.18 vs control 8.22/12.11/16.96/25.64) because a +28.7% anchor-day move enters the volatility features its scale model reads. ⚠️ **One published sign flips:** `sigma`'s served marginal at h=3 is **81.83%**, not 77.80% — so it over-covers at **4 of 4** (81.8/87.8/85.9/88.7) and h=3 was never the under-covered horizon. ✅ The control's tilt survives (ramp +8.81/+3.60/+9.86/+11.55, positive 4/4) and both flags still stay off. **Drop 07-09 and guard the anchor** — not by substituting a neighbour (07-10's own `return_1d` is the −22.7% reversion; 07-01-07-08 resolve across the 07-11 regime change). ✅ **GUARD SHIPPED the same day** (`audit_anchor_feed` in `scripts/replay_serving.py`, ten tests, refuses before the artifact loads, `ALLOW_DIRTY_ANCHOR=1` to override) — **and it immediately found a SECOND bad anchor of the four.** `2026-04-15` is the last day of the three-source era (`buff163+csfloat+youpin`, ~32,420 items); from 04-16 the archive holds `steam_17mafo` alone at ~25,010, so the anchor is quoted on one collection and **all four horizons resolve on another** — median 0.979 across the boundary but **30.1% of served items move >±10%** against 3.9-7.0% on an ordinary pair. ⚠️ **So the "four-anchor mean" behind this whole row rested on TWO clean anchors.** On 05-16 and 06-16 alone the control covers **85.78 / 90.20 / 84.14 / 86.24%** (over-covers 4/4 by **+4.1 to +10.2pp**, against the −2.2 to +7.1pp published), its ramp is positive 4/4 and larger (+11.18/+6.77/+10.24/+12.09), and the learned arm overcorrects at 3/4 (−25.18/−16.93/−12.34/+1.58). Same verdicts, stronger: **the over-coverage LEVEL is the only open quantity.** The audit refuses 12 of the 187 h=30-resolvable dates of 2026; usable stretches are `01-01..03-19`, `03-24..03-27`, `04-01..04-13`, `04-18..07-06`. **Next dispatch: `2026-04-22,2026-05-16,2026-06-16,2026-07-06`.** `docs/changelog/2026-08-12-july-09-anchor-is-a-feed-substitution.md`. ✅ **THAT DISPATCH RAN (`31657639707`, commit `d51fa32`) and the level is now measured on four audited anchors: 87.58 / 88.97 / 86.09 / 86.19%, i.e. over-coverage of +6.1 to +9.0pp at 4 of 4 horizons and 15 of 16 cells.** All four anchors passed the audit in CI with item counts matching the local read, tied shares are 33.7-56.4% with no outlier, and the two-clean-anchor prediction held to **1.8 / 1.2 / 2.0 / 0.05pp**. The tilt is confirmed a third time (ramp +7.20 / +11.09 / +11.97 / +11.03, positive 15 of 16; elasticities 0.431 / 0.368 / 0.335 / 0.317). 🔑 **New lead on the LEVEL, and it is flat in horizon:** the served band is **1.529 / 1.548 / 1.553 / 1.520×** the median half-width the same `q_hat` produces on its own calibration records — a constant, where the known served-`sigma` mix of 1.28-1.29× predicts only 1.29×, leaving **1.19× unaccounted**. Candidates are the denominator (served mid ≠ calibration mid) and the cohort (~1,030 served items on 4 dates vs 155-176K pooled OOF rows). ⚠️ It is one comparison of two medians from a log, not a paired read — **next is to compute both on the same items, which needs no retrain.** One unplanned gain: `2026-07-06` is up-majority at h=3 (33.89% down), where the retired set was down-majority at 4 of 4. `docs/changelog/2026-08-13-band-level-on-audited-anchors.md`. |
| `C5` (superseded) | ⚠️ **Half done, 2026-08-12, and the half that is left is calendar-blocked.** The over-coverage (87.2 / 91.8 / 90.6 / 89.0% vs 80%) was a **calibration-basis** defect, not a scheme defect: `q_hat` was fitted on the raw-anchor label while the band is served and scored on the smoothed anchor. Built in `ef455ab` and **gated off — REFUTED as a remedy** on a clean paired read (arm `31564924194` vs control `31564943172`, same commit and folds: `q_hat` moves UP at 4/4, +9.5 / +5.0 / +0.7 / +1.6%, where over-coverage needs it 25–39% smaller). Three causes are now excluded — the centre, quiet dates, the denominator — and the next candidate is `sigma` itself. The quiet-dates alternative was measured and rejected (median `rel_cal` 1.01 / 0.96 / 1.07 / 0.97 on the calibration window's own dates). **ACI is still the right answer to conditional coverage** — per (h, date) the band runs 58.2%–99.2% and `sigma` has no date term — but it needs a realised-coverage feedback series and production has 7 / 6 / 3 / **1** forecast dates. Not runnable yet. The *cost* half (one split slice instead of the CV block) is untouched and must not be confused with it: `forecaster.py:5218-5232` explains why the OOF set is the only unseen calibration data. `docs/changelog/2026-08-12-conformal-basis-follows-serving.md`. |
| `C6` | Regime decision — the `SKIP_REGIMES=1` half landed 2026-08-10; `N_ENSEMBLES` untouched. |
| `C7` | Deflate the accumulated A/Bs. Every stored verdict predates the 2026-08-09 re-vote. |
| `O2` | The config-description half is still open. |
| `G2` | Part 2 not started. |
| `D2`–`D5` | Unblocked, none started. |

## Do not re-propose

Everything in the "Do not re-propose" list of `2026-08-09-next-steps.md`, plus:

- **Differencing model DA against `constant_call_accuracy`.** It is hindsight-selected. Use
  `realised_down_rate` as the runnable baseline and the PT verdict as the test.
- **Quoting `DA − realised_down_rate` on a short date panel.** Per-date sd is 14.96 / 22.40 /
  21.56 / 24.39pp at 3/7/14/30d and a zero-skill model lands ≤ −11pp on 30–38% of archive dates, so
  a 5-date pooled excess carries no information about the model. Report the within-date term.
  `changelog/2026-08-11-the-da-gap-is-the-market-direction-of-five-dates.md`.
- **Reading either Parquet copy of `ops/forecast_outcomes.parquet` for a panel figure.** Neither is
  the full panel (2026-08-12): the **local working copy** is deep but stale, holding 14,668 of the
  23,073 ≥$1 scored rows, missing two whole dates at h=7, with its surviving rows in a deficient cell
  selected on "the verdict changed"; the **durable archive** is fresh and cell-complete but only 10
  dates deep, missing 2025-12-01 and 2026-07-17 entirely. **The publish leg is not broken** — don't
  chase one. Query prod Postgres read-only, or a CI run's `prediction_accuracy`. Same entry.
- **Shaving the retrain further as a priority.** It is inside the cap; the loop is bounded by
  experiment power, not by training seconds.
- **A time-varying / date-conditional `q_hat`**, in any spelling — a trailing window, a
  regime regression, or ACI keyed on a date-level state. Level-matched, all of it is worse than
  pooled at 3–4 of 4 horizons while shuffled placebos move ≤0.11pp. A date-level `q_hat` has to
  forecast the date's realised cross-sectional dispersion, which is the market factor, which is
  already refuted from its own history in both directions.
  `changelog/2026-08-12-the-band-is-tilted-in-sigma.md`.
- **Comparing coverage schemes on `mean_d |cov[d] − 80%|` without matching the marginal level
  first.** That statistic falls whenever marginal coverage moves toward target, so a uniformly
  narrower band scores as a conditional-coverage fix — a *shuffled* state variable passed a
  pre-registered bar on it. Force each arm to 80% marginal, then compare. Same entry.
