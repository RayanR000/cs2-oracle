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
| `C5` | Split conformal + ACI. **Do F1 first.** |
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
