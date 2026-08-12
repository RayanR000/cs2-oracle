# The 11–30pp DA gap is the realised market direction of five to seven anchor dates

**Date:** 2026-08-11
**Closes:** §3 of `docs/changelog/2026-08-11-clean-anchor-gate-measured-on-realised-outcomes.md`
— "the largest unexplained gap currently on the board".
**Against:** the published CV figure of **+3.5 / +0.1 / −1.3 / +4.4pp** at 3/7/14/30d
(`2026-08-10-constant-call-is-hindsight-picked.md` §2, CI run `31418286692`).
**Method:** read-only, in-memory DuckDB over `price-archive/ops/forecast_outcomes.parquet` and
`price-archive/prices-*.parquet`. No `--rescore`, no Postgres write, no code changed.
**Also records:** the diagnosed cause of 2026-07-19's flat calls (a ±0.5% dead band, no classifier),
and a **correction** — 07-19 is the production daily run and 07-18 is the ablation arm, which is the
reverse of what the repo had recorded.

**The gap is not a model failure and not a measurement error. It is composition** — the direction
the market happened to take over five to seven anchor dates, measured against a metric whose
per-date standard deviation is 15–24pp.

## Cohort

`served_identity` matching `lgbm-v3*`, `base_price >= 1.0` (`MIN_SERVED_PRICE_USD`), scored rows
only: **14,668 scorable rows over 11 forecast dates and 1,199 items**, of which the 2025-12-01
backdated batch is reported separately throughout. §3's 14,586 is this cohort minus **82 rows on
6 slugs** that fall outside `archive_universe_sql_filter` — the same 82 rows §3 dropped rather than
defaulted. The recomputed `_tied_mask` reproduces §3's tied/deviating split exactly, so this is the
same population, read with a different statistic.

## The decomposition

`excess = DA − q_down` is additive per (horizon, forecast date), with `p_c` the model's served call
share, `q_c` the realised class share on that date, and `q̄_c` the archive-average class share:

- **`side_ref = Σ_c p_c q̄_c − q̄_down`** — the model's call mix costed against a *typical* market.
  What its side selection is worth before this window's direction is known.
- **`composition = (Σ_c p_c q_c − q_down) − side_ref`** — this window's realised direction.
- **`assoc = DA − Σ_c p_c q_c`** — within-date directional information. **The only skill term.**

`q̄_c` is measured on the cohort's own **1,193 items** over **1,282–1,307 anchor dates** of archive
history: `(down, flat, up)` = `(.453, .100, .447)` at h=3, drifting to `(.487, .033, .480)` at h=30.

Always-down has `assoc ≡ 0` and `side_ref ≡ 0` by construction — it makes one call and that call is
the reference — which is what makes the attribution clean rather than a change of basis.

## The result: composition is first-order, skill is ±1pp

Live dates pooled, served basis (components sum to `excess` to within rounding):

| h | dates | n | excess | side_ref | composition | assoc |
|---|---|---|---|---|---|---|
| 3 | 7 | 5,413 | **−13.12** | −4.62 (35%) | **−7.31 (56%)** | −1.19 (9%) |
| 7 | 6 | 4,022 | **−14.35** | −4.20 (29%) | **−11.14 (78%)** | +0.99 (−7%) |
| 14 | 3 | 1,242 | **−25.52** | −1.65 (6%) | **−22.80 (89%)** | −1.08 (4%) |
| 30 | — | — | no live forecast date exists at all | | | |

56–89% of the gap is the direction of the window. The skill term is **−1.19 / +0.99 / −1.08pp** —
the model has approximately no within-date directional edge on this panel, and approximately no
within-date deficit either. Nothing in the 11–30pp is a statement about the model's ordering.

**The panel mixes three direction rules** — a zero-threshold sign rule, a ±0.5% dead band, and the
classifier, in that order (diagnosed below). Almost all of the `side_ref` column is a single
dead-band date, 2026-07-19. That does not move the conclusion — neither `side_ref` nor `composition`
is skill either way — but it means the `side_ref` column is largely a one-date configuration
artifact rather than a property of the served classifier's call mix.

## The backdated batch differs in exactly one respect, and it flips the sign

Same model, same metric, 2025-12-01: excess **+1.80 / +3.99 / +5.00 / +5.96** at 3/7/14/30d —
**inside the CV band at all four horizons.** Its forward window *rose*, by an up-rate minus
down-rate of +15 to +33pp. That is the only respect in which it differs from the live dates. A
cohort that is 11–30pp below the baseline on falling windows and 2–6pp above it on a rising one is
describing the windows, not the artifact.

## The decisive number: on the CV label basis, the two measurements agree

Recomputing the *same live item-dates* on the CV label basis — `p[d+h] / p[d]`, `prepare_targets`'
own raw-anchor denominator with `LABEL_SMOOTHED_ANCHOR` off:

| h | `assoc`, CV basis | per-date mean (dates with n ≥ 100) | published CV edge |
|---|---|---|---|
| 3 | **+2.70** | +2.33 ± 0.85 (5 dates) | **+3.5** |
| 7 | **+0.17** | +0.33 ± 1.35 (5 dates) | **+0.1** |
| 14 | **+0.42** | one date | **−1.3** |

They agree. The published CV edge was never in conflict with production; it was being compared
against a pooled `excess` that carries a composition term CV averages away.

**Basis moves `excess` by +2.6 / −3.5 / +9.5pp, with no consistent sign.** DA is far less
basis-sensitive than rank IC — where the anchor denominator carried **+0.1398 of a +0.1464** swing,
all in one direction (`2026-08-11-the-gap-is-the-anchor-denominator.md`) — because the ±0.5% flat
band (`DIRECTION_FLAT_TOLERANCE_PCT`) absorbs most of the denominator perturbation before it can
change a class label.

## The model is not below always-down on every date

§3's "on every cohort at every horizon" is true of the pooled cells and false of the panel. The
model is **above** the baseline on 2026-08-06 (+4.1), 2026-08-01 (+19.7), 2026-08-02 (+6.4) and on
the whole backdated batch — the dates where the market rose. Over live dates with n ≥ 20:

- `r(down_rate, excess)` = **−0.87** at h=3 and **−0.83** at h=7.
- Dates with negative excess: 5/6 at h=3, 4/6 at h=7, 3/3 at h=14.

Pooling also over-weights the falling dates: `r(n, q_down)` = +0.47 at h=3 and +0.49 at h=7, so
pooled **−13.12 / −14.35 / −25.52** reads against unweighted per-date means of
**−11.63 / −11.45 / −32.65**, sd across dates 11.4 / 20.7 / 7.6. That is second-order and it does
not run one way. Composition is the first-order term.

## Sampling: the entire published gap is one to two standard errors

A zero-skill model carrying the live call distribution, evaluated on every archive anchor date with
≥ 200 item-pairs, lands **≤ −11pp on 30.0 / 37.8 / 34.8 / 33.5%** of dates and **≥ +3.5pp on
30.0 / 40.5 / 43.8 / 48.4%**. Per-date sd of `excess` is **14.96 / 22.40 / 21.56 / 24.39pp**, so
the SE of a mean over the handful of dates each horizon actually has is 6.7–15.3pp. The live draw
sits at **z = −1.0 to −1.6**.

An ordinary unlucky sample. A statistic this dispersed produces a reading at or below −11pp on
roughly a third of dates from a model with **no skill at all**, and a reading at or above +3.5pp
about as often.

## The one actionable defect: a ±0.5% dead band shipped 42 minutes before the 07-19 run

**2026-07-19 predicted `flat` for 64.0% of items at h=3** (34.3% at h=7, 25.2% at h=14). That date
alone carries `side_ref` of **−22.7 / −14.5 / −11.8pp** — essentially the entire pooled `side_ref`
term — and **no other date in the panel emits a single flat call** (`p_flat` = 0.00% on 2025-12-01,
07-17, 07-18, 07-29, 08-01, 08-02, 08-04, 08-05, 08-06, 08-07). Pooled flat across the ≥$1 scored
panel is 10.5 / 7.2 / 1.3 / 0.0% at 3/7/14/30d and **0.00% once 07-19 is dropped**.

**Diagnosed in a parallel read, and the mechanism is unambiguous.** `f71ffb4` ("fix: Correct
accuracy evaluation — pct_error denominator, multi-source voting, direction threshold",
2026-07-19 18:57:46 −0400 = **22:57 UTC**) introduced `DIRECTION_FLAT_TOLERANCE_PCT = 0.5` and
replaced `predict()`'s `up if mid_ret > 0 else down if mid_ret < 0 else flat` with a global ±0.5%
dead band. The 07-19 rows were written at `created_at = 2026-07-19 23:39:12` — **42 minutes
later.** 07-17 (22:04) and 07-18 (18:33) predate the commit, which is exactly why they show no
flat. Reconstructing `mid_ret = price_mid/current_price − 1` and testing every row against a global
±0.5% band with a half-cent rounding allowance is **100% consistent in 12 of 12 (horizon × call)
cells** — 668/668 flat, 151/151 up, 234/234 down at h=3, and the same at 7/14/30d. The apparent
boundary fuzz (flat rows out to ±0.95%, down rows in to −0.376%) is cent-quantisation of
`round(price, 2)`, **not** per-tier thresholds.

**It was not a classifier at all**, so the "confident flat versus three-way near-tie" question does
not arise — there were no class probabilities. `direction_models` was empty until `a332c2b`
(2026-07-24), and `_recenter_on_direction` demonstrably did not run: only 252 of the 668 flat rows
at h=3 have `price_mid = current_price`, and those are cent-rounding.

**Why the band caught the majority: the served mid is shrunk far harder than realised returns are.**
On 07-19 at ≥$1, **62.2%** of `|mid_ret|` fell under 0.5% at h=3 (35.8% at h=7) against a **23.1%**
realised flat rate — the gap to the 64.0% served flat share is the same half-cent rounding
allowance. On 07-17 it was **99.8%** (p25 = p50 = p75 = 0.000): had the band been live that day it
would have called almost everything flat. A symmetric dead band against a shrunk prediction
distribution.

⚠️ **Two flat rates, and they must not be conflated.** `q̄_flat` = **10.0%** at h=3 in the
decomposition above is the *archive average* over 1,282–1,307 dates. **The realised flat rate in
this cohort is 23.1% / 12.7% / 4.5% / 2.8% at 3/7/14/30d**, and that is the number a flat call on
07-19 was up against. So a flat call there was a **low-hit-rate call — 18.3% hit against a 23.1%
base rate — not "close to a guaranteed miss"**, which an earlier draft of this entry claimed off
the archive-wide figure. One consequence of the identity: `side_ref` costs the call mix at `q̄_flat`
while that date realised more than twice it, so part of the flat penalty sits in that date's
`composition` term rather than in its `side_ref`.

**Counterfactual cost** — the mirror sign joined to the realised direction, with ties still counted
flat, so these are lower bounds on the call mix's cost:

| h | served DA | zero-threshold sign rule | Δ | realised down-rate | ties |
|---|---|---|---|---|---|
| 3 | 27.6% | 36.3% | **+8.7** | 52.3% | 251 |
| 7 | 29.8% | 38.4% | **+8.6** | 54.4% | 87 |
| 14 | 29.6% | 34.8% | **+5.2** | 65.2% | 10 |

So at least +8.7pp of that date's penalty at h=3 is the call *mix*. The remainder is that 07-19 was
wrong-sided anyway — 21.9% down calls into a 52.3% down-rate.

**Blast radius: nothing served today is affected, and the rule is still reachable.** Flat is
**0.0% at every horizon on every classifier-era ≥$1 date** — 07-29 and 08-05 in the
`item_forecasts` mirror, 08-05 / 08-06 / 08-07 in the scored outcomes. Below $1 the classifier
*does* emit flat, on 29.4–33.0% of rows, but that is under `MIN_SERVED_PRICE_USD = 1.0`
(`backend/api/serving_policy.py:26`) — written to the mirror, never served and never scored.
**The dead band remains live as a fallback:** `predict()`'s no-classifier `else` branch
(`backend/models/forecaster.py:6965-6976` as of `67324fb`) reapplies `t_down` / `t_up` defaulting to
±`DIRECTION_FLAT_TOLERANCE_PCT` whenever `self.direction_models` has no entry for a horizon, so an
artifact missing its classifier re-enters this exact mode **silently**.

### Which run each date is — the labels are the reverse of what the repo recorded

⚠️ **`da-is-dominated-by-market-date` and any prose following it have this backwards, and this is a
correction, not a refinement.** At that day's code state (`git show
f71ffb4:backend/scripts/forecast_prices.py`), **run A — the unconditional daily path — writes
`f"{MODEL_VERSION}-regime"`** (`:212-213`), while `-global-only` is written **only** by
`--compare-regime`'s run B (`:217-228`), which writes to the same
`(item_id, forecast_date, horizon_days)` key and therefore **overwrites** run A's rows.

Therefore **2026-07-19 (`lgbm-v3-regime`) is the production daily run**, and **2026-07-18
(`lgbm-v3-global-only`) is an ablation arm's rows that overwrote that day's production forecast.**
Not the other way round. The overwrite-on-one-key mechanism is the same one recorded in
`2026-08-11-model-version-is-not-a-config.md`; what is new is which direction it ran.

**Coverage caveat on that argument.** The `item_forecasts` mirror holds only 6 forecast dates
(2025-12-01 — itself backdated on 07-17 — plus 07-17 / 07-18 / 07-19, 07-29, 08-05) and last synced
2026-08-05, so it carries **no `model_config` column** for these dates. The arm-label reading rests
on the **writer code at that commit**, not on an observed config. The scored panel jumps 07-19 →
07-29, consistent with the known 2026-07-14 → 07-31 CI outage.

## Also measured: no date-level market-direction information

`r(p_down, q_down)` = **−0.33** at h=3, **−0.09** at h=7, **−0.17** across all 15 readable live
cells. The model called down on **71.7%** of items into a 25.7% down-rate (2026-08-06) and on
**11.0%** into a 54.4% down-rate (2026-07-19 at h=7).

⚠️ One confound, which weakens rather than reverses this: **2026-07-19's call shares are the
dead-band rule's, not a classifier's** (see above), so its `p_down` is not comparable to the other
dates' and it is one of the cells inside these correlations. They were not recomputed with 07-19
excluded.

This is consistent with the closed own-history leg of N2 — the market factor is not forecastable
from its own history (`2026-08-11-market-factor-is-not-forecastable-from-its-own-history.md`,
refuted in both directions by `2026-08-11-mean-reversion-does-not-replicate.md`). It is **not new
evidence against the cross-sectional model**, which is not asked to call the market's level.

## Why the two measurements ever differed: different units of observation

A CV fold validates on `VALIDATION_WINDOW_DAYS = 30` distinct dates
(`_compute_cv_splits`: `val_d = sorted_dates[end:end + val_window]`), which is **15,532–24,979 rows
over ~30 anchor dates** per fold at h=3 (`n_val`, local `meta.json`; fold 1 spans 2023-02-26 →
2023-03-27). So a fold's `realised_down_rate` is *already averaged over a month of market
direction*, and its composition term averages to the archive mean — **−4.6 to −0.2pp**.

A live "date" is **one anchor**. Five of them do not average to anything. The two numbers were
never the same measurement, and §3 was right to say so; what it lacked was the size.

## The consequence, and it is the recommendation

**Never quote `excess` on fewer than ~50 forecast dates. Report `assoc` instead.**

`MIN_FORECAST_DATES = 20` (`backend/backtest/scoring.py:84`) is documented in-source as "a judgement
call, not a derivation", set to span more than one market swing. **It is not calibrated for a
statistic whose per-date sd is 15–24pp**: taking the measured per-date sd at face value, `sd/√20`
is 3.3–5.5pp — arithmetic, not a measurement — which cannot resolve a +3.5pp effect at 20 dates.
Twenty dates fixes the "which two dates did the cohort contain" failure it was written for; it does
not make `excess` a readable number. `assoc` is readable on the dates that exist, because the market
term is differenced out per date — which is also exactly what the PT statistic does, and why the
published headline is PT and not DA.

## What was deliberately not done

- **The `num_only` / `den_only` basis legs were not read.** Stored `base_price` sits a median
  **8.2% / 6.8% / 1.2% / 0.0%** from today's recomputed voted quote at *d*, so mixing one frozen
  leg with one recomputed leg measures the level wedge, not the basis: those legs return a 94.7%
  down-rate on one and an 86.7% up-rate on the other. Reported as artifacts and dropped. (This is
  a different pair from the `base_price`-vs-`current_price` wedge of median 5.70% / p90 37.82% in
  `2026-08-11-actionable-selection-is-the-base-wedge.md`.)
- **The CV figure was not recomputed.** `+3.5 / +0.1 / −1.3 / +4.4` is taken from CI run
  `31418286692` as cited in `2026-08-10-constant-call-is-hindsight-picked.md`. The local
  `meta.json` has `mean_classifier_acc = None` — `CV_DIAGNOSTIC_CLASSIFIER=0` on the production
  path, and the per-fold classifier is 52% of a retrain
  (`2026-08-10-arm64-and-scheduled-diagnostics.md`). **`mean_classifier_acc` is the correct CV
  counterpart**, because `forecast_outcomes.direction_predicted` is the served classifier's call
  (`predict`: "Served signal: the directional classifier"), **not `mean_dir_acc`** — which is the
  q50 sign and reads 39.9 / 42.0 / 43.1 / 48.4 in the local `meta.json`.
- **No `--rescore`.** It refits served bias corrections and is a production write; the ops mirror
  answers this question without one.
- **No PT test.** 3–7 dates is below what the `|t| > 3.0` hurdle and the HAC bandwidth rule were
  designed for — the same reason §4 of the prior entry reported none.
- **No interval on `assoc`.** Sign and size across dates are what is readable here.
- **The 2026-08-10 next-steps ranking was not rewritten.** Only the claims this contradicts are
  annotated.

## What this does not establish

- **The decomposition is new methodology, built for this read.** It is not an existing repo metric,
  has no test, and is not computed by `scoring.py`. **Flag it for review before it becomes the
  reported number.**
- **The CV-basis agreement rests on 5 / 5 / 2 / 0 dates.** h=14 is one usable date and h=30 has
  none, so "CV and production agree" is established **at 3d and 7d only**.
- **h=30 has no live forecast date.** Its only resolved rows are the backdated batch.
- Unreadable cells, explicitly: h=3 2026-08-04 (n=16, under the 20-row bar); h=14 2026-07-18
  (n=34, excluded from every mean — and see above: 07-18's rows are an ablation arm, so its
  membership in the panel is itself in question); h=14 overall rests on the 2026-07-17 cell alone. CV basis is
  unavailable for h=3 08-06 and 08-07, h=7 08-02, and h=14 07-19 — targets or anchors beyond the
  archive's 2026-08-08 max day, or on the missing days 07-27 / 07-30 / 08-02 / 08-03.
- `mean_realised_down_rate` of **46.06 / 48.73 / 50.56 / 48.48** is the **pooled** CV value, not a
  ≥$1-specific one. A minor caveat on the CV side of the comparison, unquantified.
- **The pooled table was not recomputed with 2026-07-19 excluded** (which would leave 6 / 5 / 2
  dates at 3/7/14d), so its `side_ref` and `assoc` are mixtures of serving configurations. And it is
  **three** rules, not two: `f71ffb4`'s ±0.5% dead band covers 07-19 only; 2025-12-01, 07-17 and
  07-18 were written by the **zero-threshold sign rule** that commit replaced; the classifier
  (`a332c2b`, 07-24) serves 07-29 onward. Which of those dates lands in which horizon's panel was
  not tabulated, so how much of each pooled cell is classifier-era is **unmeasured**.

## Verification

Cohort counts, `_tied_mask` reproduction and every figure above come from one read-only DuckDB
session over the ops mirror and the archive Parquet; the CV-side numbers (`n_val`, per-fold
`val_start` / `val_end`, `mean_realised_down_rate`, `mean_dir_acc`, `mean_classifier_acc = None`)
were read from `backend/models/saved_models/meta.json` (`trained_at` 2026-08-09). `VALIDATION_WINDOW_DAYS = 30`,
`CV_STEP_DAYS = 150`, `DIRECTION_FLAT_TOLERANCE_PCT = 0.5`, `MIN_FORECAST_DATES = 20` and the
`prepare_targets` denominator were read at source. No tests were added — there is no new code to
test, which is itself a reason the decomposition should not be published from a one-off read.

The 07-19 diagnosis rests on a **separate, parallel read** plus git archaeology, re-verified here at
source: `git show f71ffb4` carries the `DIRECTION_FLAT_TOLERANCE_PCT = 0.5` definition and the
`predict()` line it replaced; `git show -s f71ffb4` gives 2026-07-19 18:57:46 −0400; `a332c2b` is
dated 2026-07-24; `git show f71ffb4:backend/scripts/forecast_prices.py:206-228` is the run-A /
run-B writer that fixes the arm labels; the live fallback branch is `forecaster.py:6965-6976` and
`MIN_SERVED_PRICE_USD = 1.0` is `api/serving_policy.py:26`. The row-level consistency check,
`created_at` timestamps, per-date flat shares and the counterfactual table come from the parallel
read over the same ops mirror.

## Still open

Both recommendations from the 07-19 diagnosis are **open, not done.** Nothing in this entry changed
code or data.

1. **Exclude 2026-07-19 from the scored panel as a distinct serving config** — a different direction
   rule, not an ablation arm, because it *is* the production run of that day. Two things make this
   non-trivial: `served_identity()` deliberately merges `-regime` and `-global-only` and **cannot see
   the direction rule at all**, so it will not separate this by itself; and 07-18's membership
   should be reviewed in the same pass, since those rows are the ablation arm.
2. **Guard the fallback branch** (~10 min): a WARNING plus a flat-call count when
   `direction_models.get(horizon) is None`, so an artifact missing its classifier cannot silently
   re-enter the ±0.5% dead band. *Status at the time of writing:* a change matching this
   (`_warn_no_classifier`, plus `backend/tests/test_no_classifier_fallback_warns.py`) is present
   **uncommitted in the working tree** and was written by someone else. This entry does not verify
   it — the test was not run here — and it is recorded as open until it lands.
3. **Whether `assoc` should be a stored metric** beside DA and the PT statistic. It is the term that
   is readable on the panel the repo actually has. Raised here as a question; not a decision this
   entry makes.
4. **A production DA headline still needs the calendar** — 8 / 7 / 4 / 1 dates at 3/7/14/30d after
   `served_identity` merged the panel (`2026-08-11-model-version-is-not-a-config.md`). This entry
   says the 20-date bar is the wrong bar for `excess`, not that the panel is now sufficient.

## Related

- `2026-08-11-clean-anchor-gate-measured-on-realised-outcomes.md` — §3 is what this closes. Its §1
  and §2 (rank IC +0.037; the gate does not move direction) are untouched.
- `2026-08-10-constant-call-is-hindsight-picked.md` — the runnable-baseline correction this builds
  on. `constant_call_accuracy` is still not to be differenced against model DA.
- `2026-08-03-accuracy-is-clustered-by-forecast-date.md` — the clustering that makes composition
  this large; "the down-rate swings 32.7 → 76.9% between dates" is the same fact, now decomposed.
- `2026-08-11-the-gap-is-the-anchor-denominator.md` — the rank-IC basis swing this contrasts with.
