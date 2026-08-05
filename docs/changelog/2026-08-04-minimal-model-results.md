# Minimal model — measured results

**Date:** 2026-08-05
**Spec:** `docs/superpowers/specs/2026-08-04-minimal-model-design.md`
**Plan:** `docs/superpowers/plans/2026-08-04-minimal-model.md`
**Commits:** `1902aab` (grid 40→8), `deddc79` (`--save-records`), `b1bab03` (rig fixes)

## The headline

The 40-model grid collapsed to 8 models. Training went **381.2s → 176.7s (−54%)**
warm-to-warm. Directional accuracy is **unchanged at 3d and 7d, better at 14d
(+3.14pp), unchanged at 30d**. Both cheap baselines lose decisively: naive by
10–13pp, ridge by 22–25pp.

**The naive baseline is not competitive.** The spec required this finding to lead
if it were, so stating it plainly: predicting the sign of the trailing return is
10.0–12.7pp worse than the model at every horizon, with every confidence interval
entirely below zero. The tree ensemble earns its cost. Ridge on the same features
is worse still (−22 to −25pp, absolute DA 36.7–43.9%) — *below* naive, so the
linear model is not merely weaker, it is actively misleading on these features.

## What the bar was, and that it predates these numbers

Pre-registered in `2af8917` (2026-08-04). The arms ran 2026-08-05 13:27, so the
bar was committed **before** any arm was run, as the spec required.

| Horizon | MDE (pp) | Bar |
|---|---|---|
| 3d | 0.54 | paired `ci_lower_pp ≥ −0.54` |
| 7d | 0.74 | paired `ci_lower_pp ≥ −0.74` |
| 14d | deferred | measured post-rewrite, below |
| 30d | deferred | measured post-rewrite, below |

**What an MDE means, in plain terms.** The gate resolves 0.54pp at 3d, so the
claim it supports there is *"not worse by more than 0.54pp"* — not "equivalent."
Where a horizon's MDE is 1.2pp, the claim is only "not worse by more than 1.2pp."
No arm result below can be read as proving equality.

## Arm results — paired, date-clustered, identical folds

Config: `--arm {gbm,ridge,naive} --max-items 60 --step-days 120`, all 4 horizons.
Arm A ran from a worktree at `1890f41` (pre-rewrite code **and** pre-rewrite
`meta.json`); arms B/C/D on the post-rewrite checkout. All four share a
**100.00%-identical** `(item_id, forecast_date)` key set at every horizon — see
"The first arm run was invalid" below for why that is stated rather than assumed.

Absolute classifier DA, gate universe (**not** production DA — see caveats):

| Horizon | A: 40-model | B: 8-model | C: ridge | D: naive |
|---|---|---|---|---|
| 3d | 59.56% | 59.55% | 36.72% | 49.52% |
| 7d | 60.53% | 60.53% | 38.87% | 50.76% |
| 14d | 61.54% | **64.68%** | 39.35% | 51.30% |
| 30d | 68.57% | 68.79% | 43.86% | 55.84% |

### Arm B (8-model minimal) vs arm A (40-model pre-rewrite)

| Horizon | mean diff | 95% CI (clustered) | CI half-width | dates | n paired | Verdict |
|---|---|---|---|---|---|---|
| 3d | −0.011pp | [−0.029, +0.000] | 0.014 | 289 | 17,331 | **PASS** (see note) |
| 7d | +0.000pp | [−0.017, +0.017] | 0.017 | 288 | 17,268 | **PASS** (see note) |
| 14d | **+3.143pp** | [+1.955, +4.373] | 1.209 | 285 | 17,084 | **PASS — improvement** |
| 30d | +0.215pp | [−0.942, +1.287] | 1.115 | 280 | 16,779 | see below |

⚠️ **The fourth column is the half-width of that row's own CI, not a
seed-varied MDE.** It is arithmetically derived from column 3 and carries no
independent information — an earlier draft labelled it "own MDE", which invited
exactly the misreading that the deferred 14d/30d measurement had been delivered.
It has not been. See "Deferred MDE" below for why that turned out not to matter.

### Arm C (ridge) vs arm A

| Horizon | mean diff | 95% CI (clustered) | dates | n paired |
|---|---|---|---|---|
| 3d | −22.843pp | [−24.434, −21.188] | 289 | 17,331 |
| 7d | −21.659pp | [−23.859, −19.275] | 288 | 17,268 |
| 14d | −22.184pp | [−24.405, −19.914] | 285 | 17,084 |
| 30d | −24.715pp | [−27.069, −22.281] | 280 | 16,779 |

### Arm D (naive) vs arm A

| Horizon | mean diff | 95% CI (clustered) | dates | n paired |
|---|---|---|---|---|
| 3d | −10.040pp | [−11.645, −8.389] | 289 | 17,331 |
| 7d | −9.764pp | [−12.388, −7.271] | 288 | 17,268 |
| 14d | −10.238pp | [−13.148, −7.316] | 285 | 17,084 |
| 30d | −12.736pp | [−15.357, −9.998] | 280 | 16,779 |

`interval_coverage` and every `conf_*` field are omitted for arms C and D: their
band is a fixed `BASELINE_BAND_PCT` placeholder and their confidence is uniform
`low`, so those fields are structurally meaningless. **They are also omitted for
arm B** — for a different reason, given below.

## ⚠️ The 3d and 7d "PASS" is structurally guaranteed, not evidence

This is the most important caveat in the document and it constrains what the
whole exercise proved.

`scripts/walkforward_backtest.py` holds its **own** module-level
`QUANTILES = [0.1, 0.5, 0.9]` and fits one booster per quantile per fold. It
never reads `ItemForecaster.QUANTILES`, `N_ENSEMBLES`, `ENSEMBLE_SEEDS` or
`BOOSTING_TYPE_MAP`. So:

- **The quantile collapse (3 → 1) and the ensemble collapse (3 → 1 member) are
  invisible to this gate.** It never ensembled, and it still trains three
  quantiles in both arms.
- The only channel by which Task 11 reaches the gate is `meta.json`, which it
  reads for tuned params. There, the classifier's boosting type comes from
  `_get_tuned_params(meta, h, 0.5)["boosting_type"]`.

Verified directly: at **3d and 7d** the `_direction_tree_params` extracted from
the old and new `meta.json` are **byte-identical**, and both are `gbdt`. The gate
therefore *cannot* see any effect of the rewrite at those two horizons, and the
measured +0.000pp / −0.011pp with a CI half-width of 0.014–0.017pp is the signature of
two near-identical computations, not of a design that was tested and preserved
accuracy. (The ~2-record residual at 3d out of 17,331 is LightGBM's
`n_jobs=-1` non-deterministic reduction order.)

**Consequence:** the pre-registered 3d/7d bars were met, but they were met
vacuously. The quantile and ensemble collapse remain **unmeasured by this gate**.

Why that is nonetheless defensible: the served direction — the headline DA metric —
comes from the directional classifier, which was **never** ensembled and never
used p10/p90. One classifier per horizon, before and after. `QUANTILES` and
`N_ENSEMBLES` only ever fed the median price and the band, so they can move MAE,
MAPE and interval width, but they cannot by construction move served DA. What
*is* genuinely untested is their effect on the **median price**, which the gate
reports but which no pre-registered bar covers.

The horizons where the gate saw a real change are **14d and 30d** (dart→gbdt),
and that is where the result is real: 14d **+3.14pp better**, 30d unchanged.
14d was the horizon the spec flagged as most at risk — best production DA of the
four, and the one DART was supposedly earning its cost on. Removing DART
*improved* it.

## Deferred MDE for 14d and 30d — NOT measured, and why that is acceptable

The spec deferred these because measuring a DART design's noise floor needs two
DART passes. Post-rewrite both horizons are `gbdt`, so the measurement became
affordable (`compute_mde.py --max-items 60 --step-days 120 --horizons 14 30`,
~40 min). **It was deliberately not run.** The reasoning, recorded so the
decision can be reversed by anyone who disagrees:

An MDE exists to make a *null* result readable — it converts "we saw nothing"
into "nothing larger than X is present." Neither horizon needs that service:

- **14d is not a null result.** +3.143pp, CI [+1.955, +4.373], entirely above
  zero. A noise floor does not change how a measured improvement whose interval
  excludes zero is read. If anything the MDE could only have been *smaller* than
  the effect, which is already implied.
- **30d's non-inferiority claim comes free from its own CI.** [−0.942, +1.287]
  already licenses "not worse by more than 0.94pp at 95% confidence," which is
  precisely the statement an MDE-based bar would have produced. The bar and the
  interval are the same inference twice.

⚠️ **What is genuinely lost by skipping it.** Both arms of a paired run share a
fixed LightGBM seed, so the clustered CI captures sampling variation across
items and dates but **not** the design's own seed-to-seed jitter. A seed-varied
MDE is the only thing that would reveal whether these intervals are falsely
precise. At 3d/7d the pre-registered figures (0.54pp, 0.74pp) are *far* wider
than the fixed-seed half-widths measured there (0.014–0.017pp), which is direct
evidence that seed jitter dominates sampling error at those horizons. Assume the
same holds at 14d/30d: **treat the 30d interval as optimistically narrow, and
the 30d verdict as "no evidence of harm" rather than a bounded guarantee.**

The 14d conclusion is robust to this, because 3.14pp is more than 2× the widest
noise floor measured anywhere in this exercise.

⚠️ Note also that had it been run, it would **not** be the same construction as
the 3d/7d bars: those derive from the *old* design's noise floor, these from the
*new* design's. The four rows could never have been read as one uniform table.

## Training time — measured, warm vs warm

Both runs: 10-core Mac, `SKIP_REGIMES=1`, `--train-only`, Optuna cached in both
(`optuna: 0.0s (skipped - cached HP)`), so the comparison holds HP search
constant as the Task 4b baseline requires.

| Phase | Before (2026-08-04) | After | Change |
|---|---|---|---|
| p10 + p90 boosters (24 models) | 223.2s | **0s** | deleted |
| q50 boosters (12 → 4 models) | 87.7s | 12.9s | −85% |
| Directional classifiers (4, kept) | 19.9s | 15.2s | ~flat |
| Optuna | 0.0s (cached) | 0.0s (cached) | held constant |
| CV / conformal / feature-eng / saving | 50.4s | **~148.6s** | **+195%** |
| **TOTAL** | **381.2s** | **176.7s** | **−54%** |

### The bottleneck moved; it did not disappear

**The spec projected ~35–70s total and that projection was wrong.** Booster
fitting did land where predicted — 330.8s → 28.1s, a 92% cut — but the spec
counted only booster fitting. The out-of-fold conformal CV added at Task 8 (9
folds per horizon, fitting a median model per fold to generate the residuals q̂
is calibrated on) now costs ~148.6s and is **~84% of all training time**.

So the honest framing: the rewrite bought a 54% cut, not 85%, and the cost centre
is no longer the model but its calibration. The next real lever is the number of
calibration folds — q̂ is currently fitted on n≈18,400–21,600 OOF residuals per
horizon, far more than a stable quantile estimate needs. That was not in scope
here and is not yet measured.

A cold run (no cached HP) measured **250.1s**, of which Optuna was 39.4s. The
first post-rewrite run is necessarily cold, because Task 10's artifact-version
check correctly refuses the pre-rewrite `meta.json`.

## Band coverage — and an honest gap

| | Before | After |
|---|---|---|
| Construction | 24 p10/p90 quantile GBMs | split conformal around 1 median model |
| Nominal | 80% | 80% (`ALPHA = 0.20`, pinned) |
| Empirical | **39–48%** (documented, `model-optimization.md`) | **not measured on production data** |

⚠️ **The post-rewrite figure is a gap, not a number.** q̂ is chosen so that
coverage on the OOF calibration pool equals 80% *by construction*, so quoting
that would be tautological. What justifies the band is split conformal's
distribution-free guarantee, and `tests/test_conformal.py` verifies held-out and
conditional coverage — but on synthetic data. **No realized production coverage
measurement exists yet.** Training logs q̂ and the target, never the achieved
rate.

**The gate cannot fill this gap either.** Its `IntCov` for arm B (83.3–85.2% vs
arm A's 74.5–84.1%) looks like an improvement and must not be quoted: the gate
builds low/high from three quantile boosters that production **no longer has**,
using default (untuned) p10/p90 params because the new `meta.json` holds only
q=0.5, and it never touches `models/conformal.py`. That number describes a design
which does not exist.

## ⚠️ The first arm run was invalid — universe drift

Recorded because the failure mode is silent and will recur.

The first run of all four arms was discarded. `_load_parquet_items` ordered by
`row_count DESC` with **no tie-break**, so each run drew its own 60-item
universe: arms A and B differed by 2 items, arm D by 1, giving only **96.66%**
key overlap. That is not a 3.3% data loss. The gate's `feature_cols` falls back
to every numeric column — `forecaster.feature_cols` is empty, since the gate
never trains or loads — which pulls in cross-sectional features computed *across*
the universe, and the >0.95 correlation prune is data-dependent. Swapping two
items perturbs every row of every fold.

It produced a plausible-looking lie: **−3.696pp at 7d**, CI entirely below zero,
which would have failed the pre-registered bar and been written up as a real
regression caused by the rewrite. It cannot have been: at 7d both arms run
byte-identical classifier configs. `paired_da_difference` raises only on **zero**
overlap and is silent on partial overlap, so nothing flagged it.

Fixed in `b1bab03` (`ORDER BY row_count DESC, item_slug`), with a test. **Check
the overlap fraction before quoting any paired number.**

Arm C also died silently in that run, ~5 minutes in, on `Input X contains
infinity`: `price_log` is `log(price)` and the archive holds zero prices, so
`-inf` survives the gate's `fillna(medians)` (fillna does not touch infinities).
LightGBM tolerates it; `StandardScaler`/`Ridge` do not. The baseline simply
dropped out of the comparison while the other three completed. Also fixed in
`b1bab03`.

## A serving bug the rewrite exposed

`N_ENSEMBLES = 1` broke model loading, and no test or train-only run could see it.
`train()` always stores `self.models[(h, q)]` as a **list**, so `save_models`
writes `lgb_3d_q50_e0.txt` whatever the member count. `load_models` branched on
`n_ensembles > 1` and read an unsuffixed `lgb_3d_q50.txt` in the single-member
case — a filename `save_models` never emits. Result: `self.models` **empty**
after a load, so `--predict-only` would have served nothing.

`--train-only` cannot catch it (boosters are still in memory), which is why the
regression test round-trips a real `Booster` through disk. Fixed in `1902aab`;
the warm retrain now logs `Loaded 4 model groups`, which would have read 0.

## What this gate can and cannot say

Per `backend/AGENTS.md`, `walkforward_backtest.py` does not use
`fetch_price_history`: it skips multi-source voting, collapses the archive's
1.37× duplicate item-days with a plain mean where production serves an
outlier-voted median, and runs a different item universe. Its absolute DA
(59–69%) is **far above** production's 46.7–50.8% and the two are not
comparable. Every figure here is valid for **arm-vs-arm comparison only**.

The gate also trains on a wider feature set than production: `feature_cols`
falls back to every numeric column, so cross-sectional and event features that
`FEATURE_GROUP_ALLOWLIST = ["price_technicals"]` excludes from production are
present here. Equal across arms, so it does not bias the comparison, but it is
another reason these are not production numbers.

## Part 3 — item-coverage reinvestment: measured, then declined

**Date:** 2026-08-05. **Spec:** Part 3 / plan Task 13.

Part 3 existed to spend the wall-clock Part 2 freed on training over more items,
on the reasoning that "fewer models over more items is plausibly *more* accurate,
not merely faster." The plumbing bug is fixed; the reinvestment is **declined on
measured cost**, at the user's direction, with accuracy left untested.

### The bug was real

`forecast_prices.py` passed `train(max_rows=700_000)`. That value never reached
`build_training_data`, which kept its own `max_feature_rows=100_000` default. So
the caller's 700_000 was a **no-op that read as if it were doing something** —
the single most expensive kind of dead parameter, because every reader assumed
coverage had already been raised.

### Two budgets, not one

They are now separate named parameters, deliberately not unified:

| Parameter | Applies | Controls | Value |
|---|---|---|---|
| `max_feature_rows` | *before* feature engineering | how many whole item histories the model learns from | **100_000** (`TRAIN_FEATURE_ROWS` overrides) |
| `max_rows` | *after*, per horizon | each horizon's training slice | 700_000, unchanged |

Feeding `max_rows` to both — the obvious one-line fix — is wrong twice. It would
silently raise production coverage 6.5× (the caller already passes 700_000), and
it would make one number move two things, so the per-horizon cap would begin
binding at the same moment coverage changed. `test_the_per_horizon_cap_does_not_leak_into_coverage`
guards exactly this.

### What a row budget actually buys

Measured against the real archive, not estimated. The pool is **5,377 items /
5,823,319 rows** after the backfilled filter (5,542 items from prod Postgres), the
dead-item filter and 165 corrupt items — at **1,083 rows per item**.

| Budget | Items | Rows | % of pool | Training | vs 100k |
|---|---|---|---|---|---|
| **100_000** | **99** | 116,111 | **1.8%** | **104.6s** | — |
| 400_000 | 372 | 410,780 | 6.9% | 232.3s | 2.2× |
| 700_000 | 646 | 707,913 | 12.0% | 468.7s | 4.5× |
| 5_823_319 | 5,377 | 5,823,319 | 100% | not run | — |

**The decision, and the number that drove it:** 700_000 costs **468.7s**, which is
*more* than the **462s** the pre-rewrite 40-model grid cost. Raising coverage to
the value the spec intended would hand back the entire minimal-model saving to buy
12% of the item pool. Kept at 100_000.

Timings are warm-to-warm with cached Optuna params and a warm voted cache, so they
are lower than the 176.7s headline above and comparable only within this table.
The sweep rows moved both budgets together; the shipped decoupled default was
re-verified separately at **99/5,377 items in 102.5s**, confirming zero behaviour
change.

### Corrections to the spec's figures

The spec's Part 3 said "133 of 7,879 items". Measured today: **99 of 5,377**. The
pool is smaller than the spec assumed (5,377, not 7,879) and rows-per-item larger
(1,083, not ~748), so the same budget buys fewer items than estimated. The spec's
claim that "the budget can rise substantially inside the original wall-clock" does
not survive measurement.

### Where the cost goes as the budget rises

Conformal CV dominates but its *share* falls — 75.8% of training at 100k, 70.6% at
400k, 61.3% at 700k — because the q50 ensemble degrades faster than linearly
(3d q50: 2.5s → 62.4s, a **25×** jump for 6.1× the rows). So "cut CV folds" would
not have rescued the raised budget on its own.

### Accuracy is UNMEASURED — this is not a null result

No accuracy claim is made or implied for a raised budget. Nothing here shows that
more items would fail to help; it was not tested. Two reasons, and the second is
the one that matters for anyone who picks this up:

1. The shipped change has **zero behaviour change** — 100_000 effective before,
   100_000 explicit after — so there are no two arms to pair and the
   pre-registered bar is satisfied structurally.
2. **The walkforward gate cannot measure this knob at all.**
   `walkforward_backtest.py` has no `build_training_data` and no
   `_stratified_item_subsample`; its universe is
   `_load_parquet_items(con, backfilled_only=False)[:max_items]` (`:309`). So
   `TRAIN_FEATURE_ROWS` never reaches it, and running "arm B at 700k vs arm B at
   100k" as the plan instructed would have produced two **byte-identical** arms and
   a ~0.00pp difference that looked like a clean pass.

Raising `--max-items` instead is *not* a substitute. It perturbs the shared items'
features through `_add_cross_sectional_features`, changes which features survive
the data-dependent >0.95 correlation prune, and past ~185 items trips
`MAX_TRAIN_ROWS = 200_000`, whose `.tail(MAX_TRAIN_ROWS)` truncates the **training
window** — so it would measure shorter history, not wider coverage. This is the
same class of confound that produced the fake −3.696pp above.

**Measuring this honestly needs a rig that does not exist yet:** a fixed scored
cohort with a varying trained universe. `_stratified_item_subsample` also draws
non-nested samples across budgets (`RandomState(seed)` re-sampled per rarity group
at each `k`), so even the production CV path cannot pair 99 items against 646
without changing the selection rule. That is its own spec.

## Open follow-ups

0. **Whether item coverage buys accuracy is open, not closed.** Part 3 declined it
   on cost alone. If it is ever revisited, the blocker is the rig described above,
   plus nested subsampling so budgets are comparable — not another timing run.
1. **Calibration CV is the new bottleneck** (~84% of training). Cutting folds or
   subsampling the OOF pool is the next lever; unmeasured.
2. **Realized band coverage is unmeasured on production data.** Neither the
   training path nor the gate can supply it.
3. **The quantile/ensemble collapse is unmeasured by this gate**, and cannot be
   measured without driving the gate's quantiles from `ItemForecaster.QUANTILES`
   and building its band via `models/conformal.py`. Argued safe for served DA on
   structural grounds; the effect on the served **median price** is untested.
4. **38 stale artifacts** (32 orphaned boosters + 6 residual `.pkl`) remain in
   `models/saved_models`. Gitignored and provably inert — `load_models` reads
   `range(n_ensembles)` and a test guards it — but `save_models` purges only
   orphaned *regime* files. Deliberately left: a purge would let a partially
   failed retrain delete still-good artifacts.
