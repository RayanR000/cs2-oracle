# Model review: training, evaluation, and cost

Measured 2026-08-08 on `feat/training-price-floor` (926 items / 984,615 rows, the
shipped ≥$1 config). Every number below was produced in this review; nothing is
carried over from a stored result.

Instruments (scratch, not committed): a CI-equivalent cold retrain with per-phase
timing; a constant-call/PT audit over the CV folds; an OOF prediction dump scored
for rank IC; an early-stopping on/off contrast; and an archive-level test of the
reversal signal against source composition.

---

> **Status 2026-08-08, same day: items 1-6 of the Recommended order are implemented.**
> Early stopping is replaced by calibrated fixed rounds (`FIXED_BOOST_ROUNDS`),
> `n_jobs` is `-1`, the CV diagnostic classifier is gated off in CI, and
> `_cv_evaluate_horizon` now emits the invariant-#4 trio plus rank IC and the
> `-return_1d` baseline. Measured effect: every degenerate booster is gone
> (clf_14d 3 -> 450 trees) and out-of-fold rank IC rises 23-88% at 7/14/30d.
> **The cost projection in this document was wrong** — training rose rather than
> fell. Fixing early stopping necessarily buys rounds, and the old configuration
> was cheap by not training. Items 7-9 remain open.
>
> **Cost re-measured 2026-08-09** (uncontended, shipped constants verified from
> the boosters' tree counts): the weekly retrain is **872s**, not the 1098s first
> reported — that run used 7d=1000 rounds, which never shipped. So early stopping
> cost **487s -> 872s, +79%**, not +125%. See
> `docs/changelog/2026-08-09-shipped-retrain-cost-measured.md`.

## Summary

The model has real cross-sectional signal. Three separate defects stop it reaching
the product, and the largest cost centre buys nothing that is served.

1. **The offline metric is base-rate dominated and never applies the project's own
   invariant #4.** `_cv_evaluate_horizon` reports persistence and momentum baselines
   only; `constant_call_baseline`, `realised_down_rate` and `pesaran_timmermann` all
   exist in `backtest/` and none is called. 51% of fold-to-fold variance in
   `classifier_accuracy` is explained by the fold's realised direction mix.
2. **Early stopping against the trailing validation window destroys 23–88% of the
   signal** at 7/14/30d and produces degenerate boosters — 9 of 33 CV folds stop at a
   single tree, and the served 14d classifier fits 3 trees.
3. **A one-line baseline beats the 33-feature model at every horizon.** Ranking items
   by `-return_1d` scores a higher rank IC than the model at 3d, 7d, 14d and 30d.
4. **~45% of the weekly retrain is not served.** ~32–37% trains a throwaway classifier
   per CV fold to populate a `meta.json` diagnostic; ~11% trains regime models that
   only CI produces. ⚠️ **The 32–37% was measured under early stopping and is now
   low by half** — re-measured 2026-08-09 at **52% of a classifier-on retrain**
   (872s gated off vs 1804s on). Early stopping was collapsing the classifier to ~3
   trees; fixed rounds train it full-length at 3 trees per round. CI gates it off.
5. **The signal is partly a quoting artifact.** The reversal effect falls from rank IC
   0.168 to 0.101 once source composition is held stable, and to ~0 on the cleanest
   subset available.

Fixing 1, 2 and 4 makes the model simultaneously **lighter (487s → ~250s)** and **more
accurate**. 3 and 5 decide whether the modelling approach is worth continuing at all.

---

## 1. Where the training time goes

CI-equivalent cold retrain (`--train-only`, no `SKIP_REGIMES`, 10-core Mac):
**487s training + ~59s data build.**

⚠️ **This table describes the early-stopping config that this review replaced.** For the
shipped fixed-rounds config, re-measured 2026-08-09 at **872s**, see
`docs/architecture/model-optimization.md` → "Where the time goes now". The shares moved:
conformal CV 50.4%, q50 ensemble 18.2%, regime models 10.9%, direction classifier 10.6%.
Note also that per-phase timings on this machine swing ±25% between clean runs at identical
rounds, so the rows below carry less precision than they appear to.

| Phase | Seconds | Share | Served? |
|---|---|---|---|
| Conformal CV (33 folds over 4 horizons) | 260.8 | 53.6% | partly |
| Remainder (targets, splits, medians, save) | 73.8 | 15.2% | yes |
| Regime models | 54.1 | 11.1% | CI only |
| q50 ensemble | 53.8 | 11.0% | yes |
| Direction classifier | 22.6 | 4.6% | yes |
| Optuna | 21.9 | 4.5% | yes |

**The CV phase fits two boosters per fold.** Only the q50 predictions become
`oof_records` → conformal `q_hat` and the confidence thresholds. The 3-class
direction classifier fitted beside it feeds `classifier_accuracy` /
`classifier_accuracy_ge1` in `cv_results`, which is logged and serialised to
`meta.json` and read by nothing else in `models/`, `api/` or `scripts/`. Multiclass
builds 3 trees per round at the same 200 rounds; benchmarked at fold shape it is
**69% of a fold's fit cost** — so **~155–180s, 32–37% of the whole retrain, computes a
reported number.** ⚠️ **Superseded: measured directly on 2026-08-09 at 932s, 52% of a
classifier-on retrain.** The benchmark above was run under early stopping, which collapsed
the classifier to ~3 trees.

**Regime models are trained only in CI.** The model-cache restore step is
`if: mode == 'predict-only'`, so the Monday `mode=full` run is always cold,
`_warm_retrain` is False, and CI never sets `SKIP_REGIMES`. Local retrains yield
`trained_regimes: []`; the deployed artifact carries `['bull','range','bear']` and 7
regime boosters, and `predict()` uses them. The served model therefore depends on
where it was trained.

### Corrections to `docs/architecture/model-optimization.md`

- It reports conformal CV at 84% and booster fitting at 16%. That was the 100K
  config; at the shipped config it is **54% / 16%**.
- Its #1 recommended lever ("stop computing discarded feature groups") assumes a 35s
  feature-engineering phase. **Measured: 7s.** The lever is worth ~1% and belongs last.
- It states `meta.json` carries `trained_regimes: []`. The deployed artifact carries
  three regimes.
- `n_jobs = max(1, cpu_count // 2)` (`forecaster.py:3866`) is vestigial from the
  parallel-ensemble code deleted 2026-07-21 and contradicts the comment above it. It
  is worth less than it looks: benchmarked at frame shape, 1→10 threads is only
  **1.55×** (memory-bandwidth bound), so ~25% of the booster-fit phases.

---

## 2. The offline metric does not measure skill

`_cv_evaluate_horizon` reports `persistence_accuracy` (the flat-call share) and
`momentum_accuracy`. Invariant #4 requires DA to be quoted beside
`constant_call_accuracy` and `realised_down_rate`, with Pesaran–Timmermann as the
headline. None of the three is computed offline, so every A/B result and every
`mean_classifier_acc_ge1` in the project's history is quoted against baselines far
weaker than the constant call.

Applying the project's own `backtest/directional_test.py` to the same folds:

| h | DA (≥$1) | constant call | PT excess | PT t | verdict |
|---|---|---|---|---|---|
| 3d | 50.07 | 46.16 (down) | +4.91 | 13.75 | skill |
| 7d | 49.02 | 49.08 (down) | +2.30 | 5.48 | skill |
| 14d | 50.33 | 51.36 (down) | +2.58 | 5.07 | skill |
| 30d | 53.65 | 49.96 (down) | +1.44 | 2.08 | **no_skill** |

Two readings, both true. PT confirms the predictions are **not** independent of
outcomes at 3/7/14d. But at 7d and 14d the model's raw accuracy is *below* the
always-down constant, so as a decision rule it adds nothing there.

Per-fold, the metric is mostly the base rate: `corr(fold model DA, fold constant-call
DA) = 0.715`, **R² = 0.51** over 33 folds. Fold 1 at 30d reads 88.2% classifier
accuracy against a 91.8% constant call. In 4 folds the model's accuracy equals the
constant call to one decimal — it emitted a single class for the entire window.

This is a known trap, not a local quirk; see ["When Directional Accuracy
Lies"](https://arxiv.org/abs/2607.12248) (2026) and [Pesaran &
Timmermann](https://econweb.ucsd.edu/~atimmerm/windowjef.pdf).

---

## 3. Early stopping is destroying the signal

Booster sizes in the fresh artifact:

| Model | Trees |
|---|---|
| `clf_3d` / `clf_7d` | 597 / 597 (hit the 200-round cap) |
| `clf_14d` | **3** (one boosting round) |
| `clf_30d` | 138 |
| `lgb_14d_q50` | **4** |
| `lgb_30d_q50_range` | **1** |

The deployed pre-floor artifact has 123 trees at `clf_14d`, so this is new with the
≥$1 universe. In the CV folds, **9 of 33 stop at a single tree and 13 at five or fewer.**

The cause is structural. Early stopping scores against `_build_production_split`'s
trailing 30-day window. At h=14 the last 14 days carry no label and frozen-run
voiding removes ~30% more, leaving on the order of a dozen distinct dates — and
because all items move together within a date (the project's own
"accuracy is clustered by forecast date" result), the *effective* sample size is
roughly that dozen, not the row count. Any real fitting looks like a val-loss
regression against a single different market period, so training halts at round 1.

**Measured fix.** Same folds, same features, early stopping replaced by a fixed 300
rounds — mean within-date rank IC on the ≥$1 cohort:

| h | with early stopping | fixed 300 rounds | ICIR before → after |
|---|---|---|---|
| 3d | 0.1813 | 0.1830 | 1.45 → 1.45 |
| 7d | 0.0951 | **0.1297** (+36%) | 0.75 → 1.17 |
| 14d | 0.0941 | **0.1155** (+23%) | 0.74 → 0.98 |
| 30d | 0.0489 | **0.0919** (+88%) | 0.34 → 0.69 |

This is a large accuracy gain that also removes work and removes the val-set
dependency. It is the single highest-value change in this review.

---

## 4. The model loses to `-return_1d`

Rank IC is the cross-sectional metric this product actually needs — the dashboard
serves a per-item forecast, so what matters is ordering items within a date. Measured
on OOF predictions, ≥$1 cohort, against the naive signal "rank by minus yesterday's
return":

| h | model IC (fixed rounds) | naive `-return_1d` | model wins? |
|---|---|---|---|
| 3d | 0.1830 | 0.1973 | no |
| 7d | 0.1297 | 0.1645 | no |
| 14d | 0.1155 | 0.1475 | no |
| 30d | 0.0919 | 0.1019 | no |

The model's predictions correlate **−0.63** with `return_1d` at 3d: it is a mean-
reversion rule, and a lossier one than the raw input. `return_1d` and `log_return_1d`
are 49% of 3d gain.

Context: an ICIR above 0.5 is considered good and 1.0+ excellent, so these are not
weak numbers in absolute terms — the problem is that the ML stack is subtracting from
its own best feature. Two structural reasons:

- **The objective is not the metric.** Quantile (median) loss on raw % returns targets
  a conditional median per row; nothing in training optimises within-date ordering.
  This is the case for a ranking objective — see
  [LambdaRankIC](https://arxiv.org/abs/2605.00501) and
  [Building Cross-Sectional Strategies by Learning to Rank](https://arxiv.org/pdf/2012.07149).
- **`price_tier` and `macd_missing` carry 11.8% and 12.1% of 30d gain.** A static tier
  and a missingness indicator are item-identity and data-availability, not dynamics.

Note the boosters here used default LightGBM params, not the Optuna-tuned production
set, so absolute IC may understate production. Both arms share params, so the
early-stopping contrast and the naive comparison are internally valid.

---

## 5. How much of the signal is a quoting artifact

> **REFUTED 2026-08-09. The composition rows of the table below do not mean what
> they say.** The partition classified every item-day whose `source` is NULL as
> "composition changed" — and `source` is NULL for **every archive row before
> 2026** (9,417,947 item-days, 0 with a label). So the "stable" cell contains no
> pre-2026 data at all: its 188 dates are 2026 dates, and ~716 of the 775
> "changed" dates are pre-2026 dates marked changed for want of a label rather
> than because anything changed. The measurement also applied no
> `_snapshot_dates` / `_collection_shift_dates` exclusion.
>
> **The +0.1676 → +0.1006 fall is a 2013-2025 → 2026 regime difference, not
> composition control.** Measured with the committed instrument (pre-`873148b`):
> restricted to 2026, the **all-rows** rank IC — no composition control of any
> kind — is **+0.1023 on 185 dates**, against this table's "stable" +0.1006 on 188.
>
> **Corrected answer (pre-`873148b`):** over 2026, with composition defined as
> the *set* of source names, the composition-stable cell is **+0.1027 on 181
> dates at 3d** against an unconditional **+0.1023** — and **+0.0842 vs +0.0842
> on 167 dates at 7d**. Holding composition still does not touch the signal, so
> the reversal is **not** a composition artifact and accuracy work is **not**
> gated on this. The one row that survives is the weakest: "stable & three
> agreeing sources" is 25 dates at 3d and 19 at 7d, both under the 30-date
> reporting floor.
>
> **The committed instrument no longer produces `+0.1023` / `+0.1027` /
> `+0.0842`.** Commit `873148b` (2026-08-09) removed
> `aggregator_steam_7d/30d/90d` from the consensus vote after this measurement
> was published; re-run post-exclusion, the same cells read +0.1011 / +0.1017
> (3d) and +0.0838 / +0.0838 (7d) — every delta ≤0.0015, same conclusion. See
> the "Post-exclusion re-run — after `873148b`" section of
> **`docs/research/2026-08-09-composition-stability.md`** for the full
> before/after tables.
>
> The numbers above are kept as published, not deleted. They are reproducible:
> the committed script reproduces 188 and 185 dates exactly when the
> NULL-never-equal rule is applied. See
> **`docs/research/2026-08-09-composition-stability.md`** for the corrected
> measurement and `backend/scripts/measure_composition_stability.py` for the
> instrument.

The reversal is large enough to be suspicious. The archive carries `source` per
item-day, so it can be tested directly. Voted daily series, ≥$1, 2024-01-01 onward,
universe rules applied; rank IC of `-r_t` predicting the forward 3-day return:

| Subset | Dates | rank IC | t |
|---|---|---|---|
| All rows | 932 | +0.1676 | 43.6 |
| ~~Source composition **stable** across t−1…t+3~~ (refuted) | 188 | +0.1006 | 14.1 |
| ~~Source composition changed~~ (refuted) | 775 | +0.1804 | 43.1 |
| ~~Stable & single source~~ (refuted) | 185 | +0.1081 | 14.0 |
| ~~Stable & **three agreeing sources**~~ (refuted) | 28 | +0.0044 | 0.1 |

75% of item-days are composition-stable, but only 188 dates have a full stable
window. Roughly 40% of the effect is associated with composition change — consistent
with the documented basis errors (the BUFF bid quoting ~11% low, and
`aggregator_steam_7d/30d/90d` being trailing-window means rather than point
observations). A component survives at +0.10, but the cleanest subset available —
three sources agreeing, none changing — shows **no reversal at all**, on 28 dates.

*(The paragraph above is the original reading and is what the 2026-08-09 work
refutes. "40% of the effect is associated with composition change" is an artifact
of the NULL rule; the "28 dates" cell is real but was and remains underpowered.)*

Pooled 1-day return autocorrelation is ~0 (−0.0001) whether or not composition
changes, so this is a purely cross-sectional effect, not Roll-style
[bid-ask bounce](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1984.tb03897.x)
in the time series.

~~**This is underpowered and it gates everything else.** If the effect is a quoting
artifact, no feature or architecture work can help, because the label is not a
tradeable return.~~

**The gate is lifted (2026-08-09).** The conditional was right and the antecedent
is false: composition change is not what the reversal is made of. What remains
open is narrower — whether the reversal survives *several independent sources
agreeing*, which is 25 dates at 3d and cannot be answered until more multi-source
days accumulate.

**Two things this conclusion leans on, stated here rather than left implicit.**
First, the "composition changed (present)" cell — item-days whose source set
genuinely differed with every window day observed, as opposed to a window with
a missing day — is *also* underpowered, at the same floor as the "≥3 sources"
cell: 25 dates at 3d, 19 at 7d. There is no dataset in this archive on which the
reversal's behaviour under a *known* composition change can be quoted — "holding
composition still does not touch the signal" is a stable-vs-unconditional
equality, not a measured stable-vs-changed contrast. Second, the stable cell is
not an independent check on the unconditional one: it is **93.7% of all rows**
(1,912,872 / 2,040,460, pre-`873148b`) and **77.0% of that is single-source
item-days** (1,472,932 / 1,912,872) whose composition cannot change by
construction. See `docs/research/2026-08-09-composition-stability.md` for the
full disclosure — this section now matches its candour rather than falling
short of it.

---

## Recommended order

**Free, no accuracy risk — do first**
1. Gate the per-fold diagnostic classifier off in CI (env flag, default on for
   research). −155–180s, −32–37%. Touches nothing served. **DONE; and the saving is
   twice what was predicted here — measured −932s, −52%** (2026-08-09).
2. Fix `n_jobs` to `-1`. ~25% of the booster-fit phases.
3. Correct `model-optimization.md` (four stale claims above).

**Accuracy, measured**
4. Replace early stopping on the production and CV fits with a fixed round count
   calibrated once. **+23–88% rank IC at 7/14/30d**, and it removes work.
5. Add `constant_call_accuracy`, `realised_down_rate` and PT to `_cv_evaluate_horizon`,
   and make rank IC the offline headline. Without this, item 4 is unverifiable.
6. Adopt `-return_1d` as the baseline every arm must beat. It is the honest bar and
   the model does not currently clear it.

**Decisions, not free**
7. Regime models: `SKIP_REGIMES=1` in CI (−11%) or justify them. The deployed set
   includes 1-tree and 3-tree boosters, so they are plausibly harming serving.
   **Re-measured 2026-08-09: 95.4s, still 10.9%** — the absolute cost rose with fixed
   rounds but the share did not, so this remains a ~1/9 lever and the case for cutting
   it rests on the degenerate boosters, not on cost.
8. Power up §5 properly (more dates, the 3-source subset). This gates any further
   feature or architecture work.
9. Only if §5 survives: try a ranking objective (lambdarank / rank-IC) and put the
   `cross_sectional` feature group back in the allowlist — the one measured positive
   structure in this archive (expensive tier leads cheap tier) lives in a group that
   is engineered on every row and then discarded.

Items 1–4 together take the weekly retrain from **487s to roughly 250s while
increasing signal**, which is the direction this project has not been able to move in
for several months.

⚠️ **That prediction was wrong, and the error was structural.** Items 1–4 shipped and the
weekly retrain went **487s → 872s** (measured 2026-08-09). Item 4 is the reason: early
stopping was not a cost that could be removed, it was the thing suppressing the cost. Any
estimate that treats "fix early stopping" as neutral-to-cheaper will be wrong the same way.
The project bought +23–88% rank IC for +79% training time.
