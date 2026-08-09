# Breadth re-measured at the shipped 1.2M budget: the 30d penalty was the early-stopping leak, and the fixed-budget argument is void

**Date:** 2026-08-09
**Change:** `backend/scripts/ab_test_training_breadth.py` only (uncommitted, branch
`feat/price-history-import`). No production code, no retrain, no knob moved.
**Re-measures:** `docs/changelog/2026-08-06-breadth-beats-depth-item-age-does-not.md` § breadth
**Bears on:** `docs/changelog/2026-08-06-data-acquisition-ranking.md` Tier-1 item 1 and its
`atalantus` rejection

## Why re-run something that was already measured

The breadth result that caps the value of the Steam listing-page backfill — "+1.18pp at 14d,
saturating by ~350 items" — was measured at a **200K per-fold budget** standing in for a
production `TRAIN_FEATURE_ROWS` of **100K**. On 2026-08-08 production shipped the $1 floor with
`TRAIN_FEATURE_ROWS = 1_200_000` and no subsample at all
(`2026-08-08-training-price-floor-shipped.md`). The ceiling constraining a Tier-1 acquisition
decision was therefore measured under a configuration that no longer exists.

## At 1.2M the budget does not bind, so this is a different experiment

On the rebuilt frame — 1,436,114 rows, 870 deep ≥$1 items, 33 features after correlation
pruning — **zero of 26 folds exceed 1.2M training rows in any arm at any horizon**:

| arm | items | per-fold rows, median | per-fold rows, max |
|---|---:|---:|---:|
| narrow | 150 | 137,948–140,580 | 234,609 |
| mid | 350 | ~352,924–359,184 | 579,219 |
| wide | 700 | ~678,896–690,807 | **1,120,798** |

(median ranges span the four horizons.) Confirmed empirically rather than by arithmetic: the
`wide_unbudgeted` arm returned **identical** results to `wide` at every horizon — same DA to two
decimal places, same rows per fold to the row.

So `_stratified_sample` returns every row untouched, the three "budgeted" arms degenerate into
the unbudgeted one, and the contrast stops being *"the same rows spread over more items"* and
becomes *"more items **and** more rows"*.

**That is production's actual regime at 1.2M, so it is the right question to ask now — but it is
not the question the 2026-08-06 run answered, and the two series must not be read as one
curve.** Every figure below describes the second question.

The corollary matters more than the numbers. `target_items = budget / rows_per_item` no longer
describes anything at 1.2M, because nothing is being divided. The arithmetic that put the Steam
backfill in Tier 1 (`target_items` 521 → 758) and rejected
`atalantus/buff-price-history-archive` (521 → 426) is **void at the current budget** — not
refuted, inapplicable.

## The harness early-stops on the window it scores, and that inflates DA by 1.5–2.7pp

`ab_test_training_breadth.py` passed the fold's validation window as `valid_sets`, stopped on it
via `lgb.early_stopping(15)`, and then reported directional accuracy on those same rows. That is
the pathology production removed on 2026-08-08 (`models/forecaster.py`, `FIXED_BOOST_ROUNDS`),
plus a direct selection leak on the scored rows.

Measured, early stopping minus fixed rounds, same folds and same arms:

| | 14d narrow | 14d wide | 30d narrow | 30d wide |
|---|---:|---:|---:|---:|
| early stopping | 58.34% | 59.45% | 52.56% | 51.85% |
| fixed rounds | 55.87% | 56.77% | 50.27% | 50.31% |
| **inflation** | **+2.47** | **+2.68** | **+2.29** | **+1.54** |

The mechanism is not specific to this script: the whole `ab_test_*` family early-stops on its own
fold window, so their stored results are inflated on the same channel. Only `training_breadth` was
measured, so the size elsewhere is unknown.

## The curve, under both trainers

Two runs, both at a 1.2M budget. **Run A** leaves early stopping untouched — the literal re-run of
the 2026-08-06 design at the new budget. **Run B** uses production's `CV_FIXED_BOOST_ROUNDS`
(14d 100, 30d 750). Paired against the 150-item arm with `paired_da_difference(cluster_key="fold_id")`,
held-out items, 25–26 folds.

**Run A (early stopping — inflated, all four horizons):**

| h | DA 150 / 350 / 700 | mid vs narrow | wide vs narrow | MDE (mid / wide) | verdict |
|---|---|---|---|---|---|
| 3d | 53.47 / 54.46 / 54.09 | +0.99 [−0.20, +2.98] | +0.63 [−0.27, +1.96] | 1.59 / 1.12 | null |
| 7d | 52.18 / 52.13 / 52.16 | −0.05 [−0.65, +0.50] | −0.02 [−0.62, +0.54] | — / 0.58 | null |
| 14d | 58.34 / 59.15 / 59.45 | +0.82 [−0.13, +2.26] | **+1.11 [+0.12, +2.58]** | 1.20 / 1.23 | positive |
| 30d | 52.56 / 52.84 / 51.85 | +0.28 [−0.80, +1.82] | **−0.71 [−1.37, −0.12]** | 1.31 / 0.62 | negative |

7d mid's MDE was not recorded.

**Run B (production fixed rounds — 14d and 30d only):**

| h | DA 150 / 350 / 700 | mid vs narrow | wide vs narrow | MDE (mid / wide) | verdict |
|---|---|---|---|---|---|
| 14d | 55.87 / 56.53 / 56.77 | +0.66 [−0.11, +1.56] | **+0.90 [+0.05, +1.96]** | 0.83 / 0.95 | positive |
| 30d | 50.27 / 50.56 / 50.31 | +0.30 [−0.56, +1.15] | +0.04 [−0.86, +0.97] | 0.86 / 0.92 | null |

**The reversal is the headline: run A's significant −0.71pp at 30d does not survive the trainer
fix.** It was the early-stopping leak interacting with arm size — a wider arm's early stopping
behaves differently on the same scored window — not a real cost of breadth. 14d's positive does
survive, at +1.11 → **+0.90pp**, which sits essentially *at* its own 0.95pp measurement floor.

## Invariant 4 changes the reading of both horizons

Constant call and realised down rate, under fixed rounds:

| h | best constant call | model DA 150 / 350 / 700 | edge over the constant call |
|---|---|---|---|
| 14d | **down, 55.41%** | 55.87 / 56.53 / 56.77 | **+0.46 / +1.12 / +1.36pp** |
| 30d | **up, 52.63%** | 50.27 / 50.56 / 50.31 | **−2.36 / −2.07 / −2.32pp** |

At 14d, breadth roughly **triples** the model's edge over the trivial strategy — and that edge is
+1.36pp at its best. At 30d the model **loses to a fixed "up" call at every breadth level**, and
no amount of breadth closes it.

Pesaran–Timmermann reads `skill` at every horizon and every arm (t = 6.43 to 17.83 across both
runs). Both statements are true at 30d simultaneously, and the reason is what PT tests: excess
over the *independence* null, not whether the model beats the trivial strategy. Reading PT alone
would have called 30d a success; this is exactly the failure mode invariant 4 exists to surface.

## Verdict on the Steam listing-page backfill

- **The accuracy case is weak.** One horizon, +0.90pp, against a 0.95pp MDE, on held-out-item CV.
  Do not run 12–17h of rate-limited scraping for accuracy.
- **The fixed-budget breadth argument that made it Tier 1 is void** at 1.2M — the budget it
  divides no longer binds.
- **The coverage argument is untouched** by this measurement and is now the whole of the honest
  case: a training pool that resembles what `predict()` scores. It remains **unquantified**; this
  run measures accuracy on a fixed universe and says nothing about it.
- **Two items outrank it now:** fixing the early-stopping leak across the `ab_test_*` family, and
  30d losing to a constant call under honest training.

## What changed in the harness

`backend/scripts/ab_test_training_breadth.py`, uncommitted:

- `--row-budget` overrides `ROW_BUDGET`, with a docstring note in `run_evaluation` that above
  ~1.2M it stops binding on this universe and the experiment silently changes question.
- `--fixed-rounds` bypasses early stopping entirely (no `valid_sets`, no callback).
- Per-row records now carry `actual_direction` / `predicted_direction` in
  `backtest/directional_test.py`'s label vocabulary. They previously stored only
  `direction_correct`, which made invariant 4 **unsatisfiable from the saved artifact**.
- Per-arm `constant_call` / `constant_call_accuracy`, `realised_down_rate` and
  `pesaran_timmermann` are computed in `run_evaluation`, deliberately **outside** `records`, so
  they survive `--out`'s record-stripping into the JSON.

## What was deliberately not done

- **No retrain, no production knob moved, no accuracy claim about the served model.** Nothing was
  written to `models/saved_models/` or `price-archive/`.
- **Run B was not extended to 3d and 7d.** Those two horizons exist only as run A
  (early-stopping) numbers and are therefore inflated by an amount this pass did not measure.
  Do not quote them as the fixed-rounds curve.
- **The round counts were not tuned for this harness.** Run B borrows production's
  `CV_FIXED_BOOST_ROUNDS`, calibrated at learning rates of 0.0053–0.01, while this harness runs
  `learning_rate: 0.03`. That is a defensible fixed schedule, not a matched one, and the fixed
  arm's absolute DA should be read with that in mind.
- **The other `ab_test_*` harnesses were not fixed or re-run**, even though the same leak is in
  all of them. That is a separate change with its own re-derivation cost.
- **The coverage argument was not quantified.** It would need an instrument that varies what the
  training universe *is* relative to the served cohort, which this harness — fixed 870-item deep
  ≥$1 universe, held-out evaluation items — cannot do.
- **The old 200K/100K series was not re-run for comparison.** It measures a configuration
  production no longer uses; reproducing it would answer a retired question.

## What these numbers are not

Held-out-item CV on the 870-item deep ≥$1 universe, 26 folds over ~523–546 dates per horizon.
**Not production DA**, not comparable to the pipeline's ≥$1 tier figure, and not comparable to
`walkforward_backtest.py` gate numbers. The evaluation items are in no arm's training set, so the
absolute level is a generalisation-to-new-items score.

## Docs updated in this pass

- `docs/changelog/2026-08-06-data-acquisition-ranking.md` — amendment banner: the Tier-1 breadth
  arithmetic and the `atalantus` rejection both rest on a budget that no longer binds.
- `docs/references/data-inventory.md` §3 — the same pointer, where the `target_items` arithmetic
  is restated as the reason for the depth-cliff verdicts.

## Still open

- **The early-stopping leak in the rest of the `ab_test_*` family.** Every stored A/B verdict in
  this repo is inflated on this channel by an unmeasured amount, on top of already predating the
  2026-08-08 statistics fix.
- **30d loses to a constant "up" call at every breadth level under honest training.** Breadth is
  not the lever; nothing here identifies what is.
- **The coverage half of the Steam backfill case.** Unquantified, and now the only argument left
  for the source.

## Related

- `docs/changelog/2026-08-06-breadth-beats-depth-item-age-does-not.md` — the 200K-budget run this
  supersedes for the current configuration
- `docs/changelog/2026-08-06-data-acquisition-ranking.md` — the ranking whose Tier-1 arithmetic
  this voids
- `docs/changelog/2026-08-08-training-price-floor-shipped.md` — the $1 floor and the 1.2M budget
  that make the fixed-budget question moot
- `docs/changelog/2026-08-09-shipped-retrain-cost-measured.md` — notes that all 15 A/B harnesses
  still early-stop on their own fold windows; this entry puts a size on it for one of them
- `docs/changelog/2026-08-07-pesaran-timmermann-headline.md` — why PT is the headline and why it
  does not settle "beats the trivial strategy"
- `.claude/rules/ab-statistics.md`, `.claude/rules/backtest-scoring.md`
