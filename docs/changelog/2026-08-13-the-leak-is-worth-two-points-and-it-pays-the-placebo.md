# The early-stopping leak is worth ~2pp of pooled DA, it pays the *shuffled* arm most — and the primitives stay shelved

**Date:** 2026-08-13
**Corrects:** `docs/changelog/2026-08-13-ab-harnesses-follow-productions-trainer.md`, which concluded
the leak's size "is not pinnable here" and left it open. **It reproduces on real folds.** That entry
is left as written, per the no-retroactive-edit rule; this is the correction.
**Instrument:** `scripts/ab_test_price_primitives.py`, the first harness re-read after the trainer
fix (`92a5687`). Two runs, same frame cache, same 26 folds, same 10,429 paired rows — **only the
trainer differs.** `--max-items 200 --horizon 7 --q50-only --n-jobs 3`.
**Verdict:** ✅ the magnitude is real and ~2pp. ❌ **it does not move this harness's verdict** — the
six primitives stay shelved. 🔑 **The leak's real damage is to the placebo arm.**

## The size reproduces, and my synthetic was simply underpowered

Pooled directional accuracy, 56,991 scored samples per arm:

| arm | fixed rounds (production) | `EARLY_STOPPING=1` | Δ |
|---|---:|---:|---:|
| baseline | 51.97% | 53.95% | **+1.98pp** |
| treatment | 51.01% | 53.43% | **+2.42pp** |
| placebo | 51.22% | 53.95% | **+2.73pp** |

`2026-08-09-breadth-curve-at-1p2m-budget.md` measured **+1.5 to +2.7pp** on a different harness.
This lands **+1.98 to +2.73pp** on this one. The two agree, and the earlier entry's synthetic
reproduction (selection −0.26pp t = −0.40, trainer +0.11pp t = +0.72, 40 seeds) was measuring
nothing — **a 900-row iid frame has no power against this, exactly as that entry suspected.** The
conclusion to carry forward is methodological: **do not try to size a fold-structured harness defect
on synthetic data.** Run the harness.

## It is a level shift, so the verdict survives

Paired, fold-clustered on `fold_id`, 26 clusters, 10,429 paired rows — **identical pairing in both
runs**, so this is the trainer and nothing else:

| trainer | arm | Δ vs baseline | 95% CI | MDE | verdict |
|---|---|---:|---|---:|---|
| fixed rounds | treatment | **+0.182pp** | [−1.164, +1.365] | 1.265 | `null` |
| fixed rounds | placebo | **−0.393pp** | [−1.426, +0.492] | 0.959 | `null` |
| `EARLY_STOPPING=1` | treatment | **+0.614pp** | [−0.610, +1.804] | 1.207 | `null` |
| `EARLY_STOPPING=1` | placebo | **+0.700pp** | [−0.592, +2.244] | 1.417 | `null` |

Every arm rises by about the same amount, so the *contrast* is `null` under both trainers. **The six
primitives stay shelved** — `vol_semidev_down_30d`, `vol_semidev_up_30d`, `vol_skew_30d`,
`rsi_divergence_7d`, `rsi_price_divergence_7d`, `macd_hist_slope_7d`. All six survive the 0.95
correlation prune (180 → 138 features), so the null is not the columns being pruned away.

## 🔑 The leak pays the shuffled arm most, which is the finding

**Placebo swings +1.09pp between trainers** — from **−0.393pp** to **+0.700pp** — and takes the
largest pooled rise of the three arms (+2.73pp). Under early stopping, **six columns of shuffled
noise buy as much as the six real ones** (+0.700 vs +0.614).

That is precisely what the placebo arm exists to detect. It is a same-width capacity control, and it
fires **only** under early stopping. So the leak does not reward signal, it rewards *capacity* — an
arm that widens the model gets to select an iteration on the rows it is scored on, and a shuffled
column is enough to collect. `ab_test_item_metadata.py`'s existing comment predicted this shape
("the leak is worth MORE to it than to baseline"); this measures it.

⚠️ **Consequence for the other twelve harnesses:** the arms most exposed are the ones that *add*
columns, which is most of them. An arm that measured **positive** under early stopping is the
suspect case, not an arm that measured null or negative. `price_primitives` measured negative and
survives; that is the easy direction.

## What this does not establish

- ⚠️ **One harness, one horizon.** The +2pp level shift and the placebo inflation are measured on
  `price_primitives` at h=7. Do not carry either number to the other twelve.
- ⚠️ **This is not a reproduction of the published −1.47pp** at 7d
  (`2026-07-31-price-primitives-decision-scale.md`). Neither trainer returns it: paired is +0.182 /
  +0.614, pooled `t−b` is −0.96 / −0.52. **Five other things changed underneath** — the embargo fix,
  `paired_mde` date-clustering, the bid and trailing-window label exclusions, the ≥$1 scoring — and
  the feature set grew 174 → 180. What is cleanly isolated here is fixed-rounds-vs-early-stopping on
  *today's* frame. The published refutation is not re-validated; its conclusion is re-reached.
- ⚠️ **Pooled and paired disagree in sign under both trainers** (pooled `t−b` −0.96 / −0.52 against
  paired +0.182 / +0.614). Composition across folds. Per `.claude/rules/ab-statistics.md` the
  fold-clustered interval is the verdict; the `SHIP DECISION SUMMARY` line the harness prints is
  pooled and must not be read as one.

## Two harness defects found while reading, neither fixed

- **`improvement_over_baseline_pp` is against a coin flip, not the `baseline` arm.**
  `ab_test_price_primitives.py:531` hardcodes `baseline_2class = 50.0`, so line 557 logs
  `DirAcc=51.0% (1.0pp above baseline)` for a number that is **0.96pp below the arm named
  `baseline`**. Pre-existing, affects no verdict, and a misreading waiting to happen — it also
  violates invariant #4, which names `realised_down_rate` as the runnable baseline and not 50%.
- **`EARLY_STOPPING=1` leaks LightGBM eval lines to stdout.** Production's
  `_train_ensemble_member` calls `lgb.early_stopping(50)` without `verbose=False`, so
  `[60] valid_0's quantile: 4.7662` appears mid-log. Only reachable under the flag; cosmetic.

## Cost

Fixed rounds **8m22s** for three arms against early stopping's **3m34s** — `_boost_rounds(7,
cv=True)` is 500 rounds where the old cap was 100 and stopping usually fired well under it. **The
honest trainer is ~2.4x slower**, which is the real budget line for re-reading the remaining twelve
harnesses. Frame cache build: 39s, 905,384 rows.
