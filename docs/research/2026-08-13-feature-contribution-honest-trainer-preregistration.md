# Pre-registration: does removing cross-sectional features survive the honest trainer?

> **Status (as of 2026-08-21): SCORED 2026-08-13 — the founding number DOES NOT REPRODUCE; `C4`
> reopened.** `changelog/2026-08-13-feature-contribution-plus-3.5-does-not-reproduce.md`. The
> `+3.5pp @30d` for removing cross-sectional features — the entire basis of
> `FEATURE_GROUP_ALLOWLIST = ["price_technicals"]` — does not survive the honest trainer.
> **What happened next:** the feature set was pruned on a different, direct read — five
> dead-weight features were shelved from the served set after a **null** drop-5 ablation
> (commit `83f5e22`). A broad re-run of the nine repaired A/B harnesses was subsequently ruled
> **out** by `2026-08-19-deep-model-review.md` §11/§12: the A/B family was never powered
> (MDEs 1.15–7.13pp), so most stored "null" verdicts are **UNRESOLVED**, not null.


**Date:** 2026-08-13, written and committed **before** any re-run.
**Instrument:** `backend/scripts/ab_test_feature_contribution.py`, run paired against
`EARLY_STOPPING=1`. One small arm addition (the placebo, below) must be committed **before**
dispatch; nothing else changes.
**Opens/closes:** this settles the standing of `FEATURE_GROUP_ALLOWLIST = ["price_technicals"]`
on the honest trainer and the ≥$1 cohort. A pass does **not** license adding features back; a
fail reopens **C4** as a cohort problem, not merely a trainer problem.
**Traces:** `C4` in `docs/research/2026-08-09-next-steps.md`; the `C7` re-read programme in
`docs/research/2026-08-10-next-steps.md`; the leak sizing in
`docs/changelog/2026-08-13-the-leak-is-worth-two-points-and-it-pays-the-placebo.md`.

## The founding number, and everything wrong with it

`no_cross_sectional − full = +3.5pp @30d` (`docs/research/2026-07-19-feature-contribution-by-horizon.md`)
is the entire basis of the allowlist that removes `cross_sectional`, `events`, supply/rarity and
item-identity from **every** served horizon. It carries **four** stacked defects, and this re-read
is the first to strip all of them at once:

1. **Trainer leak.** The number early-stopped on `valid_sets=[dval]` and scored `X_val`, the same
   rows, under a trainer production abandoned 2026-08-08. The leak is worth ~2pp of pooled DA and
   **pays the higher-capacity arm most** — see the asymmetry argument below.
2. **Penny cohort.** Item selection is `ORDER BY row_count DESC LIMIT max_items` filtered to
   `source IS NULL OR source = 'STEAMCOMMUNITY'` — longest-history backfill items, **15 of 200 ≥$1**,
   median price **$0.059**, **41.0% of 3d returns exactly zero**, **31.27%** of scored rows free
   `sign(0)==sign(0)` hits.
3. **PT null read as skill.** "+3.5pp vs 50%" is the Pesaran–Timmermann null; a constant always-down
   call beats the model on every stored date (invariant #4).
4. **Un-embargoed.** No purge gap before 2026-08-08; every training row within `horizon` days of the
   boundary carried a label resolved from inside the validation window.

The harness already fixes (2), (3) and (4): the paired contrast `_paired_vs_full` scores on
**≥$1, non-flat rows** (`ab_test_feature_contribution.py:345-346`), the split now purges through
`_purge_overlapping_train_rows` at `embargo_days(horizon)`, and the paired fold-clustered interval
is a skill contrast, not a DA level. **What remains is (1) plus a power question the cohort forces.**
So the honest read is `_paired_vs_full["no_cross_sectional"]`, **not** the pooled `directional_accuracy`
that produced +3.5.

## The capacity-leak asymmetry — why this re-read can move either way

The leak rewards **capacity, not signal**: under early stopping, six shuffled columns bought as much
pooled DA as six real ones (`+0.700` vs `+0.614pp`, price_primitives h=7). `full` carries **more**
columns than `no_cross_sectional` (115 → 97 before the corr-prune), so the leak inflates `full`'s
level **more** than the reduced arm's. That biases `(no_cross_sectional − full)` **downward** under
the leak.

The old, leaky reading was already **+3.5pp positive**. If the leak was suppressing it, the honest
contrast is **even more positive** and the allowlist is vindicated. If instead the +3.5 was a
penny/PT/embargo artifact that survives on the pooled number but never existed on the ≥$1 paired
interval, the honest contrast collapses to **null**. These point in opposite directions, which is
what makes one dispatch information-rich — unlike the item_metadata re-read, where a null is
near-certain.

## Design

Shard to **h=30** (the horizon the allowlist rests on and the one whose claim is largest), two runs
on one commit, same `--max-items`, same frame:

- **Arm A — honest (default).** `_early_stopping_enabled()` off, `_boost_rounds(30, cv=True)` = 500.
- **Arm B — leaky control.** `EARLY_STOPPING=1`, byte-identical otherwise. Reproduces the old
  regime, so the honest-vs-leaky delta on the same contrast **is** the leak's contribution.

Each run emits all three feature configs (`full`, `no_cross_sectional`, `no_events`) and their
`_paired_vs_full` contrasts in one pass, so `no_events` (+3.0 @30d) comes free and is reported
beside the primary.

### The placebo arm — required, committed before dispatch

The leak's transferable lesson is that a **shuffled** manipulation buys as much as a real one, so a
column-**removal** harness must show that removing *these* columns beats removing *any* columns.
Add a fourth config, `no_random_k`, that drops a **random `k` columns** from `pruned`, with
`k = len(pruned) − len(subsets["no_cross_sectional"])` (the exact count cross-sectional removal drops
after the corr-prune), seeded `random_state = 20260813`. Commit it before either run. It is scored
by the identical `_paired_vs_full` path, so it pairs row-for-row with the other arms.

**This is the deciding placebo.** If `no_random_k − full` clears the same interval as
`no_cross_sectional − full`, the "+3.5" is generic capacity reduction (fewer columns → less
overfit), not evidence against cross-sectional features specifically.

## Legs and bars — fixed before any number

**(1) Primary — does the shipped decision survive?** `no_cross_sectional − full` on the honest arm
(A), ≥$1 non-flat paired fold-clustered interval at h=30.
**Bar: the interval excludes 0 and is positive (lower bound > 0).** `unresolved` (< 2 shared
clusters, or a non-finite bound) is **not** a pass and **not** a null — it is its own outcome (see
the cohort caveat). Report `format_paired` verbatim.

**(2) Placebo gate — is it cross-sectional-specific?** Same interval for `no_random_k − full` on arm
A. **Bar: leg (1) passes AND `no_cross_sectional − full` is strictly greater than `no_random_k − full`
by more than the width of the tighter interval.** If the random-removal arm clears leg (1)'s bar too,
the primary is **downgraded to "capacity, not cross-sectional"** regardless of its own interval.

**(3) Attribution — how much was the trainer?** `no_cross_sectional − full` on arm A minus the same
contrast on arm B (leaky), the two paired on the same rows. **Descriptive, no bar** — but the sign
and size are pre-committed as reportable whatever they are, so a null cannot be dropped. The
asymmetry argument predicts arm A ≥ arm B on this contrast.

## Predictions, before the run

**Pre-registered point prediction: leg (1) returns NULL or unresolved.** The +3.5 lived on the
pooled penny DA against the PT null; the ≥$1 paired interval strips the free `sign(0)` hits, the
level, and the hindsight baseline at once, and against a fold-clustered MDE of 2.21–3.69pp a genuine
+3.5pp effect is at the very edge of resolvable. I expect the interval to straddle 0.

**Secondary prediction: if leg (1) does pass, leg (2) fails it** — `no_random_k` removal helps nearly
as much, because the mechanism is capacity, not cross-sectional signal.

**Attribution prediction: arm A > arm B** on the contrast (the leak suppressed the positive delta by
inflating `full`), so the leaky number understated rather than manufactured it. This is the one leg
whose direction I am least sure of and it is the reason to run both arms rather than arm A alone.

## The cohort power caveat — a real possible outcome

Item selection is unchanged and structurally penny: longest-history backfill items skew cheap, so the
≥$1 fraction stays low even at large `--max-items` (15/200 at the default). **`--max-items` must be
raised** until the h=30 paired contrast has **≥ 2 shared clusters** — report the resulting ≥$1 item
count and shared-cluster count in the writeup. If the interval is still `unresolved` at the largest
`--max-items` that fits the 30-minute cap, **that is the finding**: the founding number cannot be
honestly re-derived on this harness's cohort, and C4 becomes "rebuild the cohort or drop the
allowlist," not "re-run the trainer." Do not paper over an `unresolved` as a null.

## Void conditions

- The two runs land on different commits, caches, `--max-items`, or fold counts.
- Any h=30 paired cell whose `n` differs between arm A and arm B (they must pair row-for-row).
- Reading the pooled `directional_accuracy` (the +3.5 basis) as the verdict instead of the ≥$1
  paired interval. The pooled number may be reported as descriptive **only** beside
  `realised_down_rate`, never differenced against 50% or `constant_call_accuracy`.
- The placebo arm absent, or its `random_state` not `20260813`, or `k` not matched to the
  cross-sectional removal count.
- `improvement_over_baseline_pp` quoted for anything — it differences against a hardcoded 50%
  (`ab_test_feature_contribution.py:386`), which is the PT null.

## Cost

Two runs × one horizon. The honest trainer is ~2.4× the leaky one (500 rounds vs the old ~100 cap),
and runtime scales with `--max-items` through the corr-prune and the per-fold fit. **Dry-time arm A
once at the chosen `--max-items` and confirm < 30 min before committing to the pair**; if it exceeds
the cap, hold h=30 and drop `--max-items`, never widen the horizon shard. Arm B is the same cost
minus the round budget (early stopping cuts iterations), so the pair fits whatever arm A fits.

## What a pass and a fail each mean

**Pass (leg 1 clears, leg 2 confirms specificity):** the allowlist's cross-sectional removal is
sustained on the honest trainer and the served cohort — the first time that decision has been
measured rather than assumed. It still does not license adding any feature group back; each removed
group is its own question.

**Fail (leg 1 null/unresolved, or leg 2 shows capacity):** production is excluding four feature
groups from every horizon on the strength of a penny-cohort, PT-null, un-embargoed, leaky number that
does not reproduce. The next step is not another feature experiment — it is rebuilding this harness's
cohort onto the ≥$1 served universe (the same rebuild C4 has needed since 2026-08-09) so the
allowlist can be decided on the population it governs.
