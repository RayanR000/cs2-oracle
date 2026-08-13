---
paths:
  - "backend/backtest/{paired_mde,walkforward_records}.py"
  - "backend/scripts/ab_test_*.py"
  - "backend/scripts/merge_*.py"
---

# A/B verdicts

**An A/B verdict is a fold-clustered interval, never a win count.** Ten harnesses decided
on a fold win-count, a ±0.5pp pooled-delta threshold or a bare `a > b` until 2026-08-08 —
none is a test, against a measured item-level MDE of 2.21–3.69pp.

- Use `backtest/paired_mde.py`: `paired_da_difference` for a hit rate,
  `paired_metric_difference(value_key=…)` for anything else (pass `higher_is_better=False`
  for a loss), and `paired_arm_contrasts` / `format_paired` / `verdict` for the reporting.
- Records come from `backtest/walkforward_records.py`: `paired_records` at row grain —
  always prefer it — and `fold_level_records` only where the rows are genuinely gone, as in
  the three pinball harnesses that shard folds across processes.
- `unresolved` (fewer than 2 shared clusters, or a non-finite bound) is **not** `null`.
- Strip `records` with `without_records` before printing. Under `--arm` sharding no shard can
  contrast in-process, and `merge_price_primitives_ab.py` pairs the shards at fold grain off
  `per_fold`.
- Only `NoPairedRows` is catchable — a missing `cluster_key` is a wiring bug and propagates,
  because swallowing it restores the 2026-08-07 under-dispersion bug behind a confident wrong
  message.

**A harness trains the model production trains, or its verdict describes nothing served.**
Rounds come from `ItemForecaster._boost_rounds(horizon, cv=True)` and the booster from
`ItemForecaster._train_ensemble_member(...)`, with `early_stopping=_early_stopping_enabled()`
— never `lgb.train` directly and never a private `num_boost_round`. Until 2026-08-13 all
thirteen training harnesses early-stopped on `valid_sets=[dval]` and then scored `X_val`, the
same rows, so the iteration was selected on the window the verdict was read from. `EARLY_STOPPING=1`
reproduces the old arm, which is what makes a re-read *paired*.

**The leak is worth ~2pp of pooled DA and it pays the *shuffled* arm most.** Measured on
`price_primitives` h=7, same frame and same 26 folds: pooled DA rises **+1.98 / +2.42 / +2.73pp**
(baseline / treatment / placebo), matching the breadth harness's +1.5–2.7pp. It is a **level
shift**, so that harness's contrast stays `null` under both trainers — but **placebo swings
+1.09pp** (−0.393 → +0.700) and takes the largest rise, so six shuffled columns buy as much as six
real ones. The leak rewards **capacity, not signal**, so 🔑 **an arm that measured *positive* under
early stopping is the suspect case**; null and negative results are the easy direction. ⚠️ One
harness, one horizon — don't carry the numbers. And **never size a defect of this shape
synthetically**: a 900-row iid frame returned −0.26pp (t = −0.40) for the same effect.
`docs/changelog/2026-08-13-the-leak-is-worth-two-points-and-it-pays-the-placebo.md`, correcting
`docs/changelog/2026-08-13-ab-harnesses-follow-productions-trainer.md`.

**Every A/B result stored in this repo predates all of it, and the defects are stacked.** A
verdict from before 2026-08-13 sits downstream of the embargo fix (2026-08-08), date-clustered
`paired_mde` (2026-08-07), the bid and trailing-window label changes (2026-08-07 / 08-09), the
≥$1 universe (2026-08-08) **and** the trainer — so re-running one harness settles one harness.
Three (`training_breadth`, `item_metadata`, `csfloat_basis`) were re-run on 2026-08-08 and all
three refutations survived, but those results live in a scratchpad, not here, and they predate
the trainer fix too. See
`docs/changelog/2026-08-08-migrated-archive-emptied-eight-harnesses.md`.
