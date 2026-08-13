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
reproduces the old arm, which is what makes a re-read *paired*. ⚠️ **Its size is not
established** — the breadth harness measured early-stopping-minus-fixed-rounds at +1.5–2.7pp DA
on real folds, but that is two channels at once and neither reproduced synthetically (selection
−0.26pp, t = −0.40; trainer +0.11pp, t = +0.72). Don't quote a decomposition.
`docs/changelog/2026-08-13-ab-harnesses-follow-productions-trainer.md`.

**Every A/B result stored in this repo predates all of it, and the defects are stacked.** A
verdict from before 2026-08-13 sits downstream of the embargo fix (2026-08-08), date-clustered
`paired_mde` (2026-08-07), the bid and trailing-window label changes (2026-08-07 / 08-09), the
≥$1 universe (2026-08-08) **and** the trainer — so re-running one harness settles one harness.
Three (`training_breadth`, `item_metadata`, `csfloat_basis`) were re-run on 2026-08-08 and all
three refutations survived, but those results live in a scratchpad, not here, and they predate
the trainer fix too. See
`docs/changelog/2026-08-08-migrated-archive-emptied-eight-harnesses.md`.
