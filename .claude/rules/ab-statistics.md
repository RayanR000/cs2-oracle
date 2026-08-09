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

**Every A/B result stored in this repo predates this.** Three harnesses
(`training_breadth`, `item_metadata`, `csfloat_basis`) were re-run on 2026-08-08 and all
three refutations survived, but those results live in a scratchpad, not here; the other
**ten have not been re-run**. See
`docs/changelog/2026-08-08-migrated-archive-emptied-eight-harnesses.md`.
