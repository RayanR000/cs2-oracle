# Training breadth is accuracy-neutral at a fixed row budget; only raw volume helps, weakly, at 30d

**Date:** 2026-08-18

`ab_test_training_breadth` measured breadth-vs-depth (GH run `32168044697`) on
`DA(strict, >=$1)`, held-out items, ~25–26 folds, paired-vs-narrow with dates clustered. This
answers whether serving *more items* costs accuracy — the question the iflow serve-universe
expansion (`superpowers/specs/2026-08-17-iflow-serve-universe-expansion-design.md`) turns on.

## Result

- **Breadth is free at a fixed training-row budget.** narrow (150 items) vs mid (350) vs wide
  (700) are **NULL at 3/7/14/30d** — every CI spans 0, every effect is below its MDE. Spreading the
  same budget over more items costs zero accuracy.
- **Only raw volume helps, marginally, and only at 30d.** `wide_unbudgeted` (700 items **and**
  5.3× the rows) is **+0.88pp [+0.06, +1.80] at 30d**, null at 3/7/14. Breadth doesn't move
  accuracy; depth barely does, and only at the longest horizon.

## Implication

Expanding the served universe is **accuracy-neutral** provided items arrive as **deep,
Steam-consistent history**. This test used the deep ≥$1 cohort as a faithful proxy for
newly-backfilled Steam items — **not** stale/Buff iflow rows. So the breadth path that preserves
accuracy is **Steam-consistent backfill, not the iflow/Buff merge**. Accuracy is not gained by
adding items; it is (weakly) gained by adding depth.

## Operational note

Same run: the heavy harnesses (`recency_weights`, `train_universe`, `ensemble`, `regime`,
`supply_side`) were OOM-killed (exit 143) at "Engineering price features" after ~4M rows on the
`ubuntu-24.04-arm` standard runner. This is a **runner-size** ceiling on the heavy harnesses,
distinct from the cold train-universe schema-drift OOM fixed the same day
(`2026-08-18-train-universe-derived-from-archive.md`). Give them a bigger runner or a `max_items`
cap.
