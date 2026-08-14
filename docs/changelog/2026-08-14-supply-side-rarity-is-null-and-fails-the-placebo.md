# Supply-side (rarity) features are null and fail the placebo — item 0b closed

**Date:** 2026-08-14
**Item:** 0b of `docs/research/2026-08-13-next-steps.md` ("re-read supply depth — the only
feature never measured").
**Prereq:** item 0 (`2026-08-13-harness-family-repinned-off-steamcommunity.md`, committed
`5de382b`) — `ab_test_supply_side.py` selected **zero items** until the `STEAMCOMMUNITY`→`source
IS NULL` repin, so it had never run.

## Result

On the honest ≥$1 served universe (200 items, one shared frame cache of 759,690 rows,
33 production-allowlist base columns + 12 `rarity_*` columns), sharded arm×horizon, merged at
fold grain with `merge_supply_side_ab.py`, the **supply-side rarity feature group is null at all
four horizons** and **fails the ship rule** (treatment must beat both control and the
shuffled-rarity placebo):

| Horizon | treatment − control | placebo − control | folds | verdict |
|---|---|---|---|---|
| 3d  | +0.050pp [−0.127, +0.215] | +0.081pp [−0.150, +0.335] | 26 | null; placebo ≥ treatment |
| 7d  | +0.177pp [−0.100, +0.442] | +0.081pp [−0.119, +0.258] | 26 | null; CI straddles 0 |
| 14d | +0.242pp [−0.081, +0.546] | +0.162pp [−0.119, +0.477] | 26 | null; CI straddles 0 |
| 30d | +0.032pp [−0.428, +0.508] | +0.036pp [−0.140, +0.212] | 25 | null; placebo ≥ treatment |

Per-fold win counts are ~50% (treatment>control 15/15/16/10; treatment>placebo 13/13/15/11).
Mean treatment−control across horizons +0.13pp (context, not a verdict). n ≈ 105K scored rows
per horizon.

## Why this is the decisive read, not just another wide interval

The point is not only that every CI straddles zero. It is that **treatment never separates from
the shuffled-rarity placebo** — at 3d and 30d the placebo reads *higher* than treatment. That is
the capacity signature the 2026-08-13 leak audit
(`.claude/rules/ab-statistics.md`, `2026-08-13-the-leak-is-worth-two-points-and-it-pays-the-placebo.md`)
requires a real feature to clear: the columns buy nothing beyond their column count. The drop of
this feature group had rested on mechanism plus analogy to trade volume, with no A/B, no folds
and no MDE; it now has all three and reads null. Item 0b is closed.

## Scope caveat

This measured the `rarity_*` supply *proxy* — the feature group `ab_test_supply_side.py` adds.
It did **not** measure the live listing-count depth feeds (`supply-*.parquet` sidecar, the
archive `volume` column), which remain a distinct, still-unwired, off-by-default input. Depth-as-
listing-counts is a separate wiring + A/B, not settled here.

## Reproduce

```
cd backend
venv/bin/python scripts/ab_test_supply_side.py --max-items 200 --build-cache-only \
    --frame-cache /tmp/frame.parquet
for arm in control treatment placebo; do for h in 3 7 14 30; do
  venv/bin/python scripts/ab_test_supply_side.py --max-items 200 \
    --frame-cache /tmp/frame.parquet --arm $arm --horizon $h --out /tmp/shard_${arm}_${h}.json
done; done
venv/bin/python scripts/merge_supply_side_ab.py /tmp/shard_*.json
```

One arm×horizon shard is ~88s; the full set ~16 min.
