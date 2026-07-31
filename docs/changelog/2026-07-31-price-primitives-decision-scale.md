# Price technical primitives — decision-scale re-run (2026-07-31)

**Status:** SHELVING CONFIRMED. The six primitives measure **negative at all four
horizons** at 200 items. Keep them in `ItemForecaster.SHELVED_FEATURES`.
**Supersedes the "to revisit" note in** `2026-07-31-price-primitives-shelved.md`.

## Why this run happened

The shelving entry earlier the same day was explicit that its verdict came from a
**40-item smoke** and was therefore *unproven, not refuted* — "the effect could be
real and simply invisible at this sample size." This is that re-run, at the
`--max-items 200` the plan always specified as decision scale.

## Result — treatment loses at every horizon

Walk-forward, 200 items, `--q50-only`, 26 folds/horizon, ~109k validation
samples/horizon. Paired per-fold deltas (treatment − baseline on identical folds):

| Horizon | baseline | treatment | delta | fold wins | paired t p | fold sd |
|---------|:--------:|:---------:|:-----:|:---------:|:----------:|:-------:|
| 3d  | 62.29% | 61.60% | **−0.69pp** | 10/26 | 0.099 | 2.09pp |
| 7d  | 62.77% | 61.30% | **−1.47pp** | 12/26 | 0.149 | 5.03pp |
| 14d | 64.24% | 63.36% | **−0.88pp** | 11/26 | 0.443 | 5.70pp |
| 30d | 71.63% | 70.25% | **−1.38pp** | 10/26 | 0.594 | 12.98pp |

Pooled across horizons: **−1.10pp**, **43/104** fold wins, p=0.136.

The 40-item smoke's **+0.13pp flips to −1.10pp** at decision scale, and the sign
is negative at *all four* horizons. No individual horizon is significant, but
nothing here argues for shipping and the consistency of the sign argues against.

All six primitives survive the 0.95 correlation prune (134 features retained of
174), so the treatment arm genuinely carried them — this is not a null from the
columns being pruned away.

### Two honest caveats

- **The placebo arm was not run.** Placebo exists only to catch capacity
  inflation *inflating* treatment. Treatment lost to baseline outright, so the
  guard is moot — there is no positive result needing to be explained away.
- **`--q50-only` means interval coverage was not measured.** The pre-registered
  ship rule reads directional accuracy only (`interval_coverage` is explicitly
  `None` in this mode), so this costs the decision nothing. If these are ever
  revived, measure coverage before serving p10/p90.

## The more important finding: the gate was never falsifiable

The pre-registered gate asks for a "meaningful, non-flat" improvement, which the
project reads as ~0.5pp. **The harness cannot resolve that.** From the measured
paired per-fold delta sd above:

| Horizon | paired fold sd | min detectable effect @ 26 folds (80% power) |
|---|:---:|:---:|
| 3d  | 2.09pp  | 1.15pp |
| 7d  | 5.03pp  | 2.76pp |
| 14d | 5.70pp  | 3.13pp |
| 30d | 12.98pp | 7.13pp |

Detecting 0.5pp at 7d would need **~795 folds**. Two structural facts make that
unreachable:

1. **Fold count is set by `step=60` and the archive date range, not by
   `--max-items`.** It is 26 folds at 40 items and 26 folds at 200. More items
   sharpens each fold's estimate; it never adds a fold.
2. The most folds obtainable with *disjoint* 21-day validation windows is **~73**
   (`step=21`), which only lowers the 7d floor to ~1.65pp — still 3x the target.

So the 40-item smoke and a 200-item run were always going to land in the same
noise band. The smoke was not "too small to see the effect"; the *design* is too
noisy to see a 0.5pp effect at any item count.

**Generalises to every `ab_test_*.py` in this repo.** Before running a feature
A/B here, run two arms on one horizon, compute the paired per-fold delta sd, and
derive `MDE = 2.8 * sd / sqrt(n_folds)`. If the gate's target is below the MDE,
the run cannot produce a decision — say so and skip it. Buying more items is not
the fix; more folds or variance reduction is.

## Cost — the run is ~10 minutes, not hours

Three levers, all already present in the harness and all previously unused:

| Lever | Effect |
|---|---|
| `--q50-only` | **3x** — the gate reads dir-acc only; p10/p90 were pure waste |
| `--frame-cache` | Frame built **once in 34s** (903k rows, 134 features), reused by every shard |
| Shard + low `--n-jobs` | Sequential `--n-jobs 10` ran at only **303% CPU** — LightGBM stops scaling near 3 threads on this shape |

One horizon×arm unit is ~40s. Six units sharded at `--n-jobs 3` finish in roughly
the wall time of two sequential ones.

### Reproduce (from `backend/`)

```bash
# 1. Build the engineered frame once (~34s)
DATABASE_URL="sqlite:///./cs2_market.db" python scripts/ab_test_price_primitives.py \
  --max-items 200 --build-cache-only --frame-cache /tmp/frame200.parquet

# 2. Shard horizon x arm, 6 processes at 3 threads each
for h in 3 7 14 30; do for a in baseline treatment; do
  DATABASE_URL="sqlite:///./cs2_market.db" python scripts/ab_test_price_primitives.py \
    --max-items 200 --frame-cache /tmp/frame200.parquet --horizon $h --arm $a \
    --q50-only --n-jobs 3 --out /tmp/${a}_${h}.json &
done; done; wait
```

Thread count does not change results — `force_row_wise` is pinned in the harness
precisely so sharded low-thread runs stay comparable.

## `walkforward_backtest.py` was deliberately skipped

Not run as a confirming gate, and it should not be for A/B work. Per
`backend/AGENTS.md`, its own `_load_all_prices` bypasses multi-source voting, the
`source` filter, and the dead-item filter, so it scores a different price
consensus over a different item universe than production. It is an *absolute*
accuracy benchmark; for a *delta* between two arms the shared-fold design already
cancels every confound it would introduce. It would have cost 60–90 minutes and
changed nothing.

## Disposition

- The six columns stay **engineered but withheld** via
  `ItemForecaster.SHELVED_FEATURES` (`forecaster.py:209`). Unchanged.
- The gate matters because all six are price technicals *by name* — `_feature_group()`
  keys off the `vol_`/`rsi_`/`macd_` prefixes, so `FEATURE_GROUP_ALLOWLIST =
  ["price_technicals"]` would otherwise admit them into production.
- Regression cover unchanged: `test_build_training_data_*` asserts the shelved
  names stay out of `feature_cols` while remaining in the engineered frame.
- **Do not re-run at larger item counts.** Item count does not add folds and
  cannot resolve this. If the columns are ever deleted outright, the regression
  tests asserting their exclusion go with them.
