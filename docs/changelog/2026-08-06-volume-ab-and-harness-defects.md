# Volume features refuted on measurement — and the A/B harness was scoring the wrong cohort

**Date:** 2026-08-06
**Change:** new `backend/scripts/ab_test_volume_features.py`. No production code
changed; the volume features stay in `SHELVED_FEATURES` and
`test_volume_features_shelved.py` still passes 7/7.

Three results, in increasing order of importance:

1. CSFloat is not worth collecting — measured, not inferred.
2. `volume-data.md`'s stated rationale for refuting volume is wrong, even though
   its conclusion is right.
3. **The A/B harness reports a directional accuracy that is ~31pp free hits on a
   cohort that is 92% unserved.** That defect is not specific to volume and calls
   every prior null measured through it into question.

## 1. The CSFloat pilot: measured weak, do not build it

Cohort defined to mirror prod `is_backfilled = 1` exactly: the 5,542 pre-2026
archive items. **All 5,542 are still live in the daily aggregator**, and only
**1,079 are ≥$1** — the backfilled cohort is mostly penny items. Seeded reservoir
sample of 500 from that pool (median $3.24, range $1–$350).

Fetch: **500/500 items, 0 errors, 173 s, 391,775 daily rows**, median 820
days/item, `count` = daily completed sales.

**Rate limit, now pinned** (`volume-data.md` and the 08-06 feeds entry both left
this unestablished): `x-ratelimit-limit: 500` against a **fixed reset timestamp
~8 h out**, decrementing 1 per request. Burst throughput is **6.04 req/s with no
429 in 32 requests** — so the quota, not bandwidth, binds: it empties in 83 s.
The counter is per-backend, which is what produced the earlier "bounced between
145 and 498" observation. Practical consequence: a 500-item pilot is free and
immediate, but the ≥$1 cohort is 51 windows ≈ 17–51 days and the full archive is
83 windows.

Date-clustered Spearman against forward returns (~2,100 dates, 2020–2025):

| Feature | 3d | 7d | 14d | 30d |
|---|---|---|---|---|
| `csfloat_vol_change_7d` | 0.022 | 0.021 | 0.014 | 0.012 |
| `csfloat_vol_z` | 0.026 | 0.017 | 0.001 ns | −0.000 ns |
| `steam_vol_z` (already in the archive) | 0.095 | 0.110 | 0.098 | 0.072 |

CSFloat is ~5× weaker than the volume column the archive already carries, and it
sees a median of **4 sales/day/item against the archive's 95** (Spearman between
them only **0.145** over 288,687 overlapping item-days). It is a thin,
Poisson-noisy view of a series already on disk. **Do not build the collector.**

## 2. `volume-data.md`'s |r| < 0.002 is a cohort artifact

Reproduced the audit's exact window and got **n = 4,472,754** against its stated
"4.47M samples" — same cohort. Pearson vs 7d forward return:

| | Full 4.47M cohort | ≥$1 cohort (764,073) |
|---|---|---|
| `volume_level` | −0.00111 | −0.00307 |
| `volume_zscore_30d` | −0.00021 | **+0.0424** |
| `price_momentum_7d` | **−0.00143** | **−0.09399** |

The bottom row is the finding. **The momentum feature the whole model is built on
also reads 0.001 on that cohort.** On the unfiltered penny-heavy rows nothing
correlates with anything. The doc reports momentum at +0.0796, which cannot have
come from the same computation as its volume numbers.

So "|r| < 0.002, therefore volume is noise" is not a valid inference — the same
test says price momentum is noise. The conclusion happens to be right (see §4),
but the reasoning on record would teach the next reader a false lesson.
`docs/research/volume-data.md:27` and `:142` still carry it.

## 3. The harness defects

The first A/B run returned DA of 62–71% and a clean null. Both were artifacts.

**Cohort.** Item selection is `ORDER BY row_count DESC` — longest history —
which systematically picks cheap high-turnover cases and stickers:

| | |
|---|---|
| Items with median price ≥ $1 | **15 of 200** |
| Median item price | **$0.059** (p25 $0.03) |
| Share of rows ≥ $1 | **8.22%** |

**Metric.** At $0.03 one cent is a 33% move, so prices simply do not move:
**41.0% of 3d forward returns are exactly zero** (64.0% for sub-$0.10 items vs
0.07% for >$5). A quantile objective at α=0.5 on a target with 41% mass at zero
learns to emit **exactly 0.0 on 38.98% of rows**, and the scoring counts
`sign(0) == sign(0)` as a hit. Measured directly:

| | |
|---|---|
| Rows where actual *and* prediction are exactly 0 (free hits) | **31.27%** |
| DA as the harness scores it | **64.38%** |
| DA excluding flat-actual rows | **51.24%** |
| Majority-class base rate (non-flat) | 50.09% |

The 64.38% reproduces the run's 3d numbers. The model had **1.2pp** of real
directional skill presented as 64%. A tie is not a directional call.

The three-arm structure itself (baseline / treatment / **placebo**, permuted
per-fold in both train and val) is correct, and the placebo is what caught the
capacity inflation. The defects are upstream of the A/B.

## 4. Corrected run — a well-powered null

Fixes, both in the new script: `MIN_MEDIAN_PRICE = 1.0` on the universe query,
and a `dir_acc_strict` metric that scores only non-flat-actual rows. They
reinforce each other — the ≥$1 cohort barely has flat rows, so the metric fix
became a safety net.

| | before | after |
|---|---|---|
| Median item price | $0.059 | **$2.51** |
| Items ≥$1 | 15/200 | **199/200** |
| Rows ≥$1 | 8.22% | **87.13%** |
| Flat-actual (3d) | 41.0% | **0.78%** |

Frame: 729,005 rows, 200 items, 181 → 143 features after prune, 11 of the 13
volume columns surviving. Strict metric, ≥$1:

| h | baseline | treatment | placebo | t−b | **t−p** | 95% CI on t−p | n |
|---|---|---|---|---|---|---|---|
| 3d | 59.12 | 59.40 | 59.37 | +0.28 | **+0.03** | [−0.42, +0.49] | 99,915 |
| 7d | 59.15 | 57.88 | 58.08 | −1.27 | **−0.20** | [−1.67, +0.93] | 98,617 |
| 14d | 61.17 | 59.61 | 59.53 | −1.56 | **+0.08** | [−0.38, +0.53] | 97,142 |
| 30d | 67.07 | 67.56 | 67.54 | +0.49 | **+0.02** | [−0.15, +0.25] | 95,278 |

Validity checks: flat-actual now 0.15–0.59%; majority-class base rate
50.05–50.32%, so DA of 57–67% is 7–17pp of genuine skill; strict and pooled
metrics agree within ~0.3pp. Fold-paired bootstrap (25–26 folds, dates as the
unit) resolves **0.20–0.46pp at 3d/14d/30d** and 1.30pp at 7d.

**The claimed +1–2pp for the change/velocity variant is excluded at three of four
horizons.** The features stay shelved, now on measured grounds in addition to the
dead feed.

Two by-products worth keeping:

- **Uninformative columns actively hurt.** At 7d/14d, t−b is −1.27/−1.56 while
  t−p ≈ 0: both arms lose to baseline equally, so the damage is dilution — with
  `feature_fraction = 0.7`, 11 noise columns crowd out useful ones.
- **The in-sample signal was real but redundant.** `steam_vol_z` gives Spearman
  0.110 and a +2.35pp quintile-rule lift in-sample (quintile % up: 42.7 / 45.4 /
  47.4 / 51.1 / 55.9). It does not survive a purged walk-forward because the
  price technicals already encode it.

## What this does NOT settle

- **Supply depth is still untested.** This measured trade volume (how many
  *sold*). Scarcity is listing count (how many *exist*) — a different quantity,
  and `ops/supply_snapshots.parquet` holds exactly one day (2026-07-15, 35,037
  items, zero `skinport_quantity`). Note also that scarcity plausibly prices into
  the *level*, which is already in the anchor and the rarity features; only
  *unpriced changes* in scarcity could move a percentage-return target.
- **Bid-ask spread, from `aggregator_buff163_buy`** (32,943 items, 21 days), is
  untouched by this result: it predicts return *magnitude*, not direction (mean
  7d return −0.78% → −3.13% across spread deciles, Pearson −0.078, but `% up`
  non-monotone). That points at conformal band width, not the direction model,
  and needs no new collection.

## Follow-up this opens

`scripts/ab_test_price_primitives.py` shares **both** defects — no price filter
and the tie-counting metric — and it is what shelved the six price primitives on
a measured null (`2026-07-31-price-primitives-shelved.md`). Anything else scored
through this harness inherits the same doubt. Re-checking that is a bigger lever
than the volume question was.

## Related

- `docs/research/volume-data.md` — the audit whose rationale §2 corrects
- `docs/changelog/2026-08-06-volume-features-shelved.md` — the train/serve gap
  that shelved these features; still the binding reason for production
- `docs/changelog/2026-08-06-free-bulk-supply-depth-feeds-exist.md`,
  `2026-08-06-retroactive-supply-feeds.md` — the CSFloat and depth-feed audits
- `docs/changelog/2026-07-16-drop-supply-depth.md` — its accuracy leg now has
  direct walk-forward support, for trade volume at least
- `docs/research/lis-skins-snapshot-plan.md` — its "revisit in 6–8 weeks" is
  optimistic: `STEP_DAYS = 60` means a new feature needs ~300–400 days of its own
  history before the harness can score it
