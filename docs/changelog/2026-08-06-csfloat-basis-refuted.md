# CSFloat's cross-market price basis is refuted, and its budget is 500 requests per day

**Date:** 2026-08-06
**Change:** two new scripts, `backend/scripts/probe_csfloat_history.py` and
`backend/scripts/ab_test_csfloat_basis.py`. No production code changed, no retrain, no
change to `models/saved_models/`.
**Bears on:** `2026-08-06-data-acquisition-ranking.md` Tier 1 item 3, and the
`docs/references/data-sources.md` entries that call this endpoint "viable and unblocked;
simply not integrated" and "under-rated". It was ranked on two hand-checked items; this
is the measurement.

`csfloat.com/api/v1/history/<market_hash_name>/graph` was the highest-ranked untried
source with a live mechanism argument. Every price feature in this project is a
repackaging of "what the market did that day" — which is why demeaning the label by the
market factor drops accuracy below a constant call
(`2026-08-06-market-relative-labels-refuted.md`). A **cross-market basis** is the one
feature form that is structurally immune to that: a difference between two prices for one
item at one instant, so the market term cancels by construction.

It does not work. **Nothing here recommends building a collector.**

## 1. The budget is 500 requests per DAY, not per minute

`x-ratelimit-limit: 500` with `x-ratelimit-reset` ~86,400 s out, verified on live calls.
The ranking's "3.7 req/s" is bandwidth and should not be read as throughput: at one IP a
full-catalogue backfill of ~26K items is **~52 days**, and even the 870-item A/B cohort
takes two. This bound belongs in any future decision about the endpoint, independent of
the accuracy result below.

`probe_csfloat_history.py` is built around it — checkpoints after every item, resumes on
re-run, never sleeps waiting for a reset.

## 2. Coverage clears both gates

400 items probed (200 from the 870-item deep ≥$1 harness universe, 200 sampled from the
1,432 served ≥$1 items in `item-metadata.parquet`). 336 returned history; 312,201 series
rows.

| | harness | served ≥$1 |
|---|---:|---:|
| Any history | **98.0%** | **70.0%** |
| History spanning the fold split | **92.5%** | 54.0% |
| Median depth | 1,058 days | 854 days |
| Median first day | 2020-08-12 | 2022-05-04 |

Response shape is `{day, avg_price, count}`, descending, `avg_price` **in cents**.

## 3. There is no fee offset, but the daily price is not a clean price

Median CSFloat/archive ratio is **0.955** over 272,410 overlapping item-days — parity with
the archive's net price, with no constant to correct. Unlike the Steam listing pages'
1.1607 divisor, there is no silent-corruption trap here.

The dispersion is the problem. Within-item CV of the ratio is **1.347**, and it falls
monotonically as the daily sale count rises:

| min sales/day | item-days | items | within-item CV |
|---:|---:|---:|---:|
| 1 | 272,410 | 312 | 1.347 |
| 3 | 174,195 | 312 | 0.620 |
| 10 | 80,560 | 233 | 0.240 |
| 50 | 7,495 | 37 | 0.098 |

Median sales/day is **4**. The reading: `avg_price` is the mean of whichever specific
float/pattern instances sold that day, and CSFloat sells by float, so on a typical day the
composition term dominates the market term. This is an inference from the monotonicity
rather than a direct measurement of float, but nothing else explains the shape. Even at
n≥50 the p10/p90 band is 0.750/1.248.

**This is a durable property of the source, not a calibration bug.** Any future use of
this endpoint has to deal with it.

## 4. The A/B

`ab_test_csfloat_basis.py`. Universe, fold geometry, row budget, metric, seeds and the
significance test all match `ab_test_item_metadata.py` and `ab_test_training_breadth.py`.

* **Universe:** the deep ≥$1 definition intersected with what the probe pulled — **271
  items** (against those scripts' 870, because of the daily budget). 80 held-out eval /
  190 training.
* **Features:** production's exact 33 `price_technicals` columns, the same 33 the shipped
  tree carries.
* **No lookahead:** every CSFloat column is computed from t-1 and shifted, *including the
  archive leg* — `cf_basis` at t is `log(cf[t-1] / archive[t-1])`. CSFloat's `avg_price`
  for day t averages sales during day t and is not complete when production predicts.
* **Window:** rows before 2020-04-01 dropped. This is not cosmetic — see §6.
* 12–13 folds, ~17,700 paired rows over 252 dates per comparison.

Arms: `baseline`; `basis_raw` (the naive daily spread); `basis_smooth` (7-day basis,
30-day deviation, 7-day change); `basis_all`; `count_only` (the sale count alone, to
separate "the spread helped" from "a liquidity proxy helped"); `placebo` (all five
columns permuted).

### Held-out items — paired diff vs baseline, pp, `*` = 95% CI excludes zero

| arm | 3d | 7d | 14d | 30d |
|---|---:|---:|---:|---:|
| `basis_raw` | +0.12 | −0.02 | **+0.65\*** | **−1.42\*** |
| `basis_smooth` | −0.30 | −0.07 | **+1.03\*** | −0.80 |
| `basis_all` | +0.28 | +0.27 | +0.74 | +1.82 *(MDE 1.92)* |
| `count_only` | +0.27 | −0.41 | +0.82 | +0.65 |
| `placebo` | +0.03 | −0.06 | +0.01 | −0.23 |

MDE is **0.32–0.95pp at 3d/7d/14d** — better than the production floor of 1.15–7.13pp,
because each comparison pairs ~17,700 rows across 252 dates. The test could have resolved
a 1pp effect. It did not find one.

Three specifics worth keeping:

1. **The raw daily basis is significantly harmful at 30d** — −1.42pp held-out, −2.38pp
   trained, the largest magnitudes in the whole table. Exactly what §3 predicts: at long
   horizons the composition noise swamps the spread.
2. **The 7d trained-cohort gain was liquidity, not spread.** `basis_all` +1.17pp
   [+0.38, +1.94] against `count_only` +1.06pp [+0.40, +1.77] — almost all of it is the
   sale-count column, which is a volume proxy, and volume is refuted at |r| < 0.002 over
   4.47M rows. `basis_raw` and `basis_smooth` alone are null at 7d.
3. **Placebo is clean at every horizon** (max |0.23|pp, all CIs spanning zero), so none of
   this is capacity inflation from a wider model.

## 5. A methodological finding: 7 folds is not enough for the date-clustered CI

A second run restricted to 2023+ — where train/val basis coverage is most balanced
(58%/96%) — produced a **significantly positive placebo**: +0.94pp held-out at 7d, CI
[+0.43, +1.52], and −1.32pp trained at 14d. Permuted noise columns cannot help. With only
7 folds a single unlucky permutation draw is not averaged out, so
`paired_da_difference`'s interval is too narrow at that fold count.

Every number in that run is therefore discarded, including the ones favouring the basis.
It was never independent evidence anyway — the dense era is a *subset* of the full window.

**This generalises beyond CSFloat.** Any A/B in this repo running at ≲10 folds should
check that its placebo arm is null before its treatment arm is believed.

## 6. The window restriction was necessary, and the residual asymmetry is structural

The frame carries each item's full archive history back to 2013, and the fold split is
taken over the **date index**, not over rows. On the unrestricted frame the split lands at
**2022-01-27**, leaving the training region with **0.8%** basis coverage against 59% in
validation — the model would be fitted almost entirely on the median fill and then scored
on real values, which tests nothing. Restricting to 2020-04 moves the split to 2024-04-14
and coverage to 18%/88%.

18%/88% is still asymmetric, and it cannot be fixed by choosing a better window or a
better item subset: filtering to the 176 items whose CSFloat series *starts* before
2021-01 still yields 28% train coverage. CSFloat's per-item day coverage simply grows over
time as the marketplace grew. Pushing the cutoff to 2023-01 reaches 58%/96% but costs half
the folds, which is the run §5 discards.

So the null is measured against a genuinely sparse historical feature. That is a real
limitation of this test — and also a real property of the data any collector would land.

## 7. The loose end

`basis_smooth` at 14d is positive and CI-excluding-zero in both windows (+1.03 full,
+1.32 dense, held-out). Against it: the same arm is flat or negative at 3d, 7d and 30d;
the two windows overlap heavily so this is not two independent confirmations; and across
~80 comparisons a couple of hits at p<0.05 is what chance produces.

Treat as unresolved, not as a positive. The cheap way to settle it is finishing the
870-item harness pull — two more days of budget, tripling the item count, giving a
genuinely independent item sample. `probe_csfloat_history.py` resumes into the existing
checkpoint.

## 8. What this does not close

**CSFloat supply is untested and untouched by this.** The `/history/.../graph` endpoint
returns *completed sales* — demand side. Live listing depth is `/api/v1/listings`, which
`data-sources.md` records as needing an API key and not running. It is live-only, so it
carries the same forward-accumulation problem as Skinport `/v1/items`, Waxpeer and
Bitskins — and those three measured as **one** feature (Spearman 0.65–0.82), so a fourth
correlated listing-count feed adds coverage rather than information.

**Supply depth generally remains untested.** `2026-07-16-drop-supply-depth.md` dropped it
on an availability premise since refuted twice; its accuracy claim was never measured.
`docs/research/lis-skins-snapshot-plan.md` is still the vehicle, revisit early October.

## Docs touched

- `docs/references/data-sources.md` — the two CSFloat `/history/…/graph` rows described it
  as unblocked and not-yet-integrated with no verdict attached; both now carry the
  measurement and the 500/day budget.
- `docs/changelog/2026-08-06-data-acquisition-ranking.md` — Tier 1 item 3 called this
  source "under-rated". Corrected in place.

## Related

- `docs/changelog/2026-08-06-data-acquisition-ranking.md` — the ranking this tests
- `docs/changelog/2026-08-06-breadth-beats-depth-item-age-does-not.md` — the harness this
  one is cloned from, and the source of the comparable 33-feature baseline
- `docs/changelog/2026-08-06-market-relative-labels-refuted.md` — the market-factor result
  that motivated a cross-market spread in the first place
- `docs/research/lis-skins-snapshot-plan.md` — where the supply question actually lives
