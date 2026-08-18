# Wash-trade screen: null. Volume-spike exceedance: one episode wearing a signal's clothes

**Date:** 2026-08-16
**Ran:** read-only over `price-archive/volume-panel.parquet` (10,675,502 rows; 4,342 items;
2013-08-14 → 2026-06-15; Steam **trade counts** + sale median, same source and day).
**Bears on:** `references/cs2-market-domain.md` §8 and §10 item 1, which proposed both of these;
the `P(|r|>cost)` exceedance target of `research/2026-08-15-p-exceed-cost-target-scope.md`.
**Status:** research only. No code changed. Scripts were scratchpad-only and are not committed.

## Bottom line

Two results, one null and one that *looks* positive and is not:

1. **The wash-trade fingerprint does not exist in our data.** Volume spikes are strongly
   *informative* about price, which is the exact opposite of the wash-trade signature. There is
   no listing-count hygiene win here.
2. **A volume spike does lead |return| — but 81% of all spikes in 13 years are October 2025.**
   Vol-matched lift is 1.81× at h=3 with the crash in, **1.15× with it out**. This is one episode,
   not a feature. It is the same power trap the repo has hit five times.

## 1. The wash-trade screen is null

Fingerprint sought (`arXiv 2312.16603` / `2102.07001`, via `references/cs2-market-domain.md` §8):
volume explodes while price does not move. Spike = volume ≥ 6× the trailing-30d median
(the ">500% day-over-day" heuristic), baseline ≥10 sales/day so ratios mean something,
consecutive days only, `steam_sale_median ≥ $1` to match the served floor.

**4,710,086 usable item-days across 3,624 items. 1,787 spikes.**

| | P(\|Δlog p\| < 5%) | median \|Δlog p\| |
|---|---|---|
| volume spike | **0.2429** (n=1,787) | **0.1158** |
| no spike | **0.7795** (n=4,708,299) | **0.0237** |
| lift | **0.312×** | 4.9× |

A wash-trade regime predicts lift **> 1** — volume without price discovery. We measure **0.31×**:
spike days move ~5× as far as normal days. Per item the flag rate is ~0 (median 0.00000,
p99 0.00240, **max 0.00820** = 3 flagged days out of 366), with no concentration in any price
tier. No series in this panel looks systematically washed.

**Do not read this as "there is no wash trading in CS2."** It is a null on the cohort we can see,
and that cohort is the least likely place to find it:

- This panel is **Steam only**, and Steam is the venue where washing is most expensive — 15% fee
  per round trip plus the 7-day trade hold. Manipulation is alleged mostly on thin items and on
  third-party venues, neither of which is here.
- 4,342 items with ≥10 sales/day is the **liquid** tail. Thin-float items — where §8 says the risk
  concentrates — are filtered out by the `BASE_MIN` floor that makes the ratio meaningful at all.
- We have no trade-graph data, and the laundering fingerprint in §8 is *graph structure, not
  price*. It is invisible to us by construction, so this screen could never have found it.

**Consequence:** the "cheapest hygiene win" in `cs2-market-domain.md` §10 item 1 is **half
refuted**. The volume/price co-movement screen finds nothing and should not be built. The
listing-count floor is untouched by this — it rests on published index methodology, not on this
screen, and remains open.

## 2. The volume-spike exceedance result, and why it dies

The screen's residual is the interesting part: if spikes coincide with big moves, do they
*precede* them? That is exactly `P(|r| > cost)`, the one surviving signal.

Forward |return| from a spike day, panel-wide:

| h | n_spike | median \|r\| spike | median \|r\| base | P(\|r\|>5%) spike | base | lift |
|---|---|---|---|---|---|---|
| 3 | 1,775 | 0.1397 | 0.0333 | 0.7949 | 0.3384 | **2.35×** |
| 7 | 1,769 | 0.2340 | 0.0459 | 0.8632 | 0.4660 | **1.85×** |
| 14 | 1,760 | 0.3201 | 0.0610 | 0.9239 | 0.5753 | **1.61×** |

It survives the two obvious confounds. Conditioning on the day having **already** moved >5%:
lift 1.59 / 1.48 / 1.41. Matching spike to non-spike days **within trailing-realized-vol decile**
(the volatility-clustering null): lift **1.81 / 1.58 / 1.46**. At that point it looks like a real
feature for the exceedance model.

**Then count the episodes.** The 1,784 vol-matched spikes are:

- **654** distinct items
- **158** distinct dates
- **62** distinct year-months
- **1,447 of them — 81% — in 2025-10 alone** (next largest month: 68)

The spikes are the trade-up crash. Excluding 2025-10 and 2025-11:

| h | vol-matched spike | base | lift | n_spike |
|---|---|---|---|---|
| 3 | 0.5389 | 0.4705 | **1.145** | 193 |
| 7 | 0.6825 | 0.5683 | **1.201** | 189 |
| 14 | 0.8315 | 0.6494 | **1.280** | 184 |

269 spikes remain, across 200 items / 133 dates / 60 months. The h=3 lift falls from **1.81× to
1.145×** — and 133 dates spread over 60 months is not 133 independent draws either, since spikes
cluster within dates.

**Verdict: not a feature.** What the panel-wide number measures is "the trade-up crash had huge
volume and huge subsequent moves," which is true, market-wide, and already the date effect. The
residual out-of-crash lift is small, low-powered, and exactly the size that has failed to transfer
to serving every previous time.

## What this changes

- **Don't build** the volume/price wash screen (§10 item 1, first half). Measured, null.
- **Don't build** a volume-spike exceedance feature. Measured, one-episode.
- **Still open:** the listing-count floor on the served cohort — independent of both results above.
- **Method note worth keeping:** this took three passes to kill. The panel-wide table, the
  same-day control, and the vol-decile match all said "signal." Only the **episode count** said
  otherwise. Any future arm on this data should report distinct-months and the top-month share
  *before* reporting a lift, not after.

## Caveats

- The panel ends **2026-06-15** and this local archive runs behind the durable one; nothing here
  touches the 2026 serving regime directly.
- 3,624 items is ~15% of the ≥$1 served cohort. Both results are conditional on Steam-liquid items.
- Forward windows overlap, so even the raw n overstates independence — the episode count is the
  honest denominator, and it is ~1 for the headline and ~60 months for the remainder.
- `SPIKE=6.0`, `FLAT=0.05`, `BASE_MIN=10`, `WIN=30` were fixed a priori from the cited heuristic
  and not tuned. No threshold search was run — which also means no threshold was *rescued*.
