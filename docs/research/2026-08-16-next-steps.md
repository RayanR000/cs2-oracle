# Next steps, 2026-08-16

**Supersedes the ordering in `2026-08-14-next-steps.md`**, whose #1 item failed on 2026-08-15 and
which predates a week of work that lived only in changelogs. Earlier lists keep their item-level
detail; this one owns the ranking.

**Written after** `2026-08-16-research-docs-review.md` (audit of all 32 research docs) and the
three measurements taken the same day. Two items below are recovered from
`2026-08-07-next-steps.md`, which had dropped them without closing them.

## What closed since 2026-08-14

| Item | Outcome |
|---|---|
| 08-14 **item 1** — Oct-22-2025 / Armory `tier × post` | ❌ **FAILED** both bars. Wrong sign on knives/gloves (t=2.55) *and* placebo diverges more than the event window. `changelog/2026-08-15-armory-tier-post-fails-and-fails-placebo.md` |
| Wash-trade / volume-price co-movement screen | ❌ **Null.** Lift 0.31× where the fingerprint predicts >1. `2026-08-16-wash-trade-screen-and-volume-spike-exceedance.md` |
| Volume-spike exceedance feature | ❌ **One episode.** 81% of 13 years of spikes are 2025-10; lift 1.81× → **1.145×** without it |
| Listing-count floor | ⚠️ **Effect real, threshold not portable.** ≥30 costs 69% of the cohort to buy 0.011. Use `log1p(listing_count)` as band width, not a gate. `2026-08-16-listing-count-floor.md` |
| **R13** cohort inversion | ⚠️ **Measured, premise refuted.** See item 1 |

## The list

### 1. Decide the cohort question that R13 was really asking 🔴 *(product call, not mine)*

Measured 2026-08-16 (`2026-08-16-r13-cohort-inversion-measured.md`): the served cohort is **16.1%
≥$1**, median served price **$0.09**, p25 **$0.03** — worse than the 16.4% R13 recorded, because
the 07-29 expansion from 5,542 → 8,691 items was almost entirely sub-dollar.

But R13's stated premise is **false**. Sub-$1 is the *best-calibrated tier we have* (coverage
0.832 / 0.822 / 0.824 / 0.882 against an 80% nominal, closest to nominal at every horizon), and the
**over-coverage lives in the ≥$1 tiers** — the ones every published metric uses.

So this is not a data-plumbing fix and should stop being filed as one. The question is: **serve
8,691 well-calibrated forecasts that mostly cannot be traded, or 1,398 that can?** Raising
`MIN_SERVED_PRICE_USD` cuts the catalogue 84% and *worsens* average calibration.

**Recommended:** don't raise the floor. Ship the **two-tier framing** — serve everything, label the
sub-$1 tier not-tradeable using the 35.5% spread and the piecewise-fee numbers already in
`references/cs2-market-domain.md`. Costs no catalogue, states the truth, and is honest about a
$0.09 forecast. **Needs a decision either way — a week of silence is what put it here.**

### 2. Repair the volume feed 🔴 *(the largest measured accuracy item open)*

`changelog/2026-08-15-volume-features-remeasured.md` measured **+1.503 / +1.879 / +1.442 pp** DA at
3/7/14d, placebo-clean, 11 of 13 features surviving the prune. It is shelved **only because the
live feed is dead**, and `volume-data.md`'s banner had been stating the opposite verdict until it
was corrected today.

`count_in_24` in the iflow backfill (`2026-08-16-refutation-power-tiers-and-iflow-backfill.md`) is
exactly that repair, and it also multiplies backtest episodes ~10×. That makes it the highest-value
data item on the board — it is no longer optional infrastructure, it gates a measured gain.

⚠️ DA is not a shippable claim on its own (`AGENTS.md` invariant 4). Justify this on **band
quality**, not DA, or it cannot ship regardless of the number.

### 3. Re-run the nine repaired A/B harnesses 🟠

`changelog/2026-08-13-harness-family-repinned-off-steamcommunity.md` repaired three harnesses that
matched **0 rows** and six that never applied the production allowlist. **None were re-run**, so
every stored `ab_test_*` verdict in the repo is unmeasured, including the ones cited as settled in
`2026-08-07-cs2-forecasting-research.md`. Until this runs, "we tested that" is not a true statement
about nine arms.

### 4. Over-coverage on the ≥$1 tiers 🟠

Item 1's measurement sharpens C5: the ≥$1 tiers run **84–94%** against an 80% nominal while sub-$1
sits at 82–83%. The band is too wide exactly where the product reports. Prior offline remedies are
exhausted; the dormant served-outcome `q_hat` feedback self-activates at 20 served dates. **Free —
read it off the next retrain**, don't build anything new. *(= 08-14 item 2, still open.)*

### 5. Cross-market basis diagnostic 🟡

Still open and now better motivated: the Steam wallet-lock makes the Steam−Buff wedge
**structural and persistent**, not a latency artifact, and our blended `aggregator_sync` destroys
it before it can be tested. Un-blend the per-venue closes we already ingest.
⚠️ Do **not** encode a lead-lag in *hours or days* — no public source has ever measured one
(`references/cs2-market-domain.md` §9). *(= 08-14 item 3.)*

### 6. `log1p(listing_count)` into the band width — ❌ CLOSED, refuted 2026-08-18

**Refuted offline, do not re-run.** `changelog/2026-08-18-listing-count-conditioner-refuted.md`.
0/3 horizons pass (h=3 worsens the worst-bucket miss; h=7/14 worsen dispersion), and the decisive
finding is structural: the thin buckets (1–5, 6–15 listings) that are the whole target hold <20
items each in the served ≥$1 cohort and drop out — the per-item `sigma` already absorbs listing
information for the population actually served. Only reopen if the iflow backfill materially grows
the thin-listing ≥$1 cohort; re-check bucket `n` first.

### 7. Recovered from 2026-08-07, still unclosed 🟡

Neither was carried into any list after 08-09 and neither was ever refuted:

- **Item 8 — hedonic market index** (`2026-08-07-next-steps.md:577`, "NOT STARTED").
- **R14 — mechanical supply-position features** (`:813`): trade-up fuel/output as a per-item
  column. Distinct from the `tier × post` regression that failed; that was an event read, this is
  a feature.

Both are cheap to *decide*. Close them explicitly or rank them — do not let them fall out again.

## Do not run

- Another **band-width scale** (three measured; width is not the lever — `AGENTS.md`).
- Any **per-item directional** arm, or another **cross-sectional ranker**. Lambdarank is refuted
  for serving; the "pivot to a ranker" thesis is closed.
- The **wash-trade screen** and the **volume-spike exceedance feature** — both measured null today.
- A **listing-count gate** at ≥30 or ≥50 — measured, not portable from the Steam-based index it
  came from.
- Anything ranked on a **stored rank IC, DA, or `−return_1d`** comparison. None are safe
  (`README.md:19`).
- The **label-denominator** thread. Built, refuted, dead end.
- The **`log1p(listing_count)` band-width conditioner** (was item 6). Refuted 2026-08-18 — the thin
  buckets it targets barely exist in the served ≥$1 cohort.
  `changelog/2026-08-18-listing-count-conditioner-refuted.md`.
- A **static cross-venue arbitrage / raw-feed spread scanner.** No capturable edge even at a 5%
  fee; low-fee venues are already efficient.
  `changelog/2026-08-17-low-fee-arb-static-retest-negative.md`.

## Hygiene

- Index the **13 unindexed research docs**, including all 10 preregistrations
  (`2026-08-16-research-docs-review.md` §6). Done for the 2026-08-16 docs; the preregs remain.
- Three preregistrations had their gate rewritten after the result was seen, all on 2026-08-13
  (§7 of the review). Write the bar so it can bind, or don't write one.
