# Next steps, 2026-08-16

> ## 📌 Status header added 2026-08-21 — item-by-item, and this list is no longer the live plan
>
> **The live ranked plan is §12 of `docs/research/2026-08-19-deep-model-review.md`** (its
> unblock/free-win/measurement-integrity items 1–12 have all landed). This document is the last
> entry in the next-steps chain and is kept for item-level detail. Every item below, re-checked:
>
> | Item | Status as of 2026-08-21 |
> |---|---|
> | **1.** R13 cohort question / two-tier framing | ✅ **DECIDED as recommended.** The floor was *not* raised; the sub-$1 tier is labelled not-tradeable on served forecasts. `changelog/2026-08-17-tradeability-label-on-served-forecasts.md` (commit `cda16ab`). |
> | **2.** Repair the volume feed | ✅ built, ❌ **CLOSED NET-NEGATIVE.** iflow `count_in_24` was wired as `IFLOW_VOLUME=1` and *passed* band quality at h=14 on a re-fold — but on the **retired** q10/q90 band. Volume in the production learned scale degrades the dominant `sigma`-tilt axis. `changelog/2026-08-17-volume-in-scale-is-net-negative.md`. Known ingest defect if revisited: BUFF `count_in_24` and Steam volume are spliced into one mislabeled column. |
> | **3.** Re-run the nine repaired A/B harnesses | ⚠️ **RECOMMENDATION REVERSED — still open, but as a *labelling* task, not a re-run.** The batch workflow exists (`7d05b40`) and the OOM blocker is fixed (`changelog/2026-08-18-train-universe-derived-from-archive.md`), but `2026-08-19-deep-model-review.md` §11–§12 rules a broad re-run **out**: the A/B family was never powered (MDEs 1.15–7.13pp), so most stored "null" verdicts are **UNRESOLVED**, not null. The fix is to put the MDE beside each stored verdict and relabel. Also: no 2026 A/B is interpretable until the consensus-estimator composition breaks (§1) are controlled. |
> | **4.** Over-coverage on the ≥$1 tiers | ✅ **LARGELY ADDRESSED, by a route this item did not name.** Centre fix shipped 2026-08-19 (signed conformal band + `DIRECTION_UPWEIGHT = 1.0`); the `sigma` denominator was **replaced** on 2026-08-20 by a featureless per-item climatology band — 33–46% narrower at matched 80% and better calibrated on served replay (`docs/research/2026-08-19-climatology-vs-gbm-band.md`, `changelog/2026-08-19-climatology-band-scale-default-on.md`). The regime-reactive follow-up `CLIMATOLOGY_REACTIVE` was **shelved** after a prod A/B (`changelog/2026-08-20-climatology-reactive-band-scale.md`). ⏳ Residual: the served-outcome `q_hat` feedback is armed but dormant until 20 served dates accrue. |
> | **5.** Cross-market (Steam−Buff) basis diagnostic | ❌ **RUN AND SHELVED 2026-08-19. Do not build the ingest.** `docs/research/2026-08-16-cross-venue-basis-steam-buff.md` carries the full forward-pointer. Every skeptical control moved it the wrong way; the recorded corr −0.12 was a market-factor + shared-quote wedge artifact, and the wedge-controlled cross-sectional IC is ~−0.06 decaying to 0 by h=30. |
> | **6.** `log1p(listing_count)` into band width | ❌ **CLOSED, refuted 2026-08-18** — already recorded in-document below. Confirmed accurate. |
> | **7.** Recovered items: hedonic market index (08-07 item 8); **R14** mechanical supply-position features (08-07 `:813`) | 🟡 **STILL GENUINELY OPEN — the only open items on this list.** Neither has been run or closed since. R14 is now better grounded: only **5 containers still drop**, a verified finite-supply rule that replaces the old age proxy. |
>
> **Two closures that postdate this document and change its "Do not run" section:**
>
> - **`P(|move| > cost)` exceedance is the one signal that survived serving** and has **shipped** —
>   as served volatility/stability tags, not as a trade
>   (`changelog/2026-08-20-exceedance-served-signal.md`, `…-volatility-stability-tags.md`, PRs #28/#29).
> - **The trade hunt is closed at every horizon.** The long-horizon (h=90/180) move magnitude does
>   clear the friction bar, but *selection* is null (AUC 0.51–0.53 over 3 OOS windows, top-decile
>   lift ≤1.0×) and the unconditional bet lost money in every window. The product is
>   **range-only**. Add "another horizon-extension trade hunt" to the Do-not-run list.


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
