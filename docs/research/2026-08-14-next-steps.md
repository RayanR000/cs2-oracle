# Next steps after the 2026-08-14 audit — re-ranked, survivors first

**Supersedes ordering in:** `docs/research/2026-08-13-next-steps.md` and the ranked plan in
`docs/research/2026-08-14-what-moves-skin-prices-web-reconsideration.md`. Labels (`C2`–`C7`, `N2`,
`D2`–`D5`, `O2`, `G2`) trace back to `2026-08-10-next-steps.md`, which remains the reference for the
**content** of each. This document only re-ranks: it puts the items still worth running at the top,
marks what closed on 2026-08-14, and quarantines the ones that are refuted or chasing a dead number.

**Sources:** the 2026-08-14 changelog run — `lambdarank-serving-transfer-measured`,
`recency-decay-ships-at-30d-honestly`, `supply-side-rarity-is-null-and-fails-the-placebo`,
`cv-fold-grid-end-anchored`, `harness-family-repinned-off-steamcommunity` — plus the web
cross-check `2026-08-14-what-moves-skin-prices-web-reconsideration.md`.

---

## What the 2026-08-14 changelogs closed

Six of the 2026-08-13 items are no longer open, and the newest doc's top recommendation was
answered the same day it was filed:

- **Item 0 (harness repin)** — done, `harness-family-repinned-off-steamcommunity` / `5de382b`.
- **Item 0b (supply-side rarity)** — **null and fails the shuffled placebo** at 4/4 horizons; the
  drop is now measured, not argued. *The listing-count depth sidecar is a distinct input and was NOT
  measured here.*
- **Item 1 (C1 per-fold)** — done; the CV edge is not a pre-2026 artefact.
- **Item 2 (end-anchor the CV grid)** — shipped (`b1cff21`); its open leg is a **free** re-read, now
  ranked below.
- **Item 4 (recency weights)** — **ships at 30d** under the honest trainer (+1.44% pinball, 6/8
  folds); it was the only genuine un-re-read positive, and it was not the leak. A band/pinball gain,
  not DA.
- **Item 5 / web-doc #1 (lambdarank, "pivot to a cross-sectional ranker scored on Rank IC")** —
  **measured and refuted.** On its own pre-registered Rank-IC ruler it transferred at **1 of 4**
  horizons (h=7 only, net-negative after the 40% round-trip cost), h=3/h=30 sign-flipped, and h=14
  cannot train (>10K-rows-per-query lambdarank cap). Reproduced on the durable archive to four
  decimals — the same CV+/serving− pattern as C1. **This closes the web doc's central thesis.**

**The recurring pattern that governs the ranking below:** every relative / cross-sectional signal
tried so far is CV-positive and serving-negative (C1, then lambdarank — 0 for 2 on serving
transfer). Weight any new relative-value idea by that record.

---

## Ranked — still worth running

### 1. Oct-22-2025 `tier × post` natural experiment — pure archive read, no retrain

The single best untested idea, and the cheapest. The largest datable event in the archive — the
**Oct 22 2025** trade-up update — moved categories in **opposite** directions (knives/gloves
−~70%, Coverts +10–20×). A market-wide daily flag and market-factor demeaning both
**cancel** that by construction, which is exactly why the event-calendar feature went null once
stripped of its clock. This is the only "wrong functional form" (category × event) test that is both
cheap and unexplored.

- **Do:** regress the Oct-22-2025 window as a `tier × post` interaction on the existing archive.
- **Cost:** an `archive-analyst` dispatch, no retrain, no prod write.
- **Bar:** pre-register the tiers, the window, and a null before reading. If positive, generalise to
  a patch / container-retirement date table as `tier × event-window` regime shifts.
- **Why first:** it costs runner-minutes only and it is the one bucket-C mechanism our design
  structurally cannot express today.

### 2. Read `q_hat` / PT off the next retrain — free, closes item 2's open leg

End-anchoring the CV grid moves two served quantities — the band width (`q_hat` is calibrated on the
out-of-fold residuals the new windows produce) and the PT sample (`invariant_4_signal` must be
**re-read**, not carried). Neither has been read on the corrected grid.

- **Cost:** zero marginal — read `meta.json` (`q_hat` / `fold_q_hat`, `invariant_4_signal`) after
  the next scheduled retrain.
- **Why:** a `q_hat` fitted on a window ending 2026-02-09 and served in 2026-08 carried the exact
  misalignment the grid fix removes; confirm the fix landed where it was supposed to.

### 3. Cross-market basis diagnostic — genuinely untested, but discount for the transfer record

Our served price is a **blended** `aggregator_sync` construct, so there is no Steam−Buff spread to
mean-revert against — the basis signal was **never actually tested**. The per-venue components (Buff
bid, Skinport live) are already ingested.

- **Do:** keep venues un-blended; test `spread = P_buff − P_steam` as a convergence predictor
  **offline first**, at ~3 min, before any retrain is justified.
- ⚠️ **Discount the prior.** C1 and lambdarank were both CV+/serving−; relative-value is 0-for-2 on
  serving transfer. Do not budget a retrain until a cheap diagnostic clears, and pair any live claim
  with the 15% Steam fee + 35%→5% spread — a basis edge must clear costs to be tradeable.

---

## Worth doing as hygiene — not experiments, none urgent

Ranked, but all below the accuracy work above because the retrain is inside the 30-minute cap.

- **Delete the regime branch (item 6).** Under the current allowlist `market_return_30d` is never
  engineered, so every row labels `range` and the regime booster is **byte-identical** to the global
  one (md5 match 4/4). CI already skips it; every local / research retrain and A/B still pays 95.4s
  of 872s (10.9%). If kept, gate it on the **allowlist**, not an env flag. Also a live footgun for
  any future C4 work.
- **Dead-code removal (item 8)** and **cost hygiene (item 7)** — the 11 always-zero volume features,
  the stale-comment-protected `distance_to_*` / `high_low_range_30d`, `label_vol_30d`, the unreachable
  `MOMENTUM_FALLBACK` / `SAMPLE_WEIGHT_HALFLIFE` branches, the per-run `nvidia-smi` probe; and the
  four cost levers (split-conformal calibration, `skip_unused_groups` on predict, the redundant
  `prepare_targets` scans, the repeated `_compute_sample_weights`). All uncontroversial, none
  urgent. ⚠️ Split-conformal calibration must carve the calibration split **before** HP selection or
  it recreates the under-coverage bug.

---

## Do not run — refuted, underpowered, or chasing a dead number

- **lambdarank as a serving change (item 5 / web #1)** — refuted above; 1/4 transfer, net-negative
  after costs, h=14 untrainable. Do not re-run at this data/horizon setup.
- **C4 — rebuild the feature allowlist (item 3).** Its founding **+3.5pp is null** on the honest ≥$1
  paired read (−0.35pp [−4.15, +3.18]; the `no_random_k` placebo beats removing either real group).
  Rebuilding is ~75 min/arm to chase a number that is already gone.
- **Re-running the null / negative A/B arms (item 4 remainder).** The early-stopping leak pays
  **capacity**, so only *positive* arms were ever suspect. Recency (the one genuine positive) is
  done. Re-running ten harnesses × four horizons is expensive and low-information.
- **N2 exogenous leg, C3 residual-reversal, idiosyncratic mean-reversion (web #5).** N2's own-history
  leg is closed in both directions; the exogenous leg is a market-wide common factor (a true null by
  construction), and its only live-variance feed, FX(CNY), is ~10× too small. The `−return_1d` win
  that C3 rests on is a bid-ask-bounce quote artifact that will not survive the 15% fee.
- **A fourth band-width scale or any date-conditional `q_hat` (C5).** Three width scales (`sigma`,
  `sigma**beta`, learned) all calibrate to exactly 80% on their own records and land elsewhere when
  served; the date-level rescaling was pre-registered and **failed 3/3**. The residual defect is a
  trailing-vs-forward volatility gap that no within-date reweighting can fix. The width variable is
  not the lever. **The over-coverage LEVEL is the only open quantity** and it is a hard one.
- **Supply depth as a 2nd-moment signal (web #4).** Permissible and genuinely untested (Δlistings vs
  `|return|` / realized vol, on the ~71%-cohort sidecars we already collect), but it is a band-width
  play, and band-width work has a long non-transfer record. Low priority, not "significant impact."

---

## Carried forward unchanged

`O2` (config-description half), `G2` Part 2, `D2`–`D5` (unblocked, unstarted), and the two
serving-hygiene items (exclude 2026-07-19 as a distinct serving config; WARN + flat-call count on
`predict()`'s no-classifier fallback). `D5` — consolidating the harnesses — is worth re-reading
against any future C7 re-read, but C7 itself is de-prioritised above.
