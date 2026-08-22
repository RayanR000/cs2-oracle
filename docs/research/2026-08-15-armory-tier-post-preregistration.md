# Pre-registered read: does the 2026-07-08 Armory update move price tiers differentially?

> **Status (as of 2026-08-21): SCORED 2026-08-15 — FAILS its bar AND its placebo.**
> `changelog/2026-08-15-armory-tier-post-fails-and-fails-placebo.md`: wrong sign on
> knives/gloves (t = 2.55) *and* the placebo window diverges more than the event window. This
> was the #1 item of `2026-08-14-next-steps.md`; its failure is what triggered
> `2026-08-16-next-steps.md`. **Event-study `tier × post` designs are closed on this data.**


**Written 2026-08-15 before any return/outcome was scored.** Only coverage and tier-membership
counts were read to fix the void thresholds below (the same order the mean-reversion prereg used).
No per-item return over any window has been computed.

## Hypothesis and provenance

`docs/research/2026-08-14-next-steps.md` item 1 proposed the **Oct-22-2025 trade-up update** as a
`tier × post` natural experiment: the largest datable category shock in the archive, moving tiers in
opposite directions (knives/gloves −~70%, Coverts +10–20×), which market-wide demeaning cancels by
construction — the one bucket-C mechanism the current design structurally cannot express.

That specific event is **data-blocked** (recorded 2026-08-15): the only price panel spanning Oct 22
2025 is the ~5,500-item CSMarketAPI-historical series, which carries **10** ★ (knife/glove) items
and **45** coverts — far too few for a `tier` coefficient to clear a `|t|>3` hurdle. The broad
~32K-item panel does not begin until **2026-03**.

This read **preserves the mechanism and re-anchors it to an event inside the broad-panel era**:
the **2026-07-08 "Season 5, Armory, and More"** update (event-news `gid 1836506165581825`), with the
follow-up **2026-07-16 "Call II Arms-ory"** (`gid 1838407329258098`). The Armory is the trade-up /
economy subsystem — the same *class* of event as Oct-22-2025 — so an Armory change is the
mechanism-relevant shock, not a cosmetic release. There were **no new case releases** in the
2026-03…08 window (the classic tier-shock trigger), so this is the strongest available substitute.

## Why this is a real test and not a re-read

Nothing is re-fitted; no production model runs. It is a pure archive read of the durable
`prices-2026-*.parquet` broad panel joined to `item-metadata-bymykel.parquet`. Coverage was checked
first and is dense — usable items per tier (≥5 daily blended prints in each window, pre-mean ≥ $1):

| tier | event window | placebo window |
|---|---|---|
| knife_glove (★) | 3,272 | 3,040 |
| covert | 850 | 815 |
| control_mid (milspec/restricted/classified) | 5,453 | 5,228 |

All three clear the 200-item floor by more than an order of magnitude at both windows.

## Tiers, fixed in advance

Assigned by joining each `item_slug` to `item-metadata-bymykel.parquet`:

- **`knife_glove`** — `item_slug LIKE '★%'` (the star prefix is the durable knife/glove marker;
  independent of the rarity join). *Expected DOWN.*
- **`covert`** — `rarity_meta = 'covert'` and not a ★ item. *Expected UP.*
- **`control_mid`** — `rarity_meta IN ('milspec','restricted','classified')` and not ★. The
  **reference** tier. *Expected ≈ flat / market-mean.*

Every other rarity (stickers, agents, music kits, base/consumer/industrial, extraordinary gloves
already caught by ★) is excluded — the tiers are pre-declared, not swept.

## Windows, fixed in advance

- **Event episode** E = 2026-07-08 → 2026-07-16 (both Armory announcements bracketed inside).
  - **Pre:** `[2026-06-24, 2026-07-07]` (14 days before the first announcement).
  - **Post (settled):** `[2026-07-24, 2026-08-06]` (8–21 days *after the second* announcement, to
    read the durable relevel, not the transient overshoot).
- **Placebo episode** P anchored at T0′ = 2026-05-01 (no Armory/economy event in range — only the
  cosmetic Cache-map return 04-28 and NIGHTMODE II music kits 05-07).
  - **Pre:** `[2026-04-17, 2026-04-30]`. **Post:** `[2026-05-09, 2026-05-22]` (identical 14d/14d
    geometry with an 8-day gap).

**Outcome per item** `i`: `r_i = ln(post_mean_i) − ln(pre_mean_i)`, where each window mean is the
average of the daily **blended** `aggregator_*` price. Inclusion: ≥5 daily prints in *each* window
of that episode and pre-mean ≥ $1.

## The bar, fixed in advance

**Primary.** OLS `r_i ~ 1 + I(knife_glove) + I(covert)` on the **event** returns, `control_mid` the
omitted reference, HC3 heteroskedastic-robust SEs. Because `r_i` already differences out each item's
own level, the tier dummy *is* the `tier × post` interaction.

- **PASS** requires **both** `|t(β_knife_glove)| > 3.0` **and** `|t(β_covert)| > 3.0` (the
  Harvey–Liu–Zhu hurdle used across this project) **and** opposite signs
  (`β_knife_glove < 0 < β_covert`).

**Placebo gate (must also hold to claim identification).** The identical regression on the
**placebo** returns must **FAIL**: neither coefficient `|t| > 3.0`, **or** each placebo `|β|` is
below one-third of its event `|β|`. If the placebo passes, the tier gap is normal-time volatility,
not the Armory shock.

Both conditions are required. Primary-passes-but-placebo-also-passes means the effect is not
identified — recorded and stopped, not developed.

**Void conditions.**
- Fewer than 200 usable items in any tier at an episode → that tier is unreadable; if a *treatment*
  tier voids, the whole read voids.
- Fewer than 5 days with the blended `aggregator_*` source present in any window → that window is
  unreadable (enforced by the ≥5-print inclusion rule).

## Robustness (reported, never gating)

1. Cluster SEs by `type_meta_collection_id` — same-collection items co-move, violating cross-item
   independence; the naive HC3 SE is a floor.
2. ≥ $5 pre-mean cohort — guards the knife/glove result against thin-print noise on expensive items.
3. Earlier pre-window `[2026-06-24 → 2026-06-24−14]` i.e. `[2026-06-10, 2026-06-23]` — guards against
   pre-announcement leak into the primary pre-window.
4. Immediate post `[2026-07-17, 2026-07-23]` vs the settled post — overshoot vs durable relevel.

## Declared confounds

- **Control is trade-up-adjacent to covert.** Restricted/classified skins feed covert trade-ups, so
  if the Armory changed trade-up odds, `control_mid` is not a cleanly untreated group. The contrast
  stays interpretable as knife-vs-mid and covert-vs-mid *differentials*, not absolute treatment
  effects.
- **Liquidity/composition.** ★ and covert items are pricier and thinner; a settled-level mean over
  ≥5 days plus the ≥$5 robustness is the defence, not a proof.
- **Two Armory announcements** (07-08, 07-16) — bracketed inside the episode; post starts 8 days
  after the second.
- **Market-wide drift** cancels in the tier contrast by construction (each `β` is relative to
  `control_mid`; a common factor shifts the intercept, not the dummies).

## Committed interpretation

- **Primary passes and placebo null** → the Armory update is a real `tier × event` cross-sectional
  shock the market-demeaned design cannot express. Next step is a `tier × economy-update` date-table
  regime feature and a paired A/B — **not a ship**, and weighted against the 0-for-2 serving-transfer
  record of every relative-value arm (C1, lambdarank).
- **Primary passes, placebo also passes** → tiers differ in normal times; the effect is not
  identified. Record and stop.
- **Primary fails** → the largest datable 2026 economy event does not move tiers differentially
  beyond noise once item-level differencing is applied. The last untested bucket-C mechanism closes
  on data, not merely on the Oct-22 feasibility block.

## Query

`scratchpad/armory_tier_post.py` (DuckDB over the durable archive; pins the blended `aggregator_*`
source, prints per-tier event/placebo coefficients, HC3 robust t-stats, and the placebo gate).
Read-only; no prod write.

## Result (scored 2026-08-15) — FAIL

Primary fails and the placebo gate fails. Coverts outperformed mid by +4.8pp (t=6.67), but
knives/gloves did not crash (+1.7pp, t=2.55, wrong sign) and diverge from mid *more* in the quiet
placebo window (t=7.45) than in the event — a market-beta artifact across two opposite-direction
regimes, not the Armory shock. Full write-up:
`docs/changelog/2026-08-15-armory-tier-post-fails-and-fails-placebo.md`.
</content>
</invoke>
