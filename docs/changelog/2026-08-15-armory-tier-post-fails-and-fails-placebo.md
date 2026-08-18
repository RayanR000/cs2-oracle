# The Armory `tier × post` natural experiment fails its bar and its placebo

**Date:** 2026-08-15. Read-only archive analysis, no code/model/prod change.
**Pre-registration:** `docs/research/2026-08-15-armory-tier-post-preregistration.md` (design fixed
before any return was scored). **Query:** `scratchpad/armory_tier_post.py`.

## Context

`2026-08-14-next-steps.md` item 1 — the Oct-22-2025 `tier × post` trade-up shock — was the top-ranked
untested idea. It is **data-blocked**: the only panel spanning Oct 2025 carries 10 ★ and 45 covert
items. Re-anchored to the **2026-07-08 "Season 5, Armory" update** (+ 07-16 "Call II Arms-ory"), the
same class of economy shock, inside the broad ~32K-item panel (3,272 knives/gloves, 850 coverts,
5,452 control-mid usable).

## Result — FAIL on the pre-registered bar

OLS `r_i ~ 1 + I(knife_glove) + I(covert)`, reference `control_mid`, HC3 SEs. Settled post window.

| leg | event β | event t | placebo β | placebo t |
|---|---|---|---|---|
| knife_glove | +0.0165 | 2.55 | +0.0240 | **7.45** |
| covert | +0.0477 | **6.67** | +0.0076 | 1.81 |
| intercept (control_mid level) | +0.0447 (rose) | 9.91 | −0.0341 (fell) | −14.58 |

- **Primary fails.** The mechanism predicts `β_knife_glove < 0 < β_covert`, both `|t|>3`. Coverts
  outperformed mid by +4.8pp (t=6.67) — the covert half is event-consistent — but knives/gloves did
  **not** crash: +1.7pp vs mid, **wrong sign**, `|t|=2.55` below the hurdle.
- **Placebo gate fails (decisive).** In the quiet May window knives/gloves diverge from mid *more*
  (t=7.45) than during the Armory event. The event window rose 4.5% and the placebo window fell
  3.4% — opposite market regimes — so tiers with different market-betas separate under either. The
  knife/glove "gap" is **beta, not Armory**.

## Reading

The clean opposite-sign DiD the `tier × event` mechanism requires did not materialise. The only
survivor is a **one-sided** covert-outperformance leg (event t=6.67 vs placebo 1.81, β ~6× larger) —
suggestive that Armory helped coverts specifically, but riding on top of a +4.5% market-wide move and
without the matching knife/glove crash. A one-legged, non-placebo-clean result is not the bucket-C
structure the design cannot express, and it is not shippable.

**This closes the last untested bucket-C mechanism.** Every `tier`/category/cross-sectional axis is
now measured; the covert-only leg does not change the range-forecaster positioning
(`2026-08-15-cs2-oracle-is-a-range-forecaster.md`, `backend/AGENTS.md` invariant 4). The only lever
that would change the deliverable remains item 1 of
`2026-08-15-directional-accuracy-and-data-inventory.md` — retarget to `P(|return| > round-trip cost)`
— and it stays validation-blocked until 2026+ durable serving dates accumulate.

## Covert-leg difference-in-differences (scored 2026-08-15) — survives isolation

Follow-up on the one event-consistent leg. Per-item double difference `d_i = r_event − r_placebo`
(removes each item's level and market-beta), regressed on the covert dummy vs `control_mid`, with
collection fixed effects and SEs clustered by `type_meta_collection_id`. Sample: 810 covert / 5,087
control items, 91 collections.

| estimator | covert−control DiD | t (clustered) |
|---|---|---|
| mean d | covert +11.4% vs control +5.8% | — |
| naive DiD, clustered by collection | +5.7% | 7.03 |
| **within-collection FE, clustered** (58 collections with both tiers, 5,215 items) | **+4.0%** | **5.12** |

The covert leg is **not** the market-beta artifact that sank the knife/glove leg: comparing coverts
to control skins *inside the same collection*, coverts still moved ~4% more, event-vs-placebo,
t=5.12 > 3. This is a real, placebo-clean, collection-isolated cross-sectional effect — the kind the
market-demeaned design cannot express.

**It still does not ship.** (1) ~4% < the ~15% round-trip fee — not tradeable. (2) It is a **single**
event (n=1 Armory shock); no folds or CI across events exist, so it is the episode-count bind in a
new guise. (3) It is one-legged — coverts up with no knife/glove crash — so it reads as a
covert-specific demand story, not the trade-up reallocation the mechanism predicted.

The only way to promote this to a testable feature is a **date-table of multiple dated economy
events** (Armory/case updates) scored as `tier × event-window` with real folds. That is worth
noting but not scheduling: it needs several more datable shocks to accumulate (2026+ calendar time),
and every relative-value arm to date is 0-for-2 on serving transfer.
</content>
