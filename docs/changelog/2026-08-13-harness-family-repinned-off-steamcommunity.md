# The A/B harness family repinned off STEAMCOMMUNITY and onto production's model

**Date:** 2026-08-13
**Change:** nine `backend/scripts/ab_test_*.py` harnesses edited. No harness re-run yet — this is
the repair `research/2026-08-13-next-steps.md` item 0 requires *before* any re-run.
**Bears on:** `C7`, item 0b (supply-depth re-read), item 3 (`C4`), `D5`.
**Implements:** `changelog/2026-08-13-the-null-verdicts-were-not-all-tested.md` and next-steps item 0.

## What was broken

`source = 'STEAMCOMMUNITY'` matches **0 rows** in the archive post rebuild — values are `NULL`
(pre-2026, ends 2025-12-31) and thirteen `aggregator_*` feeds. Three harnesses
(`supply_side`, `regime`, `ensemble`) put that predicate inside an `item_slug IN (...)` subquery
and so selected **zero items** — unrunnable. Four more (`feature_contribution`, `direction_labels`,
`q50_sampling`, `interval_sampling`) degenerated to a NULL-only (pre-2026) frame. Separately, six
harnesses never applied `_apply_feature_allowlist`, so they measured a 138+-column model where
production serves **33 of 123**; and `supply_side` / `feature_contribution` applied the `≥$1` filter
only at scoring while training on a pool that is ~92% penny items.

## What changed

**Repin (all 7).** `STEAMCOMMUNITY` → `source IS NULL` semantics, which is the backfilled
(pre-2026) series the filter always meant. Verified against the local archive: `source IS NULL`
selects **5,536 items** (`regime` went 0 → 5,536). A **non-zero row assertion** was added to every
universe query so an empty predicate now fails loudly instead of returning an empty set. Target
confirmed against the two production-correct harnesses (`csfloat_basis`, `item_metadata`), which
already pin `(source IS NULL OR source = 'aggregator_sync')` — i.e. this repin matches working
production code, not a predicate invented from a stale local view.

**Allowlist — applied five ways, deferred once.** The correct application depends on what each arm
varies:
- `regime`, `ensemble` vary *training config*, not features → both arms get the fixed production
  allowlist (`price_technicals`, 33 cols).
- `price_primitives`, `volume_features` add *shelved price_technicals* primitives → base =
  shelve+allowlist (33 cols), treatment/placebo add the primitives back explicitly.
- `supply_side` adds *rarity* (`item_identity`, which the allowlist would otherwise strip) → both
  arms get the 33-col `price_technicals` base; treatment adds back only the `rarity_*` columns.
  Verified on a synthetic frame with the real monkey-patch: **control = 33, treatment = 33 + 12
  rarity, rarity present only in treatment.**
- `feature_contribution` (`C4`) **deliberately keeps no fixed allowlist** — it measures the marginal
  contribution of *non-allowlisted* groups (`cross_sectional`, `events`) by ablation, and a fixed
  allowlist would collapse `no_cross_sectional` onto `full`. The honest form (base = production 33
  cols, arm = base + the group under test) is `C4`'s rebuild, item 3 — recorded in an in-code note.

**≥$1 floor lifted to universe selection** in `supply_side` and `feature_contribution` via
`MEDIAN(mean_price) >= 1.0` in the pool `HAVING`, matching `item_metadata`'s `MIN_MEDIAN_PRICE`.
Measured: the old `ORDER BY row_count DESC LIMIT 200` pool held **14 of 200** items ≥$1; the
universe-level filter yields **871** ≥$1 items.

**Tiebreaker** `, item_slug` added to the four pools that lacked it (`supply_side`,
`feature_contribution`, `regime`, `ensemble`); `price_primitives` / `volume_features` already had it.

**Placebo arm** added to `supply_side`: a third arm carrying the rarity columns but permuting each
per fold (seeded per fold so all three arms score the same rows), reported as a paired contrast vs
control alongside treatment. The 2026-08-13 leak audit found the early-stopping leak pays
*capacity*, so a real rarity effect must beat shuffled rarity, not merely the no-rarity control.

## Not done, and why

- **`feature_contribution`'s allowlist / arm redesign** — item 3 (`C4`). A fixed allowlist here
  breaks the ablation; the rebuild onto a production-base contrast is the separate, larger job the
  next-steps doc already scopes.
- **`training_breadth`'s placebo** — deferred. A correct capacity control for *nested breadth* arms
  evaluated on held-out items is a research-design question, not a mechanical repair, and a
  mis-specified placebo would be cited as a control it is not. It is also lower-urgency: the
  capacity leak it would guard against is gone under the fixed trainer (`EARLY_STOPPING` off by
  default). Suggested design when taken up: a `wide_shuffled` arm at the wide item set with the
  added items' targets permuted, so breadth is matched but the added items carry no signal.

## Verification

All nine files byte-compile and import. The repin selection (5,536 items) and the ≥$1 lift
(871 items) were measured directly against `price-archive/prices-*.parquet`. The `supply_side` arm
column counts (33 / 33+12) were verified on a synthetic frame exercising the real monkey-patch. No
harness has been run end-to-end — a full run is the next action (item 0b: re-read supply depth
first). ⚠️ Verify the pins against the durable `cs2-oracle-data` archive before any CI re-run; this
work was confirmed against the local mirror, which runs behind it (the pre-2026 NULL history is
stable, so the finding is safe, but the durable copy is the source of truth for CI).
