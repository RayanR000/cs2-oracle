# Six of sixteen accuracy nulls survive audit — and three harnesses select zero items

**Date:** 2026-08-13
**Change:** none. Read-only audit of `backend/scripts/ab_test_*.py` (15 files) and every
published accuracy verdict in `changelog/` and `research/`. No code edited, nothing dispatched,
no verdict re-run.
**Bears on:** the claim in `research/2026-08-13-next-steps.md` §2 that "the accuracy surface is
nearly exhausted"; `C7`; and `D5`.
**Verdict:** ❌ **the exhaustion claim is not supported.** Of ~16 accuracy verdicts, **6 survive
scrutiny, 5 were underpowered against their own MDE, and 5 were never measured at all.** Four
instrument defects are found that none of the five documented harness defects covers, and one of
them is fatal.

## 🔑 `source = 'STEAMCOMMUNITY'` matches zero rows in the archive

Measured directly against `price-archive/prices-*.parquet`:

```
SELECT count(DISTINCT item_slug) FROM {rel} WHERE source='STEAMCOMMUNITY'  -->  0
```

The real values are `NULL` (pre-2026, 9,429,275 rows, **ends 2025-12-31**) and thirteen
`aggregator_*` feeds from 2026-01-01 on. Seven harnesses pin a string that does not exist, in
two forms with very different consequences:

**Form A — zero items, unrunnable.** `ab_test_supply_side.py:106`, `ab_test_regime.py:87`,
`ab_test_ensemble.py:95` put the bare predicate inside an `item_slug IN (...)` subquery. The
subquery returns the empty set, so the enclosing query returns **0 rows**. These three cannot
execute today. Their published verdicts — the supply-side bundle, regime training, `N_ENSEMBLES`
3→1 — describe an archive state that no longer exists.

**Form B — silently pre-2026 only.** `ab_test_feature_contribution.py:100`,
`_direction_labels:100`, `_q50_sampling:131`, `_interval_sampling:184` use
`(source IS NULL OR source = 'STEAMCOMMUNITY')`, which degenerates to the NULL branch: a frame
**ending 2025-12-31 with zero 2026 rows**. That matches the stated intent —
`interval_sampling.py:176` calls it "the pre-2026 CSMarketAPI STEAMCOMMUNITY series" — so it is
not wrong, but it means C4's founding measurement never saw the **2026-03-22 consensus break**
or the **2026-07-09 feed substitution**, and nothing re-checks that the OR branch is dead.

**Likely mechanism:** `aggregator-update.yml` rebuilds the archive by orphan commit and
force-push, so a source rename rewrites all history and retroactively empties a pinned
predicate. Nothing errors — DuckDB returns an empty set. ⚠️ **This is a failure mode none of the
five stacked defects in `2026-08-13-ab-harnesses-follow-productions-trainer.md` covers**: a
verdict can silently describe a cohort the archive no longer contains.

⚠️ Confirmed on the **local** `price-archive/`, which runs behind the durable
`RayanR000/cs2-oracle-data` repo. The pre-2026 files are stable history so the NULL finding is
safe, but the durable copy should be checked before anyone edits the pins.

## Three further instrument defects

**1. Six harnesses do not apply the allowlist production serves.**
`FEATURE_GROUP_ALLOWLIST = ["price_technicals"]` keeps **33 of 123** candidate columns
(`forecaster.py:400-404`). `price_primitives`, `volume_features`, `supply_side`,
`feature_contribution`, `regime`, `ensemble` never call `_apply_feature_allowlist` — the
primitives re-read reports its own frame at "180 → 138 features". Adding six columns to a
138-column model and adding them to a 33-column model are different experiments, and dilution
biases toward null. Only `item_metadata.py:306` and `csfloat_basis.py:340` do it as production
does.

**2. Two harnesses train on the penny pool and score on a sliver.** `supply_side.py:110` and
`feature_contribution.py:116` select the universe as `ORDER BY row_count DESC LIMIT 200` and
apply `price >= 1.0` **only at scoring** (`:316`, `:363`). `volume_features.py:57-64` already
measured that exact universe: *"15 of 200 items >= $1, median item price $0.059, 8.22% of rows
in the served cohort."* So the supply-depth arm and the C4 founding number trained on ~92% penny
items. `csfloat_basis.py:291` and `item_metadata.py:249` filter at the universe level, correctly.

**3. No capacity control, and a nondeterministic pool.** `supply_side` and `training_breadth`
have **no placebo arm** — and per `the-leak-is-worth-two-points-and-it-pays-the-placebo.md` the
leak pays capacity, which is exactly what a placebo detects. Separately, `ORDER BY row_count
DESC` carries no tiebreaker in `supply_side`, `feature_contribution`, `regime`, `ensemble`;
`price_primitives:195` and `volume_features:229` add `, item_slug`. `supply_side:383-387` runs
its two arms as two separate DuckDB queries, so its row-for-row pairing rests on an unguaranteed
tie order.

## Two suspected defects that are NOT defects

Recorded because each looks like an invariant violation on a grep and is not:

- **The embargo is fine.** `csfloat_basis`, `item_metadata`, `training_breadth` and
  `frozen_runs` never mention `embargo_days`, which reads as an invariant-3 violation. They
  delegate to `_purge_overlapping_train_rows`, which applies `embargo_days(horizon)` =
  `horizon + 13` **internally** (`forecaster.py:3674`). Passing a bare `horizon` at those call
  sites is correct.
- **The phantom-slug omission is inert.** Nine harnesses apply only
  `phase_collapsed_sql_filter()` and skip `phantom_slug_sql_filter()`. That admits 78,220 rows /
  3,143 phantom slugs into the `aggregator_sync` frame — but every phantom slug has **at most 26
  distinct days** (median 26, min 21, all dated 2026-07-09 → 2026-08-08), and every harness gates
  its pool at `MIN_ITEM_DAYS` 90 or 180. Phantoms in the drawn pool: **0** at both thresholds.
  A real correctness gap in the read with **zero effect on any pool actually drawn**.
  Secondary correction to the `item-universe` rule text: the phantom rows are not a 13-year
  backlog from `migrate_historical_data.py` — they fall entirely in a 31-day 2026 window.

## The power audit

MDE and fold counts as each doc reports them.

| Verdict | Effect | MDE / folds | Class |
|---|---|---|---|
| Regime training | md5 identical 4/4 | identity, not a test | ✅ **true null** — a proof |
| Mean reversion, own history | −12.2 to −21.9pp | ~2,000 windows | ✅ **true null**, pre-registered OOS |
| Market-relative labels | −2.1 to −12.7pp | 8–9 folds | ✅ **true null** |
| Volume features 3d/14d/30d | +0.03 / +0.08 / +0.02pp | 0.20–0.46pp, 25–26 folds | ✅ **true null**, best-powered in repo |
| CSFloat basis 3d/7d | +0.28 / +0.27pp | 0.32–0.95pp, 12–13 folds | ✅ **true null** (not 14d, not 30d) |
| Allowlist `+3.5pp` claim | −0.35 [−4.15, +3.18] | 2.21–3.69pp, 25 folds | ✅ **true null for +3.5 only** |
| Six price primitives | −0.69 / −1.47 / −0.88 / −1.38 | **1.15 / 2.76 / 3.13 / 7.13** | ⚠️ **underpowered at 3 of 4**; placebo never run |
| ByMykel metadata | +0.66 → +3.89 | 2.01–10.07, 8–9 folds | ⚠️ **underpowered**; placebo read **+1.20 [+0.44, +1.96]** |
| Exogenous FX / events | −3.41 [−7.95, +1.23] | 2.21–3.69pp | ⚠️ **underpowered**; its own MDE gate never run |
| Served-cohort weighting | −0.43 / +1.18 / −0.07 / −0.97 | no MDE, 8–9 folds | ⚠️ underpowered for ±1pp; excludes the pre-registered +2pp |
| **Supply depth** | — | **none — no A/B, no folds** | ❌ **never measured** |
| tier_lead | +0.0078 rank IC | no CI, no folds, no placebo | ❌ never measured as a test |
| Ensemble 3→1 | — | — | ❌ never measured (the doc says so) |
| Recency weights | 30d **+1.17pp, 6/8 folds** | no MDE | ❌ **the 30d arm passed its gate** and was declined on a mechanism argument |
| Direction-label sweep | max-of-ten selection | no error control | ❌ no verdict is derivable |
| q50 / interval sampling | q50 was a SHIP, not a null | 8 folds, 1 member | ❌ `interval_sampling` has no published verdict |

The primitives doc says it outright: *"the gate was never falsifiable — detecting 0.5pp at 7d
would need ~795 folds."*

## What this changes

**Three items are live experiments, not closed ones:** supply depth (never tested, and both
premises that killed it have since been refuted — see
`free-bulk-listing-count-feeds-exist` and the supply-depth collector), recency weights at 30d
(passed, declined on a story), and `N_ENSEMBLES` 3→1. The primitives and ByMykel nulls are each
consistent with a real +1–2pp effect.

⚠️ **Do not re-run anything before the pins are fixed.** Three harnesses return zero rows, and a
re-read on a 138-feature penny frame answers a question production does not ask. Suggested order:

1. Repin the seven harnesses off `STEAMCOMMUNITY`, and add a **non-zero row assertion** to every
   universe query so an empty predicate fails loudly. ~30 min. This unblocks `C7` for
   `supply_side`, `regime` and `ensemble`, which are currently unrunnable.
2. Apply `_apply_feature_allowlist` in the six harnesses that skip it, and lift the `>= 1.0`
   filter in `supply_side` / `feature_contribution` from scoring to universe selection. ~1 hour.
3. Re-read supply depth first — it is the only feature in the family that was never measured.

`D5` (consolidating the fifteen harnesses) gains a fourth argument: the source pin, the
allowlist, the price floor and the tiebreaker are each spelled four different ways across the
family, and every one of those spellings is load-bearing for a published verdict.
