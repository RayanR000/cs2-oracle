# The "quoting artifact" composition test was reading a NULL as a change

**Date:** 2026-08-09
**Type:** refutation — research instrument + published result

## Why

`docs/research/2026-08-08-model-review.md` §5 tested whether the project's reversal signal
(`-return_1d` predicting the forward return, rank IC ≈ 0.17 at 3d) is a tradeable return or an
artifact of the archive's consensus price changing measurement basis. It reported that the
signal fell from **+0.1676 to +0.1006** once source composition was held stable, and to
**+0.0044 on 28 dates** on the cleanest subset — and concluded that this "gates everything
else", because a label that is not a return cannot be modelled.

Rebuilding that measurement as a committed instrument showed the composition partition was
wrong.

## The cause

The §5 partition treated an item-day whose `source` is **NULL** as a composition that never
matches itself, so every such window read as "composition changed". **`source` is NULL for
every archive row before 2026** — verified: 9,417,947 pre-2026 item-days after the universe
filter, 0 with more than one row and 0 carrying a source label. The `aggregator_*` labels begin
in 2026.

So the §5 "stable" cell contained no pre-2026 data at all. Its 188 dates are 2026 dates, and
roughly 716 of its 775 "changed" dates are pre-2026 dates marked changed for want of a label
rather than because anything changed. The measurement also applied no `_snapshot_dates` /
`_collection_shift_dates` exclusion.

This is the failure mode `backend/AGENTS.md` invariant #2 exists for — a NULL-unsafe comparison
against `source` silently reclassifying thirteen years — appearing in an analysis rather than in
a loader.

**The published fall from +0.1676 to +0.1006 is a 2013-2025 → 2026 regime difference, not
composition control.** Restricted to 2026, the *all-rows* rank IC — no composition control of
any kind — is **+0.1023 on 185 dates**, against §5's "stable" +0.1006 on 188.

The diagnosis is not inference: the new instrument reproduces §5's **188** and **185** date
counts *exactly* when the NULL-never-equal rule is applied to it, on two independent cells.

## The corrected answer

Measured over 2026 — the only era whose rows carry a source label — with composition defined as
the **set** of source names voting on each item-day. Each cell has its own date count; "all
rows" and "composition stable" are not drawn from the same dates, because a handful of dates
fall into the changed or gapped cells instead:

| | all rows | dates | composition stable | dates |
|---|---|---|---|---|
| 3d | +0.1023 | 185 | **+0.1027** | 181 |
| 7d | +0.0842 | 172 | **+0.0842** | 167 |

Holding composition still does not touch the signal. The same holds over the whole archive on
the count basis — which assumes a NULL `source` is one constant source, so 2013-2025 reads as
composition-stable by construction (+0.1941 vs +0.1937 at 3d, on 4,602 vs 4,598 dates). **The
reversal is not a composition artifact, and accuracy work is not gated on this** — with the
caveat that this conclusion rests on the stable population matching the unconditional one, not on
a measured contrast against a changed population (next paragraph).

Two parts of §5 survive, and both are its weakest: the "composition changed (present)" cell —
composition demonstrably different with every window day observed, as opposed to a window that
merely had a missing day — is **25 dates at 3d and 19 at 7d**, and the "stable and ≥3 agreeing
sources" cell is the same **25 dates at 3d and 19 at 7d**. Both are under the 30-date reporting
floor, so no number is quoted for either. There is consequently no dataset in this archive on
which the reversal's behaviour under a *known* composition change can be quoted — the "not a
composition artifact" conclusion above is supported by the stable cell equalling the
unconditional cell, not by a stable-vs-changed contrast. The strictest forms of the question are
blocked on calendar time — the multi-source era is a few dozen days deep — not on effort.

## What changed

- **New:** `backend/scripts/measure_composition_stability.py` — rank IC of `-r_t` against the
  forward h-day return on the voted daily series, partitioned three ways over `t−1 … t+h`:
  composition stable, composition changed with every window day present, and window incomplete
  (a gap, which is not evidence about composition at all). Also reports a paired
  stable-vs-changed difference on the dates both cells occupy, and takes `--min-items-per-date`
  as a flag so that sensitivity can be published rather than estimated. Reads through
  `prices_relation` + `archive_universe_sql_filter`, votes through
  `ItemForecaster._apply_multi_source_voting` rather than reimplementing it, opens no database
  session, and applies production's endpoint (snapshot) and span (collection-shift) voiding
  rules itself. Two composition bases: the **set** of source names (default) and the **count**
  (`n_ask_sources`). `_apply_multi_source_voting` and `VOTED_CACHE_VERSION` were deliberately
  left untouched.
- **New:** `backend/tests/test_composition_stability.py` — the stability predicate, the
  within-date IC aggregation, the void rules, and the source-set encoding.
- **New:** `docs/research/2026-08-09-composition-stability.md` — the corrected measurement.
- **Marked:** `docs/research/2026-08-08-model-review.md` §5 — composition rows struck through
  and annotated. The numbers are kept, not deleted; they are reproducible on demand.

## Also recorded

`_collection_shift_dates` is a function of the item universe it is handed, which had not been
written down. On the backfilled-only frame it fires on 5 dates in 2024-2026
(`2026-03-22, 07-09, 07-10, 07-11, 07-12`); on the full-universe frame it also fires on
**`2026-04-16`, `2026-07-14` and `2026-07-15`**. "Which dates are void" is therefore not a
property of the archive alone. Neither detector was modified.
