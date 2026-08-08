# Frozen-price runs voided from the label set; `stale_run_days` measured

**Date:** 2026-08-08
**Step 6** of `docs/research/2026-08-07-next-steps.md`. Review §22 D2, §10 Tier 3 #15.
**Spec:** `docs/superpowers/specs/2026-08-08-frozen-price-runs-design.md`.

A label measured to or from a bit-identical price run is the Getmansky–Lo–Makarov MA(k)
artifact, not a market move. `models/staleness.py` now counts those runs, `prepare_targets`
voids the labels that touch one on either leg, and `forecast_outcomes` carries the anchor's run
length as a frozen column so the backtest can report a staleness axis.

## The premise did not survive re-measurement, and that is the main result

Two figures the step was justified on are wrong as applied. Both were re-measured on the voted
series (13,657,848 item-days, 41,534 items, 2013-08-14 → 2026-08-07, universe predicate imported
live from `models/item_parser.py`).

**1. "0–1.8% at every tier ≥$1" is a different quantity.** It was measured on resolved backtest
anchors, which are 3-day smoothed medians — a smoothed anchor is almost never bit-identical to
the one before it. On the raw voted daily series, which is what the label path actually sees,
≥$1 staleness is **12–27%** (2024+: `$1–5` 26.65%, `$5–20` 19.36%, `$20–100` 13.09%,
`$100–1000` 12.24%, `$1000+` 17.92%). Both numbers are real. **Neither may be used to size the
other**, and any future citation has to say which series it means.

**2. It is a 2026 feed property, not a market fact.** Holding the item set fixed to what was
already live in Q4 2025, the ≥$1 `stale_run_days >= 1` rate runs `2025-10 0.54% · 11 0.57% ·
12 0.76% → 2026-01 6.01% · 04 18.58% · 05 30.11% · 06 33.11% · 07 28.03%`. So the rule voids
**~30% of 2026 ≥$1 labels and ~0.6% of everything before 2024**. Defensible — 2026 is the window
production trains and serves on — but it is not the even 13-year cleaning the step implies.

Labels voided, 2024+, anchor rule alone: **19.20 / 18.19 / 16.00 / 13.78%** at h=3/7/14/30 on the
≥$1 subset; **39.80 / 39.48 / 38.76 / 38.77%** on the whole unfiltered universe.

**The rule is not surgical.** It takes **13.7–15.9% of *non-zero* ≥$1 labels** with it
(20.3–22.5% on the whole universe). It can be net-harmful, which is why it is verified on
interval width rather than on accuracy.

## Why this is not a source-exclusion change

The cheaper alternative was measured and rejected as a *substitute*, not as an idea.

`aggregator_steam_7d/30d/90d` are Steam's trailing-window **mean sale price** — MA(7), MA(30),
MA(90) — voting on equal terms against point-in-time asks
(`collectors/csgotrader_aggregator.py:308-312`, `collectors/pipeline.py:132-135`). That is the
MA(k) mechanism installed as an ingest decision, and it is a basis error by the same argument
that removed `aggregator_buff163_buy`.

It cannot be step 6's fix, because the smoothing is in the **fields**, not the three feeds.
`aggregator_sync` is `last_24h` falling back to `last_7d`→`30d`→`90d` and is bit-equal to
`last_7d` on **41.98%** of the 319,269 ≥$1 item-days where both exist;
`aggregator_steam_17mafo` is the same construct (`scripts/merge_17mafo_gap.py:57-60`).
**There is no point-in-time Steam price in this archive**, and the fallback fires precisely on
the illiquid items.

Sized: dropping the three window feeds costs almost no coverage (670 lost item-days of
3,093,793) but clears only **2.30pp of the 20.25pp** ≥$1 stale rate — about 11% — while moving
the voted median on **17.13%** of 2026 ≥$1 item-days (median **−7.16%** where it moves, 29.8% of
moves upward) and flipping **5.75%** of consecutive-day return directions. Roughly half the bid
exclusion's magnitude, same character. Dropping `aggregator_sync` too buys a further 1.4pp and
**deletes 2026-01 and 2026-02 in full** for the ≥$1 cohort (52,048 item-days) — it is the only
source in that window.

**Disposition:** the run-length rule is the more robust lever because it is mechanism-agnostic —
it catches the fallback-to-MA wherever it hides. The source exclusion is a real, separate,
**not-yet-taken** decision that needs its own level-displacement measurement and its own
`VOTED_CACHE_VERSION` bump, and it subsumes the `aggregator_steam_17mafo` item still open from
step 1.

## What landed

- **`models/staleness.py`** — `stale_run_days` / `stale_run_lookup`. A pure module, because both
  the label path and `scripts/backtest_accuracy.py` need it and the latter should not import a
  6,000-line trainer for a run-length scan. A gap wider than `MAX_WINDOW_SPAN_DAYS` (7) **breaks**
  a run: nothing was observed across a collection outage to be frozen. That costs 4.06% of
  consecutive pairs on the 2024+ ≥$1 cohort. Cost is 1.65 s over 13.66M rows.
- **`LABEL_MAX_STALE_RUN_DAYS = 0`** in `models/forecaster.py`, applied in `prepare_targets`
  inside the existing `bad` mask beside `_snapshot_dates` / `_collection_shift_dates`. `None`
  disables it. **Both legs**, matching the snapshot rule's endpoint semantics: the anchor leg
  alone catches 65–72% of the exact-zero ≥$1 return mass at h=3–14 and **31.22% at h=30**;
  anchor-or-target reaches 86–91% and 51.98%. The h=30 residual is deliberate — roughly half of
  30d zero returns are genuine round trips and are labels, not artifacts.
- **Migration `0021`**, `forecast_outcomes.base_stale_run_days`, written by `resolve_outcomes`,
  absent from `_REFRESH_VERDICTS_SQL`, moved only by `--reresolve`. Historical rows stay **NULL**;
  backfilling would mean a full `--reresolve`, which unfreezes every `base_price`/`actual_price`.
- **`score_by_staleness`** in `backtest/scoring.py`, published as `metrics["staleness_bands"]`.
  This is the axis `2026-08-07-friction-conditioned-tier-scoring-design.md` deferred to step 6,
  but as **four fixed bands plus `unknown`, not quartiles** — ~80% of the ≥$1 cohort sits at zero,
  so `q1 = q2 = q3 = 0` and a data-driven cut collapses to one populated bucket while still being
  reported as four.
- **`scripts/ab_test_frozen_runs.py`** — the verification. Not an on/off A/B: the rule voids rows,
  so the arms would pair only on the intersection, which is the rows the rule did not touch.
  Instead it runs `compute_mde.py`'s seed-only placebo design **once per label regime** and
  compares the two noise floors.
- **1,532 tests pass**, up from 1,466.

## Caught while building

- **A date-keyed stale set would have emptied the label set.** The first draft collapsed stale
  rows to a set of *dates*; staleness is per (item, day), so at a ~20% archive-wide rate that
  would have voided nearly every item's label on nearly every day. Locked by
  `test_a_frozen_run_voids_only_the_frozen_item`.
- **The `NO_ARCHIVE_READ` exemption in `test_ab_harness_universe.py` was an unguarded hole.**
  Adding a name to it disabled every universe check for that module. It now has to be earned:
  `test_the_exempt_harnesses_really_do_not_read_the_archive` fails if an exempt module contains
  an archive read. The check distinguishes DuckDB's `read_parquet` (an archive glob) from
  `pd.read_parquet` (a caller-named file, which is how `recency_weights` takes `--frame`).

## Not done

- **Nothing was re-scored or re-resolved.** `base_stale_run_days` is NULL on every stored
  outcome, so `staleness_bands` reads 100% `unknown` in production until new outcomes resolve.
- **No retrain.** The rule takes effect at the next `mode=full` Monday run.
- **`stale_run_days` is not a model feature.** That is step 11, and it carries a specific hazard:
  the source mix changed four times in 2026 (Jan–Feb `sync`-only; Mar–Apr multi-source; May–Jun
  `17mafo`-only; Jul+ eleven sources), so a feature on run length partly encodes the collection
  schedule. **Leak-adjacent — check before shipping it.**
- **The MA-feed exclusion was not made.** See above.

## Relationship to step 7

Step 7 (`TRAIN_MIN_MEDIAN_PRICE = 1.0`) removes the sub-$1 items outright. This rule drops ~39%
of labels on the whole universe and 13.8–19.2% on the ≥$1 subset. **The two overlap heavily and
are not additive** — once step 7 lands, step 6's measured effect falls to roughly a third of what
it is today. Do not sum a before/after across both.
