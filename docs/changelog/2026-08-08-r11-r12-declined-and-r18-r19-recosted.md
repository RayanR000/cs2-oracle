# R11 and R12 declined on coverage, R18 and R19 re-costed in opposite directions, and the Steam backfill has a second blocker

**Date:** 2026-08-08
**Plan:** `docs/research/2026-08-07-next-steps.md`, the R11–R19 section
**Scope:** documentation and verification only — **no code changed**. Every number below was
measured today against the local archive, the staged SQLite DB, or a live probe.

Four of the six never-tracked §10 items had effort or feasibility estimates that do not survive
being checked. Two are now decided, two keep their rank at a corrected cost, and one new blocker
came out of the checking.

## 1. R11 (kieranpoc Kaggle dump) — declined, and not on licence

The dump is **frozen**: `dateModified` 2024-06-15, data snapshot 2024-05-04, 901,195,556 bytes.
The archive's `volume` column last carried a non-zero value on **2026-04-15** and is zero on
100% of rows for 2026-05 through 2026-08 (verified across `prices-2026-*.parquet`). So the dump
supplies nothing for **2024-06 → 2026-08**, which is the only window that needs it — that is the
whole decision.

Both legs of its stated purpose (§24 rank 16, `sale_count_24h`, the counting-noise denominator)
are covered elsewhere:

- **Forward:** `collectors/sales_volume.py`, wired 2026-08-08. Not re-verified here — the
  endpoint answered **403 with a 422 KB HTML challenge** to this machine's egress, which is the
  WAF behaviour that module's own docstring documents for Cloudflare-owned egress. That is a
  statement about where the call came from, not about the feed.
- **Historical:** the Steam listing pages, and this is demonstrated rather than argued —
  `runtime/steam_listing_history.db` already holds **262 items / 528,573 daily rows back to
  2013-08-15**, with `purchases` beside `price_median`. The dump's own source is Steam.

The licence is **CC BY-NC-SA 4.0** (read off the page; the review's mark was "unconfirmed"). It
is recorded because the review asked for it, and it played no part in the decision.

## 2. R12 (atalantus listing counts) — declined on fold count and splicability

The listing-count window is 2023-01-25 → 2024-01-19, **359 days**. Against
`CV_STEP_DAYS = 150` / `CV_MIN_TRAIN_DAYS = 200` that is **~2 folds**, so it does technically
break the zero-additional-folds arithmetic the entry was written to break. That is the strongest
thing available for it.

It cannot join the served present. Live supply depth begins **2026-08-06** (`supply-2026-08.parquet`
holds that one day, 30,330 items), leaving a **2.5-year gap**, a different venue (BUFF, CNY) and a
different quantity than the lis-skins / market.csgo / Waxpeer counts. So R12 buys a **history-only
~2-fold A/B**, which against the 2.21–3.69pp item-level MDE resolves `unresolved` by construction.
Also corrected in `data-sources.md`: the raw dump is **113 MB via Git LFS**, not 24 MB xz.

## 3. R18 is harder than "one query away", on three counts

The entry claimed the per-regime coverage read was one query from being decidable. It is not.

1. **No regime window is defined in the code at all.** `REGIME_WINDOWS`, `regime_window` and
   `regime_stress` return zero hits across every `.py` in the repo. §12's dated breaks exist only
   in prose, so the windows have to be written before anything can be grouped by them.
2. **The walkforward gate cannot supply the number, deliberately.** Every arm gets
   `PLACEHOLDER_BAND_PCT = 10.0`, the gate never calls `models/conformal.py`, and
   `interval_coverage` is neither logged nor persisted there — `walkforward_backtest.py:195-201`
   and the omission comment at `:621-623` state the reason: the figure would describe the
   placeholder, not a model. Realised coverage can only come from `scripts/backtest_accuracy.py`,
   which scores the actually-served band.
3. **Those outcomes span 6 forecast dates**, 2025-12-01 → 2026-07-19
   (`ops/forecast_outcomes.parquet`), in two clusters — a backdated batch and one week of July.
   Two clusters cannot be partitioned into regime windows.

R18 is therefore gated on the same `MIN_FORECAST_DATES` calendar wait as everything else, plus a
definition that has to be authored first.

## 4. R19's instrument does not have to be built

R19 says there is no CPCV path in this repo and that building Track B is medium. The first half
is still true; the second is now wrong. **`purgedcv`** (`github.com/eslazarev/purged-cross-validation`,
**MIT**, PyPI **0.1.3**, also conda-forge) exports `CombinatorialPurgedCV`, `PurgedGroupKFold`,
`PurgedKFold`, `WalkForwardSplit`, `purge`, `apply_embargo`, `deflated_sharpe_ratio` and
`probability_of_backtest_overfitting` (PBO via CSCV) — both Bailey/López de Prado instruments plus
the fold geometry. All four splitters follow the sklearn splitter protocol; dependencies are
numpy / pandas / scikit-learn / scipy at `python >= 3.10`, so nothing new enters the image. It
exists because mlfinlab, the canonical implementation, went closed-source.

Two caveats before leaning on it. It is at **0.1.3**, which is young for a dependency a ship
decision rests on. And adopting the splitter leaves this repo's own work: CPCV has to be fed the
same `cluster_key` fold geometry `backtest/paired_mde.py` uses, or the deflation runs on a
different clustering than the intervals it deflates. Re-costed as **small-to-medium, mostly
wiring**.

## 5. New: 5d, two things blocking the Steam listing backfill

Found while re-deriving R11. This gates R11, R13's expensive half, and 5c.

**(a) `load_targets` raises on `--min-price`.** It filters `HAVING MAX(median_price)`, and the
2026-08-08 schema normalisation left the archive with `mean_price`, so the script's own documented
usage line dies on a DuckDB `BinderException`. It also reads through a raw
`SELECT * FROM read_parquet('prices-*.parquet')` — invariant 1 in `backend/AGENTS.md`, and exactly
the mechanism by which a renamed column vanishes without an error. One-line fix.

Target counts on the 2026-08-07 archive day, once fixed: **32,617** non-gated name-keyed items with
no floor, **27,935** at ≥$1 (35,766 / 28,594 before the mangled-key filter removes 3,149 / 659
phantom slug rows). This **independently corroborates R13's extrapolated ~31,590**.

**(b) This IP's soft-block has not decayed in three days.** The block itself is not new — dated
2026-08-05 with "no decay over 5 h" in `2026-08-06-data-acquisition-ranking.md`, and the staged DB
stops there. What is new is that it is **still in force on 2026-08-08**, moving the decay bound
from ≥5 h to ≥3 days. Measured today: a logged-out `GET /market/listings/730/<name>` returns
302 → **5.0 MB** carrying 20 embedded `pricehistory` caches — 9 variants of one base name, daily
`price_median` + `purchases` back to **2014-02-21** — for the first ~3–4 fetches, then flips to the
shell (200, **230 KB**, no redirect, zero `pricehistory`) and stays flipped at every idle interval
probed: **60 s, 180 s, 360 s and 660 s all returned the shell**, byte-identical to within one byte.
Whatever the recovery period is, it is longer than 11 minutes of idling on top of a three-day-old
block.

The flip is **not header-shaped**, which was worth ruling out: all three rotated user-agents and
the `Accept` / `Accept-Language` / `Accept-Encoding` variants returned the shell once flipped,
while the same request had succeeded a minute earlier. `classify()` labels it `SOFT_BLOCK`
correctly and `canary_ok` aborts before request 1, so **the guard is working** — the earlier
reading of this as a code fault was wrong.

**What the staged data does establish:** 47 requests harvested 262 items, i.e. **5.6 series per
request**, above the docstring's ~4.5. At its ~82% target share that is ~4.6 targets/request, so
32,617 targets is **~7,100 requests**, ~5.7 h at `REQUEST_DELAY = 2.5` *if the block allowed it*.
It does not, so the delay is not the binding constraint and no plan should quote that figure as a
schedule.

## 6. Two smaller corrections

- **6b is done: verified absent.** No occurrence of `api.dmarket.com/exchange/v1` anywhere.
  `dmarket` appears only as fee constants in `backtest/friction.py` and its test — no dead
  integration to remove.
- **The "free bug fix" is not free.** The prefix mismatch is real: `_feature_group` matches
  `support_` while the columns are `distance_to_support` / `distance_to_resistance` /
  `high_low_range_30d` (`models/forecaster.py:201` against `:1416-1420`), so they group as `other`
  and `FEATURE_GROUP_ALLOWLIST = ["price_technicals"]` drops them — confirmed by running
  `_apply_feature_allowlist` on them. But the 2026-07-24 ablation that set that allowlist ran with
  these three already absent, so fixing the prefix restores nothing measured; it **admits three
  never-measured features**. One line plus an A/B against a 2.21–3.69pp MDE.

## What this does not do

No code changed, so nothing here alters a model, a label or a stored number. R13 is untouched in
rank — its cheap half still runs through step 3's `FLOOR_SWEEP` and its expensive half is now
explicitly blocked on egress rather than on effort. The Borri/Liu/Tsyvinski citation in step 11
was re-checked and is **still SSRN 4052045 with no journal version**, so the "do not cite as
published" caveat stands.
