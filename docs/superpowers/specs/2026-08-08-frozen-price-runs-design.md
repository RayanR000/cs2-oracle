# Frozen-price runs: `stale_run_days`, and dropping them from the label set

**Step 6** of `docs/research/2026-08-07-next-steps.md`. Review §22 D2, §10 Tier 3 #15.

Compute consecutive bit-identical price runs, void the training labels that sit inside one,
freeze the run length onto backtest outcome rows so the staleness axis can be reported in
quartiles, and measure whether any of it narrows the paired interval.

No source-vote change. No model feature. No retrain in this step.

---

## What the measurement changed before anything was built

Two numbers in the step's own justification do not survive re-measurement on the voted series
(13,657,848 item-days, 41,534 items, 2013-08-14 → 2026-08-07; universe predicate imported live
from `models/item_parser.py::archive_universe_sql_filter`, plain median rather than production's
outlier-voted median, which is identical on the 88.8% of item-days that are single-source).

**1. "0–1.8% at every tier ≥$1" measures a different quantity.** That figure came from resolved
backtest anchors, which are 3-day smoothed medians — a smoothed anchor is almost never
bit-identical to the previous one. On the raw voted daily series, which is what the label path
sees, ≥$1 staleness is **12–27%**. Both numbers are real. Neither may be used to size the other,
and `stale_run_days` must be documented as a property of the **unsmoothed** voted series.

**2. It is a 2026 feed property, not a market fact.** Holding the item set fixed to what was
already live in Q4 2025, the ≥$1 `stale_run_days >= 1` rate runs
`2025-10 0.54% · 11 0.57% · 12 0.76% → 2026-01 6.01% · 04 18.58% · 05 30.11% · 06 33.11% · 07 28.03%`.

So the rule specified below **deletes ~30% of 2026 ≥$1 labels and ~0.6% of everything before
2024**. It is a 2026-feed filter wearing a 13-year mask. That is defensible — 2026 is the window
production trains and serves on — but it is not the even 13-year cleaning the step implies, and
a reader must not be left thinking it is.

### Prevalence, voted series, `stale_run_days >= 1 / >= 2`

| band | n (2024+) | ≥1 | ≥2 |
|---|---|---|---|
| <$0.05 | 1,905,325 | 86.00% | 81.15% |
| $0.05–1 | 2,451,682 | 30.29% | 16.49% |
| $1–5 | 1,196,291 | 26.65% | 11.75% |
| $5–20 | 814,292 | 19.36% | 9.23% |
| $20–100 | 588,489 | 13.09% | 7.22% |
| $100–1000 | 490,796 | 12.24% | 7.47% |
| $1000+ | 38,589 | 17.92% | 11.57% |

≥$1 cohort (27,410 items / 3,094,463 item-days, 2024+): **20.25% / 9.97%**, mean 0.616.

### Label impact of "drop if anchor `stale_run_days >= 1`", 2024+

| h | ≥$1 labels | dropped | whole universe | dropped |
|---|---|---|---|---|
| 3 | 2,879,800 | **19.20%** | 7,159,007 | **39.80%** |
| 7 | 2,745,272 | **18.19%** | 6,966,698 | **39.48%** |
| 14 | 2,554,221 | **16.00%** | 6,683,673 | **38.76%** |
| 30 | 2,182,093 | **13.78%** | 6,116,567 | **38.77%** |

**The collateral is the risk.** The rule takes **13.7–15.9% of *non-zero* ≥$1 labels** with it
(20.3–22.5% on the whole universe). It is not surgical, and that is precisely why the
verification below is on interval *width* and not on a point estimate.

---

## Why this is not a source-exclusion step

The obvious cheaper fix was measured and rejected as a substitute.

`aggregator_steam_7d/30d/90d` are Steam's trailing-window **mean sale price** — MA(7), MA(30),
MA(90) — voting on equal terms against point-in-time asks
(`collectors/csgotrader_aggregator.py:308-312`, `collectors/pipeline.py:132-135`). That is the
Getmansky–Lo–Makarov MA(k) mechanism installed as an ingest decision, and it is a basis error by
the same argument that removed `aggregator_buff163_buy`.

It is nonetheless **not** step 6's fix, because the smoothing is in the *fields*, not the three
feeds. `aggregator_sync` is `last_24h` **falling back** to `last_7d` → `30d` → `90d`, and is
bit-equal to `last_7d` on 41.98% of the 319,269 ≥$1 item-days where both exist.
`aggregator_steam_17mafo` is the same construct. **There is no point-in-time Steam price in this
archive**, and the fallback fires precisely on the illiquid items.

Sized: dropping the three window feeds is free on coverage (670 lost item-days of 3,093,793) but
clears only **2.30pp of the 20.25pp** — about 11% of the stale mass — while moving the voted
median on **17.13%** of ≥$1 2026 item-days (median **−7.16%** where it moves, 29.8% of moves
upward) and flipping **5.75%** of consecutive-day return directions. Roughly half the bid
exclusion's magnitude, same character. Dropping `aggregator_sync` as well buys a further 1.4pp and
**deletes 2026-01 and 2026-02 in full** for the ≥$1 cohort (52,048 item-days, `sync` is the only
source in that window) — do not do it.

**Disposition:** the run-length rule is the more robust lever because it is mechanism-agnostic —
it catches the fallback-to-MA wherever it hides, without an inventory of which field each feed
degraded to on which day. The source exclusion is a real and separate finding; it belongs in its
own step with its own level-displacement measurement and its own `VOTED_CACHE_VERSION` bump, and
it subsumes the `aggregator_steam_17mafo` item still open from step 1. It is **not** built here.

---

## Design

### 1. `models/staleness.py` — one pure function

```python
stale_run_days(df, *, item_col, date_col, price_col) -> pd.Series
```

Per item, over rows sorted by date: the count of consecutive preceding observed rows carrying a
bit-identical price. `0` on a fresh level. A new module rather than a method on
`ItemForecaster` because both the label path and the backtest resolution path need it, and
`scripts/backtest_accuracy.py` should not import a 6,000-line trainer for a run-length scan.

**A gap longer than `MAX_WINDOW_SPAN_DAYS` (7) breaks the run.** Identical prices either side of a
collector outage are not evidence of a frozen feed, and 7 days is the single staleness convention
already derived from `collectors.pipeline.FALLBACK_MAX_AGE_DAYS` everywhere else in the codebase.
This is not free: **4.06%** of consecutive observed pairs on the 2024+ ≥$1 cohort are more than
one calendar day apart, concentrated in 2026 (the four missing days are 2026-07-27, 07-30, 08-02,
08-03). Like `embargo_days`, the width therefore tracks an env-overridable constant and is not
itself a constant.

Cost is not a consideration: 1.65 s in pandas over the full 13.66M-row frame, against a 206 s
daily step. Cast the slug to `category` first — the frame is 1.47 GB with object dtype.

### 2. The label drop, in `prepare_targets`

Folded into the existing `bad` mask beside `_snapshot_dates` / `_collection_shift_dates`, and
reported in the same log line. `_snapshot_dates` voids a day where the *cross-section* repeats;
this voids a row where the *item* repeats. Same block, same idiom.

**Both legs, not just the anchor.** The anchor rule targets the MA(k) mechanism (a stale anchor
under-reports, so the next move is a catch-up); the target rule targets a fabricated endpoint.
Measured, anchor-only catches 65–72% of the exact-zero ≥$1 return mass at h=3–14 and **31.22% at
h=30**; anchor-or-target reaches 86–91% and 51.98%. `_snapshot_dates` already applies exactly this
endpoint rule, so the two read alike.

The residual at h=30 is deliberate. Roughly half of ≥$1 zero returns at 30d are caught by neither
leg — those are genuine round-trips to the same price, and they are labels, not artifacts.

**Threshold is a constant, not a literal:** `LABEL_MAX_STALE_RUN_DAYS = 0` voids any row whose
anchor or target day carries `stale_run_days > 0`. Named so the harness can contrast 0 / 1 / 2
without a code edit. The review's "run ≥2" is 1-indexed and means the same rows.

### 3. `base_stale_run_days`, frozen on the outcome row

Migration `0021`, a nullable `Integer` on `forecast_outcomes` beside `base_price`, plus the
column on the `ops/forecast_outcomes.parquet` mirror.

It must be a **frozen observation**, not a score-time computation: `backtest/scoring.py` is pure
and `--rescore` is deliberately archive-free, so the only layer that can know an outcome's run
length is `resolve_outcomes`, which already holds the voted frame. It therefore joins the
never-`UPDATE`d set — `_REFRESH_VERDICTS_SQL` must not name it, and only `--reresolve` moves it.

**Historical rows stay NULL.** Backfilling means a full `--reresolve`, which unfreezes every
`base_price` / `actual_price` in the table — far too much for a reporting axis. A NULL means
"run length unknown", never "zero".

### 4. The staleness axis goes to four bands plus `unknown`

`backtest/scoring.py` currently partitions on `actual_price == base_price`, two buckets. It gains
a `base_stale_run_days` partition, never pooled, which is what
`docs/superpowers/specs/2026-08-07-friction-conditioned-tier-scoring-design.md` deferred to this
step.

**Fixed bands, not quartiles — a deliberate departure from the deferring spec.** That spec asked
for a 4-quartile staleness axis, and quartiles are the wrong estimator against the distribution
actually measured: **~80% of the ≥$1 cohort sits at zero**, so `q1 = q2 = q3 = 0`, a data-driven
cut collapses to one populated bucket, and it would then be *reported* as four. The bands are
`fresh` (0), `repeat_1` (1), `run_2_6` (2–6), `run_7_plus` (7+) — chosen to straddle the
mechanism rather than the mass, and stored in `metrics` so a row is self-describing.

`unknown` is a band, not a default: every outcome resolved before 2026-08-08 carries a NULL, and
folding those into `fresh` would report 13 years of unmeasured rows as measured-fresh.

### 5. Verification: interval width, not accuracy

The step's claim is that it **lowers the MDE**, not that it raises accuracy. One `ab_test_*` arm
on and off, contrasted on the **width** of the paired fold-clustered interval from
`backtest/paired_mde.py`. A narrower interval is the claim. A narrower interval accompanied by a
moved point estimate is a *different* claim and is reported as one, not merged into it.

Given the 13.7–15.9% non-zero collateral, a null or a widened interval is a real possible outcome
and is a publishable result.

---

## Explicitly out of scope

- **`stale_run_days` as a LightGBM feature.** That is step 11: it needs a retrain and an A/B
  against a 2.21–3.69pp MDE. It also carries a specific hazard — the source mix changed four
  times in 2026 (Jan–Feb `sync`-only; Mar–Apr multi-source; May–Jun `17mafo`-only; Jul+ eleven
  sources), so a feature built on run length partly encodes the collection schedule. That is
  **leak-adjacent** and needs an explicit check before anyone ships it.
- **Excluding the MA feeds from voting.** Its own step, per above.
- **Re-resolving historical outcomes** to populate the new column.
- **Moving `MIN_SERVED_PRICE_USD` or `TRAIN_MIN_MEDIAN_PRICE`.** Step 7.

## Relationship to step 7

Step 7 sets `TRAIN_MIN_MEDIAN_PRICE = 1.0`, which removes the sub-$1 items outright. On the
whole universe this rule drops ~39% of labels; on the ≥$1 subset it drops 13.8–19.2%. **The two
are substantially overlapping, not additive**, and once step 7 lands step 6's measured effect
falls to roughly a third of what it is today. Anyone reading a before/after across both steps must
not sum them.
