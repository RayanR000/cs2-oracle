# Next steps after the 2026-08-13 model and pipeline audit

**Supersedes ordering in:** `docs/research/2026-08-10-next-steps.md`. Every item keeps its label
(`C2`–`C7`, `N2`, `D2`–`D5`, `O2`, `G2`) so it traces back; that document remains the reference for
the **content** of each. Items closed since then (`C1`, `N1`, `N2`'s own-history leg, `F1`, `F2`,
`F3`) are not repeated here.

**Sources:** the 2026-08-13 changelog run — `ab-harnesses-follow-productions-trainer`,
`the-leak-is-worth-two-points-and-it-pays-the-placebo`, `c1-refutation-survives-the-audited-anchors`,
`cohort-geometry-is-not-c1s-gap`, `serving-transforms-do-not-explain-the-cv-gap`,
`feature-contribution-plus-3.5-does-not-reproduce` — plus two new entries filed the same day:
`cv-folds-are-not-time-aligned-with-serving` and `regime-training-duplicates-the-global-fit`.

---

## What the audit changed about the priorities

**1. Speed is no longer the constraint.** The warm arm64 retrain is **996.6s / 17m48s**
(run `31407938154`), inside the 30-minute cap. Every remaining cost lever is real and none is
urgent. Cost work below is ranked as hygiene, not as a goal.

**2. The constraint is experiment validity, and one confound was never controlled for.** For
7d/14d/30d the last CV validation window ends **2026-02-09**, while every served arm of the last two
weeks is read on anchors `2026-04-18 … 2026-06-08`. The two legs share no dates. See
`changelog/2026-08-13-cv-folds-are-not-time-aligned-with-serving.md`. This is not one of the four
explanations C1 was closed against — those are all about *which rows*; this is about *which months*.

**3. ~~The accuracy surface is nearly exhausted~~ — WITHDRAWN the same day.** This section read:
*"C1 closed, N1 null, N2's own-history leg dead in both directions, primitives / `tier_lead` /
ByMykel / CSFloat / volume all null. What is left: `lambdarank` (C2, never attempted) and the
allowlist itself (C4)."* The audit in
`changelog/2026-08-13-the-null-verdicts-were-not-all-tested.md` does not support it. Of ~16
accuracy verdicts, **6 survive, 5 were underpowered against their own MDE, and 5 were never
measured at all.** Specifically: **supply depth has no A/B, no folds and no MDE** — it was dropped
on mechanism plus analogy to trade volume, and both premises have since been refuted; **recency
weights' 30d arm passed its gate** (+1.17pp, 6/8 folds) and was declined on a mechanism argument;
**`N_ENSEMBLES` 3→1 is documented as unmeasured**; and the primitives and ByMykel nulls are each
consistent with a real +1–2pp effect (`tier_lead` has no interval at all). The survivors are
regime, mean reversion, market-relative labels, volume at 3d/14d/30d, CSFloat at 3d/7d, and the
`+3.5pp` number specifically.

**4. 🔑 And three of the instruments do not run.** `source = 'STEAMCOMMUNITY'` **matches zero rows
in the archive** — the real values are `NULL` (pre-2026, ends 2025-12-31) and thirteen
`aggregator_*` feeds. `ab_test_supply_side.py:106`, `ab_test_regime.py:87` and
`ab_test_ensemble.py:95` put that predicate inside an `item_slug IN (...)` subquery, so they select
**zero items today**; four more degenerate silently to a NULL-only frame that never sees the
2026-03-22 consensus break or the 2026-07-09 feed substitution. This is a **sixth** stacked defect,
and unlike the other five it is fatal rather than distorting. **Nothing in the family should be
re-run until the pins are fixed** — see item 0 below.

---

## Ranked

### 0. Repair the harness family before re-running any of it — ~1.5h, blocks items 3, 4 and 5

Source: `changelog/2026-08-13-the-null-verdicts-were-not-all-tested.md`.

- **Repin off `STEAMCOMMUNITY`** (7 files) and add a **non-zero row assertion** to every universe
  query, so an empty predicate fails loudly instead of returning an empty set. ~30 min. This is
  what unblocks `C7` for `supply_side`, `regime` and `ensemble`, which cannot run at all today.
  ⚠️ Verify against the durable `cs2-oracle-data` archive first — the local `price-archive/` runs
  behind it.
- **Apply `_apply_feature_allowlist`** in the six harnesses that skip it (`price_primitives`,
  `volume_features`, `supply_side`, `feature_contribution`, `regime`, `ensemble`). Production
  serves **33 of 123** columns; those six measure a 138–180-column model, and the dilution biases
  every feature-addition arm toward null.
- **Lift the `>= $1` filter from scoring to universe selection** in `supply_side:316` and
  `feature_contribution:363`. Their pool is `ORDER BY row_count DESC LIMIT 200`, which
  `volume_features:57-64` already measured as *15 of 200 items ≥ $1, median item price $0.059,
  8.22% of rows in the served cohort* — so both arms train on ~92% penny items.
- **Add the `, item_slug` tiebreaker** to the four pools that lack it, and a placebo arm to
  `supply_side` and `training_breadth`. The leak pays capacity, which is exactly what a placebo
  detects.

⚠️ Do **not** "fix" the embargo or add `phantom_slug_sql_filter` on the strength of a grep. Both
look like invariant violations and neither is: the purge delegates to `embargo_days(horizon)`
internally (`forecaster.py:3674`), and every phantom slug has ≤26 distinct days against a
`MIN_ITEM_DAYS` gate of 90 or 180, so **zero** phantoms enter any drawn pool.

### 0b. Then re-read supply depth — the only feature never measured

It has no A/B, no folds and no MDE; the drop rested on mechanism plus analogy to trade volume, and
both premises have since been refuted. Its harness is one of the three that currently selects zero
items, so item 0 is a hard prerequisite. Recency weights at 30d and `N_ENSEMBLES` 3→1 queue behind
it on the same grounds.

### 1. Read C1's CV edge per fold — free, no retrain, settles §2 above

`rank_ic` is already serialised per fold beside `val_start` / `val_end`
(`meta.json: cv_results[h].per_fold`). Pull it from the C1 pair that has already run — control
`31663300447`, arm `31663312585`, commit `ee76a9c` — and compute the arm−control edge per fold.

- **Bar (pre-register before reading):** if the +0.0684 / +0.0759 / +0.0588 / +0.0408 pooled CV gain
  is carried by pre-2026 folds and collapses on the 2026-01/02 fold, the CV and serving legs are
  measuring different regimes. If the edge is flat across the fold sequence, this explanation is
  spent and should be written up as spent.
- **Cost:** one script, ~20 minutes, no dispatch.
- **Why first:** it costs nothing and it changes how items 2, 4 and 5 must be read.

### 2. Anchor the CV fold grid to the end of the frame — ~5 lines

`_compute_cv_splits` (`forecaster.py:3646`) strides forward from `CV_MIN_TRAIN_DAYS`, so the last
fold lands up to `CV_STEP_DAYS − 1 = 149` days short of the frame end. h=3 wins the rounding by
single-digit dates; 7/14/30d lose a full 150-day stride. Walk the grid backwards from the end
instead.

- **Cost:** fold count unchanged or +1 per horizon; the conformal CV is 50.4% of a retrain, so
  budget ~1% of training. Not free.
- ⚠️ **It moves two published quantities and neither is accuracy.** `q_hat` is calibrated on those
  out-of-fold residuals, so the served band width changes; and the PT sample changes, so
  `invariant_4_signal` must be re-read rather than carried. Both are arguments for doing it — a
  `q_hat` fitted on a window ending 2026-02-09 and served in 2026-08 carries the same defect.
- **Do item 1 first**, so the before/after is interpretable.

### 3. C4 — the feature allowlist has no surviving basis

`changelog/2026-08-13-feature-contribution-plus-3.5-does-not-reproduce.md`: on the honest trainer,
≥$1 non-flat rows, 25 folds, n = 104,843, `no_cross_sectional − full` is **−0.35pp [−4.15, +3.18]`**
(leg 1 fails, and the upper bound essentially excludes the claimed +3.5), and the `no_random_k`
placebo at **+0.28pp** is *better* than removing either real group (leg 2 fails). Production serves
**33 of ~123 candidate features** on a number that does not reproduce.

- **Next instrument:** rebuild the harness's item selection onto the ≥$1 served universe — not
  another run of it as it stands. Its cohort is expensive to power (arm A alone was 74.6 min at
  `--max-items 1500`).
- ⚠️ **Do not re-admit `cross_sectional` without first resolving item 6** — the regime branch
  reactivates silently inside whatever A/B is measuring the feature change.
- ⚠️ Re-admitting groups also un-mutes `HORIZON_EXCLUDED_GROUPS` (`forecaster.py:389-392`), inert
  today, and raises `PREDICT_TAIL_ITEM_DAYS` past 368 for `market_return_30d_percentile`
  (`:7149-7158`). `test_cross_sectional_features_are_not_served` is the tripwire; fix the constant,
  do not delete the test.

### 4. C7 — re-read the A/B arms that measured POSITIVE

The early-stopping leak is sized at **+1.98 / +2.42 / +2.73pp** pooled DA and **pays the shuffled
arm most** (placebo swings +1.09pp between trainers). So a positive verdict under the old trainer is
the suspect case; negatives measured negative *despite* a leak that rewards capacity, and are the
safer ones.

- **Shard by horizon** — ~8 min per harness-horizon, four horizons exceeds the cap.
- Ten of thirteen harnesses are still un-re-run.

### 5. C2 — `lambdarank` within date, scored by rank IC

The only untried structural accuracy change. Cautions from `2026-08-09-next-steps.md` still bind:
raise `lambdarank_truncation_level` above its default 30 over a ~900-item cross-section, linear
`label_gain`, `lambdarank_norm`, and a ranker emits a score rather than a level — so it **adds** four
boosters rather than replacing them.

- **Read after item 1.** If C1's CV gain turns out to be a pre-2026 artefact, the rank-IC diagnosis
  that motivates C2 needs restating before four more boosters are justified.

### 6. Delete the regime branch — ~200 lines, and it is a live footgun for item 3

`changelog/2026-08-13-regime-training-duplicates-the-global-fit.md`: under the current allowlist
`market_return_30d` is never engineered, so every row labels `range`, so the regime booster is
**byte-identical** to the global one (`md5` match at 4/4 horizons). CI already skips it; every local
retrain, research retrain and A/B harness still pays 95.4s of 872s (10.9%), and none of the thirteen
harnesses sets `SKIP_REGIMES`.

If kept rather than deleted, gate it on the **allowlist**, not on an env flag.

### 7. Cost hygiene — none urgent, all uncontroversial

In descending size:

| | Lever | Size | Site |
|---|---|---|---|
| a | **Split-conformal calibration**: one held-out calibration fit per horizon instead of 33 refits; move the expanding-window CV to a schedule for the PT verdict and the rank-IC series | ~700s, 50.4% of training | `2026-08-10-training-cost-levers.md` §3 |
| b | `predict` calls `engineer_features` **without** `skip_unused_groups=True` (`:7307`, `:7222`), unlike training (`:4636`), then adds cross-sectional and supply-depth features nothing reads — and the chunked path pays it **twice** | 8.5s of 17.5s on the training frame, ×2 on predict | `forecaster.py:7205-7314` |
| c | `prepare_targets` recomputes `_snapshot_dates`, `_collection_shift_dates` and `stale_run_days` identically once per horizon; all three are pure functions of `(item_id, date, price)` | 4× redundant whole-frame scans | `forecaster.py:4180-4220` |
| d | `_compute_sample_weights` — a Python-lambda `groupby.transform` of `pct_change().rolling(30).std()` — runs ~74× (train+val per horizon, per regime, per CV fold) on the same deterministic quantity | up to 300K rows × 74 | `forecaster.py:4947-4949` |

⚠️ (a) has a design constraint that must not be lost: the calibration split has to be carved out
**before** HP selection, not after, or it recreates the under-coverage bug `forecaster.py:4274-4292`
documents.

**Levers that are not levers**, re-confirmed: boost rounds (calibrated at the knee; cutting them
costs 30d's PT), `CV_MAX_TRAIN_ROWS` (already binding at 300,000), `CV_STEP_DAYS` (fold count is the
PT sample), `max_bin` (31 measured *slower* than 63), `N_ENSEMBLES` (already 1).

### 8. Remove — computed every run, never served

- **11 volume features** (`:2114-2193`, called unconditionally at `:2105`), every one in
  `SHELVED_FEATURES`; the archive's `volume` column has been identically 0 since 2026-05.
- **`distance_to_support` / `distance_to_resistance` / `high_low_range_30d`** (`:2072-2077`).
  `_skipped_feature_groups` deliberately never skips group `other` to preserve them (`:5961-5964`),
  and `_apply_feature_allowlist` (`:6012-6021`) then drops them anyway because
  `other ∉ ["price_technicals"]`. **The comment protecting them is stale.**
- **`label_vol_30d`** (`:1973-1976`) — production passes `sigma_train=None` at both call sites
  (`:5288`, `:8078`); only `scripts/ab_test_direction_labels.py` reads it. Already recorded as dead
  code; still present.
- **`MOMENTUM_FALLBACK_HORIZONS = []`** (`:553`) → `_recenter_on_momentum` (`:6024`) and the branch
  at `:7599-7604` are unreachable.
- **`SAMPLE_WEIGHT_HALFLIFE_DAYS = 0.0`** (`:164`) → the recency-decay block `:4961-4964` never
  executes.
- **`_gpu_available()`** (`:192`) spawns `nvidia-smi` plus a Python subprocess probe on every fresh
  process and always returns False here.
- ⚠️ **Keep** the six shelved price primitives (`:2038-2067`) and `target_{h}d_cal` (`:4147-4168`) —
  both have live harness readers.

---

## Carried forward unchanged

`N2` (exogenous leg only — FX, events, player counts; the ingest exists and was never used), `C3`,
`C5` (band level; three width scales refuted, do not propose a fourth), `C6`'s `N_ENSEMBLES` half,
`O2`, `G2` Part 2, `D2`–`D5`, and the two unlabelled serving-hygiene items (exclude 2026-07-19 as a
distinct serving config; WARN plus a flat-call count on `predict()`'s no-classifier fallback).

`D5` — consolidating the fifteen harnesses — is worth re-reading against item 4: nine still use
`phase_collapsed_sql_filter()` where invariant 2 requires `archive_universe_sql_filter()`, and every
C7 re-read runs through one of them.
