# Next steps — 2026-09-28

The live action list. It replaces the August series (`2026-08-07` → `2026-08-16-next-steps.md`),
which was pruned in `2b5330d` on 2026-09-15 without a successor. By then the 08-16 list had
already handed its ranking to §12 of `2026-08-19-deep-model-review.md`. Recover any pruned file with
`git show 2b5330d^:docs/research/2026-08-16-next-steps.md`.

It was compiled from every changelog, research note, spec, plan and experiment-log row dated
2026-08-16 or later. Items marked **(verified 09-28)** or **(verified 09-30)** were checked against code or CI on
the day. The rest restate what the cited doc says. Where the docs conflict, the item says so.

**Standing constraint:** this is a range forecaster (`AGENTS.md`). Nothing below proposes a
directional product claim, a fourth band-width denominator, or a serving ranker. See §7.

## State on 2026-09-30

- **Chunked vote confirmed on a scheduled predict-only run (verified 09-30).** Price Forecast run
  `36657821345` (02:00Z) logged `Applying multi-source voting in DuckDB (20 sources, 8 item
  chunks)` and `Wrote 22144 forecasts`. The Monday retrain half is still open (item 3).
- **h=14 feedback factor is live (verified 09-30).** The same run logged `14d served-coverage
  factor 0.7059 on 7,169 served rows over 8 dates (>= 8)`. The first forecast date served with it
  is **2026-09-30**. h=30 is still pending, mid-October (item 8).
- **Archive day 09-19 is restored (verified 09-30).** #74 (`ca5c21b`) backfilled it. Item 2 is done.
- **CSMarketAPI access is gone.** The backfill will not resume (item 14), and nothing now adds
  newly released items to the catalog (item 23).

State on 2026-09-28, still true unless an item below says otherwise:

- **Pipeline green again (verified 09-28).** The Monday `mode=full` retrain was OOM-killed twice
  during multi-source voting. After #69 and #70 (DuckDB voting, item-hash chunks), run
  `36473131841` trained in 22.5 min and wrote 22,144 forecasts.
  `changelog/2026-09-28-voting-moved-into-duckdb.md`.
- **h=7 feedback factor now serves (verified 09-28).** It is 0.6111, on 13 dates. The retrain
  refit it alongside h=3's 0.5385, on the composition-fixed code (#64). As of 09-22, h=7 was still
  served at 1.0. The code's wake gate is **8 dates** (`MIN_FEEDBACK_DATES`); the 09-14 workplan's
  "do not lower the gate (20)" predates that split.
- **Served coverage is still the open problem:** 91–94% against an 80% target at ≥$1
  (`specs/2026-09-14-wait-window-workplan.md`, `changelog/2026-09-22-post-csmarketapi-served-performance.md`).
  The feedback factors are the live remedy, and the PID read (item 4) is the next test of it.

## 1. This week

1. ~~**Run the archive-basis h=14 served-rank gate.**~~ **DONE 2026-09-28: KILL.** The gate cleared
   at exactly 20 dates. Model IC is +0.089 [+0.061, +0.117], but model minus naive is +0.026
   [−0.005, +0.055], which spans zero. Both sensitivities VOID on the primary cohort. Item 6 loses
   its archive-basis support. `changelog/2026-09-28-archive-basis-centre-rank-h14-kill.md`.
2. ~~**Decide whether archive day 2026-09-19 can be backfilled, before 10-23.**~~ **DONE 2026-09-29.**
   #74 (`ca5c21b`, "restore lost archive day 09-19 and lease every data-repo push") restored the
   day. Archive Restore Day runs `36527080625` and `36527193991` succeeded on 09-29. H=3 forecast
   dates 09-16/17/18 are scoreable again and go back into the PID window (item 4). The prereg
   excludes them only conditionally ("until 09-19 is backfilled") and
   `scripts/measure_conformal_pid.py` hard-codes no exclusion, so this is consistent with the
   frozen prereg, which governs and is not edited. The window is still 09-07..10-18.
   `changelog/2026-09-23-feedback-factor-refit-composes.md`.
3. **Confirm the first scheduled runs on the chunked vote.** The predict-only half is **CONFIRMED
   (verified 09-30):** run `36657821345` (2026-09-30 02:00Z) logged `Applying multi-source voting
   in DuckDB (20 sources, 8 item chunks)` and `Wrote 22144 forecasts`. **Still open:** the Mon
   10-05 `mode=full` retrain. Check the output, not the badge: `forecast_date` advances, and the
   voting line prints `8 item chunks`.

## 2. Scheduled reads — do not run early

4. **Conformal PID vs the batch feedback factor, h=3, served panel.** The prereg froze on 09-23
   and the instrument is built (`scripts/measure_conformal_pid.py`). The window is forecast dates
   09-07..10-18, read **once** around **10-23**; the script refuses to read earlier. It is VOID if
   the incumbent's per-date error is under 3pp. 09-28 is out of the window (no forecasts that
   day). Forecast dates 09-16/17/18 are scoreable again after item 2, so they count.
   ⚠️ **Fixed 2026-09-30:** the loader's prior-row read selected columns `item_forecasts` does not
   have and would have raised at the read. Young releases (item 23) are joined out, so the panel
   stays the established one the prereg froze. `research/2026-09-23-conformal-pid-served-h3-preregistration.md`.
5. **Centre promotion (last-price vs `gbm_q50`) through the champion–challenger shadow.** It reads
   INSUFFICIENT_EVIDENCE until 20 shared dates. Promotion is a reviewed PR, never automatic
   (invariant 5).
   - ⚠️ The centre was retrained on CSMarketAPI data on 09-20 and served rank IC turned positive
     on 09-22. The 09-14 `centre_vs_lastprice` read (skill −5.6 to −12.5%, all CIs below zero)
     describes the **old** model.
   - `changelog/2026-09-18-multi-head-champion-challenger-built.md`,
     `research/2026-08-25-gbm-retirement-decision.md`.
6. **Ranking-head transfer (`RANKING_HEAD=1`, sidecar `rank_score`).** It is decided by
   `scripts/ranking_transfer_report.py` (a 20-date paired gate) together with item 1. **No rank
   into band or direction until SUPPORTED.**
   `changelog/2026-09-18-lambdarank-transfer-screen-closed.md`.
7. **Post-CSMarketAPI served confirmation at h=14 and h=30**, early to mid October. The ~40%
   narrowing is attributed to the combined PR `ad2d43e`, not to CSMarketAPI alone.
   `changelog/2026-09-22-post-csmarketapi-served-performance.md`.
8. ~~**h=14 feedback factor.**~~ **WOKE 2026-09-30 (verified 09-30).** Price Forecast run
   `36657821345` logged `14d served-coverage factor 0.7059 on 7,169 served rows over 8 dates
   (>= 8)` and `Served-coverage factors (live panel read for [14, 30]): {14: 0.7059}`. The first
   forecast date served with it is **2026-09-30**. It woke on the first **predict** run after the
   gate crossed (#72), not at a retrain. **Still pending:** h=30 reaches 8 dates around
   mid-October and wakes the same way. Confirm it in the same log line.
   ⚠️ Item 7's h=14 coverage read spans a band-width change, so **split it at forecast date
   2026-09-30**. `changelog/2026-09-13-served-feedback-wake-dated.md`,
   `changelog/2026-09-28-served-feedback-live-refresh-per-horizon.md`.
9. **Exceedance served-reliability re-check at maturity (`--served-only`).** Watch the h=14 bulk
   bin: 0.032 predicted vs 0.007 realised. If it persists past 20 dates, the fix is "a served
   refit, not a fancier map". The maturity date is not stated.
   `changelog/2026-09-13-exceedance-calibration-measured.md`.
10. **Conditions to watch, not actions:** `musickit` h=3 under-coverage (84.7%) gets a re-read at
    20 dates (`changelog/2026-09-09-conditional-coverage-by-stratum.md`). `DIRECTION_DISCLOSED`
    reopens only on a positive PT at 20 or more dates
    (`changelog/2026-09-10-served-direction-withheld.md`).

## 3. Accumulating — built, gated off, waiting on data

11. **`SKINPORT_VOLUME=1` volume-repair A/B.** It waits on `volume-*.parquet` depth.
    - Run `scripts/archive/compute_mde.py` first, and **stop if MDE > ~2pp**.
    - Judge on band quality (invariant 4), not DA. It is the successor to the 08-16 list's #2,
      whose iflow route closed net-negative.
    - `changelog/2026-09-08-accuracy-panels-and-orderbook-accumulation.md`.
12. **`ORDERBOOK_FEATURES=1` (`ob_*` ask-side proxies).** Default off.
    - `ob_churn` promotion is forbidden (`changelog/2026-09-13-supply-velocity-transfer-killed.md`).
    - Slope, depth concentration, inflow, age and turnover were never explicitly closed.
    - Bid-side features (spread, imbalance, price impact) have no free source.
13. **Case panel, sticker panel and the Reddit event detector: decide whether to wire them or
    drop them.** The builders exist, but **none is in a workflow, so nothing is accumulating**. The
    09-08 note sequences Reddit after the market-flow panels.
14. ~~**CSMarketAPI backfill resume.**~~ **CLOSED 2026-09-30: will not resume.** The user no
    longer has CSMarketAPI access. The 305 knives/gloves checkpointed at "★ Ursus Knife | Marble
    Fade (Factory New)" stay as they are, and the `--only-catalog` run is dropped. The 09-20
    retrain data is the last CSMarketAPI data there will be.
    `changelog/2026-09-20-csmarketapi-backfill-and-retrain.md`.
23. ~~**Decide how newly released items reach the catalog.**~~ **DONE 2026-09-30.** The
    Aggregator now inserts newly priced names (`collectors/new_item_discovery.py`), and
    `models/serve_universe.py` serves a discovered release **at h=3 only** once it has 60 days of
    history. Replayed on the 2026-07-08 release, h=3 covered 81–83% but h=7/h=14 covered 70–75%,
    so the longer horizons are withheld. Young rows are excluded from the feedback refit and the
    PID panel (item 4). **Re-read h=7/h=14 at ~120 days, mid-November.**
    `changelog/2026-09-30-serve-new-releases-at-h3.md`.

## 4. Decide or delete — code and docs that disagree

15. ~~**Learned source-quality vote weights and the data-quality Isolation Forest have no
    callers.**~~ **DONE 2026-09-30 (#76): deleted.** Removed `models/source_weight_model.py`,
    `models/data_quality.py`, `conformal.calibrate_adaptive` / `update_adaptive` and
    `stratum_factors_from_panel`, each with its tests. Nothing was live. The 09-22 changelog
    got a dated correction: source weights and adaptive conformal never shipped.
    `changelog/2026-09-30-delete-uncalled-code.md`.
16. ~~**`SHRINK_K_GBM` / vol-rank flag plumbing.**~~ **DONE 2026-09-30, ships with the 10-05
    retrain.** The env reads, `meta.json` keys, `_*_served()` accessors, serve-time vol-rank
    multiplier, workflow key and the archived replays that called them are removed. Old
    artifacts still load. `changelog/2026-09-30-composition-break-calendar.md`.
17. ~~**Fix stale reproduce paths.**~~ **DONE 2026-09-30 (#76).** The seven moved scripts are
    now cited at `backend/scripts/archive/` in architecture, research and code comments.
    Three frozen preregistrations got the path fix plus a dated path-only erratum.
    Changelogs were left as written. `run_forecast_local.sh` had been silently skipping
    its informational centre gate on the old path; it now runs it again (read-only against
    prod Postgres).

## 5. Hygiene

18. ~~**`experiment_log.csv` is missing rows.**~~ **DONE 2026-09-30 (#78).** 21 rows added,
    each linked to the note that holds its verdict (TFT links `tft_eval.csv`, since it has no
    changelog). NGBoost and CQR are logged `void` against
    `specs/2026-09-18-ml-upgrade-design.md`, because they were deleted with no measurement. The two
    08-17 volume band-quality notes are a sequence, not a conflict. The 8-fold run is
    `inconclusive`, the 16-fold refold is `measured` (it passed on the retired q10/q90 band),
    and `volume-in-scale` already refutes it on the served band. `feature-native-nan` is
    corrected to `shipped` (on since #30, 08-21), and `research/2026-08-19-deep-model-review.md`
    carries a dated update saying so (#79).
19. ~~**The `docs/README.md` research index stops at 08-18.**~~ **DONE 2026-09-30 (#77).**
    Seven preregistrations and five research docs are indexed, with verdicts taken only from
    changelogs. The README's specs, plans and changelog sections are still selective by design.
20. **Deep-review §11 relabelling** (`research/2026-08-19-deep-model-review.md`):
    - Put the MDE beside each stored A/B verdict, and relabel underpowered nulls UNRESOLVED.
    - Fix `paired_mde`'s frame-construction nondeterminism.
    - Run one capacity-matched `age_only` placebo.
    - Fix three stale console labels.
    - Re-validate the synthetic 1.1607 Steam fee constant (about half a day).
21. **Decide archive history retention.** The orphan force-push keeps no history, so revisions
    can't be diffed; the 08-25 note says it is "worth deciding deliberately"
    (`changelog/2026-08-25-centre-shrinks-to-zero-and-the-dollar-band-is-the-wedge.md`).
22. ~~**`mapie` / `scoringrules` in the `dev` extra is blocked.**~~ **DONE 2026-09-30 (#80).**
    The exact `requests==2.31.0` pin had no recorded reason (it dates from the original
    pin-everything `requirements.txt`) and is now `>=2.31,<3`. `uv lock` had been failing on
    main since `evidently` landed in #59, so `uv.lock` was stale; CI never noticed because it
    syncs `--frozen`. Runtime changes: requests 2.34.2, and protobuf drops 7.36.1 → 6.33.6
    through the universal resolution (an mlflow/evidently cap). Only `onnxruntime` pulls
    protobuf at runtime, and nothing in `backend/` imports it. The MAPIE oracle and
    interval-score tests now run in the gate. Still unstarted: a statsforecast outside baseline
    for `PORTFOLIO.md` (`references/open-source-shortlist.md`).

## 6. Unstarted and unranked — open ideas, not commitments

- ~~**Composition-break calendar (deep review §12.8).**~~ **DONE 2026-09-30, ships with the 10-05
  retrain.** Labels now void across source-composition breaks too, which adds 2026-01-01 and
  2026-04-16 to the voided days. Break-aware lookbacks (`BREAK_AWARE_LOOKBACKS`) are built but
  off, pending the paired A/B in the spec's Part B gate. Per-item source churn is still open.
  `changelog/2026-09-30-composition-break-calendar.md`.
- A cross-venue FX ratio monitor (§12.11, about 2h).
- R14 mechanical supply-position features and the hedonic market index. Both carry over from
  08-16, untouched.
- Re-score boost rounds and the Optuna objective on width at matched coverage (§12.15, to be
  re-scoped against climatology).
- "A stale anchor refuses to serve" (`changelog/2026-08-19-drop-replayed-2025-12-01-cohort.md`).
  `ANCHOR_AUDIT=1` shipped instead, and 08-25 says to build no anchor arms, so this is probably
  superseded.
- The §9 alternatives (AR(1) unsmoothing with its free kill test, an h=3 hurdle model, an h=30
  term-structure band, partial pooling of nuisance parameters, weighted calibration scores). The
  centre-side ones are likely moot after 08-25.

## 7. Do not run

The full lists live in the cited docs; this is the index.

**Band width and calibration**
- Any sigma-shaped or modelled-sigma scale, or a fourth width denominator (`AGENTS.md`).
- Width conditioned on sigma, the sigma exponent, a learned scale, exceedance, Mondrian or
  per-tier cuts, bagged `q_hat`, vol-rank GBM, `SHRINK_K_GBM`, `log1p(listings)` or supply
  velocity (`specs/2026-09-14-wait-window-workplan.md`).
- `q_hat` fold-design fixes (09-15). Retuning `CLIMATOLOGY_REACTIVE` (08-20). WACI (09-09).
- Another offline ACI replay; only a new served prereg reopens it (09-22).
- Pooling across band-geometry eras. Fitting on served rows at 6 dates or fewer.

**Centre, direction and ranking**
- A per-item directional arm, or a serving ranker.
- Rank into band or direction before SUPPORTED.
- Re-opening the centre ("no second panel, no second basis"). Anchor arms A/B/C (08-25).
- Further `init_score` variants (09-22).
- Quote-basis outcomes as a ranking referee (09-13 VOID).
- A broad re-run of the nine harnesses; relabel instead (item 20).
- NGBoost or a Gaussian head; CQR with new boosters.

**Features and data**
- Cross-venue basis ingest or static arbitrage (08-16/08-17).
- The wash-trade screen or volume-spike exceedance; a listing-count gate.
- iflow volume ingest, or a universe-expanding iflow backfill.
- `EXCEEDANCE_META` without a new column family; ByMykel metadata.
- A per-item exceedance rate; the 30d anomaly head as a width or ranking arm.
- The supply-churn width GBM, or `ob_churn` promotion.
- Horizon-extension trade hunts. Generic sentiment.
- darts, sktime, mlforecast or TFT.

## 8. Calendar

| Date | Event |
|---|---|
| ~~09-27~~ 09-28 | archive-basis h=14 gate: KILL (item 1) |
| 09-29 (done) | archive day 09-19 restored (item 2, #74) |
| 09-30 (done) | h=14 feedback factor live, 0.7059 (item 8); chunked vote confirmed predict-only (item 3); new releases served at h=3 (item 23) |
| 09-29 | h=3 at 20 post-floor dates |
| 10-03 | h=7 at 20 post-floor dates |
| Mon 10-05 | weekly `mode=full` retrain (then 10-12, 10-19, …); first check of the chunked vote on a retrain (item 3); first retrain with the composition-break calendar and item 16 removed (check the `(composition: …)` list includes 04-16) |
| ~10-10 | h=14 at 20 post-floor dates |
| ~10-12 | end of the 09-14 workplan's wait window |
| early–mid Oct | h=14/h=30 post-CSMarketAPI confirmation (item 7); h=14 read split at 09-30 |
| mid-Oct | h=30 reaches 8 served dates; its feedback factor wakes (item 8) |
| mid-Oct at earliest | 20 shared champion–challenger dates (items 5–6) |
| 10-18 | last forecast date in the PID window |
| ~10-23 | the single PID read (item 4) |
| ~11-15 | young-release h=7/h=14 re-read at ~120 days (item 23) |
| November | h=30 at 20 post-floor dates |

## Carry-over from the 08-16 list

| # | Item | Status |
|---|---|---|
| 1 | R13 cohort question | **closed, shipped:** two-tier tradeability label, no floor (`changelog/2026-08-17-tradeability-label-on-served-forecasts.md`) |
| 2 | Repair the volume feed | iflow route **refuted, net-negative** (`changelog/2026-08-17-volume-in-scale-is-net-negative.md`); successor is item 11 |
| 3 | Re-run nine harnesses | **reversed into relabelling**, item 20 |
| 4 | ≥$1 over-coverage | **partly addressed** (climatology, K=320, feedback factors); still 91–94% served, items 4 and 8 |
| 5 | Cross-venue basis | **refuted** 08-19 |
| 6 | Listing count as band width | **refuted** 08-18 |
| 7 | R14 / hedonic index | **open, untouched** (§6) |
