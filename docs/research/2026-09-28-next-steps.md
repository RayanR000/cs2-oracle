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

## State on 2026-10-04

- **musickit watch closed (item 10).** Its h=3 under-coverage is the known width tilt, not a
  family effect. `changelog/2026-10-04-musickit-h3-is-the-width-tilt.md`.

## State on 2026-10-02

- **Bitskins supply feed dropped (verified 10-02).** Its public export has been empty upstream
  since ~08-24 (last archived day 08-23), and every run logged it as a failed feed while staying
  green. No feature reads it. Three scalar feeds plus the lis-skins ladder remain (items 11–12).
  `changelog/2026-10-02-drop-bitskins-supply-feed.md`.
- **Item 21's first CI run is verified:** zero revisions on a populated snapshot.

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
3. ~~**Confirm the first scheduled runs on the chunked vote.**~~ **DONE 2026-10-05 (verified
   10-05).** The predict-only half was confirmed 09-30: run `36657821345` logged `Applying
   multi-source voting in DuckDB (20 sources, 8 item chunks)` and `Wrote 22144 forecasts`. The
   Mon 10-05 `mode=full` retrain, Price Forecast run `37252926432` (`FORCE_RETRAIN: 1`), logged
   the same voting line on both the train and the predict pass, `TRAINING COMPLETE in 1320s
   (22.0min)` and `Wrote 22499 forecasts`. The extra rows are young releases: `355 items served at
   h=[3] only (0 had no h=3 forecast)`. Both stores carry forecast date 2026-10-04. The same run
   is the first retrain on the composition-break calendar: the break list includes 2026-01-01
   and 2026-04-16, and every horizon voided across `7 break days`.

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
   ⚠️ The h=14 band width changed twice, so **split the h=14 read at forecast dates 2026-09-30 and
   2026-10-05** (item 8). `changelog/2026-09-22-post-csmarketapi-served-performance.md`.
8. ~~**h=14 feedback factor.**~~ **WOKE 2026-09-30 (verified 09-30).** Price Forecast run
   `36657821345` logged `14d served-coverage factor 0.7059 on 7,169 served rows over 8 dates
   (>= 8)` and `Served-coverage factors (live panel read for [14, 30]): {14: 0.7059}`. The first
   forecast date served with it is **2026-09-30**. It woke on the first **predict** run after the
   gate crossed (#72), not at a retrain. **Still pending:** h=30 reaches 8 dates around
   mid-October and wakes the same way. Confirm it in the same log line.
   **Refit at the 10-05 retrain (verified 10-05):** run `37252926432` logged `{3: 0.5385, 7:
   0.5812, 14: 0.6757}`. h=3 is unchanged; h=7 moved 0.6111 → 0.5812 (19 dates) and h=14 moved
   0.7059 → 0.6757 (12 dates). Bands at h=7 and h=14 narrow again from forecast date 2026-10-05.
   ⚠️ Item 7's h=14 coverage read now spans two band-width changes, so **split it at forecast
   dates 2026-09-30 and 2026-10-05**. `changelog/2026-09-13-served-feedback-wake-dated.md`,
   `changelog/2026-09-28-served-feedback-live-refresh-per-horizon.md`.
9. **Exceedance served-reliability re-check at maturity (`--served-only`).** Watch the h=14 bulk
   bin: 0.032 predicted vs 0.007 realised. If it persists past 20 dates, the fix is "a served
   refit, not a fancier map". The maturity date is not stated.
   `changelog/2026-09-13-exceedance-calibration-measured.md`.
10. **Conditions to watch, not actions:** ~~`musickit` h=3 under-coverage (84.7%) gets a re-read at
    20 dates.~~ **CLOSED 2026-10-04: it is the width tilt.** At 38 dates musickit reads 83.3%, but
    52% of its rows sit in the narrow-band tertile. Width-matched, it is within the CIs of the
    other families in the narrow and mid tertiles. No family arm.
    `changelog/2026-10-04-musickit-h3-is-the-width-tilt.md`. `DIRECTION_DISCLOSED`
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
13. ~~**Case panel, sticker panel and the Reddit event detector: decide whether to wire them or
    drop them.**~~ **DONE 2026-09-30: dropped.** The two panels are derived views over the
    daily `supply-*.parquet`, so they never needed a workflow to accumulate. Run on the live
    archive, the case panel populated only visible supply (status 100% `unknown`, EV/openings
    0%), and the sticker panel matched **0** of 12,185 sticker names (slug-format mismatch).
    Reddit has no credentials in the repo. A case-only or sticker-only study rebuilds its panel
    from the archive when it is preregistered.
    `changelog/2026-09-30-drop-case-sticker-reddit-builders.md`.
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
20. ~~**Deep-review §11 relabelling.**~~ **DONE 2026-10-01, except the fee constant.**
    `changelog/2026-10-01-deep-review-s11-relabel.md`.
    - `experiment_log.csv` has an `mde` column. ByMykel is relabelled `inconclusive`, and seven
      §11 arms got rows.
    - The `paired_mde` nondeterminism was reproduced (16 rows between two identical builds) and
      fixed. The order of duplicate rows flipped the last bit of the daily mean, and with it the
      bit-exact frozen-run voiding.
    - The capacity-matched `age_only` placebo is **refuted**: null at all four horizons held-out,
      and the 08-08 positives don't reproduce on the honest trainer.
    - The stale labels are fixed. So is the 09-15 move, which had broken every path in
      `scripts/archive/` (the archive directory, `sys.path`).
    - **Still open: the 1.1607 Steam fee constant.** `runtime/steam_listing_history.db` is gone;
      re-validating means re-scraping 262 items with a Steam session.
21. ~~**Decide archive history retention.**~~ **DECIDED 2026-10-01: no history, log revisions instead.**
    `changelog/2026-10-01-price-revision-log.md`. **First CI run verified 10-02:** Aggregator run
    `36954043078` (snapshot date 10-01) logged `{"revisions": 0, "logged_rows": 0, "wrote": false}`.
    The before-snapshot was populated (no "No monthly price files to snapshot" warning), so the
    zero is a real zero, not an empty input.
    Original question: The orphan force-push keeps no history, so revisions
    can't be diffed; the 08-25 note says it is "worth deciding deliberately"
    (`changelog/2026-08-25-centre-shrinks-to-zero-and-the-dollar-band-is-the-wedge.md`).
22. ~~**`mapie` / `scoringrules` in the `dev` extra is blocked.**~~ **DONE 2026-09-30 (#80).**
    The exact `requests==2.31.0` pin had no recorded reason (it dates from the original
    pin-everything `requirements.txt`) and is now `>=2.31,<3`. `uv lock` had been failing on
    main since `evidently` landed in #59, so `uv.lock` was stale; CI never noticed because it
    syncs `--frozen`. Runtime changes: requests 2.34.2, and protobuf drops 7.36.1 → 6.33.6
    through the universal resolution (an mlflow/evidently cap). Only `onnxruntime` pulls
    protobuf at runtime, and nothing in `backend/` imports it. The MAPIE oracle and
    interval-score tests now run in the gate. ~~Still unstarted: a statsforecast outside baseline
    for `PORTFOLIO.md`.~~ **MEASURED 2026-10-04.** `scripts/outside_baseline.py` scores the
    served band against a naive empirical-quantile band and AutoETS + conformal. Pooled over
    every date, both baselines beat it at all four horizons, but that pools band eras. Since
    09-21, h=3 ties: narrowly better than naive and level with ETS (10 dates). h=7 still loses
    (6 dates, mostly from before its feedback factor). **Re-read with `--since 2026-09-28` at
    20 dates per horizon. If h=7 still loses, preregister a naive-quantile band arm.**
    `changelog/2026-10-04-outside-baseline-statsforecast.md`.

## 6. Unstarted and unranked — open ideas, not commitments

- ~~**Composition-break calendar (deep review §12.8).**~~ **DONE 2026-09-30, ships with the 10-05
  retrain.** Labels now void across source-composition breaks too, which adds 2026-01-01 and
  2026-04-16 to the voided days. Per-item source churn is still open.
  `changelog/2026-09-30-composition-break-calendar.md`.
- ~~**Break-aware lookbacks (`BREAK_AWARE_LOOKBACKS`), Part B gate.**~~ **MEASURED 2026-10-05:
  inconclusive, flag stays off.** Matched width moved −0.28/−0.70/−0.88/−4.43% at h=3/7/14/30,
  and interval score +0.33/+0.61/+0.18/−0.98%. Every CI spans zero, so the gate (SUPPORTED at
  h=14 and h=30) fails. The achieved half-widths (4.3% at h=14, 11% at h=30) are above the ~2%
  bar, so this is not a refutation. Re-read only with more folds past 07-12 from a fresh
  data-repo archive. The masks lift around 2027-01 anyway.
  `changelog/2026-10-05-break-aware-lookbacks-ab-inconclusive.md`.
- ~~**A cross-venue FX ratio monitor (§12.11).**~~ **DONE 2026-10-05, warn-only.** The
  Aggregator's "FX ratio monitor" step compares median buff163/csfloat and youpin/csfloat with
  their trailing 14-day medians and warns past 3% (`fx_signature` when both CNY venues move and
  the USD controls don't). It never fails the run. Replayed from 03-22 it raised zero alarms;
  the largest CNY move, +2.9% on 07-11, is the composition break.
  **First CI run to check:** the step prints `"status": "ok"` with `baseline_days` ≥ 13.
  `changelog/2026-10-05-fx-ratio-monitor.md`.
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
| Mon 10-05 (done) | weekly `mode=full` retrain verified (item 3): chunked vote, composition-break calendar incl. 04-16, item 16 removed; factors refit to h=7 0.5812, h=14 0.6757 (item 8). Band-era break: split h=14 reads (item 7) and the h=7 outside-baseline re-read (item 22) at forecast date 10-05. Break-aware lookbacks Part B gate: inconclusive, flag stays off (§6). Next retrains 10-12, 10-19, … |
| ~10-10 | h=14 at 20 post-floor dates |
| ~10-12 | end of the 09-14 workplan's wait window |
| early–mid Oct | h=14/h=30 post-CSMarketAPI confirmation (item 7); h=14 read split at 09-30 and 10-05 |
| mid-Oct | h=30 reaches 8 served dates; its feedback factor wakes (item 8) |
| mid-Oct at earliest | 20 shared champion–challenger dates (items 5–6) |
| 10-18 | last forecast date in the PID window |
| ~10-23 | the single PID read (item 4) |
| ~10-23 / late Oct | outside-baseline re-read, `--since 2026-09-28`, h=3 then h=7 at 20 dates (item 22) |
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
