# Next steps — 2026-09-28

The live action list. It replaces the August series (`2026-08-07` → `2026-08-16-next-steps.md`),
which was pruned in `2b5330d` on 2026-09-15 without a successor. By then the 08-16 list had
already handed its ranking to §12 of `2026-08-19-deep-model-review.md`. Recover any pruned file with
`git show 2b5330d^:docs/research/2026-08-16-next-steps.md`.

It was compiled from every changelog, research note, spec, plan and experiment-log row dated
2026-08-16 or later. Items marked **(verified 09-28)** were checked against code or CI on the
day. The rest restate what the cited doc says. Where the docs conflict, the item says so.

**Standing constraint:** this is a range forecaster (`AGENTS.md`). Nothing below proposes a
directional product claim, a fourth band-width denominator, or a serving ranker. See §7.

## State on 2026-09-28

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
2. **Decide whether archive day 2026-09-19 can be backfilled, before 10-23.** Until it is, h=3
   forecast dates 09-16/17/18 are unscoreable. The PID prereg excludes those dates rather than
   imputing them, so a backfill before the read restores three dates to its window. The archive
   is CI-written only (orphan force-push), so this is a data-repo operation.
   `changelog/2026-09-23-feedback-factor-refit-composes.md`.
3. **Confirm the first scheduled runs on the chunked vote:** tonight's predict-only and the
   10-05 Monday retrain. Check the output, not the badge: `forecast_date` advances, and the voting
   line prints `8 item chunks`.

## 2. Scheduled reads — do not run early

4. **Conformal PID vs the batch feedback factor, h=3, served panel.** The prereg froze on 09-23
   and the instrument is built (`scripts/measure_conformal_pid.py`). The window is forecast dates
   09-07..10-18, read **once** around **10-23**; the script refuses to read earlier. It is VOID if
   the incumbent's per-date error is under 3pp. 09-28 is out of the window (no forecasts that
   day). `research/2026-09-23-conformal-pid-served-h3-preregistration.md`.
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
8. **h=14 feedback factor.** It wakes when h=14 reaches 8 resolved served dates (the code gate),
   at the first retrain after that. h=30 follows in November.
   `changelog/2026-09-13-served-feedback-wake-dated.md`.
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
14. **CSMarketAPI backfill resume.** 305 knives/gloves are checkpointed at
    "★ Ursus Knife | Marble Fade (Factory New)". Run `--only-catalog` when the monthly key resets
    (no date recorded). `changelog/2026-09-20-csmarketapi-backfill-and-retrain.md`.

## 4. Decide or delete — code and docs that disagree

15. **Learned source-quality vote weights and the data-quality Isolation Forest (09-19 spec items
    13–14) have no callers (verified 09-28).**
    - `models/source_weight_model.py` has no importers. `models/data_quality.py` is imported only
      by its own test. Spec item 12's `stratum_factors_from_panel` is also uncalled.
    - Yet `changelog/2026-09-22-post-csmarketapi-served-performance.md` says the retrain
      "shipped … source weights, and adaptive conformal". Correct the doc, then wire-and-measure
      or delete.
    - Adaptive conformal is part of the refuted conditioning family
      (`research/2026-09-22-adaptive-conformal-preregistration.md`), so deletion is the default
      for that piece.
16. **Delete `SHRINK_K_GBM` / vol-rank code.** This was a 09-10 follow-up, and ~12 references
    remain in `forecaster.py` (`changelog/2026-09-10-shrink-k-gated-off-vol-rank-unwired.md`).
17. **Fix stale reproduce paths.** The 09-22 cleanup moved `archive_basis_centre_rank.py`,
    `centre_vs_lastprice.py`, `compute_mde.py`, `ab_test_volume_features.py`,
    `build_case_panel.py`, `build_sticker_panel.py` and `run_reddit_events.py` into
    `backend/scripts/archive/`. The preregistrations still cite the old paths.

## 5. Hygiene

18. **`experiment_log.csv` is missing rows** (Workflow Rule 3):
    - TFT centre (refuted 09-10, commit `5c94230`, no changelog)
    - `SHRINK_K_GBM` and vol-rank (09-10)
    - `CLIMATOLOGY_REACTIVE` (08-20); shrink-K 320 (08-26); centre vs last price (08-25)
    - clean-era centre, exceedance calibration, anomaly 30d calibration (09-13)
    - `q_hat` dispersion (09-15); per-horizon objective and ranking head (09-17)
    - direction withheld (09-10)
    - NGBoost and CQR (deleted "as refuted" in `ff568ca` with no measurement doc)

    Also check `feature-native-nan`: it is logged as "measured", but it is **on** in
    `price-forecast.yml`.
19. **The `docs/README.md` preregistration and research index stops at 08-18.** A dozen research
    docs since 08-19 are unindexed.
20. **Deep-review §11 relabelling** (`research/2026-08-19-deep-model-review.md`):
    - Put the MDE beside each stored A/B verdict, and relabel underpowered nulls UNRESOLVED.
    - Fix `paired_mde`'s frame-construction nondeterminism.
    - Run one capacity-matched `age_only` placebo.
    - Fix three stale console labels.
    - Re-validate the synthetic 1.1607 Steam fee constant (about half a day).
21. **Decide archive history retention.** The orphan force-push keeps no history, so revisions
    can't be diffed; the 08-25 note says it is "worth deciding deliberately"
    (`changelog/2026-08-25-centre-shrinks-to-zero-and-the-dollar-band-is-the-wedge.md`).
22. **`mapie` / `scoringrules` in the `dev` extra is blocked:** `requests==2.31.0` conflicts with
    `evidently>=0.5` in `uv lock`. Also unstarted: a statsforecast outside baseline for
    `PORTFOLIO.md` (`references/open-source-shortlist.md`).

## 6. Unstarted and unranked — open ideas, not commitments

- **Composition-break calendar (deep review §12.8).** Persist `n_ask_sources` plus a source
  bitmask, and exclude windows that span a consensus break. There's no record it landed, and the
  vote now spans 20 sources. Two breaks are already known: 2026-03-22 (cash venues outvote a
  lone Steam feed) and 2026-07-09 (feed substitution). If anything here gets promoted, this is
  the candidate: it is data integrity, not another model arm.
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
| 09-29 | h=3 at 20 post-floor dates |
| 10-03 | h=7 at 20 post-floor dates |
| Mon 10-05 | weekly `mode=full` retrain (then 10-12, 10-19, …) |
| ~10-10 | h=14 at 20 post-floor dates |
| ~10-12 | end of the 09-14 workplan's wait window |
| early–mid Oct | h=14/h=30 post-CSMarketAPI confirmation (item 7) |
| mid-Oct at earliest | 20 shared champion–challenger dates (items 5–6) |
| 10-18 | last forecast date in the PID window |
| ~10-23 | the single PID read (item 4) |
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
