# The research docs were fact-checked; three load-bearing numbers did not survive

**Date:** 2026-08-09
**Type:** documentation audit — no code changed

## Why

All 13 docs in `docs/research/` (~5,700 lines) were checked claim by claim against the working
tree, the local Parquet archive via DuckDB, the GitHub Actions run history, the canonical
`RayanR000/cs2-oracle-data` repo, and the cited literature. This is the companion to
`2026-08-09-roadmap-status-reconciled.md`, which fixed *statuses*; this pass fixed *facts*.

The measurement work held up well. **13 of 19 archive claims in
`2026-08-09-model-and-data-research.md` reproduce to the digit**, the four statistical tables in
`2026-08-09-composition-stability.md` partition exactly, the MDE arithmetic in
`accuracy-opportunities.md` is internally correct, and **no citation anywhere is fabricated** —
all three 2026 arXiv IDs exist and are the papers described. What had drifted was provenance.

## The three that mattered

### 1. `~0.3pp date-level MDE` — a required effect size marked `[MEASURED HERE]`

This is the number that **reopened the closed accuracy roadmap**
(`accuracy-opportunities.md`'s REOPENED banner) and the sole justification for six sections of
`2026-08-07-cs2-forecasting-research.md`. It is not a measurement. 0.3pp originates in the same
file's *stop* banner as the MDE you would **need**, declared "not reachable" — and the changelog
cited as its source retracts it outright:
`2026-08-06-date-level-exogenous-ingest.md:29-38` says the operative date-level floor is the
fold-clustered **2.21–3.69pp** and "**Nothing in these two tables can be resolved below roughly
2pp**."

The measurement-side reopening survives on its other three legs (the Pesaran–Timmermann null, the
unpurged backtest number, three refutations with wrong intervals). What does not survive is
*"aggregate to the date level and the floor drops 10×."*

### 2. The composition gate was refuted by a commit on the same branch

`2026-08-09-next-steps.md` still hard-gated all of Track C and Track D on a rank-IC fall
(0.168 → 0.101, and +0.0044 on 28 dates) that `f833882` had already refuted — those rows are a
2013-2025 → 2026 regime contrast, not a composition contrast, because the partition read a NULL
`source` as never equal to itself. The source doc reproduced the same table in §1c and named it as
finding #1 in its Summary, with no banner. Corrected measurement: **stable +0.1027 vs
unconditional +0.1023** at 3d, equal to four decimals at 7d.

**Scope note, added after `2026-08-09-composition-stability.md` was revised mid-audit.** The revised
primary tables split the old "composition changed" cell into "changed (present)" and "window
incomplete", and the changed cell is now **underpowered at both horizons** (25 dates at 3d, 19 at
7d). So the contrast is **stable-vs-unconditional, not stable-vs-changed**, and no directional claim
about the changed population is supported. The gate still lifts — its premise was that the reversal
is *made of* composition change, and removing composition change does not move the signal on 181 and
167 dates — but two banners written earlier in this pass quoted the withdrawn `changed +0.0965` and
a "7d changed cell is higher" reading, and both were corrected.

### 3. `2026-08-09-model-and-data-research.md` had no record of its own outcome

All six Track A recommendations shipped within hours of the review being written and three
predictions failed: Optuna re-tuning priced at **35s** measured at **392.3s** (11×); the CV fold
cap projected at **−194.6s** moved the conformal-CV phase 439.3s → **434.8s** with the cap binding
on every fold; and ≈32s of recoverable feature work measured **16.5s**. Every share in the
document is against a retired 872s baseline (measured cold retrain is now 1426.3s), which
re-sizes the split-conformal argument from ~50% to ~30% and leaves the regime-model share
unmeasured.

## Claims that were wrong on their own terms

- `2026-08-09-composition-stability.md:4` — "4,967 days". The span 2013-08-14 → 2026-08-08 is
  **4,743** calendar days (4,739 present). 4,967 is the event-calendar row count from the sibling
  doc.
- `_warm_retrain` suppresses **four** things, not three — the fourth is `HORIZON_EXCLUDED_GROUPS`,
  and it was already present at the doc's own commit. **A warm retrain therefore trains 14d/30d on
  the full feature set**, so warm and cold are not the same model at those horizons. Not previously
  recorded anywhere.
- "15 of 15 harnesses still early-stop" is **14 of 15**; `ab_test_direction_labels.py` calls the
  production estimator, whose `early_stopping` default is `False`. (Counting *direct* references
  gives 13 and is wrong — `ab_test_frozen_runs.py` inherits one through
  `walkforward_backtest.py:490`. Verified directly after two audit passes disagreed.)
- `2026-07-19-feature-contribution-by-horizon.md`'s Optimal Config table attributes **72.2%** to a
  "No CS + No Ev" arm that **was never run** — 72.2% is the No-CS-only figure, and the doc's own
  "~+6pp stacked" claim implies ~74.7%.
- `2026-07-21-training-time-optimization.md` never closed its own arithmetic: 64 − 31 min of stated
  savings is 33 min, not the title's "~14 min", and its After table sums to 12.0.
- Two `+3.50pp @30d` results have been conflated repo-wide. `model.md:210,533` cite the
  `TRAIN_MIN_MEDIAN_PRICE` one (re-derived to **+1.642pp [−0.809, +4.505], null**). The
  `ab_test_feature_contribution` cross-sectional `+3.5pp` that founds `FEATURE_GROUP_ALLOWLIST` is a
  **different result and has never been re-derived** — neither refuted nor validated.

## Citation defects

- **The LambdaRankIC table is correct digit-for-digit, but the doc's reading of it is the one
  inference it does not license.** In the paper's own portfolio columns the two arms reachable from
  a stock LightGBM objective are *worse* than the regression baselines — Sharpe 0.566/0.501 against
  0.740 (OLS) and 0.831 (MLP), max drawdown 81–83% vs 44–46%. Only the authors' own method
  dominates, and that needs custom gradient code. The review also omitted an MLP regression row
  whose ICIR (0.806) beats both LTR arms. This weakens **C2**, the next planned accuracy experiment,
  specifically in its cheap form.
- The cross-conformal bound is misattributed: it is Vovk et al. **2018** + Barber et al. 2021
  App. B.2.1. **Vovk (2015) explains why cross-conformal *lacks* the inductive guarantee.**
- `econweb.ucsd.edu/~atimmerm/windowjef.pdf` is the wrong paper for the PT test — it is "Market
  Timing under Model Instability" (2002). The test is Pesaran & Timmermann (1992), *JBES* 10(4).
- "ICIR above 0.5 good, 1.0+ excellent" is **Grinold & Kahn's convention for the Information
  Ratio**, a different statistic. No standard ICIR convention exists.
- Da/Liu/Schaumburg's published multiple is **4×**, not ~3× (the 3× is the earlier working paper).
- The Qlib "Rank IC ≈ 0.05" bar is real but **not like-for-like** — Qlib reaches it at Rank ICIR
  0.39–0.40 against this model's claimed 0.092–0.183 at ICIR 0.34–1.45. Higher IC at
  comparable-or-lower ICIR is a different sampling regime, not a better model.

## Data claims that did not reproduce

Re-queried through `db/archive.py::prices_relation` with `archive_universe_sql_filter()`:

- **`2026-07-09` is `aggregator_sync` alone, not `17mafo`.** The 2026-04-16 → 07-10 window is
  single-feed on all 86 days but `17mafo` on only 85, which is why n on 07-09/07-10 is exactly
  2,372 — the panel is the sync intersection, not the 27,194-item universe.
- §4b mixes three different filters with no filter column. `aggregator_buff163_buy` is **603,243**,
  not "~570k", and `historical_fallback:aggregator_sync` (12,655 rows) is missing from the table.
- §4f's 8,622,895 zero-volume rows is not reproducible under any of eight filter combinations. The
  substance is stronger than stated: 2026-04-16 → 08-07 is **100%** zero, non-NULL. 2026-08-08 is
  327,840 **NULLs** — the crash O1 uncovered.
- §4f's "871 at ≥$1" does not reproduce (898/923/1,050/3,021 depending on an unstated definition),
  and its per-source volume medians are medians over `volume > 0` rows only — the literal median per
  row is **0** for every 2026 source.
- The `STEAM_FEE_MULTIPLIER` circularity finding rests on `backend/runtime/steam_listing_history.db`,
  which **does not exist on this machine**. The nearest available pairing shows no flat-constant
  signature (median 1.163, IQR 0.088, 2.59% identical). Different pairing, so neither confirmed nor
  refuted — but currently unverifiable.
- The 2026-03-22 break is understated: −7.77% at ≥$1 but **−18.10% pooled** median.

## Operational claims

Price Forecast is fixed (run `31337078991`), the canonical archive is complete, and the full
scheduled chain went green. But: **"Backtest is skipped so no labels are maturing" is wrong** —
Backtest has an independent cron and ran green on 08-08; the missing files were **22, not 17** and
were fixed by a *manual push*, not CI; "compaction was never dispatched" is wrong (210.7 MB →
122.6 MB landed, one stranded file left); and **"the one live stale number in `architecture/`" is
one of at least eight**.

Three things nobody had claimed:

1. **`MODEL_ARTIFACT_VERSION` is now 6 and is not on `origin/main`.** Merging without a `mode=full`
   dispatch re-breaks predict-only with `5 != expected 6`.
2. **A CI wall-clock for the shipped config now exists** — `TOTAL training: 1884.2s` (31.4 min),
   job 35m24s. The doc's derived ≈20–25 min was low; the real ratio to the 872s local cold is
   **2.16×**, not the assumed 1.4–1.5×. It is over the project's 30-minute cap.
3. **`ops/collection_runs.parquet` has not been written by CI since 2026-07-21** — the aggregator's
   own evidence-of-collection channel. Aggregator freshness is currently only verifiable from
   `max(day)` in `prices-2026-08.parquet`.

## What was edited

Banners follow `2026-08-08-model-review.md` §5's pattern: numbers kept in place, refuted lines
struck through, every point naming the commit or changelog that settles it.

| Doc | Change |
|---|---|
| `2026-08-09-model-and-data-research.md` | Corrections banner + inline pointers at §Summary 1, §1c, §2b, §3a, §5c |
| `2026-08-09-next-steps.md` | **Gate lifted**; G1 closed; six entry criteria satisfied; C2's evidence, C5/C6 baselines, O2's scope, O3's count and the Answers section corrected |
| `2026-08-09-composition-stability.md` | 4,967 → 4,743 days; reproduce commands for the 2024 window the caveats cite |
| `accuracy-opportunities.md` | The 0.3pp reopening withdrawn; Measurement Floor table marked superseded |
| `2026-08-07-cs2-forecasting-research.md` | Corrections banner (six load-bearing numbers, a shipped-since table, R11–R19 status) |
| `2026-08-07-next-steps.md` | R11 "declined vs unblocked" contradiction resolved; 5d pointer inside the body |
| `2026-07-19-feature-contribution-by-horizon.md` | Banner; the 72.2% table row corrected |
| `2026-07-21-training-time-optimization.md` | **RETIRED** |
| `volume-data.md` | Banner — conclusion stands, reasoning does not |
| `lis-skins-snapshot-plan.md` | Status corrected to "built"; banner on field drift |
| `competitor-analysis.md` | Four inline caveats |
| `README.md` | Reordered so the current review is "start here"; every stale entry annotated |

## The pattern worth keeping

`2026-08-08-model-review.md` is the only doc in the set that was corrected *in place* as its
findings were overturned — struck through, annotated, numbers retained. It was also the easiest to
audit, because every claim carried its own provenance. The docs that were hardest to check were the
ones asserting a number without saying which filter, cut date, or cohort produced it. **§4b's
missing filter column caused more unreproducible numbers in this audit than any actual error did.**
