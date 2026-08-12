# docs/

Refreshed 2026-08-10 against the code. Where a doc and the code disagree, the code wins —
report it rather than working around it.

> ⚠️ **The CV metric that ranked every accuracy arm is measured against a target the
> serving path does not use (2026-08-11).** `prepare_targets` divides by the raw quote at
> the anchor; `predict` quotes against a smoothed median. Swapping only that denominator
> recovers **+0.1398 of the +0.1464** CV↔serving rank IC gap — confirmed in CI on a fresh
> artifact at four non-overlapping anchors, **16 cells of 16**. **No stored rank IC, DA or
> `−return_1d` comparison in this repo is safe to rank arms on.** The corrected label ships
> gated as `LABEL_SMOOTHED_ANCHOR=1`. ⚠️ **Measured 2026-08-11 and NOT shippable**: it swings
> pooled served rank IC to +0.17–0.31 at 4/4 anchors, but the entire gain is the anchor
> deviation `p/S` entering the label as a free factor — on the tied cohort, where that factor
> is 1, it is −0.033/−0.017/−0.020/+0.008. Both label bases are contaminated by `p/S` with
> opposite signs; read arms on the tied subset.
> `changelog/2026-08-11-smoothed-anchor-label-measured.md`.
> `changelog/2026-08-11-the-gap-is-the-anchor-denominator.md`,
> `changelog/2026-08-11-clean-anchor-confirmed-in-ci.md`,
> `changelog/2026-08-11-label-smoothed-anchor.md`.
>
> The same run found the project's **first measured served signal**: rank IC +0.13/+0.16/+0.17
> at 3/7/14d on the third of the cohort whose anchor quote equals its local median, against
> nothing usable (−0.20 at h=3) on the rest. 30d does not replicate (+0.05, 3 of 4 anchors).
>
> **Attacking `p/S` in the SERVING basis works, and it is not an accuracy lever (2026-08-11).**
> `SERVE_OUTLIER_GATED_ANCHOR=1` serves the raw quote unless it deviates >10%; dollar error on
> the deviating cohort improves at **14 of 16 cells** above a composition placebo. But the
> model's edge over simply republishing the served price moves at 10 of 16 — a coin flip — and
> on that cohort the quote **beats** the forecast in dollars at 12 of 16 control cells. Off by
> default; the blocker is the shared backtest resolver.
> `changelog/2026-08-11-serving-anchor-freshness-measured.md`.

## Architecture (`architecture/`)

- `model.md` — the forecaster as it stands: 4 q50 LightGBM models + 4 directional
  classifiers, the split-conformal band, sequential training, age-based retrain
- `model-optimization.md` — size/speed levers, split into already-applied, still-available,
  and 🛑 do-not
- `pipeline.md` — the aggregator and the workflows chained off it; what collects and what
  no longer does
- `data.md` — Parquet archive layout, the `ops/` mirror layer, Supabase serving tables

> **No production directional-accuracy figure is currently quotable.**
> `MIN_FORECAST_DATES = 20` (`backend/backtest/scoring.py`) and live cohorts span 1–5
> distinct forecast dates, so every horizon reports NO HEADLINE. Offline CV DA and
> production DA are not comparable until the served series accumulates ~20 dates.
>
> ⚠️ **Corrected 2026-08-10: this is only half a calendar problem.** `item_forecasts` holds
> **6 distinct forecast dates in total**, split three ways by `model_version`
> (`lgbm-v3-regime` 3 / `lgbm-v3` 2 / `lgbm-v3-global-only` 1). `score_cohort` keys on that
> field and it encodes the *configuration*, so every config change resets the panel — twenty
> daily runs yield twenty dates only if nothing about the config moves for twenty days, and
> `SKIP_REGIMES=1` landed 2026-08-10. Tracked as **F3**.
>
> ✅ **The code half is fixed, 2026-08-11 — and the figure is still not quotable.**
> `served_identity()` keys the cohort on the artifact rather than the configuration, so the
> panel merges to **8 / 7 / 4 / 1** forecast dates at 3/7/14/30d (from a best single cohort of
> 5 / 4 / 2 / 1) and now accumulates instead of resetting. Still under 20 at every horizon:
> the remainder is genuinely the calendar, ~12 more daily runs at h=3 and 20 maturing
> 30-day-old forecasts at h=30. Quote `config_dates` beside any pooled number.
> `changelog/2026-08-11-model-version-is-not-a-config.md`.
>
> ⚠️ **And 20 dates is the wrong bar for `DA − realised_down_rate` specifically (2026-08-11).**
> That statistic's per-date sd is **14.96 / 22.40 / 21.56 / 24.39pp**, so `sd/√20` is 3.3–5.5pp
> against effects of a few pp. `MIN_FORECAST_DATES` was set to span more than one market swing —
> stated in-source as "a judgement call, not a derivation" — and it does that; it does not make
> pooled excess readable. Report the within-date term.
> `changelog/2026-08-11-the-da-gap-is-the-market-direction-of-five-dates.md`.

> ⚠️ **Two published metrics do not mean what they appear to (audited 2026-08-10).**
> `constant_call_accuracy` is **hindsight-selected per fold**, so `edge_vs_constant_call*` is a
> comparison to an oracle and must not be read as a defeat — the runnable baseline is
> `realised_down_rate`, against which the served classifier is +3.5 / +0.1 / −1.3 / +4.4pp at
> 3/7/14/30d. And the served band is calibrated around the q50 mid but served around a
> **recentred** mid, so its 80% coverage claim does not hold (production `IntCov` 34.6–61.8%).
> `changelog/2026-08-10-constant-call-is-hindsight-picked.md`,
> `changelog/2026-08-10-band-and-confidence-are-miscalibrated.md`.
>
> ✅ **The apparent 12–16pp production deficit is composition (2026-08-11, figures corrected
> 2026-08-12).** Decomposing per-date `excess` on the ≥$1
> `lgbm-v3*` panel (**23,073** scorable rows, **12** dates, read from prod Postgres) gives
> **52% / 82% / 103%** of the gap at
> 3/7/14d to the realised direction of the 3–8 anchor dates, against a within-date term of
> **−2.26 / +0.34 / +4.98pp** — positive at h=14, where the model beats its own call mix.
> ⚠️ **The CV-basis agreement table (+2.70 / +0.17 / +0.42 against the published +3.5 / +0.1 / −1.3)
> was NOT recomputed and is unverified**, so "the classifier half is confirmed" is not yet a result;
> what is established is that composition is the first-order term. Per-date sd of `excess` is
> 14.96–24.39pp, so **never quote `excess` on fewer than ~50 forecast dates**; report the
> within-date term, which is what the PT statistic already differences. `MIN_FORECAST_DATES = 20`
> does not rescue it. ⚠️ **The superseded figures came from the *local working copy* of
> `ops/forecast_outcomes.parquet`, which holds only 14,668 of those 23,073 rows with
> verdict-selected gaps. Do not read the local copy for panel work — and note that neither Parquet
> copy is the full panel:** the durable archive CI publishes is fresh and cell-complete (70,409 rows,
> 0.1% by the `evaluated_at > resolved_at` test) but **shallow at 10 dates**, so **the publish leg is
> not broken**. The one defect the read surfaced is
> **diagnosed**: 2026-07-19 called `flat` on 64.0%
> of items at h=3 against a **23.59%** realised flat rate (an 18.3% hit rate, not a near-certain
> miss), because `f71ffb4` shipped a global ±0.5% dead band in `predict()` 42 minutes before that
> run and no directional classifier existed yet. Still reachable as `predict()`'s no-classifier
> fallback; the guard is open. **Correction to the record: 2026-07-19 is the production daily run and
> 2026-07-18's `-global-only` rows are an ablation arm that overwrote that day's production
> forecast** — the reverse of what `da-is-dominated-by-market-date` and prose following it state.
> `changelog/2026-08-11-the-da-gap-is-the-market-direction-of-five-dates.md`.

> ⚠️ **The band half of that is resolved, and the recentring was not the cause (2026-08-11).**
> Served centre, q50 centre and recentring-off agree within **1.1pp** in all 8 cells. The
> 34.6–61.8% was a **basis** artifact: `in_interval` tested an archive-resolved actual against
> a band `predict` had quoted from `current_price`, two anchors that disagree on 85% of rows by
> a median 5.70% / p90 37.82% against half-widths of 10–31%. The band is now rebased before the
> predicate, and **both** figures are reported — `interval_coverage` (calibrated, the basis
> `q_hat` was fitted in and the one `replay_serving.py` reads at ~81%) beside
> `interval_coverage_dollar_basis` (published dollars, the old number). Their gap is the anchor
> wedge. **Do not difference an `interval_coverage` across 2026-08-11.**
> **Migrated in prod the same day** (25,288 rows): all ten ≥$1 cells now read **73.8–93.6%**
> against an 80% nominal, and the `$-basis` column reproduces the old 34.6–61.8% range to the
> decimal — that range was never a calibration figure. The live open question is now
> **over**-coverage: 5 of 10 cells sit at 88–94%, which is C5's problem, well posed at last.
> `changelog/2026-08-11-in-interval-basis.md`,
> `changelog/2026-08-11-conformal-centre-follows-serving.md`.

## Reference (`references/`)

- `steam-api.md` — Steam Market endpoints and response formats, empirically tested.
  The rate-limit envelope applies to **residential IPs only** — hosted CI runners are
  429'd on the first request.
- `data-sources.md` — per-source status, freshness, known issues
- `data-inventory.md` — the canonical coverage audit: what is actually on disk, how much of
  the market it covers, and where the history is thin. Companion to `data-sources.md`, which
  says where the data comes from rather than what arrived
- `catalog-build.md` — Steam catalog scrape: rate-limiting strategy, gap repair
- `backfill.md` — **Dead capability.** CSMarketAPI multi-market backfill; the free-key
  quota never resets and the local DB is empty. Kept for the key-rotation and
  priority-queue design only.

## Research (`research/`)

- `2026-08-09-model-and-data-research.md` — **Start here for anything accuracy-related.** The
  current review. ⚠️ **Read its corrections banner first** — every Track A recommendation in it
  shipped within hours, three of its cost predictions failed, and its §1c gate is refuted.
- `2026-08-07-cs2-forecasting-research.md` — ⚠️ **SUBSTANTIALLY STALE, banner at the top.** Still
  the best analysis and literature review in the repo, but six of its load-bearing numbers are
  refuted or category errors — including the `~0.3pp` date-level MDE, which is a *required* effect
  size marked `[MEASURED HERE]`. 1,706-line external design review, and its C1–C5 correction block
  overturns four of its own first-pass claims. Of those, the BUFF **bid** voting into the consensus
  as an ask is **fixed** (2026-08-07) and `walkforward_backtest.py --purge` being default OFF is
  **fixed** (2026-08-08; the flag is now `--no-purge`); the synthetic 1.1607 Steam fee constant is
  **still unfixed**, tracked as `5c`. Also carries the one measured positive — expensive tiers lead
  cheap tiers by a day, z = 9.1. See `changelog/2026-08-07-cs2-forecasting-research-review.md`.
- `2026-08-10-next-steps.md` — ⭐ **the live action list.** Ranked by accuracy-per-minute after the
  2026-08-10 audit. Adds **Track N** (close the `−return_1d` gap: `init_score`, then a market/rank
  decomposition, then `lambdarank`) and **Track F** (three cheap fixes that gate what can be
  published). Deprioritises anything scoped as closing the constant-call gap, and further
  retrain-cost work — the warm arm64 retrain is **996.6s / 17m48s**, inside the cap, and the
  bottleneck is now experiment power.
- `2026-08-09-next-steps.md` — the previous action list; **ordering superseded**, but still the
  reference for the *content* of every O/G/A/C/D item and its cautions. **Track A is closed
  (all six cost levers shipped 2026-08-09); D1 answered — the Steam listing page works.** **The gate
  is lifted** — Tracks C and D are unblocked. ⚠️ Its "loses to a constant call" framing throughout
  is a comparison to a hindsight-selected baseline.
- `2026-08-10-training-cost-levers.md` — the cost accounting, measured against CI runs
  `31337078991` and `31356483719`; supersedes the cost tables in
  `changelog/2026-08-09-training-cost-levers.md`. ⚠️ **Read its own corrections banner** — three of
  its claims were overturned when levers 1 and 4 landed the same day, including that lever 1 is free
  of served effects (it is not; the regime half moves the served mid). ⚠️ **Also now stale on the
  headline:** both runs it budgets against (1884s, 2306s) predate the warm cache, `SKIP_REGIMES=1`
  and arm64. Run `31407938154` measured **996.6s training / 17m48s job** — inside the 30-minute cap,
  so its remaining levers are real but no longer urgent. Its §"The structural option" (serve the
  naive predictor) is superseded by **N1**, which gets the same floor via `init_score` without
  giving up the q50.
- `2026-08-08-model-review.md` — the model review. ⚠️ **§5's composition rows are refuted in
  place** (2026-08-09); the fall attributed to composition control was a pre-2026-vs-2026 regime
  difference caused by a NULL-unsafe comparison.
- `2026-08-09-composition-stability.md` — the corrected measurement, **complete** (`f833882`).
  **Composition control does not move the reversal** (+0.1027 stable vs +0.1023 unconditional at
  3d; equal to four decimals at 7d), so the quoting-artifact gate on accuracy work is **lifted**.
  Note what is compared: the contrast is stable-vs-unconditional, because both the
  "changed (present)" and "stable & ≥3 sources" cells are 25 dates at 3d / 19 at 7d and carry no
  number. Both are calendar waits.
- `2026-08-07-next-steps.md` — **superseded for ordering** by the 2026-08-09 doc; the descriptions
  remain valid. Steps 1–7 are DONE. Steps 8–11 are NOT STARTED **except** step 10's rank-IC half
  and step 11's reversal measurement. **Blocker 5d is refuted** — it was a false positive.
  Read step 5's "not done" list before citing any A/B result: the harnesses were repaired but
  **none has been re-run**, so every stored A/B number predates the repair.
- `lis-skins-snapshot-plan.md` — ⚠️ **built, not proposed.** Shipped 2026-08-06 as
  `collectors/supply_depth.py` and runs daily; banner records which fields were dropped.
- `accuracy-opportunities.md` — closed 2026-07-31, **reopened 2026-08-07** by the review
  above, which relocates the binding constraint from input data to measurement. The stop
  banner is intact and the tables are still a record of what was tried, not a backlog. Read
  both before proposing accuracy work.
- `2026-07-19-feature-contribution-by-horizon.md` — ⚠️ **bannered.** The ablation behind
  `HORIZON_EXCLUDED_GROUPS` (now a **no-op** — the allowlist already removes those groups at every
  horizon) and the founding `+3.5pp` behind `FEATURE_GROUP_ALLOWLIST`, which has **never been
  re-derived**. Penny cohort, un-embargoed, scored against a 50% benchmark the project rejects.
- `2026-07-21-training-time-optimization.md` — 🛑 **RETIRED.** Its phase ranking is inverted
  (Optuna 59% / CV 2%; measured is CV 50% / Optuna 4%) and levers E, G and J are refuted in code.
  Use `architecture/model-optimization.md` → "Where the time goes now".
- `volume-data.md` — ⚠️ **bannered.** The conclusion (volume adds no predictive lift) stands; the
  |r| < 0.002 reasoning at `:27`/`:142` does not, the free archive source died 2026-04-15, and
  post-2026-03-22 "volume" is a listing count.
- `competitor-analysis.md` — landscape and differentiators; four inline caveats added 2026-08-09
- `2026-07-27-direction-label-sweep-raw.txt` — raw sweep output. ⚠️ **It is a crashed run** — dies
  on an SSL timeout partway through 14d; 30d never ran and no summary line was printed. Nothing
  cites it. Every treatment arm loses to control on the three completed horizons, which is
  consistent with the vol-scaled branch being dead code (`sigma=None` on both paths).

## Design docs and plans (`superpowers/`)

`specs/` holds designs (16), `plans/` the execution checklists (13). Each shipped change is
also recorded in `changelog/`, which is the durable record. Load-bearing ones:

- `specs/2026-08-12-sigma-exponent-design.md` — **designed, not implemented.** The band divides by
  `sigma ** beta`. Read the matched-pair invariant before touching it: `sigma` is ~0.07 so
  `sigma ** 0.4` is ~5× larger and `q_hat` absorbs that, which makes a `q_hat` applied at the wrong
  exponent wrong by ~5×, not partially fixed. Four call sites, one persisted float, `beta = 1.0` the
  no-op default for every pre-existing artifact.
- `specs/2026-07-25-monthly-parquet-partitioning-design.md` — the live partitioning scheme
  in `scripts/append_to_parquet.py`
- `specs/2026-08-01-deterministic-backtest-design.md` — shared-estimator backtest
- `specs/2026-08-03-served-forecast-surface-design.md` — confidence-gate removal, $1 floor
- `specs/2026-08-04-minimal-model-design.md` — the 40→8 model collapse
- `specs/2026-08-05-cv-cohort-parity-design.md` — CV/production cohort mismatch. Its
  residual-gap table rests on 1–2 market days; read it with the NO HEADLINE caveat above.

## Changelog (`changelog/`)

Append-only dated decision records: bug fixes, features, audits, and refuted experiments.
155 entries, 2026-07-08 to 2026-08-12. Entries are never edited to match later reality —
several describe code that has since been deleted, which is the point. Per `AGENTS.md`
workflow rule 2, non-trivial decisions get a new dated note here.

The newest:

- `2026-08-12-sigma-exponent-implemented.md` — the band can divide by `sigma ** beta`, behind
  `SIGMA_EXPONENT=1`, off by default. On the real calibration path the deciles go from
  `41 61 71 78 83 87 90 94 97 99` to **flat 80 at all ten**, marginal coverage unchanged at 80.0% (the
  property that hid the tilt for months, now asserted by a test), width 0.90×. **`q_hat` moves
  833.43 → 151.18 — a 5.5× units shift** — which is why both keys are written together and a missing
  `conformal_beta` defaults to 1.0. It also found a live bug in the shipped `elasticity` diagnostic:
  `denom <= 0` does not catch a constant `sigma`, where `x - x.mean()` is 1e-16 noise, so it returned
  a plausible **0.5** that would have been persisted and served. Two documented deviations from the
  spec (the per-fold `q_hat` stays at β = 1.0 for comparability, plus a new `fold_beta`). 17 tests,
  suite 2052 → 2069. No dispatch, nothing promoted.
- `2026-08-12-the-sigma-scale-is-one-exponent-per-horizon.md` — the open 14d/30d half of the tilt
  remedy, **decided**. Walk-forward over 507–588 dates with production's 14-day refit cadence, four
  arms: shrinkage is a **no-op** (`beta`'s departure from 1 is 6–14× its standard error, so λ =
  0.996–0.999 and the shrunk arm reproduces the plain one), and flexibility **buys nothing** — a
  per-`sigma`-decile `q_hat` is worse at 3/4 and a non-parametric binned scale is better only inside
  the noise, while losing on marginal coverage at 4/4. So: **one fitted exponent per horizon**, which
  held out cuts the level-matched tilt **−89% / −94% / −84% / −73%** and narrows bands to
  **0.87 / 0.86 / 0.84 / 0.77×**. ⚠️ That **contradicts the −26% / −12% at 14d/30d on record** — 43
  independent refits here against one held-out CV fold there, on a model-free panel rather than real
  OOF residuals, so the dispatch settles it. ⚠️ **The pinned selection rule MISFIRED and picked
  production at 4/4**, because it gated on a marginal coverage that drifts 5–10pp between periods for
  every arm; reported as a misfire, with the comparison labelled post-hoc. ⚠️ The tilt and width wins
  are stable across both periods; the **marginal win is not** — on the earlier period the exponent
  arms cover 74–77%, so nothing here fixes the level. Spec:
  `superpowers/specs/2026-08-12-sigma-exponent-design.md`. Nothing shipped.
- `2026-08-12-marginal-over-coverage-is-half-the-sigma-mix.md` — the seventh cause, and the first one
  that **sizes**. The served `sigma` distribution runs **1.28–1.29×** the calibration median
  (measured directly, not implied as `half_pct / q_hat`), and pushing the measured coverage-vs-`sigma`
  curve through it buys **+4.0 to +4.9pp** of the 7.2/11.8/10.6/9.0pp excess — **36–68%**, verdict
  **PARTIAL** at 4/4. Every pre-registered leg passes: validity MAE **0.89–1.43pp**, footprint
  corr **+0.50 to +0.61**, and the `β = 1` placebo at **0.0000pp**, which proves the channel is the
  tilt and nothing else. ⚠️ **Refutes the `0.93×` at 30d** — `sigma` has no horizon term, so that
  rested on production's single 30d forecast date, and the "predicts the opposite at 30d" objection
  is withdrawn (`A` −0.22 → **+0.51**). At **7d the residual excess is unreachable** from this
  channel at any market state (`k* = 2.45×` exceeds every date in two years). The remaining 32–64%
  is **named by identity**: the residual law at given `sigma`, measurable on resolved outcomes.
  Second argument for shipping `β`, now at all four horizons. Nothing shipped.
- `2026-08-12-sigma-tilt-confirmed-on-oof-residuals.md` — ✅ the tilt below is **confirmed on real
  OOF residuals** (run `31619383780`, ~157K records): elasticity **0.429 / 0.369 / 0.350 / 0.313**,
  within **0.014–0.032** of what the model-free instrument predicted, so that instrument is now
  validated against real residuals. The **0.798 / 0.692 / 1.034 / 1.150** on record is refuted at
  4/4 and hardest at 14d/30d, which it had called fine — those are the *worst* horizons
  (deciles **57→96** and **52→97**). Not the clip. ⚠️ **The remedy's reach is narrower than the
  tilt:** held out, the conditional error falls **−76% / −62%** at 3d/7d but only **−26% / −12%** at
  14d/30d, so one global exponent is implementable at the short horizons and needs a shrunk or
  non-parametric scale at the long ones. ⚠️ **It does not explain the marginal over-coverage** —
  level-matching removes that quantity first, and after six causes 87.2/91.8/90.6/89.0% vs 80% is
  still unattributed. `q_hat` and every `fold_q_hat` are byte-identical to `31611508808`. Nothing
  shipped.
- `2026-08-12-the-band-is-tilted-in-sigma.md` — the conditional `q_hat` is designed, costed and
  measured, and **the axis was wrong**. A model-free instrument (validated at
  **1.020 / 0.958 / 0.887 / 0.785×** the shipped `q_hat`) puts 731 dates behind the question for
  seconds per candidate. Its pre-registered read is **VOID** — the shuffled-state **placebo passed
  the bar**, because `mean_d |cov[d] − 80%|` falls whenever *marginal* coverage moves toward
  target, information or not. Level-matched to 80% first, **every date-conditional scheme is worse
  than pooled** (placebos at ≤0.11pp, so there is no noise floor hiding an effect) — a
  time-varying `q_hat` is refuted, at the same wall N2 hit. The defect is the **`sigma`
  exponent**: coverage ramps **62→95%** (h=3) to **58→98%** (h=30) across `sigma` deciles,
  monotone in all ten, stratum error **8.0–9.7pp → 0.6–1.8pp** at a fitted
  **β = 0.408 / 0.401 / 0.363 / 0.327**. Not the clip (1.2% of rows; β moves 0.389→0.395
  excluding them). ⚠️ **Contradicts the 0.798 / 0.692 / 1.034 / 1.150 already on record** — that
  read implied `sigma` as `half_pct / q_hat` from ~20K rows; the magnitude is unresolved and one
  report-only dispatch settles it. Nothing shipped.
- `2026-08-12-expanding-window-refuted-for-band-width.md` — ❌ **REFUTED.** Per-fold `q_hat` is now
  reported: the pooled value sits **0.94 / 0.91 / 0.92 / 0.84×** the p80 of folds already at the
  300K cap (the hypothesis needs it *above* 1), dropping the one small-`n` fold **widens** the
  calibration to 1.06–1.13×, and at identical `n_train` the fold spread is still **1.56–2.25×**.
  That closes the whole *"calibrate on different rows"* class, "use the late folds" included
  (a trailing 2–3 fold window is worse than pooled at 1.13–1.21×). Do not read the audit rho as
  support — `CV_MAX_TRAIN_ROWS` leaves 4 distinct `n_train` values, so it is one fold's leverage.
- `2026-08-12-conformal-basis-follows-serving.md` — ❌ **diagnosed, built, and NOT confirmed.**
  The served band over-covers (**87.2 / 91.8 / 90.6 / 89.0%** against 80%) and `q_hat` is fitted
  on the raw-anchor training label while the band is served and scored on the smoothed anchor.
  The quiet-dates alternative was measured against the calibration window's own dispersion and
  **rejected** (median `rel_cal` 1.01 / 0.96 / 1.07 / 0.97). But the pre-registered check
  **failed and the paired read refuted it**: arm against control on one commit, `q_hat` moves
  *up* at 4/4 horizons where over-coverage needs it 25–39% smaller. Ships off as
  `CONFORMAL_SERVED_BASIS=1`. The wrong sign is itself the finding — it can only happen if the
  booster's own prediction carries `p[d]/S[d]`. `sigma` was then measured and is **also not the
  cause**: `p80(s)` is below 1 in **19 of 20** `sigma` strata, including the lowest quintile at
  every horizon. Leading hypothesis is now the **expanding window** — OOF residuals come from
  fold models trained on 87k–300k rows against the shipped model's 1.2M budget, so a pooled
  `q_hat` is conservative by construction. Conditional coverage (**58.2–99.2%** per
  date) is untouched and ACI cannot be validated on 1–7 forecast dates.
- `2026-08-12-served-confidence-withdrawn.md` — F2. The `confidence` tag leaves `PredictionOut`
  and `TrendAnalysisOut`. Within-date, on the 11 cells with `n_high >= 30`, the `high` cohort is
  right **29.0–45.8%** against a stated 80% target and its gap to `low` is mixed-sign — so it is
  withdrawn as uninformative and mislabelled, not as inverted. **The backtest's `conf_gap_pp` is
  pooled across dates and must not be quoted for this**; its −38.7pp at h=30 is one item. Column,
  writer and metric all kept.

Behind it, the instrument panel and the 2026-08-10 audit, both of which carry corrections that
reach back into earlier entries:

- `2026-08-10-instrument-panel-first-read.md` — ⭐ four arms on one commit. The cross-sectional
  rank transform (`C1`) is the first arm to beat `−return_1d` on rank IC, at all four horizons,
  and it lifts served PT excess 45–96%. `init_score` (`N1`) does not clear the bar and never
  touches the served classifier; `tier_lead` closes the gap nowhere. Re-ranks
  `research/2026-08-10-next-steps.md`, which had put Track N first.
- `2026-08-10-rank-transform-reference-cohort.md` — the transform is fitted on 916 items and
  `predict`'s frame holds 5,536, so serving ranked against the wrong population. Fixed by ranking
  every row against the >= $1 cohort's distribution; sub-$1 items keep their unserved forecast
  rows. `predict` now refuses an artifact that does not record its cohort.
- `2026-08-10-constant-call-is-hindsight-picked.md` — `constant_call_accuracy` is selected with
  hindsight per fold, so `edge_vs_constant_call*` compares to an oracle; the runnable baseline is
  `realised_down_rate`. Also: `model_version` fragments the scoring panel, which is why no headline
  publishes. Corrects three earlier 2026-08-10 entries in place.
- `2026-08-10-band-and-confidence-are-miscalibrated.md` — the conformal band is calibrated around
  the q50 mid then served around a recentred one; the served `confidence` label is an uncalibrated
  0.5 cut, and the thresholds that *were* fitted describe a path production does not take.
  Diagnosis only, nothing fixed.

## Other

- `code-review-2026-07-21.md` — **Live punch list**, findings re-verified 2026-08-05.
  Most are still open, and the security findings cluster (SQL f-strings, default secret
  key, session token in a redirect URL). Separates LIVE from DORMANT.
- `operations.md` — runbook: workflow schedules, required secrets, load-bearing steps,
  troubleshooting
- `design.md` — ⚠️ **describes the frontend deleted 2026-08-10.** Visual design system:
  OKLCH palette, typography, spacing, components. Bannered, and kept as **rebuild input
  only** — the `frontend/app/*` paths it references no longer exist
- `product.md` — ⚠️ **same: rebuild input, not a live spec.** Positioning, users, brand
  personality, design principles, written in the present tense about an interface that
  no longer ships

## Removed 2026-08-05

`historical/` (5 files) and `retrain-optimization-analysis.md` were deleted — the first
documented only resolved issues, the second optimized a 36-model quantile grid that no
longer exists. Both are recoverable from git history if needed.
