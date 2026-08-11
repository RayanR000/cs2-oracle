# Next steps after the 2026-08-10 model audit

**Source:** the audit recorded in `docs/changelog/2026-08-10-constant-call-is-hindsight-picked.md`
and `docs/changelog/2026-08-10-band-and-confidence-are-miscalibrated.md`.
**Supersedes ordering in:** `docs/research/2026-08-09-next-steps.md`. Every item carried forward
keeps its old label (`C1`, `C2`, `C5`, `O2`, `G2`, `D2`–`D5`) so it can be traced. That document
remains the reference for the *content* of those items; this one re-ranks them and adds two tracks.

**Ranked by accuracy gain per minute of work**, which is the stated objective: fast training, fast
A/B, maximum accuracy.

---

## What the audit changed about the priorities

Three things, and the first is the reason for the re-rank.

**1. The model is not badly beaten by anything runnable on directional accuracy.** The published
"edge vs constant call" of −4.08 / −9.65 / −16.01pp at 7/14/30d is measured against a baseline that
picks its direction with hindsight, per fold, and switches sides on 4 of 8 folds at 30d. Against the
runnable always-down call the served classifier is **+3.5 / +0.1 / −1.3 / +4.4pp** — a wash to
modest positive, inside per-fold noise. Full derivation in the companion changelog.

*Consequence:* work aimed at closing a 16pp DA gap is aimed at a gap that does not exist. Deprioritised.

**2. The one real, runnable baseline the model loses to is `−return_1d`, on rank IC, at every
horizon** (−0.0159 / −0.0371 / −0.0433 / −0.0091). This uses no hindsight and survived the
2026-08-09 re-vote. It is the only baseline gap that is both measured and legitimate.

*Consequence:* Track N below, and it is now the top priority.

**3. Two serving-path calibration defects and one structural measurement blocker were found**, none
of which is an accuracy problem in the modelling sense. They are cheap and they gate what can
honestly be published.

*Consequence:* Track F.

**What did not change.** The PT verdict is `skill` at all four horizons on the served signal
(t = 13.18 / 8.00 / 5.40 / 4.31). The model has real within-date signal. Nothing here is an argument
that it is a null model — the argument is that a free one-liner currently has more of the same
signal.

---

## Track N — close the naive-baseline gap (new, highest priority)

### N1. Fit the q50 on top of `−return_1d` via `init_score`

> ## ⚠️ MEASURED 2026-08-10, and this ranking is now wrong.
> `docs/changelog/2026-08-10-instrument-panel-first-read.md`. N1's rank-IC edge is
> **+0.0090 / −0.0115 / +0.0131 / −0.0167** at 3/7/14/30d — still negative at 7d and 30d, so the
> offset does **not** floor the model at the baseline. Worse for the framing here: N1's served
> figures are byte-identical to the control's, because the offset reaches the q50 boosters and
> not the directional classifier that production serves.
> **`C1` below is the arm that cleared the bar**, at all four horizons (+0.0556 / +0.0561 /
> +0.0371 / +0.0316), and it raises served PT excess by 45–96% as well. Track N is no longer the
> top priority; `C1` is.

> **Instrument shipped 2026-08-10, gated off (`NAIVE_INIT_SCORE=1`).** See
> `docs/changelog/2026-08-10-naive-init-score-instrument.md` — it covers all five `Dataset`
> seams and every predict path, and adds a `naive_init_score` arm to `model-diagnostics.yml`.
> What remains is the read: one dispatch per horizon against a control on the same commit.
> Two things the changelog corrects about the plan below: the floor is **empirical, not
> algebraic** (a boosted model can fit its way back below the offset), and a CV rank-IC gain
> does **not** transfer to the served mid until **F1** lands, because `predict` recentres the
> mid on the classifier's call afterwards.

- **Do:** pass `init_score = −return_1d` (in the same units as the target) on the training
  `lgb.Dataset`, and add it back at predict time. The model then fits the *residual* to the naive
  predictor instead of competing with it.
- **Why:** the measured gap is −0.0091 to −0.0433 rank IC at all four horizons. `init_score` makes
  "loses to the one-liner" structurally impossible rather than empirically true: the worst case is
  the model learns nothing and reproduces the baseline. This is the standard boosting-from-an-offset
  pattern and LightGBM supports it natively.
- **Why this and not the swap:** `docs/research/2026-08-10-training-cost-levers.md` §"The structural
  option" records replacing the served mid with the naive predictor, correctly flagging it as a
  product decision rather than a cost tweak. `init_score` gets the same floor without giving up the
  q50, and keeps the conformal band's level.
- **Read:** `mean_rank_ic` against `mean_naive_rank_ic` on the same folds. The bar is
  `rank_ic_edge_vs_naive ≥ 0` — currently negative at all four.
- ⚠️ **Two things to get right.** (a) The offset must be applied on the predict path too, or the
  served mid silently loses the baseline; assert it in a test rather than trusting the symmetry.
  (b) `tuned_params` were selected against the un-offset target, so hold HP fixed for the first
  read and confirm any positive with `FORCE_HP_SEARCH=1` before believing its size — the same
  caveat the tier-lead instrument carries.
- **Vehicle:** `model-diagnostics.yml`, one arm per horizon. It reports rank IC and served
  classifier accuracy and cannot promote its artifact.
- **Cost:** ~10 lines plus a test; one dispatch (~10 min wall clock) to read.

### N2. Split the problem into a market layer and a within-date rank layer

- **Do:** forecast the market factor as a single per-date series (one number per day, not ~900), and
  keep the existing cross-sectional model for relative position. Served item return = market
  forecast + relative component.
- **Why the decomposition is the right shape here:** three independent measurements in this repo say
  the cross-section is dominated by one common factor —
  `da-is-dominated-by-the-market-date`, `no-idiosyncratic-signal-in-features`, and the audit finding
  that the hindsight constant call scores 68.99% at 30d precisely *because* it re-picks that
  factor's sign each fold. The factor is where the predictable variance is; the model currently has
  no term for it.
- **Why it is measurable and the current work is not:** the item-level A/B floor is **2.21–3.69pp**
  fold-clustered (`docs/research/accuracy-opportunities.md`, Measurement Floor) against candidate
  effects of 0.5–1.5pp — those experiments cannot return a decision. The date-level series is a few
  thousand observations of one quantity; the exogenous ingest built for it already exists
  (`docs/changelog/2026-08-06-date-level-exogenous-ingest.md`). Experiments at this layer are
  cheap to run and cheap to power.
- **Cost:** small. One series, seconds to fit, no change to the item pipeline for the first read.
- ⚠️ **Read the prior refutation before starting, and note it is a different change.**
  `docs/changelog/2026-08-06-market-relative-labels-refuted.md` demeaned the **label** and made
  things worse; its own diagnosis is that demeaning spread the label distribution 3.3–9.5× while a
  pointwise quantile loss fought the residual. N2 leaves the label alone and predicts the two
  components separately. The mechanism that killed the earlier attempt does not apply — but that is
  an argument, not a result, and it should be stated as such in whatever this produces.
- ⚠️ **The market factor may be unforecastable.** That is a legitimate outcome and it is worth
  knowing: it would mean the constant call's apparent strength is unreachable in principle, which
  closes the question rather than leaving it open. Budget one experiment, not a programme.

### C2 (carried forward). `lambdarank` within date, scored by rank IC

- **Unchanged from `2026-08-09-next-steps.md`, and its cautions still bind** — the truncation level
  defaults to 30 over a ~900-item cross-section, `label_gain` is exponential, `lambdarank_norm`
  matters, and a ranker emits a score rather than a level so it **adds** four boosters rather than
  replacing them. The paper evidence supports a rank-IC gain and argues *against* expecting a
  performance win.
- **Why it moves up:** the audit sharpened the diagnosis it rests on. Rank IC 0.10–0.18 alongside
  DA at or below the runnable base rate is the signature of a model that orders well and calibrates
  badly, and the loss is pointwise while the served question is within-date ordering.
- **Do N1 first.** It is a tenth of the work against the same metric and the same bar, and if it
  closes the gap the case for adding four boosters weakens considerably.

---

## Track F — fixes that gate what can be published (new)

None of these is an accuracy improvement. All three are cheap, and all three block an honest claim.

### F1. Calibrate `q_hat` around the mid that is actually served

- **Do:** apply `_recenter_on_direction` inside the conformal records path before
  `_calibrate_conformal`, so the calibration centre and the serving centre are the same.
- **Why:** `q_hat` is fitted on residuals to the **q50** mid, then `predict` moves the centre three
  times — prior-day blend, tier bias, and the directional recentring that can translate the mid by
  twice its magnitude or pin it to exactly zero. Half-widths are preserved, which is what breaks
  coverage. Production reports `IntCov` 34.6–61.8% on the `lgbm-v3` cohort against an 80% target.
- **Cost:** no extra model fit. Needs classifier predictions in CV, which
  `CV_DIAGNOSTIC_CLASSIFIER=1` already supplies in `model-diagnostics.yml`.
- **Relationship to C5** (split conformal + ACI, carried forward as R18): F1 is prior and much
  smaller. C5 changes the calibration *scheme*; F1 fixes which centre the current scheme calibrates
  to. Do F1 first — C5 built on the wrong centre inherits the same defect.
- **Add the test that would have caught it:** pooled served coverage on a held-out window against
  `NOMINAL_COVERAGE`. The existing check is unfalsifiable — `q_hat` measured on the OOF it was fitted
  on is ≥80% by construction (`2026-08-10-served-classifier-scored.md` §4).

### F2. Calibrate or withdraw the served `confidence` label

- **Do:** either calibrate the classifier-probability cut against realised hits and store it under
  its own key, or stop publishing `confidence` until it is calibrated.
- **Why:** the served label is a bare `>= 0.5` cut on a 3-class argmax
  (`DIRECTION_CONFIDENCE_HIGH`), never validated. The `confidence_thresholds` that *were* fitted
  describe `_compute_confidence`, which only runs on the no-classifier fallback — so `high_accuracy`
  in `meta.json` and `conf_gap_pp` in the backtest are measuring two different things and have been
  read as one. Separately, the fitted path shows `range_pct` carries no directional information at
  all: it never reaches its 80% target, falls back to maximising coverage, and lands at 39.9 / 41.8 /
  42.6 / 47.1% against a q50 mean of 39.9 / 42.0 / 43.1 / 48.4%.
- **Cost:** small. Withdrawal is a one-line product change and is the right interim state — an
  uncalibrated confidence tag is worse than none, because consumers weight on it.

### F3. Stop encoding configuration in `model_version`

- **Do:** separate the served-model identity from its configuration, so a config flag does not fork
  the scoring panel. Alternatively, score across versions where the artifact is materially the same.
- **Why:** `ops/item_forecasts.parquet` holds **6 distinct forecast dates in total**, split
  `lgbm-v3-regime` (3) / `lgbm-v3` (2) / `lgbm-v3-global-only` (1). `score_cohort` keys on
  `model_version`, so run `31409508960` scored 110,615 frozen outcomes and returned
  `NO HEADLINE (insufficient_dates)` at every horizon. `global-only` and `regime` are the same
  artifact with `SKIP_REGIMES` flipped.
- **Why waiting does not fix it:** `docs/README.md` calls the 20-date failure "a calendar problem,
  not a code problem". Half of it is a code problem. Twenty daily runs yield twenty dates only if
  nothing about the config moves for twenty days — and `SKIP_REGIMES=1` landed 2026-08-10.
- **Cost:** small, and it is the binding constraint on ever publishing a live accuracy figure.

---

## Deprioritised by this audit

- **Anything scoped as "close the constant-call gap".** The gap is against an oracle. This includes
  the framing in `2026-08-10-tier-lead-and-rank-transform-instruments.md`'s "Follows" line; the
  instruments themselves are unaffected and still worth measuring on rank IC.
- **The 30d-horizon question**, but only in priority, not in substance. It remains the only horizon
  whose `feature_validation` fails (`price_technicals` drop 0.18pp, p = 0.40) while its served PT
  reads t = 4.31, and it is ~40% of runtime. The tension is real and unresolved; it is simply
  cheaper to resolve after N1, which changes what the 30d model is fitting.
- **Further retrain-cost work.** The warm arm64 retrain measured **996.6s training / 17m48s job**
  (run `31407938154`), inside the 30-minute cap. `docs/research/2026-08-10-training-cost-levers.md`
  budgets against 1884–2306s runs that predate the warm cache, `SKIP_REGIMES=1` and arm64; its
  remaining levers are real but no longer urgent. **The velocity problem is the experiment loop, not
  the retrain** — see below.

## The measurement problem is now the bottleneck

Worth stating plainly, because it bounds everything in Track N and C:

- The item-level A/B floor is **2.21–3.69pp** fold-clustered; every candidate effect in this repo is
  0.5–1.5pp. Those experiments **cannot return a decision** regardless of how they come out.
- **No `ab_test_*` harness reports rank IC** — `grep -l rank_ic scripts/ab_test_*.py` returns
  nothing — and rank IC is the metric every item in Track N is read on. `model-diagnostics.yml` is
  the only vehicle that reports it.
- The one thing working in your favour: rank IC reproduced **exactly** across two runs at identical
  cached HP and folds (`2026-08-10-served-classifier-scored.md` §1). That bit-reproducibility is the
  control a paired harness would otherwise have to supply, and it is why one dispatch per arm is
  readable — provided the arm does not perturb the item draw.

**Do not build more `ab_test_*` harnesses.** Invest in `model-diagnostics.yml` and in the date-level
layer (N2), where the floor is low enough that an experiment can conclude. This is the same
conclusion `accuracy-opportunities.md` reached — *"the shortest path to any further accuracy work is
a measurement problem, not a feature problem"* — and the audit does not overturn it.

## Carried forward unchanged

Content in `docs/research/2026-08-09-next-steps.md`; re-ranked below N and F.

| Item | State |
|---|---|
| `C1` | ⚠️ **CV-positive, and it does not appear at serving.** CV edge +0.0556 / +0.0561 / +0.0371 / +0.0316 on cached HP (re-confirmed 2026-08-11 with its own control: Δ vs control +0.0686 / +0.0883 / +0.0755 / +0.0483). But the serving replay of the same two artifacts is **worse in 6 of 8 cells**, mean Δ −0.0176 / −0.0478 / −0.0763 / +0.0447. Two anchors, no interval — not a refutation, and not shippable on the CV number either. `changelog/2026-08-11-rank-transform-does-not-transfer-to-serving.md`. |
| `C3` | Residual reversal as a feature. Not started. Note the overlap with N1 — both are `−return_1d` derivatives. |
| `C4` | Re-derive the feature allowlist with `cross_sectional` restored. Compute the MDE first; it may not clear the floor. |
| `C5` | Split conformal + ACI. **Do F1 first.** |
| `C6` | Regime decision — the `SKIP_REGIMES=1` half landed 2026-08-10; `N_ENSEMBLES` untouched. |
| `C7` | Deflate the accumulated A/Bs. Every stored verdict predates the 2026-08-09 re-vote. |
| `O2` | The config-description half is still open. |
| `G2` | Part 2 not started. |
| `D2`–`D5` | Unblocked, none started. |

## Do not re-propose

Everything in the "Do not re-propose" list of `2026-08-09-next-steps.md`, plus:

- **Differencing model DA against `constant_call_accuracy`.** It is hindsight-selected. Use
  `realised_down_rate` as the runnable baseline and the PT verdict as the test.
- **Shaving the retrain further as a priority.** It is inside the cap; the loop is bounded by
  experiment power, not by training seconds.
