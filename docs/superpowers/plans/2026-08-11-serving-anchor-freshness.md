# Quote against a fresher anchor, 2026-08-11

> ## Status — 2026-08-11, all four tasks done
>
> **Arm A passes the gate: dollar error improves at 14 of 16 cells above the placebo, on the
> deviating cohort.** Control `31512455612` / arm `31514765343`, both on `e2c2c80`. The tied
> cohort is 0.00 different in all 16 cells — the negative control works.
>
> **It is a quoting correction and buys no forecasting skill.** The model's edge over its own
> no-change quote moves at 10 of 16 cells (coin flip), and `pinnedIC`'s +0.06–0.40 is the same
> `p/S` free-factor arithmetic that refuted the label arm — moved from the label to the
> prediction. **That corrects task 1's claim**: pinning the denominator does not make rank IC
> comparable across arms, because this arm moves the numerator's level and both legs carry
> `p/S` together. Dollar error is the only metric here no arm touches.
>
> Uncomfortable control-side fact: on the deviating cohort the served quote beats the forecast
> in dollars at **12 of 16 cells**. Full read: `docs/changelog/2026-08-11-serving-anchor-freshness-measured.md`.
>
> **Not shipped.** The arm stays off by default. What blocks the default is the backtest
> resolver (below), not the measurement.

**Raised by** `docs/changelog/2026-08-11-smoothed-anchor-label-measured.md`. The wedge
`p[d]/S[d]` — the last raw quote over the span-bounded median production quotes from — is
what both label bases are contaminated by, in opposite directions. Neither choice of
denominator removes it. This plan attacks the wedge itself, at source, in the **serving
basis** rather than in the label.

## What production does today

`forecaster.py:6443-6462`. Features come from the item's latest row (raw); the base price the
dollar conversion uses is `_smoothed_anchor_prices` — the median of the last 3 observations
within 7 days. Two things about that block matter here:

- **The substitution is unconditional.** The code detects items whose latest price deviates
  >10% from the median, logs `"using smoothed price"` — and then replaces `price` with the
  smoothed value for **every** item, outlier or not. The warning describes a narrower
  behaviour than the line below it implements.
- **So every served item carries the wedge**, not just the spiky ones, and the served
  `current_price` is a number no venue quoted.

## The three arms, cheapest first

| # | Arm | Change | Note |
|---|---|---|---|
| A | Gate the substitution on the outlier test | ~2 lines | Makes the code do what its own comment says. Serves the raw quote unless it deviates >10% |
| B | Shrink the window | `SMOOTH_WINDOW` 3→2 or the span 7→3 | One constant, but it is shared with the backtest resolver — see the coherence hazard |
| C | Serve raw, guarded by staleness | Fall back to the median only when the latest quote sits on a frozen run | Needs `stale_run_days` on the predict path; largest of the three |

**Start with A.** It is the smallest, it is the one the existing comment already claims, and
it isolates the wedge to the items where the smoothing was actually motivated.

## The measurement problem, which is the real work

**This experiment cannot be read the way the last one was.** Two reasons, and both are
structural:

1. **The tied subset is inert here.** Tied means `p[d] = S[d]`, so any arm that changes the
   *choice* between them serves an identical price on that cohort. The rule from
   `2026-08-11-smoothed-anchor-label-measured.md` — read arms on the tied subset — cannot
   referee this one. The effect lives entirely in the deviating cohort.
2. **Rank IC is basis-relative and the arm moves the basis.** `replay_serving.py:321-322`
   divides *both* `actual_ret` and `pred_ret` by `frame["current"]` — the served
   `current_price`. An arm that changes `current_price` changes the label and the prediction
   together, so a rank IC across arms measures nothing. This is the same trap the label arm
   fell into, wearing different clothes.

### What to measure instead

**Primary: absolute dollar error, which is basis-free.** `median(|mid − realised| / realised)`
over the served cohort. A quoting change is a claim about dollars, and dollars are the only
quantity both arms express in the same units. This is the metric that answers "is the number
we publish closer to what happens".

**Secondary: rank IC against a PINNED denominator.** Score both arms with
`D_fixed` = the shipped smoothed anchor, computed from the shipped constants and *not*
following the arm's flag. Then `mid/D_fixed` and `realised/D_fixed` are comparable across
arms. `_basis_frame` already builds `anchor_smooth` this way — it must be pinned explicitly,
or an arm that edits `_smoothed_anchor_prices` silently moves the referee too.

## Tasks

| # | Task | Cost |
|---|---|---|
| 1 | Pin the replay's scoring denominator so it cannot follow the arm; test that it is unchanged when the serving flag flips | ✅ done |
| 2 | Add dollar-error columns to the replay (median and p90 of `\|mid − realised\|/realised`), control and naive alongside | ✅ done |
| 3 | Arm A behind `SERVE_OUTLIER_GATED_ANCHOR=1`, plus the `model-diagnostics.yml` input; test both branches and the >10% boundary | ✅ done |
| 4 | Two dispatches on one commit, four anchors, sequential | ✅ done — **gate PASSED** |

**~2.5h of work plus ~1h of CI.** I estimated "~2h to prototype" in conversation; that was
the arm alone and ignored tasks 1–2, without which the read is not interpretable.

## Gate

**Dollar error on the deviating cohort, control vs arm, at four anchors.** Improvement at
3 of 4 anchors and at 3 of 4 horizons, or the arm is refuted. No interval — four anchors is a
replication, not power, so the sign consistency is the whole read.

If it passes, the follow-on question is whether the *backtest's* resolver should move with it
(below), and only then whether to change the default.

> ✅ **Task 1 landed.** `_pinned_anchor` / `_pinned_rank_ic` / `_pin_matches_production` in
> `scripts/replay_serving.py`, a `pinnedIC` column on the headline table, and `_basis_frame`'s
> `anchor_smooth` pinned too — so the tied/deviating split stays a constant across arms. 10 new
> tests.
>
> **One hazard surfaced that this plan did not anticipate:** `MAX_WINDOW_SPAN_DAYS` is
> `collectors.pipeline.FALLBACK_MAX_AGE_DAYS`, which reads `os.environ` at import — so
> production's smoothing span can move with no commit in this repo, and a literal pin can
> silently stop describing production. `_pin_matches_production()` is checked at run time and
> warns; the reproduction test pins the shipped constants to their defaults so it does not
> pass or fail according to a variable set outside the repo.
>
> Confirmed while pinning: `forecaster.py:6482` is the **only** site that sets the served
> `current_price` (`latest_rows["price"]`, after the unconditional smoothed substitution), so
> arm A has exactly one seam to change.

> ✅ **Task 2 landed.** A `DOLLAR ERROR` table prints on every replay run, unconditionally —
> not behind `--basis-sweep`, because a run missing the gate metric costs an hour to redo.
> Three rows per horizon (pooled / tied / deviating) and four columns: `model` and its `p90`,
> `quote` (**no change** — the served `current_price` itself, which is the reference an arm's
> claim is actually about), and `naive` (`−return_1d` in dollars). 12 new tests.
>
> **Two decisions the plan did not specify.** The naive baseline is converted to dollars
> against the **pinned** anchor, not `frame["current"]` — otherwise the arm moves the baseline
> too and "the model beat naive" shifts for a reason that is not the model. And the tied/
> deviating split is now `_tied_mask`, one definition shared with `_basis_frame`, so the
> dollar gate's deviating cohort and the basis sweep's cannot drift apart.
>
> `n` and `n_naive` are reported per row, which is hazard 4's mitigation.

> ✅ **Task 3 landed.** `ItemForecaster.outlier_gated_anchor_enabled()` and
> `_serving_base_price(price, smoothed) -> (base, deviates)`, called from the one seam in
> `predict()`. Off, the base is `smoothed.fillna(price)` — the shipped unconditional
> substitution, unchanged. On, it is `price.where(~deviates, smoothed)`. 16 new tests, plus a
> `model-diagnostics.yml` `serve_outlier_gated_anchor` input wired into the train step (for
> `predict_smoke`), the replay step, and the summary's arm label.
>
> **`deviates` is arm-invariant, and that is load-bearing.** The `>10%` mask is computed the
> same way under both arms, so the `n items have latest price >10%` warning sizes the same
> cohort in both dispatches, and `ANCHOR_OUTLIER_TOLERANCE = 0.10` is now a named constant
> read by both branches. The boundary is `> 0.10`, exclusive, inherited from the shipped mask:
> a boundary that moved with the flag would sort items into different cohorts for a reason
> that is not the arm.
>
> **Two things the plan did not call out.** The arm leaves **no trace in `meta.json`** — the
> artifact is identical either way — so `predict()`'s warning now names the flag and its state,
> and that log line plus the workflow summary heading are the only record of which price a
> stored replay number was scored against. And the summary's replay grep did not match the
> dollar table: its rows are `h` then a word, the same shape as the basis sweep's split rows,
> so the heading had to be matched or two tables would interleave into one unlabelled block.
>
> Degenerate cases go the safe way under both arms: a NaN or zero smoothed value is not a
> deviation and the raw quote survives, because serving degrades and never drops.

## Hazards

1. **Serving and the backtest share `SMOOTH_WINDOW`.** `backtest/price_resolution.py`
   resolves `base_price` with the same window, and `_smoothed_anchor_prices`' docstring
   records that serving and backtest already diverge by a median 3.6% (p90 35%) when they
   disagree. Arm A changes serving only, so it *widens* that divergence by construction —
   acceptable for a measurement, not acceptable to ship without deciding the backtest side.
   Arm B changes the constant and therefore moves **both**, which is a different experiment,
   not a bigger version of the same one.
2. **A fresher quote is not always a better one.** The raw voted series is 12–27% stale
   across 2026 feeds (`labels-and-embargo`), so "latest" can be a re-publication. The >10%
   outlier gate does not catch a stale quote — a repeat deviates by 0%. That is the argument
   for arm C, and the reason arm A's result should not be read as "raw beats smoothed".
3. **The conformal band rides on the denominator.** `q_hat` is fitted in return space, so a
   different base rescales every served dollar band. F1 (band calibrated around a mid it is
   not served around) is already open and is prior to this — do not read band coverage off
   this experiment.
4. **`MIN_SERVED_PRICE_USD` is applied to `current_price`.** Changing what that is moves
   cohort membership at the floor. Small, but it means the two arms do not score exactly the
   same item set; report `n` per cell.

## Do not

- **Do not change the label to match.** Measured 2026-08-11 and refuted: it wins by
  arithmetic on the deviating cohort and is worth −0.02 to −0.03 where the arithmetic cannot
  operate.
- **Do not read this on rank IC against each arm's own basis.** See the measurement problem
  above; that comparison is meaningless in a way that looks like a large result.
