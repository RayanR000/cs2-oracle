# CI stops training regime models, and two accuracy instruments land gated off

**Date:** 2026-08-10
**Follows:** `docs/changelog/2026-08-10-served-classifier-scored.md`, which established that
the served classifier loses to a constant call at 7d/14d/30d, and
`docs/research/2026-08-10-training-cost-levers.md` for the cost budget these fit inside.
**Implements:** Track C's **C1** (the rank transform) and the `cross_sectional`-adjacent half
of **C4** (the tier lead-lag), plus **C6**'s `SKIP_REGIMES` decision.

⚠️ **No accuracy claim is made here. Nothing below has been measured.** Two of the three
changes are instruments that ship **off by default**, and the third is a served change
justified on the shape of the models it removes, not on a measured delta. The A/B is the next
step, not this entry.

## 1. `SKIP_REGIMES=1` in CI — the one change that is live

`price-forecast.yml` now sets it in the `Run ML price forecasting` env block.

**This is a change to the served mid, not a cost saving, and the distinction is the whole
point.** `predict` at `forecaster.py:5658` prefers the regime model whenever
`_detect_current_regime` matches. On this artifact `range` is the only trained regime
(`trained_regimes: ["range"]`; bear and bull hold 0 rows) and it covers **818K–893K of ~985K
rows**, so the "specialist" branch was serving ~90% of the cohort. Dropping it moves those
items onto the global model.

Two reasons, neither of them cost:

- The deployed regime set has included **1-tree and 3-tree boosters**, so it is plausibly
  degrading the mid it displaces (`docs/research/2026-08-08-model-review.md` item 7). That was
  already the standing argument for cutting them; what changed is that the 2026-08-10 warm-cache
  work made the regime path production's steady state rather than a local-only quirk.
- **The served artifact depended on its runner.** CI trained regimes; the documented local
  retrain passes `SKIP_REGIMES=1`. Same code, same data, two different served models. This
  makes the paths agree.

Worth ~183s of a ~1120s warm retrain, which is a side effect and not the justification.

`test_ci_skips_regime_models_so_the_artifact_does_not_depend_on_its_runner` pins it, and its
docstring records the relationship to `test_warm_retrain_still_trains_regime_models` — that
test pins a *different* invariant (regime training is not coupled to the warm-retrain gate) and
must stay true. Revert by deleting the env line; nothing else is coupled.

## 2. `tier_lead_return_1d` — `TIER_LEAD_FEATURE=1`, default off

For an item in `price_tier` t on date d, the mean `return_1d` of tier **t+1** on the previous
observed date. One column, new feature group `tier_lead`, gated the way `bymykel_metadata` is —
the allowlist alone cannot admit it, because an allowlisted-but-unengineered group yields
columns absent and median-filled to zero.

**Why this feature and not another.** It is the only cross-sectional structure ever measured
positive in this archive: lag-1 corr **+0.213 (z = 9.1)**, Granger incremental R² **9.0%**,
stable in 4 of 5 years, and — the clause that matters — it **survives removing the market
factor** (0.122, R² 4.5%). Every refuted feature here died with the common factor;
`2026-08-06-market-relative-labels-refuted.md` showed there is nothing left at the item level
once `m` is subtracted. This is the one candidate that was measured on the other side of that
line. `docs/research/2026-08-07-cs2-forecasting-research.md` §rank-0.

**Direction matters and the folk version is backwards.** Cheap → expensive is +0.043, inside
the noise band. `test_direction_is_expensive_to_cheap_not_the_reverse` pins the sign.

### Three implementation decisions worth knowing

**Mean, not median, and that is a real cost.** `PREDICT_CHUNK_ITEMS` defaults to 1000 against
~5,600 eligible items, so **chunked engineering is the production predict path** — and a tier
index over one chunk is an index over the wrong cross-section. A median does not compose from
`(sum, count)` partials, so a chunked median would need every chunk resident, which is the
thing chunking exists to avoid. The exposure accepted is outlier sensitivity. It lands where it
does least harm: the feature reads the tier *above* each item, so the indices consumed are the
more liquid ones, and tier 0 (sub-$1, the noisiest) is never read because nothing leads it.
`test_chunked_partials_reproduce_the_whole_frame_result` pins that the two paths agree exactly.

**The lag is positional over observed dates, not calendar.** The archive is missing whole days
(`aggregator-archive-day-gaps`). A calendar shift would emit NaN for the day after a gap; every
other lag in this file resolves against observed item-days for the same reason.

**Its own group, not `cross_sectional`.** Folding it in would have inherited that group's
`PREDICT_TAIL_ITEM_DAYS` problem — `market_return_30d_percentile` uses `rolling(365)` against a
240-item-day predict tail, which is why admitting `cross_sectional` requires raising the
constant past 368 and why `test_cross_sectional_features_are_not_served` exists. The tier lead
is lag-1 and carries no such dependency.

## 3. The within-date rank transform — `CROSS_SECTIONAL_RANK=1`, default off

Every surviving feature column mapped to its centred within-date percentile, range [−1, 1].
Applied after the allowlist and the correlation prune, so it runs over ~33 columns rather than
123.

**This is not the refuted experiment**, and the distinction is not cosmetic.
`2026-08-06-market-relative-labels-refuted.md` changed the **label** and left a pointwise
quantile loss fighting a noisy residual — the mechanism its own §"An assumption the spec got
backwards" section diagnoses, where demeaning *spread* the label distribution 3.3–9.5×. This
changes the **features** and leaves the label alone. The features here are scale-free *per
item* (pinned by `tests/test_scale_free_features.py`), which is not the same as
cross-sectionally normalised: every column stays loaded on the common factor on every date,
which is the diagnosed mechanism behind "DA is dominated by the forecast date".

### It deviates from C1's stated formula, deliberately

C1 writes it as `2 * (rank(pct=True) − 0.5)`. Pandas' `pct=True` divides by n, so that maps the
top item to exactly **+1.0** and the bottom to **−1 + 2/n**. The cross-section width varies by
date here — items enter and exit, the same fact that makes `lambdarank_norm` matter — so an
endpoint that moves with n injects a **date-varying artifact into the one transform whose
purpose is removing date effects**. The mid-rank form `(rank − 0.5) / n` is symmetric at every
n. At n ≈ 900 the difference is ~0.1%; it is fixed because it was free to fix, not because it
would have dominated. This was caught by a test written against the intended property rather
than against the implementation.

### Two guards that are not optimisations

**Date-constant columns are skipped.** Ranking a column identical across the cross-section
gives all-ties → pct 0.5 → exactly 0 after centring, i.e. the transform would *delete* it.
Nothing in the current allowlist is date-constant, but the tier lead's upstream index is, and a
date-level feature added later would be erased without a word.

**The predict path raises rather than median-filling.** The engineered cache's key fingerprints
`forecaster.py`'s bytes, so a code change invalidates it — but flipping `TIER_LEAD_FEATURE` is
an environment change the key cannot see, and a frame cached with the flag off is otherwise a
valid hit. Feature alignment would add the missing column as NaN and the median fill would turn
it into a constant, silently. It now fails with the remedy in the message.

## 4. Serving follows the artifact, not the environment

Both flags are written to `meta.json` (`tier_lead`, `cross_sectional_rank`) and restored into
`_artifact_tier_lead` / `_artifact_xs_rank`. `_tier_lead_served` and
`_cross_sectional_rank_served` prefer the artifact; the environment is consulted only when the
artifact does not say.

**Training deliberately reads the environment instead.** A warm retrain restores the previous
artifact's meta, so inheriting its flag would make the treatment arm untestable — it would
train the control.

The default is `None`, not `False`, so "an artifact written before these flags existed" is
distinguishable from "trained with it off". Reading a stored `False` as `None` would let a stray
env var change what an existing model is served — a booster fitted on ranks served raw values is
not a degradation, it is a different input space.

Neither flag joins `MODEL_ARTIFACT_VERSION`, for the reason `bymykel_metadata` does not: with
both off the artifact is byte-identical to one written before they existed, so bumping would
force every checkout into a needless retrain for a change none of them enabled.

## 5. How to measure these — `model-diagnostics.yml`, not an `ab_test_*` harness

**None of the 15 `ab_test_*` harnesses can run either arm, and the reason is worth recording
because it is not obvious from their names.** `ab_test_training_breadth.py`'s arms are item
counts (`N_NARROW` / `N_MID` / `N_WIDE`), not feature configurations. More decisively:
**no harness reports rank IC at all** — `grep -l rank_ic scripts/ab_test_*.py` returns nothing.
Rank IC exists only in the retrain's own CV (`forecaster.py:6088-6116`). And
`ab_test_feature_contribution.py`, which is otherwise the right *shape* for a column ablation,
selects items with `ORDER BY row_count DESC LIMIT n` and no price floor — the penny cohort whose
median item is $0.059.

So `model-diagnostics.yml` gained two dispatch inputs, `tier_lead` and `cross_sectional_rank`,
both defaulting **off**. It is the correct vehicle and it already existed:

- It computes rank IC *and* the served classifier accuracy (`CV_DIAGNOSTIC_CLASSIFIER=1`).
- It reads on production's ≥$1 cohort, folds, and `horizon + 13` embargo.
- It **cannot promote its artifact** — restore step, no save step, `FORECAST_MODEL_DIR`
  deliberately unset. Pinned by `test_diagnostics_workflow_cannot_promote_its_artifact`.
- One horizon per arm64 matrix job: ~9–10 min wall clock, inside `keep-runs-under-30-minutes`.

**Run three dispatches on the same commit:** control, `cross_sectional_rank` alone,
`tier_lead` alone. Track C's instruction to test C1 by itself first stands — it is the cheaper
half and isolates cleanly.

**Read `mean_rank_ic` against `mean_naive_rank_ic` on the same folds, not DA and not 50%.** The
bar is closing the current gap: **−0.0159 / −0.0371 / −0.0433 / −0.0091** at 3/7/14/30d.

**Why one run per arm is readable**, given `ab-fold-count-floor` and `training-item-draw-variance`:
`2026-08-10-served-classifier-scored.md` found rank IC reproduced the retrain *exactly*
(0.1774 / 0.1267 / 0.1023 / 0.0967) at identical cached HP and folds. That bit-reproducibility
is the control a paired harness would otherwise have to supply. It does **not** extend to
anything that perturbs the item draw — this instrument does not, because the ≥$1 cohort fits the
1.2M budget whole and the subsample never runs.

**Two caveats on the read.** The restored cache carries `tuned_params` selected *without* the
tier-lead column, so the tier-lead arm holds HP fixed — the intended first read, but confirm any
positive with `FORCE_HP_SEARCH=1` before believing its size. And the >0.95 correlation prune runs
on raw values before the rank transform, so the `cross_sectional_rank` arm inherits the control's
column set by construction, which is what makes it a clean single-factor contrast.

Do not compare either arm to a stored A/B verdict — every one predates the 2026-08-09 re-vote
(`873148b`).

## Still open

- **The staleness objection on the tier lead is not closed.** Cheap skins have the highest
  zero-change rate (1.33% vs 0.16%), so a partially-updating cheap index could fake this
  signature. The argument against is that the cheap index's own AR(1) is 0.125 — too small a
  footprint for a partial-adjustment model to generate a 0.21 lag-1 cross-correlation. Judged
  mostly real, not proven clean. The decisive test is dropping high-`stale_run_days` item-days
  and re-measuring; it is a filter, not a model, so it is cheap.
- **`SKIP_REGIMES=1`'s served effect is unmeasured.** The next daily run's mids will differ from
  the previous one's for ~90% of the cohort and nothing here quantifies by how much. Diff the
  served mids on a fixed date rather than inferring it from a retrain's CV.
- **C4 proper is untouched.** `FEATURE_GROUP_ALLOWLIST = ["price_technicals"]` still rests on a
  `+3.5pp @30d` that has **never been re-derived**, measured un-embargoed at 100 items with
  median price $0.059, 41% of 3d forward returns exactly zero, 31% free `sign(0)==sign(0)` hits,
  on pre-2026 Steam-only rows. Not to be confused with the `TRAIN_MIN_MEDIAN_PRICE` +3.50pp,
  which *has* been re-derived (to +1.642pp [−0.809, +4.505], null). This entry adds one targeted
  column from that neighbourhood; it does not re-derive the allowlist.
- **The 30d horizon question is untouched.** It is ~40% of the run (751s of 1884s cold) and the
  only horizon whose `feature_validation` fails — `price_technicals` drop 0.18pp, base 50.57 vs
  shuffled 50.38, p = 0.40. That sits in real tension with the served PT reading t = 4.31
  "skill", and the tension should be resolved before anything is cut.

## Verification

Full backend suite green: **1800 passed**, +27 over 1773 (25 new in
`tests/test_tier_lead_and_xs_rank.py`, 2 in `tests/test_minimal_model_shape.py`).

`test_diagnostics_arms_default_to_the_control` pins that the Sunday scheduled run stays a
control: a scheduled event carries no inputs, so both flags must resolve to `'0'` there. If one
ever defaults on, the weekly series silently becomes a treatment arm and stops being comparable
to its own history — the failure this repo already has with pre-`873148b` verdicts.

The arm label in the step summary is composed with `if` blocks rather than `[ test ] && x=y`:
`defaults.run.shell: bash` means `bash -e`, under which a bare failing test fails the step. All
four flag combinations were verified to label correctly and exit 0.

The tests that carry the most weight are the gating ones — both instruments off by default, only
the exact value `"1"` enabling, and `test_engineering_and_allowlist_cannot_disagree`, which pins
that the group is on the same side of the engineering skip and the allowlist whatever the flag
holds. A gated instrument that silently turns itself on is worse than one that does not exist,
because it makes every subsequent A/B uninterpretable.

`test_chunked_partials_reproduce_the_whole_frame_result` is the correctness test for the
production predict path, and `test_rank_transform_is_within_date_not_global` is the one that
would catch the transform degenerating into a no-op.
