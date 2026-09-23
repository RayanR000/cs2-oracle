# Multi-head champion–challenger forecasting backend — design

**Date:** 2026-09-18  
**Status:** approved design; implementation not started  
**Scope:** backend training, serving, persistence, resolution, evaluation, and reporting only  
**Out of scope:** frontend work, automatic production promotion, new data sources, and a generic
model-registry platform

## Summary

CS2 Oracle will present an impressive ML system by assigning each predictive task to the method
that has earned it, rather than forcing one model to produce every output. The production backend
will become a component-aware forecasting graph with five responsibilities:

1. **Centre:** the expected price level.
2. **Interval:** an asymmetric calibrated range around a supplied centre.
3. **Anomaly:** the probability of an unusually large move relative to prior item volatility.
4. **Exceedance:** the probability that an upside move clears estimated round-trip cost.
5. **Ranking:** a cross-sectional score for ordering opportunities within a forecast date.

The existing LightGBM q50 centre remains the initial production champion. A last-price centre runs
in shadow mode on every daily serve. Both are resolved against the same frozen outcome, and a
manual promotion report decides whether last-price may replace the GBM independently at each
horizon. LambdaRank is also shadow-only until it demonstrates transfer on served outcomes.

The direction classifier leaves routine production training. Its code and reproducible benchmark
remain available offline, while the API continues to disclose neutral direction. The measured
anomaly and exceedance heads remain production ML outputs.

This design deliberately avoids an automatic self-modifying system. Evaluation produces evidence;
a reviewed configuration change promotes a candidate.

## Goals

- Preserve the existing production forecast and API while collecting exact, contemporaneous
  challenger predictions.
- Compare centres at identical interval geometry so a centre win cannot be caused by a width
  change.
- Promote or reject candidates per horizon using independent forecast dates, date-blocked
  uncertainty, explicit baselines, and frozen resolved outcomes.
- Keep anomaly and exceedance as first-class ML signals with their existing meanings.
- Evaluate LambdaRank on future served dates without exposing an unvalidated score.
- Remove the failed direction classifier from routine retraining without deleting the negative
  experiment or its reproducibility path.
- Separate publication sufficiency from early served-calibration feedback.
- Produce machine-readable and human-readable reports suitable for a portfolio-quality ML system.

## Non-goals

- This design does not claim that last-price, LambdaRank, or any future challenger has already
  earned production authority.
- It does not change the 3/7/14/30-day horizons, the >=$1 headline cohort, the voted price frame,
  the `horizon + 13` embargo, or canonical outcome resolution.
- It does not replace LightGBM with a transformer or introduce a general experiment-tracking
  service.
- It does not add a frontend or expose shadow predictions through existing public endpoints.
- It does not allow an evaluation job to edit environment variables, model metadata, workflow
  files, database configuration, or champion selection.

## Evidence behind the decomposition

The component boundaries reflect measured results already recorded in the repository:

- The GBM centre produced 6–11% more absolute error than last-price at 3/7/14 days on the served
  replay; the 30-day apparent win had only two clean dates.
- The directional output was anti-informative on the latest clean served panel and is already
  withheld from the API.
- Per-item climatology beat modelled volatility scales for interval width and is already the
  production scale.
- The anomaly head is the first GBM in the project to beat a featureless null at every horizon on
  held-out ranking, with probability calibration supported at 3/7/14 days.
- The exceedance head has useful calibrated short-horizon output and already serves independently
  of the interval scale.
- LambdaRank has a positive clean-anchor offline result, but earlier cross-sectional models failed
  at serving transfer. It therefore starts in shadow mode.

These are task-specific conclusions. A model that fails as a price centre may still succeed as an
anomaly classifier or ranker.

## Architecture

### Component graph

```text
voted price frame
    |
    +-- centre champion ------------------+
    +-- centre challenger ----------------+-- forecast assembler
    +-- anomaly head ---------------------+       |
    +-- exceedance head ------------------+       +-- item_forecasts (production only)
    +-- ranking challenger ---------------+       +-- forecast_candidates (shadow only)
                                                   |
canonical resolver --------------------------------+-- frozen production/candidate outcomes
                                                           |
                                                           +-- operational scoring
                                                           +-- centre promotion report
                                                           +-- ranking transfer report
```

Each component has one typed output and no authority over neighboring components. In particular:

- A centre does not size an interval.
- The interval component does not select a centre.
- Anomaly and exceedance probabilities never recenter or resize the production interval.
- A ranking score is not a return prediction and never becomes `predicted_price_mid`.
- Evaluation does not select or mutate a champion.

### Initial component state

| Component | Production | Shadow | Routine training |
|---|---|---|---|
| Centre | `gbm_q50` | `last_price` | GBM required while configured as champion or challenger |
| Interval | signed conformal + climatology | same offsets on both centres | existing calibration path |
| Anomaly | `anomaly_gbm_v1` | none | enabled |
| Exceedance | `exceedance_gbm_v1` | none | enabled |
| Ranking | none | `lambdarank_v1` | enabled for shadow collection |
| Direction | API-neutral | offline benchmark only | disabled |

After a reviewed centre promotion, the last-price centre becomes production for that horizon and
the GBM becomes its shadow challenger. Promotion at one horizon does not affect another.

## Typed component contracts

Create contracts outside `models/forecaster.py` so training and serving orchestration can depend on
stable interfaces rather than private methods of the 11,000-line class.

```python
@dataclass(frozen=True)
class CentrePrediction:
    name: str
    version: str
    return_pct: np.ndarray
    price: np.ndarray


@dataclass(frozen=True)
class SignedIntervalOffsets:
    lower_pct: np.ndarray
    upper_pct: np.ndarray


@dataclass(frozen=True)
class RankingPrediction:
    name: str
    version: str
    score: np.ndarray
```

Required invariants:

- Arrays in a contract have equal length and preserve the input row order.
- Centre prices are finite and positive for rows eligible to persist.
- `lower_pct <= upper_pct`; both are offsets from the supplied centre, not absolute prices.
- Ranking scores are finite but have no cross-date cardinal interpretation. Only within-date order
  is meaningful.
- Names and versions are stable identifiers, not display labels.

The forecast assembler accepts one `CentrePrediction` and one `SignedIntervalOffsets` and emits
absolute `low`, `mid`, and `high`. Applying the same offsets to two centres produces intervals of
identical percentage geometry:

```text
mid  = centre.price
low  = mid * (1 + lower_pct / 100)
high = mid * (1 + upper_pct / 100)
```

This operation is the only supported way to construct a candidate interval.

## Champion policy

Champion selection is explicit, versioned code configuration:

```python
CENTRE_CHAMPIONS = {
    3: "gbm_q50",
    7: "gbm_q50",
    14: "gbm_q50",
    30: "gbm_q50",
}
```

The mapping must contain every supported horizon and only registered centre names. Startup and
training fail fast on an incomplete or unknown mapping. There is no database override and no
evaluation-driven write path.

The selection module exposes:

```python
def centre_champion(horizon: int) -> str
def centre_challengers(horizon: int) -> tuple[str, ...]
```

Exactly one champion exists per horizon. The first release supports one centre challenger per
horizon but stores candidates in a schema that permits more later.

## Persistence

### Production record

`item_forecasts` remains the authoritative served forecast. Its uniqueness, API behavior, anomaly
field, exceedance field, and Parquet mirror remain unchanged. No shadow output is added to public
schemas in this release.

### Shadow prediction record

Add `forecast_candidates`:

```text
id                       bigint primary key
item_id                  text not null
forecast_date            date not null
horizon_days             integer not null
component                text not null       # centre | ranking
candidate_name           text not null       # last_price | gbm_q50 | lambdarank_v1
candidate_version        text not null
centre_price             double nullable     # centre candidates only
predicted_price_low      double nullable     # centre candidates only
predicted_price_high     double nullable     # centre candidates only
score                    double nullable     # ranking candidates only
anchor_price             double not null
feature_cutoff_at        timestamp not null
artifact_version         text nullable       # null for featureless candidates
config_fingerprint       text not null
created_at               timestamp not null
```

Unique constraint:

```text
(item_id, forecast_date, horizon_days, component, candidate_name, candidate_version)
```

Component-specific checks:

- `component = centre` requires `centre_price`, `predicted_price_low`, and
  `predicted_price_high`; `score` is null.
- `component = ranking` requires `score`; all three price columns are null.
- Centre intervals require `0 < low <= centre <= high`.
- `horizon_days` is one of 3, 7, 14, or 30.

### Frozen candidate outcome

Add `forecast_candidate_outcomes`:

```text
candidate_id             bigint primary key references forecast_candidates(id)
base_price               double not null
actual_price             double not null
resolved_at              timestamp not null
absolute_error           double nullable
percentage_error         double nullable
in_interval              boolean nullable
resolution_version       text not null
```

Outcome values are immutable after first successful resolution. Only an explicit existing-style
`--reresolve` maintenance path may replace them. Ordinary rescoring recomputes derived metrics from
the frozen prices and never reopens the archive.

### Idempotency and contamination guard

A retry may treat an existing candidate row as identical only when its `config_fingerprint`
matches. The fingerprint covers:

- candidate name and version;
- champion mapping;
- interval geometry and calibration identifiers;
- artifact version where applicable;
- relevant feature flags;
- feature cutoff timestamp semantics.

An attempted write to an existing candidate identity with a different fingerprint is a hard error.
The system must never overwrite one experiment arm with another under the same identity.

Candidate writes are atomic per `(forecast_date, horizon, component)`. A partial candidate batch is
rolled back rather than leaving a selected subset of items.

## Daily serving behavior

For each horizon:

1. Build the production centre and its configured shadow centre from the same row set and anchor.
2. Compute the production signed interval offsets once.
3. Assemble production and challenger intervals by applying those identical offsets to each
   centre.
4. Compute anomaly and exceedance outputs from their production heads.
5. Compute the LambdaRank shadow score when its artifact is present.
6. Persist `item_forecasts` exactly as today.
7. Persist candidate centre and ranking batches to `forecast_candidates`.

The production write is authoritative and independent of shadow success. Failure policy:

- A missing or invalid production component fails the forecast run.
- Failure to compute a shadow component logs a structured error and omits that entire
  date/horizon/component batch; it does not fail production.
- Failure to persist a successfully computed shadow batch fails the run after the production write
  only if the failure is a fingerprint conflict or constraint violation. A transient database
  failure logs an error and leaves the batch absent for retry.
- Reports count expected and observed shadow dates so missing collection is visible and cannot be
  mistaken for a model result.

No candidate prediction may be reconstructed later from current prices, a new archive snapshot, or
a newer artifact.

## Resolution

Candidate resolution reuses the canonical production price resolver and all existing rules for
maturity, archive coverage, stale runs, excluded forecast dates, and universe membership.

For an item/date/horizon shared by production and a candidate:

- `base_price`, `actual_price`, and `resolved_at` must be identical;
- the candidate's centre and interval remain those captured at serve time;
- absolute error is `abs(centre_price - actual_price)`;
- percentage error uses the canonical resolved base and existing zero/invalid guards;
- interval membership compares the candidate's stored absolute low/high to `actual_price`.

The resolver reports mismatched shared outcomes as an invariant violation. It must not silently
score the two arms on different estimators.

Ranking outcomes use the canonical realised return from the same frozen base and actual prices.
They do not create an alternative outcome definition.

## Centre promotion report

### Unit of evidence

The independent unit is `forecast_date`, not item row. All intervals and bootstraps resample dates.
Item counts are reported for observability but never used as the effective sample size.

The primary paired metric is:

```text
delta_mae = MAE(last_price) - MAE(gbm_q50)
```

Negative favors last-price. Compute the item-level mean absolute error inside each shared date,
then average paired daily differences. Produce a two-sided 90% date-block bootstrap interval using
a fixed repository seed.

### Eligibility

A horizon is eligible for a promotion verdict only when all are true:

- At least **20 shared resolved forecast dates** exist after all production exclusions.
- Both candidates have at least 95% row overlap on every included date.
- No included date contains mixed candidate versions or config fingerprints.
- At least 80% of the expected daily shadow batches since activation are present.
- Neither arm has a resolution mismatch.

An ineligible horizon reports `INSUFFICIENT_EVIDENCE` with explicit failed conditions.

### PASS gate

Last-price earns `PASS_FOR_MANUAL_PROMOTION` only when all are true:

1. The upper bound of the 90% confidence interval for `delta_mae` is `<= 0`.
2. The point estimate of `delta_mae` is `< 0`.
3. Let `delta_coverage = coverage(last_price) - coverage(gbm_q50)`. The lower bound of its paired
   90% date-block interval is `>= -0.02`, limiting a coverage regression to two percentage points.
4. Median percentage interval width differs by at most `0.001` percentage points. This should be
   mechanically true because the same offsets are used; failure indicates an assembly bug.
5. Production guardrails show no candidate data-integrity failure.

The report also includes median absolute error, dollar coverage, calibrated-basis coverage where
available, row counts, distinct dates, per-date deltas, and version/fingerprint summaries. These are
diagnostics, not additional implicit gates.

### FAIL and unresolved outcomes

- If the lower bound of the 90% interval for `delta_mae` is `> 0`, the candidate is
  `REJECTED` for that horizon.
- If eligibility passes but neither PASS nor rejection is established, the verdict is
  `UNRESOLVED`.
- Verdicts never mutate `CENTRE_CHAMPIONS`.

The report command writes JSON plus a Markdown rendering and exits:

- `0` when every requested horizon is `PASS_FOR_MANUAL_PROMOTION`;
- `1` for `UNRESOLVED` or `INSUFFICIENT_EVIDENCE`;
- `2` for data-integrity or execution failure;
- `3` when any requested horizon is `REJECTED`.

Promotion requires a reviewed code change to `CENTRE_CHAMPIONS`, a dated changelog entry, and an
`experiment_log.csv` row. The first post-promotion run continues collecting the previous champion
as the challenger, providing a rollback comparison.

## Ranking transfer report

LambdaRank remains shadow-only throughout this design. Its report asks whether it transfers to
future served dates; it does not authorize API disclosure.

For every eligible date, compute Spearman rank correlation between:

- `lambdarank_v1.score` and canonical realised return;
- the production q50 centre return and canonical realised return.

The paired metric is daily `rank_ic_lambdarank - rank_ic_q50`. Eligibility requires at least 20
shared dates, at least 100 shared items on each date, consistent candidate version/fingerprint, and
80% expected-batch completeness. Report a 90% date-block bootstrap interval.

The research signal is `SUPPORTED` only when the interval lower bound is `> 0`; otherwise it is
`UNRESOLVED` or `REJECTED`. Even `SUPPORTED` leaves the candidate shadow-only. Public disclosure is
a separate future design because a ranking field changes API and product semantics.

## Direction classifier disposition

Routine production training must not fit, persist, restore, or require the three-class direction
classifier. The API continues returning neutral through the existing serving policy.

The model remains reproducible through an offline command that:

- trains with the production universe, features, fixed rounds, and `horizon + 13` embargo;
- reports Pesaran–Timmermann, directional accuracy, realised down rate, and hindsight constant-call
  accuracy together;
- never writes production artifacts or forecasts;
- exits without changing champion configuration.

Old artifacts containing direction boosters remain loadable during the transition, but production
ignores those files. New artifacts do not require them.

## Publication and calibration thresholds

Split the current global threshold into two named policies:

```python
MIN_HEADLINE_DATES = 20
MIN_FEEDBACK_DATES = 8
```

`MIN_HEADLINE_DATES` gates publishable performance claims, centre promotion eligibility, ranking
transfer conclusions, and Pesaran–Timmermann headlines. It must not be lowered to activate an
operational calibration mechanism.

`MIN_FEEDBACK_DATES` applies only to served-outcome width feedback. Because eight dates remain a
small effective sample, the feedback report must disclose date count and clamp hits. Changing this
constant does not change headline or promotion sufficiency.

The existing `[0.5, 2.0]` feedback clamp remains. A clamp hit is a warning and prevents that panel
from serving as promotion evidence until a subsequent unclamped window is available.

## Artifacts and versioning

The model metadata manifest records component presence and versions independently:

```json
{
  "components": {
    "centre": {"gbm_q50": "..."},
    "interval": {"signed_conformal_climatology": "..."},
    "anomaly": {"anomaly_gbm_v1": "..."},
    "exceedance": {"exceedance_gbm_v1": "..."},
    "ranking": {"lambdarank_v1": "..."},
    "direction": {}
  },
  "centre_champions": {"3": "gbm_q50", "7": "gbm_q50", "14": "gbm_q50", "30": "gbm_q50"}
}
```

The manifest is descriptive, not authoritative: code configuration selects champions. On load,
the backend verifies that every configured learned champion has a matching artifact. Featureless
`last_price` requires no artifact. A missing shadow artifact disables only that shadow component
and is reported.

No feature flag is removed as part of this design. Existing artifact/flag matching remains valid
until a later cleanup accompanies any necessary retrain.

## Module boundaries

The implementation should add focused modules rather than expanding `forecaster.py`:

- `models/prediction_contracts.py` — immutable typed outputs and validation.
- `models/centre_policy.py` — registered centres, per-horizon champion mapping, challengers.
- `models/forecast_assembly.py` — apply signed offsets to a supplied centre.
- `models/candidate_predictions.py` — convert component outputs to persistence records and compute
  fingerprints; no database imports.
- `backtest/candidate_resolution.py` — resolve candidates with the canonical resolver.
- `backtest/candidate_scoring.py` — pure paired daily metrics and date-block intervals.
- `backtest/promotion.py` — eligibility and verdict state machine.
- `scripts/centre_promotion_report.py` — read-only JSON/Markdown report command.
- `scripts/ranking_transfer_report.py` — read-only served ranking report.
- `scripts/direction_benchmark.py` — offline-only direction reproduction.

Database models and write orchestration remain in their existing repository locations. The
stateful `train`, `predict`, and `_train_horizon_inline` methods stay in `ItemForecaster`; the new
modules isolate contracts and policy without attempting a high-risk orchestrator rewrite.

## Observability

Every daily forecast log includes, per horizon:

- configured centre champion;
- candidate names and versions;
- production row count;
- shadow rows computed and persisted;
- config fingerprint prefix;
- reason for any missing shadow batch;
- whether direction training was skipped;
- component training and prediction durations.

Add counters to the existing task result so the zero-row guard can detect silent shadow loss:

```text
forecasts_written
centre_candidates_written
ranking_candidates_written
candidate_batches_expected
candidate_batches_written
```

Production success still depends on `forecasts_written`. Shadow incompleteness is visible in the
promotion report and workflow logs but does not turn a valid production forecast into a false
failure, except for integrity errors defined above.

## Testing strategy

### Contract and assembly tests

- Reject unequal array lengths, non-finite centre prices, invalid signed offset order, and
  non-finite ranking scores.
- Apply one signed offset set to two centres and prove percentage widths are identical.
- Prove champion selection is total over all four horizons and rejects unknown names.
- Prove last-price uses the exact captured serving anchor, not a reconstructed archive price.

### Persistence tests

- Migration creates both tables, constraints, indexes, and foreign key.
- Valid centre and ranking records persist with their component-specific nullability.
- An identical retry is idempotent.
- The same identity with a different fingerprint fails.
- A failed batch leaves zero rows for that date/horizon/component.
- Existing `item_forecasts` writes and API reads are unchanged.

### Resolution tests

- Production and candidate rows share byte-identical base/actual prices and resolution timestamps.
- Candidate outcomes freeze after first resolution.
- Ordinary rescore does not read the archive or move frozen legs.
- `--reresolve` is the only path that may update them.
- Archive maturity, excluded dates, stale-run handling, and unresolvable rows match production.

### Promotion-statistics tests

- Nineteen dates produce `INSUFFICIENT_EVIDENCE`; twenty may produce a verdict.
- Thousands of rows across fewer than twenty dates still fail eligibility.
- A synthetic clear MAE win with neutral coverage passes.
- A point MAE win whose confidence interval crosses zero is `UNRESOLVED`.
- A centre win with coverage lower-bound below −2pp is `UNRESOLVED`, not PASS.
- A clear MAE loss is `REJECTED`.
- Mixed fingerprints, low row overlap, low batch completeness, and resolution mismatches fail
  eligibility.
- Bootstrap output is deterministic under the fixed seed.
- The report never changes champion configuration.

### Ranking tests

- Rank IC is computed within date, never across pooled rows.
- Dates with fewer than 100 shared items are excluded and reported.
- Positive, unresolved, and negative paired-IC fixtures produce the expected research verdict.
- No public API schema or `item_forecasts` column exposes `rank_score`.

### Direction-removal tests

- A routine retrain does not call the direction trainer or require direction artifacts.
- A legacy artifact containing direction boosters loads without affecting production output.
- The API remains neutral.
- The offline benchmark reports PT, DA, realised down rate, and constant-call accuracy together.

### Integration test

On a small synthetic two-date frame:

1. Generate production GBM-centred forecasts, last-price centre candidates, anomaly/exceedance
   outputs, and LambdaRank candidates.
2. Persist production and shadow batches.
3. Resolve both through one canonical outcome fixture.
4. Assert identical outcome legs and different centre errors.
5. Run both reports and verify they remain below the independent-date gates.

The full backend test suite and the existing minimal-model/forecaster gates remain required before
merge. No test may connect to production.

## Rollout

### Phase 1 — contracts and storage

Land typed contracts, centre policy, interval assembly, migrations, persistence, and tests. Keep all
production outputs unchanged.

### Phase 2 — shadow collection

Enable last-price centre and LambdaRank candidate writes. Remove direction from routine training.
Verify candidate batch counts and fingerprints for at least three consecutive daily runs before
treating the collection clock as started.

### Phase 3 — resolution and reports

Resolve matured candidates, publish read-only centre and ranking reports as workflow artifacts, and
keep all verdicts non-authoritative until their gates are met.

### Phase 4 — manual centre promotion

When a horizon reports `PASS_FOR_MANUAL_PROMOTION`, review the report and change only that
horizon's `CENTRE_CHAMPIONS` entry in a separate PR. Retrain only if the new champion requires an
artifact; promoting `last_price` does not. Preserve GBM as the shadow challenger.

### Phase 5 — future ranking decision

A `SUPPORTED` ranking-transfer report justifies a new design for API semantics, calibration, and
consumer use. It does not automatically expose the score.

## Acceptance criteria

The design is implemented when:

- Production output is unchanged while shadow mode is enabled.
- Exact last-price centre and LambdaRank candidate predictions are stored daily with stable
  identities and fingerprints.
- Both arms resolve against identical frozen outcome legs.
- Promotion and ranking reports enforce the stated independent-date gates and confidence rules.
- No report or scorer can mutate champion configuration.
- Routine production training performs no direction-classifier fit and the API stays neutral.
- Headline and feedback date thresholds are separate and correctly routed.
- Existing anomaly and exceedance behavior remains intact.
- Tests demonstrate idempotency, contamination protection, statistical gating, legacy artifact
  compatibility, and absence of public shadow disclosure.

## Deferred decisions

The following require new evidence or a separate design and are intentionally not decided here:

- Whether any centre should be promoted before all four horizons mature.
- Whether a supported ranking score becomes a public API field or only powers a future discovery
  endpoint.
- Whether anomaly probability at 30 days should be recalibrated or withheld.
- Whether served-feedback factors should be shrunk toward 1.0 instead of only clamped.
- Whether the generic candidate store should later support interval-scale, anomaly, or exceedance
  challengers.
