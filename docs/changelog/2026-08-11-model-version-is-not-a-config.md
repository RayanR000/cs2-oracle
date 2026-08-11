# `model_version` is the served identity, not its configuration

F3 from `docs/research/2026-08-10-next-steps.md`. Shipped 2026-08-11.

**A config flag was forking the scoring panel.** `scripts/forecast_prices.py` wrote
`f"{MODEL_VERSION}-regime"` on the daily path and `f"{MODEL_VERSION}-global-only"` on the
comparison run, and `backtest/scoring.py::score_cohort` keys the cohort on that string. So
`lgbm-v3-regime`, `lgbm-v3-global-only` and the pre-suffix `lgbm-v3` scored as three
unrelated models, and every one of them sat below `MIN_FORECAST_DATES = 20`: run
`31409508960` scored 110,615 frozen outcomes and returned `NO HEADLINE
(insufficient_dates)` at every horizon.

**The suffix bought nothing in exchange, which is what settles it.** `item_forecasts` is
unique on `(item_id, forecast_date, horizon_days)` — no `model_version` in the key — so two
configs writing the same day never coexisted as rows. `on_conflict_do_update` includes
`model_version` in its update set, so the second write **overwrote the first and relabelled
it**. The label never separated a row from another row; it only forked the cohort the row
was scored in.

And since 2026-08-10 it was not even accurate. `SKIP_REGIMES=1` is production, so
`regime_models` is empty and the served mid comes from the global model — while the writer
appended `-regime` unconditionally. The label named a configuration that did not run.

## What changed

**`served_identity(model_version)`** in `backtest/scoring.py` strips a
`SERVING_CONFIG_SUFFIXES` suffix (`-regime`, `-global-only`) and passes everything else
through. Both grouping sites in `scripts/backtest_accuracy.py` use it — the frozen-outcome
scoring path and the resolution path, the latter because its group key is what
`resolve_outcomes` writes to `forecast_outcomes.model_version`, so a suffixed key there
would re-fork the panel from the outcome side on every later run.

**An allowlist, deliberately, not a prefix strip.** `lgbm-v1` and `lgbm-catboost-v2` are
different models; `-ens3` / `-ens6` / `-clustered` are A/B harnesses' own
`prediction_accuracy` series. Merging any of those would pool cohorts that never shared an
artifact. A genuinely new version must be a new identity, and adding one to the list has to
be a deliberate edit.

**The writer now separates the two.** `_write_forecasts_to_db` takes `model_config` and
writes it to the **Parquet mirror only** — the same split `item_slug` already follows: the
ops store is where the archive is analysed, the DB carries what is served. The label is
**observed**, `"regime" if forecaster.regime_models else "global-only"`, not inferred from
which branch is running, so `SKIP_REGIMES=1` describes itself truthfully.

No migration and no production write. Legacy rows keep their stored labels;
`served_identity` canonicalises at the scoring boundary, which is why the historical dates
merge without touching prod and why the suffix stays recoverable.

## The merge is disclosed, not silent

`score_cohort` reports **`config_dates`** — distinct forecast dates per stored label —
and `_headline_line` prints `pooling lgbm-v3-global-only:1d, lgbm-v3-regime:5d` in the
prefix when a cohort pooled more than one. Dates rather than row counts because dates are
the unit `MIN_FORECAST_DATES` counts: a config contributing 1 date to a 20-date panel is a
different claim from one contributing 10.

The raw label is read from the **joined `item_forecasts.model_version`**, not from the
outcome's own column, because the outcome now stores the identity. `config_dates` is
**absent rather than empty** when no record carries a label — an empty dict reads as "pooled
nothing", absent reads as "this payload does not say", which is what a legacy row supports.
This follows the `interval_coverage_dollar_basis` precedent: when a stored series changes
meaning, report the thing that changed alongside it.

## Measured on the ops mirror (`≥$1`, `lgbm-v3*`)

Local mirror, last synced 2026-08-07 — production is a few days further along.

| h | forked cohorts | best single cohort | merged | rows |
|---|---|---|---|---|
| 3 | 3 | 5 dates | **8 dates** | 6,412 |
| 7 | 3 | 4 dates | **7 dates** | 5,024 |
| 14 | 3 | 2 dates | **4 dates** | 2,242 |
| 30 | 1 | 1 date | 1 date | 990 |

**This does not produce a headline and it is not supposed to.** 8 of 20 dates at h=3 is
still `NO HEADLINE`. What changes is that the counter now *accumulates*: before this, twenty
daily runs yielded twenty dates only if nothing about the configuration moved for twenty
days, and `SKIP_REGIMES=1` moved on 2026-08-10. `docs/README.md` calls the 20-date failure
"a calendar problem, not a code problem" — half of it was a code problem, and that half is
fixed. The remaining half is genuinely a calendar: ~12 more daily runs at h=3, and h=30
needs 20 forecast dates each of which matures 30 days later.

## What this does NOT change

- **Bias corrections.** `update_bias_corrections_from_outcomes` already pooled: its query
  filters `fo.model_version LIKE 'lgbm-v3%'` and does not group by version. The
  `--rescore` warning in `2026-08-11-in-interval-basis.md` implied F3 would unblock that fit;
  it does not. That fit is short of dates per **tier**, which is the same calendar problem.
- **The A/B route.** `api/routes/ab_test.py` filters `prediction_type = 'ab_test_regime'`,
  written only by `scripts/ab_test_regime.py`, which sets its own labels. The backtest writes
  `prediction_type = 'forecast'`. The two never shared rows.
- **Any stored number.** No metric moves; only the cohort key does. Historical
  `prediction_accuracy` rows under `lgbm-v3-regime` / `-global-only` stay where they are and
  will simply stop being extended — they are per-evaluation-date outputs, and the series
  continues under `lgbm-v3`. **Do not read a cohort-count change across 2026-08-11 as a data
  change.**

## Out of scope, and worth recording

Two pre-existing defects this touched and deliberately left alone:

1. **`--compare-regime` destroys run A's rows.** Both runs write the same
   `(item_id, forecast_date, horizon_days)` key, so what is stored for that day is the
   global-only prediction. It is not the daily path, and the behaviour is unchanged here.
2. **`forecast_prices.py` returns no field in `run_task.ROW_COUNT_FIELDS`.** `forecasts`
   and `items` are counts but are not on the list, so a forecast run that writes zero rows
   still exits 0 — the failure mode `AGENTS.md` records for collectors, on the forecast leg.

Tests: `tests/test_model_version_identity.py` (the DB row carries the identity, the mirror
carries the config, the config is observed off the forecaster, the DB model deliberately has
no such column, and no writer can rebuild a suffixed version string) and
`tests/test_backtest_scoring.py` (the suffix allowlist, the labels it must *not* merge, the
`unknown` mapping, `config_dates` and its absence, an end-to-end two-config cohort scoring as
one over two dates, the frozen outcome inheriting the canonical label, and both headline
branches).
