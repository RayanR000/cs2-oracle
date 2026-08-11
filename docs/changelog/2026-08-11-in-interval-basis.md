# `in_interval` is measured on the basis the band was quoted from

The open question `2026-08-11-conformal-centre-follows-serving.md` surfaced, and it was the
same defect `2026-08-11-actionable-selection-is-the-base-wedge.md` fixed one metric earlier.
Shipped 2026-08-11.

## The discrepancy that named it

Production reports `IntCov` **34.6–61.8%** on the `lgbm-v3` cohort against an 80% target.
`scripts/replay_serving.py`'s `BAND COVERAGE` reads **~81% pooled** on the same band — six of
eight cells at or above nominal across two anchors. Two figures that far apart describing one
band do not both describe the calibration.

They do not, and the difference is entirely mechanical. The replay computes
`actual_ret = realised / current - 1` (`replay_serving.py:617`) — band and outcome share the
anchor the forecast was quoted from. Production's `_derive_verdict` tested
`low <= actual <= high` where `low`/`high` were built by `predict()` as
`current_price x (1 + ret)` and `actual` is resolved off `base_price` by `resolve_anchors`.
The two bases disagree on **85.3%** of rows, by a median **5.70%** / p90 **37.82%**, against
half-widths of **10–31%**. The wedge was deciding coverage.

The F1 arms already ruled out the alternative: served centre, q50 centre and recentring-off
agree within **1.1pp** in all eight cells, so the conformal centre is not the cause either.

## What changed

`_derive_verdict` takes a **keyword-only, required** `quote` and rebases the band by
`base / quote` before the predicate:

    low/quote - 1  <=  actual/base - 1  <=  high/quote - 1

which is the inequality with the divisions cleared. `quote` has no default deliberately —
a default is how this defect survived being fixed on the actionable leg and not here. All
three verdict paths pass it: the first resolution (`f.current_price`), the re-derived scoring
records, and the Task 8c refresh. `_quote_basis` falls back to `base_price` for the two
populations with no separate quote — legacy outcomes predating the column, and
`walkforward_records`, which builds `mid = base * (1 + mid_ret)` so the resolved base *is*
its quote.

**Neither leg of the outcome moves.** `base` and `actual` both stay on `resolve_anchors`.
This does not go near the rule that broke a cohort into 61.76%/33.74% on consecutive days;
only the prediction is rebased, exactly as with `r_hat`.

## Both figures are reported, because there are two questions

Rebasing `in_interval` and stopping there would have deleted a true number. The API publishes
a **dollar** band and a consumer reads it in dollars — "did the band we served contain the
price" is a real product question, and its answer really is 34.6–61.8%.

| key | question |
|---|---|
| `interval_coverage` | the **calibrated** band: the basis `q_hat` was fitted in, and the predicate the replay reports |
| `interval_coverage_dollar_basis` | the **published** band: did the served dollars contain the resolved price |
| `interval_n_served_basis` / `interval_n_fallback_basis` | which convention formed each row's band |

Their gap **is** the anchor wedge, which turns it from a mystery into an attributable
quantity. The counts follow the `actionable_n_served_basis` precedent for the same reason: the
stored series breaks at 2026-08-11, and a payload that cannot name its own convention is not
self-describing. `_headline_line` prints both together — the calibrated figure alone invites
the reading that the served band covers, and the dollar figure alone is what was mistaken for
a calibration defect.

`in_interval_dollar` and `interval_basis_served` live on the scoring record only.
`_verdict_for_storage` now filters to `VERDICT_COLUMNS`, so `_REFRESH_VERDICTS_SQL` binds
exactly the columns it names and no migration is needed.

## What did NOT move, and why

**`abs_error` and `pct_error` are not rebased.** A dollar error is basis-free: "how far was
the published price from the realised one" is answerable without asking what the prediction
was quoted from, and a forecast quoted off a stale anchor really is that wrong. Only a
predicate about a *calibrated width* needs the width's own basis. This is the same reasoning
that made dollar error the metric two anchor arms could share (`4109a43`).

**The conformal calibration is untouched.** `q_hat` was already fitted in return space off the
quote; it was the *comparison* that was wrong, which is what the F1 changelog predicted.

## Migration

`_refresh_verdict_columns` rewrites `in_interval` on every stored row whose verdict now
disagrees, through the existing Task 8c path — no backfill script, and the frozen actuals are
structurally out of reach. **Run 2026-08-11 against production: 25,288 rows refreshed, exit 0**,
after the fix was pushed as `e4b57e9` — order matters, because the daily path refreshes only
that run's mature ids (`:1346`) while `--rescore` walks the table (`:1014`), so migrating ahead
of the push would have left prod mixed-basis as CI kept refreshing fresh rows on the old
predicate. **Do not difference an `interval_coverage` across 2026-08-11** in either direction;
the pre-change series is `interval_coverage_dollar_basis` under the old name.

⚠️ **`--rescore` also refits production bias corrections.** `update_bias` is hardcoded `True`
(`backtest_accuracy.py:1418`) with no flag to disable it, so the run loads the models and calls
`update_bias_corrections_from_outcomes`. On this run every tier refused to fit (1–5 distinct
forecast dates against the 20 required) and stored thresholds were left unchanged, so it was a
no-op — but it is a served-config write, not a scoring refresh, and it will stop being a no-op
once F3 lets the date panel accumulate.

## Measured in production, `--rescore` 2026-08-11 (≥$1 cohort)

25,288 stored rows refreshed; frozen actuals untouched; exit 0.

| cohort | n | `interval_coverage` | `$-basis` |
|---|---|---|---|
| 3d / lgbm-v3 | 2,092 | 73.8% | 54.6% |
| 3d / global-only | 1,053 | 90.9% | 87.8% |
| 3d / regime | 5,283 | 88.3% | 46.0% |
| 7d / lgbm-v3 | 2,095 | 80.9% | 57.3% |
| 7d / global-only | 1,053 | 93.3% | 91.8% |
| 7d / regime | 5,346 | 93.6% | 78.2% |
| 14d / lgbm-v3 | 2,093 | 87.0% | 61.8% |
| 14d / global-only | 1,053 | 92.6% | 91.6% |
| 14d / regime | 1,052 | 82.8% | 79.8% |
| 30d / lgbm-v3 | 990 | **88.8%** | **34.6%** |

**The `$-basis` column reproduces the published range exactly.** `lgbm-v3` reads 54.6 / 57.3 /
61.8 / 34.6 — the "34.6–61.8%" that has been quoted as a calibration defect since 2026-08-10 is
its minimum and maximum, recovered to the decimal. That is the confirmation: the old figure was
never measuring the band's calibration, and nothing about the band changed today.

On the corrected basis all ten cells land at **73.8–93.6%**, bracketing the 80% nominal and
agreeing with `replay_serving.py`'s ~81%. The two instruments now describe one band.

**The new open question is over-coverage, not under.** Five of ten cells sit at 88–94%, which
is a band wider than its target — the opposite failure, and a real one, since an interval that
covers 94% at a nominal 80% is uninformative at the width it charges for. That is C5's problem
and it is now well posed for the first time. Note also that every cell still spans 1–5 forecast
dates against `MIN_FORECAST_DATES = 20`, so each reads `NO HEADLINE`; these are diagnostics, not
a published series, and **F3** is what gates that.

## Caveats

- The ~81% replay figure is **two anchors**, and the 05-16 → 07-09 swing (85.5 → 64.4 at h=3)
  dwarfs every arm difference. That is replication, not power, and it is the market-date
  dominance already on record (`da-is-dominated-by-the-market-date`). The prediction that
  production's `IntCov` will land near the replay's is a prediction, not a result, until a
  `--rescore` reports it.
- `conf_high_interval_cov` follows `in_interval` and so changes basis with it. It is still
  gated on the uncalibrated `>= 0.5` confidence tag that **F2** exists to withdraw, so it was
  not quotable before this change and is not quotable after it.

Tests: `tests/test_backtest_scoring.py` — the rebase in both directions, the fallback for the
three no-quote cases, a missing band staying `None`, the stored-column filter, both reported
figures with their basis counts, the `walkforward_records` default, and one end-to-end run
through the real resolver where a +5% move inside a [-10%, +15%] band was previously scored as
a miss.
