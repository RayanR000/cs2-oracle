# The published gate had never written a row, and the embargo it was flipped to costs 10.15pp at 30d

**Date:** 2026-08-08
**Plan:** `docs/research/2026-08-07-next-steps.md` step 5
**Follows:** `docs/changelog/2026-08-08-embargo-and-harness-hygiene.md` — this closes the first
item in its "Still open" list ("the published gate was not re-run")
**Commits:** `f63ae76` (`backend/scripts/walkforward_backtest.py`,
`backend/tests/test_walkforward_embargo.py`)
**Targeted suite:** 111 pass across `tests/test_walkforward_embargo.py`,
`tests/test_walkforward_arms.py`, `tests/test_walkforward_gate.py`,
`tests/test_walkforward_records.py`

Two things landed. A persistence bug that meant the walkforward gate had **never stored a row
in production**, and — once it could store — the first measurement of the embargo
discontinuity the prior entry accepted sight-unseen. Everything below is an **offline gate
number**, not production DA, and the two are not comparable.

## 1. The gate had zero rows in prod: the write reused a session closed 30 minutes earlier

`run_walkforward` opened a DB session at the top of the function, closed it immediately after
`fetch_events`, and then reused that same closed handle for the `_upsert_accuracy` writes at
the very end. Everything between those two points is the fold loop: at the default 500 items
the measured purged run took **29.6 minutes** end to end (the in-code comment rounds it to
~35). So the write asked SQLAlchemy for a pooled connection the Supabase pooler had long since
dropped, and died with `SSL SYSCALL error: EOF detected` — **after all four horizons had been
scored**. The outer `except` then reported "Backtest failed" with every number already
computed and nothing persisted.

The consequence is not "some rows were lost". `prediction_type='walkforward_backtest'` had
**zero rows in production**; no walkforward row had ever landed.

The fix is one line plus its reason: the `if not skip_db:` block opens its own
`SessionLocal()`. **The alternative was rejected.** Holding the first session open across the
whole run would have left an idle transaction against the pooler for half an hour, which is a
worse thing to do to a pooled connection than reopening late.

## 2. The first embargoed gate run, and it is in prod

Run from `backend/` — so, production Supabase — at the defaults: 500 items, all four horizons,
`gbm` arm, `purge=True`. 29.6 minutes. Four rows written with
`model_version = 'lgbm-v4-embargoed'`, `metrics.purge = true`, and `metrics.embargo_days` of
16 / 20 / 27 / 43 — `embargo_days(horizon) == horizon + 13` at each. Presence verified in prod
by direct query after the run.

Headline cohort is `price_tier >= HEADLINE_MIN_TIER` (≥$1), the population production serves.

| horizon | classifier DA | clustered 95% | constant call | PT excess | PT t | PT verdict | headline n | stored `sample_count` |
|---:|---:|---|---:|---:|---:|---|---:|---:|
| 3 | 52.25% | [51.34, 53.20] | 46.17% (up) | +7.68pp | 18.80 | skill | 65,770 | 276,533 |
| 7 | 52.19% | [50.98, 53.41] | 48.67% (up) | +5.83pp | 13.27 | skill | 64,078 | 271,800 |
| 14 | 53.70% | [52.44, 54.99] | 50.30% (up) | +5.30pp | 12.35 | skill | 63,963 | 272,295 |
| 30 | 52.92% | [51.57, 54.26] | 52.97% (up) | +4.74pp | 10.12 | skill | 63,012 | 271,782 |

**The two count columns are not interchangeable.** Every DA in this table is computed on
`headline n` — the ≥$1 rows. `sample_count` is what the row stores, and it counts *all* tiers,
~76% of which are tier-0 penny items whose pooled DA runs 62–66%. Reading a DA across to
`sample_count` overstates the evidence behind it by roughly 4×, and the all-tiers figure
(62.53% at 3d) is the penny metric, not a model result.

### There was no prior series to step from

The prior entry predicted the flip "will appear in the stored `lgbm-v3-clustered` series as a
step of unknown magnitude at the first run after this lands". **There is no such series.** The
persist bug meant nothing had ever been stored under any version string, so there was nothing
to step from and the version bump protected an empty table. The discontinuity could not be read
off consecutive runs; it had to be measured by running a paired arm (§3).

### The constant call is UP in this window, not down

`constant_call_direction` is `up` at all four horizons and `realised_down_rate` is 43–45%
throughout. Down and up do not sum to 100 because the label carries a flat band, and that band
is already reported: `baseline_directional_accuracy` is the always-**flat** call
(`backtest/scoring.py:187`), so the three rates close exactly at every horizon —
44.15 + 46.17 + 9.68, 45.01 + 48.67 + 6.32, 45.09 + 50.30 + 4.61, 43.49 + 52.97 + 3.54, all
100.00. The flat band shrinks monotonically with horizon, which is what lets the constant call
climb from 46.17% at 3d to 52.97% at 30d. This does
**not** refute the recorded "always-down beats the model" observation — it **dates** it. That
observation was a fact about the market periods then in the store; this is a fact about this
window. A constant call's hit rate is a property of the period, which is the entire reason
`backend/AGENTS.md` forbids quoting a DA without it.

### At h=30 the model loses to the constant call

52.92% against 52.97%. PT still returns `skill` with t = 10.12, and both statements are true at
once: **PT tests the model against independence from the realised sign, not against the
constant call.** Any quotation of the 30d number has to carry both, or it is misleading in
whichever direction the quoter prefers.

### h=30 produced folds normally, despite a 43-day embargo

`embargo_days(30) = 43` exceeds `VALIDATION_WINDOW_DAYS = 30`, and the prior entry's framing —
"a 30d fold cannot be built out of 30 days of history" — invites the expectation that h=30
would starve. It did not: **545 dates, 271,782 records, 26 folds**, in line with the other three
horizons. The embargo trims the *train* side of an expanding window that has ample history
behind it; it does not consume the validation window. Worth stating explicitly so a later
reader does not treat a normal h=30 row as evidence the embargo failed to apply — the stored
`embargo_days: 43` is what says it applied.

### `actionable_da` is unbacked and should not be quoted

The run reports `actionable_da` of **94.79% at h=14** (`actionable_n` 1,324, 2.07% of the
scoreable cohort) and **89.11% at h=30** (`actionable_n` 1,487, 2.36%). Both are accompanied by
`actionable_pt_verdict = insufficient_dates`, with `pt_n_dates: 0` and 362 / 343 dates dropped.

So there is **no significance test behind either figure**. A 94.79% hit rate on a 2%
sub-cohort is the single most quotable number this run produced and it is the one number in it
with nothing standing behind it. Recording it here so the next reader who finds it in a report
finds this sentence too.

## 3. The discontinuity, measured: +10.15pp at 30d, unresolved at 3d and 7d

A second arm was run with `--no-purge --skip-db`, on the same 500 items and the same 2,392,831
loaded rows, producing **identical per-horizon record counts** to the purged arm — so the two
arms pair on the same cohort rather than on an intersection.

**It was deliberately not persisted.** `--no-purge`'s own help text says the result must not be
published from, and the risk is concrete: the dashboard headline filters on
`prediction_type == "forecast"`, but **nothing filters by `model_version`**, so a stored
`lgbm-v3-clustered` row would sit beside the `lgbm-v4-embargoed` row on the same
`evaluation_date` with no consumer distinguishing them. `--skip-db` is what keeps the contrast
out of the series it exists to characterise.

Pairing: row grain on `(item_id, forecast_date)`, clustered on `fold_id` (both arms use
`window_end`, per the prior entry's rule that a running counter drifts when one arm skips a
fold), cohort `price_tier >= HEADLINE_MIN_TIER` to match the published headline. Sign
convention: **positive = how much the unpurged split inflates accuracy**.

| horizon | purged | unpurged | inflation | 95% CI | MDE | verdict |
|---:|---:|---:|---:|---|---:|---|
| 3 | 52.25% | 52.71% | +0.47pp | [−1.05, +2.06] | 1.55 | null |
| 7 | 52.19% | 56.53% | +4.29pp | [−0.26, +9.14] | 4.70 | null |
| 14 | 53.70% | 59.14% | +5.44pp | [+1.68, +9.12] | 3.72 | positive |
| 30 | 52.92% | 63.14% | +10.15pp | [+5.00, +15.79] | 5.40 | positive |

`n_paired` 65,770 / 64,078 / 63,963 / 63,012; 27 / 26 / 26 / 26 folds.

**Leakage is confirmed at 14d and 30d**, both intervals excluding zero. At 30d the old split
read 63.14% and the honest one reads 52.92%.

**h=3 and h=7 are `null`, which means UNRESOLVED, not "no leakage."** The design resolves
1.55pp at h=3 and 4.70pp at h=7; h=7's +4.29pp point estimate sits *inside its own MDE*. The
honest statement is that this run cannot tell whether the unpurged h=7 split was inflated by
4pp or not at all.

**A correction from earlier in this same session.** An earlier reading compared h=7's +4.29pp
against the **2.21–3.69pp** fold-clustered floor from
`docs/changelog/2026-08-07-training-item-universe.md` and called it resolved. That was wrong:
that floor belongs to a different design (item-level arms in the training-universe harness),
not to this row-grain gate contrast, whose own MDE at h=7 is 4.70pp. The comparison is not
transferable and the conclusion it produced does not hold.

**At h=30 the embargo reverses the verdict, not just the level.** Unpurged 63.14% against a
52.97% always-up rule looks decisive. Purged 52.92% loses to that same rule. The leak was not
shading a result; it was manufacturing one.

**This retires the imported +12.1pp → +6.1pp figure.**
`docs/changelog/2026-08-08-embargo-and-harness-hygiene.md` §2 quoted that pair from the research
review and pointedly declined to endorse it as evidence about this gate. There is now a local
measurement of the same quantity: **+10.15pp inflation at 30d**, on this gate, this cohort, this
window. That is the number to cite. The imported pair should not be cited at all.

### Caveats that apply to every interval above

- **26 folds is thin.** These are bootstrap intervals over 26–27 clusters.
- **Fold clustering is an improvement, not a proof of independence.** Adjacent expanding-window
  folds share most of their training data.
- **The `paired_mde` run-to-run divergence (~0.155pp between identical commands) is still
  unexplained**, per the prior entry, and applies here too.

## What was deliberately not done

- **The unpurged arm wrote nothing to the database.** Only the four `lgbm-v4-embargoed` rows
  were persisted. §3 above is the reason.
- **The first session was not held open across the run.** §1: a 30-minute idle transaction
  against the pooler is a worse fix than reopening at the write.
- **No connection-level retry or `pool_pre_ping` change was made.** The bug was a stale handle
  reused after `close()`, not a flaky pool; a retry would have masked it.
- **The contrast artifacts were not committed.** Both record sets (~469MB each),
  `paired-embargo-contrast.json` and the pairing script live in a session scratchpad outside
  the repo. **The contrast is therefore reproducible only by re-running both arms** — a measured
  29.6 min purged plus 27.5 min unpurged, 57 minutes in total — not by re-reading a stored file.
- **No h=3 / h=7 follow-up was run to shrink their MDE.** Resolving a sub-2pp effect at h=3
  needs a different design, not a repeat of this one.
- **`actionable_da` was not fixed or suppressed.** It is reported as-is with its
  `insufficient_dates` scope; changing what the gate emits is a separate decision.

## Verification

- `venv/bin/python -m pytest` over `tests/test_walkforward_embargo.py`,
  `tests/test_walkforward_arms.py`, `tests/test_walkforward_gate.py`,
  `tests/test_walkforward_records.py` from `backend/`: **111 pass, 0 failures**.
- New `TestTheGateCanActuallyPersist` in `backend/tests/test_walkforward_embargo.py`, two
  cases: the write block opens its own session (`SessionLocal()` appears after
  `if not skip_db:`), and the early session is still closed promptly (`fetch_events` and
  `db.close()` both still precede the fold loop). The second test exists to forbid the rejected
  fix — without it, "hold the session open" would pass the first test.
- **Prod:** four rows present under `prediction_type='walkforward_backtest'`,
  `model_version='lgbm-v4-embargoed'`, `purge=true`, `embargo_days` 16/20/27/43, queried
  directly after the run. Before this run: zero rows, all-time.

## Still open

- **Every stored A/B still predates the embargo and the universe filter.** This run re-ran the
  *gate*, not the thirteen `ab_test_*` harnesses. The prior entry's "no harness was run" item
  stands unchanged.
- **The archive migration has not been run**; `ingested_at` is still absent from every stored
  Parquet file, so the embargo is still computed from `day` and remains a lower bound on the
  true one.
- **The `paired_mde` run-to-run divergence is still unexplained.**
- **Whether the h=7 split leaks is unknown**, and this design cannot answer it.
- **The three published refutations awaiting re-derivation** (CSFloat, ByMykel, breadth) are
  untouched.
- **Step 6 (frozen-price labels) has not been started.** It is the step that would lower the
  MDE that left h=3 and h=7 unresolved above.

## Related

- `docs/changelog/2026-08-08-embargo-and-harness-hygiene.md` — the instrument change this
  measures; its "Still open" first item is what §2 and §3 close, and its §2 holds the imported
  +12.1pp → +6.1pp pair that §3 retires
- `docs/research/2026-08-07-next-steps.md` step 5 — the plan, updated in this pass
- `docs/changelog/2026-08-07-pesaran-timmermann-headline.md` — why the constant call and
  `realised_down_rate` are quoted beside every DA above
- `docs/changelog/2026-08-07-paired-mde-fold-clustering.md` — the fold-clustered bootstrap
  behind every interval in §3, and the divergence caveat
- `docs/changelog/2026-08-07-training-item-universe.md` — the 2.21–3.69pp item-level floor that
  §3 records as **not** transferable to this contrast
- `docs/changelog/2026-08-07-friction-conditioned-tier-scoring.md` — `actionable_*`, whose
  scope states §2 reads

## Docs touched

This entry, and `docs/research/2026-08-07-next-steps.md` (step 5's "the published gate has not
been re-run" item marked done, pointing here). Nothing under `docs/architecture/` moved —
`backend/AGENTS.md`'s statement that "no run has happened under the new default" is now stale,
but `AGENTS.md` files were out of scope for this pass and are flagged here rather than edited.
