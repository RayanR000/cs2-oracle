# SHRINK_K_GBM gated off, VOLATILITY_RANK_GBM stays off: the asymmetry fix

2026-09-10. Two band-geometry GBMs sat in opposite unjustified states, and the
02:59 UTC 2026-09-09 retrain (run 34305388876, at 4949714) baked one of them
into the live artifact:

- `SHRINK_K_GBM=1` shipped in 4949714 with no measurement at all — no harness,
  no paired interval, no changelog. Its per-item K table is in the served
  climatology right now.
- `VOLATILITY_RANK_GBM` sits off behind a commit-message-only local A/B
  (3bfd9a9: 8.1%/6.1%/3.6% narrower at matched 80% for h=3/7/14d, neutral at
  h=30). No harness was committed, so per backend/AGENTS.md the verdict cannot
  be cited — and it stacks with the unmeasured per-item K above, so wiring it
  now would compound two unvalidated geometry changes in one q_hat.

## What changed

1. `price-forecast.yml`: `SHRINK_K_GBM` "1" → "0". The next full/train-only
   retrain rebuilds the climatology table at the flat `CLIMATOLOGY_SHRINK_K=320`
   — the last measured win (f02320b: narrower in 23/23 served-gated paired
   cells). Predict-only runs are unaffected either way: they serve whatever K
   the artifact's table was built with (matched-pair rule), so the revert lands
   atomically at the retrain, in the same train that already ships the
   EXCEEDANCE_META removal, the strictly-prior anomaly label, and the isotonic
   exceedance layer.
2. `VOLATILITY_RANK_GBM` stays off. Not wired, deliberately: the write-down is
   this doc. A band-width claim without a committed harness is exactly what the
   AGENTS.md gotcha forbids citing, and the 6–8% number was measured against a
   flat-320 control that the live artifact no longer serves (it serves
   per-item K). Re-measure, then wire.
3. `backend/scripts/shrink_k_vol_rank_ab.py` (+
   `tests/test_shrink_k_vol_rank_ab.py`): the paired replay both arms need.
   Control is the flat-320 table built by production's own
   `_build_climatology_table` on the fit split; the shrink arm runs
   production's `_compute_per_item_optimal_k` → `_fit_shrink_k_model` →
   `_build_climatology_table_adaptive` per fold; the vol arm runs production's
   `_fit_vol_rank_model` with the multiplier normalised on the fit split.
   Metric is mean half-width at matched 80% coverage per fold, verdict is the
   paired log-width delta plus mean/worst ratio and win count. An arm earns the
   served table only with a negative 95% interval AND no losing fold.

## Measured 2026-09-10: both arms fail — the gating is permanent

Ran the harness full (102 paired walk-forward folds, 26/26/25/25 at h=3/7/14/30,
`/tmp/skvr_full.json`): width at matched 80% coverage, paired log-width delta
arm − control (negative = narrower):

| h | shrink_k_gbm delta [95%] | ratio / wins | vol_rank delta [95%] | ratio / wins |
|---|---|---|---|---|
| 3d | −0.003 [+/−0.006] | 0.997, 16/26 | **+0.112\*** | 1.119, 0/26 |
| 7d | −0.001 [+/−0.004] | 0.999, 15/26 | **+0.097\*** | 1.103, 0/26 |
| 14d | −0.001 [+/−0.004] | 1.000, 12/25 | **+0.062\*** | 1.065, 0/25 |
| 30d | +0.002 [+/−0.006] | 1.002, 11/25 | **+0.100\*** | 1.108, 0/25 |

- Per-item K is a clean NULL: every interval covers zero, mean ratios round to
  1.000, losing folds at every horizon. The GBM transfer adds nothing over the
  flat 320 — which is unsurprising in hindsight: `_compute_per_item_optimal_k`
  picks train-optimal K per item, and the GBM has five item-stat features to
  carry that optimum across items. There is no cross-item structure to carry.
- Vol-rank is refuted, not null: 6–12% WIDER at every horizon, 0 wins in 102
  folds, every interval clear of zero. The 3bfd9a9 commit-message claim (6–8%
  narrower) does not replicate out of sample — it was never committed as a
  harness, so there is nothing to reconcile against; the walk-forward read
  stands. Do not wire it without a new mechanism, not a re-run.
- Caveat, stated not buried: this ran on the local archive copy, which trails
  prod. Staleness moves levels, not a 0/102 shutout — and the shrink null fails
  the deploy rule (no losing fold) on any data. A confirmatory run on the fresh
  archive is welcome but cannot promote either arm; only a NEW arm re-opens
  this.

Follow-up, not this change: delete the SHRINK_K_GBM code path
(`_compute_per_item_optimal_k`, `_fit_shrink_k_model`,
`_build_climatology_table_adaptive`, `shrink_k_models` persist/load) once the
post-revert artifact is serving — load must keep tolerating the baked-in
`shrink_k_*` keys from the current artifact, so the deletion needs its own
careful pass. The vol-rank path (`vol_rank_models`, multiplier) was never
served and has no artifact footprint; same treatment, lower risk.

## What is still open
- `anomaly_p` at h=30 is withheld (None) as of this change
  (`ANOMALY_SERVED_HORIZONS = (3, 7, 14)`): the 30d head ranks (+0.129 AUC) but
  ties the null on log loss, so it is not a quotable probability. The 30d head
  still trains — calibrating it against the pooled rate needs no cold start.
- Backtest is still the throughput constraint: the leg-window resolver fix
  (2a0ace3) landed after the last red run and is unproven, and the panel sits
  at 14/16/11/2 dates against MIN_FORECAST_DATES=20. Nothing here changes that;
  it is wall-clock days, not code.
