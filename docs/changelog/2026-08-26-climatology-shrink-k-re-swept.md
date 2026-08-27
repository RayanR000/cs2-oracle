# The climatology shrink constant was tuned on a truncated sweep

**2026-08-26.** `CLIMATOLOGY_SHRINK_K` goes 20 -> 320. At matched 80% coverage
the served band is **6.5 / 7.4 / 9.7 / 15.4% narrower** at h=3/7/14/30, winning
4 of 4 walk-forward folds at every horizon with every worst-fold ratio below 1.
No new flag, no new denominator, no retuning of anything else.

## How it was missed

The climatology band scale shipped 2026-08-20 with K=20, recorded as
"insensitive 5..100" (`docs/research/2026-08-19-climatology-vs-gbm-band.md`).
That sweep **stopped at 100 and the optimum is past it.** Re-swept past the old
ceiling, the optimum sits on a broad **240..640 plateau**, flat to under 1%
across it. K=320 is taken mid-plateau so the choice does not depend on the exact
value.

| K | h=3 | h=7 | h=14 | h=30 |
|---|---|---|---|---|
| 0 (pure per-item) | 1.101 | 1.116 | 1.131 | 1.133 |
| 20 (**was prod**) | 1.000 | 1.000 | 1.000 | 1.000 |
| 40 | 0.974 | 0.971 | 0.960 | 0.947 |
| 80 | 0.954 | 0.946 | 0.927 | 0.896 |
| 160 | 0.939 | 0.931 | 0.905 | 0.860 |
| **320 (now prod)** | **0.936** | **0.926** | **0.903** | **0.846** |
| 640 | 0.938 | 0.930 | 0.907 | 0.842 |
| pooled constant | 0.948 | 0.945 | 0.924 | 0.850 |

Ratios are the mean of the PER-FOLD matched-coverage width against K=20, so they
are not distorted by folds sitting in wider eras.

## What the finding actually is

**Per-item dispersion over-fits item-level noise, worse at longer horizons.** A
single pooled constant -- featureless AND itemless -- also beats K=20 at every
horizon, and loses to the plateau by under 1%. So a little per-item information
is real; K=20 simply over-weighted it. At K=20 a 200-observation item carried
weight 0.91 on its own estimate; at K=320 it carries 0.38.

This is **variance reduction, not signal extraction**, which is why it survives
out of sample where the modelled-sigma denominators did not. The root
`AGENTS.md` rule stands unchanged: the dead end is the modelled-sigma family,
not band width as such.

## How it was found, and what was refuted on the way

`scripts/magnitude_vs_climatology.py` was built to ask a different question --
whether a LightGBM trained directly on `|r_h|` beats climatology, on the theory
that magnitude is predictable where direction is not. On a single temporal split
it looked like a clear win (0.962 / 0.920 / 0.835 at h=7/14/30, CIs clear of 1,
reproduced under an L1 objective).

**`--folds 4` refuted it.** Two h=30 folds won by ~16% after only ONE boosting
round, which prompted adding a pooled-constant arm. Over 16 folds the booster
matched the pooled constant (`mag/pool` ~ 1.0 in 14 of 16) while the pooled
constant beat per-item climatology in all 16. The features add nothing; the
shrinkage was doing all the work. A `MAGNITUDE_BAND` denominator was planned and
is **not being built.**

## Serving consequence

Pooling harder changes the served half-widths, so this is a third
band-geometry change on the panel that `models/served_recalibration.py` reads.
`SHRINK_K_SERVING_START` is added to `_geometry_floor()` and **ships as `None`**:
the forecast chain is paused for billing, so no prod forecast has served K=320
yet. Set it to the first deploy whose artifact was BUILT at K=320 -- the tables
are persisted in `meta.json`, so a predict-only run on a K=20 cache still serves
the old geometry.

### Gated at serving (2026-08-26, local paired replay)

Two artifacts were trained from the same archive into scratch directories, one
at K=320 and one at K=20, and replayed through `predict()` at two anchors.
**`tuned_params` and `feature_cols` came out IDENTICAL between the arms** — K
enters only at `_fit_climatology_scale` / `_calibrate_conformal`, after tuning —
so the pair differs in exactly one value. q_hat fell at every horizon
(0.850/0.820/0.812/1.242 -> 0.765/0.757/0.745/1.123) while the per-item tables
lost cross-item dispersion (h=14 stdev 9.02 -> 6.77), which is the mechanism.

Served half-width %, K=320 vs K=20:

| anchor | h=3 | h=7 | h=14 | h=30 |
|---|---|---|---|---|
| 2026-06-08 (n=1,061) | 7.61 / 7.99 | 9.99 / 10.49 | 13.04 / 13.78 | 26.51 / 28.25 |
| 2026-05-08 (n=1,072) | 7.53 / 7.84 | 9.90 / 10.38 | 12.93 / 13.59 | 26.30 / 27.50 |

**Narrower in 8 of 8 horizon-anchor cells, by 4.0-6.2%** — the same sign as the
offline sweep but SMALLER than its 6.5-15.4%, because the offline figure is a
matched-coverage width while the replay serves the q_hat it was calibrated with
and lets coverage move. Coverage duly falls 0.8-1.9pp everywhere, and **mean
|coverage - 80| improves at both anchors** (3.94 -> 3.70 and 5.75 -> 5.10), so
the band is narrower AND better calibrated on average.

It is not a free win: coverage falls everywhere, which helps where the band
over-covers and hurts where it under-covers.

### Extended to seven anchors

The two-anchor read was the weakest part of this change, so it was widened to
**7 paired anchors** spanning 2026-03-05..2026-06-08 (23 horizon-anchor cells;
anchors chosen so no horizon's outcome lands on the 03-22 / 07-09 / 07-10 label
seams, and h=30 resolves for only 4 of them).

| h | anchors | mean ratio | worst | narrower | mean abs(cov-80) K=320 | K=20 | calib wins |
|---|---|---|---|---|---|---|---|
| 3 | 7 | 0.9626 | 0.9921 | 7/7 | 5.25 | 5.73 | 5/7 |
| 7 | 6 | 0.9578 | 0.9820 | 6/6 | 6.47 | 7.49 | 5/6 |
| 14 | 6 | 0.9552 | 0.9862 | 6/6 | 4.31 | 4.83 | 4/6 |
| 30 | 4 | 0.9469 | 0.9564 | 4/4 | 5.42 | 6.31 | 4/4 |

**Narrower in 23 of 23 cells** -- the worst single cell is still 0.9921 -- and
**mean calibration error improves at EVERY horizon**, with 18 of 23 cells
individually better.

**The per-horizon question is now settled: don't.** The five calibration losses
scatter across h=3 (2), h=7 (1) and h=14 (2), with none at h=30, and every one
of them is a date where K=20 already sat at or below nominal. That is a generic
property of narrowing a band that is already too narrow on that date -- not a
horizon-stable effect -- so a per-horizon K would be fitting date noise.

### The cutover is self-describing

`SHRINK_K_SERVING_START` still ships `None` -- the chain is paused, so no prod
forecast has served K=320 and there is no date to set. But identifying that
deploy no longer depends on remembering which run retrained: **`meta.json` now
records `climatology_shrink_k`**, and `climatology_geometry_matches_code()`
returns False whenever the loaded cache predates a K change. A predict-only run
on a K=20 cache serves the old geometry while looking identical in every other
respect, which is exactly the mistake this prevents.

It is deliberately NOT folded into `MODEL_ARTIFACT_VERSION`: an artifact written
before the key exists is otherwise byte-identical, so bumping would force every
checkout into a needless retrain. Verified on the two scratch artifacts -- the
untagged one falls back to the code constant and is not flagged, one tagged
`climatology_shrink_k: 20` reports a mismatch.

**Still not a prod A/B.** Two anchors is two dates, the artifacts are locally
trained on an archive that lags the durable one, and a replay measures serving
arithmetic on historical features rather than a historical run. Re-gate on the
served panel once the chain resumes; `CLIMATOLOGY_REACTIVE` looked good offline
and died on prod A/B.

## Changed

- `models/forecaster.py` — `CLIMATOLOGY_SHRINK_K` 20 -> 320.
- `models/served_recalibration.py` — `SHRINK_K_SERVING_START` (None) into `_geometry_floor()`.
- `models/forecaster.py` — persists `climatology_shrink_k` in `meta.json`, reads it back at load,
  and adds `climatology_shrink_k_served()` / `climatology_geometry_matches_code()`.
- `scripts/magnitude_vs_climatology.py` — new gate: `--folds`, `--k-sweep`, `--k-grid`,
  `--served-cohort`, `--objective`, and the sigma / climatology / pooled / booster arms.
- `scripts/replay_serving.py` — honours `FORECAST_MODEL_DIR`, so an arm and its
  control can be replayed at one anchor without swapping `models/saved_models/`.
- `tests/test_climatology_shrink_level.py`, `tests/test_magnitude_gate_folds.py`,
  `tests/test_magnitude_gate_shrink.py`, `tests/test_served_recalibration_wiring.py`,
  `tests/test_replay_serving_model_dir.py`, `tests/test_climatology_shrink_k_persisted.py`.
