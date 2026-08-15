# Volume features remeasured — real ~1–2pp directional signal, conditional on repairing the feed

**Date:** 2026-08-15
**Change:** none shipped. `scripts/ab_test_volume_features.py` re-run on the
post-2026-08-08 embargo and post-2026-08-13 trainer fixes. The shelving from
`2026-08-06-volume-features-shelved.md` stands (the live feed is still dead);
this note records that the *counterfactual* — "if the feed were repaired, would
volume earn its place?" — now answers **yes**, and quantifies it.

Motivation: `backend/AGENTS.md` flags that all thirteen `ab_test_*` verdicts
predate the 2026-08-08 statistics fix and the 2026-08-13 trainer fix, so each
needs a fresh run before it can be cited. This is that run for volume.

## What was found

200-item universe, frame cut at `VOLUME_LIVE_THROUGH = 2026-04-30` (past which
the archive's `volume` is identically 0 — see the shelving note), embargoed
walk-forward folds (`horizon + 13`), 25–26 folds, ~94–98K paired rows. Three
arms: **baseline** (volume columns dropped, 31 features), **treatment** (volume
in, 42 features — 11 of the 13 survive the >0.95 prune), **placebo** (the volume
columns column-shuffled, the capacity-inflation guard).

Directional accuracy, treatment vs baseline (fold-clustered paired interval),
and the placebo:

| Horizon | Baseline | Treatment | Treatment − baseline (paired) | Placebo − baseline |
|---|---|---|---|---|
| 3d  | 55.27% | 56.75% | **+1.503pp** [+0.631, +2.499] | null −0.103 [−0.307, +0.096] |
| 7d  | 54.55% | 56.32% | **+1.879pp** [+0.031, +3.952] | null −0.008 [−0.194, +0.194] |
| 14d | 57.02% | 58.43% | **+1.442pp** [+0.277, +2.538] | null +0.186 [−0.082, +0.440] |
| 30d | 51.61% | 52.55% | **+0.959pp** [+0.025, +2.012] | null −0.165 [−0.338, +0.009] |

**It passes the harness's ship criteria at all four horizons:** treatment beats
baseline with a CI that excludes zero (4/4), the shuffled placebo is null (4/4,
every interval straddles zero), and no horizon regresses. Treatment-vs-placebo
margins agree with treatment-vs-baseline (+1.61 / +1.78 / +1.22 / +1.10pp on the
STR≥$1 cohort), so the gain is signal, not the two extra columns' capacity.

This is the first evidence in the recent thread that a **new data source** adds
idiosyncratic directional signal. It qualifies — does not overturn —
`2026-08-15-cs2-oracle-is-a-range-forecaster.md`: that note's "no idiosyncratic
per-item signal once the market factor is removed" was measured on the
price-technicals set, which does not include volume.

## What this does NOT license

- **Un-shelving volume against the current feed.** The live column is 0 from
  2026-07 (a train/serve gap), so the frame is cut at 2026-04-30 and every late
  fold would otherwise score a dead column. **There is no collector to "repair":**
  the csgotrader.app dumps carry no volume field, `prices.volume` was only ever
  populated by one-shot backfills (last expired 2026-04-15), and the signal here
  was trained on the `volume-panel.parquet` sidecar — `steam_volume` from a
  **static third-party Kaggle dataset** (`ingest_volume_panel.py`, reaching
  2026-06-15), not a live series. See
  `2026-08-08-volume-null-not-zero-and-skinport-sales.md`. Capturing the gain
  therefore needs a *live* volume source, and the only wired one
  (`collectors/sales_volume.py`, Skinport `volume-YYYY-MM.parquet`) is a
  **different quantity** — one cash venue, trailing/cumulative windows, not the
  Steam sale count measured here. So this result does NOT transfer to production
  until the A/B is re-run on that live series; a positive here with a null there
  would be the usual CV-positive / serving-negative pattern.
- **A production DA number.** This harness uses a 200-item universe and its own
  mean loader, not the ≥$1 voted-median cohort production trains on
  (`labels-and-embargo`), so read the *delta*, never the 51–58% absolute levels.
- **Treating the win as large.** ~1–2pp, and the 7d and 30d lower bounds are
  razor-thin (+0.031, +0.025).

## Reproduce

```
cd backend && venv/bin/python scripts/ab_test_volume_features.py --out <path>.json
```

Run local 2026-08-15, exit 0. Full log and JSON captured off-repo.
