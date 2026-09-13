# 30d anomaly head: isotonic calibrates it, and it still does not earn disclosure (2026-09-13)

Closes the STILL OPEN item from `2026-09-09-anomaly-head-beats-its-null.md`: at h=30 the
anomaly head RANKS (held-out AUC +0.129 over a featureless null) but its held-out LOG LOSS
ties the pooled constant, so `ANOMALY_SERVED_HORIZONS = (3, 7, 14)` withholds `anomaly_p`
there. That is the shape of defect isotonic calibration fixes, and the repo already ships
PAV machinery for the exceedance head. This asks whether the layer earns 30d its disclosure.

Instrument: `backend/scripts/anomaly_calibration_ab.py` (+ `tests/test_anomaly_calibration_ab.py`,
7 tests). 25 paired folds, served >=$1, strictly-prior (clean) label, base rate 0.1764 over
1,321,684 rows. Calibration slice is the last 25% of each train fold's DATES, with
`_purge_overlapping_train_rows` applying the H+13 embargo at the inner boundary.

## Arm scores (held-out, 25 folds)

| arm | AUC | log loss |
|---|---|---|
| gbm | 0.6294 | 0.49378 |
| gbm_inner | 0.6027 | 0.53394 |
| **gbm_cal** | 0.6027 | **0.48648** |
| item_rate | 0.5000 | 0.48844 |
| global_rate | 0.5000 | 0.48844 |

## The two reads that matter

**The calibration layer WORKS.** `gbm_cal` vs `gbm_inner` (the honest control — same training
rows, so the only difference is the map): held-out log loss **-0.04746 [-0.08703, -0.00788],
SIG**, with AUC delta **exactly 0.00000** at both ends of the CI. That zero is the signature
of a correct isotonic fit: the map is monotone, so it cannot reorder, and every bit of the
gain is calibration rather than discrimination.

**The shipping test FAILS.** `gbm_cal` vs `global_rate` on held-out log loss:
**-0.00196 [-0.01815, +0.01423], ns.** The calibrated head is numerically ahead of the
constant (0.48648 vs 0.48844) and orders items far better (AUC +0.1027 [+0.0642, +0.1412],
SIG) — but as a *probability* it is a dead heat with a number that ignores every feature.
The pre-registered bar was a negative log-loss delta with a CI clear of zero. It is not.

## Verdict

**30d stays withheld. `ANOMALY_SERVED_HORIZONS = (3, 7, 14)` is confirmed correct** — now
against a calibrated head, not just a raw one. A served `anomaly_p` is read as a probability,
and at 30d this one still cannot beat a constant at being a probability.

Note the raw `gbm` arm's held-out log loss (0.49378) is WORSE than the constant while its
calibrated sibling is better: uncalibrated 30d output is actively misleading, which is why
disclosing it raw was never the fallback.

Do not re-run this as a band-width or ranking experiment. The 30d head's ranking was never in
doubt (AUC +0.10 to +0.13 across three independent runs now); the question was calibration,
and calibration was achieved without changing the answer.

## Sanity

`item_rate` collapses onto `global_rate` exactly (0.48844 / AUC 0.5000) on held-out items —
they are unseen in train, so the per-item rate correctly falls back to pooled. Third
reproduction of pooled-beats-per-item.

## Reproduce

```
venv/bin/python -m scripts.anomaly_calibration_ab --horizon 30 --clean-label \
  --metadata-parquet ../price-archive/item-metadata-bymykel.parquet \
  --frame-cache /tmp/anom_frame.parquet --out /tmp/anom_cal_h30.json
```

`--metadata-parquet` is REQUIRED (the docstring usage line omits it; the frame build exits 1
without it).
