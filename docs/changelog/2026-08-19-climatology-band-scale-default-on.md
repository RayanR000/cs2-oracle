# Climatology band scale defaulted on, with a served-geometry guard

Cuts the climatology scale over from gated-off to the production default. It is the one band
denominator that wins OUT of sample (variance reduction, not signal extraction): 43-47% narrower
than the sigma band at matched 0.80 coverage on 285 test dates, DECISIVE on the out-of-sample
served-cohort gate (`docs/changelog/2026-08-19-climatology-band-scale-implemented.md`,
`docs/research/2026-08-19-climatology-vs-gbm-band.md`). The signed-band recentring win shipped
the same day (`2026-08-19-signed-conformal-band.md`); this is the next-largest width change and it
does not compete with it — the signed band is a two-quantile centring change, not a scale
denominator, so the two compose.

## What

- `climatology_scale_enabled()` (`models/forecaster.py`) now returns
  `os.environ.get("CLIMATOLOGY_SCALE", "1") != "0"` — **on by default, only an explicit `"0"`
  disables** (an unparseable value keeps it on, deliberately, mirroring the training price floor).
  Serving is unchanged: `_climatology_scale_served` follows `meta.json`'s `climatology_scale`, so
  **the cutover is atomic at the next `full` retrain**, not the moment this flag flips — loaded
  sigma-scaled artifacts keep serving until then.

## The served-geometry guard (`models/served_recalibration.py`)

The served-coverage feedback factor reads the STORED band shape (`r = (actual - mid)/half`) and
applies one multiplier to the next artifact's `q_hat`. Sigma-scaled rows over-cover; climatology
rows are near-calibrated. Pooling the two across the cutover computes a blend optimal for neither
and over-shrinks the climatology band — the same geometry-mixing hazard the signed band already
guards with `SIGNED_BAND_SERVING_START`.

- `CLIMATOLOGY_SERVING_START: Optional[str] = None` — the date the climatology band first serves
  prod; left `None` until that retrain deploys.
- `_geometry_floor()` — the LATEST of the configured cutovers (signed band, climatology). The
  panel floors to it, so the factor is fit only on rows served under the shape it is applied to.
- `served_coverage_factors` defaults `since` to `_geometry_floor()` (via an `_UNSET` sentinel, so
  an explicit `since=None` still means dormant). The sole caller (`forecaster.py:5501`) passes no
  `since`, so it picks up the floor automatically.

**Behavior today:** `CLIMATOLOGY_SERVING_START` is `None`, so the floor stays `2026-08-19` and the
feedback is unchanged. When the first `full` retrain carrying `climatology_scale: true` deploys,
**set `CLIMATOLOGY_SERVING_START` to that serve date** — the feedback then restarts its
`MIN_FORECAST_DATES` count on climatology-only rows (the correct, temporary dormancy cost).

## Not done (intentional)

- No `MODEL_ARTIFACT_VERSION` bump: the on-disk format is unchanged; the served band changes only
  when a retrain re-stamps `climatology_scale: true`.
- `CLIMATOLOGY_SERVING_START` is left `None` — it is an operational value set at deploy, not merge.

## Tests

- `tests/test_climatology_scale_default.py` (new) — default-on, only `"0"` disables, the
  `_calibrate_conformal` mutual-exclusion predicate is false under production defaults, and the
  signed-band symbols are absent from that guard block.
- `tests/test_served_recalibration_wiring.py` — floor takes the later cutover; a climatology start
  before the signed band (or unset) leaves the signed-band floor; floor is `None` only when both
  unset; `served_coverage_factors` with no `since` reads the panel at the combined floor.
- `tests/test_climatology_scale.py` — the gating test now disables via `"0"` (deleting the env var
  is now on). 22 climatology/signed-band + 21 served-recalibration tests pass.
