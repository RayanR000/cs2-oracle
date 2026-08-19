# DIRECTION_UPWEIGHT set to 1.0 (neutral) — the durable centre fix

**2026-08-19.** `DIRECTION_UPWEIGHT` drops from `1.5` to `1.0`, the neutral value, and becomes an
env-overridable constant (`float(os.environ.get("DIRECTION_UPWEIGHT", "1.0"))`) so the old control
can be reproduced for a paired read. This is the upstream half of the signed-conformal band work
(`2026-08-19-signed-conformal-band.md`): the signed pair absorbs a biased q50 centre post-hoc; this
removes the bias at its source.

## Why

At `1.5`, positive-return samples carried 1.5× the gradient weight in `_compute_sample_weights`.
The intent was to counter a conservative "up" bias, but the served **sign** comes from the
directional classifier, not the q50 — so the upweight did not correct direction. What it did was
pull the q50 `|mid|` up, seating the served median near the 58–71st percentile of the outcome
(`P(actual < mid)` = 0.60 / 0.65 / 0.71 at 3/7/14d). That off-centre median is the biased centre
the signed band was measured to absorb — 11–13% of band width comes from recentring alone.

Setting the upweight to `1.0` centres the q50 directly, so the band's asymmetry reflects the
return distribution's real skew rather than a fitting artifact.

## What changed

- `models/forecaster.py:136` — `DIRECTION_UPWEIGHT = float(os.environ.get("DIRECTION_UPWEIGHT", "1.0"))`.
  Read at import (training reads it; the classifier is untouched). The `1.5` control is reproducible
  with `DIRECTION_UPWEIGHT=1.5`.
- `_calibrate_conformal` logs a per-horizon centre-bias line
  (`P(actual<mid)`, `q_lo`, `q_hi`) from the OOF residuals, so the centre is auditable in the
  training log without a separate harness.

## Measured

Local paired retrain, production config (1.2M feature rows, ≥ $1 floor, no subsample), each arm's
own OOF calibration records. `P(actual<mid)`, target 0.50:

| h | 1.5 (control) | 1.0 (arm) |
|---|---|---|
| 3 | 0.614 | 0.532 |
| 7 | 0.65\* | 0.533 |
| 14 | 0.71\* | 0.550 |
| 30 | ~0.60\* | 0.468 |

\* 7/14/30 at 1.5 are the prior recorded baseline; the control retrain was stopped after 3d (0.614)
reproduced it, its only remaining value being a stale-archive width read.

The q50 now centres at the median. Signed quantiles tighten toward symmetric at the short horizons
(3d `−94.7 / +90.2`, 7d `−125.3 / +114.3`) while 14d/30d retain genuine right-skew
(`−136.9 / +153.0`, `−208.2 / +280.4`) — the return asymmetry the signed band is meant to carry.

## What this does NOT settle

Served coverage and width. This is measured on OOF residuals, robust to the stale local archive
(`local-db-is-a-fixture`), but the served-basis width read needs a prod A/B — a background clock on
the 14-date panel. Ship on the next scheduled retrain and let it accumulate.
