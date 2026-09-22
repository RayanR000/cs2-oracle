# Time-adaptive conformal (ACI) fails the preregistered bar — h=3 is a real near-pass

**Date:** 2026-09-22
**Prereg:** `docs/research/2026-09-22-adaptive-conformal-preregistration.md` (committed `3ec74e9`;
γ grid amended in `1823c1a`, both before this run)
**Instrument:** `backend/scripts/archive/measure_aci.py` (+ `tests/test_measure_aci.py`, 9 tests)
**Status:** REFUTED per the prereg. ACI is closed for the band. No serving change.

## Result (scoring block, ~700 dates per horizon, panel 2022-09-22 → 2026-09-08, ≥$1)

| h | arm | M1 | LM date err | P10 date cov | LM width vs S0 | min σ-decile |
|---|---|---|---|---|---|---|
| 3 | S0 pooled | 80.5% | 6.34pp | 69.1% | 1.000 | 62% |
| 3 | R_roll | 80.1% | 5.24pp | 71.2% | 1.036 | 64% |
| 3 | **A_aci (γ=0.2)** | 80.1% | **4.12pp** | **74.2%** | 1.045 | 65% |
| 3 | P_aci placebo | 51.1% | 10.60pp | 61.1% | 1.325 | 67% |
| 7 | S0 pooled | 79.9% | 5.99pp | 70.1% | 1.000 | 63% |
| 7 | A_aci (γ=0.005) | 80.3% | 6.10pp | 69.9% | 0.999 | 63% |
| 14 | S0 pooled (report-only) | 77.8% | 5.93pp | 70.6% | 1.000 | 60% |
| 14 | A_aci (γ=0.01) | 81.0% | 6.21pp | 70.3% | 1.036 | 60% |

Full CSV: `docs/research/data/2026-09-22-measure-aci.csv`.

## Verdict against the bar

- **h=3:** passes (1)–(4) — date error −35%, beats both R_roll and the placebo, P10 +5.1pp, width
  +4.5% (cap 5%). **Fails (5)**: min σ-decile 65% < 70%.
- **h=7:** fails (2) and (3). Tuning picked the smallest γ; adapting does not help.
- The prereg requires PASS at both h=3 **and** h=7, so the verdict is **closed** however (5) is
  read. No void condition fired (S0 date err ≥ 3pp; placebo did not clear (2)).

## What is worth keeping

1. **At h=3 the gain is real timing information.** The placebo — same γ, same err values, shuffled
   order — is *worse* than S0 (10.6pp), so the improvement is not machinery.
2. **The benefit dies with feedback lag.** At h=3 the update reads 3-day-old misses; at h=7/14 the
   tuning block rejects adaptation entirely. This matches the deep review's caution that feedback
   lags at longer horizons.
3. **Prereg design flaw, recorded rather than fixed post hoc:** guard (5) was copied from the WACI
   read without checking the baseline, and S0 itself is at 62%, so no arm could pass it on this
   panel. ACI *raises* the worst decile (62 → 65), the opposite of the trade the guard was built to
   catch. Changing the guard now would be a post-hoc rescue and is not done.

## If anyone reopens this

Only as a **new** preregistration, h=3 only, with a guard relative to S0, scored on data this run
has not seen — i.e. the served panel going forward (≥ 20 post-2026-09-06 dates at h=3, ~09-29),
not another offline replay of this archive.

## Reproduce

```
cd backend && venv/bin/python -m scripts.archive.measure_aci --horizons 3,7,14 --out /tmp/measure_aci.csv
```
Runtime 21s, 2.4 GB peak. Local panel `voted_2905e0703fc3cd85dc4b1172.parquet` (1,437 dates).
