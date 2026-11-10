# Conformal PID at h=3: the preregistered read is void (placebo clears the bar)

**Date:** 2026-11-09
**Prereg:** `research/2026-09-23-conformal-pid-served-h3-preregistration.md` (A7: panel read from the
ops partitions; `--prove-equal` passed, 226,976 rows identical from both stores)
**Script:** `backend/scripts/measure_conformal_pid.py`; log `direction-reads/2026-11-09-pid-read.log`
(outside the repo)
**Status:** final. One read; rerun only under a new window declared before its first date resolves.

## Read

Window 2026-09-07..2026-10-18, 41 scoreable dates (≥ 25), served h=3, established items.

| Arm | primary (pp) | P10 | M1 | mean m | clamps |
|---|---|---|---|---|---|
| F1 (m = 1.0) | 13.50 | 91.43 | 93.51 | 1.0000 | 0 |
| B (incumbent batch refit) | 7.10 | 73.27 | 84.30 | 0.6637 | 0 |
| Q (quantile tracker) | 6.02 | 74.50 | 82.69 | 0.6221 | 12 |
| QI (candidate, P + I) | 6.59 | 72.67 | 82.57 | 0.6228 | 15 |
| P_QI (placebo, permuted errors) | 5.04 | 76.16 | 82.95 | 0.6313 | 7 |

QI − B primary: −0.51pp, 95% CI [−1.62, +0.56]. Criteria: (1) holds, (2a) margin fails,
(2b) CI includes 0, (2c) QI is not below the placebo, (3) P10 fails, (4) width holds, (5) deciles
hold. QI and Q both FAIL.

**Verdict: VOID.** The placebo clears (2a) against B (5.04 vs 7.10, −2.06pp), which the prereg
names as a void condition. QI would fail on its own anyway (−0.51pp against a 1.0pp margin, CI
spanning zero), so the void changes no decision.

## Action (prereg outcome table, "Void / unresolved")

- Recorded; `experiment_log.csv` row added (`void`).
- PID is not wired in. The batch refit B stays.
- A rerun needs a new window declared before its first date resolves.
- A permuted-error placebo beating the incumbent by 2pp means most of the gap to B is the level
  (clamped, lower multiplier), not tracking. Worth remembering if online conformal comes back.
