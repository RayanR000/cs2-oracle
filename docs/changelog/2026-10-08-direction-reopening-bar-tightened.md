# The direction reopening bar is tightened and its window fixed

**2026-10-08.** Docs only. `DIRECTION_DISCLOSED` stays `False`; no code or flag changed.

The 09-10 withdrawal (`2026-09-10-served-direction-withheld.md`) reopens direction on "a positive
PT at 20 or more dates". A read-only probe on 2026-10-08 showed why that is not enough:

- All-date PT has washed out from perverse to null (h=3 −0.58pp p=0.29; h=7 −0.03pp; h=14 +0.06pp;
  h=30 −0.13pp).
- Since the 09-20 retrain PT is positive at h=3 (+1.41pp, t=2.67, 13 dates) and h=7 (+0.88pp,
  t=2.20, 9 dates), but h=3 DA is 36.0% against an always-down rate of 47.9%. PT measures
  information against the model's own marginals, so a PT-positive call can still lose to the
  runnable baseline.

`research/2026-10-08-direction-reopening-preregistration.md` replaces the rule. Per primary
horizon (h=3, h=7) on the first 20 usable forecast dates **≥ 2026-10-05** (none of which the
probe saw): PT t ≥ 3.0, DA ≥ `realised_down_rate`, and a drop-one-date guard. Expected reads
are about 10-29 (h=3) and 11-02 (h=7). Calibrated power at the observed effect is about 60%, so a
miss is probably "unresolved" and not a refutation.

The 09-10 note stays as written; its "Reopening" section is superseded by the preregistration.
