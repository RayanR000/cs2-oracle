# musickit h=3 under-coverage is the width tilt, not a family effect

**Date:** 2026-10-04
**Script:** `backend/scripts/archive/conditional_coverage_by_stratum.py` (unchanged), plus the
width-matched breakdown under Reproduce
**Status:** diagnostic only. No modelling or serving change. Closes the musickit watch in
next-steps item 10.

`2026-09-09-conditional-coverage-by-stratum.md` flagged `musickit` at h=3: 84.7% (n = 484,
CI [81.4, 87.4]) against pistol's [88.7, 91.3]. It asked for a re-read at 20 dates and did not
build an arm. This is that re-read.

## Panel

Data-repo `forecast_outcomes.parquet` as archived on 2026-10-04: served cohort (>=$1),
`excluded_forecast_date` applied, rebased to the model's own quote as in the 09-09 note. That
gives 124,966 rows over 47 forecast dates. Musickit has **38 dates** at h=3 and 40 at h=7, past
`MIN_HEADLINE_DATES = 20`. The script still prints "panel immature" on those horizons because
its gate takes the minimum over all strata, and `unknown` (1 date) and `tool` (13) sit below 20.

## Family read

| h=3 | 09-09 (14 dates) | 10-04 (38 dates) |
|---|---|---|
| musickit | 84.7% [81.4, 87.4] | 83.3% [80.5, 86.0] |
| pistol | ~90% [88.7, 91.3] | 87.0% [85.0, 88.9] |

The gap persists at about 3.7pp against pistol. At h=7 musickit reads 89.9% [87.7, 92.0],
within the bulk.

## Width explains it

Musickit bands are narrow: median h=3 half-width 4.95% against ~6.3% for the gun families.
**52%** of musickit h=3 rows (689 / 1,314) fall in the narrow within-horizon tertile, against
the 33% that tertile holds by construction. The narrow tertile under-covers for everyone
(80.7%). That is the known sigma tilt, `2026-08-12-the-band-is-tilted-in-sigma.md`.

Width-matched coverage, musickit vs every other family (90% date-bootstrap CI):

| h | tertile | musickit | other |
|---|---|---|---|
| 3 | narrow | 77.9% [74.8, 81.4], n 689 | 80.9% [79.0, 83.0] |
| 3 | mid | 90.7% [87.6, 93.3], n 492 | 91.0% [89.7, 92.1] |
| 3 | wide | 84.2% [77.4, 89.2], n 133 | 94.3% [93.1, 95.4] |
| 7 | narrow | 88.8% [85.9, 91.7], n 893 | 89.5% [87.7, 91.3] |
| 7 | mid | 91.3% [89.3, 93.2], n 345 | 92.1% [91.0, 93.1] |
| 7 | wide | 93.0% [86.7, 96.5], n 142 | 96.3% [95.6, 97.1] |

- In the narrow and mid tertiles the CIs overlap at both horizons.
- If musickit had the other families' per-tertile coverage on its own tertile mix, it would
  read ~86.0%, so the mix accounts for about half the gap.
- The remainder sits in the 133-row wide cell. There, musickit's half-width is still narrower
  (8.67% vs 10.27%), so it is the same tilt within the tertile.

## Verdict

**No family arm. Watch closed.** The width tilt is the cause, and conditioning width on sigma
is on the do-not-run list (`specs/2026-09-14-wait-window-workplan.md`). Nothing new to build.

## Caveat

The panel pools August and September across the h=3 feedback-factor change (0.5385, served
since the 09-28 retrain), which narrowed bands. That is acceptable for a within-panel family
comparison but not quotable as a level. h=3 coverage among the other families fell from 91.3%
(August) to 87.0% (September). Musickit's gap to them narrowed from 7.3pp to 4.1pp.

## Reproduce

Extract `price-archive/ops/forecast_outcomes.parquet` and `price-archive/item-metadata.parquet`
from the data repo's `origin/main` (`git show origin/main:<path>`) into `<dir>/ops/` and
`<dir>/`, then from `backend/`:

```
venv/bin/python scripts/archive/conditional_coverage_by_stratum.py --archive-dir <dir>
```

Width-matched breakdown:

```python
import sys; from pathlib import Path
sys.path[:0] = ["scripts/archive", "."]
import numpy as np, conditional_coverage_by_stratum as c
df = c.prepare(c._read_parquet(Path("<dir>")))
d = df[df.horizon_days.isin([3, 7])].assign(
    mk=lambda x: np.where(x.family == "musickit", "musickit", "other"))
rng = np.random.default_rng(0)
for (h, w, m), g in d.groupby(["horizon_days", "width", "mk"]):
    per = g.groupby("forecast_date").covered.agg(["sum", "count"])
    i = rng.integers(0, len(per), (1000, len(per)))
    lo, hi = np.percentile(per["sum"].values[i].sum(1) / per["count"].values[i].sum(1), [5, 95])
    print(h, w, m, len(g), round(g.covered.mean() * 100, 1), round(lo * 100, 1), round(hi * 100, 1))
```
