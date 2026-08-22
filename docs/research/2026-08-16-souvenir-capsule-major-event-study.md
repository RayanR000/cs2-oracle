# Souvenir / capsule × Major event study (spike)

> **Status (as of 2026-08-21): the underpowered verdict stands, and the trade it pointed at is
> DEAD.** This doc's one suggestive result is a **6–12 month** post-Major appreciation, i.e. a
> long-horizon trade. That line was tested directly on 2026-08-20 and closed:
> at h=90/180 the realised move magnitude *does* clear the friction bar, but **selection is null**
> (AUC 0.507/0.526/0.532 across 3 out-of-sample windows, top-decile lift ≤1.0×) and the
> unconditional bet **lost money in every window**. There is no tradeable edge at any horizon;
> the product is range-only and the trade hunt is closed. Do not re-scope this as a strategy.


**Date:** 2026-08-16
**Type:** Spike (pre-registered probe). Deliverable is a finding + go/no-go, not shipped code.
**Question:** Around CS Majors, do capsule + souvenir-package items earn abnormal returns — net of the 15% Steam fee — at announcement, tournament start, or tournament end?

## Verdict

**Suggestive but not statistically established.** There is *no* short-horizon event pump.
There is a large, mostly-positive **supply-cycle pattern**: Major capsule/souvenir-package
prices flood *down* during/after the event, bottom ~90 days out, then appreciate above
comparable ordinary skins over the following 6–12 months. But when aggregated to the level
of *independent events* (the correct unit), the effect is **underpowered and insignificant**
(see "Per-Major aggregation" below): 4 of 7 Majors positive at +365d (t=1.78, p=0.13), 5 of
6 at +455d (t=2.09, p=0.09), and the mean is driven heavily by the 2021–22 (peak
bull-market) Majors. Mechanism, if real, is finite/retired-supply container appreciation
(driver #16) timed by the Major flood — not esports hype. Treat the +30–48pp item-level
excess as an **inflated upper bound**: within a Major all capsules move together, so 60
items are ~1 bet, not 60.

## Method

- **Cohort:** 320 capsule + souvenir-package + autograph-capsule items, matched 1:1 to a
  Major by the location+year token embedded in the slug (356 candidates; 36 non-Major
  community/licensed capsules correctly dropped). Souvenir *weapon skins* excluded — their
  slug carries no unambiguous tournament tag.
- **Calendar:** 25 Majors (2013→2026), start/end dates from Liquipedia (MediaWiki API,
  rendered `Majors` page). See `scratchpad majors_cal.py`.
- **Prices:** price archive collapsed to one price per (item, day) = median `mean_price`
  across sources excluding buff163_buy and Steam trailing-window means. Daily log returns.
- **Abnormal return:** item log-return minus the broad market factor (cross-sectional
  median daily log-return across the whole archive).
- **Anchors:** release (first obs), Major start, Major end. Windows measured post-anchor.
- **Placebo:** identical calendar windows applied to 800 random ordinary weapon skins with
  a *random* 2021+ Major-date assignment. Two-sample t on capsule − control.

## Results

### Post-Major decline (not tradeable — can't short on Steam)
Market-demeaned CAR from Major **start**: −21% at 30d, **−54% at 90d (t=−10.3, ~85% of
items negative)**. Clean supply-flood signal, but it is a *decline* — not a long trade,
and the fee makes shorting moot.

### Recovery leg (the tradeable part) — buy the flooded dip, hold
Capsule market-demeaned abnormal return vs placebo control:

| window (from Major start) | capsule AR | control AR | excess | t(diff) | net-of-fee excess |
|---|---|---|---|---|---|
| buy+90 / sell+270 | +30.1% | +1.6% | **+28.4pp** | +4.78 | +24.2pp |
| buy+90 / sell+365 | +48.0% | +18.2% | **+29.9pp** | +3.32 | +25.4pp |
| buy+90 / sell+455 | +60.6% | +12.3% | **+48.2pp** | +3.37 | +41.0pp |
| buy+120 / sell+455 | +50.8% | +8.7% | **+42.1pp** | +3.33 | +35.8pp |

The naive demeaned capsule number (+48%) overstates the edge — the placebo shows ~18pp of
it is just "hold any skin a year during the 2021–25 bull market" (imperfect demeaning).
**These item-level t≈3.3–4.8 values are themselves inflated** by within-Major correlation
and are superseded by the per-Major aggregation below.

### Per-Major aggregation (the honest test — independent events)
Each Major = one observation; capsule basket mean vs control basket anchored to the *same*
Major date; paired one-sample t across Majors.

| window | Majors | mean excess | median | % Majors positive | paired t | p | leave-one-out range |
|---|---|---|---|---|---|---|---|
| buy+90 / sell+365 | 7 | +42pp | +63pp | 57% (4/7) | +1.78 | 0.13 | +29 .. +53pp |
| buy+90 / sell+455 | 6 | +78pp | +68pp | 83% (5/6) | +2.09 | 0.09 | +52 .. +99pp |

Per-Major excess (buy+90/sell+455): Stockholm 2021 **+208pp**, Antwerp 2022 **+138pp**,
Shanghai 2024 +71pp, Rio 2022 +65pp, Copenhagen 2024 +14pp, Paris 2023 **−26pp**. The
mean leans on the two 2021–22 bull-market Majors; drop Stockholm and it roughly halves.
**Not significant at conventional thresholds; ~6–9 events cannot establish this.**

## Caveats (material)

1. **Slow, thin trade.** 6–15 month hold; capsules are low-liquidity, wide-spread. Paper
   edge ≠ capturable edge at size.
2. **Few independent events.** Item-level n=57–67, but only **~9 Majors** with full forward
   windows (2019/2021–2024). Events overlap in calendar time → effective independent n ≈ 9;
   the item-level t-stats overstate true significance.
3. **Coverage.** Pre-2019 Major capsules aren't priced in the archive; 2025–26 Majors are
   truncated by the forward window. Recovery cohort is effectively 2021–2024.
4. **Mechanism is supply, not hype.** This is retired/finite-supply appreciation (driver
   #16), not the esports-attention catalyst the original hypothesis imagined.
5. **Demeaning is imperfect** (control AR ≠ 0). Trust the difference, not the level.

## Go / no-go

**No-go as a shippable signal; park as a watch-item.** The point estimate is large and
positive in most Majors, but at the correct (per-event) unit it is not significant
(p=0.09–0.13) and leans on the 2021–22 bull-market Majors. Six to nine independent events
cannot carry a trade. Do **not** wire it into the model.

Cheap ways it could be revived (only worth it if the theme resurfaces):
- **More events** via the **iflow BUFF backfill** (already staged) — adds pre-2021 Major
  depth and a second venue, the only real path to statistical power here.
- **Better control** — the demean under-removes drift in the 2021–22 windows exactly where
  the excess concentrates; a beta-matched (not median) factor could erase much of it.
- If revived, treat as a **held-inventory tilt** (buy retired-container basket ~90d
  post-Major), never a daily-model feature, and gate on liquidity.

## Reproduction

Throwaway scripts in scratchpad `/private/tmp/.../souvtest/`: `majors_cal.py` (calendar),
`build_panel.py` (market factor + capsule panel), `event_study.py` (anchors/CARs),
`placebo.py` / `diff.py` (control + two-sample). Nothing written to the pipeline.
