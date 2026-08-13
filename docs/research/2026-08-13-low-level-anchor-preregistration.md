# Pre-registration: does the date-level rescaling RAISE the band where it must?

**Date:** 2026-08-13, written and committed **before any coverage number was computed.**
**Instrument:** `backend/scripts/attribute_band_level.py --low-legs` (offline, ~2 min, writes
nothing). Selection ran first, through `--select-low`, which sees `L[t]`, the two audits and row
counts and **never a coverage rate**.
**Opened by:** `docs/changelog/2026-08-13-the-rescaling-is-a-level-fix-not-a-spread-fix.md`, open
question 2 — *"every readable anchor is a high-`sigma` date, so the arm has been observed only where
it lowers coverage; its behaviour on a low-vol date, where it must raise the band, is unmeasured."*

## The one question

`sigma_tilde = sigma / L[t] ** gamma` with `gamma ≈ 0.70` lands production's over-covering band on
80.67% / 82.12% — measured on six anchors that are **all** above the panel's volatility norm
(1.079–2.255×). Dividing by `L ** gamma` **widens** the band on a date whose trailing level is below
the calibration average. Nobody has measured that direction. If the arm only ever narrows, it is a
constant shrink wearing a date-level mechanism's clothes, and shipping it would move calm dates from
whatever they cover now to something worse.

**Fitted `b` = 0.246–0.315 says forward dispersion repays only a quarter to a third of a trailing-
volatility swing. Applied downward that predicts the control UNDER-covers on calm dates** — the same
arithmetic that makes it over-cover on volatile ones, with the sign flipped.

## What selection found before any read, and it bounds the whole question

Of 724 panel dates, 639 pass **both** audits at all four horizons. Of the 183 dates in 2026 — the
served collection regime — **exactly one** is in the panel's bottom quartile of `L[t]` and passes:
`2026-02-19`. 2026's calm quarter (Q1, median `sigma` 0.0655) sits at the panel median (0.0667), not
below it.

🔑 **So the direction where the arm must raise the band is not measurable inside the served regime.**
That is not a choice made here; it is what the archive holds. Two legs follow from it, and the split
is declared now so neither can be promoted over the other afterwards.

**Leg B — the mechanism test, cross-regime.** Six anchors, lowest eligible `L[t]` first, ≥30 days
apart so no two outcome windows overlap, ≥300 rows, at least 60 days after the panel start:

| anchor | `L[t]` | vs panel median | rows |
|---|---:|---:|---:|
| 2024-09-21 | 0.0526 | 0.787× | 625 |
| 2024-11-04 | 0.0575 | 0.861× | 699 |
| 2024-12-04 | 0.0574 | 0.859× | 706 |
| 2025-01-03 | 0.0454 | **0.679×** | 710 |
| 2025-04-01 | 0.0581 | 0.870× | 792 |
| 2025-07-12 | 0.0549 | 0.822× | 852 |

⚠️ All six are pre-2026, where the archive carries `source IS NULL` — **a different collection
regime from the one production serves.** Leg B is therefore a test of the *mechanism*, not of a ship.

**Leg A — in-regime, and it carries NO BAR.** `2026-02-19` (L = 0.0582, 0.871×, 745 rows). One
anchor is a replication, not evidence. It is reported because the in-regime number is the one a
reader will want, and it is barred from deciding anything **because that is fixed here rather than
after its value is known.**

**The warm-up exclusion, recorded because it changed the set.** The first selection pass returned
`2024-07-09` as the lowest `L[t]` in the panel by a factor of two. It has **100% of its rows pinned
at the sigma clip floor**: `sigma` is `price_std_60d / price` and the panel's first 60 days hold a
window that is not yet 60 days long. The rule added is the feature's own window
(`SIGMA_TRAILING_WINDOW_DAYS = 60`), pinned by a test against the feature set. No coverage had been
computed when this changed.

## The arm

Unchanged from `2026-08-13-date-level-sigma-rescaling-preregistration.md`: `gamma = 1 − b` refit per
horizon on the whole panel by `fit_level_elasticity`, bootstrap over **dates**. It is deterministic
and was published as **0.700 / 0.685 / 0.708 / 0.754**; a departure from those is a void condition,
not a result.

## Bars

**h=14 is excluded from every bar count.** The prior read's leg (V) measured the panel against the
served control at **4.53pp MAE** there, above its own 3pp bar, and that document's rule is that a
panel which cannot reproduce the control cannot referee the arm. 14d is reported and never counted.
**So "≥3 of 4" throughout that document becomes ≥2 of the 3 readable horizons here.**

**(L1) Direction — primary.** Row-weighted pooled coverage over leg B must move **UP** under the arm
at **≥2 of 3** readable horizons. Down at 2 or more refutes the date-level mechanism outright: an arm
that narrows on volatile dates *and* on calm ones is re-absorbing a constant into `q_hat`.

**(L2) Calibration — primary.** Mean per-anchor `|coverage − 80|` over leg B must **fall** at ≥2 of 3,
and must not **rise by more than 2pp** at any readable horizon.

**(L3) Overshoot guard.** No anchor sitting within **±5pp** of 80% under the control may be moved
beyond **±10pp** by the arm, at any readable horizon. The 30d high-vol read already overshot to
77.68% pooled and 73–77% on two anchors; the symmetric failure here is inflating a well-calibrated
calm date to 90%+.

**(P) Placebo.** `L[t]` shuffled across dates, **200 permutations, seed 20260813** — the same seed
already fixed in the instrument. Bar: `|mean Δ coverage| ≤ 1.0pp` on leg B. A set selected on `L[t]`
is exactly where a rescaling artifact would hide, so this leg is not optional here.

**(J) Joint — secondary, and it is the shipping-relevant one.** Mean per-anchor `|coverage − 80|`
over leg B **∪** the high-vol arm set (`2026-02-14, 03-10, 04-06, 04-22, 06-16, 07-06`) must fall at
≥2 of 3. One global `gamma` has to help across the level range rather than trade one end for the
other. Secondary because the two sets sit in different collection regimes and the union inherits
both.

## Predictions, before the run

1. **The control under-covers on leg B**: pooled coverage below 80% at ≥2 of 3 readable horizons.
2. **The arm raises it** — (L1) passes.
3. **(L3) is the leg most likely to fail.** `gamma ≈ 0.70` is fitted on the whole panel, and the
   30d high-vol read overshot in the other direction at that same value; nothing constrains it to
   land inside ±10pp when it is applied at a 0.68× dose.
4. **Leg A is a coin flip and will not be interpretable either way**, which is why it has no bar.

## Void conditions

- Fitted `b` outside [0, 1], or a CI containing both 0 and 1, at any horizon.
- Any selected anchor refused at run time by either audit (the archive moves).
- Refitted `gamma` departing from 0.700 / 0.685 / 0.708 / 0.754 by more than 0.01 — the panel and the
  fit are unchanged, so a move means the instrument changed underneath the comparison.
- Reading absolute coverage on leg B as production's. **There is no validity leg on the low set** —
  no served band was ever quoted on a 2024 date — so only control→arm *differences* are admissible,
  and the transfer of the panel's 0.86–2.33pp agreement at 3/7/30d from the 2026 anchors to these is
  an **assumption**, stated here as one.
- Selecting, dropping or re-ordering an anchor after a coverage number is seen.

## Cost and what a pass buys

~2 minutes, offline, no dispatch and no retrain. A pass does **not** license shipping: it removes
the second of the two open questions the level-fix argument has to settle, leaving 2026-05-29 (one
anchor moving 10–13pp against its own `L[t]` at serving) and the walk-forward across regimes that
`2026-08-13-date-level-rescaling-passes-its-offline-gate.md` already named as the real gate. A
**failure** on (L1) is worth more: it would retire the arm and, with it, the last live candidate for
production's band level.
