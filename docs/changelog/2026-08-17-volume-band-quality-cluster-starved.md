# Volume feed repair (iflow) tested on band quality — cluster-starved, no clean ship

*2026-08-17*

Wired the iflow `count_in_24` backfill into `scripts/ab_test_volume_features.py` as a
feed-repair arm (`IFLOW_VOLUME=1`) and re-scoped the harness's endpoint from directional
accuracy to **band quality** — `rel_width` (interval width / price, lower better) subject to a
coverage gate — because `AGENTS.md` invariant 4 forbids shipping the volume lead on its DA gain.
Prereg: `docs/research/2026-08-17-volume-band-quality-preregistration.md`.

## What changed in the harness
- `IFLOW_VOLUME=1` sources `volume` from the iflow `steam_volume` (= `count_in_24`) series in
  `buff-iflow-staging/price-archive/iflow-liquidity-*.parquet`, as the **single** volume series
  across the panel (never coalesced with the dead native feed — mixing two scales inside a
  30d/60d rolling window is the same corruption the cliff-straddle causes).
- Eval panel restricted to the iflow era **2022-04-18 → 2026-05-20** so the volume features are
  live on every row they are scored on. Without the lower bound, 68% of rows were pre-2022 (no
  iflow volume) and diluted the treatment arm; the first run mis-flagged a spurious width win off
  that dilution, which the 80% join-coverage void gate caught.
- Added `in_interval` and `rel_width` as per-row paired metrics; added fold-clustered width and
  coverage contrasts (`paired_arm_contrasts`, `higher_is_better=False` for width) alongside the
  now-diagnostic DA contrast. Cache fingerprint extended with `IFLOW_VOLUME` / `IFLOW_ERA_FROM`.

## Result (≥$1 cohort, join coverage 80.9%, 8 folds/horizon)

| h | rel_width | coverage Δ | placebo | DA (diagnostic) |
|---|---|---|---|---|
| 3 | null +0.048 [−0.17, +0.22] | null | clean | +1.41pp positive |
| 7 | null −0.076 [−0.26, +0.10] | null | tiny, < ½ treatment | +1.77pp positive |
| 14 | **positive −0.204 [−0.35, −0.06]** | null (holds ~81%) | clean | +1.32pp null |

Band tightens **only at h=14** — where DA goes null, so it is a real band-quality effect, not a
DA artifact. Width trends tighter with horizon.

## Why there is no clean verdict
The preregistered **≥14-cluster power floor fails at every horizon (8 < 14)**. Restricting to the
iflow era to make volume live everywhere costs folds: 2013–2026 gave 26, 2022–2026 gives 8. The
void condition drops all three horizons — including the h=14 pass — for lack of clustered power.
The floor was **not** rewritten post-hoc (cf. the three preregs whose gates were rewritten after
the result was seen).

**The volume → band-quality lead is neither confirmed nor cleanly refuted.** The iflow era is too
short (starts 2022-04) to reach ≥14 non-overlapping folds at these horizons. The h=14 tightening
is worth remembering but is one horizon at the power boundary; do not ship on it alone. A clean
test needs a re-scoped fold scheme under a fresh preregistration.

**Prereg premise corrected:** the "~10× episodes" claim was backwards — the iflow era is *shorter*
than the existing panel and yields *fewer* folds; the 10× only held versus a 2026-only baseline
this harness never used.
