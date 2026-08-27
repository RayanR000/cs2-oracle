# 2026-08-22 — The GBM band is decorative on a seam-free label (verdict, no serving change)

Closes the deep-model-review §12.12 gate's last escape hatch. The review's own
gate (`docs/research/2026-08-19-climatology-vs-gbm-band.md`) found the featureless
per-item climatology band beats the 33-feature GBM band on the served cohort, and
climatology shipped as the default scale on 2026-08-20. The open objection was
that the label the GBM trains on is corrupt — the consensus price is five stitched
estimators with visible market-wide phantom moves at the source cutovers — so the
comparison might be measuring a broken label rather than a decorative model.

This settles it. On a **seam-free within-source label**, the GBM loses at every
horizon by *more* than on the dirty one. Cleaning the label confirms the verdict;
it does not reverse it.

## What was measured (no retrain, no serving change)

Two new artifacts, both read-only against the durable archive:

- `backend/scripts/check_label_seams.py` — the label-seam detector and the
  within-source label. Flags any date whose cross-sectional **median** one-day
  return is too large to be a market and attributes it to the share of the cohort
  whose source set turned over. On the `voted` (production) label, 2026 flags
  **2026-03-22, 07-09, 07-10**, each with ~100% source turnover; the
  `within-source` basis flags **none** at a cost of **0.5%** of returns. Every
  2025 flag reproduces in an independently collected Steam series
  (`runtime/steam_listing_history.db`) at 0% source change — those are the real
  late-2025 market, not seams, and the 2025 legacy series needs no repair. Tests:
  `tests/test_label_seams.py` (13).
- `backend/scripts/climatology_vs_gbm.py --label within-source` — the §12.12 gate,
  now able to run on the clean label. Archive mode engineers `price_std_60d`
  itself, so **no retrain is required** — which matters, because the retrain/deploy
  chain is billing-paused and prod is frozen on the pre-clean artifact.

`within_source_index` rebuilds the price from the **voted** level (so it keeps the
consensus 2σ outlier vote) and zeroes only the pure cross-source steps — the days
with no source overlap at all, i.e. the seams. It deliberately does **not** rebuild
from raw single-source returns: that discards the vote and compounds a single
fat-finger print to ~8e6% (tried and reverted the same day).

## Result — served cohort, clim/GBM width ratio at matched 80% coverage

Ratio < 1 means the featureless climatology band is narrower at the same coverage.

| h | voted (dirty) | within-source (clean) |
|---|---|---|
| 3 | 0.74 | **0.70** |
| 7 | 0.75 | **0.69** |
| 14 | 0.74 | **0.71** |
| 30 | 0.80 | **0.74** |

Climatology is ~30% narrower on the clean label (~25% dirty). The GBM's
out-of-sample coverage also **falls** 0.78 → 0.73 once the seams are gone: the
dirty label's phantom moves were inflating the rolling-std scale and manufacturing
the band's over-coverage.

Caveat: on the broad ≥$1 universe a <0.1% near-zero-price residual tail compounds
to implausible h≥14 forward returns and dominates the *mean* width (the per-item
median and tails are tighter than voted). Use `--served-cohort`, which is the
cohort the band is served on and excludes that tail.

## Consequence

The GBM band is **instrumentation, not the deliverable** — a clean label confirms
it. This is a docs/verdict change only: climatology is already the served scale, so
nothing in the serving path moves. The GBM q50 still provides the forecast centre,
and the exceedance head still powers `move_odds`; neither is in scope here. Whether
to also replace the centre with a featureless one is a separate, measurable
question that would require the deploy chain unpaused — not opened here.

Do not re-run the label-seam question, and do not re-investigate the 2025 dates:
both are settled above.
