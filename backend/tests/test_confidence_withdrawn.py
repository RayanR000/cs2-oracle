"""The API must not publish the served ``confidence`` label, and must keep
recording it so the backtest can tell us when it becomes publishable.

The label is a bare ``>= 0.5`` cut on the directional classifier's max class
probability (``DIRECTION_CONFIDENCE_HIGH``), never calibrated against realised
hits. Measured within each forecast date on the >=$1 ``lgbm-v3%`` panel in prod
(19,917 outcomes, 2026-08-12), on the 11 cells carrying ``n_high >= 30``:

- the "high" cohort's own directional accuracy is **29.0-45.8%** — every cell
  below a coin flip, against ``CONFIDENCE_TARGET_ACCURACY = 80.0``;
- the gap to the "low" cohort is **-8.26 to +4.65pp with mixed sign** (7 of 11
  negative), so the tag does not order accuracy either.

A label that says "high" on calls landing below 50% is a false claim whichever
way the gap points, and consumers weight on it.

Do NOT read the backtest's ``conf_gap_pp`` as the evidence here: it pools across
forecast dates, so it inherits the market-composition term that
``docs/changelog/2026-08-11-the-da-gap-is-the-market-direction-of-five-dates.md``
rules out for exactly this kind of contrast. Its -38.7pp at h=30 is one item.

``tests/test_opportunity_selection.py`` removed the *selection* on this tag in
2026-08-03. This removes the *publication* of it. The column, the writer and
``conf_gap_pp`` all stay: withdrawing the claim must not destroy the instrument
that would show the claim becoming true.

See docs/changelog/2026-08-12-served-confidence-withdrawn.md.
"""

from __future__ import annotations

from pathlib import Path

from database import ItemForecast


class TestInstrumentSurvives:
    def test_column_still_exists(self):
        """Dropping it would silently zero ``conf_gap_pp``, which is the only
        measurement that can retire this withdrawal."""
        assert "confidence" in ItemForecast.__table__.columns

    def test_daily_run_still_writes_it(self):
        source = (Path(__file__).resolve().parents[1] / "scripts" / "forecast_prices.py").read_text()
        assert '"confidence": fcast.get("confidence")' in source
