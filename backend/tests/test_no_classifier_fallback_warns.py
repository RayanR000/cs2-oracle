"""The dead-band direction fallback must announce itself.

On 2026-07-19 `predict` served directions from a global ±0.5% dead band on
`mid_ret`, because `direction_models` was empty — and called `flat` on 64.0% of
the ≥$1 cross-section against a 23.1% realised flat rate. Nothing logged it, so
it took a per-date decomposition three weeks later to find. The branch is still
reachable: any artifact missing its classifier for a horizon re-enters it.

See `docs/changelog/2026-08-11-the-da-gap-is-the-market-direction-of-five-dates.md`.
"""

import logging

from models.forecaster import DIRECTION_FLAT_TOLERANCE_PCT, ItemForecaster


def _bare() -> ItemForecaster:
    """A forecaster without __init__ — the warning reads no instance state."""
    return ItemForecaster.__new__(ItemForecaster)


def test_fallback_warns_and_reports_the_flat_share(caplog):
    with caplog.at_level(logging.WARNING, logger="models.forecaster"):
        _bare()._warn_no_classifier(3, 1052, 673)

    assert len(caplog.records) == 1
    rec = caplog.records[0]
    assert rec.levelno == logging.WARNING
    # The horizon, the item count, and the flat share all have to be in the
    # line: 64% flat is the signature that distinguishes this from a healthy
    # fallback, and the horizon is what makes it actionable.
    assert "h=3" in rec.message
    assert "1052" in rec.message
    assert "673" in rec.message
    assert "64.0%" in rec.message
    assert str(DIRECTION_FLAT_TOLERANCE_PCT) in rec.message


def test_no_warning_when_nothing_was_served_from_the_fallback():
    """Guard the guard: an empty horizon must not log or divide by zero."""
    f = _bare()
    with caplog_at(logging.WARNING) as records:
        f._warn_no_classifier(7, 0, 0)
    assert records == []


class caplog_at:
    """Minimal handler capture, so the zero case cannot pass on a filtered logger."""

    def __init__(self, level):
        self.level = level
        self.records = []

    def __enter__(self):
        self.logger = logging.getLogger("models.forecaster")
        self.handler = logging.Handler(level=self.level)
        self.handler.emit = self.records.append
        self.logger.addHandler(self.handler)
        self._prev = self.logger.level
        self.logger.setLevel(self.level)
        return self.records

    def __exit__(self, *exc):
        self.logger.removeHandler(self.handler)
        self.logger.setLevel(self._prev)
        return False
