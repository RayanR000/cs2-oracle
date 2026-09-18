"""Per-horizon centre champion policy.

Champion selection is explicit, versioned code configuration. No database
override and no evaluation-driven write path.
"""

from __future__ import annotations

HORIZONS = (3, 7, 14, 30)
REGISTERED_CENTRES = frozenset({"gbm_q50", "last_price"})
CENTRE_CHAMPIONS = {3: "gbm_q50", 7: "gbm_q50", 14: "gbm_q50", 30: "gbm_q50"}


def _validate_policy() -> None:
    if set(CENTRE_CHAMPIONS) != set(HORIZONS):
        raise RuntimeError("centre policy must name every supported horizon exactly once")
    unknown = set(CENTRE_CHAMPIONS.values()) - REGISTERED_CENTRES
    if unknown:
        raise RuntimeError(f"unknown centre champion(s): {sorted(unknown)}")


def centre_champion(horizon: int) -> str:
    if horizon not in HORIZONS:
        raise ValueError(f"unsupported horizon: {horizon}")
    return CENTRE_CHAMPIONS[horizon]


def centre_challengers(horizon: int) -> tuple[str, ...]:
    champion = centre_champion(horizon)
    return tuple(name for name in ("gbm_q50", "last_price") if name != champion)


_validate_policy()
