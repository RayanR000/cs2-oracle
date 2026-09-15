# mypy baseline (2026-09-15)

No type checking existed: no mypy/pyright in deps, 12K-line `forecaster.py`
entirely un-gated. `lint.yml` now runs `mypy models backtest` from `backend/`
(config in `[tool.mypy]`), with `check_untyped_defs` on.

## Fixed to reach green (all real, all cheap)

- `models/item_parser.py` (4 errors) and `models/steam_types.py` (4 errors):
  heterogeneous parse dicts inferred as `bool|None`-valued; every later
  str/int assignment flagged. Annotated `result: dict[str, Any]` — the dicts
  are heterogeneous by construction, so this is the honest type.
- `models/served_recalibration.py` (1): `since: str | None = _UNSET` where
  `_UNSET = object()`. Annotated the sentinel `_UNSET: Any`.
- `backtest/price_resolution.py` (1): `fetchone()[0]` on a `tuple|None`.
  Aggregate queries always return one row; an assert documents it (the
  existing None check below stays the empty-archive guard).

## Explicitly NOT gated yet (overrides in `[tool.mypy]`, remove as fixed)

- `models.forecaster` (79 errors): LightGBM stub strictness (callbacks,
  Dataset kwargs), untyped-SQLAlchemy-`Base` cascades, `object`-arithmetic
  from untyped defs. Needs stub ignores + annotation passes, not a drive-by.
- `models.scale_model` (7): sklearn `LGBMRegressor(**dict[str, object])`
  stub strictness.
- `models.tft.trainer` (3): `None`-model paths when torch is absent.
- Followed imports: `database` (28, untyped `Base`), `config` (2,
  pydantic-settings version skew), `collectors.pipeline` (17),
  `collectors.data_validation` (2), `db.parquet` (8).

## What mypy does and does not buy here

It catches annotation-level drift (wrong defaults, None handling, bad kwargs)
— the class above. It does NOT catch DataFrame schema bugs (`price_tier`
`UndefinedColumn`, the served_recalibration panel shape): pandas frames are
untyped. Those stay the test suite's job (see 2026-09-15-ci-test-gate.md).
Do not cite a green mypy run as schema evidence.
