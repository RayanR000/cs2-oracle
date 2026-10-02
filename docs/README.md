# docs/

Where a doc and the code disagree, the code wins. Report the mismatch instead of working
around it.

## Start here

| Doc | What it is |
|-----|------------|
| [`PORTFOLIO.md`](PORTFOLIO.md) | Project write-up: what was built, what was found, what was refuted |
| [`operations.md`](operations.md) | Runbook: workflow schedules, required secrets, troubleshooting |
| [`architecture/`](architecture/) | `model.md`, `model-optimization.md`, `pipeline.md`, `data.md` |
| [`references/`](references/) | Market domain, Steam API, data sources and inventory |
| [`experiment_log.csv`](experiment_log.csv) | Every shipped, refuted and void experiment |

## Decision records

- [`changelog/`](changelog/): dated records of each change with its measured effect. This
  is the durable record; search it before re-running an experiment.
- [`research/`](research/): preregistrations and research notes, one file per question.
- [`specs/`](specs/) and [`plans/`](plans/): designs and execution checklists. Shipped work
  is also recorded in `changelog/`, so treat these as history, not as current instructions.
  New ones are named `YYYY-MM-DD-<topic>.md`.

## Read before quoting a number

Two things changed after most August figures were measured:

- **Band geometry.** A signed two-quantile conformal band went live 2026-08-19
  (`SIGNED_BAND_SERVING_START`) and a per-item climatology band scale on 2026-08-20
  (`CLIMATOLOGY_SERVING_START`). Never difference a coverage number across either date.
- **Product claim.** CS2 Oracle is a range forecaster. Directional accuracy is tracked but
  is not a product claim (`changelog/2026-08-15-cs2-oracle-is-a-range-forecaster.md`).

Always quote both coverage columns: `interval_coverage` (calibrated) beside
`interval_coverage_dollar_basis` (published).

## Historical notes

[`INDEX-2026-08.md`](INDEX-2026-08.md) is the old annotated index, last refreshed
2026-08-21. It keeps the per-doc caveats and the August measurement banners. It is
accurate as a record of what was known then, not as a current map.

`product.md` describes the frontend deleted 2026-08-10. Treat it as rebuild input, not a
live spec.
