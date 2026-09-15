# Experiment log (2026-09-15)

Arms were tracked in /tmp files, changelogs, and memory — so dead ideas got
re-proposed (modelled-sigma band scales four times, relative arms repeatedly).
`docs/experiment_log.csv` is the append-only answer to "has this been tried":
one row per arm — date, name, hypothesis, metric, estimate, CI, n, verdict,
link. 58 rows backfilled from the changelog's filename-verdicts; estimates
filled only where the note states a number, blank otherwise (a qualitative
refuted beats a precise nothing).

## Why CSV, not MLflow

No server, no client, no schema migration: the log is 58 lines of git-diffable
text beside the notes it cites, and `test_experiment_log.py` enforces the
schema (columns, verdict enum, ISO dates, no duplicate arms, every link
resolves to a repo file, the named dead ideas stay present). MLflow would add
a tracking server to a repo whose experiments run as CI-dispatched scripts and
whose results must be reviewable in a PR diff. Revisit if arms ever need
parameter-level search bookkeeping (Optuna studies) rather than verdicts.

## Verdicts

shipped (in the serving path), refuted (measured, absent/wrong-signed), void
(the bar was invalid — placebo matched, contaminated), measured (built an
instrument or number, no arm decision), inconclusive. `closed`/`killed`
changelog titles map to refuted — the inquiry ended against the arm.

## Workflow

Logging is rule 4 in root AGENTS.md workflow rules: a shipped/refuted/void
arm gets a dated changelog note AND a log row. The row is the cheap part —
write it when the verdict lands, not when the archaeology starts.
