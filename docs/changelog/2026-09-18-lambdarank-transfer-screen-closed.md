# LambdaRank serving-transfer screen: closed without merge (2026-09-18)

Branch `lambdarank-serving-transfer-screen` (commit `174353d`: `CV_END_ANCHORED`
fold-grid flag + Path A six-anchor spec) is superseded and will not be merged.

## Why

The Path A question — does the vs-q50 tied rank IC edge survive onto the serving
window — was answered the next day by the fuller instrument
(`docs/specs/2026-08-14-lambdarank-serving-transfer-design.md`,
biweekly retrain scoring daily, ~52 anchors) and recorded in
`docs/changelog/2026-08-14-lambdarank-serving-transfer-measured.md` with a
same-day durable-archive confirmation: h3 sign-flips (−0.174), h30 sign-flips
(−0.069), h14 is degenerate (constant output locally; LightGBM 10K-rows/query
cap on the durable cross-section), h7 alone confirms (+0.035, attenuated ~44%,
decile net −0.399, PT `no_skill`). The cross-sectional ranker pivot is refuted
for serving. The missing `experiment_log.csv` row for that verdict is added
alongside this note (`lambdarank-serving-transfer`, refuted).

The branch's cheap screen (`CV_END_ANCHORED` on `model-diagnostics.yml`) reads CV
geometry, not served predictions, and moves `q_hat` and the PT sample — the 08-14
design explicitly rejects touching the production fold grid for this read. Merging
it would re-ask a closed question on the wrong instrument with production-adjacent
side effects.

## What stays open

The Sep-17 shipped head (`RANKING_HEAD=1`, sidecar `rank_score`, never driving the
band; `CENTRE_OBJECTIVE=30:regression`) was trained on CV-geometry evidence
(`scripts/objective_comparison_ab.py`), the same geometry whose edges failed to
transfer. Its transfer verdict is therefore pending, not assumed: the honest read
is served `rank_score` vs served q50 on archive-basis outcomes with a 20-date
paired gate — the shadow `ranking_transfer_report` in
`docs/plans/2026-09-18-multi-head-champion-challenger.md` plus the
armed archive-basis leg (`docs/changelog/2026-09-13-archive-basis-centre-rank-armed.md`,
power gate ~09-27). No Path B wiring (rank into direction/band) until that
report returns SUPPORTED.
