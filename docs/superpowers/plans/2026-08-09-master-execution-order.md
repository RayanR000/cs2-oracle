# Master execution order, 2026-08-09

The single ordered list. Every row names a concrete task in a plan document or a one-line
operation; nothing here restates *why* — that lives in
`docs/research/2026-08-09-model-and-data-research.md` and
`docs/research/2026-08-09-next-steps.md`.

**Read the three decision points before starting.** Two of them can end a phase, and one of them
can end the accuracy roadmap entirely. Work that runs past a decision point without reading it is
wasted.

| Plan | Covers |
|---|---|
| `docs/superpowers/plans/2026-08-09-training-cost.md` | Phase 1, tasks 1–7 |
| `docs/superpowers/plans/2026-08-09-label-integrity.md` | Phase 2, tasks 1–4 |

---

## The sequence

| # | Step | Where | Depends on | Cost | Gate |
|---|---|---|---|---|---|
| **0.1** | Dispatch `price-forecast.yml` with `mode=full`, `FORCE_RETRAIN=1` | ops | — | 1 dispatch | — |
| **0.2** | Re-verify the Steam listing page from this egress | ops, one `curl` | — | 5 min | **D1** |
| **0.3** | Correct the retired +3.50pp citation, `model.md:210,533` | docs | — | tiny | — |
| **0.4** | Publish the 17 files missing from the canonical archive | `aggregator-update.yml` | — | small | — |
| **1.1** | Cap CV fold training rows at 300k | cost plan, Task 1 | — | small | — |
| **1.2** | Optuna selects on within-date rank IC | cost plan, Task 2 | 1.1 | small | — |
| **1.3** | Allowlist before the correlation prune | cost plan, Task 3 | — | small | — |
| **1.4** | Skip the discarded feature blocks | cost plan, Task 4 | 1.3 | small | — |
| **1.5** | `CV_DIAGNOSTIC_CLASSIFIER` default off | cost plan, Task 5 | — | tiny | — |
| **1.6** | Content-hash the archive fingerprint | cost plan, Task 6 | — | small | — |
| **1.7** | Cold retrain, verify, correct the docs | cost plan, Task 7 | 1.1–1.6 | ~15 min run | **D2** |
| **2.1** | Persist the label-voiding audit | label plan, Task 1 | — | small | — |
| **2.2** | Add `n_ask_sources` to the voted frame | label plan, Task 2 | — | small | — |
| **2.3** | Run the powered-up composition test | label plan, Task 3 | 2.1, 2.2 | medium | **D3** |
| **2.4** | Exclude the Steam trailing-window feeds from voting | label plan, Task 4 | 2.2 | small + re-vote | — |
| **2.5** | Re-run 2.3 on the post-6c consensus | label plan, Task 3 | 2.4 | small | — |
| **3.x** | Accuracy work | branches on **D3** | 2.3 | — | — |

Phase 0 is roughly an hour. Phase 1 is a day. Phase 2 is a day plus a re-vote. Phase 3 does not
have an estimate until D3 returns.

---

## Decision points

### D1 — after 0.2. Does the Steam listing page answer from this egress?

The research environment got 10 back-to-back 200s at ~1.3 req/s against the rebuilt SSR page, with
`purchases` back to 2014 and full order-book depth at ~10 names per fetch.
`2026-08-07-next-steps.md` §5d records this machine soft-blocked since 2026-08-05.

- **Answers** → `R11`, `R13` and `5c` all unblock at ~0.1 requests per item. Promote **D3
  (devynpruden cross-check)** and the listing backfill into Phase 3 regardless of which branch
  D3 takes — they are data collection, not modelling, and they accumulate the multi-source days
  the composition test needs.
- **Still blocked** → leave 5d as written. Do not plan a backfill on the research environment's
  result.

Five minutes, and it changes the value of three roadmap items. That is why it is in Phase 0.

### D2 — after 1.7. Did band coverage hold?

Read empirical coverage against `NOMINAL_COVERAGE = 0.80` from the fresh `meta.json`.

- **≥ 0.80** → proceed. Over-coverage is the expected direction: the fold model fits on less data
  than the served one, so `q_hat` comes out larger.
- **< 0.80** → **stop and raise `CV_MAX_TRAIN_ROWS`.** A narrower band means the cap hit the wrong
  axis. Check `n_folds` and the OOF record count first — if either moved, 1.1 thinned dates
  instead of rows.

Also check that `horizon_feature_cols` is set-identical to the previous artifact. If it moved,
1.3 or 1.4 changed which features reach a booster, which neither is allowed to do.

### D3 — after 2.3. Is the label a tradeable return?

**This is the one that matters.** Three outcomes, and they lead to different projects.

| Outcome | Next |
|---|---|
| **Signal survives** composition control at a reportable date count | Branch A below |
| **Signal vanishes** when composition is held stable | Branch B — and this is the headline result of the project |
| **Clean cells stay under 30 dates** | Branch C — blocked on calendar time, not effort |

**Branch C is the likely one.** The multi-source era is 24 days deep. Do not work around it by
lowering `MIN_DATES_TO_REPORT`.

---

## Phase 3, by branch

### Branch A — the signal survives

In order. Each is gated on the one before clearing its MDE.

| # | Step | Ref | Note |
|---|---|---|---|
| A1 | Cross-sectional rank transform of every feature, per date | next-steps C1 | One groupby. Test alone, before A2 |
| A2 | `lambdarank` within date, scored by rank IC | next-steps C2 | Raise `lambdarank_truncation_level` above 30; linear `label_gain`; budget **+4 boosters**, a ranker has no level |
| A3 | Residual reversal feature | next-steps C3 | Demean `return_1d` within weapon/tier/crate |
| A4 | Re-derive the feature allowlist with `cross_sectional` restored | next-steps C4 | Compute the MDE first. 1.4's skip set is derived from the allowlist so this cannot silently drop columns |

### Branch B — the signal is a quoting artifact

Stop modelling work. Write it up as the primary finding — it is a stronger result than any
accuracy number this project could publish, and `2026-08-08-model-review.md` §5 already says so:
*"If the effect is a quoting artifact, no feature or architecture work can help, because the label
is not a tradeable return."*

Then reconsider what the product is. `2026-08-07-next-steps.md` step 3's expected result —
failure on `n_actionable` first — belongs at the top of that document.

### Branch C — underpowered

Accumulate multi-source days, and spend the wait on work that does not depend on D3:

| # | Step | Ref |
|---|---|---|
| C1 | Fix the csgotrader collector to per-provider paths | next-steps D2 |
| C2 | devynpruden Kaggle volume cross-check | next-steps D3 |
| C3 | `stattrak_premium_z30` — free, 13 years deep, no fetch | next-steps D4 |
| C4 | Split conformal + ACI | next-steps C5 |
| C5 | Regime models and `N_ENSEMBLES` decision | next-steps C6 |
| C6 | Deflate the accumulated A/Bs | next-steps C7 |
| C7 | Consolidate the 15 experiment harnesses | next-steps D5 |

Re-run 2.3 monthly. It becomes answerable on its own.

---

## The two experiments you asked for

Neither is on the critical path. Both are cheap, and both are more interpretable after D3.

**"100 items vs 200 items."** `backend/scripts/ab_test_training_breadth.py`. Edit
`N_NARROW`/`N_MID`/`N_WIDE` at `:105-107`; there is no CLI flag. **Pass `--fixed-rounds`** — it is
the only harness of 15 with that escape hatch, and without it both arms early-stop on the window
they score. Nested arms, so the item draw differences out and the MDE is 0.69–2.15pp. Under 2
minutes for one horizon × two arms.

Run it **after 1.2**, not before: the harness inherits the tuned params, and until 1.2 lands those
were selected on a criterion the project discarded.

Prior: the 2026-08-08 re-run found breadth positive at **14d only** (+2.33pp [+0.44, +4.74]), null
at 30d, saturating by 350.

**"Does volume affect accuracy."** Runnable now, but four things must be right or the answer is
wrong: cut the frame at **2025-12-31** (not the harness's `VOLUME_LIVE_THROUGH = "2026-04-30"`);
never pool across 2026-03-22, where `ask_volume` starts masquerading as `volume`; know that the
pre-2026 series has **zero** zero-volume rows so a no-sale day is an absent row; and note the
repaired feed would be a different quantity from what the pre-2026 data holds. Detail in
`2026-08-09-model-and-data-research.md` §4f.

"Does volume affect speed" is not an experiment — per-phase timings swing ±25% between clean runs.

---

## Sequencing hazards

These are the ways a correct set of changes produces a wrong result if landed in the wrong order.

1. **`VOTED_CACHE_VERSION` is bumped by three separate tasks** — 2.2 (4→5), 2.4 (5→6), and 1.6
   adds a CI cache key naming a version. **Whichever lands last must move the workflow key in the
   same commit**, or CI restores a frame voted under the old rule and silently trains on it.
2. **2.4's exclusion must run before 2.2's count**, or a trailing-window leg inflates
   `n_ask_sources` and the composition partition is wrong in the direction that hides the problem.
3. **Run 2.3 before 2.4, then again after.** The baseline must reproduce on the *current*
   consensus, or the 6c change confounds the comparison with the thing being measured.
4. **1.2 depends on 1.1's signature change.** Land them in order.
5. **1.4's skip set must stay derived from the allowlist.** If A4 later re-admits
   `cross_sectional` and the skip set is a literal, the frame carries the group allowlisted and
   its columns absent — median-filled to zero, undetectably.
6. **Every stored A/B verdict predating 2.4 becomes uncitable.** Labels from 2026-03 onward sit
   downstream of the re-vote. Do not compare a post-2.4 number to a stored one.

---

## If you only do four things

1. **0.1** — the pipeline is red and no labels are maturing.
2. **1.1** — 22% of the retrain, nothing structural breaks.
3. **1.2** — 35 seconds, and it fixes params for 14d/30d chosen under a refuted criterion.
4. **2.3** — it decides whether any of the rest is worth doing.
