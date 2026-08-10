# Master execution order, 2026-08-09

> ## Status — 2026-08-10 05:00 UTC
>
> **Phases 0, 1 and 2 are complete.** All of the label-integrity work is merged and pushed;
> `main` = `origin/main` = `58d681e`. Do not re-run anything marked ✅.
>
> - **D1 returned: the Steam listing page ANSWERS** (0.36 req/item). Blocker 5d was a false
>   positive. So `R11`, `R13` and `5c` are unblocked on access — but nothing has been built, and
>   `backfill_steam_listing_history.py` still probes the wrong URL.
> - **D2 could not be evaluated as written.** `meta.json` stores no coverage field, and `q_hat` is
>   the conformal quantile of its own calibration pool, so it cannot under-cover it. The fallback
>   check is the one that mattered: folds 9/8/8/8, every fold capped at exactly 300,000, OOF pools
>   six figures, `q_hat` up at all four horizons — band **wider**, the safe direction, and
>   `horizon_feature_cols` set-identical under the A3/A4 knobs. **Verdict: proceed.**
> - **D3 is RULED, and it is Branch A** (`f833882`, re-measured post-exclusion in `3c62a49` /
>   `09e945e`). The reversal survives composition control: on the primary 2026 set basis,
>   composition-stable **+0.0838** against unconditional **+0.0838** at 167–172 dates. The label is
>   a return, not a quoting artifact. **Phase 3's Branch A is live.**
> - ~~⚠️ Live defect from hazard 1 below: `VOTED_CACHE_VERSION` is 5 but `price-forecast.yml` still
>   keys on `voted-v4-`.~~ **Fixed in `873148b`** — constant and key moved together, straight to
>   **v6**, leaving no v5 key behind.
>
> ✅ **The retrain landed 2026-08-10** (run `31356483719`, `mode=train-only` on `a0c215e`, 39m32s).
> PT returns **skill at all four horizons**; the model still **loses to `−return_1d`** on rank IC
> at all four. `docs/changelog/2026-08-10-post-revote-retrain.md`.
>
> ⚠️ **Two things it did not settle.** Served accuracy is still unmeasured —
> `CV_DIAGNOSTIC_CLASSIFIER=0`, so the classifier that serves direction is unscored and every DA
> figure in the log is the q50 sign. And the run went **over the 30-minute cap** at 39m32s, with
> Optuna alone costing 692.9s against 63.3s the run before:
> `docs/research/2026-08-10-training-cost-levers.md`.
>
> **Phase 1 did not deliver a speedup.** The cold retrain measured **1426.3s** against a predicted
> ≈600s, because 1.2 removed early stopping from the trial loop and made Optuna the largest phase
> at 392.3s. No like-for-like control was run. `docs/changelog/2026-08-09-training-cost-levers.md`
> has the full accounting. Do not quote a speedup from this work.

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

| # | Step | Status | Where | Depends on | Cost | Gate |
|---|---|---|---|---|---|---|
| **0.1** | Dispatch `price-forecast.yml` with `mode=full`, `FORCE_RETRAIN=1` | ✅ run `31337078991`, 35m24s | ops | — | 1 dispatch | — |
| **0.2** | Re-verify the Steam listing page from this egress | ✅ **it answers**, 0.36 req/item | ops, one `curl` | — | 5 min | **D1** |
| **0.3** | Correct the retired +3.50pp citation, `model.md:210,533` | ✅ `895005a` — both sites now read "withdrawn, re-derives to +1.642pp, null" | docs | — | tiny | — |
| **0.4** | Publish the 17 files missing from the canonical archive | ✅ all 17 published; ⬜ compaction never dispatched | `aggregator-update.yml` | — | small | — |
| **1.1** | Cap CV fold training rows at 300k | ✅ `6b6fc81` | cost plan, Task 1 | — | small | — |
| **1.2** | Optuna selects on within-date rank IC | ✅ `8be48c5` — but cost 392.3s, not 35s | cost plan, Task 2 | 1.1 | small | — |
| **1.3** | Allowlist before the correlation prune | ✅ `a20b5a2` | cost plan, Task 3 | — | small | — |
| **1.4** | Skip the discarded feature blocks | ✅ `17306b0` (1.3+1.4 = −16.5s, not −33.7s) | cost plan, Task 4 | 1.3 | small | — |
| **1.5** | `CV_DIAGNOSTIC_CLASSIFIER` default off | ✅ `dfafdfb` | cost plan, Task 5 | — | tiny | — |
| **1.6** | Content-hash the archive fingerprint | ✅ `db8d5d5` + `1b5ee06` | cost plan, Task 6 | — | small | — |
| **1.7** | Cold retrain, verify, correct the docs | ✅ retrain + changelog; ⬜ **doc correction = 0.3** | cost plan, Task 7 | 1.1–1.6 | ~15 min run | **D2** |
| **2.1** | Persist the label-voiding audit | ✅ `3d582ab`, `5e7dc60` | label plan, Task 1 | — | small | — |
| **2.2** | Add `n_ask_sources` to the voted frame | ✅ `48fd352`, `4f828fd` | label plan, Task 2 | — | small | — |
| **2.3** | Run the powered-up composition test | ✅ instrument `80c3e07`, write-up `f833882` + `cddcf76` | label plan, Task 3 | 2.1, 2.2 | medium | **D3** |
| **2.4** | Exclude the Steam trailing-window feeds from voting | ✅ `873148b` (+ `ac713cc`, `09e945e`); cache key moved v4 → **v6** in the same commit | label plan, Task 4 | 2.2 | small + re-vote | — |
| **2.5** | Re-run 2.3 on the post-6c consensus | ✅ `3c62a49` — every measured cell moved 0.0000–0.0015; conclusion unchanged | label plan, Task 3 | 2.4 | small | — |
| **2.6** | Retrain on the post-2.4 consensus | ✅ run `31356483719`, 39m32s. PT skill at all 4; still loses to `−return_1d` | ops, `price-forecast.yml` `mode=train-only` | 2.4 | done | — |
| **3.x** | Accuracy work — **Branch A** | ⬜ unblocked by D3; start after 2.6 gives a post-exclusion baseline | branches on **D3** | 2.3, 2.6 | — | — |

Phase 0 is roughly an hour. Phase 1 is a day. Phase 2 is a day plus a re-vote. Phase 3 does not
have an estimate until D3 returns.

**What is actually left:** 2.6 (the retrain), 0.4's stranded `snapshots-2026-08.parquet` deletion,
and O2 (the `architecture/model.md` + `model-optimization.md` rewrite, tracked in
`2026-08-09-next-steps.md`). Then Phase 3, Branch A.

---

## Decision points

### D1 — after 0.2. Does the Steam listing page answer from this egress? — ✅ **ANSWERED: YES**

> **Resolved 2026-08-09.** It answers, at **0.36 requests per item**. §5d's soft-block was a false
> positive — the 230 KB shell from the `market_hash_name` URL is normal SSR. You must resolve the
> canonical `G<id>` first; the two-step `render/?…` → `G<id>` route returns the full page.
>
> **Take the "Answers" branch below.** `R11`, `R13` and `5c` unblock on access, and D3
> (devynpruden cross-check) plus the listing backfill are promoted into Phase 3 regardless of which
> branch D3 takes. Neither has been started, and `backfill_steam_listing_history.py` still probes
> the name URL with a block detector keyed on the shell — fix that first.


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

### D2 — after 1.7. Did band coverage hold? — ✅ **PROCEED** (but the gate as written was not executable)

> **Resolved 2026-08-09.** The check below cannot fail and cannot be run: `meta.json` stores no
> coverage field, and `q_hat` *is* the finite-sample conformal quantile of the pooled OOF scores,
> so coverage on that pool is ≥ 0.80 by construction at n = 155,923–174,784.
>
> The informative read was the fallback — did 1.1 thin **rows** or **dates**? Folds **9/8/8/8**
> (matching the expectation), every fold's `n_train` capped at exactly **300,000**, OOF pools
> 155,923–174,784, and `q_hat` **rose at all four horizons** (95.25 / 137.68 / 205.99 / 309.49).
> The band got wider, which is the over-coverage direction the cap predicts.
> `horizon_feature_cols` is set-identical under the A3/A4 knobs at 33 columns.
>
> ⚠️ **What is still not established:** coverage on *held-out* data. Nothing measured that. If the
> served band is ever observed under-covering, `CV_MAX_TRAIN_ROWS` is the first knob to raise.


Read empirical coverage against `NOMINAL_COVERAGE = 0.80` from the fresh `meta.json`.

- **≥ 0.80** → proceed. Over-coverage is the expected direction: the fold model fits on less data
  than the served one, so `q_hat` comes out larger.
- **< 0.80** → **stop and raise `CV_MAX_TRAIN_ROWS`.** A narrower band means the cap hit the wrong
  axis. Check `n_folds` and the OOF record count first — if either moved, 1.1 thinned dates
  instead of rows.

Also check that `horizon_feature_cols` is set-identical to the previous artifact. If it moved,
1.3 or 1.4 changed which features reach a booster, which neither is allowed to do.

### D3 — after 2.3. Is the label a tradeable return? — ✅ **RULED: YES. Branch A.**

> **Ruled 2026-08-09 (`f833882`), re-measured post-exclusion (`3c62a49`, `09e945e`).** The
> reversal **survives** composition control at a well-powered date count, so the label is a
> tradeable return and Phase 3 Branch A is the live branch.
>
> Primary basis — 2026, composition = the **set** of source names, post-`873148b` vote:
> all rows **+0.0838**, composition-stable **+0.0838**, window-incomplete +0.1095,
> stable & single-source +0.0926, at 167–172 dates. Secondary — full archive on the count
> basis, h=3: all rows +0.1940, stable +0.1936 at 4,598 dates. Holding composition still does
> not move the signal on either basis.
>
> The `changed` and `≥3-source` cells remain **underpowered** (19–25 dates, below
> `MIN_DATES_TO_REPORT = 30`) and carry no number, before and after the exclusion. The paired
> stable−changed contrast is therefore still unmeasured; the ruling rests on stable ≈
> unconditional, not on a paired difference. Full write-up and reproduction commands:
> `docs/research/2026-08-09-composition-stability.md`.

<details>
<summary>How the question changed on the way to that ruling (2026-08-09)</summary>

> **The instrument shipped 2026-08-09 (`80c3e07`) and it changed the question.** The published §5
> baseline **does not reproduce**, for a definitional reason rather than a bug: "composition
> stable" was computed treating a NULL source as never equal to itself, and *every* pre-2026 row
> has `source IS NULL` (9,417,947 item-days). So the published "stable" cell contains **no pre-2026
> data at all**, and its 775 "changed" dates are ~716 pre-2026 dates classified as changed for want
> of a label. **The 0.1676 → 0.1006 fall attributed to composition control is a 2013-2025 → 2026
> regime difference.**
>
> Within 2026, where composition is actually observed and the cells are **powered at 181–185
> dates**: **stable +0.1027, changed +0.0967, unconditional +0.1023.** Holding composition stable
> does not move the signal.
>
> **Caveat that the ruling must carry:** `n_ask_sources` is a count, not a set, so a source swap at
> constant count reads stable. The instrument **under-detects** composition change, biasing toward
> the answer it found. Agreed basis: primary = 2026 observable era on **set**-based composition
> (immune to the NULL assumption, closes the count-vs-set weakness); secondary = full archive on
> the count basis with its assumption stated.
>
> ~~⬜ Blocking the ruling (as of 23:44 UTC, uncommitted): the write-up stops at a
> `<!--RESULTS-->` placeholder.~~ **Cleared** — `f833882` committed the results; the placeholder is
> gone and the document is 378 lines with both bases and the post-exclusion re-run.

</details>

**This is the one that matters.** Three outcomes, and they lead to different projects.

| Outcome | Next |
|---|---|
| **Signal survives** composition control at a reportable date count | ✅ **this is what happened** — Branch A below |
| **Signal vanishes** when composition is held stable | Branch B — and this is the headline result of the project |
| **Clean cells stay under 30 dates** | Branch C — blocked on calendar time, not effort |

~~**Branch C is the likely one.**~~ **Superseded 2026-08-09** — the 2026 cells came back powered at
181–185 dates, so underpowerment is not the answer. The standing rule is unchanged: do not work
around a thin cell by lowering `MIN_DATES_TO_REPORT`.

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
   ⚠️ **This went wrong once and is now fixed.** 2.2 bumped the constant to 5 and left
   `price-forecast.yml:115-116` on `voted-v4-`. The consequence was a permanent cache **miss**, not
   a stale hit — the cached parquet's filename embeds the version, so a v4 frame cannot be read as a
   v5 one. `873148b` moved the constant to 6 and the key straight from `voted-v4-` to `voted-v6-` in
   the same commit. **No v6 key has been populated yet**, so the next run votes cold.
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

**All four are done as of 2026-08-09.** Kept for the record; the live list is the four ⬜ rows in
the sequence table.

1. ~~**0.1** — the pipeline is red and no labels are maturing.~~ ✅ green since run `31337078991`.
2. ~~**1.1** — 22% of the retrain, nothing structural breaks.~~ ✅ `6b6fc81`. Nothing structural did
   break; the retrain nonetheless got slower, from 1.2.
3. ~~**1.2** — 35 seconds, and it fixes params for 14d/30d chosen under a refuted criterion.~~
   ✅ `8be48c5`. **It was 392.3s, not 35s.** The params are now selected on the right criterion;
   whether they forecast better is unmeasured.
4. ~~**2.3** — it decides whether any of the rest is worth doing.~~ 🟡 measured, unruled. Writing it
   up is now the highest-value remaining item.

### The current three

1. **Training cost** — both recent runs are over the 30-minute cap (35m24s, then 39m32s). The
   free lever is restoring the model cache on training runs, which removes an Optuna phase that
   measured 63.3s and then 692.9s. `docs/research/2026-08-10-training-cost-levers.md`.
2. **O2** — `architecture/model.md` and `model-optimization.md` carry ~25 stale claims between
   them (training defaults in eight places, `MODEL_ARTIFACT_VERSION` 5 vs 6, `VOTED_CACHE_VERSION`
   v3 vs 6, early stopping described as live). 0.3's number is fixed; the surrounding config
   description is not.
3. **0.4's last leg** — delete the stranded `snapshots-2026-08.parquet` from the canonical repo.
   Compaction itself already landed; only that one file is left.

~~2.6, the retrain~~ ✅ done 2026-08-10, run `31356483719`.

Then Phase 3, Branch A, in order: A1 → A2 → A3 → A4. **Do not compare any A/B result to a stored
one** — hazard 6 — every stored verdict predates 2.4's re-vote.
