# The next-steps docs were re-proposing finished work

**Date:** 2026-08-09
**Type:** documentation audit — no code changed

## Why

Nine of the ten plans in `docs/plans/` described work that had already landed, with
every step still an unchecked `- [ ]` and no completion marker anywhere in the file. The live
roadmap, `docs/research/2026-08-09-next-steps.md`, marked all six Track A items **NOT STARTED**
hours after all six shipped, and marked Track O and D1 **NOT STARTED** after both were resolved.

The cost of that is not tidiness. A reader following the docs would have re-run the entire
training-cost analysis, re-dispatched a green pipeline, and re-verified a Steam blocker that had
already been refuted.

## What was reconciled

Verified against the working tree, `git log`, the GitHub Actions run history, and the canonical
`RayanR000/cs2-oracle-data` repo — not against the local archive copy, which runs behind.

| Doc | Change |
|---|---|
| `research/2026-08-09-next-steps.md` | Status block added. **Track A closed** (A1–A6, commits `6b6fc81`, `8be48c5`, `a20b5a2`, `17306b0`, `dfafdfb`, `db8d5d5`). **O1 done** (run `31337078991`), **O3 done** bar the compaction dispatch, **D1 answered**, **G2 part 1 done**, **G1 instrument shipped**, **C2's rank-IC half done**. O2 and G3 confirmed still open. |
| `plans/2026-08-09-master-execution-order.md` | Status column on every row; D1/D2/D3 carry their answers; "if you only do four things" replaced with the four that remain. |
| `plans/2026-08-09-training-cost.md` | 47 boxes ticked, banner added. Task 7 Step 4 left open — it is the one step that did not land. |
| `plans/2026-08-09-label-integrity.md` | Tasks 1–3 ticked, Task 4 left open, the four pre-flight rulings from the SDD ledger surfaced into the plan. |
| 7 closed plans | Completion banners with commit evidence + boxes ticked: deterministic-backtest, served-forecast-surface, minimal-model, remove-accidental-retrain-work, market-relative-labels, friction-conditioned-tier-scoring, price-history-source-import. |
| `plans/2026-08-03-direction-prior-correction.md` | Tasks 1–4 ticked; Tasks 5–7 marked **NEVER BUILT** at the heading, since Task 4's gate returned STOP. |
| `plans/2026-08-07-data-organization-next-steps.md` | Item 1 marked shipped; items 2–6 re-verified as open. |
| `plans/2026-08-08-per-fold-price-filter.md` | Answer recorded: the +3.50pp **does not survive** re-derivation. |
| `research/2026-08-07-next-steps.md`, `research/accuracy-opportunities.md`, `references/catalog-build.md`, `README.md`, `specs/2026-08-01-...-design.md` | Superseded-by pointers and per-item corrections. |

## Five things the audit found that were not just missing checkmarks

1. ~~**`VOTED_CACHE_VERSION` is 5; the CI cache key is still `voted-v4-`.** Task 2 bumped the
   constant without moving `price-forecast.yml:115-116`. The consequence is a permanent cache
   **miss**, not a stale hit — the cached parquet's filename embeds the version — but the next
   bump must move the key **v4 → v6**, not v5 → v6. This is exactly the sequencing hazard the
   master order lists first.~~ **Resolved on `label-integrity`**: the key moved straight to
   `voted-v6-` (`.github/workflows/price-forecast.yml:115`) when `VOTED_CACHE_VERSION` bumped to
   6, leaving no v4 or v5 key behind. Do not re-file this.

2. **The Steam soft-block (blocker 5d) was a false positive**, and nothing downstream was
   updated. `backfill_steam_listing_history.py:67` still fetches the `market_hash_name` URL, whose
   normal 230 KB SSR shell the block detector at `:271` reads as a block. Three roadmap items
   (`R11`, `R13`, `5c`) were filed as blocked on a blocker that does not exist.

3. ~~**The composition-stability write-up is a stub.** `research/2026-08-09-composition-stability.md`
   stops at a `<!--RESULTS-->` placeholder. **D3 is therefore not answered**, even though the number
   exists.~~ **Resolved on `label-integrity`**: `f833882` committed the results, and the number did
   point at the opposite branch from the one every plan predicted — **D3 rules Branch A**, the
   signal survives composition control. Re-measured post-`873148b` in `3c62a49`; every cell moved
   ≤0.0015 and the ruling held.

4. **The orphaned-outcomes problem got worse, not better.** The canonical
   `ops/forecast_outcomes.parquet` has 11,084 of 48,241 rows carrying a slug — **77% NULL**,
   against the 30% the plan was written against. The row count also fell from 104,642, so the file
   was rewritten between readings and the two numbers are not comparable.

5. **The `+3.50pp` withdrawal never propagated.** ~~`architecture/model.md:210,533` still ship it
   as settled evidence~~ — **fixed in `895005a`**; both sites now record the withdrawal and the
   null re-derivation. The second half stands: `:189-217` / `:529-537` still describe the 99-item /
   100K-row config as production against a shipped default of 926 items at 1.2M rows. That remains
   **O2**.

## Not done here

**`architecture/model.md` and `architecture/model-optimization.md` were not corrected.** They
carry roughly 25 stale factual claims between them — the training defaults in eight places,
`MODEL_ARTIFACT_VERSION` 5 vs 6, retrain timings from a superseded config, `VOTED_CACHE_VERSION`
v3 vs 5, early stopping described as live, and a `SKIP_REGIMES` claim that contradicts the same
file 470 lines earlier. That is a rewrite of two reference documents, not a status pass, and it is
tracked as **O2** rather than absorbed here.

Changelog entries were also left untouched by design: they are dated records, and this repo's
convention is that they are never edited to match later reality.
