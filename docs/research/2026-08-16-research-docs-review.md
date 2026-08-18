# Research docs review — 32 docs, 9,548 lines

**Date:** 2026-08-16
**Scope:** every file in `docs/research/`. Four parallel read-only passes (foundational, the
next-steps chain, model/data, preregistrations + recent), plus index and reference-integrity
checks run directly.
**Status:** review only. **No doc was edited.** Every finding below was verified against the
filesystem or git by hand — three subagent claims were checked and refuted, and are recorded in
§9 so they don't get re-raised.

## Bottom line

The docs are unusually well self-audited — the corrections-banner convention works, and all 10
preregistrations have recorded outcomes, which is rare. Two structural problems:

1. **Both designated entry points now mislead a new reader**, each still presenting a headline
   result that later work killed.
2. **A named product-level defect (R13) fell out of tracking on 2026-08-09 and is still unfixed
   in the code.** Nothing closed it. It just stopped being carried forward.

## 1. R13 — the cohort inversion was silently dropped 🔴

`research/2026-08-07-next-steps.md:782` calls it *"the review's most consequential product fact"*:

| Cohort | ≥$1 | <$1 | % ≥$1 |
|---|---:|---:|---:|
| All items with recent data | 26,468 | 14,955 | **63.9%** |
| Items actually forecast | 1,423 | 7,268 | **16.4%** |

The archive is two-thirds dollar-plus; the served cohort is **84% sub-dollar** — precisely the tier
carrying the 35.5% spread (`backtest/friction.py`) and the 37–42% carry-forward rate. The review's
own words: *"a larger lever than any feature, and it is a data-plumbing problem, not a modelling
one."*

**Last tracked 2026-08-09.** It appears in `2026-08-07-next-steps.md`, `2026-08-09-next-steps.md`,
`2026-08-07-cs2-forecasting-research.md`, `2026-08-09-model-and-data-research.md` and two
changelogs — and then vanishes. It is in **none** of the 08-10, 08-13, or 08-14 lists, and no doc
or changelog closes, refutes, or defers it.

**Verified still unfixed in code (2026-08-16):** `backend/database.py:105` gates serving on
`Item.is_backfilled == 1`, exactly as R13 describes, and **`MIN_SERVED_PRICE_USD` does not exist
anywhere in the backend.** The `FLOOR_SWEEP` half that R13 says was already instrumented never
had its decision taken.

This also makes R13 the direct parent of today's `2026-08-16-listing-count-floor.md`: that doc
measured a *listing-count* floor and found the threshold doesn't transfer, while the *price* floor
— the one already instrumented, and the one with a measured 35.5%-spread justification — has been
sitting undecided for a week.

**Two smaller drops** from the same doc, same pattern: **item 8** (hedonic market index,
`2026-08-07-next-steps.md:577`, "NOT STARTED") and **R14** (mechanical supply-position features,
`:813`). Neither carried forward, neither closed.

## 2. The live action list is stale on its own #1 item 🔴

`docs/README.md:173` calls `2026-08-14-next-steps.md` "⭐ the live action list." Its top-ranked,
cheapest, "single best untested idea" — item 1, the `tier × post` natural experiment — **failed on
2026-08-15**: `changelog/2026-08-15-armory-tier-post-fails-and-fails-placebo.md` reports the wrong
sign on knives/gloves (t=2.55) *and* a failed placebo gate (the placebo diverges more than the
event window).

Items 2 and 3 are genuinely still open. But a full week of work now sits only in changelogs and is
reflected in no list: the range-forecaster reclassification, the exceedance-target scope, the
served-outcome feedback calibration, and today's three results.

**A 6th list is warranted** — and `2026-08-15-directional-accuracy-and-data-inventory.md` has
already effectively written it, proposing `P(|return|>cost)` as the top priority, which 08-14 does
not contain at all.

## 3. Both entry points mislead 🔴

**`2026-08-07-cs2-forecasting-research.md`** — README calls it "the best analysis and literature
review in the repo."

- It still presents **C1 / cross-sectional reversal** (`:652`, `:668`) unqualified as *"the
  project's strongest measured predictor,"* which *"beats the model at all four horizons"* and
  *"survives composition control."* Refuted since:
  `changelog/2026-08-13-c1-refutation-survives-the-audited-anchors.md:9` — "FAILS at 3 of 4
  horizons. C1 stays shelved." Its banner stops at 2026-08-09 and never mentions this. This is the
  doc's most-cited positive result.
- **Its own corrected headline is itself invalid.** The 2026-08-09 banner withdraws the accuracy
  number and substitutes `mean_rank_ic 0.1307 vs naive −return_1d 0.1656` as the trustworthy
  figure — but the 2026-08-11 label-denominator finding (README:19) concludes **"No stored rank
  IC, DA or −return_1d comparison in this repo is safe to rank arms on."** A correction that
  corrected itself into invalidity.

**`2026-08-09-model-and-data-research.md`** — README: "Start here for anything accuracy-related."
Fit for that role only **with caveats it doesn't carry.** It predates the range-forecaster pivot,
the DA findings, and the label-denominator result, and its §3 reads as if a ranking/directional
objective is the fix. That thesis is closed (`2026-08-14-next-steps.md`: "the web doc's 'pivot to
a ranker' thesis is closed"). **None of the seven model/data docs mentions the range-forecaster
reclassification at all** — a reader starting here concludes directional objectives are the
frontier, which is the opposite of the project's own verdict.

## 4. Every stored A/B number predates the harness repair 🟠

`changelog/2026-08-13-harness-family-repinned-off-steamcommunity.md`: three harnesses matched 0
rows on `source='STEAMCOMMUNITY'`; six more never applied the production feature allowlist. They
were repaired and **none were re-run** (README:250). Every `ab_test_*`-sourced figure in
`2026-08-07-cs2-forecasting-research.md` — feature-contribution deltas, supply, regime, ensemble
arms — rests on that family, and neither doc flags it.

## 5. `volume-data.md`'s banner inverts its own verdict 🟠

The banner says "the conclusion stands" — volume features refuted. **That is no longer the
verdict.** `changelog/2026-08-15-volume-features-remeasured.md` re-ran the arm on the post-08-08
and post-08-13 fixes and found a real gain at every horizon, placebo-clean:

| Horizon | Treatment − baseline | Placebo |
|---|---|---|
| 3d | **+1.503pp** [+0.631, +2.499] | −0.103 (null) |
| 7d | **+1.879pp** [+0.031, +3.952] | −0.008 (null) |
| 14d | **+1.442pp** [+0.277, +2.538] | +0.186 (null) |

The correct verdict is **feed-blocked, not refuted** — the shelving stands only because the live
feed is dead. A reader stopping at `volume-data.md` gets the opposite conclusion, and this
directly changes the value of the iflow backfill in
`2026-08-16-refutation-power-tiers-and-iflow-backfill.md`, whose `count_in_24` is exactly the
repair.

## 6. Index drift — 13 of 32 docs are unindexed 🟠

`docs/README.md` omits: **all 10 preregistrations**, plus
`2026-08-14-what-moves-skin-prices-web-reconsideration.md`,
`2026-08-15-p-exceed-cost-target-scope.md`, and
`2026-08-16-refutation-power-tiers-and-iflow-backfill.md`.

The preregistrations are the repo's best discipline artifact and they are invisible from the index.
(README's own outbound links all resolve — the drift is one-directional.)

## 7. Preregistration discipline: 10/10 outcomes, 3 moved gates 🟡

Every one of the 10 preregistrations has an explicit scored outcome, either inline or in a
same-day changelog that names the prereg file. **No orphans.** That is the healthiest thing in
this review.

Three, though, changed the bar *after* seeing the result:

| Prereg | What happened |
|---|---|
| `2026-08-13-c1-audited-anchor` | Reproducibility bar (≤0.002) violated at 3/4 horizons; write-up rules the bar mis-specified and substitutes an undeclared pairing check (`changelog/…-c1-refutation-survives…:59`) |
| `2026-08-13-low-level-anchor` | Primary bar (pooled marginal coverage) fails; write-up declares the bar itself "a defect" and pivots to an unregistered h=30-only axis (`…-the-low-level-read-fails-its-axis:38`) |
| `2026-08-13-date-level-sigma-rescaling` | h=14 fails validity (MAE 4.53pp vs ≤3pp) but is reclassified "unrefereeable" rather than failed |

All three self-flag the change, which is honest. But a bar that gets rewritten whenever it binds
isn't a bar — and all three are from the same day, which suggests the bars were being written
faster than they were being thought through.

## 8. Reference rot 🟡

- **`backend/scripts/merge_hf_dataset.py` is cited in 12 docs and no longer exists.** Deleted
  2026-08-10 in `fd49d5d` — a bulk cleanup of `.bak`/`catboost_info` artifacts whose commit
  message is about served-classifier verdicts. It was collateral. The claim it backs is *true and
  still binding* — `ask_volume AS volume` is at line 99 of `fd49d5d^`, which is why
  `volume-data.md` says "never pool across 2026-03-22" — but the citation now dangles. The
  surviving prose reference is `backend/collectors/sales_volume.py:13`.
- **`superpowers/sdd/2026-08-09-label-integrity/task-5-report.md`** cited in
  `2026-08-09-composition-stability.md` — `superpowers/` contains only `plans/` and `specs/`.
  The only confirmed-broken doc link in the whole tree (127 unique refs checked).
- `2026-08-10-training-cost-levers.md`'s **in-file** banner never records that both runs it
  budgets against (1884s, 2306s) are superseded by 996.6s / 17m48s. Only the README index line
  says so, so a reader who opens the file directly treats 1884s as current.

## 9. Checked and refuted — do not re-raise 🟢

Three claims surfaced during this review that do not hold:

- *"`volume-panel.parquet`'s volume dies 2026-04-15."* **No.** That fact applies to the prices
  archive's `volume` column. `volume-panel.parquet` is a separate sidecar: 100% non-zero through
  **2026-06-15** (60,306 rows in June alone, median 19). The wash-trade result stands.
- *"`supply-history.parquet` end-date contradiction, 2024-02-15 vs 2024-02-16."* **No.** The file
  spans to 02-16; `2026-08-16-listing-count-floor.md` reports 02-15 as the end of the *joined*
  frame after the `volume-panel` merge and the `buff_listing_count > 0` filter. Different
  quantities.
- *"`competitor-analysis.md` is undated."* **No** — it carries `2026-07-15` at `:3`. Old, not
  unowned.

## Recommended fixes, in order

1. **Decide R13** — or re-file it explicitly. A week of silence on the repo's self-declared
   most consequential product fact is the finding here, not the item itself.
2. **Write the 6th next-steps list**, with 08-14 item 1 marked failed.
3. **Banner both entry points** — C1-is-dead and no-stored-rank-IC-is-safe on the 08-07 doc;
   range-forecaster-pivot on the 08-09 doc.
4. **Rewrite `volume-data.md`'s banner** to "feed-blocked, not refuted."
5. Index the 13 missing docs; repoint the `merge_hf_dataset.py` citations at `fd49d5d^`; fix the
   `superpowers/sdd/` link; add the 996.6s line to the cost-levers in-file banner.

Items 3–5 are mechanical, ~30 minutes total. Items 1–2 are judgment calls and are the real output
of this review.
