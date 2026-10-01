# Deep-review §11: MDEs beside the verdicts, the `paired_mde` nondeterminism found, the age placebo run

2026-10-01. Item 20 of `research/2026-09-28-next-steps.md`, against §11 of
`research/2026-08-19-deep-model-review.md`.

## 1. MDE beside each stored verdict; underpowered nulls relabelled

`experiment_log.csv` gains an `mde` column: the half-width of the paired interval, which is what
`backtest/paired_mde.py` reports as `mde`, in the row's metric units. Per-horizon values are
written `3d/7d/14d/30d`. Values come only from the cited notes. Where a row already stored
`ci_lo`/`ci_hi` for a paired interval, `mde` is computed from them.

Underpowered nulls are relabelled **`inconclusive`**, the log's existing verdict, rather than a new
`unresolved`. In `paired_mde.verdict()`, `unresolved` means "no interval could be computed", which
is a different claim from "an interval exists and is too wide".

| Row | Before | After | Why |
|---|---|---|---|
| `bymykel-metadata` | refuted | **inconclusive** | MDE 2.29/2.01/5.80/10.07 at 8–9 folds, and the 7d placebo is positive (+1.20 [+0.44, +1.96]) |
| `csfloat-basis` | refuted | refuted | Holds at 3d/7d only (MDE 1.19/1.40). The 14d placebo excludes zero and 30d sits under its MDE; the estimate cell now says so |
| `volume-features` | refuted | refuted | Best-powered null in the repo (MDE 0.20–0.46) |
| `supply-side-rarity` | refuted | refuted | MDE 0.17–0.47, and the placebo ≥ treatment |
| `served-cohort-weighting` | refuted | refuted | No MDE was stored. It refutes only the preregistered +2pp, and the cell says so |

New rows, for §11 arms that had none: `quality-spread` and `price-primitives` (inconclusive: no
test, and an MDE of 2.21–3.69 respectively), `scale-free-model-fixes-accuracy` (inconclusive, MDE
4.11/3.37/2.92/3.52), `age-only-item-metadata` (see §4), `embargo-discontinuity` (measured; h7
+4.29 sits inside its 4.70 MDE), `cross-sectional-group-removal` (refuted: the interval excludes
the claimed +3.5 only) and `event-group-removal` (inconclusive, MDE 4.59).

Not changed: `training-breadth-neutral` stays `measured`. Its note says every effect is below its
MDE but never states the MDE, so "breadth is free" is unverified at an unknown power. That isn't
enough to relabel it.

## 2. The `paired_mde` frame nondeterminism: found and fixed

The 08-07 observation (`mean_diff_pp` −0.1581 → −0.0026, `n_paired` 468,759 → 469,359) was
never diagnosed. Reproduced today: two separate processes built `walkforward_backtest`'s h=3
frame (`max_items=300`, events frozen) and got **683,461 vs 683,477 rows**. The items were
identical, and the price rows were the same multiset in a different order.

**Mechanism.** `_load_all_prices` orders by `(item_slug, day)` only, and 716,572 of 1,043,039
item-days have several source rows, which DuckDB returns in any order. `engineer_features`
collapses them with a plain mean, whose last bit follows summation order: 320,359 daily means
differed, by at most 7.3e-12. `stale_run_days` compares prices bit-for-bit, so 45 frozen-run
flags flipped, and frozen-run label voiding moved the frame. Production's vote is unaffected. It
takes a `median` (order-free), and its 2σ cut already carries `VOTE_TIE_RTOL` for the same
reason (`changelog/2026-09-28-voting-moved-into-duckdb.md`).

**Fix.** `engineer_features` sorts by `(item_id, date, price, volume)` before the collapse, but
only when duplicate item-days exist. The voted path has none and is unchanged. Two builds now
hash identically. The fixed frame has 683,412 rows: genuinely frozen quote runs are now bit-equal
on every build, so they are voided every time rather than at random. Every harness that skips
the vote (`walkforward_backtest`, `compute_mde`, the paired gates) shifts by a few dozen rows
from its pre-fix frames.
`tests/test_engineer_features_row_order.py` fails without the fix.

⚠️ The 08-07 pair was also not the "identical command" it was recorded as: `2f061ea` (bid
exclusion, explicit universe) landed 10 seconds before the fold-clustering fix. Both causes may
have contributed. Only this one can be reproduced.

## 3. Stale console labels — and the move that broke the harnesses

The three harnesses that printed `(paired, dates clustered)` while passing `cluster_key="fold_id"`
now print `folds clustered`.

Fixing them turned up a larger defect. `fd68af8` (09-15) moved these scripts into
`scripts/archive/` without re-anchoring their paths. Every `Path(__file__).parent.parent` that
meant `backend/` now resolved to `backend/scripts/`. `ARCHIVE_DIR` pointed at a
`backend/price-archive` that does not exist, and the `sys.path` inserts broke direct invocation
(`python scripts/archive/compute_mde.py --help` failed). All 67 affected files are re-anchored (the two panel builders are left to #83, which deletes them) to
`Path(__file__).resolve().parents[2]`, and every resolved target was checked to exist. Item 17
(#76) fixed the cited paths but not these.

## 4. The capacity-matched `age_only` placebo

`ab_test_item_metadata.py` gains an `age_placebo` arm: the same two columns as `age_only`
(`item_age_days`, `rarity_meta_rank`), permuted the way `placebo` permutes its nine. It also gains
an `age_only_vs_age_placebo` contrast. The existing `placebo` carries nine columns, so it could
not say whether two real columns beat two columns of noise. Run 2026-10-01 on the local archive,
production's trainer, raw labels, 25–26 folds, after the §2 fix. Held-out ≥$1 items, pp:

| h | `age_only` − baseline | 08-08 (pre-trainer-fix) | `age_only` − `age_placebo` | MDE |
|---:|---|---:|---|---:|
| 3 | +0.16 [−0.28, +0.55] | +0.46 | +0.11 [−0.29, +0.52] | 0.41 |
| 7 | +0.80 [−0.14, +1.87] | +0.75 | +0.62 [−0.24, +1.68] | 0.96 |
| 14 | −0.47 [−1.59, +0.47] | +0.60 | −0.62 [−1.84, +0.53] | 1.18 |
| 30 | +0.77 [−1.61, +2.51] | +1.60 | +0.89 [−1.52, +2.69] | 2.10 |

`age_placebo` itself is clean at all four (+0.05 / +0.18 / +0.15 / −0.12, every interval spans
zero).

**Verdict: refuted, and the thread is closed.** The three positive cells from 08-08 don't
reproduce on the honest trainer even against baseline. Against the matched placebo, no held-out
horizon resolves. That is the pattern `ab-statistics` warns about: an arm that read positive under
early stopping is the suspect case. On the trained cohort, one cell of eight clears zero (h=3,
+0.61 [+0.03, +1.19]). With eight looks that is about what chance produces, and h=14 and h=30 go
the other way or span zero. ⚠️ h=30's MDE is 2.10pp, so an effect of that size is not excluded
there. h=3's 0.41pp is the reading this rests on.

## 5. The 1.1607 Steam fee constant: blocked

`backend/runtime/steam_listing_history.db`, which the review found on disk at 98 MB, is gone: it
is not anywhere under the home directory. It was gitignored local state and nothing records its
deletion. Re-validating the constant means re-scraping the 262 items' Steam price history with a
logged-in session (`scripts/archive/backfill_steam_listing_history.py`), which is a decision for
the owner, not part of this item.
