# The archive migration silently emptied eight A/B harnesses, and the three refutations re-derived after the fix all survive

**Date:** 2026-08-08
**Follows:** `docs/changelog/2026-08-08-embargo-and-harness-hygiene.md` and
`docs/changelog/2026-08-08-embargo-discontinuity-measured.md` — this is the third entry of the
same day. It closes the prior entry's "Still open" item *"no harness was run"* **partially**:
three of thirteen.
**Commits:** `e930850` (`backend/scripts/ab_test_{csfloat_basis,direction_labels,`
`feature_contribution,item_metadata,price_primitives,q50_sampling,training_breadth,`
`volume_features}.py`, `backend/tests/test_ab_harness_universe.py`)
**Suite:** 1,490 pass, 0 failures (was 1,477 at `c0f0349`; `f63ae76` added 2 and `e930850`
added 11)

Two things landed. A defect the 2026-08-08 archive migration introduced into eight A/B
harnesses — it emptied their item universe outright — and, after the fix, the first re-run of
three of them. Everything measured below is an **offline harness number** on the ≥$1 cohort
(`MIN_SERVED_PRICE_USD = 1.0`); none of it is production DA and none of it is comparable to
the walkforward gate figures in the sibling entry.

## 1. Schema presence was a proxy for era, and the migration retired the proxy

The archive migration the prior entry listed as outstanding has now run: all 21
`price-archive/prices-*.parquet` files carry an mtime of **04:38 today**, thirteen pre-2026
(`prices-2013` … `prices-2025`) and eight 2026 (`prices-2026-01` … `prices-2026-08`).
`normalize_price_schema.py` materialises `source` on the thirteen pre-2026 files as a **typed
all-NULL column**. That is the correct migration. `source IS NULL` is what "the pre-2026
series" means, and `db/archive.py::prices_relation`'s docstring says so in as many words.

What it broke is eight `ab_test_*` harnesses that decided *whether* to filter rows on `source`
by asking whether the column **exists**:

```
if "source" in cols:   ->  WHERE source = 'aggregator_sync' AND {_UNIVERSE}
else:                  ->  WHERE {_UNIVERSE}          # take the file whole
```

Those were the same question only until the migration. Afterwards every pre-2026 file takes
the filter branch, `NULL = 'aggregator_sync'` evaluates to NULL rather than false, and thirteen
years of prices contribute nothing. The universe query then requires `MIN(day) < DATE
'2026-01-01'`, which no surviving item can satisfy — so **the universe is empty**.

Measured on the real archive with the harnesses' own `phase_collapsed_sql_filter()` and
recorded in `e930850`'s message: **0 items before the fix, 876 after**. (This entry cites that
measurement rather than repeating it.) For scale, `ab_test_training_breadth`'s universe query
carries `LIMIT N_UNIVERSE = 870` and its guard requires `N_EVAL_ITEMS + N_WIDE = 850`; the
post-fix run logged `Universe: 870 deep >=$1 items with >=180 days`, i.e. capped by the LIMIT.

**It surfaced as three unrelated-looking failures, and not one of them said "zero rows":**

| harness | what it raised |
|---|---|
| `ab_test_training_breadth` | `IndexError: list index out of range` |
| `ab_test_item_metadata` | `_duckdb.ParserException` on `WHERE item_slug IN ()` |
| `ab_test_csfloat_basis` | a stale frame-cache fingerprint refusal |

The fix is `WHERE source = 'X'` → `WHERE (source IS NULL OR source = 'X')`. **The `if "source"
in cols` branch is kept**, because a file that genuinely lacks the column cannot be queried for
it — but the branch is now about *projection safety*, not about which era the file belongs to.
That distinction is the whole bug.

Applied to all eight, not only the three being re-run:

| filtered value | harnesses |
|---|---|
| `aggregator_sync` | `csfloat_basis`, `item_metadata`, `training_breadth`, `price_primitives`, `volume_features` |
| `STEAMCOMMUNITY` | `feature_contribution`, `direction_labels`, `q50_sampling` |

## 2. The `STEAMCOMMUNITY` three are a separate and worse finding

That value appears **nowhere in the archive** — 0 rows across all 21 files, which hold 13
distinct `source` values (measured in `e930850`). So those three harnesses were **already
reading pre-2026 data only, before the migration, silently**: their 2026 files matched nothing
and their pre-2026 files fell through to the `else` branch that takes the file whole.

The NULL-safe form therefore **restores exactly their prior behaviour** rather than changing
what they measure. It does not fix the dead constant, and that was deliberate: deciding what
`feature_contribution`, `direction_labels` and `q50_sampling` *should* read is a question about
their design, not about NULL semantics, and guessing at it would silently change three
instruments in the same commit that was fixing a different silent change. Left open in §7.

`walkforward_backtest.py::_load_parquet_items` carries the same dead constant at line 101,
reachable only under `backfilled_only=True`. Nothing in the served path sets that (§3), but
anyone who flips it will select zero items.

## 3. The walkforward gate is not affected, so this morning's prod rows stand

Verified by reading `scripts/walkforward_backtest.py`:

- its `"source" in cols` sniff only **projects** the column, with an explicit
  `NULL::VARCHAR AS source` fallback in the `else` branch (lines 89–92). It never filters on it
  there.
- `conds.append("source = 'STEAMCOMMUNITY'")` sits behind `if backfilled_only:` (line 101), and
  `main()` calls `_load_parquet_items(con, backfilled_only=False)` (line 373). The branch is
  dead in the served path.

So the four `lgbm-v4-embargoed` rows written to production earlier today, and the
purged-vs-unpurged discontinuity contrast in
`docs/changelog/2026-08-08-embargo-discontinuity-measured.md`, are unaffected by this bug and
by its fix.

## 4. The empty-universe guard existed; it fired three lines too late

`ab_test_training_breadth._build_frame_uncached` already had the right diagnostic — a
`SystemExit` naming the item count and the two knobs to change:

```
Universe has {len(slugs)} items but the design needs {N_EVAL_ITEMS + N_WIDE}
(N_EVAL_ITEMS + N_WIDE). Lower MIN_ITEM_DAYS or N_WIDE.
```

Three lines **above** it, a `logger.info` dereferenced `rows[0][1]` and `rows[-1][1]` to print
the days-per-item range. On an empty universe the `IndexError` fired from inside the *logging*
and the message that would have explained everything never ran. The guard now runs first and
the log line after it.

This is the second defect in this pass whose cost was diagnostic rather than numerical, and it
is the reason the failure looked like three unrelated bugs instead of one.

## 5. Three of thirteen harnesses re-run; all three refutations survive

`training_breadth`, `item_metadata` and `csfloat_basis` were re-run from `backend/` after the
fix. All three exited 0 (`EXIT_BREADTH=0 13:28:32`, `EXIT_METADATA=0 13:35:10`,
`EXIT_CSFLOAT=0 13:39:07`; 5m53s + 6m38s + 3m57s, **16m28s** in total). These are the three
published refutations that `2026-08-07-paired-mde-fold-clustering.md` and both sibling entries
left awaiting re-derivation — the first A/B numbers in this repo produced under the wider
`H + 13` embargo, the `BID_SOURCES` / phase-collapsed universe filter, and fold-clustered
intervals at once.

**All three survive.** Every figure below is the **held-out**, paired, fold-clustered contrast
on the ≥$1 cohort. `null` throughout means **unresolved at this design's power**, not "no
effect"; the fold counts and MDEs are given so that distinction is checkable.

### CSFloat basis — null at every horizon, on the weakest design of the three

`basis_all_vs_placebo`, held-out:

| h | mean diff | 95% CI | MDE | folds |
|---:|---:|---|---:|---:|
| 3 | −0.05pp | [−1.26, +1.12] | 1.19 | 13 |
| 7 | −0.16pp | [−1.57, +1.23] | 1.40 | 12 |
| 14 | +0.34pp | [−2.81, +2.57] | 2.69 | 12 |
| 30 | −1.68pp | [−5.63, +1.75] | 3.69 | 12 |

Every interval crosses zero. The standing refutation holds.

**This is the weakest design of the three and could not have resolved anything but a very
large effect.** 80 eval items, 190 train items, 512,106 rows after the `2020-04-01` min-date
cut, 12–13 folds, and MDEs across arms running **up to 4.58pp at 30d** (`count_only`). A real
1pp basis effect would be invisible here.

**A blemish worth recording:** the **14d placebo** reads **−0.47pp [−0.93, −0.05]**, excluding
zero. A permuted arm should not do that. It is the only placebo cell in this run that does, and
it means the **14d column of this harness is unreliable** — the design is producing a
significant result from a permutation at that horizon, so a significant result from a real arm
there would not be believable either. Placebo is clean at 3d, 7d and 30d
(+0.14 [−0.31, +0.53], −0.20 [−0.48, +0.04], −0.36 [−0.91, +0.12]).

### ByMykel item metadata — the standing refutation holds; item age is the one live thread

`treatment_vs_placebo`, held-out — null at all four:

| h | mean diff | 95% CI | MDE | folds |
|---:|---:|---|---:|---:|
| 3 | +0.20pp | [−0.57, +1.07] | 0.82 | 26 |
| 7 | +1.18pp | [−0.34, +3.17] | 1.75 | 26 |
| 14 | −0.34pp | [−1.62, +0.84] | 1.23 | 25 |
| 30 | +1.18pp | [−1.01, +3.38] | 2.19 | 25 |

`placebo` itself is clean at all four (−0.03, −0.35, −0.23, +0.25pp, every interval spanning
zero), so the null is a null and not a dead instrument.

**The one live thread: `age_only` reads positive at three horizons.**

| h | `age_only` vs baseline | 95% CI |
|---:|---:|---|
| 3 | **+0.46pp** | [+0.17, +0.78] |
| 7 | +0.75pp | [−0.07, +1.80] |
| 14 | **+0.60pp** | [+0.08, +1.18] |
| 30 | **+1.60pp** | [+0.12, +3.04] |

**This is measured against `baseline`, not against `placebo`, so it is NOT yet a result and
must not be reported as one.** `treatment_vs_placebo` is the contrast this harness treats as
its verdict, for the reason its own docstring gives — an added column buys capacity whether or
not it carries signal. Reading `age_only` against baseline is the reading the placebo arm
exists to discipline. It is flagged here as the single thing in this pass worth following up,
with the follow-up being an `age_only`-vs-placebo contrast the harness does not currently
compute.

Note what it is *not*: `docs/changelog/2026-08-06-breadth-beats-depth-item-age-does-not.md`
already **withdrew** its item-age refutation in an amendment the same day, so a positive
`age_only` does not contradict a live claim. What it does do is put a positive item-age reading
under **raw** labels, where that entry's raw-label reading was sign-inconsistent (+0.73pp at
7d, −0.69pp at 30d for age's marginal contribution over `static_only`).

### Training breadth — positive at 14d only, where all three arms agree

Paired vs `narrow`, held-out:

| h | mid (350) | wide (700) | wide_unbudgeted | folds |
|---:|---|---|---|---:|
| 3 | +0.42 [−0.53, +1.76] | +0.77 [−0.44, +2.47] | +0.33 [−0.44, +1.36] | 26 |
| 7 | −0.72 [−2.41, +0.38] | −0.93 [−2.94, +0.38] | −0.45 [−1.19, +0.19] | 26 |
| **14** | **+2.33 [+0.44, +4.74]** | **+2.38 [+0.55, +4.77]** | **+1.58 [+0.32, +3.59]** | 25 |
| 30 | −0.04 [−1.74, +2.11] | −0.10 [−2.13, +2.12] | −0.52 [−1.34, +0.19] | 25 |

MDE 0.69–2.15pp. `narrow` absolute DA: 53.85 / 53.01 / 58.28 / 52.35%.

**The 14d row is the result, and all three arms agreeing is what makes it one** — a single
positive cell among twelve would be a multiplicity artifact. **The saturation qualifier
survives and strengthens:** 150 → 350 items buys +2.33pp; 350 → 700 adds +0.05pp on top.

**Two claims from 2026-08-06 did not survive.** That entry read breadth as positive at *both*
14d and 30d (mid +0.73 [+0.16, +1.36], wide +1.06 [+0.46, +1.70] at 30d); **30d is now null**.
And its "decisive column" argument — that `wide_unbudgeted` is *worse* than `wide` at 7d, 14d
and 30d despite 5.8× the rows, so what pays is item diversity and not row count — rested on
`wide_unbudgeted` reading −0.59 [−0.96, −0.22] at 14d. It now reads **+1.58 [+0.32, +3.59]**, a
sign flip into significance. `wide_unbudgeted` is no longer significantly worse than `narrow`
at any horizon. The point-estimate ordering `wide > wide_unbudgeted` still holds at 14d
(+2.38 vs +1.58), so the *conclusion* is not overturned; the evidence for it is much weaker
than the 2026-08-06 table made it look.

**The budgeted arms are not equal-rows and never were.** At 3d the re-run logs 131,137 /
119,364 / 124,321 rows/fold for narrow / mid / wide against a nominal `ROW_BUDGET = 200_000`,
because `_stratified_sample` caps at `budget // len(items)` per item and items short of that
quota contribute everything they have. The 2026-08-06 run had the same property and a wider
spread (~213K / ~172K / ~185K implied by its published rows/item). Row totals are ~40% lower
than then; this pass did **not** separate how much of that is the embargo trimming train rows
from how much is the changed item universe.

### The `trained` column is systematically more flattering, and is not a result

`csfloat_basis` and `item_metadata` each report a `trained` figure beside the `heldout` one —
the same arms scored on a slice of their own training items. It is systematically kinder:

| contrast | held-out | trained |
|---|---:|---:|
| metadata `treatment` @ 30d | **+1.43pp** [−0.63, +3.61] — null | **+5.25pp** [+1.79, +8.89] — positive |
| metadata `treatment_vs_placebo` @ 7d | +1.18pp [−0.34, +3.17] — null | +2.58pp [+0.99, +4.60] — positive |
| csfloat `basis_all` @ 30d | −2.04pp [−6.01, +1.28] | −4.65pp [−10.80, +0.19] |

In two of the three the trained column **changes the verdict**. **Anything quoted from the
`trained` column is not a result.** `training_breadth` does not have this column at all — it
reports one `dir_acc_strict_ge1` per arm on held-out items only — so the two-column caveat
applies to the other two harnesses, not to all three.

### These three do not use the new verdict helper, and their own labels are wrong

None of `csfloat_basis`, `item_metadata` or `training_breadth` calls `verdict()`,
`format_paired()` or `paired_arm_contrasts()`. All three import only `paired_da_difference`.

This is **not** a contradiction of `2026-08-08-embargo-and-harness-hygiene.md` §5 — those three
were fixed on 2026-08-07 and were never among the ten that entry covered — but the helper now
exists, they do not use it, and so **every `positive` / `null` verdict recorded above was
derived by hand from the stored interval bounds**. That is a manual step in front of numbers
that are about to be cited.

Relatedly, all three print **`(paired, dates clustered)`** / **`paired vs narrow (dates
clustered, held-out items)`** while passing `cluster_key="fold_id"` and while their stored JSON
records `"cluster_key": "fold_id"` with `n_clusters` of 12–26 against `n_dates` of 252–546. The
printed label is a leftover from the pre-`180c426` date-clustered estimator and now says the
opposite of what the code does. Anyone reading a console transcript rather than the JSON will
read the interval as date-clustered, which is exactly the mistake
`2026-08-07-paired-mde-fold-clustering.md` was written to prevent.

## 6. What was deliberately not done

- **The other ten harnesses were not re-run.** Five got the same NULL-safe fix
  (`price_primitives`, `volume_features`, `feature_contribution`, `direction_labels`,
  `q50_sampling`) and five were never source-filtered, but none of the ten has been executed
  since the embargo, the universe filter or the migration. Every A/B result they have stored in
  this repo still predates all three.
- **The dead `STEAMCOMMUNITY` constant was left in place** in three harnesses and in
  `walkforward_backtest.py`'s unreached branch. Choosing a replacement changes what those
  instruments measure; that is a separate decision and guessing at it inside a NULL-semantics
  fix is how the original defect happened.
- **The artifacts were not committed.** `ab-breadth.json`, `ab-metadata.json`,
  `ab-csfloat.json` and the combined run log live in a session scratchpad outside the repo.
  **Every number in §5 is therefore reproducible only by re-running the three harnesses**
  (16m28s in total), not by re-reading a stored file in this repository.
- **The re-runs' configuration flags are not recorded in the log.** `ab-csfloat.json` stores
  `market_relative: false`; nothing in the transcript states whether `--no-early-stop` was
  passed to `item_metadata`. The 2026-08-06 entry publishes both configurations, and the
  qualitative comparisons in §5 were checked to hold against either — but a like-for-like
  cell-by-cell diff against that entry is **not** fully determined by what was captured.
- **`verdict()` was not retrofitted into the three**, and the stale "dates clustered" print
  labels were not corrected. Both are code changes to harnesses that had just been re-run, and
  changing them now would mean the stored artifacts no longer match the source that produced
  them.
- **No `age_only`-vs-placebo contrast was computed.** The harness does not emit one and adding
  it is a code change, not a reading of this run.
- **No archive query was re-executed for this entry.** The 0 → 876 figure and the "0 rows,
  13 distinct sources" figure are cited from `e930850`'s commit message, which is where they
  were measured.

## 7. Verification

- **Suite: 1,490 pass, 0 failures.** The delta from the 1,477 recorded by
  `2026-08-08-embargo-and-harness-hygiene.md` spans two commits: `f63ae76` added 2
  (`TestTheGateCanActuallyPersist`) and `e930850` added **11**.
- **`backend/tests/test_ab_harness_universe.py`, +11 cases in two groups**, both of which
  **failed first for the right reason**:
  - a `migrated_archive` fixture building a two-file archive in the **post-migration shape** —
    a pre-2026 file *with* an all-NULL `source` column beside a 2026 file with a real one — and
    one behavioural case per re-run harness (3) asserting the pre-2026 row survives
    `_archive_union_sql`. Pinned behaviourally rather than by reading the source precisely
    because none of the three real failures said "zero rows".
  - a family-wide static guard (8, parametrised over `SOURCE_FILTERED`) that any `source = '…'`
    filter carries a NULL-safe clause. **Its regex deliberately anchors on `source = '` and not
    on `WHERE source = '`**: after the fix the clause reads
    `WHERE (source IS NULL OR source = '…')`, so the tighter anchor would have made the test
    pass by failing to match — a green test proving nothing.
- **The three re-runs exited 0** with the fold counts, item counts and intervals in §5.
- **`walkforward_backtest.py` was read, not run**, to establish §3.

One incidental observation from the run log, recorded but not diagnosed: the `training_breadth`
process logged `sqlalchemy.pool.impl.QueuePool - ERROR - Exception during reset` with
`psycopg2.OperationalError: SSL SYSCALL error: EOF detected` at teardown (13:28:31), after all
results were written, and still exited 0. It is the same pooler-drop symptom as the walkforward
persistence bug in the sibling entry, here at interpreter shutdown rather than at a write.

## 8. Still open

- **Ten of thirteen harnesses have not been re-run.** The prior entry's "no harness was run"
  item is now partially closed, not closed.
- **The three re-run harnesses' docstrings still cite the `+12.1pp unpurged → +6.1pp purged`
  pair** (`ab_test_training_breadth.py:53`, `ab_test_csfloat_basis.py:42`,
  `ab_test_item_metadata.py:51`).
  `docs/changelog/2026-08-08-embargo-discontinuity-measured.md` §3 retired that imported figure
  in favour of a locally measured **+10.15pp at 30d**. Three docstrings now carry a number the
  project has withdrawn.
- **`age_only` is positive against baseline at 3d, 14d and 30d** and has no placebo contrast.
  The single follow-up worth running out of this pass.
- **The CSFloat 14d placebo excludes zero.** Until that is understood, no 14d number from that
  harness is citable.
- **The dead `STEAMCOMMUNITY` constant** in three harnesses and in
  `walkforward_backtest.py:101`.
- **`verdict()` is not wired into the three re-run harnesses**, and their console output still
  claims date clustering.
- **The `paired_mde` run-to-run divergence is still unexplained** (~0.155pp between identical
  commands), and applies to every interval in §5.
- **Whether folds are independent.** 12–26 clusters, adjacent expanding windows sharing most of
  their training data; fold clustering is an improvement, not a proof.

## 9. Related

- `docs/changelog/2026-08-08-embargo-and-harness-hygiene.md` — the instrument change these
  three harnesses are the first to exercise, and the "Still open" item this partially closes
- `docs/changelog/2026-08-08-embargo-discontinuity-measured.md` — the gate run §3 shows this
  bug did not touch, and the source of the +10.15pp that retires the docstring figure
- `docs/changelog/2026-08-07-paired-mde-fold-clustering.md` — the fold clustering behind every
  interval in §5, and the estimator the stale print labels contradict
- `docs/changelog/2026-08-06-breadth-beats-depth-item-age-does-not.md` — the entry the breadth
  and metadata figures re-derive; its 30d breadth positive and its `wide_unbudgeted` argument
  are the two claims §5 weakens
- `docs/changelog/2026-08-07-archive-schema-and-keys.md` — `prices_relation` and the
  `source IS NULL` convention the eight harnesses had diverged from
- `docs/changelog/2026-08-08-phase-collapsed-names-dropped.md` — the other rule the re-run
  universes now apply

## 10. Docs touched

This entry; `backend/AGENTS.md` (the A/B verdict bullet's "no harness has been re-run" clause,
now stating three of thirteen and that the results are not stored in the repo); and
`docs/research/2026-08-07-next-steps.md` ("Open questions", the three-refutations item, now
recording that all three were re-derived on 2026-08-08 and survived). Nothing under
`docs/architecture/` moved.
