# Is the reversal a return, or a change of measurement basis?

**Measured 2026-08-09** on branch `label-integrity`, against the local Parquet archive
(4,743 calendar days, 4,739 present, 2013-08-14 → 2026-08-08, 38,413 items after the universe
filter). Instrument:
`backend/scripts/measure_composition_stability.py`, committed. No database session is opened
anywhere in it.

This supersedes the composition rows of `docs/research/2026-08-08-model-review.md` §5, which are
marked refuted in place.

## The question

The project's strongest predictor is a reversal: ranking items by `-return_1d` predicts the
forward return with a rank IC near 0.17 at 3d, better than the model that consumes it
(`docs/research/2026-08-08-model-review.md` §4). The archive's consensus price is a median over
whichever sources reported that item that day, so when the reporting set changes between `t` and
`t+h` the measured return partly records a change of *measurement basis* rather than a change of
price. If the reversal survives holding composition still, it is a return and modelling work has
something to reach. If it vanishes, the label is a quoting artifact and no feature or
architecture change can help.

## The instrument

| Choice | What it is |
|---|---|
| Universe | `db/archive.py::prices_relation` with `archive_universe_sql_filter()`, plus production's `historical_fallback:%` exclusion. Never a raw glob. |
| Consensus price | `ItemForecaster._apply_multi_source_voting`, called unbound on the class. Not reimplemented — a reimplementation would drift from the label production actually trains on. |
| Composition, **primary** | The **set of source names** that voted on the item-day, as a bitmask. Built in the script from the same rows the vote consumed. `n_ask_sources` is unchanged and `VOTED_CACHE_VERSION` is untouched. |
| Composition, **secondary** | `n_ask_sources` — the *count* of distinct ask sources (Task 2's column). |
| Stable over `t−1 … t+h` | The composition takes the **same value on every one of those days for that item**, and **every one of those days is present**. A gap is not stable: an absent day is an unobserved composition, not a matching one. |
| Returns | Exact calendar-day lookups on both legs, matching `prepare_targets`. An as-of lookup substitutes a stale price for a missing day, which is the artifact under test. |
| Price floor | `p_t ≥ $1`. Tiers are never pooled. |
| Rank IC | Spearman **within each date**, then the unweighted mean across dates; `t = mean / (sd / √n_dates)`. A pooled Spearman is a different and much larger number because it absorbs the cross-sectional market factor. |
| Reporting floor | `MIN_DATES_TO_REPORT = 30`. A cell below it reads `underpowered` and carries **no number**. |
| Degeneracy guard | `MIN_ITEMS_PER_DATE = 5`. A date needs five items in a cell to contribute an IC; a two-item Spearman is ±1 whatever the data says. |

### Void dates

The script reads the archive directly, so `prepare_targets` never runs and nothing else drops
the dates whose labels it voids. It applies both of production's rules itself, widened to the
whole `t−1 … t+h` window because `-r_t` is itself a return over `(t−1, t]`:

- `_snapshot_dates` — **endpoint** rule. A snapshot at `d` voids anchors `{d+1, d, d−h}`.
- `_collection_shift_dates` — **span** rule. A cutover at `d` voids anchors `[d−h, d]`.

This makes every result below **conservative**: the archive's largest known basis changes are
gone from *every* cell, including "all rows", before composition is partitioned on at all.

## Primary result — 2026, composition = the set of source names

This is the measurement immune to both objections: 2026 is the only era whose rows carry a
source label at all, so nothing rests on what a NULL means, and the set basis sees one source
being swapped for another, which a count cannot.

`--horizon 3 --from 2026-01-01 --basis set`

| Cell | n_dates | n_rows | rank IC | t | verdict |
|---|---|---|---|---|---|
| All rows | 185 | 2,040,460 | **+0.1023** | 14.3 | measured |
| Composition **stable** across t−1…t+3 | 181 | 1,912,872 | **+0.1027** | 14.1 | measured |
| Composition changed (present) | **25** | 23,970 | — | — | **underpowered** |
| Window incomplete | 37 | 103,618 | +0.0917 | 1.3 | measured |
| Stable & single source | 181 | 1,472,932 | +0.1088 | 13.7 | measured |
| Stable & ≥3 sources | **25** | 407,437 | — | — | **underpowered** |

Paired difference (stable minus changed, on the dates both occupy): **25** dates, **underpowered**.

`--horizon 7 --from 2026-01-01 --basis set`

| Cell | n_dates | n_rows | rank IC | t | verdict |
|---|---|---|---|---|---|
| All rows | 172 | 1,845,202 | **+0.0842** | 14.3 | measured |
| Composition **stable** across t−1…t+7 | 167 | 1,686,266 | **+0.0842** | 14.1 | measured |
| Composition changed (present) | **19** | 27,559 | — | — | **underpowered** |
| Window incomplete | 166 | 131,377 | +0.1095 | 4.2 | measured |
| Stable & single source | 167 | 1,368,709 | +0.0919 | 13.8 | measured |
| Stable & ≥3 sources | **19** | 296,502 | — | — | **underpowered** |

Paired difference (stable minus changed, on the dates both occupy): **19** dates, **underpowered**.

**Holding source composition still does not touch the signal, but only one side of that
comparison is actually measured.** At 3d the stable cell is +0.1027 against an unconditional
+0.1023; at 7d the two agree to four decimal places (+0.0842 both). That is the whole of the
evidence: the **composition changed (present)** cell — items whose composition genuinely
differed with every window day observed, as distinct from a window that merely had a missing
day — is **underpowered at both horizons** (25 dates at 3d, 19 at 7d), so there is no number to
compare the stable cell against. The equality above is stable-vs-unconditional, not
stable-vs-changed; no directional claim about the changed population is supported by this run.
`Window incomplete` (37 dates at 3d, 166 at 7d) is gap-driven reporting, not composition change,
and is reported apart from `changed` for exactly that reason — it answers a different question
and is not part of the artifact test.

## Secondary result — the whole archive, composition = the count of ask sources

Stated assumption: `source` is NULL for every row before 2026, and this basis reads that as
**one constant source**, so 2013-2025 is composition-stable by construction. That is an
assertion from provenance (the pre-2026 series is a single Steam-derived backfill: 9,417,947
item-days, none with more than one row) rather than something the column proves. It is why this
is the secondary and not the headline.

`--horizon 3 --from 2013-08-14 --basis count`

| Cell | n_dates | n_rows | rank IC | t | verdict |
|---|---|---|---|---|---|
| All rows | 4,602 | 3,451,594 | **+0.1941** | 81.1 | measured |
| Composition **stable** | 4,598 | 3,321,918 | **+0.1937** | 80.6 | measured |
| Composition changed (present) | **25** | 23,864 | — | — | **underpowered** |
| Window incomplete | 81 | 105,812 | +0.1502 | 2.6 | measured |
| Stable & single source | 4,598 | 2,881,885 | +0.1939 | 80.7 | measured |
| Stable & ≥3 sources | **25** | 407,509 | — | — | **underpowered** |

Paired difference (stable minus changed, on the dates both occupy): **25** dates, **underpowered**.

`--horizon 7 --from 2013-08-14 --basis count`

| Cell | n_dates | n_rows | rank IC | t | verdict |
|---|---|---|---|---|---|
| All rows | 4,585 | 3,255,920 | **+0.1528** | 63.4 | measured |
| Composition **stable** | 4,580 | 3,091,865 | **+0.1515** | 62.2 | measured |
| Composition changed (present) | **19** | 27,493 | — | — | **underpowered** |
| Window incomplete | 552 | 136,562 | +0.2228 | 12.8 | measured |
| Stable & single source | 4,580 | 2,774,251 | +0.1518 | 62.3 | measured |
| Stable & ≥3 sources | **19** | 296,544 | — | — | **underpowered** |

Paired difference (stable minus changed, on the dates both occupy): **19** dates, **underpowered**.

Same answer over thirteen years and 4,598/4,580 dates: stable and unconditional are the same
number. And, as on the primary basis, the direct comparison is out of reach here too — the
composition changed (present) cell is underpowered at **exactly the same 25 and 19 dates** as
the primary basis, because a genuine composition change (as opposed to a gap) can only be
observed where `source` is non-NULL, which is 2026 regardless of which basis names it.

### How much does the set basis actually add over the count?

Empirically almost nothing, which is worth recording because it was the reason the set basis was
made primary. At h=3 over 2026 the two bases disagree about exactly **106 windows out of
1,912,978** (0.006%): the count basis calls them stable, the set basis calls them changed
because one source was swapped for another with the cardinality unchanged. The concern that the
count basis was biased toward "composition does not matter" was correct in principle and
immaterial in this archive.

## Which of the three outcomes occurred

**Outcome 1: the signal survives composition control at a reportable date count — but only the
stable-vs-unconditional half of that test is actually powered.** On the primary basis, at both
horizons, the composition-stable cell equals the unconditional cell to within noise on 181 and
167 dates; on the secondary basis the same holds on 4,598 and 4,580 dates over the whole
archive. The label is not explained by source composition changing under it, in the specific
sense that removing the changed rows (and the gapped rows) from the sample does not move the
number.

Two parts of outcome 3 attach, and neither is a small caveat:

- **The direct comparison this test was built to make — "composition changed (present)" against
  "composition stable" — is underpowered everywhere it was run**: 25 dates at 3d, 19 at 7d, on
  both the primary and the secondary basis, and the paired stable-minus-changed difference is
  therefore unreportable at every horizon too. There is no dataset in this archive, on either
  composition basis, in which the reversal's behaviour under a *known* composition change can be
  quoted. The "no directional shift" conclusion above rests on the stable population matching the
  unconditional population, not on any measured contrast against the changed population.
- **The "stable and ≥3 agreeing sources" cell is 25 dates at 3d and 19 dates at 7d** — the same
  counts, because both draw on the same short multi-source window — both below the 30-date
  reporting floor, so no number is quoted for either. That cell is the one §5 leant on, at 28
  dates.

The strictest available tests of the artifact hypothesis — a measured "changed" cell, and
"several sources agree" — are both still unavailable, and both are blocked on calendar time
rather than effort: the multi-source era is a few dozen days deep.

**Track C is unblocked**, but on narrower grounds than "the changed cell disagrees with the
stable cell": the evidence is that the composition-stable population's signal equals the
unconditional population's, at a well-powered date count, on both composition bases. No cell in
this archive currently supports a direct stable-vs-changed comparison.

## What this does *not* establish

- It does **not** show the reversal is tradeable. It removes one specific mechanism — the
  reporting set changing between the two ends of the window — from the *unconditional* sample,
  by showing that sample's number survives when composition-changed and gapped rows are excluded.
  It does not show what happens *in* a changed sample, because that sample is underpowered
  everywhere measured. A single-source series can still produce cross-sectional reversal from
  noise in its own quotes; §5's own observation that pooled 1-day autocorrelation is ≈0 argues
  against the time-series (Roll bid-ask bounce) form of that, but not against a cross-sectional
  measurement-error form.
- It does not touch the *level* errors already documented and unfixed: the BUFF bid voting
  ~11% low, and `aggregator_steam_7d/30d/90d` being trailing-window mean sale prices rather than
  point asks. Those bias the consensus price whether or not the source set is moving.
- Both the 25/19-date "≥3 sources" cells and the 25/19-date "changed (present)" cells being
  underpowered means the "does the reversal survive an *observed* composition change, and does it
  survive several independent sources agreeing" versions of the question are genuinely open.

## Caveats on the instrument

- **`MIN_ITEMS_PER_DATE = 5` is a stated choice.** Re-run at `--min-items-per-date 2` on the
  primary window (`--horizon 3 --from 2026-01-01 --basis set`): `all rows` and `composition
  stable` are unchanged to the four decimal places this script prints (+0.1023 and +0.1027,
  identical at both settings), and `composition changed (present)` stays at exactly **25 dates,
  underpowered, at both settings** — none of its dates sit in the 2-to-4-item range the floor
  would otherwise exclude, so lowering it does not rescue the cell. The cell the floor visibly
  moves is `window incomplete` (37 dates at the default 5, 108 at 2, rank IC +0.0917 vs +0.2626)
  — but that cell is gap-driven reporting, not composition change, so its sensitivity to the
  floor is not evidence about the artifact question this instrument exists to answer.
- **`_collection_shift_dates` is a function of the item universe it is handed.** On the
  backfilled-only frame it fires on `2026-03-22, 07-09, 07-10, 07-11, 07-12`; on the
  full-universe frame used here it also fires on **`2026-04-16`, `2026-07-14`, `2026-07-15`**,
  and over the full archive on five 2013-2016 startup dates as well (13 in total). "Which dates
  are void" is therefore not a property of the archive alone. Neither detector was modified;
  this is recorded, not fixed.
- `-r_t` here is an exact-calendar-day return, where production's `return_1d` is an as-of lookup
  within `LAG_TOLERANCE_DAYS`, winsorised at ±500%. Rank IC is invariant to the winsorisation;
  the as-of/exact choice changes which rows exist, and is part of why the "all rows" cell here
  reads +0.1761/916 dates against §5's +0.1676/932 on the same 2024-01-01 window.
- Every number is from the **local** archive copy, which runs behind CI (max day 2026-08-08).

## Reproducing

```bash
cd backend
venv/bin/python scripts/measure_composition_stability.py --horizon 3 --from 2026-01-01
venv/bin/python scripts/measure_composition_stability.py --horizon 7 --from 2026-01-01
venv/bin/python scripts/measure_composition_stability.py --horizon 3 --from 2013-08-14 --basis count
venv/bin/python scripts/measure_composition_stability.py --horizon 7 --from 2013-08-14 --basis count

# The MIN_ITEMS_PER_DATE sensitivity check quoted in the caveats above
venv/bin/python scripts/measure_composition_stability.py --horizon 3 --from 2026-01-01 \
  --min-items-per-date 2
```

Roughly 3.5 minutes each over 2026 and 4 minutes over the full archive; the cost is almost
entirely `_apply_multi_source_voting`'s per-group path over 2026's multi-source item-days.
