# Label integrity: powering up the composition test, and making the detector auditable

**Track G** of `docs/research/2026-08-09-next-steps.md`. Source measurements:
`docs/research/2026-08-09-model-and-data-research.md` §1.

This is the gate. Every accuracy item on the roadmap — the cross-sectional rank transform,
`lambdarank`, residual reversal, re-deriving the feature allowlist — is conditional on the answer
here, because if the price signal is an artifact of which venues quoted on which day, the label is
not a tradeable return and no model change can help.

Three deliverables, in order: an auditable list of the dates the label path already voids (G2
part 1); a powered-up source-composition test (G1); and the removal of the Steam trailing-window
feeds from the consensus vote (G3 / step 6c).

---

## What the measurement changed before anything was built

**The premise of the original G2 was wrong, and the way it was wrong is the reason for G2 part 1.**

A first draft of this spec proposed voiding 2026-07-09/10/11 as a newly-discovered consensus break
and generalising `_collection_shift_dates` to catch basis changes rather than universe-size
changes. Both halves are withdrawn:

- The 2026-07-09/10 cutover is documented (`docs/changelog/2026-08-06-validation-window-widening.md:14`),
  named in the detector's own docstring (`models/forecaster.py:2977-2981`) and in
  `docs/architecture/model.md:259`, is caught, and the span rule already voids labels across it
  (`docs/changelog/2026-08-06-paired-retrain-measures-no-gain.md:101`).
- The magnitudes this review measured (**+26.46% / −20.65% / +14.26%** median day-over-day across
  2,372 items) differ from the published **+17.4% / −17.8%** only because one is a median over a
  filtered cohort and the other a mean market return. Same event.

**Why the mistake was possible:** `_collection_shift_dates` fires 12 times in 4,735 days — 4 in
2013, 1 in 2016, 7 in 2026 — and **the list of dates has never been written down anywhere.** It
exists only as a count in a code comment. An audit that queries the archive directly, as this one
did, sees a large market-wide move on a date and has no way to check whether the label path
already handles it.

That is a legibility defect with a measured cost: it consumed an audit cycle and briefly put a
no-op task at the top of a roadmap. Printing the list is the fix, and it is Task 1.

**The 2026-07-11 question is genuinely open.** It is named in no changelog. The universe jumped
from `17mafo`'s 27,194 items to a nine-venue panel reaching 39,366, which is well past
`COLLECTION_SHIFT_FRACTION = 0.20`, so it is *probably* in the fired set — but that is an argument,
not a measurement, and Task 1 answers it as a side effect.

---

## G2 part 1. Publish the fired-date list

`_collection_shift_dates` (`:2975`) and `_snapshot_dates` (`:2951`) both return a `frozenset` that
`prepare_targets` consumes and discards. Nothing persists either one.

The change is to record both sets — and the resulting voided-label counts per horizon — into
`cv_results` / `meta.json` and into the training log, and to pin the 2026 members in
`tests/test_degenerate_label_dates.py`.

**Do not change either detector's logic.** The universe-size rule is deliberate and load-bearing:
prices moving cannot change how many items a collector returns, so the detector **cannot mask a
real crash**, and `test_a_price_crash_is_never_flagged` exists to keep it that way. Any future
basis-change detector must key on **source-set composition** — which sources reported on a given
day — and never on the magnitude of a price move. That column does not exist yet; it is G1's
`n_ask_sources`.

**What to record, per training run:**

| Key | Content |
|---|---|
| `label_voiding.snapshot_dates` | sorted ISO dates from `_snapshot_dates` |
| `label_voiding.collection_shift_dates` | sorted ISO dates from `_collection_shift_dates` |
| `label_voiding.voided_labels_by_horizon` | `{3: n, 7: n, 14: n, 30: n}` — rows whose target was set to NaN by the `bad` mask |
| `label_voiding.frame_date_range` | `[min, max]`, so a reader can tell whether a date is absent because it did not fire or because it was outside the window |

That last row matters more than it looks. The training window is `days_back=1460`, so the 2013 and
2016 cutovers fall outside it on most runs and their absence from the list is not evidence that
they stopped firing.

---

## G1. Power up the source-composition test

The measurement to extend is `docs/research/2026-08-08-model-review.md` §5 — rank IC of `−r_t`
predicting the forward 3-day return, on the voted daily series, ≥$1, 2024-01-01 onward, universe
rules applied:

| Subset | Dates | rank IC | t |
|---|---:|---:|---:|
| All rows | 932 | +0.1676 | 43.6 |
| Composition **stable** across t−1…t+3 | 188 | **+0.1006** | 14.1 |
| Composition changed | 775 | +0.1804 | 43.1 |
| Stable & single source | 185 | +0.1081 | 14.0 |
| Stable & **three agreeing sources** | 28 | **+0.0044** | **0.1** |

Roughly 40% of the effect is associated with composition change, consistent with the documented
basis errors. A component survives at +0.10 — but the cleanest cell shows **no reversal at all**,
on 28 dates.

### What the extension must add

1. **`n_ask_sources` as a stored column.** Today the partition is recomputed ad hoc in a
   scratchpad. It should be a column on the voted frame — the count of distinct non-bid,
   non-`historical_fallback` sources that reported for that item-day — so the partition is
   auditable and so a future basis-change detector has something to key on. This is also the
   best candidate cause of the harness's documented run-to-run non-reproducibility
   (`mean_diff_pp` −0.1581 → −0.0026 on identical commands), which has no diagnosis.
2. **Exclusion of the voided dates.** G1 reads the archive directly rather than through
   `prepare_targets`, so it must apply `_collection_shift_dates` and `_snapshot_dates` itself.
   This is the one real dependency on G2 part 1 — and the reason those sets need to be importable
   rather than buried in `prepare_targets`.
3. **Every cell reported with its date count and its t-stat**, and cells below a stated date floor
   marked `underpowered` rather than reported as a number. The 28-date cell is the whole result and
   it must never be quoted without its n.

### The constraint that may make this unanswerable, and that is fine

**The multi-source era is 24 days deep.** Per-source spans:

- `aggregator_csgotrader`, `skinport`, `csmoney`, `steam_7d/30d/90d`, `buff163_buy`: 2026-07-11 →
  08-07, **24 days**
- `buff163`, `youpin`, `csfloat`: 2026-03-22 → 08-07, **49 days present** of 139
- `aggregator_sync`: 2026-01-01 → 08-07, **113 days of 219**
- `17mafo`: 2026-04-16 → 2026-07-10 — and it is the **only** feed for that whole window
- pre-2026: a single backfill source, `source IS NULL`

So two windows are effectively single-feed, and "three agreeing sources, none changing" can only
be drawn from the 24-day nine-venue window plus parts of the 49-day three-venue one. The honest
outcome may be **`unresolvable until more multi-source days accumulate`**, and that is a
decision-grade answer: it says the accuracy roadmap is blocked on calendar time, not on effort,
and it should be recorded as such rather than worked around.

**What would make it a wrong answer:** running it across 2026-03-22 or the 2026-07-09…11
handover without excluding them; pooling across price tiers; or reporting the 28-date cell without
its date count.

---

## G3 / step 6c. Stop the Steam MA feeds voting

`aggregator_steam_7d`, `aggregator_steam_30d` and `aggregator_steam_90d` are Steam's
trailing-window **mean sale price** — MA(7)/MA(30)/MA(90) — voting on equal terms against
point-in-time asks (`collectors/csgotrader_aggregator.py:308-312`,
`collectors/pipeline.py:132-135`). Same class of basis error as `aggregator_buff163_buy`, removed
2026-08-07.

Measured:

| | |
|---|---|
| Coverage cost of excluding them | **670 item-days of 3,093,793** on the ≥$1 cohort — negligible |
| Voted median moves on | **17.13%** of 2026 ≥$1 item-days |
| Median move where it moves | **−7.16%** |
| Consecutive-day return directions flipped | **5.75%** |

Half the bid's magnitude, same character.

### Three things that must be got right

**1. This is not a staleness fix.** It clears only **2.30pp of the 20.25pp** ≥$1 stale rate,
because `aggregator_sync` and `aggregator_steam_17mafo` are `last_24h` **falling back** to those
same windows — and the fallback fires on exactly the illiquid items. The structural finding
stands and is not addressed by this change: **there is no point-in-time Steam price in this
archive at all.** Anyone scoping 6c as "fixes staleness" will measure a null and conclude the
wrong thing.

**2. Do not also drop `aggregator_sync`.** That deletes 2026-01 and 2026-02 in full for the ≥$1
cohort — 52,048 item-days — to buy a further 1.4pp. The correct fix there is upstream: record
which field of the fallback chain was actually used, which is a collector change, not a voting one.

**3. It subsumes the `aggregator_steam_17mafo` open item** from the 2026-08-07 step 1. That feed
is 2,161,250 rows / 27,194 items, was the sole feed 2026-04-16 → 2026-07-10, and its raw JSON
(`price-archive/raw/17mafo/`, 84 files, 630 MB) confirms it carries `last_24h/7d/30d/90d`. It
inherits the same fallback basis error for a three-month window and remains unaudited. Excluding
it is not an option — it is the only feed for that window — so the honest treatment is to mark the
window as single-source and let G1's `n_ask_sources` expose it.

### Mechanics

Add the three to the excluded set in `_apply_multi_source_voting` alongside `BID_SOURCES`. They
are a *different kind* of exclusion — a bid is the wrong side of the book, an MA is the wrong
time basis — so give them their own named constant (`TRAILING_WINDOW_SOURCES`) rather than
overloading `BID_SOURCES`, and make the exclusion NULL-safe by construction per invariant 2.

Bump `VOTED_CACHE_VERSION` 4 → 5. **Every A/B and every label from 2026-03 onward sits downstream
of this**, so it needs its own changelog entry and the re-vote must complete before any Track C
measurement is taken.

⚠️ **Sequencing against Track A Task 6.** That task adds a workflow cache keyed `voted-v4-`.
Whichever of the two lands second must move the key to `voted-v5-` in the same commit, or CI will
restore a frame voted under the old rule.

---

## What this spec does not cover

- **`STEAM_FEE_MULTIPLIER = 1.1607` (step 5c).** Measured synthetic — flat at 1.1606–1.1607 across
  four orders of magnitude, IQR 0.0002, on 63,767 pairs, where the cent-ceiling schedule must
  swing ~1.67 at $0.03 to ~1.15 at $50, and 91.61% of pairs are identical after dividing. Rows
  below ~$0.50 are ~5% too high. It is a real defect and it is separable: it affects the
  backfill's *level*, not which sources vote, and the ≥$1 training floor already keeps the worst
  of it out of the training cohort. Own diff.
- **The four missing archive days** (2026-07-27, 07-30, 08-02, 08-03). Midnight-UTC cron drift,
  not a basis problem.
- **Any new detector.** G2 part 1 makes the existing one auditable and stops there, deliberately.
  Build a source-composition detector only if G1 produces evidence that a composition change the
  universe-size rule misses is actually damaging labels — and note that `n_ask_sources` is the
  prerequisite either way.

---

## Verification

```bash
cd backend
venv/bin/python -m pytest tests/test_degenerate_label_dates.py \
    tests/test_label_voiding_audit.py tests/test_trailing_window_sources.py -q
```

| Check | Expectation |
|---|---|
| Fired-date list in `meta.json` | contains 2026-03-22 and 2026-07-09/10; **records explicitly whether 2026-07-11 is present** |
| `test_a_price_crash_is_never_flagged` | **still passes** — the detectors' logic is untouched |
| `n_ask_sources` on the voted frame | present, integer, never NULL; `1` for every pre-2026 row |
| G1 cell report | every cell carries its date count; cells under the date floor read `underpowered`, not a number |
| Coverage cost of the 6c exclusion | ≈670 item-days on the ≥$1 cohort — if it comes back materially larger, the exclusion is matching more than the three intended sources |
| `VOTED_CACHE_VERSION` | 5, and the workflow cache key matches |
