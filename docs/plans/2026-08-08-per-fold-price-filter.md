# Per-fold price filter: re-deriving step 7 without the look-ahead

> ## ✅ STAGE 1 EXECUTED (2026-08-08) — the question below is ANSWERED
>
> **Answer: the +3.50pp does not survive.** Re-derived without the look-ahead it comes back
> **+1.642pp [−0.809, +4.505] — null**, and the placebo arm puts the instrument's item-draw noise
> floor at **±3–4pp**, wide enough to have produced the original number by itself.
>
> **Outcome: `docs/changelog/2026-08-08-per-fold-price-filter-rederived.md`.** Evidence:
> `_fold_median_price_items` at `forecaster.py:3364` (`5ebcc73`),
> `backend/scripts/ab_test_train_universe.py` (arms `prod_pool`, `prod_pool_b`, `full_sample`,
> `per_fold`, `full_sample_matched`, `full_sample_matched_b`), and `TestFoldMedianPriceItems` in
> `tests/test_training_item_coverage.py`.
>
> ⬜ **Stage 2 (the anchored production filter) is not started — and the finding is that it is not
> needed.** `build_training_data` still does not call `_fold_median_price_items`; the only callers
> are the tests and `ab_test_train_universe.py:386`. The floor shipped anyway, on determinism
> rather than accuracy (`6eb8775`, `docs/changelog/2026-08-08-training-price-floor-shipped.md`).
>
> ⚠️ **Two defects in this document's own text, found when the work ran.** They are left in place
> so the reasoning is still readable, but do not act on either:
> 1. It names `ab_test_training_breadth.py` as the instrument that produced the +3.50pp. **It did
>    not** — that harness never produced the number.
> 2. Its third arm is described backwards: it is `full_sample` that gets thinned, not `per_fold`.
>
> ⚠️ **The withdrawal has not propagated.** `docs/architecture/model.md:210,533` still cite the
> +3.50pp as settled. That correction is O2 of `docs/research/2026-08-09-next-steps.md`.

**Blocks step 7** of `docs/research/2026-08-07-next-steps.md` ("Ship `TRAIN_MIN_MEDIAN_PRICE`
at a `TRAIN_FEATURE_ROWS ≥ 1.0M` budget"), whose own caveat is *"Re-derive the result first
with a per-fold price filter — the current version selects on a full-sample median."*

Nothing here ships a model change. The deliverable is an answer to one question: **is the
+3.50pp at 30d real, or is it survivorship?**

---

## The defect

`models/forecaster.py::_filter_by_median_price` is two lines:

```python
item_median = price_df.groupby("item_id")["price"].median()
keep = set(item_median[item_median >= min_median_price].index)
```

`price_df` is the whole 1460-day frame, and the filter runs once in
`build_training_data` before any split exists. So the training universe of a fold whose
window ends in 2019 is *"items whose median price over 2013→2026 is ≥ $1"* — a set nobody
could name in 2019. An item that was a penny sticker in 2019 and a $50 item in 2025 is in
that fold's universe **because it later rose**.

`scripts/ab_test_training_breadth.py:269-276` repeats it in SQL:

```sql
HAVING COUNT(DISTINCT day) >= 180 AND MEDIAN(mean_price) >= 1.0
ORDER BY ... LIMIT 870
```

All three clauses are full-sample. **The harness that produced the +3.50pp contains the same
leak the result needs testing for**, so the number cannot adjudicate itself.

**The signature fits.** A genuine liquidity effect has no reason to be horizon-selective. A
survivorship effect concentrated in the longest-horizon, thinnest-fold cohort does — and the
effect is **h=30 only, null at 3/7/14** (§18 L2).

## Two blockers the next-steps entry does not mention

**B1. Moving the filter later silently changes the market features.** The filter runs
*before* `engineer_features`, so `_add_cross_sectional_features` →
`_market_from_partials(df)` (forecaster.py:2127-2148) builds the market factor, the
`market_return_*d` lags and `item_return_vs_market_*d` from the **≥$1 universe**. A per-fold
filter necessarily runs after feature engineering, so those features would instead describe
the *pooled* universe. That is a second, independent treatment riding along inside what is
supposed to be a one-variable arm.

**B2. Per-fold filtering destroys the budget argument that motivates step 7.** The filter is
positioned before `_stratified_item_subsample` precisely so the row budget buys ≥$1 breadth
rather than the pool's tier mix (44% stickers and graffiti at a $0.03 median). Move selection
into the fold loop and the subsample has already spent the budget on the pooled universe —
which is the exact thing step 7 exists to fix.

Both are why this is split into two stages rather than done as one rewrite.

---

## Stage 1 — the derivation

Answers the doc's question. Harness-only; production is untouched.

### Task 1.1 — `_fold_median_price_items`

New pure helper beside `_filter_by_median_price`:

```python
def _fold_median_price_items(price_df, min_median_price, cutoff) -> set:
    """Items whose median price over `date < cutoff` clears the floor."""
```

**`cutoff` is `val_start - embargo_days(horizon)`, never `val_start`.** The selection
statistic is itself a function of prices; computed up to `val_start` it reads the embargo
window, which is the same leak in a smaller form. Use `embargo_days`, never a bare horizon —
see `.claude/rules/labels-and-embargo.md`.

Unit tests: an item that only clears the floor after `cutoff` is excluded; an item that
clears it before and collapses after is included; an empty pre-cutoff window yields an empty
set rather than everything.

### Task 1.2 — arms in `ab_test_training_breadth.py`

Filter **`train_set` only.** `val_set` is left alone: the scored metric is already
`acc_ge1`, so filtering val would move the cohort rather than the treatment, and the two
would be inseparable.

| arm | training universe | isolates |
|---|---|---|
| `full_sample` | today's behaviour, reproduced | the existing +3.50pp |
| `per_fold` | median over `date < val_start − embargo(h)` | look-ahead removed |
| `per_fold_matched` | `per_fold`, downsampled to `full_sample`'s item count | look-ahead removed **at equal universe size** |

**The third arm is not optional.** Per-fold selection shrinks early folds' universes, so
without it "the effect vanished because the leak is gone" and "the effect vanished because
there is less data" are the same observation. Downsample with the fold's own seed and record
the realised item count per fold.

Hold B1 fixed across all three arms — compute the market factor once, on the pooled frame,
for every arm. It is then wrong in the same way everywhere and differences out of the paired
contrast. Note it in the output; do not quietly let it vary.

### Task 1.3 — fix the other two full-sample clauses

`COUNT(DISTINCT day) >= 180` and `LIMIT 870` are survivorship on the same axis: selecting the
870 best-covered items over 2013→2026 pre-selects items that survived to 2026. Fixing only
the median leaves the leak in and would produce a confident null. Both move inside the same
`cutoff`.

### Task 1.4 — read the result

Paired, fold-clustered, via `backtest/paired_mde.py` — `paired_arm_contrasts` against
`full_sample`, never a win count (`.claude/rules/ab-statistics.md`). Read the **placebo
first**: `ab-fold-count-floor` records a permuted placebo reading significantly positive at
~7 folds, so a thin-fold run is not trustworthy on treatment alone.

Outcomes:

- **`per_fold` null and `per_fold_matched` null** → the +3.50pp was look-ahead. Step 7 does
  not ship on this evidence, and `2026-08-07-training-item-universe.md` needs a correction
  note.
- **`per_fold` null but `per_fold_matched` positive** → the effect is real and the per-fold
  filter merely starved it. Step 7 proceeds to stage 2.
- **Both positive** → effect is real and survives honest selection. Step 7 proceeds.
- **Anything at h=30 only, again** → treat as unresolved, not confirmed. h=30 rests on 5,461
  usable rows from one backdated date, and that caveat is unchanged by this work.

---

## Stage 2 — production, only if stage 1 survives

Do **not** ship the expanding per-fold filter into `build_training_data`. Ship an **anchored**
one: compute the median once over `[window_start, first_val_start − embargo)`.

Every fold's training window contains that period, so the universe is knowable at every
fold's decision time; it is one universe rather than a fold-varying one; and it still runs
before `engineer_features`, so **B1 and B2 both disappear** — the market factor keeps
describing the ≥$1 universe and the row budget still buys ≥$1 breadth.

The cost is a stated conservatism, and it must be written into the docstring rather than
discovered later: an item that only reaches $1 in 2023 never enters the universe, even for a
2025 fold where it would legitimately qualify. That biases toward long-lived expensive items.
It is a bias, not a leak, and it is the correct trade for a knob whose whole purpose is to
make the budget deterministic.

---

## Preconditions

1. **Re-derive on the post-step-6 label set.** `2026-08-08-frozen-price-runs-dropped-from-labels.md`
   voids 13.8–19.2% of ≥$1 labels; the +3.50pp predates it. Any contrast against the stored
   number is invalid regardless of the filter.
2. **Step 6 and step 7 overlap non-additively.** Step 6 removes sub-$1 rows via their frozen
   prices; step 7 removes the sub-$1 items outright. Do not sum a before/after across both.
3. **Never read an effect off one item draw.** `_stratified_item_subsample` has a hardcoded
   `seed=42` and changing only the seed moves `mean_classifier_acc_ge1` by sd 1.5–3.1pp
   (`.claude/rules/training-budget.md`). A universe-changing arm compared against a one-seed
   baseline is not a measurement.
4. **CPU.** Stage 1 runs the walkforward harness. Do not start it while another walkforward
   job is running.

## Out of scope

- Shipping `TRAIN_MIN_MEDIAN_PRICE` or moving `TRAIN_FEATURE_ROWS`. That is step 7 proper,
  and it waits on stage 1.
- Recomputing the market factor per fold (B1 done properly). Real, and a separate change —
  it affects every harness, not this one.
- `MIN_SERVED_PRICE_USD`. Unrelated; it tracks `HEADLINE_MIN_TIER`, not the training floor.
