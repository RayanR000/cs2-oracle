# The serving path can be replayed, and it reads worse than CV, 2026-08-10

`REPLAY_ANCHOR` rewinds the serving clock so `predict()` can be scored against outcomes the
archive already holds. Four transforms run inside `predict` and nowhere else — the prior-day
blend, the tier bias, `_recenter_on_direction`, and the conformal band — so no CV number has
ever described what production serves. Until now the only way to read them was to publish a
forecast and wait `horizon` days.

Commits: `e6a9c43` (the harness), `1485662` (a defect it found in itself).

## What was built

`REPLAY_ANCHOR=YYYY-MM-DD` makes every window `predict` derives resolve as if today were that
date. Five seams had to move together, and each was a leak on its own:

- `_now()` returns the anchor, which carries the fetch cutoff, the model age gate and the
  prior-day blend. `_fetch_prior_forecasts` took `date.today()` directly and now takes
  `self._now().date()` — otherwise the blend mixes real forecasts dated after the anchor into a
  backdated prediction.
- The archive read gains an **upper bound**. `WHERE day >= cutoff` alone lets a rewound run read
  every row after its anchor.
- The **voted cache key** carries the anchor, or the replay restores the live frame — which holds
  exactly the rows the upper bound exists to exclude.
- The **engineered cache is ignored** under a replay. Its key fingerprints `forecaster.py`, which
  a replay does not change, so a live cache is a perfectly valid hit carrying post-anchor rows.
- An unparseable anchor **raises**. Ignoring it would replay against today and score a forecast on
  its own answer — a pass, and a meaningless one.

`scripts/replay_serving.py` writes nothing to the database. It resolves outcomes through
`fetch_price_history` rather than a hand-rolled archive read (invariant 2, and a plain
`median(mean_price)` lets BUFF's bid and Steam's trailing-window means vote — a systematic wedge
in every return), and it scores the cohort at the artifact's own `train_min_median_price`.

## The harness found a defect in itself first

The `-return_1d` baseline read **−0.6187 / −0.3815 / −0.3249 / −0.2610** rank IC at anchor
2026-06-01, against a CV value of about +0.19. Sign-inverted, and decaying monotonically in
horizon — the anchor-overlap signature the script's own docstring names.

It was a **basis mismatch**: the baseline was built from the last two raw quotes while the return
it was scored against divides by `predict`'s span-bounded median (`SMOOTH_WINDOW = 3`,
`MAX_WINDOW_SPAN_DAYS = 7`). Raw and smoothed differ for **29% of items** at that anchor. A fresh
jump moves `return_1d` and barely moves the median, and the outcome window then carries the jump.

Both bases were measured before anything changed:

| baseline built from | scored vs raw anchor | scored vs smoothed anchor |
|---|---|---|
| raw last-two quotes | +0.056 / −0.008 / +0.008 / +0.021 | **−0.619 / −0.382 / −0.325 / −0.261** |
| the smoothed anchor | +0.065 / +0.161 / +0.153 / +0.132 | −0.034 / +0.070 / +0.080 / +0.072 |

So the sign flip is the mismatch, not either series. `_naive_baseline` now builds on the served
basis, and truncates the frame before each `_smoothed_anchor_prices` call rather than leaning on
the anchor argument: an item with nothing inside the span window falls back to `last()` over the
**whole** frame, which in a replay reaches past the anchor into the outcome. That leak has its own
test.

## The control read, four anchors

Artifact trained 2026-08-09, `CROSS_SECTIONAL_RANK` off, cohort ≥ $1, n = 1,028–1,080 per anchor.

| h | 2026-05-01 | 2026-06-01 | 2026-06-15 | 2026-07-05 | mean |
|---|---|---|---|---|---|
| 3 | −0.1719 | −0.2008 | −0.0737 | −0.1846 | **−0.158** |
| 7 | +0.0776 | +0.0028 | +0.1180 | −0.0804 | +0.030 |
| 14 | +0.1818 | −0.0282 | +0.0859 | +0.0739 | +0.078 |
| 30 | +0.1665 | −0.0144 | +0.0606 | +0.1617 | +0.094 |

Against the naive baseline on the same basis, the mean edge is **−0.180 / −0.044 / +0.033 /
+0.065**.

**The one consistent result is h=3: negative at 4 of 4 anchors.** It is also the horizon where the
anchor-basis wedge is largest — the model loads on reversal, and on the smoothed basis `return_1d`
predicts the next three days *positively*, so a reversal-loaded forecast scores negatively there
almost mechanically. Treat it as a measurement question first, not a model result.

Served DA loses to the runnable constant call in **16 of 16** cells (−1.29 to −36.75pp). Read that
next to `realised_down_rate`, per invariant 4: at these anchors the down rate runs 22.3% to 91.5%,
and against an 83–91% down day *any* non-constant forecaster loses by construction. This is not
new information about the model.

## What this does not establish

- **No accuracy claim, and no comparison to a stored number.** Four anchors is four dates, each
  contributing one within-date rank IC, against a CV figure that averages hundreds. The spread
  across anchors here (−0.03 to +0.18 at 14d) is wider than the gap being discussed. Nothing here
  has an interval.
- **The artifact post-dates every anchor.** It was trained 2026-08-09 on data through that date,
  so all four replays run with training lookahead. That biases *toward* the model, which is worth
  knowing given the direction of the result — but it also means these are not historical runs.
- **The basis choice is unresolved and it is worth up to 0.6 rank IC at h=3.** Dividing the
  realised return by the smoothed anchor (what production publishes as `current_price`) and
  dividing by the raw quote are both defensible, and they disagree enormously at short horizons.
  The choice cancels in an arm-vs-control comparison at a fixed anchor, so no absolute served rank
  IC should be quoted until it is made deliberately.
- **No arm was measured.** `CROSS_SECTIONAL_RANK=1` needs its own artifact — `predict` refuses to
  serve a rank-transform artifact whose recorded cohort and skip set do not match
  (`2026-08-10-rank-transform-reference-cohort.md`), which is the guard working as designed.

## Tests

`tests/test_serving_replay.py`, 10 passed. The load-bearing ones are the leak tests: the archive
upper bound, the voted cache key, the engineered cache, the prior-day blend, and the baseline's
staleness fallback. Full suite 1,844 passed.
