# The label can divide by the price production serves from, 2026-08-11

`LABEL_SMOOTHED_ANCHOR=1`. Gated off. It changes `prepare_targets`' denominator from
the raw quote observed on the anchor day to `_smoothed_anchor_prices`' span-bounded
median — the statistic `predict` converts a return-space forecast to dollars with.

## Why this one and not one of the ranked arms

The 2026-08-11 serving replay crossed the two axes between the CV label and the served
one and attributed the CV↔serving rank IC gap to the **denominator**: swapping only it
recovers **+0.1398 of the +0.1464** mean gap, the outcome leg carries +0.0251, and the
serving transforms carry none (`2026-08-11-the-gap-is-the-anchor-denominator.md`,
`2026-08-11-serving-transforms-are-not-the-gap.md`). Confirmed in CI on a fresh artifact
at four non-overlapping anchors: the gap is zero to within ±0.021 on items whose anchor
quote already equals its local median, and positive in **all sixteen** deviating cells
(`2026-08-11-clean-anchor-confirmed-in-ci.md`).

The mechanism is a shared quote. The raw anchor sits in the label's denominator *and* in
`return_1d` and every level feature built from it, so one noisy observation deflates the
label and inflates the feature together, and a model that reads the noise scores as if it
had read the market. A median of three observations cannot be moved past its neighbours by
any one of them.

Every arm in `docs/research/2026-08-10-next-steps.md` — C1, N1, `lambdarank` — and the
`−return_1d` bar they were held to were ranked on the contaminated metric. That is why this
is ahead of them rather than beside them.

## What changed

- `ItemForecaster.label_smoothed_anchor_enabled()` — the flag, read at call time.
- `ItemForecaster._rolling_anchor_prices()` — the panel form of
  `_smoothed_anchor_prices`: the median of the most recent `SMOOTH_WINDOW` (3)
  observations within `MAX_WINDOW_SPAN_DAYS` (7) before each row's own date. Built from
  k-step shifts because the two bounds are of different kinds — a row count and a calendar
  span — and pandas' `rolling` takes one or the other.
- `prepare_targets` uses it for **both legs** of the return. Subtracting the raw anchor
  from a smoothed base would leave `(S − p_raw)/S` in the label, which is the contamination
  term itself, rescaled.
- `meta.json` records `label_smoothed_anchor`, for provenance.
- `model-diagnostics.yml` gains the arm and prints it in the heading.

`predict` is untouched and needs no offset added back: it already quotes against the
smoothed anchor, so an artifact trained under this flag is the *coherent* pairing and the
shipped default is the incoherent one — the model has been fitting returns against a raw
base and having them converted against a smoothed one.

`VOTED_CACHE_VERSION` is unchanged. The voted frame is upstream of `prepare_targets`;
nothing about the consensus price moves.

## How to read the arm — not on CV rank IC

**This is a label change, so the two arms are scored against different targets.** A CV rank
IC, a DA, and the `naive_rank_ic` bar all move for reasons that have nothing to do with
forecast quality, and differencing them across the flag is meaningless. The referee is
`scripts/replay_serving.py`, which builds its labels from the archive, never calls
`prepare_targets`, and scores both artifacts on one basis.

Dispatch `model-diagnostics.yml` twice on the same commit — control and
`label_smoothed_anchor=true` — with `replay_anchors` set to the four non-overlapping
anchors CI already used: `2026-04-15,2026-05-16,2026-06-16,2026-07-09`. Read served rank IC
per anchor and horizon. Note 2026-07-09 carries a disproportionate share of the effect
(26 tied items of 669, against 27–56% elsewhere), so read it as its own column rather than
pooling it.

## What this does not claim

Nothing here says the flag improves served accuracy. It says the metric that ranked every
prior arm was measured against a target the serving path does not use, and that this is the
axis carrying 95% of the discrepancy. Whether a model *trained* on the corrected label
serves better is the open question, and it is what the two dispatches above measure.

The outcome leg (+0.0251, a trailing median over `(anchor, anchor + h]`) is deliberately
left alone — a separate axis, separately measurable, and at short horizons its window
reaches back over the anchor and needs the `after=` guard `replay_serving._resolve` carries.

Tests: `backend/tests/test_label_smoothed_anchor.py` (12), including the invariance the
flag exists for — once a spiked quote clears its window's maximum, making it larger cannot
move the label again, where the raw label is unbounded in the size of the spike.
