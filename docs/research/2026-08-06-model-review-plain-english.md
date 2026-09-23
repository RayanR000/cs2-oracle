# What's wrong with the model, in plain English (2026-08-06)

A non-technical companion to
`docs/changelog/2026-08-06-scale-free-features-and-fabricated-labels.md`.
Same findings, no jargon. Every number here was verified directly against the
shipped model artifact or the price archive.

> ⚠️ **HISTORICAL — a snapshot of 2026-08-06, not current advice.** The four problems and
> the three fixes are an accurate record. Two things have changed since, and they change how
> the rest of the document should be read:
>
> 1. **The whole document is framed on directional accuracy, and that framing was retired
>    on 2026-08-15.** CS2 Oracle is a **range (interval) forecaster**; directional accuracy
>    is not a shippable product claim at any strength. So "Problem 4", "What good actually
>    looks like" and the 49–53% figures describe a metric the product no longer sells.
>    `changelog/2026-08-15-cs2-oracle-is-a-range-forecaster.md`.
> 2. **The "Still open" section at the bottom is closed** — see the notes there.
>
> The band geometry it predates has also moved twice: signed conformal (2026-08-19) and the
> climatology band scale, default on since 2026-08-20. See the banner in `docs/README.md`.

---

## The short version

The model wasn't stuck at a ceiling. It was being fed the wrong kind of numbers,
graded on the wrong group of items, and trained on some days the collector made
up.

---

## Problem 1 — It read dollars but predicted percentages

This is the big one.

The model is asked: *"will this item go up 3%?"* But most of the clues it was
given were dollar amounts — things like *"this item's price wobbles by $0.01."*

That clue means completely different things for different items. A one-cent
wobble is enormous for a 5-cent sticker and invisible for a $600 knife. So the
model couldn't learn "wobbly means volatile." It could only learn "small number =
cheap item, big number = expensive item." It stopped predicting price movement
and started memorising which item it was looking at.

This wasn't a corner of the model:

| Forecast horizon | Share of the model's decisions based on dollar amounts |
|---|---|
| 3 days | 55.6% |
| 7 days | 70.2% |
| 14 days | 77.5% |
| 30 days | 86.6% |

It also explains why longer horizons looked worse — the problem grows with the
horizon in lockstep.

Making it worse: the model was trained mostly on items worth about **8 cents**,
then asked to predict items worth up to **$639**. Those are so far outside
anything it saw in training that it lumps them all into a single bucket and
gives them the same answer.

**The code already knew about this.** The part that draws the confidence band
divides by price to fix exactly this problem, with a comment explaining why. The
model itself never got that fix.

---

## Problem 2 — It peeked at the answers

During training, the model sets aside the most recent 30 days to test itself on.

But a 30-day forecast made on the *first* day of that hold-out period is answered
by data inside the hold-out period. So the answer was already sitting in the
study material. The "test" wasn't a test.

This matters because that test is what decides the model's settings — how long to
train, which configuration to use. Those were all chosen against a rigged exam.

(The cross-validation step did this correctly. The final training step didn't.)

---

## Problem 3 — Some days in the archive are fake

Two kinds of bad days, both of which the model treated as real market activity:

**Copied days.** 2026-07-16 and 2026-07-22 are exact duplicates of the day
before — all ~40,000 items, identical prices, not a single one changed. Out of
4,735 days in the whole archive, only these two do this. The next "flattest" day
in history is 69%, so there's no ambiguity about what these are.

Worth knowing: **2026-07-16 is the day before 2026-07-17**, which is the one
production forecast batch the project has been treating as trustworthy.

**Source-switch days.** The archive isn't one continuous feed — it's several
price sources stitched together. When the stitching changes, prices jump. On
2026-03-22 the whole market appears to fall **31.6%**; on 2026-07-09 and 07-10 it
appears to jump **+17.4%** then **−17.8%**. Normal days move about 0.5%.

Nothing actually happened on those days. The price source changed.

The existing safety net couldn't catch either problem: it only discards moves
bigger than 500%, and a −31.6% fake crash or a 0% copied day sail right through
as confident, completely wrong lessons.

---

## Problem 4 — It was graded on the wrong students

Your site only shows items worth **$1 or more**. But the model's training data is
about **83% penny items**, and every past experiment was scored on that whole
mixed pool.

Split them apart and the picture changes completely:

|  | Score on everything | Score on items you actually show |
|---|---|---|
| 3 days | 67.3% | **49.9%** |
| 7 days | 67.2% | **49.0%** |
| 14 days | 67.8% | **51.1%** |
| 30 days | 68.6% | **53.4%** |

The left column is basically the penny-item score. The right column — the one
that describes your product — is a coin flip.

And it's a *stable* coin flip, not bad luck: across all nine test periods, the
3-day score sits at 49.9% give or take 1.9. It isn't noisy. There just isn't any
signal there.

For comparison, simply guessing "down" every single time, with no model at all,
scores 49.6% at 7 days and 55.8% at 30 days. On the items you serve, the model is
at or slightly below that.

> ⚠️ **Correction (2026-08-10):** the "always down" comparison as it was computed in the
> backtest (`constant_call_accuracy`) is **selected with hindsight per fold**, so it is an
> oracle, not a baseline the model failed to beat. The runnable baseline is
> `realised_down_rate`. `changelog/2026-08-10-constant-call-is-hindsight-picked.md`.

---

## What was fixed

1. **Every dollar-based clue converted to a percentage-based one.** Dollar-based
   decisions went from 56–87% down to **0% at every horizon**. The information
   isn't lost — "wobbles by $0.01 on a $0.10 item" simply becomes "wobbles by
   10%," which travels correctly across price ranges.
2. **The peeking stopped.** The training step now discards the overlap window, the
   same way cross-validation already did.
3. **The fake days are excluded** from what the model learns. Copied days are
   dropped as start/end points; source-switch days void any lesson that spans
   them.

Safety note on #3: source-switch days are detected by counting *how many items
the collector returned*, never by looking at prices. A real market crash doesn't
change how many items get collected — so this can never delete a genuine crash.
It triggers on 12 days out of 4,735 (0.25%).

39 new tests were added. The full test suite passes (840 tests as of 2026-08-06; ~2,460 as
of 2026-08-21).

**One requested change was deliberately not built.** A fourth item — a
statistical weighting adjustment — was measured and found to do nothing here: the
math cancels out for more than 90% of the data. Building it would have been
busywork.

---

## The honest limitation

**The causes are fixed. The improvement is not proven.**

We can show the broken mechanism is gone. We cannot yet show the accuracy number
moved — because the system currently can't measure its own accuracy:

- Production has only 1–2 real forecast dates, and 20 are required before a
  number can be quoted.
- The offline testing tool measures a meaningfully different setup than what
  actually ships.

Running the proper before/after comparison is about a 20-minute job and hasn't
been done yet.

> **Status 2026-08-21.** The date-counting half was a code bug as much as a calendar one:
> the scoring cohort keyed on the *configuration*, so every config change reset the panel.
> Fixed 2026-08-11 (`changelog/2026-08-11-model-version-is-not-a-config.md`) — the panel now
> accumulates. The before/after accuracy comparison was subsequently run many times over and
> came back **null**: the accuracy surface is where this project stopped, which is why it was
> reclassified as a range forecaster. Do not schedule the 20-minute job described above.

---

## What "good" actually looks like

Worth calibrating, because the target has probably been too high:

- The best academic work in this field — thousands of stocks, hundreds of
  features — achieves about **0.4% predictive power**. That is considered an
  outstanding result.
- Both published papers claiming CS2 skin prediction works have bugs where the
  answer leaks into the clues. One reports a result roughly 1,000× better than
  the best equities research, using a feature that contains today's price to
  predict today's price change.
- A realistic target is **beating a constant "always down" guess by 1.5–3
  percentage points**, consistently. Not "reach 60%."

This project's 49–53% is, as far as we can tell, the most honestly measured
number that exists for this market. That is worth something on its own.

---

## Still open

> ⚠️ **All three items below are closed. Kept for the record only.**

**A live bug (reported, not fixed by request).** ✅ **Fixed.** In certain market conditions
the 3-day forecast was served by a model containing **one decision tree** instead of the
real 379-tree model. Production stopped early-stopping on 2026-08-08 and now trains a fixed
round count; the old behaviour survives only behind an opt-in `EARLY_STOPPING=1`
(`forecaster.py:887`), which nothing in CI sets.
`changelog/2026-08-13-ab-harnesses-follow-productions-trainer.md`.

**Free speed.** ✅ **Taken.** The cost work closed on 2026-08-09; a warm retrain now runs
**176.7s** (250.1s cold), inside the cap, and the bottleneck moved from wall-clock to
experiment power. `docs/operations.md` → Healthy state; `research/2026-08-10-training-cost-levers.md`.

**Free data.** ❌ **Measured and shelved.** The bid/order-book side was built behind
`BID_FEATURES` and the volume/demand panels behind `VOLUME_FEATURES` / `IFLOW_VOLUME`; both
are off, and the cross-venue Steam↔BUFF basis feature was refuted under every skeptical
control. `changelog/2026-08-06-volume-features-shelved.md`,
`changelog/2026-08-17-volume-in-scale-is-net-negative.md`,
`research/2026-08-16-cross-venue-basis-steam-buff.md`.
