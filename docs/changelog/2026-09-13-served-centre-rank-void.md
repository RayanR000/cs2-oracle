# Served centre rank: VOID — and the quote basis cannot referee ranking (2026-09-13)

Scores `docs/research/2026-09-13-served-centre-rank-preregistration.md` (+ same-day
amendment). Instrument: `backend/scripts/served_centre_rank.py` (+
`tests/test_served_centre_rank.py`, 9 tests). Official verdict: **VOID** — and the
audit found the design would have misled at any date count.

## The run (as pre-registered)

Served h=14 panel (22,705 rows / 24 dates), production `r_hat` vs quote-basis realised,
naive `−return_1d` paired on identical rows. Serve-time mask coverage 0.525 < 0.70, so
the amendment's mechanical rule selected wedge-negligible primary — no discretion.

| cohort | rows | share | dates | model | naive | paired |
|---|---|---|---|---|---|---|
| clean | 3,488 | 0.19 | 11 | **+0.073 [+0.034, +0.113]** | −0.091 | +0.163 |
| wedge (primary) | 6,744 | 0.38 | 18 | −0.009, null | **−0.193** | +0.184 |
| exact | 6,715 | 0.37 | 17 | −0.030, null | −0.190 | +0.160 |

Primary has 18 qualifying dates < 20 → **VOID by the letter**. September served dates
score no naive signal: both archives end 2026-08-25/09-08 for prices, and September
anchors need trailing days past the gap.

## Why the VOID is substantive, not calendar: the anchor lag

Naive at −0.19 on 18/18 dates (min −0.60, max −0.02) is not a market fact. Decomposed
on 2026-08-11 (942 paired rows, same rows, swapped outcome legs):

| outcome basis | −r1 IC |
|---|---|
| archive forward (contemporaneous anchor) | +0.043 |
| served quote basis | −0.138 |

Same rows, same signal — the flip is the outcome leg. Mechanism, measured directly:
the serve-time quote is a lagging median-of-3 while the voted price is contemporaneous,
and their gap is **+0.60 [+0.46, +0.73] rank-correlated with trailing 1d returns**.
Quote-basis outcomes therefore hand momentum a mechanical boost and reversal a
mechanical penalty *in either sign*. The overlay's paired +0.18 is naive's anchor
handicap, not model skill — CONFIRMED would have been a lie at 20 dates too.

Design lesson, extending 08-11 ("both bases carry the wedge") to a third basis:
**quote-basis outcomes cannot referee ranking.** Any future served-rank read must score
against the archive basis (or another contemporaneous anchor), never the served quote.

## Exploratory, labelled as such (no bars, motivates the next prereg)

One pre-specified mirror cell, served `r_hat` vs ARCHIVE-basis forwards on wedge rows
(working-copy archive through 09-08; September rows unscorable yet):

| cell (13 dates) | IC |
|---|---|
| served r_hat vs ARCH f14 | +0.003 [−0.103, +0.110] |
| arch −r1 vs ARCH f14 | −0.012, null |
| paired on ARCH basis | +0.014, null |

The 2025 tied ranking (+0.094, clean-trained walk-forward) does not transfer to
production models on current evidence — consistent with trainer/labels mattering, with
CIs (±0.10) too wide to call it a refutation either. The next prereg (archive-basis
confirmation on served rows, gated on September archive freshness + 20 dates) is
motivated, not substituted, by this.

## Reproduce

```
venv/bin/python -m scripts.served_centre_rank --out /tmp/scr_h14.json
venv/bin/python -m pytest tests/test_served_centre_rank.py -q
```

Naive leg defaults to the durable checkout; September anchors need the working copy
(`--archive-dir price-archive`), which carries 2026-09 through 09-08.
