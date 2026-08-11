# The smoothed-anchor label, measured: the gain is the deviation, not skill

Control run `31459789143` and arm run `31461219973`, both on `4d0c8bc`, same four
non-overlapping anchors (2026-04-15, 05-16, 06-16, 07-09), `label_smoothed_anchor` confirmed
`false` / `true` in each job's `meta.json`. Served rank IC from the replay, which builds its
labels from the archive and never calls `prepare_targets`.

**Verdict: do not ship it. Keep `LABEL_SMOOTHED_ANCHOR` off.** It is not a null result — it
identifies a second contamination with the opposite sign to the first, and it retires the
"serving gate on anchor cleanliness" as a separate idea.

## The control reproduces exactly

Every one of the 16 cells in `2026-08-11-clean-anchor-confirmed-in-ci.md` came back to four
decimals on a different commit and a different runner — tied +0.1321 / +0.1562 / +0.1747 /
+0.0536, deviating −0.2014 / −0.0529 / −0.1009 / −0.0563, and both gap columns. The replay is
deterministic, so anything the arm moved, the arm moved.

## What the arm did

Mean across the four anchors; every arm cell positive at 4 of 4 anchors except 7d tied (3/4).

| h | subset | control | arm | Δ |
|---|---|---|---|---|
| 3 | pooled | −0.0815 | **+0.3088** | +0.3904 |
| 3 | tied | +0.1321 | +0.1501 | +0.0180 |
| 3 | deviating | −0.2014 | **+0.4387** | +0.6401 |
| 7 | pooled | +0.0224 | **+0.2932** | +0.2708 |
| 7 | tied | +0.1562 | +0.0934 | −0.0629 |
| 7 | deviating | −0.0529 | **+0.3964** | +0.4494 |
| 14 | pooled | −0.0208 | **+0.1743** | +0.1952 |
| 14 | tied | +0.1747 | +0.0946 | −0.0800 |
| 14 | deviating | −0.1009 | **+0.2264** | +0.3273 |
| 30 | pooled | −0.0066 | **+0.1815** | +0.1881 |
| 30 | tied | +0.0536 | +0.1035 | +0.0498 |
| 30 | deviating | −0.0563 | **+0.2266** | +0.2828 |

A pooled served rank IC of +0.17 to +0.31, positive at every anchor and every horizon, would
be the largest result this project has ever measured. It is not one.

## Why it is arithmetic

```
label_arm = P[d+h]/S[d] − 1 = (P[d+h]/p[d]) · (p[d]/S[d]) − 1
```

`p[d]/S[d]` — the raw quote over its own trailing median — is **known at the anchor**, is
readable off `return_1d` and the lags, and is precisely what the tied/deviating split measures.
So the corrected label carries a free multiplicative factor that a booster can predict with no
market information at all. It is the mirror of the defect it was built to remove: the shipped
label shares the raw quote through its *denominator* and reads negative on deviating items; this
one carries `p/S` as a *factor* and reads positive on exactly the same items.

**The tied subset is the natural control**, because there `p[d] = S[d]` and the factor is
identically 1. Excluding 2026-07-09, whose tied cell holds 26 items against 301–598 elsewhere:

| h | control | arm | Δ |
|---|---|---|---|
| 3 | +0.1017 | +0.0687 | **−0.0330** |
| 7 | +0.1518 | +0.1350 | **−0.0169** |
| 14 | +0.1173 | +0.0971 | **−0.0202** |
| 30 | +0.0807 | +0.0888 | +0.0081 |

**Where the mechanical factor cannot operate, the corrected label buys nothing** — slightly
negative at three horizons of four. All of the pooled gain lives where the factor exists.

Corroborating: the arm's `cv − served` gaps go *negative* (deviating −0.19 at 14d, was +0.30).
CV now understates serving, which is what a free factor present at serving and absent from the
CV basis produces.

## What this changes

1. **The label rebuild is not the fix.** The CV metric is still measured against a target the
   serving path does not use, and that remains true — but replacing the denominator does not
   produce better forecasts on the cohort where the answer is readable. Both bases are
   contaminated by `p/S`, with opposite signs. Read arms on the **tied subset**, which is the
   only place neither contamination operates.
2. **The serving gate on anchor cleanliness is no longer a separate idea, and it is not an
   accuracy lever.** Under the shipped label the deviating two-thirds looks anti-predictive;
   under the corrected one it looks brilliant. Neither is about those items' prices. What the
   split actually measures is how far production's quoted `current_price` sits from the last
   observation.
3. **The one durable number is the control's tied cell**: +0.10 to +0.15 at 3/7/14d excluding
   the outlier anchor, +0.08 at 30d. That is the project's measured served signal, it is on the
   clean third of the cohort, and neither run moved it.

## What was not measured

Whether quoting against a *fresher* anchor — narrowing `p/S` at source rather than choosing
which side of it to divide by — improves anything. That is a serving-basis change, not a label
change, and it is the question this pair of runs raises. No interval on any figure here: four
anchors is a replication, not power, and the sign consistency is what is readable.

Artifacts: `diagnostics-{3,7,14,30}d` on both runs, 90-day retention.
