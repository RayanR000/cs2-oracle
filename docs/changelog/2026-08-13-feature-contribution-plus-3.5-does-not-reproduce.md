# The +3.5pp feature-contribution number does not reproduce on the honest trainer

**Date:** 2026-08-13.
**Pre-registered:** `docs/research/2026-08-13-feature-contribution-honest-trainer-preregistration.md`,
committed as `d0717f2` **before** any number was read — bars, placebo seed, and point predictions
all fixed in advance.
**Refutes:** the `+3.5pp @30d` for removing cross-sectional features
(`docs/research/2026-07-19-feature-contribution-by-horizon.md`), the founding and never-re-derived
basis of `FEATURE_GROUP_ALLOWLIST = ["price_technicals"]`. Reopens **C4**.
**Instrument:** `backend/scripts/ab_test_feature_contribution.py`, honest trainer
(`EARLY_STOPPING=0`), `--horizon 30 --max-items 1500`, one run, 25 shared folds, **n = 104,843**
≥$1 non-flat paired rows.

## Result — legs 1 and 2 both fail, and both against the allowlist

The founding number carried four stacked defects (trainer leak, penny cohort, PT null read as skill,
no embargo). The harness already strips three of them: the paired contrast scores on **≥$1 non-flat
rows** with the H+13 embargo applied, and a fold-clustered interval is a skill contrast, not a DA
level. This run stripped the fourth — production's trainer — and read the pre-registered
`_paired_vs_full` interval:

| Contrast (arm − full, +Δ = removing helps) | Verdict | Interval |
|---|---|---|
| `no_cross_sectional − full` | **null, −0.35pp** | [−4.15, **+3.18**] |
| `no_events − full` | null, −3.41pp | [−7.95, +1.23] |
| `no_random_k − full` (placebo) | null, +0.28pp | [−0.49, +1.19] |

**Leg 1 (primary) FAILS its pre-registered bar** ("interval excludes 0 and is positive"). The point
estimate is *negative* and the upper bound (+3.18) essentially excludes the claimed **+3.5pp**. The
cohort resolved cleanly (25 folds, 104,843 rows), so this is a genuine null — not the cohort-power
fallback the pre-registration flagged. The interval width (~7.3pp) is consistent with the documented
fold-clustered MDE of 2.21–3.69pp; a real +3.5pp effect would have shown.

**Leg 2 (placebo) FAILS.** Removing a random equal-sized column set (`no_random_k`, +0.28pp) is if
anything *better* than removing the cross-sectional group (−0.35pp) or the event group (−3.41pp). So
there is no cross-sectional-specific — or event-specific — noise to remove; the real feature groups
are weakly *more* useful than noise columns, the opposite of "they add noise." No support survives
for the allowlist's removal of either group.

## What the old +3.5 actually was

Descriptive pooled DA (penny-inclusive, **not** the verdict — reported only beside the base rate):
full **55.3%**, no_cross_sectional **54.8%**, no_events **51.7%** at h=30. Two things:

- The sign already flips on the pooled number too — removing cross-sectional now *lowers* pooled DA
  by 0.5pp, where 2026-07-19 read +3.5.
- The whole level collapsed from the old **65–72%** to **51–55%** once the embargo and the item
  universe filter were applied. The old number lived on free `sign(0)==sign(0)` penny hits (31% of
  its rows) and on labels resolved from inside the validation window; both are gone.

## Attribution leg (3) declined on cost

The pre-registration's third leg — the leaky arm (`EARLY_STOPPING=1`) at matched `--max-items`, to
size how much of the old number was the trainer leak specifically — was **not run**. It is descriptive
(no bar), and the verdict is already decided by legs 1 and 2. Arm A alone cost **74.6 min** of wall
clock at 1500 items (2.5× over the 30-minute cap for a single arm; the full pair would be ~2 hours),
which is not worth spending to quantify a contribution to a number already refuted. If ever wanted,
it must match `--max-items 1500` to pair with this arm A per the void condition.

## Operational finding: this harness's cohort is expensive to power

`--max-items 1500` is what it took to resolve the ≥$1 paired interval (the default 200 yields **15**
≥$1 items), and it puts one honest arm at ~75 min. A future re-read of this harness must either shard
harder or accept it does not fit the daily cap. The cheaper fix is the one C4 has always pointed at:
rebuild the item selection onto the ≥$1 served universe instead of `ORDER BY row_count DESC` over
penny backfill items, so ≥$1 rows are not a thin minority of a cheap cohort.

## Consequence

`FEATURE_GROUP_ALLOWLIST = ["price_technicals"]` removes `cross_sectional`, `events`, supply/rarity
and item-identity from **every** served horizon on the strength of a number that does not reproduce.
This read refutes the two headline justifications (cross-sectional +3.5, events +3.0 at 30d) on the
honest, embargoed, ≥$1 paired interval. It does **not** by itself license restoring any group — each
removed group is its own question, and the full "keep only `price_technicals`" configuration has
never been tested even now. What it does establish: **C4 is reopened, and the allowlist can no longer
cite 2026-07-19 as its basis.** The next honest step is to decide the allowlist on the served cohort,
which needs the cohort rebuild above, not another run of this harness as it stands.

## The C7 asymmetry held

The pre-registered prediction was that leg 1 returns null/unresolved, and it did. The capacity-leak
argument — that the leak inflates the higher-capacity `full` arm most and so *suppressed* a true
positive delta — is **not** supported: the honest delta is negative, not more positive, so the old
+3.5 was not a leak-suppressed real effect but an artifact of the penny cohort and the missing
embargo. Consistent with the leak being a level shift that cancels in a within-run contrast
(`docs/changelog/2026-08-13-the-leak-is-worth-two-points-and-it-pays-the-placebo.md`): the trainer was
never the main thing wrong with this number.
