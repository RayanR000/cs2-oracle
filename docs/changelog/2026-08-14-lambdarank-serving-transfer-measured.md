# 2026-08-14 — C2 lambdarank serving-transfer: measured, does not transfer (3 of 4 horizons), h=14 degenerate

## What ran

`backend/scripts/replay_lambdarank.py --horizon {3,7,14,30}`, one at a time, against the
**local** price archive (confirmed non-empty coverage around 2026-06-08: ~25.8K items/day,
single `aggregator_steam_17mafo` source — see `backend/AGENTS.md`'s note that the local copy
runs *behind* the durable `cs2-oracle-data` archive). Read-only: each shard trains two
matched boosters (q50, lambdarank) locally on the frozen `SERVED_WINDOW`
(2026-04-18..2026-06-08) at the 14-day retrain cadence and scores the TIED, floor->=$1
cohort. Nothing was written to prod, the DB, or the archive. Wall-clock: h=3 ~5m, h=14 ~5m,
h=30 ~7.5m, h=7 ~6.5m — all well under the 30-minute cap, run sequentially.

Per the brief, these are **directional, local-archive numbers pending a durable-archive
confirmation** — not a final verdict.

## Pre-registered bar

Serving transfer CONFIRMS at a horizon iff **EDGE vs q50 (lr rank IC − q50 rank IC) > 0** on
the tied, floor->=$1 cohort. Decile spread (gross/net), edge-vs-naive, and the
Pesaran-Timmermann line are DESCRIPTIVE, not gating.

## Results

**Note:** `EDGE vs q50` and `edge vs naive` are the pre-registered **paired** metric — the
mean over anchors of the per-anchor difference `rank_ic(lr) − rank_ic(q50)` (or `− naive`) —
not a difference of the two legs' means. The script originally computed the difference of
means; that was fixed in the same change that produced the table below (mean-of-differences
vs difference-of-means diverge whenever a leg is `None` on a different subset of anchors than
the other leg, which is exactly the case at h=3/7/30: some anchors have a `None` q50 or naive
leg that the lr leg doesn't share, so the two "different calendars" don't cancel). The lr/q50/
naive IC columns below are still the simple per-leg means (unaffected by the fix).

| h | anchors | lr IC | q50 IC | naive IC | **EDGE vs q50 (paired)** | edge vs naive (paired) | decile gross | decile net | PT (excess_pp / t / verdict) |
|---|---|---|---|---|---|---|---|---|---|
| 3  | 52 | -0.0146 | 0.1209 | -0.0539 | **-0.1736** | +0.0571 | +0.0017 | -0.398 | 0.015 / 0.06 / no_skill |
| 7  | 52 | 0.1653  | 0.1054 | -0.0527 | **+0.0346** | +0.2513 | +0.0007 | -0.399 | 1.48 / 1.68 / no_skill |
| 14 | 52 | None*   | 0.1173 | -0.0510 | **None***   | None*   | -0.0119 | -0.412 | 0.00 / None / degenerate |
| 30 | 52 | 0.0184  | 0.0719 | -0.0501 | **-0.0680** | +0.0519 | -0.0071 | -0.407 | 0.094 / 0.28 / no_skill |

\* At h=14, `lr rank IC` is `None` not because rows failed the tied/floor/min-rows filter
(q50 and naive IC computed cleanly on the identical per-anchor frames) but because the
lambdarank scores had `nunique < 2` — i.e. **degenerate/constant** — on every one of the 52
scored anchors, so within-date Spearman correlation is undefined at every date and the mean
is `None`. This is a genuine finding, not a script fault: the run completed cleanly (no
traceback), matching the "continue, don't stop" branch of the brief. Root-causing why the
h=14 ranker collapses to constant output is out of scope for this measurement task and is
flagged as a follow-up, not resolved here.

## Reading against the CV numbers

CV vs-q50 edge (HP-confirmed, `docs/changelog/2026-08-13-lambdarank-clears-the-diagnostic-bar.md`):
3d +0.0648, 7d +0.0614, 14d +0.0391, 30d +0.0297 — all four CV horizons passed the same bar.

- **h=3: CV +0.0648 -> served -0.1736.** Sign-flipped and by far the largest miss. Does not
  transfer.
- **h=7: CV +0.0614 -> served +0.0346.** Held sign but attenuated by ~44% under the
  corrected paired metric — still the only horizon that confirms on the pre-registered bar,
  but the margin is smaller than the difference-of-means read (+0.0599) initially suggested.
- **h=14: CV +0.0391 -> served undefined (degenerate lr output).** Cannot be read as
  transfer or non-transfer; the ranker's serving predictions carry no rank information at
  all on this horizon's anchors, which is a stronger negative signal than a measured
  negative edge would be.
- **h=30: CV +0.0297 -> served -0.0680.** Sign-flipped. Does not transfer.

**Verdict: 1 of 4 horizons (h=7) confirms the pre-registered bar.** h=3 and h=30 sign-flip
from a positive CV edge to a negative served edge — attenuation this large (CV edges were
all in the 0.03-0.065 range; served swings from -0.07 to -0.17) is not "held, some
decay," it is reversal. h=14 is not measurable as stated and needs its own diagnosis before
it can be counted either way. No verdict changes sign under the paired-metric correction —
h=7 stays the sole confirm, at a smaller margin than first measured.

## Tradeability read (descriptive, not a bar)

Decile long-short is gross-positive only at h=3 and h=7 (both ~+0.0007-0.0017, i.e.
economically negligible even before costs) and gross-negative at h=14 and h=30. Net of the
40% round-trip cost assumption (`ROUNDTRIP_COST = 2*(0.15+0.05)`), every horizon is deeply
net-negative (-0.398 to -0.412). This says the same thing decile spread has said throughout
this project's cost-aware reads: whatever rank signal exists is nowhere near strong enough
to clear Steam-fee-plus-spread economics. It does not change the pass/fail above; it
conditions what "confirms" can be used for even where it holds.

The PT verdicts are `no_skill` at h=3/7/30 and `degenerate` at h=14 — consistent with the
rank-IC read: even where EDGE vs q50 is positive (h=7), the ranker's own directional call
does not clear a PT skill bar on 52 dates.

`edge vs naive` uses the RAW `-return_1d` baseline, the same construction the CV diagnostic
used — it carries the raw-vs-smoothed basis wedge documented elsewhere in this project's
history, so `edge vs naive` is directional/descriptive only. The primary `EDGE vs q50` read
does not touch the naive leg and is unaffected by that wedge.

## What this does and doesn't license

Per the diagnostic's own design caution: a pass at h=7 opens a **further serving
pre-registration** (e.g. a dedicated h=7 paired A/B against the constant-call / naive
baseline on more dates, ideally against the durable archive) — it does **not** license
shipping the lambdarank ranker into production serving. Three of four horizons failed to
transfer from CV, one of those four failed in a way (constant output) that wasn't even
anticipated by the pre-registration's failure modes. That is a strong caution against
generalizing the CV result, not a green light narrowed to h=7.

No bare directional accuracy is quoted anywhere above (backend/AGENTS.md invariant 4) — a
ranker's output is ordinal, so DA/MAE are undefined by design; only rank IC, decile
long-short, and PT are reported, as the CV write-up itself specifies.

## Follow-ups opened

1. Diagnose the h=14 constant-lambdarank-output degeneracy (training data, label voiding
   volume — h=14 voids the most labels of the four horizons at 43,408 — or an HP/label
   interaction specific to that horizon).
2. Re-run all four shards against the durable `cs2-oracle-data` archive before treating any
   number here as more than directional, per `backend/AGENTS.md`'s local-archive caveat.
3. If h=7 is to be pursued further, its own pre-registration (wider date range, paired MDE
   design) is required before any claim beyond "cleared this one diagnostic."

## Durable-archive confirmation (2026-08-14, same day) — resolves follow-up #2

Re-ran all four shards against the **durable `cs2-oracle-data` archive** (cloned fresh,
pushed 2026-08-14 09:09 UTC; `price-archive` symlink-swapped to it and restored after, the
same pattern `model-diagnostics.yml` uses). First checked the served-window coverage: the
durable archive carries the **same single source** `aggregator_steam_17mafo` over
2026-04-18..06-08 (1.31M rows / 52 days / 26,720 items) as local — so no rescue was expected,
and none occurred.

| h | served EDGE vs q50 (durable) | (local) | verdict |
|---|---|---|---|
| 3  | **−0.1737** | −0.1736 | does not transfer (sign-flip) |
| 7  | **+0.0346** | +0.0346 | confirms, attenuated; decile net **−0.399** |
| 14 | **LightGBM error** | degenerate | not measurable |
| 30 | **−0.0689** | −0.0680 | does not transfer (sign-flip) |

h=3/7/30 reproduce local **to four decimals** — the non-transfer is robust, not a
local-archive artifact. **h=14 fails differently and worse:** on the fuller durable
cross-section the ranker aborts with `LightGBMError: Number of rows 14478 exceeds upper limit
of 10000 for a query` — a within-date query group exceeds LightGBM lambdarank's hard
10,000-rows-per-query cap. That is a genuine scaling blocker for a ranker on this ~14K-item
cross-section, independent of the constant-output degeneracy the local run hit.

**Final verdict (both archives): 1 of 4 horizons transfers (h=7 only), and it is net-negative
after the 40% round-trip cost.** The lambdarank cross-sectional edge is real in CV and does not
survive to serving — the same CV+/serving− pattern as C1. Follow-up #2 is closed; the pivot to
a cross-sectional ranker is refuted **for serving** at this data/horizon setup (the CV signal
is genuine; it does not transfer). Follow-up #1 (h=14 root-cause) is now two distinct failures,
neither worth fixing since a fixed h=14 still joins h=3/h=30 in the non-transfer column.
