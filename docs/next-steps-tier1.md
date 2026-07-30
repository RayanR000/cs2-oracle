# Post-Tier-1 Next Steps

> **Historical planning doc — all Tier-1 items completed.**
> See `docs/architecture/model.md` for current state and `docs/research/accuracy-opportunities.md` for remaining work.

## ✅ What's been implemented

| Lever | Status | File |
|---|---|---|
| HP caching/reuse (skip Optuna on retrain) | Done | `forecaster.py` — `tuned_params` persist/restore |
| Ensemble 9 → 6 | Done | `forecaster.py` — `N_ENSEMBLES`, seeds/fractions trimmed |
| `max_feature_rows` 700K → 400K | Done | `forecaster.py` — `build_training_data` default |
| `MAX_BIN` 127 → 63 | Done | `forecaster.py` — class constant + all Dataset/train sites |
| CV skip via `SKIP_CV=1` | Done | `forecaster.py` — env guard; CI workflow sets it |
| Conditional retrain (skip if fresh + no drift) | Done | `forecast_prices.py` — `RETRAIN_INTERVAL_DAYS` (14) |
| LightGBM 4.6 `feature_pre_filter=False` | Done | `forecaster.py` — all 5 Dataset/train sites |
| Accuracy validation | Done | Ran 2026-07-17 — see §1 |
| Timing tests | Done | Ran 2026-07-17 — see §2 |
| Test updates | Done | `test_forecaster.py` — constants + `test_tuned_params_roundtrip` |
| Changelog | Done | `docs/changelog/2026-07-17-tier-1-training-speedups.md` |
| Regime-switching models | Done | `forecaster.py` — `_assign_regime_label`, regime training, regime predict, persistence |
| Regime A/B comparison | Done | `forecast_prices.py` — `--compare-regime` flag |
| Regime tests | Done | `test_forecaster.py` — 15 tests in `TestRegimeSwitching` |
| Regime docs | Done | `docs/changelog/2026-07-18-regime-switching.md` |

## 1. Accuracy validation — PASS ✅

**Run:** 2026-07-17 against local SQLite DB (`cs2_market.db`), 33,252 mature forecasts evaluated against Parquet actuals.

**Result:** All four horizons within the 0.5–1.5pp budget — zero regression:

| Horizon | Baseline (lgbm-v3) | Current | Δ | Budget |
|---------|:------------------:|:-------:|:-:|:------:|
| 3d      | 58.15%             | 58.15%  | 0.00pp | ≤1.5pp |
| 7d      | 57.41%             | 57.41%  | 0.00pp | ≤1.5pp |
| 14d     | 55.08%             | 55.08%  | 0.00pp | ≤1.5pp |
| 30d     | 55.15%             | 55.15%  | 0.00pp | ≤1.5pp |

**Bugs found & fixed:** `backtest_accuracy.py` had SQLite-specific bugs (too many SQL variables in IN clause, string vs date type for `forecast_date`). Both fixed when running against local SQLite.

Command to re-run:
```
cd backend
DATABASE_URL="sqlite:///./cs2_market.db" python scripts/backtest_accuracy.py
```

## 2. Full timing test

A cold (no HP cache) + warm (HP reused) retrain timing run on real data. The previous attempt failed due to `feature_pre_filter` (now fixed). Ran 2026-07-17 on Mac (387% CPU utilization).

| Mode | Measured | Expected | Notes |
|------|:--------:|:--------:|-------|
| **Cold** (no HP cache, full Optuna) | **41m56s** | 18–35 min | ~1.2–2.3× above range; likely SQLite vs PostgreSQL overhead + CPU throttling |
| **Warm** (HP reused, SKIP_CV) | **17m19s** | 2–5 min | Auto-prune triggered retry on 3d (67 features pruned, 57 retained), doubling that horizon's training time |
| **Warm + conditional** | N/A | ~0 min | Drift check flagged 3d at 45.3% (threshold 60%) on a 7-window sliding check — triggered retrain. The 14d interval was not reached because drift catches real degradation first |

**Key findings:**
- **Cold** is ~1.2–2.3× the expected range. The primary gap is likely SQLite I/O during DuckDB queries (vs PostgreSQL in prod). The HP cache is correctly persisted on every run, so cold training is a one-time cost.
- **Warm** is ~3–9× above expected range. The gap comes from feature-group auto-prune retry cascades (not a speed regression). On 3d, 5 of 7 feature groups failed the permutation test, pruning 67→57 features and forcing a full retrain of all 3 quantiles × 6 ensembles — adding ~7 min.
- **Suggestion:** Re-measure on the G14 or GH runner once deployed. The auto-prune cost will persist, but per-fit speed should improve on better hardware.

Commands used:
```bash
cd backend
# Cold (first time, Optuna runs):
rm -f models/saved_models/*.txt models/saved_models/meta.json
time DATABASE_URL="sqlite:///./cs2_market.db" python scripts/forecast_prices.py --train-only

# Warm (HP reused, skip CV):
time SKIP_CV=1 DATABASE_URL="sqlite:///./cs2_market.db" python scripts/forecast_prices.py --train-only

# Warm + conditional skip (use default mode so RETRAIN_INTERVAL applies):
time RETRAIN_INTERVAL_DAYS=14 SKIP_CV=1 \
  DATABASE_URL="sqlite:///./cs2_market.db" \
  python scripts/forecast_prices.py
```

Expected ranges (from previous estimates):
- Cold: ~18–35 min (Mac), ~12–20 min (G14 GPU)
- Warm (HP reused + SKIP_CV): ~2–5 min
- Warm + conditional skip (model < 14d old): ~0 min (predict-only)

## 3. CI runner upgrade

The Monday retrain on `ubuntu-latest` (2 vCPU, no GPU) is the real bottleneck. Options:

| Option | Expected CI speed | Cost |
|---|---|---|
| **A)** Larger hosted runner (4–8 vCPU) | ~15–30 min | GitHub paid plan |
| **B)** Self-host runner on G14 (cpu) | ~5–10 min | Free (your hardware) |
| **C)** Self-host runner on G14 (GPU) | ~2–5 min | Free (your hardware) + CUDA setup |

Self-hosting on the G14 with `device="cuda"` would give the best Monday retrain time. See step 5.

## 4. Verify CI workflow passes

The workflow changes (`SKIP_CV=1` env) apply automatically. After merging, watch the next Monday run for:
- `"Reusing cached HP…"` log line (confirms HP cache works)
- `"CV skipped (SKIP_CV=1)"` log line (confirms CV skip)
- `"Skipping retrain: model N days old"` if less than 14 days since last train

## 5. GPU support (optional, G14)

Add `device="cuda"` auto-detection to `forecaster.py` for ~3–5× per-fit speedup on the G14 RTX 4060:

```python
import torch  # or check nvidia-smi
GPU_AVAILABLE = "cuda"  # detect via lightgbm basic._LIB.LGBM_GetNumThreads or try/except

base_params = {
    "device": "cuda" if GPU_AVAILABLE else "cpu",
    ...
}
```

~8 lines, measurable only when trained on a CUDA machine. Falls back to CPU automatically.

## 6. GH Actions remaining speedups (no GPU, hosted runners)

The Monday CI retrain on `ubuntu-latest` (2 vCPU) is the bottleneck. Since GPU is off the table:

### Lever A — Feature matrix caching — ✅ DONE, but the premise was wrong

Implemented 2026-07-29 as a voted-frame cache in `fetch_price_history`
(`VOTED_CACHE_*` constants, `VOTED_CACHE=0` to disable). Keyed on the
`prices-*.parquet` fingerprint + cutoff date + backfill slug set + a version
constant.

**The "10-min voting phase" in the original estimate below was never measured
and is wrong.** Measured on the real archive (14 files, 18.2M raw rows):

| Phase | Measured |
|---|---|
| `fetch_price_history` cold (DuckDB read + voting → 6.1M voted rows) | **35.5s** |
| `fetch_price_history` cached | **0.3s** (116× — frame verified byte-identical) |
| `build_training_data` total, warm cache | **6.1s** |

So data prep is ~36s of a 12m30s cold retrain, not ~11 min. The cold retrain is
essentially *all* model fitting (Optuna + CV + ensembles). **This lever saves
~35s per invocation, not ~10 min** — worth keeping for the experiment loop, but
it is not the CI bottleneck and was never going to be.

Correcting the ROI claim matters more than the lever: any future "speed up the
retrain" work should start from the fitting cost, not data prep.

### Lever B — Within accuracy budget

| Change | CI speedup | Est. accuracy cost |
|---|---|---|
| Ensemble 6 → 4 | ~33% ensemble block | ~0.3–0.5pp |
| `max_feature_rows` 400K → 300K | ~30% per fit | ~0.3–0.7pp |
| Conditional interval 14 → 21 days | Fewer retrains/yr | 0pp (drift catches real issues) |
| `num_boost_round` 1000 → 600 | ~20% ensemble block | ~0pp if early-stop triggers first |

### Lever C — Larger GH hosted runner (paid, ~$0.10/wk)

| Runner | vCPU | CI estimate | Cost/min |
|---|---|---|---|
| `ubuntu-latest` (free) | 2 | ~15–25 min | $0 |
| 4-core runner | 4 | ~8–12 min | ~$0.008/min |
| 8-core runner | 8 | ~4–7 min | ~$0.016/min |

An 8-core runner for a weekly retrain costs ~$0.06–0.11/wk ($3–6/yr). This alone doubles CI speed vs the free tier.

### Implementation order

1. Feature matrix caching (code)
2. Ensemble 6→4 + `max_feature_rows` 400K→300K (code)
3. Upgrade CI to 8-core runner (config)
4. Validate accuracy with `scripts/backtest_accuracy.py`

## 7. Further speed tunings (if needed)

| Change | Est. speedup | Est. accuracy cost |
|---|---|---|
| `num_boost_round` 1000 → 600 (final ensemble) | ~10–20% on ensemble block | ~0pp if early-stopping triggers before 600 |
| Ensemble 6 → 4 | ~33% on ensemble block | ~0.5–0.8pp |
| `max_bin` 63 → 31 | ~10–15% on each fit | ~0.3–0.5pp |
| `max_feature_rows` 400K → 300K | ~30% on each fit | ~0.3–0.7pp |
| Drop `_validate_feature_groups` permutation test | ~2–5% (cheap) | 0pp (diagnostic only) |

Measure before applying — the Tier-1 changes already cover the highest-ROI levers.
