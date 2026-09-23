# CS2 Oracle ML Upgrade — Design Spec

**Date:** 2026-09-18
**Goal:** Make cs2-oracle an impressive, portfolio-grade ML/AI project while maintaining accuracy and quality.
**Constraints:** CPU-only, free tooling only, training time under 30 minutes total.

## Current State

- LightGBM quantile regression (q50) for 4 horizons (3d, 7d, 14d, 30d), ~926 items, ~993K rows
- Hand-rolled conformal calibration (signed split conformal, climatology scale with James-Stein shrinkage)
- Exceedance head (P(|move|>cost)), anomaly GBM, ranking head — all gated by env flags
- Training time: ~14 min (conformal CV is 57% of that)
- FastAPI with ~25 endpoints, no frontend
- Point prediction is decorative (centre ≈ last price). Product is the calibrated range.
- Every band-conditioning experiment (7/7) loses to flat pooling
- Label noise ceiling: IC achieved ~0.03 vs ceiling 0.61–0.96. Headroom exists but is extremely hard to capture.

## Phase 2: MLOps & Experiment Rigor (Build First)

### 2.1 MLflow Tracking (Local)

Wrap the existing training loop so every run logs:

- **Parameters:** all hyperparameters, feature flags, data window, item count
- **Metrics:** MAE, IC, coverage, band width, exceedance AUC — per horizon
- **Artifacts:** trained model files, feature importance plots, calibration curves

MLflow UI (`mlflow ui`) provides free comparison dashboards. No external service needed.

**Files:**

- `models/forecaster.py` — add MLflow logging around existing training
- `mlflow/` or `mlruns/` — local tracking directory (gitignored)

### 2.2 Walk-Forward Backtesting

Replace train-once-score-once with temporal walk-forward evaluation:

- Train on data up to date T
- Predict T+h
- Slide T forward by stride S (e.g., 7 days)
- Repeat across the full evaluation window
- Score: coverage, interval width, MAE, IC at each step

This is the gold standard for time series evaluation. It directly proves understanding of temporal leakage.

**Files:**

- New `models/walk_forward.py` — walk-forward evaluator
- Integrates with MLflow — each walk-forward fold is a child run

### 2.3 Data Drift Detection (Evidently AI)

Generates drift reports comparing current feature distributions to a reference window. Catches: frozen quotes, source changes, label seams.

- Runs as a step in the daily GH Actions chain
- Writes HTML + JSON reports
- API exposes `/monitoring/drift` endpoint

**Files:**

- New `monitoring/drift.py` — Evidently-based drift detection
- `api/routes/monitoring.py` — new monitoring routes
- GH Actions workflow — add drift check step
- `requirements.txt` — add `mlflow`, `evidently`

### 2.4 Model Registry

MLflow model registry tracks which model version is "production" vs "staging." Walk-forward results determine promotion. Simple but shows ML maturity.

## Phase 1: Probabilistic Forecasting Upgrade (Build Second)

### 1.1 NGBoost Distributional Head

Train NGBoost alongside LightGBM. Outputs a full Normal (or LogNormal) distribution per prediction rather than a single quantile.

- **Not replacing LightGBM** — running both, letting walk-forward decide which earns production
- Train on h=7 and h=14 only (where IC ceiling headroom is largest) to stay under 30 min
- Expand to other horizons only if it wins
- Gated by `NGBOOST_HEAD=1` env flag (established pattern)

**Key risk:** Per-item conditioning has consistently overfit on this panel. NGBoost's per-prediction variance may hit the same wall. The walk-forward framework from Phase 2 makes this a fair test.

**Files:**

- New `models/ngboost_head.py` — NGBoost distributional model
- `models/forecaster.py` — wire as optional head

### 1.2 MAPIE Conformal Upgrade (CQR)

Replace hand-rolled conformal calibration with MAPIE's Conformalized Quantile Regression:

- Produces adaptive intervals (wider when uncertain, narrower when confident)
- Theoretical marginal coverage guarantees
- Conditions through residual quantiles, not a separate scale function — fundamentally different mechanism from the 7 refuted band-conditioning approaches

Keep old conformal code behind a flag for A/B comparison.

**Files:**

- `models/conformal.py` — refactor to use MAPIE CQR, old code behind flag
- `requirements.txt` — add `ngboost`, `mapie`

### 1.3 Interval Scoring Framework

Standardized evaluation comparing all interval methods:

- **Coverage:** does the interval contain the true value X% of the time?
- **Width:** narrower is better at matched coverage
- **Adaptivity:** does width correlate with actual prediction difficulty?
- All logged to MLflow for comparability

**Files:**

- New `evaluation/interval_scoring.py` — standardized interval comparison metrics

## Phase 3: React Frontend Dashboard (Build Last)

### Tech Stack

- Next.js 14 + TypeScript
- Recharts (free, React-native charting)
- Tailwind CSS
- Vercel free tier hosting
- Consumes existing FastAPI endpoints (add CORS middleware)

### Pages

**3.1 Market Overview (`/`)**

Grid of items: current price, predicted range (sparkline/bar), exceedance probability, anomaly flags. Sortable/filterable by category, price tier, volatility.

**3.2 Item Detail (`/items/[id]`)**

- Price history chart with prediction intervals overlaid
- Horizon selector (3d/7d/14d/30d)
- Accuracy panel: historical band coverage, visual hit/miss
- Feature importance
- Anomaly timeline

**3.3 Model Performance (`/performance`)**

- Coverage vs target per horizon over time
- Band width trends
- Walk-forward backtest results
- Model comparison table (LightGBM vs NGBoost vs climatology)

**3.4 Monitoring (`/monitoring`)**

- Data drift report (Evidently)
- Feature distribution shifts
- Data freshness / pipeline health
- Model version history (MLflow registry)

### API Changes Needed

- CORS middleware on FastAPI
- New aggregated endpoints: `/monitoring/drift-summary`, `/accuracy/coverage-over-time`

## Build Order

| Phase | What | ML Impressiveness | Accuracy Impact | Effort |
|-------|------|-------------------|-----------------|--------|
| 2 — MLOps | MLflow, walk-forward, drift | High (rigor) | None (infra) | Medium |
| 1 — Probabilistic | NGBoost, MAPIE/CQR, scoring | Very high (SOTA) | Possible gain | Medium-High |
| 3 — Frontend | Next.js dashboard, 4 pages | Medium (full-stack) | None (visibility) | High |

Each phase is independently shippable. Order: 2 → 1 → 3.

## Training Time Budget

| Component | Current | After Upgrade |
|-----------|---------|--------------|
| Data prep | 16s | 16s |
| Feature engineering | 9s | 9s |
| LightGBM (4 horizons) | ~335s | ~335s |
| Conformal CV | ~487s | ~487s (MAPIE may be faster) |
| NGBoost (h=7, h=14 only) | — | ~300-500s est. |
| MLflow logging | — | ~5s |
| Drift detection | — | runs separately in CI |
| **Total** | **~847s (14 min)** | **~1150-1350s (19-23 min)** |

Under 30 min cap with margin.

## Portfolio Story

"I built a production ML forecasting system for the CS2 skin market: a calibrated range forecast with proper uncertainty quantification (NGBoost + conformal prediction), rigorous temporal evaluation (walk-forward backtesting), automated drift monitoring (Evidently), full experiment tracking (MLflow), and an interactive React dashboard. I evaluated state-of-the-art distributional models against simpler baselines and documented where they do and don't help — the process and rigor matter as much as the results."
