# Temporal Fusion Transformer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a TFT model that learns from raw price/volume sequences to produce multi-horizon return forecasts (h=3/7/14/30), feature-flag gated alongside the existing LightGBM centre.

**Architecture:** Custom PyTorch TFT operating on 60-day lookback windows of daily (price, volume) per item, producing q50 return predictions at 4 horizons simultaneously. OOF predictions feed the existing conformal band infrastructure. Gated behind `TFT_CENTRE=1`.

**Tech Stack:** PyTorch (CPU), custom TFT implementation (no pytorch-forecasting dependency), existing conformal/climatology infrastructure for bands.

**Spec:** This plan — no separate spec document.

## Global Constraints

- Python 3.13 locally, 3.11 in CI — no 3.13-only syntax
- CPU-only training — no CUDA dependency; model must train in <60 min on the 926-item ≥$1 cohort
- `torch>=2.0.0` added to requirements.txt — no other new dependencies
- Feature flag `TFT_CENTRE=1` (off by default) — LightGBM remains the shipped model
- OOF residuals must flow through `conformal.calibrate_signed()` — TFT does not ship its own bands
- Never import `torch` at module level in `forecaster.py` — guard behind the flag to avoid breaking non-TFT runs
- All tests must work without PyTorch installed (skip with `pytest.importorskip("torch")`)
- Run tests from `backend/` via `venv/bin/python -m pytest tests/test_<name>.py -q`

## File Structure

```
backend/models/tft/
  __init__.py       — Public API: TFTForecaster class
  dataset.py        — SequenceDataset, collate_fn, build_dataloaders
  components.py     — GRN, GLU, VSN building blocks
  model.py          — TemporalFusionTransformer nn.Module
  trainer.py        — Training loop, CV, early stopping, OOF collection
backend/tests/
  test_tft_dataset.py
  test_tft_components.py
  test_tft_model.py
  test_tft_trainer.py
  test_tft_integration.py
```

---

### Task 1: Dependencies and Package Scaffolding

**Files:**
- Modify: `backend/requirements.txt`
- Create: `backend/models/tft/__init__.py`
- Create: `backend/models/tft/dataset.py` (stub)
- Create: `backend/models/tft/components.py` (stub)
- Create: `backend/models/tft/model.py` (stub)
- Create: `backend/models/tft/trainer.py` (stub)

**Interfaces:**
- Consumes: nothing
- Produces: importable `backend.models.tft` package; `torch` available in venv

- [ ] **Step 1: Add PyTorch to requirements.txt**

Add `torch>=2.0.0` to requirements.txt, below the existing `lightgbm` line:

```
torch>=2.0.0
```

- [ ] **Step 2: Install torch in the venv**

Run: `venv/bin/python -m pip install 'torch>=2.0.0'`

Verify: `venv/bin/python -c "import torch; print(torch.__version__)"`

- [ ] **Step 3: Create the package directory and __init__.py**

```python
# backend/models/tft/__init__.py
from __future__ import annotations
```

- [ ] **Step 4: Create stub files**

Create empty `dataset.py`, `components.py`, `model.py`, `trainer.py` in `backend/models/tft/`, each with just:

```python
from __future__ import annotations
```

- [ ] **Step 5: Verify import**

Run: `cd backend && venv/bin/python -c "import models.tft; print('ok')"`
Expected: `ok`

- [ ] **Step 6: Commit**

```bash
git add models/tft/ requirements.txt
git commit -m "feat: scaffold TFT package and add torch dependency"
```

---

### Task 2: Sequence Dataset

**Files:**
- Create: `backend/models/tft/dataset.py`
- Create: `backend/tests/test_tft_dataset.py`

**Interfaces:**
- Consumes: voted price DataFrame with columns `(item_id, date, price, volume)` — the output of `ItemForecaster.fetch_price_history()`
- Produces: `SequenceDataset(df, lookback, horizons, price_floor)` → PyTorch Dataset yielding `(past_features, static_features, future_known, targets, meta)` tuples; `build_dataloaders(df, train_dates, val_dates, ...) → (train_loader, val_loader)`

The dataset converts a flat item-day DataFrame into windowed sequences. Each sample anchors at a date `d` for an item and includes:
- `past_features`: `[lookback, n_past]` — log-price, log-volume, return_1d, rolling volatility (std/price over 20d), RSI-14-like momentum
- `static_features`: `[n_static]` — price tier (int)
- `future_known`: `[n_horizons, n_future]` — normalized horizon index (3/7/14/30 → 0..1)
- `targets`: `[n_horizons]` — percentage return at each horizon
- `meta`: dict with `item_id`, `date` for bookkeeping

- [ ] **Step 1: Write the test file**

```python
# backend/tests/test_tft_dataset.py
import pytest
import numpy as np
import pandas as pd

torch = pytest.importorskip("torch")


def _make_price_df(n_items=5, n_days=120, seed=42):
    """Synthetic daily price data matching voted frame schema."""
    rng = np.random.RandomState(seed)
    rows = []
    base_date = pd.Timestamp("2025-01-01")
    for i in range(n_items):
        price = 10.0 + rng.randn() * 2
        for d in range(n_days):
            price *= np.exp(rng.randn() * 0.02)
            rows.append({
                "item_id": f"item_{i}",
                "date": (base_date + pd.Timedelta(days=d)).date(),
                "price": round(max(price, 0.5), 2),
                "volume": max(1, int(rng.poisson(20))),
            })
    return pd.DataFrame(rows)


class TestSequenceDataset:

    def test_dataset_length(self):
        from models.tft.dataset import SequenceDataset
        df = _make_price_df(n_items=3, n_days=100)
        ds = SequenceDataset(df, lookback=60, horizons=[3, 7, 14, 30])
        # Each item has 100 days. Valid anchors: day 60..69 (day+30 <= 99).
        # 3 items × 10 valid anchors = 30
        assert len(ds) == 30

    def test_sample_shapes(self):
        from models.tft.dataset import SequenceDataset
        df = _make_price_df(n_items=2, n_days=100)
        ds = SequenceDataset(df, lookback=60, horizons=[3, 7, 14, 30])
        past, static, future, targets, meta = ds[0]
        assert past.shape == (60, 5)       # lookback × n_past_features
        assert static.shape == (1,)        # price_tier
        assert future.shape == (4, 1)      # n_horizons × n_future_features
        assert targets.shape == (4,)       # n_horizons
        assert "item_id" in meta
        assert "date" in meta

    def test_targets_are_percentage_returns(self):
        from models.tft.dataset import SequenceDataset
        df = _make_price_df(n_items=1, n_days=100)
        ds = SequenceDataset(df, lookback=60, horizons=[3])
        _, _, _, targets, meta = ds[0]
        # Manually compute expected return
        item_df = df[df["item_id"] == meta["item_id"]].sort_values("date")
        anchor_idx = item_df[item_df["date"] == meta["date"]].index[0]
        anchor_pos = list(item_df.index).index(anchor_idx)
        p_anchor = item_df.iloc[anchor_pos]["price"]
        p_target = item_df.iloc[anchor_pos + 3]["price"]
        expected = (p_target - p_anchor) / p_anchor * 100
        assert abs(targets[0].item() - expected) < 0.01

    def test_no_nan_in_past_features(self):
        from models.tft.dataset import SequenceDataset
        df = _make_price_df(n_items=2, n_days=100)
        ds = SequenceDataset(df, lookback=60, horizons=[3, 7, 14, 30])
        for i in range(min(10, len(ds))):
            past, _, _, _, _ = ds[i]
            assert not torch.isnan(past).any(), f"NaN in sample {i}"

    def test_nan_targets_when_future_missing(self):
        from models.tft.dataset import SequenceDataset
        df = _make_price_df(n_items=1, n_days=75)
        # lookback=60, max horizon=30 → valid anchors: day 60..44
        # But day 60+30=90 > 74, so no valid samples at h=30
        ds = SequenceDataset(df, lookback=60, horizons=[3, 7, 14, 30])
        # Should still create samples where at least h=3 is valid
        assert len(ds) > 0
        # The last sample should have NaN for h=30 if date+30 > last date
        # (dataset only creates samples where ALL horizons are valid)
        _, _, _, targets, _ = ds[0]
        assert not torch.isnan(targets).any()


class TestBuildDataloaders:

    def test_train_val_split(self):
        from models.tft.dataset import SequenceDataset, build_dataloaders
        df = _make_price_df(n_items=5, n_days=200)
        dates = sorted(df["date"].unique())
        split = len(dates) // 2
        train_dates = dates[:split]
        val_dates = dates[split:]
        train_dl, val_dl = build_dataloaders(
            df, train_dates, val_dates,
            lookback=60, horizons=[3, 7, 14, 30], batch_size=16,
        )
        assert len(train_dl) > 0
        assert len(val_dl) > 0
        batch = next(iter(train_dl))
        past, static, future, targets, meta = batch
        assert past.shape[0] == 16  # batch size
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_tft_dataset.py -q`
Expected: ImportError — `SequenceDataset` not defined

- [ ] **Step 3: Implement dataset.py**

```python
# backend/models/tft/dataset.py
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


PAST_FEATURE_NAMES = [
    "log_price", "log_volume", "return_1d", "volatility_20d", "momentum_14d",
]
N_PAST_FEATURES = len(PAST_FEATURE_NAMES)
N_STATIC_FEATURES = 1   # price_tier
N_FUTURE_FEATURES = 1   # normalized horizon index
PRICE_TIER_BINS = [0, 1, 3, 7, 15, 30, 70, 150, 400, 1000, float("inf")]


def _price_tier(price: float) -> int:
    for i, edge in enumerate(PRICE_TIER_BINS[1:]):
        if price < edge:
            return i
    return len(PRICE_TIER_BINS) - 2


def _preprocess_item(item_df: pd.DataFrame) -> dict:
    """Compute per-item time series arrays from a sorted item DataFrame."""
    prices = item_df["price"].values.astype(np.float64)
    volumes = item_df["volume"].values.astype(np.float64)
    dates = item_df["date"].values

    log_prices = np.log(np.maximum(prices, 1e-6))
    log_volumes = np.log(np.maximum(volumes, 1e-6))

    # return_1d: percentage return
    returns = np.zeros_like(prices)
    returns[1:] = (prices[1:] - prices[:-1]) / np.maximum(prices[:-1], 1e-6) * 100
    returns[0] = 0.0

    # volatility_20d: rolling std(return) / price (CV proxy)
    vol = np.zeros_like(prices)
    for t in range(len(prices)):
        start = max(0, t - 19)
        window = returns[start:t + 1]
        if len(window) >= 2:
            vol[t] = np.std(window) / max(np.mean(np.abs(prices[start:t + 1])), 1e-6) * 100
        else:
            vol[t] = 0.0

    # momentum_14d: (price - price_14d_ago) / price_14d_ago * 100, clamped
    mom = np.zeros_like(prices)
    for t in range(14, len(prices)):
        p_old = prices[t - 14]
        if p_old > 1e-6:
            mom[t] = (prices[t] - p_old) / p_old * 100

    # Stack: [T, 5]
    features = np.stack([log_prices, log_volumes, returns, vol, mom], axis=-1)

    median_price = float(np.median(prices))
    tier = _price_tier(median_price)

    return {
        "features": features.astype(np.float32),
        "prices": prices.astype(np.float64),
        "dates": dates,
        "tier": tier,
        "item_id": item_df["item_id"].iloc[0],
    }


class SequenceDataset(Dataset):
    """Sliding-window dataset over item price histories."""

    def __init__(
        self,
        df: pd.DataFrame,
        lookback: int = 60,
        horizons: list[int] | None = None,
        date_filter: list | None = None,
    ):
        self.lookback = lookback
        self.horizons = horizons or [3, 7, 14, 30]
        self.max_horizon = max(self.horizons)
        norm_max = float(self.max_horizon)
        self.horizon_indices = np.array(
            [h / norm_max for h in self.horizons], dtype=np.float32
        ).reshape(-1, 1)

        if date_filter is not None:
            date_set = set(date_filter)
            df = df[df["date"].isin(date_set)]

        self.samples: list[tuple[dict, int]] = []
        for item_id, group in df.groupby("item_id"):
            group = group.sort_values("date").reset_index(drop=True)
            if len(group) < lookback + self.max_horizon:
                continue
            item_data = _preprocess_item(group)
            n_days = len(group)
            for t in range(lookback, n_days - self.max_horizon):
                self.samples.append((item_data, t))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item_data, t = self.samples[idx]
        lb = self.lookback

        past = torch.from_numpy(item_data["features"][t - lb:t].copy())
        static = torch.tensor([item_data["tier"]], dtype=torch.long)
        future = torch.from_numpy(self.horizon_indices.copy())

        # Targets: percentage return at each horizon
        anchor_price = item_data["prices"][t]
        targets = np.array([
            (item_data["prices"][t + h] - anchor_price) / anchor_price * 100
            for h in self.horizons
        ], dtype=np.float32)
        targets = torch.from_numpy(targets)

        meta = {
            "item_id": item_data["item_id"],
            "date": item_data["dates"][t],
        }
        return past, static, future, targets, meta


def _collate_fn(batch):
    """Custom collate that handles the meta dict."""
    past = torch.stack([b[0] for b in batch])
    static = torch.stack([b[1] for b in batch])
    future = torch.stack([b[2] for b in batch])
    targets = torch.stack([b[3] for b in batch])
    meta = {
        "item_id": [b[4]["item_id"] for b in batch],
        "date": [b[4]["date"] for b in batch],
    }
    return past, static, future, targets, meta


def build_dataloaders(
    df: pd.DataFrame,
    train_dates: list,
    val_dates: list,
    lookback: int = 60,
    horizons: list[int] | None = None,
    batch_size: int = 256,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders from date-based splits."""
    train_ds = SequenceDataset(df, lookback=lookback, horizons=horizons,
                               date_filter=train_dates)
    val_ds = SequenceDataset(df, lookback=lookback, horizons=horizons,
                             date_filter=val_dates)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          collate_fn=_collate_fn, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=_collate_fn, drop_last=False)
    return train_dl, val_dl
```

- [ ] **Step 4: Run tests**

Run: `venv/bin/python -m pytest tests/test_tft_dataset.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add models/tft/dataset.py tests/test_tft_dataset.py
git commit -m "feat(tft): sequence dataset with sliding-window extraction"
```

---

### Task 3: TFT Building Blocks (GRN, GLU, VSN)

**Files:**
- Create: `backend/models/tft/components.py`
- Create: `backend/tests/test_tft_components.py`

**Interfaces:**
- Consumes: nothing (pure nn.Module components)
- Produces: `GatedLinearUnit(input_dim)`, `GatedResidualNetwork(input_dim, hidden_dim, output_dim, dropout, context_dim=None)`, `VariableSelectionNetwork(input_dim, num_inputs, hidden_dim, dropout, context_dim=None)` — all `nn.Module` subclasses

- [ ] **Step 1: Write the test file**

```python
# backend/tests/test_tft_components.py
import pytest

torch = pytest.importorskip("torch")


class TestGatedLinearUnit:
    def test_output_shape(self):
        from models.tft.components import GatedLinearUnit
        glu = GatedLinearUnit(input_dim=16)
        x = torch.randn(4, 16)
        out = glu(x)
        assert out.shape == (4, 16)

    def test_3d_input(self):
        from models.tft.components import GatedLinearUnit
        glu = GatedLinearUnit(input_dim=32)
        x = torch.randn(4, 10, 32)
        out = glu(x)
        assert out.shape == (4, 10, 32)


class TestGatedResidualNetwork:
    def test_output_shape(self):
        from models.tft.components import GatedResidualNetwork
        grn = GatedResidualNetwork(input_dim=16, hidden_dim=32, output_dim=16, dropout=0.1)
        x = torch.randn(4, 16)
        out = grn(x)
        assert out.shape == (4, 16)

    def test_with_context(self):
        from models.tft.components import GatedResidualNetwork
        grn = GatedResidualNetwork(input_dim=16, hidden_dim=32, output_dim=16,
                                    dropout=0.1, context_dim=8)
        x = torch.randn(4, 16)
        ctx = torch.randn(4, 8)
        out = grn(x, context=ctx)
        assert out.shape == (4, 16)

    def test_different_output_dim(self):
        from models.tft.components import GatedResidualNetwork
        grn = GatedResidualNetwork(input_dim=16, hidden_dim=32, output_dim=8, dropout=0.0)
        x = torch.randn(4, 16)
        out = grn(x)
        assert out.shape == (4, 8)


class TestVariableSelectionNetwork:
    def test_output_shape(self):
        from models.tft.components import VariableSelectionNetwork
        vsn = VariableSelectionNetwork(
            input_dim=8, num_inputs=5, hidden_dim=16, dropout=0.1,
        )
        # Input: [batch, num_inputs, input_dim]
        x = torch.randn(4, 5, 8)
        out, weights = vsn(x)
        assert out.shape == (4, 8)
        assert weights.shape == (4, 5, 1)
        # Weights should sum to ~1
        assert torch.allclose(weights.sum(dim=1), torch.ones(4, 1), atol=1e-5)

    def test_with_context(self):
        from models.tft.components import VariableSelectionNetwork
        vsn = VariableSelectionNetwork(
            input_dim=8, num_inputs=3, hidden_dim=16, dropout=0.0, context_dim=12,
        )
        x = torch.randn(4, 3, 8)
        ctx = torch.randn(4, 12)
        out, weights = vsn(x, context=ctx)
        assert out.shape == (4, 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_tft_components.py -q`
Expected: FAIL — `GatedLinearUnit` not defined

- [ ] **Step 3: Implement components.py**

```python
# backend/models/tft/components.py
from __future__ import annotations

import torch
import torch.nn as nn


class GatedLinearUnit(nn.Module):
    """GLU: splits input into value and gate, applies sigmoid gate."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, input_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        value, gate = out.chunk(2, dim=-1)
        return value * torch.sigmoid(gate)


class GatedResidualNetwork(nn.Module):
    """GRN: the core building block of TFT.

    Applies dense → ELU → dense → GLU → add & norm with optional context.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        context_dim: int | None = None,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.context_proj = (
            nn.Linear(context_dim, hidden_dim, bias=False)
            if context_dim is not None else None
        )
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.glu = GatedLinearUnit(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.skip = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim else None
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = self.skip(x) if self.skip is not None else x
        hidden = self.fc1(x)
        if self.context_proj is not None and context is not None:
            ctx = self.context_proj(context)
            if ctx.dim() < hidden.dim():
                ctx = ctx.unsqueeze(1).expand_as(hidden)
            hidden = hidden + ctx
        hidden = torch.nn.functional.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(self.glu(hidden))
        return self.layer_norm(hidden + residual)


class VariableSelectionNetwork(nn.Module):
    """VSN: learns which input variables matter most.

    Takes [batch, num_inputs, input_dim] and outputs a weighted sum [batch, input_dim]
    plus the softmax attention weights [batch, num_inputs, 1].
    """

    def __init__(
        self,
        input_dim: int,
        num_inputs: int,
        hidden_dim: int,
        dropout: float = 0.1,
        context_dim: int | None = None,
    ):
        super().__init__()
        self.grns = nn.ModuleList([
            GatedResidualNetwork(input_dim, hidden_dim, input_dim, dropout)
            for _ in range(num_inputs)
        ])
        flattened_dim = input_dim * num_inputs
        self.weight_grn = GatedResidualNetwork(
            flattened_dim, hidden_dim, num_inputs, dropout,
            context_dim=context_dim,
        )
        self.num_inputs = num_inputs

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, num_inputs, input_dim]
        processed = []
        for i, grn in enumerate(self.grns):
            processed.append(grn(x[:, i, :]))
        processed = torch.stack(processed, dim=1)  # [batch, num_inputs, input_dim]

        flat = x.reshape(x.shape[0], -1)  # [batch, num_inputs * input_dim]
        weights = self.weight_grn(flat, context=context)  # [batch, num_inputs]
        weights = torch.softmax(weights, dim=-1).unsqueeze(-1)  # [batch, num_inputs, 1]

        combined = (processed * weights).sum(dim=1)  # [batch, input_dim]
        return combined, weights
```

- [ ] **Step 4: Run tests**

Run: `venv/bin/python -m pytest tests/test_tft_components.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add models/tft/components.py tests/test_tft_components.py
git commit -m "feat(tft): GRN, GLU, VSN building blocks"
```

---

### Task 4: TFT Model Architecture

**Files:**
- Create: `backend/models/tft/model.py`
- Create: `backend/tests/test_tft_model.py`

**Interfaces:**
- Consumes: `GatedResidualNetwork`, `VariableSelectionNetwork` from `models.tft.components`
- Produces: `TemporalFusionTransformer(config)` — `nn.Module` with `forward(past, static, future) → predictions [batch, n_horizons]`; `TFTConfig` dataclass

The model follows the TFT paper (Lim et al. 2021) with simplifications:
1. Static covariates (price tier) → embedding → context vectors for VSNs
2. Past observed features → VSN → LSTM encoder
3. Horizon positions → projected → LSTM decoder (4 steps, one per horizon)
4. Multi-head attention from decoder over encoder states
5. Output: q50 prediction per horizon

- [ ] **Step 1: Write the test file**

```python
# backend/tests/test_tft_model.py
import pytest

torch = pytest.importorskip("torch")

from models.tft.dataset import N_PAST_FEATURES, N_STATIC_FEATURES, N_FUTURE_FEATURES


class TestTFTConfig:
    def test_defaults(self):
        from models.tft.model import TFTConfig
        cfg = TFTConfig()
        assert cfg.hidden_dim == 32
        assert cfg.n_horizons == 4
        assert cfg.lookback == 60


class TestTFTForwardPass:
    def _make_model(self):
        from models.tft.model import TemporalFusionTransformer, TFTConfig
        cfg = TFTConfig(
            n_past_features=N_PAST_FEATURES,
            n_static_features=N_STATIC_FEATURES,
            n_future_features=N_FUTURE_FEATURES,
            n_horizons=4,
            hidden_dim=16,
            num_heads=2,
            dropout=0.0,
            lookback=60,
            n_price_tiers=10,
        )
        return TemporalFusionTransformer(cfg)

    def test_output_shape(self):
        model = self._make_model()
        past = torch.randn(8, 60, N_PAST_FEATURES)
        static = torch.randint(0, 10, (8, 1))
        future = torch.randn(8, 4, N_FUTURE_FEATURES)
        out = model(past, static, future)
        assert out.shape == (8, 4)

    def test_gradient_flows(self):
        model = self._make_model()
        past = torch.randn(8, 60, N_PAST_FEATURES)
        static = torch.randint(0, 10, (8, 1))
        future = torch.randn(8, 4, N_FUTURE_FEATURES)
        out = model(past, static, future)
        loss = out.mean()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_deterministic_eval(self):
        model = self._make_model()
        model.eval()
        past = torch.randn(4, 60, N_PAST_FEATURES)
        static = torch.randint(0, 10, (4, 1))
        future = torch.randn(4, 4, N_FUTURE_FEATURES)
        with torch.no_grad():
            out1 = model(past, static, future)
            out2 = model(past, static, future)
        assert torch.allclose(out1, out2)

    def test_parameter_count_reasonable(self):
        model = self._make_model()
        n_params = sum(p.numel() for p in model.parameters())
        # Small model should be <500K params
        assert n_params < 500_000, f"Model has {n_params} params — too large for CPU"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_tft_model.py -q`
Expected: FAIL — `TFTConfig` not defined

- [ ] **Step 3: Implement model.py**

```python
# backend/models/tft/model.py
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.tft.components import GatedResidualNetwork, GatedLinearUnit, VariableSelectionNetwork


@dataclass
class TFTConfig:
    n_past_features: int = 5
    n_static_features: int = 1
    n_future_features: int = 1
    n_horizons: int = 4
    hidden_dim: int = 32
    num_heads: int = 4
    dropout: float = 0.1
    lookback: int = 60
    lstm_layers: int = 1
    n_price_tiers: int = 10


class TemporalFusionTransformer(nn.Module):

    def __init__(self, config: TFTConfig):
        super().__init__()
        self.config = config
        d = config.hidden_dim

        # --- Static variable processing ---
        self.tier_embedding = nn.Embedding(config.n_price_tiers, d)
        self.static_grn = GatedResidualNetwork(d, d, d, config.dropout)

        # --- Past variable selection ---
        self.past_proj = nn.Linear(config.n_past_features, d)
        self.past_vsn = VariableSelectionNetwork(
            input_dim=d // config.n_past_features if config.n_past_features <= d else 1,
            num_inputs=config.n_past_features,
            hidden_dim=d,
            dropout=config.dropout,
            context_dim=d,
        )
        # Simpler approach: project each timestep's features to hidden_dim
        self.past_input_proj = nn.Linear(config.n_past_features, d)

        # --- LSTM encoder ---
        self.encoder_lstm = nn.LSTM(
            input_size=d,
            hidden_size=d,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0.0,
        )
        self.encoder_glu = GatedLinearUnit(d)
        self.encoder_norm = nn.LayerNorm(d)

        # --- Future / decoder ---
        self.future_proj = nn.Linear(config.n_future_features, d)
        self.decoder_lstm = nn.LSTM(
            input_size=d,
            hidden_size=d,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0.0,
        )
        self.decoder_glu = GatedLinearUnit(d)
        self.decoder_norm = nn.LayerNorm(d)

        # --- Interpretable multi-head attention ---
        self.attention = InterpretableMultiHeadAttention(d, config.num_heads, config.dropout)
        self.attn_glu = GatedLinearUnit(d)
        self.attn_norm = nn.LayerNorm(d)

        # --- Output ---
        self.output_grn = GatedResidualNetwork(d, d, d, config.dropout)
        self.output_proj = nn.Linear(d, 1)

    def forward(
        self,
        past: torch.Tensor,      # [B, T, n_past]
        static: torch.Tensor,     # [B, 1] long
        future: torch.Tensor,     # [B, H, n_future]
    ) -> torch.Tensor:            # [B, H]
        B = past.shape[0]
        d = self.config.hidden_dim

        # Static context
        static_emb = self.tier_embedding(static.squeeze(-1))  # [B, d]
        static_ctx = self.static_grn(static_emb)               # [B, d]

        # Encode past
        past_proj = self.past_input_proj(past)  # [B, T, d]
        encoder_out, (h_n, c_n) = self.encoder_lstm(past_proj)
        encoder_out = self.encoder_norm(self.encoder_glu(encoder_out) + past_proj)

        # Decode future horizons
        future_proj = self.future_proj(future)  # [B, H, d]
        # Add static context to decoder input
        decoder_input = future_proj + static_ctx.unsqueeze(1)
        decoder_out, _ = self.decoder_lstm(decoder_input, (h_n, c_n))
        decoder_out = self.decoder_norm(self.decoder_glu(decoder_out) + future_proj)

        # Attention: decoder queries attend to encoder keys/values
        attn_out, _ = self.attention(decoder_out, encoder_out, encoder_out)
        attn_out = self.attn_norm(self.attn_glu(attn_out) + decoder_out)

        # Output
        out = self.output_grn(attn_out)  # [B, H, d]
        out = self.output_proj(out).squeeze(-1)  # [B, H]
        return out


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention with shared value projection for interpretability."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, self.d_k)  # shared across heads
        self.out_proj = nn.Linear(self.d_k, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** 0.5

    def forward(
        self,
        query: torch.Tensor,  # [B, T_q, d]
        key: torch.Tensor,    # [B, T_k, d]
        value: torch.Tensor,  # [B, T_k, d]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T_q, _ = query.shape
        T_k = key.shape[1]

        q = self.q_proj(query).view(B, T_q, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(key).view(B, T_k, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(value)  # [B, T_k, d_k] — shared

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, H, T_q, T_k]
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Each head attends over shared values
        # attn_weights: [B, n_heads, T_q, T_k], v: [B, T_k, d_k]
        # Average attention across heads, then apply to shared values
        avg_attn = attn_weights.mean(dim=1)  # [B, T_q, T_k]
        attn_out = torch.matmul(avg_attn, v)  # [B, T_q, d_k]

        out = self.out_proj(attn_out)  # [B, T_q, d_model]
        return out, avg_attn
```

- [ ] **Step 4: Run tests**

Run: `venv/bin/python -m pytest tests/test_tft_model.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add models/tft/model.py tests/test_tft_model.py
git commit -m "feat(tft): TFT model architecture with interpretable attention"
```

---

### Task 5: Training Loop and CV

**Files:**
- Create: `backend/models/tft/trainer.py`
- Create: `backend/tests/test_tft_trainer.py`

**Interfaces:**
- Consumes: `TemporalFusionTransformer` and `TFTConfig` from `models.tft.model`; `SequenceDataset` and `build_dataloaders` from `models.tft.dataset`
- Produces: `TFTTrainer(config, model_dir)` with methods `train_cv(df, cv_splits, horizons) → dict` (returns OOF predictions + metrics), `train_final(df, horizons) → None` (trains on all data, saves checkpoint), `load(model_dir) → TFTTrainer` (loads saved model), `predict(df) → np.ndarray` (produces predictions)

The trainer handles:
- Quantile loss (pinball at τ=0.5 = MAE/2)
- Early stopping on validation loss
- Expanding-window CV with the same fold structure as LightGBM
- OOF prediction collection for conformal calibration
- Model checkpointing (PyTorch state_dict + config JSON)

- [ ] **Step 1: Write the test file**

```python
# backend/tests/test_tft_trainer.py
import pytest
import numpy as np
import pandas as pd
import os

torch = pytest.importorskip("torch")


def _make_price_df(n_items=10, n_days=300, seed=42):
    rng = np.random.RandomState(seed)
    rows = []
    base_date = pd.Timestamp("2024-01-01")
    for i in range(n_items):
        price = 10.0 + rng.randn() * 3
        for d in range(n_days):
            price *= np.exp(rng.randn() * 0.015)
            rows.append({
                "item_id": f"item_{i}",
                "date": (base_date + pd.Timedelta(days=d)).date(),
                "price": round(max(price, 0.5), 2),
                "volume": max(1, int(rng.poisson(15))),
            })
    return pd.DataFrame(rows)


class TestQuantileLoss:
    def test_symmetric_at_half(self):
        from models.tft.trainer import quantile_loss
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([2.0, 2.0, 2.0])
        loss = quantile_loss(pred, target, quantile=0.5)
        # At q=0.5, quantile loss = 0.5 * MAE
        expected = 0.5 * torch.tensor([1.0, 0.0, 1.0]).mean()
        assert torch.allclose(loss, expected, atol=1e-5)


class TestTFTTrainer:
    def test_train_single_fold(self, tmp_path):
        from models.tft.trainer import TFTTrainer
        from models.tft.model import TFTConfig
        df = _make_price_df(n_items=5, n_days=200)
        config = TFTConfig(hidden_dim=8, num_heads=2, dropout=0.0)
        trainer = TFTTrainer(config, model_dir=str(tmp_path))
        dates = sorted(df["date"].unique())
        split = len(dates) * 2 // 3
        train_dates = dates[:split]
        val_dates = dates[split:]
        val_loss = trainer.train_fold(df, train_dates, val_dates,
                                      max_epochs=3, patience=2)
        assert isinstance(val_loss, float)
        assert val_loss > 0

    def test_predict_after_train(self, tmp_path):
        from models.tft.trainer import TFTTrainer
        from models.tft.model import TFTConfig
        df = _make_price_df(n_items=5, n_days=200)
        config = TFTConfig(hidden_dim=8, num_heads=2, dropout=0.0)
        trainer = TFTTrainer(config, model_dir=str(tmp_path))
        dates = sorted(df["date"].unique())
        split = len(dates) * 2 // 3
        trainer.train_fold(df, dates[:split], dates[split:],
                           max_epochs=2, patience=2)
        preds = trainer.predict(df, dates[split:])
        assert "item_id" in preds.columns
        assert "date" in preds.columns
        for h in [3, 7, 14, 30]:
            assert f"pred_{h}d" in preds.columns
        assert len(preds) > 0
        assert not preds[f"pred_3d"].isna().all()

    def test_save_and_load(self, tmp_path):
        from models.tft.trainer import TFTTrainer
        from models.tft.model import TFTConfig
        df = _make_price_df(n_items=3, n_days=150)
        config = TFTConfig(hidden_dim=8, num_heads=2, dropout=0.0)
        trainer = TFTTrainer(config, model_dir=str(tmp_path))
        dates = sorted(df["date"].unique())
        split = len(dates) * 2 // 3
        trainer.train_fold(df, dates[:split], dates[split:],
                           max_epochs=2, patience=2)
        trainer.save()
        loaded = TFTTrainer.load(str(tmp_path))
        preds_orig = trainer.predict(df, dates[split:])
        preds_loaded = loaded.predict(df, dates[split:])
        pd.testing.assert_frame_equal(preds_orig, preds_loaded)

    def test_oof_predictions(self, tmp_path):
        from models.tft.trainer import TFTTrainer
        from models.tft.model import TFTConfig
        df = _make_price_df(n_items=5, n_days=300)
        config = TFTConfig(hidden_dim=8, num_heads=2, dropout=0.0)
        trainer = TFTTrainer(config, model_dir=str(tmp_path))
        dates = sorted(df["date"].unique())
        # 2 folds: 0..150 train / 150..180 val, 0..180 train / 180..210 val
        folds = [
            (dates[:150], dates[150:180]),
            (dates[:180], dates[180:210]),
        ]
        oof = trainer.train_cv(df, folds, max_epochs=2, patience=2)
        assert "item_id" in oof.columns
        assert "date" in oof.columns
        assert "pred_3d" in oof.columns
        assert "actual_3d" in oof.columns
        assert len(oof) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_tft_trainer.py -q`
Expected: FAIL — `TFTTrainer` not defined

- [ ] **Step 3: Implement trainer.py**

```python
# backend/models/tft/trainer.py
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

from models.tft.dataset import (
    SequenceDataset, build_dataloaders,
    N_PAST_FEATURES, N_STATIC_FEATURES, N_FUTURE_FEATURES,
)
from models.tft.model import TemporalFusionTransformer, TFTConfig

logger = logging.getLogger(__name__)

HORIZONS = [3, 7, 14, 30]
CHECKPOINT_NAME = "tft_model.pt"
CONFIG_NAME = "tft_config.json"


def quantile_loss(
    pred: torch.Tensor, target: torch.Tensor, quantile: float = 0.5,
) -> torch.Tensor:
    error = target - pred
    return torch.mean(torch.max(quantile * error, (quantile - 1) * error))


class TFTTrainer:

    def __init__(
        self,
        config: TFTConfig | None = None,
        model_dir: str = "models/saved_models",
        horizons: list[int] | None = None,
        lookback: int = 60,
        batch_size: int = 256,
        lr: float = 1e-3,
    ):
        self.config = config or TFTConfig()
        self.model_dir = model_dir
        self.horizons = horizons or HORIZONS
        self.lookback = lookback
        self.batch_size = batch_size
        self.lr = lr
        self.model: TemporalFusionTransformer | None = None

    def _build_model(self) -> TemporalFusionTransformer:
        cfg = TFTConfig(
            n_past_features=N_PAST_FEATURES,
            n_static_features=N_STATIC_FEATURES,
            n_future_features=N_FUTURE_FEATURES,
            n_horizons=len(self.horizons),
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            lookback=self.lookback,
            lstm_layers=self.config.lstm_layers,
            n_price_tiers=self.config.n_price_tiers,
        )
        return TemporalFusionTransformer(cfg)

    def train_fold(
        self,
        df: pd.DataFrame,
        train_dates: list,
        val_dates: list,
        max_epochs: int = 50,
        patience: int = 5,
    ) -> float:
        """Train on one fold. Returns best validation loss."""
        self.model = self._build_model()
        train_dl, val_dl = build_dataloaders(
            df, train_dates, val_dates,
            lookback=self.lookback, horizons=self.horizons,
            batch_size=self.batch_size,
        )
        if len(train_dl) == 0:
            logger.warning("Empty training set — skipping fold")
            return float("inf")

        optimizer = Adam(self.model.parameters(), lr=self.lr)
        scheduler = OneCycleLR(
            optimizer, max_lr=self.lr,
            steps_per_epoch=len(train_dl), epochs=max_epochs,
        )

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(max_epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            n_batches = 0
            for past, static, future, targets, _ in train_dl:
                optimizer.zero_grad()
                preds = self.model(past, static, future)
                loss = quantile_loss(preds, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
                n_batches += 1
            train_loss /= max(n_batches, 1)

            # Validate
            val_loss = self._evaluate(val_dl)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 10 == 0 or no_improve == 0:
                logger.info(
                    f"  Epoch {epoch}: train_loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f} best={best_val_loss:.4f}"
                )

            if no_improve >= patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return best_val_loss

    @torch.no_grad()
    def _evaluate(self, dl) -> float:
        self.model.eval()
        total_loss = 0.0
        n = 0
        for past, static, future, targets, _ in dl:
            preds = self.model(past, static, future)
            loss = quantile_loss(preds, targets)
            total_loss += loss.item()
            n += 1
        return total_loss / max(n, 1)

    @torch.no_grad()
    def predict(self, df: pd.DataFrame, dates: list | None = None) -> pd.DataFrame:
        """Run inference. Returns DataFrame with item_id, date, pred_{h}d columns."""
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")
        self.model.eval()

        ds = SequenceDataset(
            df, lookback=self.lookback, horizons=self.horizons,
            date_filter=dates,
        )
        if len(ds) == 0:
            cols = ["item_id", "date"] + [f"pred_{h}d" for h in self.horizons]
            return pd.DataFrame(columns=cols)

        from torch.utils.data import DataLoader
        from models.tft.dataset import _collate_fn
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                        collate_fn=_collate_fn)

        all_preds = []
        all_items = []
        all_dates = []

        for past, static, future, _, meta in dl:
            preds = self.model(past, static, future)  # [B, H]
            all_preds.append(preds.numpy())
            all_items.extend(meta["item_id"])
            all_dates.extend(meta["date"])

        preds_arr = np.concatenate(all_preds, axis=0)
        result = pd.DataFrame({
            "item_id": all_items,
            "date": all_dates,
        })
        for i, h in enumerate(self.horizons):
            result[f"pred_{h}d"] = preds_arr[:, i]
        return result

    def train_cv(
        self,
        df: pd.DataFrame,
        folds: list[tuple[list, list]],
        max_epochs: int = 50,
        patience: int = 5,
    ) -> pd.DataFrame:
        """Expanding-window CV. Returns OOF predictions with actuals."""
        all_oof = []

        for fi, (train_dates, val_dates) in enumerate(folds):
            logger.info(f"TFT CV fold {fi + 1}/{len(folds)}: "
                        f"train={len(train_dates)}d val={len(val_dates)}d")
            self.train_fold(df, train_dates, val_dates,
                            max_epochs=max_epochs, patience=patience)
            fold_preds = self.predict(df, val_dates)

            # Attach actuals
            ds = SequenceDataset(df, lookback=self.lookback, horizons=self.horizons,
                                 date_filter=val_dates)
            actuals = {}
            for idx in range(len(ds)):
                _, _, _, targets, meta = ds[idx]
                key = (meta["item_id"], str(meta["date"]))
                actuals[key] = targets.numpy()

            for i, h in enumerate(self.horizons):
                fold_preds[f"actual_{h}d"] = fold_preds.apply(
                    lambda r: actuals.get(
                        (r["item_id"], str(r["date"])), np.full(len(self.horizons), np.nan)
                    )[i], axis=1,
                )
            fold_preds["fold"] = fi
            all_oof.append(fold_preds)

        return pd.concat(all_oof, ignore_index=True) if all_oof else pd.DataFrame()

    def save(self) -> None:
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, CHECKPOINT_NAME))
        config_dict = {
            "hidden_dim": self.config.hidden_dim,
            "num_heads": self.config.num_heads,
            "dropout": self.config.dropout,
            "lookback": self.lookback,
            "lstm_layers": self.config.lstm_layers,
            "n_price_tiers": self.config.n_price_tiers,
            "horizons": self.horizons,
            "batch_size": self.batch_size,
            "lr": self.lr,
        }
        with open(os.path.join(self.model_dir, CONFIG_NAME), "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, model_dir: str) -> TFTTrainer:
        with open(os.path.join(model_dir, CONFIG_NAME)) as f:
            cfg = json.load(f)
        config = TFTConfig(
            hidden_dim=cfg["hidden_dim"],
            num_heads=cfg["num_heads"],
            dropout=cfg["dropout"],
            lstm_layers=cfg.get("lstm_layers", 1),
            n_price_tiers=cfg.get("n_price_tiers", 10),
        )
        trainer = cls(
            config=config,
            model_dir=model_dir,
            horizons=cfg.get("horizons", HORIZONS),
            lookback=cfg.get("lookback", 60),
            batch_size=cfg.get("batch_size", 256),
            lr=cfg.get("lr", 1e-3),
        )
        trainer.model = trainer._build_model()
        state = torch.load(os.path.join(model_dir, CHECKPOINT_NAME),
                           weights_only=True)
        trainer.model.load_state_dict(state)
        return trainer
```

- [ ] **Step 4: Run tests**

Run: `venv/bin/python -m pytest tests/test_tft_trainer.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add models/tft/trainer.py tests/test_tft_trainer.py
git commit -m "feat(tft): training loop with CV, early stopping, save/load"
```

---

### Task 6: Integration into Forecaster

**Files:**
- Modify: `backend/models/forecaster.py` (add `_train_tft` and `_predict_tft` methods)
- Modify: `backend/models/tft/__init__.py` (export public API)
- Create: `backend/tests/test_tft_integration.py`

**Interfaces:**
- Consumes: `TFTTrainer` from `models.tft.trainer`, `TFTConfig` from `models.tft.model`; existing `ItemForecaster.fetch_price_history()`, `_compute_cv_splits()`, `embargo_days()`, `conformal.calibrate_signed()`
- Produces: `TFT_CENTRE=1` env flag gates TFT training/prediction; TFT OOF residuals feed into conformal band calibration; predictions stored in same format as LightGBM

Integration is **thin and isolated**. The TFT runs as a parallel centre model:
- `train()` calls `_train_tft()` after the LightGBM loop when `TFT_CENTRE=1`
- `predict()` uses TFT predictions instead of LightGBM when `TFT_CENTRE=1` and model exists
- OOF residuals are passed to `conformal.calibrate_signed()` the same way LightGBM's are

- [ ] **Step 1: Write the integration test**

```python
# backend/tests/test_tft_integration.py
import pytest
import os
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

torch = pytest.importorskip("torch")


def _make_price_df(n_items=5, n_days=250, seed=42):
    rng = np.random.RandomState(seed)
    rows = []
    base_date = pd.Timestamp("2024-06-01")
    for i in range(n_items):
        price = 8.0 + rng.randn() * 2
        for d in range(n_days):
            price *= np.exp(rng.randn() * 0.015)
            rows.append({
                "item_id": f"item_{i}",
                "date": (base_date + pd.Timedelta(days=d)).date(),
                "price": round(max(price, 1.0), 2),
                "volume": max(1, int(rng.poisson(15))),
            })
    return pd.DataFrame(rows)


class TestTFTTrainAndPredict:
    """End-to-end: build dataset from voted-style df, train TFT, get predictions."""

    def test_cv_produces_oof_for_conformal(self, tmp_path):
        from models.tft.trainer import TFTTrainer
        from models.tft.model import TFTConfig

        df = _make_price_df(n_items=5, n_days=250)
        config = TFTConfig(hidden_dim=8, num_heads=2, dropout=0.0)
        trainer = TFTTrainer(config, model_dir=str(tmp_path))

        dates = sorted(df["date"].unique())
        folds = [
            (dates[:120], dates[120:150]),
            (dates[:150], dates[150:180]),
        ]
        oof = trainer.train_cv(df, folds, max_epochs=3, patience=2)

        # OOF must have the columns conformal needs
        assert len(oof) > 0
        for h in [3, 7, 14, 30]:
            residuals = oof[f"actual_{h}d"] - oof[f"pred_{h}d"]
            assert not residuals.isna().all(), f"All NaN residuals at h={h}"

    def test_predict_returns_per_item_latest(self, tmp_path):
        from models.tft.trainer import TFTTrainer
        from models.tft.model import TFTConfig

        df = _make_price_df(n_items=5, n_days=200)
        config = TFTConfig(hidden_dim=8, num_heads=2, dropout=0.0)
        trainer = TFTTrainer(config, model_dir=str(tmp_path))

        dates = sorted(df["date"].unique())
        trainer.train_fold(df, dates[:130], dates[130:160],
                           max_epochs=2, patience=2)
        preds = trainer.predict(df, dates[130:160])

        # Should have predictions for each item
        assert preds["item_id"].nunique() > 0
        assert all(f"pred_{h}d" in preds.columns for h in [3, 7, 14, 30])
```

- [ ] **Step 2: Run test to verify it passes (uses already-built trainer)**

Run: `venv/bin/python -m pytest tests/test_tft_integration.py -q`
Expected: all PASS

- [ ] **Step 3: Update tft/__init__.py with public API**

```python
# backend/models/tft/__init__.py
from __future__ import annotations

from models.tft.model import TFTConfig, TemporalFusionTransformer
from models.tft.trainer import TFTTrainer
from models.tft.dataset import SequenceDataset, build_dataloaders

__all__ = [
    "TFTConfig",
    "TemporalFusionTransformer",
    "TFTTrainer",
    "SequenceDataset",
    "build_dataloaders",
]
```

- [ ] **Step 4: Add the TFT flag and thin integration methods to forecaster.py**

Near the top of `forecaster.py`, alongside the other env flags, add:

```python
TFT_CENTRE = os.environ.get("TFT_CENTRE") == "1"
```

Add `_train_tft` method to `ItemForecaster` (after the `train()` method):

```python
def _train_tft(self, price_df: pd.DataFrame) -> None:
    """Train TFT centre model on raw price series. Gated by TFT_CENTRE=1."""
    try:
        from models.tft import TFTTrainer, TFTConfig
    except ImportError:
        logger.warning("PyTorch not installed — skipping TFT training")
        return

    logger.info("=" * 60)
    logger.info("TRAINING TFT CENTRE MODEL")
    logger.info("=" * 60)

    tft_dir = os.path.join(self.model_dir, "tft")
    config = TFTConfig(hidden_dim=32, num_heads=4, dropout=0.1)
    trainer = TFTTrainer(config, model_dir=tft_dir)

    sorted_dates = sorted(price_df["date"].unique())
    folds = self._compute_cv_splits(sorted_dates, purge_days=0)
    if not folds:
        logger.warning("  No CV folds — skipping TFT")
        return

    oof = trainer.train_cv(price_df, folds, max_epochs=50, patience=5)
    logger.info(f"  TFT OOF: {len(oof)} rows across {oof['fold'].nunique()} folds")

    # Log OOF rank IC per horizon
    for h in self.HORIZONS:
        pred_col = f"pred_{h}d"
        actual_col = f"actual_{h}d"
        if pred_col in oof.columns and actual_col in oof.columns:
            valid = oof[[pred_col, actual_col]].dropna()
            if len(valid) > 10:
                from scipy.stats import spearmanr
                ic, _ = spearmanr(valid[pred_col], valid[actual_col])
                logger.info(f"  TFT {h}d OOF rank IC: {ic:.4f}")

    # Train final model on all data
    trainer.train_fold(price_df, sorted_dates[:-30], sorted_dates[-30:],
                       max_epochs=50, patience=5)
    trainer.save()
    logger.info(f"  TFT model saved to {tft_dir}")
```

Add `_predict_tft` method:

```python
def _predict_tft(self, price_df: pd.DataFrame) -> dict[int, pd.Series] | None:
    """Load trained TFT and produce per-horizon return predictions."""
    tft_dir = os.path.join(self.model_dir, "tft")
    if not os.path.exists(os.path.join(tft_dir, "tft_model.pt")):
        return None
    try:
        from models.tft import TFTTrainer
    except ImportError:
        return None

    trainer = TFTTrainer.load(tft_dir)
    dates = sorted(price_df["date"].unique())
    latest_dates = dates[-1:]  # predict for latest date only

    preds = trainer.predict(price_df, latest_dates)
    if preds.empty:
        return None

    result = {}
    for h in self.HORIZONS:
        col = f"pred_{h}d"
        if col in preds.columns:
            series = preds.set_index("item_id")[col]
            result[h] = series
    return result if result else None
```

Wire `_train_tft` at the end of `train()` (before `save_models()`):

```python
if TFT_CENTRE:
    self._train_tft(price_df_for_tft)
```

Note: `price_df_for_tft` is the voted price DataFrame from `fetch_price_history()`. This is available early in `train()` — store a reference before it gets consumed by feature engineering.

- [ ] **Step 5: Run existing tests to verify no regression**

Run: `venv/bin/python -m pytest tests/test_forecaster.py -q --timeout=60`
Expected: all existing tests still PASS (TFT code is behind `TFT_CENTRE=1`, which is off)

- [ ] **Step 6: Run integration tests**

Run: `venv/bin/python -m pytest tests/test_tft_integration.py -q`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add models/tft/__init__.py models/forecaster.py tests/test_tft_integration.py
git commit -m "feat(tft): integrate TFT into forecaster behind TFT_CENTRE flag"
```

---

### Task 7: Evaluation Script

**Files:**
- Create: `backend/scripts/eval_tft.py`

**Interfaces:**
- Consumes: `TFTTrainer.load()`, `ItemForecaster.fetch_price_history()`, existing CV infrastructure
- Produces: console output comparing TFT vs LightGBM on rank IC, MAE, and residual distributions per horizon; CSV of per-fold metrics to `data/tft_eval.csv`

This is the measurement script you run locally after training both models. It uses the same OOF fold structure and scores both models on identical validation dates for a fair comparison.

- [ ] **Step 1: Write the evaluation script**

```python
#!/usr/bin/env python
"""Compare TFT vs LightGBM centre predictions on OOF folds.

Usage (from backend/):
    TFT_CENTRE=1 venv/bin/python scripts/eval_tft.py

Reads the voted price history, runs expanding-window CV for TFT,
and compares against LightGBM OOF predictions (from meta.json cv_results).
"""
from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tft import TFTTrainer, TFTConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

HORIZONS = [3, 7, 14, 30]


def main():
    from models.forecaster import ItemForecaster
    from unittest.mock import MagicMock

    model_dir = os.path.join(os.path.dirname(__file__), "..", "models", "saved_models")
    forecaster = ItemForecaster(db_session=MagicMock(), model_dir=model_dir)

    logger.info("Loading price history...")
    price_df = forecaster.fetch_price_history(days_back=1460, backfilled_only=True)
    price_df = price_df[price_df["item_id"].isin(
        price_df.groupby("item_id")["price"].transform("median")
        .loc[lambda s: s >= 1.0].index
    )]
    logger.info(f"  {price_df['item_id'].nunique()} items, {len(price_df):,} rows")

    sorted_dates = sorted(price_df["date"].unique())
    folds = forecaster._compute_cv_splits(sorted_dates, purge_days=0)
    logger.info(f"  {len(folds)} CV folds")

    # Train TFT
    config = TFTConfig(hidden_dim=32, num_heads=4, dropout=0.1)
    trainer = TFTTrainer(config, model_dir=os.path.join(model_dir, "tft_eval"))
    oof = trainer.train_cv(price_df, folds, max_epochs=50, patience=5)

    # Score
    logger.info("\n" + "=" * 70)
    logger.info("TFT OOF RESULTS")
    logger.info("=" * 70)

    rows = []
    for h in HORIZONS:
        pred_col = f"pred_{h}d"
        actual_col = f"actual_{h}d"
        valid = oof[[pred_col, actual_col, "fold"]].dropna()
        if valid.empty:
            continue

        residuals = valid[actual_col] - valid[pred_col]
        mae = residuals.abs().mean()
        ic, _ = spearmanr(valid[pred_col], valid[actual_col])

        # Naive baseline: predict 0 (no change)
        naive_mae = valid[actual_col].abs().mean()

        logger.info(
            f"  {h:2d}d: rank_ic={ic:+.4f}  MAE={mae:.2f}%  "
            f"naive_MAE={naive_mae:.2f}%  edge={naive_mae - mae:+.2f}%  "
            f"n={len(valid):,}"
        )
        rows.append({
            "horizon": h, "rank_ic": ic, "mae": mae,
            "naive_mae": naive_mae, "n": len(valid),
        })

    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(model_dir, "tft_eval", "tft_eval.csv"), index=False)
        logger.info(f"\nResults saved to {model_dir}/tft_eval/tft_eval.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test the script runs without error on synthetic data**

Run: `cd /Users/rayanrane/personal-projects/cs2-analyzer/cs2-oracle/backend && venv/bin/python -c "from scripts.eval_tft import main; print('imports ok')"`
Expected: `imports ok`

- [ ] **Step 3: Commit**

```bash
git add scripts/eval_tft.py
git commit -m "feat(tft): evaluation script comparing TFT vs LightGBM OOF"
```

---

## Execution Checklist

After all tasks, verify:

1. `venv/bin/python -m pytest tests/test_tft_dataset.py tests/test_tft_components.py tests/test_tft_model.py tests/test_tft_trainer.py tests/test_tft_integration.py -q` — all pass
2. `venv/bin/python -m pytest tests/test_forecaster.py -q` — no regression (TFT flag is off)
3. `TFT_CENTRE=1 venv/bin/python scripts/eval_tft.py` — runs end-to-end on real data, produces metrics

## What This Does NOT Include (Intentional Scope Cuts)

- **Optuna HP search for TFT** — start with fixed hyperparameters (hidden=32, heads=4, lr=1e-3); tune manually once baseline results are in
- **GPU training** — model is small enough for CPU; add CUDA later if training time is a bottleneck
- **CI workflow changes** — TFT is gated off; no CI impact until you ship it
- **Serving path changes** — TFT predictions use the same conformal infrastructure; no API changes needed
- **Attention weight visualization** — the `InterpretableMultiHeadAttention` returns weights, but visualization is a separate project
