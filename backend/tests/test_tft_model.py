import pytest

torch = pytest.importorskip("torch")

from models.tft.dataset import N_FUTURE_FEATURES, N_PAST_FEATURES, N_STATIC_FEATURES


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
