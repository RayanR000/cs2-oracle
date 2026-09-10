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
