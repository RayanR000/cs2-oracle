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
