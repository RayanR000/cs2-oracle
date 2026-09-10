from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.tft.components import GatedResidualNetwork, GatedLinearUnit


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

        # --- Past observed features ---
        # NOTE: the plan drafted a per-variable VSN here (past_proj/past_vsn)
        # but forward uses the projected path below; the VSN modules were
        # never called, so they received no gradient and failed
        # test_gradient_flows. Dropped in favour of the single projection.
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
