import torch
import torch.nn as nn
import math


class PositionTimingLayer(nn.Module):
    """Dynamic Position-Timing Unity Layer (位時合一)"""

    def __init__(self, d_model, max_seq_len=1024):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Learnable position and timing embeddings
        self.position_embedding = nn.Parameter(self._init_sinusoidal(max_seq_len, d_model // 2))
        self.timing_embedding = nn.Parameter(self._init_sinusoidal(max_seq_len, d_model // 2))

        # Dynamic fusion predictor
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),  # Position weight, Timing weight
        )

        # Final integration
        self.integration_proj = nn.Linear(d_model, d_model)

    def _init_sinusoidal(self, length, dim):
        """Create sinusoidal embeddings (fixed initial pattern)"""
        pe = torch.zeros(length, dim)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x, mechanism_strengths=None):
        batch_size, seq_len, _ = x.size()

        # Get positional and timing embeddings
        pos_emb = self.position_embedding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        time_emb = self.timing_embedding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)

        # Predict fusion weights (position vs timing) per token
        fusion_logits = self.fusion_proj(x)  # [batch_size, seq_len, 2]
        fusion_weights = torch.softmax(fusion_logits, dim=-1)

        # Blend position and timing embeddings
        fused_emb = (
                fusion_weights[..., 0:1] * pos_emb +
                fusion_weights[..., 1:2] * time_emb
        )

        # 🛠️ Fix: Expand fused_emb from [batch, seq, d_model//2] to [batch, seq, d_model]
        fused_emb = torch.cat([fused_emb, fused_emb], dim=-1)  # Duplicate it

        # If mechanism strengths are provided, sharpen fusion
        if mechanism_strengths is not None:
            mech_importance = mechanism_strengths.unsqueeze(-1)
            fused_emb = fused_emb * (1 + mech_importance)

        # Integrate
        out = x + self.integration_proj(fused_emb)

        return out

