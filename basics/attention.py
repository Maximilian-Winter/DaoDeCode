import torch
import torch.nn as nn
import math


class MechanismAttention(nn.Module):
    """Sharpened Mechanism-Aware Multi-Head Attention"""

    def __init__(self, d_model, n_heads, sharpening_factor=2.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.sharpening_factor = sharpening_factor  # How strongly to bias toward mechanism points

        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Mechanism point predictor
        self.mechanism_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # Project queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, L, D]
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Standard scaled dot-product attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)  # [B, H, L, L]

        # Optional mask (padding, causal, etc.)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

        # Calculate mechanism strengths
        mechanism_logits = self.mechanism_proj(x).squeeze(-1)  # [B, L]
        mechanism_strengths = torch.softmax(mechanism_logits, dim=-1)  # Sharpened across sequence

        # Mechanism strengths influence attention
        # We add sharpening by adding mechanism bias to attention scores
        mechanism_bias = self.sharpening_factor * mechanism_strengths.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
        scores = scores + mechanism_bias  # Additive bias favors high mechanism strength tokens

        # Now compute standard attention
        attn_weights = torch.softmax(scores, dim=-1)  # [B, H, L, L]
        context = torch.matmul(attn_weights, v)  # [B, H, L, D]

        # Reshape and output
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(context)

        return output, mechanism_strengths
