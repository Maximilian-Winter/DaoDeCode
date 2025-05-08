import torch
import torch.nn as nn

from attention import MechanismAttention
from five_transformations_layer import FiveElementsTransformationLayer
from position_timing_layer import PositionTimingLayer


class MechanismTransformerBlock(nn.Module):
    """Center of the Dao: Harmonizes Position-Timing, Mechanism Attention, Five Elements, and Transformation."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, sharpening_factor=2.0):
        super().__init__()

        # Layers
        self.pos_time = PositionTimingLayer(d_model)
        self.mechanism_attn = MechanismAttention(d_model, n_heads, sharpening_factor)
        self.five_elements = FiveElementsTransformationLayer(d_model)

        # Feedforward fire (simple expansion -> activation -> compression)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 1. Integrate spacetime consciousness
        x = self.pos_time(x)

        # 2. Mechanism-Aware Attention: sharpen focus on key points
        attn_input = self.norm1(x)
        attn_output, mechanism_strengths = self.mechanism_attn(attn_input, mask)
        x = x + self.dropout(attn_output)  # Residual connection

        # 3. Transform through the Five Elements
        elem_input = self.norm2(x)
        five_elem_output, element_weights = self.five_elements(elem_input)
        x = x + self.dropout(five_elem_output)  # Residual

        # 4. Ignite the Feedforward Fire
        ff_input = self.norm3(x)
        ff_output = self.ff(ff_input)
        x = x + ff_output  # Residual

        # Return x and diagnostics
        return x, mechanism_strengths, element_weights
