import torch
import torch.nn as nn

from mechanism_tranformer_layer import MechanismTransformerBlock


class MechanismTransformer(nn.Module):
    """Full Mechanism Transformer: Dao of Mechanisms, Transformations, and Time"""

    def __init__(self, d_input, d_model=512, n_heads=8, n_layers=6, d_ff=2048, dropout=0.1, max_seq_len=1024,
                 sharpening_factor=2.0):
        super().__init__()

        self.d_input = d_input
        self.d_model = d_model
        self.n_layers = n_layers

        # Embedding layer to lift raw input into d_model space
        self.input_proj = nn.Linear(d_input, d_model)

        # Stack of MechanismTransformerBlocks
        self.blocks = nn.ModuleList([
            MechanismTransformerBlock(d_model, n_heads, d_ff, dropout, sharpening_factor)
            for _ in range(n_layers)
        ])

        # Final normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_input]
            mask: [batch_size, seq_len] (optional)
        Returns:
            output: [batch_size, seq_len, d_model]
            all_mechanism_strengths: list of mechanism strengths from each block
            all_element_weights: list of element weights from each block
        """
        # Input projection
        x = self.input_proj(x)

        # Store diagnostics
        all_mechanism_strengths = []
        all_element_weights = []

        # Pass through blocks
        for block in self.blocks:
            x, mech_strengths, elem_weights = block(x, mask)
            all_mechanism_strengths.append(mech_strengths)
            all_element_weights.append(elem_weights)

        # Final normalization
        x = self.norm(x)

        return x, all_mechanism_strengths, all_element_weights
