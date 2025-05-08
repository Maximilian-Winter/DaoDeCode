import torch
import torch.nn as nn


class FiveElementsTransformationLayer(nn.Module):
    """Dynamic Five Elements Transformation per token"""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # Element transformations
        self.wood_proj = nn.Linear(d_model, d_model)  # Expansion
        self.fire_proj = nn.Linear(d_model, d_model)  # Acceleration
        self.earth_proj = nn.Linear(d_model, d_model)  # Stabilization
        self.metal_proj = nn.Linear(d_model, d_model)  # Refinement
        self.water_proj = nn.Linear(d_model, d_model)  # Adaptation

        # Per-token element predictor
        self.element_classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 5)  # 5 elements
        )

        # Initialize with basic flavor
        self._initialize_transformations()

    def _initialize_transformations(self):
        """Initialize projections with slight element-specific biases"""
        nn.init.eye_(self.wood_proj.weight)
        self.wood_proj.weight.data *= 1.2  # Expand

        nn.init.eye_(self.fire_proj.weight)
        self.fire_proj.weight.data += torch.randn_like(self.fire_proj.weight) * 0.1  # Randomness

        nn.init.eye_(self.earth_proj.weight)
        self.earth_proj.weight.data *= 0.8  # Stabilize

        nn.init.eye_(self.metal_proj.weight)
        self.metal_proj.weight.data *= (torch.rand_like(self.metal_proj.weight) > 0.7).float()  # Sparsify

        nn.init.eye_(self.water_proj.weight)
        for i in range(self.water_proj.weight.size(0)):
            for j in range(max(0, i - 1), min(self.water_proj.weight.size(1), i + 2)):
                self.water_proj.weight.data[i, j] = 1.0 / (abs(i - j) + 1)  # Smoothing

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Apply each elemental transformation
        wood_x = self.wood_proj(x)
        fire_x = self.fire_proj(x)
        earth_x = self.earth_proj(x)
        metal_x = self.metal_proj(x)
        water_x = self.water_proj(x)

        # Predict element mixture per token
        element_logits = self.element_classifier(x)  # [batch_size, seq_len, 5]
        element_weights = torch.softmax(element_logits, dim=-1)  # Normalize across elements

        # Mix the transformations
        output = (
                element_weights[..., 0:1] * wood_x +
                element_weights[..., 1:2] * fire_x +
                element_weights[..., 2:3] * earth_x +
                element_weights[..., 3:4] * metal_x +
                element_weights[..., 4:5] * water_x
        )

        return output, element_weights
