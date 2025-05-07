import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class MechanismPointAttention(nn.Module):
    """
    Attention module that identifies and leverages mechanism points in the attention process.
    Implements the concept of position-timing unity and the Five Elements transformation.
    """

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            dropout_prob: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Position-Timing Unity projections
        self.positional_proj = nn.Linear(hidden_size, hidden_size)
        self.temporal_proj = nn.Linear(hidden_size, hidden_size)
        self.unity_gate = nn.Linear(hidden_size * 2, hidden_size)

        # Five Elements transformation matrices
        self.wood_expansion = nn.Linear(hidden_size, hidden_size)  # Expansion
        self.fire_acceleration = nn.Linear(hidden_size, hidden_size)  # Acceleration
        self.earth_stabilization = nn.Linear(hidden_size, hidden_size)  # Stabilization
        self.metal_refinement = nn.Linear(hidden_size, hidden_size)  # Refinement
        self.water_adaptation = nn.Linear(hidden_size, hidden_size)  # Adaptation

        # Element mixing weights (learnable)
        self.element_weights = nn.Parameter(torch.ones(5) / 5)

        # For the query, key, value projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout_prob)

        # Mechanism strength detector
        self.mechanism_detector = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim // 2),
            nn.GELU(),
            nn.Linear(self.head_dim // 2, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize transformation matrices according to their characteristics
        # Wood: expansion (eigenvalues > 1)
        nn.init.eye_(self.wood_expansion.weight)
        self.wood_expansion.weight.data *= 1.2

        # Fire: acceleration (amplifies differences)
        nn.init.eye_(self.fire_acceleration.weight)
        self.fire_acceleration.weight.data += torch.randn_like(self.fire_acceleration.weight.data) * 0.1

        # Earth: stabilization (eigenvalues < 1)
        nn.init.eye_(self.earth_stabilization.weight)
        self.earth_stabilization.weight.data *= 0.8

        # Metal: refinement (sparse precision)
        nn.init.eye_(self.metal_refinement.weight)
        mask = torch.rand_like(self.metal_refinement.weight.data) > 0.8
        self.metal_refinement.weight.data *= mask.float()

        # Water: adaptation (smoothing)
        nn.init.eye_(self.water_adaptation.weight)
        for i in range(self.water_adaptation.weight.size(0)):
            for j in range(max(0, i - 1), min(self.water_adaptation.weight.size(1), i + 2)):
                self.water_adaptation.weight.data[i, j] = 1.0 / (abs(i - j) + 1)

    def apply_five_elements(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Five Elements transformations with learned mixing weights"""
        # Apply each transformation
        wood_x = self.wood_expansion(x)  # Expansion
        fire_x = self.fire_acceleration(x)  # Acceleration
        earth_x = self.earth_stabilization(x)  # Stabilization
        metal_x = self.metal_refinement(x)  # Refinement
        water_x = self.water_adaptation(x)  # Adaptation

        # Get normalized element weights
        weights = F.softmax(self.element_weights, dim=0)

        # Combine transformations
        return (weights[0] * wood_x +
                weights[1] * fire_x +
                weights[2] * earth_x +
                weights[3] * metal_x +
                weights[4] * water_x)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_embeddings: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.Tensor]] = None,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_length = hidden_states.shape[:2]

        # Position-Timing Unity calculation
        if position_embeddings is not None:
            position_features = self.positional_proj(position_embeddings)
            temporal_features = self.temporal_proj(hidden_states)

            # Integrate position and timing aspects
            unity_gate = torch.sigmoid(self.unity_gate(
                torch.cat([position_features, temporal_features], dim=-1)
            ))

            # Apply position-timing unity gating
            hidden_states = hidden_states * unity_gate

        # Apply Five Elements transformations
        hidden_states = self.apply_five_elements(hidden_states)

        # Regular attention calculations with QKV projections
        query_states = self.q_proj(hidden_states).view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Handle past key values if provided (for efficient generation)
        if past_key_values is not None:
            key_states, value_states = past_key_values
            present = (key_states, value_states)
        else:
            key_states = self.k_proj(hidden_states).view(
                batch_size, seq_length, self.num_heads, self.head_dim
            ).transpose(1, 2)

            value_states = self.v_proj(hidden_states).view(
                batch_size, seq_length, self.num_heads, self.head_dim
            ).transpose(1, 2)

            if use_cache:
                present = (key_states, value_states)
            else:
                present = None

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply softmax for probabilities
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Detect mechanism points
        mechanism_strengths = self.mechanism_detector(key_states)

        # Apply mechanism strength weighting to attention
        weighted_values = value_states * mechanism_strengths

        # Compute attention output
        attn_output = torch.matmul(attn_weights, weighted_values)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_length, self.hidden_size
        )

        # Final projection
        attn_output = self.o_proj(attn_output)

        return attn_output, present, mechanism_strengths


class DaoTransformerBlock(nn.Module):
    """
    A Transformer block incorporating Dao principles of mechanism points,
    five elements transformation, and position-timing unity.
    """

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            intermediate_size: int,
            dropout_prob: float = 0.1
    ):
        super().__init__()
        self.attention = MechanismPointAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_prob=dropout_prob
        )
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        # Feedforward with five elements inspired design
        self.ff_wood = nn.Linear(hidden_size, intermediate_size)  # Expansion
        self.ff_fire = nn.Linear(hidden_size, intermediate_size)  # Acceleration
        self.ff_earth = nn.Linear(hidden_size, intermediate_size)  # Stabilization
        self.ff_metal = nn.Linear(hidden_size, intermediate_size)  # Refinement
        self.ff_water = nn.Linear(hidden_size, intermediate_size)  # Adaptation

        self.element_gate = nn.Linear(hidden_size, 5)
        self.ff_out = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_embeddings: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.Tensor]] = None,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], Optional[torch.Tensor]]:
        # Layer normalization before attention
        norm_x = self.ln1(hidden_states)

        # Apply mechanism point attention
        attn_output, present, mechanism_strengths = self.attention(
            norm_x,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            use_cache=use_cache
        )

        # Residual connection
        hidden_states = hidden_states + attn_output

        # Layer normalization before feedforward
        norm_x = self.ln2(hidden_states)

        # Calculate element gates - which transformation is most appropriate for each token
        element_gates = F.softmax(self.element_gate(norm_x), dim=-1)

        # Apply five elements to feedforward operations
        ff_wood = self.ff_wood(norm_x) * F.gelu(element_gates[..., 0:1])
        ff_fire = self.ff_fire(norm_x) * F.gelu(element_gates[..., 1:2])
        ff_earth = self.ff_earth(norm_x) * F.gelu(element_gates[..., 2:3])
        ff_metal = self.ff_metal(norm_x) * F.gelu(element_gates[..., 3:4])
        ff_water = self.ff_water(norm_x) * F.gelu(element_gates[..., 4:5])

        # Combine feedforward outputs from each element
        ff_output = ff_wood + ff_fire + ff_earth + ff_metal + ff_water

        # Final projection and residual connection
        ff_output = self.dropout(self.ff_out(ff_output))
        hidden_states = hidden_states + ff_output

        outputs = (hidden_states, present)
        if use_cache:
            outputs += (mechanism_strengths,)

        return outputs


class MechanismOptimizer(torch.optim.Optimizer):
    """
    An optimizer that prioritizes updating parameters based on their mechanism strength -
    their ability to create large downstream effects with small changes.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # Initialize mechanism strengths
        self.mechanism_strengths = {}
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    state['mechanism_strength'] = torch.ones_like(p)

    def calculate_mechanism_strengths(self, loss_fn, inputs, targets):
        """Calculate mechanism strength for each parameter"""
        # Store original parameter values
        original_params = {}
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    original_params[id(p)] = p.data.clone()

        # Calculate baseline loss
        baseline_loss = loss_fn(inputs, targets)
        baseline_loss_value = baseline_loss.item()

        # Calculate sensitivity for each parameter
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]

                    # Apply a small perturbation
                    perturbation = torch.randn_like(p) * 1e-4
                    p.data.add_(perturbation)

                    # Calculate new loss
                    perturbed_loss = loss_fn(inputs, targets)
                    perturbed_loss_value = perturbed_loss.item()

                    # Calculate sensitivity
                    sensitivity = abs(perturbed_loss_value - baseline_loss_value) / (torch.norm(perturbation) + 1e-10)

                    # Update mechanism strength using exponential moving average
                    state['mechanism_strength'] = 0.9 * state['mechanism_strength'] + 0.1 * sensitivity

                    # Restore original parameter values
                    p.data.copy_(original_params[id(p)])

    def step(self, closure=None):
        """Perform a single optimization step with mechanism awareness"""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Get mechanism strength for this parameter
                state = self.state[p]
                if 'mechanism_strength' not in state:
                    state['mechanism_strength'] = torch.ones_like(p)
                mechanism_strength = state['mechanism_strength']

                # Scale learning rate by mechanism strength
                # Higher strength → higher learning rate (more influential parameter)
                lr = group['lr'] * mechanism_strength / (torch.mean(mechanism_strength) + 1e-10)

                # Standard Adam-like update but with mechanism-scaled learning rate
                grad = p.grad.data
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Initialize momentum/variance accumulators
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['step'] = 0

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Update biased first/second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected moments
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Apply update
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = lr / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss