import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List


class FiveElementsAttention(nn.Module):
    """
    A modified attention mechanism based on the Five Elements transformation patterns.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Regular attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Five Elements transformation projections
        self.wood_proj = nn.Linear(embed_dim, embed_dim)  # Expansion
        self.fire_proj = nn.Linear(embed_dim, embed_dim)  # Acceleration
        self.earth_proj = nn.Linear(embed_dim, embed_dim)  # Stabilization
        self.metal_proj = nn.Linear(embed_dim, embed_dim)  # Refinement
        self.water_proj = nn.Linear(embed_dim, embed_dim)  # Adaptation

        # Element mixing weights (learnable)
        self.element_weights = nn.Parameter(torch.ones(5) / 5)

        # Initialize transformations with appropriate characteristics
        self._initialize_transformations()

    def _initialize_transformations(self):
        # Wood: Expansion through broadening attention span
        nn.init.eye_(self.wood_proj.weight)
        nn.init.ones_(self.wood_proj.bias)

        # Fire: Acceleration through sharpening attention
        nn.init.eye_(self.fire_proj.weight)
        self.fire_proj.weight.data += torch.randn_like(self.fire_proj.weight.data) * 0.1

        # Earth: Stabilization through smoothing
        nn.init.kaiming_uniform_(self.earth_proj.weight)

        # Metal: Refinement through precision focus
        nn.init.eye_(self.metal_proj.weight)
        mask = torch.rand_like(self.metal_proj.weight.data) > 0.7
        self.metal_proj.weight.data *= mask.float()

        # Water: Adaptation through contextual flow
        nn.init.xavier_uniform_(self.water_proj.weight)

    def _reshape_for_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape for multi-head attention"""
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _mechanism_strength(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Calculate mechanism strength for each position based on attention influence.
        Returns a tensor of shape [batch_size, seq_len] with mechanism strength scores.
        """
        # Sum attention across heads for each position
        mech_strength = attention_weights.sum(dim=1).mean(dim=1)

        # Normalize to [0, 1]
        mech_strength = (mech_strength - mech_strength.min(dim=-1, keepdim=True)[0]) / \
                        (mech_strength.max(dim=-1, keepdim=True)[0] - mech_strength.min(dim=-1, keepdim=True)[0] + 1e-9)

        return mech_strength

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        batch_size, seq_len, _ = hidden_states.size()

        # Regular attention computations
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for attention heads
        q = self._reshape_for_heads(q)
        k = self._reshape_for_heads(k)
        v = self._reshape_for_heads(v)

        # Scaled dot-product attention
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_weights = F.softmax(attention_scores, dim=-1)

        # Standard attention output
        attention_output = torch.matmul(attention_weights, v)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )

        # Calculate mechanism strength for each position
        mechanism_strength = self._mechanism_strength(attention_weights)

        # Apply Five Elements transformations based on mechanism strength
        wood_output = self.wood_proj(attention_output)  # Expansion
        fire_output = self.fire_proj(attention_output)  # Acceleration
        earth_output = self.earth_proj(attention_output)  # Stabilization
        metal_output = self.metal_proj(attention_output)  # Refinement
        water_output = self.water_proj(attention_output)  # Adaptation

        # Element weights can be context-dependent or learned
        element_weights = F.softmax(self.element_weights, dim=0)

        # Combine transformations with learned weights
        transformed_output = (
                element_weights[0] * wood_output +
                element_weights[1] * fire_output +
                element_weights[2] * earth_output +
                element_weights[3] * metal_output +
                element_weights[4] * water_output
        )

        # Final projection
        output = self.out_proj(transformed_output)

        # Return additional information about the transformation
        element_composition = {
            "wood": element_weights[0].item(),
            "fire": element_weights[1].item(),
            "earth": element_weights[2].item(),
            "metal": element_weights[3].item(),
            "water": element_weights[4].item(),
        }

        info = {
            "mechanism_strength": mechanism_strength,
            "element_composition": element_composition,
            "attention_weights": attention_weights
        }

        return output, info


class MechanismPointsLLM(nn.Module):
    """
    Language model architecture integrating the Mechanism Points and Transformation Patterns framework.
    """

    def __init__(
            self,
            vocab_size: int = 50257,
            hidden_size: int = 768,
            num_hidden_layers: int = 12,
            num_attention_heads: int = 12,
            max_position_embeddings: int = 2048,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings

        # Embeddings
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.layernorm_embedding = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

        # Layers
        self.layers = nn.ModuleList([
            TransformationLayer(hidden_size, num_attention_heads)
            for _ in range(num_hidden_layers)
        ])

        # Output head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Position-timing integration
        self.position_timing_layer = PositionTimingLayer(hidden_size)

        # Mechanism tracking
        self.register_buffer("mechanism_points", torch.zeros(max_position_embeddings))
        self.register_buffer("element_history", torch.zeros(5, 100))  # Track last 100 steps
        self.history_pointer = 0

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            return_mechanism_info: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_length = input_ids.size()
        device = input_ids.device

        # Create position ids
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Get embeddings
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)

        # Sum embeddings and apply layer norm
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Apply position-timing integration
        hidden_states = self.position_timing_layer(hidden_states, position_ids)

        # Track mechanism points and transformation patterns
        all_mechanism_strengths = []
        all_element_compositions = []

        # Process through transformer layers with five elements attention
        for layer in self.layers:
            hidden_states, layer_info = layer(hidden_states, attention_mask)

            if return_mechanism_info:
                all_mechanism_strengths.append(layer_info["mechanism_strength"])
                all_element_compositions.append(layer_info["element_composition"])

        # Get logits from final hidden states
        logits = self.lm_head(hidden_states)

        # Update mechanism tracking for analysis
        if return_mechanism_info:
            # Average mechanism strengths across layers
            avg_mechanism_strength = torch.stack(all_mechanism_strengths).mean(dim=0)
            self.mechanism_points = 0.9 * self.mechanism_points + 0.1 * avg_mechanism_strength.mean(dim=0)

            # Track element composition history
            avg_composition = torch.tensor([
                sum(comp["wood"] for comp in all_element_compositions) / len(all_element_compositions),
                sum(comp["fire"] for comp in all_element_compositions) / len(all_element_compositions),
                sum(comp["earth"] for comp in all_element_compositions) / len(all_element_compositions),
                sum(comp["metal"] for comp in all_element_compositions) / len(all_element_compositions),
                sum(comp["water"] for comp in all_element_compositions) / len(all_element_compositions),
            ]).to(device)

            self.element_history[:, self.history_pointer] = avg_composition
            self.history_pointer = (self.history_pointer + 1) % 100

            return logits, {
                "mechanism_strengths": all_mechanism_strengths,
                "element_compositions": all_element_compositions,
                "avg_mechanism_strength": avg_mechanism_strength,
                "mechanism_points": self.mechanism_points,
                "element_history": self.element_history
            }

        return logits


class TransformationLayer(nn.Module):
    """
    A transformer layer integrated with Five Elements transformations.
    """

    def __init__(self, hidden_size: int, num_attention_heads: int):
        super().__init__()
        self.attention = FiveElementsAttention(hidden_size, num_attention_heads)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(0.1)
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        # Self-attention with transformations
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        attention_output, attention_info = self.attention(
            hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = residual + attention_output

        # Feed-forward
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, attention_info


class PositionTimingLayer(nn.Module):
    """
    Implements the position-timing unity concept for LLMs.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.position_encoder = nn.Linear(hidden_size, hidden_size)
        self.timing_encoder = nn.Linear(hidden_size, hidden_size)
        self.integrator = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.size()

        # Position encoding (spatial aspects)
        position_features = self.position_encoder(hidden_states)

        # Timing encoding (temporal aspects - position in sequence)
        relative_positions = position_ids.float() / position_ids.max().float()
        relative_positions = relative_positions.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
        timing_input = hidden_states * relative_positions
        timing_features = self.timing_encoder(timing_input)

        # Integrate position and timing
        combined = torch.cat([position_features, timing_features], dim=-1)
        integrated = self.integrator(combined)

        return integrated


def analyze_transformation_patterns(model: MechanismPointsLLM, text: str, tokenizer) -> Dict:
    """
    Analyze transformation patterns in text processing.
    """
    tokens = tokenizer(text, return_tensors="pt")
    input_ids = tokens["input_ids"].to(model.device)
    attention_mask = tokens["attention_mask"].to(model.device)

    # Forward pass with mechanism tracking
    _, mechanism_info = model(input_ids, attention_mask, return_mechanism_info=True)

    # Find top mechanism points
    mechanism_strengths = mechanism_info["avg_mechanism_strength"][0].cpu().numpy()
    top_indices = mechanism_strengths.argsort()[-5:][::-1]

    # Get token strings for mechanism points
    tokens_list = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
    top_tokens = [(tokens_list[idx], mechanism_strengths[idx], idx) for idx in top_indices]

    # Analyze element composition
    element_comp = {
        "wood": sum(c["wood"] for c in mechanism_info["element_compositions"]) / len(
            mechanism_info["element_compositions"]),
        "fire": sum(c["fire"] for c in mechanism_info["element_compositions"]) / len(
            mechanism_info["element_compositions"]),
        "earth": sum(c["earth"] for c in mechanism_info["element_compositions"]) / len(
            mechanism_info["element_compositions"]),
        "metal": sum(c["metal"] for c in mechanism_info["element_compositions"]) / len(
            mechanism_info["element_compositions"]),
        "water": sum(c["water"] for c in mechanism_info["element_compositions"]) / len(
            mechanism_info["element_compositions"]),
    }

    dominant_element = max(element_comp.items(), key=lambda x: x[1])[0]

    return {
        "top_mechanism_points": top_tokens,
        "element_composition": element_comp,
        "dominant_element": dominant_element,
        "element_history": model.element_history.cpu().numpy(),
    }


def demonstrate_position_timing_unity(model: MechanismPointsLLM, tokenizer, texts: List[str]):
    """
    Demonstrate how the same word at different positions creates different effects.
    """
    results = []

    for text in texts:
        analysis = analyze_transformation_patterns(model, text, tokenizer)
        results.append({
            "text": text,
            "top_mechanism_points": analysis["top_mechanism_points"],
            "dominant_element": analysis["dominant_element"]
        })

    return results