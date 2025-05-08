"""
MechanismPointsLLM: A language model implementation incorporating the Mechanism Points
and Transformation Patterns framework based on the Five Elements system.

This implementation includes:
1. A custom tokenizer using SentencePiece
2. A transformer-based LLM with Five Elements attention
3. Training and generation functionality
4. Tools for analyzing mechanism points and transformation patterns
"""

import os
import time
import json
import math
import logging
from typing import Optional, List, Dict, Tuple, Union, Any
from dataclasses import dataclass
import random
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# For tokenization
import sentencepiece as spm
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#######################
# Tokenizer
#######################

class MechanismTokenizer:
    """
    Tokenizer using SentencePiece for subword tokenization with special tokens.
    """
    def __init__(self, model_path=None, vocab_size=32000):
        self.vocab_size = vocab_size
        self.sp_model = None
        self.model_path = model_path
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        
        # Special tokens for mechanism analysis
        self.mechanism_token_id = 4  # Token to mark mechanism points
        
        # Load or initialize
        if model_path and os.path.exists(model_path):
            self.load(model_path)
            logger.info(f"Loaded tokenizer from {model_path}")
        else:
            logger.info("No tokenizer model provided. Use train_tokenizer() to train a new one.")
    
    def train_tokenizer(self, texts, output_path="tokenizer", model_type="unigram"):
        """Train a SentencePiece tokenizer on the given texts."""
        # Create directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Write training data to a file
        data_path = os.path.join(output_path, "training_data.txt")
        with open(data_path, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text + "\n")
        
        # Train SentencePiece model
        input_args = [
            f"--input={data_path}",
            f"--model_prefix={os.path.join(output_path, 'mechanism_tokenizer')}",
            f"--vocab_size={self.vocab_size}",
            f"--model_type={model_type}",
            "--pad_id=0",
            "--bos_id=1",
            "--eos_id=2",
            "--unk_id=3",
            "--user_defined_symbols=<mechanism>",  # Special token for mechanism points
            "--control_symbols=<mask>",  # Additional control token
            "--hard_vocab_limit=false",
            "--normalization_rule_name=identity"  # No normalization for full control
        ]
        
        spm.SentencePieceTrainer.train(" ".join(input_args))
        
        # Set model path and load it
        self.model_path = os.path.join(output_path, "mechanism_tokenizer.model")
        self.load(self.model_path)
        
        logger.info(f"Trained tokenizer saved to {self.model_path}")
        return self.model_path
    
    def load(self, model_path):
        """Load a trained SentencePiece model."""
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(model_path)
        self.vocab_size = self.sp_model.get_piece_size()
        self.model_path = model_path
    
    def __call__(self, texts, padding=True, truncation=True, max_length=1024, return_tensors=None):
        """Tokenize a batch of texts."""
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize each text
        token_ids_list = [self.sp_model.encode(text) for text in texts]
        
        # Apply truncation if needed
        if truncation and max_length is not None:
            token_ids_list = [ids[:max_length] for ids in token_ids_list]
        
        # Get max length for padding
        max_len = max(len(ids) for ids in token_ids_list) if token_ids_list else 0
        
        # Apply padding if needed
        if padding:
            padded_ids = []
            attention_masks = []
            
            for ids in token_ids_list:
                padding_length = max_len - len(ids)
                padded_ids.append(ids + [self.pad_token_id] * padding_length)
                attention_masks.append([1] * len(ids) + [0] * padding_length)
        else:
            padded_ids = token_ids_list
            attention_masks = [[1] * len(ids) for ids in token_ids_list]

        if not padding and return_tensors is None and len(padded_ids) == 1:
            padded_ids = padded_ids[0]
            attention_masks = attention_masks[0]

        # Convert to tensors if requested
        if return_tensors == "pt":
            padded_ids = torch.tensor(padded_ids, dtype=torch.long)
            attention_masks = torch.tensor(attention_masks, dtype=torch.long)
            
            return {
                "input_ids": padded_ids,
                "attention_mask": attention_masks,
            }
        else:
            return {
                "input_ids": padded_ids,
                "attention_mask": attention_masks,
            }
    
    def decode(self, token_ids, skip_special_tokens=True):
        """Decode token IDs back to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy()
        
        if token_ids.ndim > 1:
            return [self.decode(ids, skip_special_tokens) for ids in token_ids]
        
        # Convert to list if needed
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
            
        return self.sp_model.decode(token_ids)
    
    def convert_ids_to_tokens(self, ids):
        """Convert token IDs to token strings."""
        if isinstance(ids, torch.Tensor):
            ids = ids.cpu().numpy().tolist()
        elif isinstance(ids, np.ndarray):
            ids = ids.tolist()
            
        if isinstance(ids, list):
            return [self.sp_model.id_to_piece(id) for id in ids]
        else:
            return self.sp_model.id_to_piece(ids)
    
    def convert_tokens_to_ids(self, tokens):
        """Convert token strings to token IDs."""
        if isinstance(tokens, list):
            return [self.sp_model.piece_to_id(token) for token in tokens]
        else:
            return self.sp_model.piece_to_id(tokens)
    
    def get_vocab_size(self):
        """Return the vocabulary size."""
        return self.vocab_size
    
    def save_pretrained(self, output_dir):
        """Save the tokenizer model to the specified directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the model file
        if self.model_path and os.path.exists(self.model_path):
            output_path = os.path.join(output_dir, "mechanism_tokenizer.model")
            import shutil
            shutil.copy(self.model_path, output_path)
            
            # Save tokenizer config
            config = {
                "vocab_size": self.vocab_size,
                "model_type": "unigram",
                "pad_token_id": self.pad_token_id,
                "bos_token_id": self.bos_token_id,
                "eos_token_id": self.eos_token_id,
                "unk_token_id": self.unk_token_id,
                "mechanism_token_id": self.mechanism_token_id,
            }
            
            with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
                json.dump(config, f, indent=2)
                
            logger.info(f"Tokenizer saved to {output_dir}")
        else:
            logger.warning("No tokenizer model to save.")


#######################
# Model Architecture
#######################

class FiveElementsAttention(nn.Module):
    """
    Attention mechanism based on the Five Elements transformation patterns.
    """
    def __init__(self, hidden_size, num_heads, dropout_prob=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        # Standard transformer attention projections
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        
        # Five Elements transformation projections 
        self.wood_proj = nn.Linear(hidden_size, hidden_size)  # Expansion
        self.fire_proj = nn.Linear(hidden_size, hidden_size)  # Acceleration
        self.earth_proj = nn.Linear(hidden_size, hidden_size)  # Stabilization
        self.metal_proj = nn.Linear(hidden_size, hidden_size)  # Refinement
        self.water_proj = nn.Linear(hidden_size, hidden_size)  # Adaptation
        
        # Element mixing weights (learnable)
        self.element_mixer = nn.Linear(hidden_size, 5)
        
        self.dropout = nn.Dropout(dropout_prob)
        
        # Initialize transformations with appropriate characteristics
        self._initialize_transformations()
    
    def _initialize_transformations(self):
        """Initialize the Five Elements transformation matrices with their characteristic properties."""
        # Wood (Expansion): Encourage broader patterns
        nn.init.normal_(self.wood_proj.weight, std=0.1)
        nn.init.ones_(self.wood_proj.bias)
        
        # Fire (Acceleration): Sharpen attention
        nn.init.normal_(self.fire_proj.weight, std=0.02)
        nn.init.zeros_(self.fire_proj.bias)
        
        # Earth (Stabilization): Create stability through smoothing
        nn.init.xavier_normal_(self.earth_proj.weight)
        nn.init.zeros_(self.earth_proj.bias)
        
        # Metal (Refinement): Create sparse precision through dropout
        nn.init.kaiming_normal_(self.metal_proj.weight)
        nn.init.zeros_(self.metal_proj.bias)
        
        # Water (Adaptation): Context-sensitive flow
        nn.init.xavier_uniform_(self.water_proj.weight)
        nn.init.zeros_(self.water_proj.bias)
    
    def transpose_for_scores(self, x):
        """Reshape for multi-head attention."""
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)  # (batch, heads, seq_len, head_dim)
    
    def calculate_mechanism_strength(self, attention_weights):
        """
        Calculate mechanism strength from attention weights.
        Higher values indicate positions with greater influence on other positions.
        """
        # Average attention across heads
        avg_attention = attention_weights.mean(dim=1)  # (batch, seq_len, seq_len)
        
        # Calculate row-wise influence (how much each token influences others)
        mechanism_strength = avg_attention.sum(dim=2)  # (batch, seq_len)
        
        # Normalize to [0, 1]
        mechanism_strength = (mechanism_strength - mechanism_strength.min(dim=1, keepdim=True)[0]) / \
                             (mechanism_strength.max(dim=1, keepdim=True)[0] - 
                              mechanism_strength.min(dim=1, keepdim=True)[0] + 1e-9)
        
        return mechanism_strength
    
    def calculate_element_composition(self, hidden_states):
        """Calculate the element composition of the current state using learnable weights."""
        # Get the average hidden state
        avg_hidden = hidden_states.mean(dim=1)  # (batch, hidden_size)
        
        # Project to element weights
        element_logits = self.element_mixer(avg_hidden)  # (batch, 5)
        element_weights = F.softmax(element_logits, dim=-1)  # (batch, 5)
        
        # Create dictionary for tracking
        element_composition = {
            "wood": element_weights[:, 0],
            "fire": element_weights[:, 1],
            "earth": element_weights[:, 2],
            "metal": element_weights[:, 3],
            "water": element_weights[:, 4],
        }
        
        return element_composition, element_weights
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False
    ):
        batch_size, seq_length, _ = hidden_states.size()
        
        # Project queries, keys, values
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Calculate attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 4:  # extended mask [b,1,1,seq_len]
                attention_scores = attention_scores + attention_mask
            else:  # basic 2-D mask [b,seq_len]
                # turn 0→masked into -10 000 and broadcast to 4-D
                extended = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * -10000.0
                attention_scores = attention_scores + extended
        
        # Apply softmax for probability distribution
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Calculate context from attention and values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_shape)  # (batch, seq_len, hidden_size)
        
        # Calculate mechanism strength
        mechanism_strength = self.calculate_mechanism_strength(attention_probs)
        
        # Calculate element composition
        element_composition, element_weights = self.calculate_element_composition(hidden_states)
        
        # Apply Five Elements transformations
        wood_output = self.wood_proj(context_layer)  # Expansion
        fire_output = self.fire_proj(context_layer)  # Acceleration
        earth_output = self.earth_proj(context_layer)  # Stabilization
        metal_output = self.metal_proj(context_layer)  # Refinement
        water_output = self.water_proj(context_layer)  # Adaptation
        
        # Combine transformations weighted by element composition
        transformed_output = (
            element_weights[:, 0:1, None] * wood_output +
            element_weights[:, 1:2, None] * fire_output +
            element_weights[:, 2:3, None] * earth_output +
            element_weights[:, 3:4, None] * metal_output +
            element_weights[:, 4:5, None] * water_output
        )
        
        # Final projection
        attention_output = self.output(transformed_output)
        
        outputs = (attention_output, mechanism_strength, element_composition)
        if output_attentions:
            outputs += (attention_probs,)
        
        return outputs


class PositionTimingIntegration(nn.Module):
    """
    Implements the position-timing unity concept for language models.
    """
    def __init__(self, hidden_size, max_position_embeddings=2048):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        
        # Position embedding
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        # Position-timing integration
        self.position_encoder = nn.Linear(hidden_size, hidden_size)
        self.timing_encoder = nn.Linear(hidden_size, hidden_size)
        self.integrator = nn.Linear(hidden_size * 2, hidden_size)
        
        # Initialize position embeddings
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.position_embeddings.weight, std=0.02)
    
    def forward(self, hidden_states, position_ids=None, past_key_values_length=0):
        seq_length = hidden_states.size(1)
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length,
                dtype=torch.long, device=hidden_states.device
            )
            position_ids = position_ids.unsqueeze(0).expand(hidden_states.size(0), -1)
        
        # Create position embeddings
        position_embeds = self.position_embeddings(position_ids)
        
        # Create timing factor based on relative position in sequence
        relative_positions = position_ids.float() / self.max_position_embeddings
        relative_positions = relative_positions.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
        
        # Position encoding (spatial aspects)
        position_features = self.position_encoder(position_embeds)
        
        # Timing encoding (temporal aspects - position in sequence)
        timing_input = hidden_states * relative_positions
        timing_features = self.timing_encoder(timing_input)
        
        # Integrate position and timing
        combined = torch.cat([position_features, timing_features], dim=-1)
        integrated = self.integrator(combined)
        
        # Add the integrated position-timing to hidden states
        integrated_hidden_states = hidden_states + integrated
        
        return integrated_hidden_states


class TransformationLayer(nn.Module):
    """
    Transformer layer with Five Elements attention mechanism.
    """
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout_prob=0.1):
        super().__init__()
        
        # Five Elements Attention
        self.attention = FiveElementsAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            dropout_prob=dropout_prob
        )
        
        # Feed-forward network
        self.intermediate = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout_prob)
        )
        
        # Layer normalization
        self.attention_layernorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.output_layernorm = nn.LayerNorm(hidden_size, eps=1e-12)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        residual = hidden_states
        
        # Layer norm before attention
        hidden_states = self.attention_layernorm(hidden_states)
        
        # Apply attention
        attention_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        
        # Extract outputs
        attention_output = attention_outputs[0]
        mechanism_strength = attention_outputs[1]
        element_composition = attention_outputs[2]
        
        # Add residual connection
        hidden_states = residual + self.dropout(attention_output)
        
        # Feed-forward with residual connection
        residual = hidden_states
        hidden_states = self.output_layernorm(hidden_states)
        hidden_states = residual + self.dropout(self.intermediate(hidden_states))
        
        outputs = (hidden_states, mechanism_strength, element_composition)
        if output_attentions:
            outputs += (attention_outputs[3],)
        
        return outputs


class MechanismPointsLLM(nn.Module):
    """
    Language model incorporating the Mechanism Points and Transformation Patterns framework.
    """
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=2048,
        dropout_prob=0.1,
        pad_token_id=0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        
        # Word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        
        # Position-timing integration
        self.position_timing = PositionTimingIntegration(
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings
        )
        
        # Embedding layernorm and dropout
        self.layernorm_embedding = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformationLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                dropout_prob=dropout_prob
            )
            for _ in range(num_hidden_layers)
        ])
        
        # Output layer
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Element history buffer (for tracking purposes)
        self.register_buffer("element_history", torch.zeros(5, 100))  # Track last 100 steps
        self.history_pointer = 0
        
        # Tie weights if needed
        self.tie_weights()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights."""
        # Initialize embeddings
        nn.init.normal_(self.word_embeddings.weight, std=0.02)
        
        # Initialize LM head
        nn.init.normal_(self.lm_head.weight, std=0.02)
    
    def tie_weights(self):
        """Tie the weights between the input embeddings and the output embeddings."""
        self.lm_head.weight = self.word_embeddings.weight
    
    def get_input_embeddings(self):
        return self.word_embeddings
    
    def set_input_embeddings(self, new_embeddings):
        self.word_embeddings = new_embeddings
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values_length=0,
        labels=None,
        output_attentions=False,
        output_mechanism_info=False,
        return_dict=True,
    ):
        # Process inputs
        batch_size, seq_length = input_ids.size()
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=input_ids.device)
        
        # Create extended attention mask for transformer [batch, 1, 1, seq_length]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Get word embeddings
        hidden_states = self.word_embeddings(input_ids)
        
        # Apply position-timing integration
        hidden_states = self.position_timing(
            hidden_states,
            position_ids=position_ids,
            past_key_values_length=past_key_values_length
        )
        
        # Apply embedding layernorm and dropout
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Storage for mechanism information
        all_mechanism_strengths = []
        all_element_compositions = []
        all_attentions = [] if output_attentions else None
        
        # Process through transformer layers
        for i, layer in enumerate(self.layers):
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=extended_attention_mask,
                output_attentions=output_attentions
            )
            
            # Extract outputs
            hidden_states = layer_outputs[0]
            mechanism_strength = layer_outputs[1]
            element_composition = layer_outputs[2]
            
            # Store mechanism information if requested
            if output_mechanism_info:
                all_mechanism_strengths.append(mechanism_strength)
                all_element_compositions.append(element_composition)
            
            # Store attention outputs if requested
            if output_attentions:
                all_attentions.append(layer_outputs[3])
        
        # Get logits from final hidden states
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
        
        # Update element history for tracking
        if output_mechanism_info and len(all_element_compositions) > 0:
            avg_wood = torch.mean(torch.stack([comp["wood"].mean() for comp in all_element_compositions]))
            avg_fire = torch.mean(torch.stack([comp["fire"].mean() for comp in all_element_compositions]))
            avg_earth = torch.mean(torch.stack([comp["earth"].mean() for comp in all_element_compositions]))
            avg_metal = torch.mean(torch.stack([comp["metal"].mean() for comp in all_element_compositions]))
            avg_water = torch.mean(torch.stack([comp["water"].mean() for comp in all_element_compositions]))
            
            self.element_history[:, self.history_pointer] = torch.tensor(
                [avg_wood, avg_fire, avg_earth, avg_metal, avg_water],
                device=self.element_history.device
            )
            self.history_pointer = (self.history_pointer + 1) % 100
        
        # Return results
        if not return_dict:
            outputs = (logits, hidden_states)
            if output_mechanism_info:
                outputs += (all_mechanism_strengths, all_element_compositions)
            if output_attentions:
                outputs += (all_attentions,)
            if loss is not None:
                outputs = (loss,) + outputs
            return outputs
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": hidden_states,
            "mechanism_strengths": all_mechanism_strengths if output_mechanism_info else None,
            "element_compositions": all_element_compositions if output_mechanism_info else None,
            "attentions": all_attentions,
        }
    
    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=100,
        min_length=0,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.0,
        pad_token_id=None,
        eos_token_id=None,
        track_mechanism_points=False,
        return_dict=True,
    ):
        """
        Generate text using the model with transformation pattern awareness.
        """
        # Set defaults if not provided
        pad_token_id = pad_token_id if pad_token_id is not None else self.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else None
        
        # Ensure input is on the correct device
        input_ids = input_ids.to(self.word_embeddings.weight.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(input_ids.device)
        
        # Storage for mechanism information if tracking
        all_mechanism_points = [] if track_mechanism_points else None
        all_element_flows = [] if track_mechanism_points else None
        
        # Initialize generation outputs with input_ids
        batch_size, input_seq_length = input_ids.size()
        generated_ids = input_ids.clone()
        
        # Create position IDs
        position_ids = torch.arange(input_seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Generation loop
        for _ in range(max_length - input_seq_length):
            # Limit sequence length if needed
            if generated_ids.size(1) > self.max_position_embeddings:
                generated_ids = generated_ids[:, -self.max_position_embeddings:]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, -self.max_position_embeddings:]
                position_ids = position_ids[:, -self.max_position_embeddings:]
            
            # Forward pass
            outputs = self.forward(
                input_ids=generated_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_mechanism_info=track_mechanism_points,
                return_dict=True,
            )
            
            # Get next token logits (last token in sequence)
            next_token_logits = outputs["logits"][:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(generated_ids[i].tolist()):
                        next_token_logits[i, token_id] /= repetition_penalty
            
            # Filter logits with top-k
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float("Inf")
            
            # Filter with top-p (nucleus sampling)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                for i in range(batch_size):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    next_token_logits[i, indices_to_remove] = -float("Inf")
            
            # Sample or greedy select next token
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Store mechanism information if tracking
            if track_mechanism_points and outputs["mechanism_strengths"] is not None:
                # Average mechanism strengths across layers for the last token
                avg_strength = torch.stack([ms[:, -1] for ms in outputs["mechanism_strengths"]]).mean(0)
                all_mechanism_points.append(avg_strength)
                
                # Average element composition across layers
                avg_elements = {
                    element: torch.stack([comp[element].mean() for comp in outputs["element_compositions"]]).mean()
                    for element in ["wood", "fire", "earth", "metal", "water"]
                }
                all_element_flows.append(avg_elements)
            
            # Check for EOS token
            if eos_token_id is not None and (next_tokens == eos_token_id).any():
                # Set all EOS tokens to pad tokens for batches that haven't reached min_length
                if min_length > 0:
                    for i in range(batch_size):
                        if position_ids[i, -1] < min_length:
                            next_tokens[i] = pad_token_id
            
            # Update inputs
            generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Update attention mask if needed
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), device=attention_mask.device)
                ], dim=-1)
            
            # Update position IDs
            position_ids = torch.cat([
                position_ids,
                (position_ids[:, -1] + 1).unsqueeze(-1)
            ], dim=-1)
            
            # Check if all sequences have hit EOS
            if eos_token_id is not None and (generated_ids[:, -1] == eos_token_id).all():
                break
        
        # Return results
        if not return_dict:
            if track_mechanism_points:
                return generated_ids, all_mechanism_points, all_element_flows
            return generated_ids
        
        return {
            "generated_ids": generated_ids,
            "mechanism_points": all_mechanism_points,
            "element_flows": all_element_flows,
        }
    
    def save_pretrained(self, save_directory):
        """Save model to the specified directory."""
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            
        # Save model config
        config = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "pad_token_id": self.pad_token_id,
        }
        
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        logger.info(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, model_path):
        """Load model from the specified directory."""
        # Load config
        with open(os.path.join(model_path, "config.json"), "r") as f:
            config = json.load(f)
        
        # Create model with config
        model = cls(**config)
        
        # Load weights
        model.load_state_dict(torch.load(
            os.path.join(model_path, "pytorch_model.bin"),
            map_location=torch.device("cpu")
        ))
        
        logger.info(f"Model loaded from {model_path}")
        return model


#######################
# Training
#######################

class TextDataset(Dataset):
    """Dataset for language modeling."""
    def __init__(self, texts, tokenizer, max_length=512, chunk_mode="overlap"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_mode = chunk_mode
        
        # Tokenize and chunk all texts
        self.examples = self._prepare_examples(texts)
        logger.info(f"Created dataset with {len(self.examples)} examples")
    
    def _prepare_examples(self, texts):
        """Tokenize texts and split into chunks."""
        examples = []

        for text in tqdm(texts, desc="Preparing examples"):
            # Tokenize
            tok_out = self.tokenizer(text, padding=False, truncation=False)
            input_ids = tok_out["input_ids"]
            if isinstance(input_ids[0], (list, tuple)):  # tokenizer returned [[...]]
                input_ids = input_ids[0]

            # Always keep a copy of the (possibly short) sequence
            if len(input_ids) <= self.max_length:
                examples.append({"input_ids": input_ids})
                continue
            
            # Split into chunks
            if self.chunk_mode == "non-overlap":
                # Non-overlapping chunks
                for i in range(0, len(input_ids), self.max_length):
                    chunk = input_ids[i:i + self.max_length]
                    if len(chunk) >= self.max_length // 4:  # Only keep chunks of reasonable size
                        examples.append({"input_ids": chunk})
            
            elif self.chunk_mode == "overlap":
                # Overlapping chunks with 50% overlap
                stride = self.max_length // 2
                for i in range(0, len(input_ids) - stride, stride):
                    chunk = input_ids[i:i + self.max_length]
                    if len(chunk) >= self.max_length // 2:
                        examples.append({"input_ids": chunk})
            
            else:  # "document" mode
                # Keep full documents, truncate if too long
                if len(input_ids) > self.max_length:
                    examples.append({"input_ids": input_ids[:self.max_length]})
                else:
                    examples.append({"input_ids": input_ids})
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        input_ids = example["input_ids"]
        
        # Pad or truncate to max_length
        if len(input_ids) < self.max_length:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        elif len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        
        # Create attention mask
        attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in input_ids]
        
        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),  # For autoregressive language modeling
        }


def collate_fn(examples):
    """Collate function for DataLoader."""
    batch = {}
    
    for key in examples[0].keys():
        batch[key] = torch.stack([example[key] for example in examples])
    
    return batch


class Trainer:
    """Trainer for the MechanismPointsLLM model."""
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset=None,
        batch_size=8,
        learning_rate=5e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,
        num_epochs=3,
        warmup_steps=0,
        output_dir="output",
        device=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.output_dir = output_dir
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up data loaders
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        
        if eval_dataset:
            self.eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                collate_fn=collate_fn,
            )
        else:
            self.eval_dataloader = None
        
        # Set up optimizer and scheduler
        self._setup_optimizer_and_scheduler()
        
        # Move model to device
        self.model.to(self.device)
        
        # For tracking training
        self.global_step = 0
        self.best_eval_loss = float("inf")
        self.training_logs = []
    
    def _setup_optimizer_and_scheduler(self):
        """Set up optimizer and learning rate scheduler."""
        # Prepare optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Set up scheduler
        total_steps = len(self.train_dataloader) * self.num_epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=total_steps - self.warmup_steps
        )
    
    def train(self):
        """Train the model."""
        logger.info("Starting training...")
        self.model.train()
        
        total_loss = 0
        logging_loss = 0
        start_time = time.time()
        
        # Training loop
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            epoch_iterator = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
            
            for step, batch in enumerate(epoch_iterator):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    output_mechanism_info=(step % 100 == 0),  # Only track mechanism info occasionally
                )
                
                loss = outputs["loss"]
                
                # Backward pass
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Update weights
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Update tracking
                total_loss += loss.item()
                self.global_step += 1
                
                # Log progress
                if self.global_step % 10 == 0:
                    avg_loss = (total_loss - logging_loss) / 10
                    logging_loss = total_loss
                    
                    # Update progress bar
                    epoch_iterator.set_postfix(loss=avg_loss)
                    
                    # Log element composition if available
                    if outputs["element_compositions"] is not None:
                        avg_elements = {
                            element: torch.stack([comp[element].mean() for comp in outputs["element_compositions"]]).mean().item()
                            for element in ["wood", "fire", "earth", "metal", "water"]
                        }
                        logger.info(f"Element composition: {avg_elements}")
                
                # Save checkpoint and evaluate occasionally
                if self.global_step % 500 == 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step}")
                    
                    if self.eval_dataloader:
                        eval_results = self.evaluate()
                        self.model.train()  # Set back to train mode
                        
                        # Save if best
                        if eval_results["loss"] < self.best_eval_loss:
                            self.best_eval_loss = eval_results["loss"]
                            self.save_checkpoint("best_model")
                        
                        # Log evaluation results
                        self.training_logs.append({
                            "step": self.global_step,
                            "train_loss": avg_loss,
                            "eval_loss": eval_results["loss"],
                            "eval_perplexity": eval_results["perplexity"],
                        })
                        
                        # Save training logs
                        with open(os.path.join(self.output_dir, "training_logs.json"), "w") as f:
                            json.dump(self.training_logs, f, indent=2)
            
            # End of epoch
            if self.eval_dataloader:
                eval_results = self.evaluate()
                self.model.train()
                logger.info(f"Epoch {epoch+1} evaluation: {eval_results}")
        
        # End of training
        logger.info(f"Training completed in {time.time() - start_time:.2f}s")
        
        # Save final model
        self.save_checkpoint("final_model")
        
        return self.training_logs
    
    def evaluate(self):
        """Evaluate the model on the evaluation dataset."""
        if not self.eval_dataloader:
            return {"loss": 0.0, "perplexity": 0.0}
        
        logger.info("Running evaluation...")
        self.model.eval()
        
        total_loss = 0
        total_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                
                loss = outputs["loss"]
                total_loss += loss.item()
                total_steps += 1
        
        avg_loss = total_loss / total_steps
        perplexity = math.exp(avg_loss)
        
        results = {
            "loss": avg_loss,
            "perplexity": perplexity,
        }
        
        logger.info(f"Evaluation results: {results}")
        return results
    
    def save_checkpoint(self, checkpoint_name):
        """Save a model checkpoint."""
        checkpoint_dir = os.path.join(self.output_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        torch.save({
            "global_step": self.global_step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_eval_loss": self.best_eval_loss,
        }, os.path.join(checkpoint_dir, "trainer_state.bin"))
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir):
        """Load a model checkpoint."""
        # Load model
        self.model = MechanismPointsLLM.from_pretrained(checkpoint_dir)
        self.model.to(self.device)
        
        # Load training state
        training_state = torch.load(
            os.path.join(checkpoint_dir, "trainer_state.bin"),
            map_location=self.device
        )
        
        self.global_step = training_state["global_step"]
        self.best_eval_loss = training_state["best_eval_loss"]
        
        # Reset optimizer and scheduler
        self._setup_optimizer_and_scheduler()
        
        # Load optimizer and scheduler states
        self.optimizer.load_state_dict(training_state["optimizer_state_dict"])
        self.scheduler.load_state_dict(training_state["scheduler_state_dict"])
        
        logger.info(f"Checkpoint loaded from {checkpoint_dir}")


#######################
# Analysis Tools
#######################

def analyze_mechanism_points(model, text, tokenizer):
    """Identify mechanism points and analyze transformation patterns in text."""
    # Tokenize input
    tokens = tokenizer(text, return_tensors="pt")
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    
    # Run forward pass with mechanism tracking
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_mechanism_info=True,
            output_attentions=True,
        )
    
    # Get token strings
    tokens_list = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    
    # Average mechanism strengths across layers
    mechanism_strengths = torch.stack([s[0] for s in outputs["mechanism_strengths"]]).mean(0)
    
    # Find top mechanism points
    top_indices = torch.argsort(mechanism_strengths, descending=True)[:5].cpu().numpy()
    
    # Get element composition
    element_composition = {
        element: torch.stack([comp[element][0] for comp in outputs["element_compositions"]]).mean().item()
        for element in ["wood", "fire", "earth", "metal", "water"]
    }
    
    # Organize results
    results = {
        "tokens": tokens_list,
        "mechanism_strengths": mechanism_strengths.cpu().numpy(),
        "top_mechanism_points": [(tokens_list[idx], float(mechanism_strengths[idx]), int(idx)) 
                                for idx in top_indices],
        "element_composition": element_composition,
        "dominant_element": max(element_composition.items(), key=lambda x: x[1])[0],
    }
    
    return results


def visualize_transformation_patterns(model, texts, tokenizer):
    """Visualize transformation patterns across different texts."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Analyze multiple texts
    all_results = []
    element_names = ["wood", "fire", "earth", "metal", "water"]
    
    for text in texts:
        all_results.append(analyze_mechanism_points(model, text, tokenizer))
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot element compositions
    compositions = np.array([[result["element_composition"][element] for element in element_names] 
                             for result in all_results])
    
    im = axes[0].imshow(compositions, cmap="viridis")
    axes[0].set_xticks(np.arange(len(element_names)))
    axes[0].set_yticks(np.arange(len(texts)))
    axes[0].set_xticklabels(element_names)
    axes[0].set_yticklabels([f"Text {i+1}" for i in range(len(texts))])
    axes[0].set_title("Element Composition Across Texts")
    plt.colorbar(im, ax=axes[0])
    
    # Plot mechanism strength distribution for first text
    mech_strength = all_results[0]["mechanism_strengths"]
    token_labels = all_results[0]["tokens"]
    
    # Truncate if too many tokens
    if len(token_labels) > 30:
        token_labels = token_labels[:30]
        mech_strength = mech_strength[:30]
    
    axes[1].bar(range(len(token_labels)), mech_strength[:len(token_labels)])
    axes[1].set_xticks(range(len(token_labels)))
    axes[1].set_xticklabels(token_labels, rotation=45, ha="right")
    axes[1].set_title("Mechanism Strength by Token (Text 1)")
    axes[1].set_xlabel("Tokens")
    axes[1].set_ylabel("Mechanism Strength")
    
    plt.tight_layout()
    plt.savefig("transformation_patterns.png")
    plt.close()
    
    print(f"Visualization saved to transformation_patterns.png")
    return all_results


def demonstrate_position_timing(model, tokenizer, template="The {position} cat", positions=["happy", "angry", "sleepy"]):
    """Demonstrate how changing a word's position affects text generation."""
    # Create texts with different positions
    texts = [template.format(position=pos) for pos in positions]
    
    # Generate completions
    results = []
    
    for text in texts:
        # Tokenize
        tokens = tokenizer(text, return_tensors="pt")
        input_ids = tokens["input_ids"].to(device)
        
        # Generate
        with torch.no_grad():
            generation = model.generate(
                input_ids=input_ids,
                max_length=50,
                do_sample=True,
                temperature=0.7,
                track_mechanism_points=True,
                return_dict=True,
            )
        
        # Decode
        generated_text = tokenizer.decode(generation["generated_ids"][0], skip_special_tokens=True)
        
        # Analyze mechanism points
        mech_analysis = analyze_mechanism_points(model, text, tokenizer)
        
        results.append({
            "input": text,
            "generated": generated_text,
            "dominant_element": mech_analysis["dominant_element"],
            "top_mechanism_points": mech_analysis["top_mechanism_points"],
        })
    
    return results


#######################
# Main Functions
#######################

def train_model_from_scratch(texts, output_dir="model", **kwargs):
    """Train a model from scratch on the provided texts."""
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Train tokenizer
    logger.info("Training tokenizer...")
    tokenizer = MechanismTokenizer(vocab_size=kwargs.get("vocab_size", 32000))
    tokenizer_path = tokenizer.train_tokenizer(texts, output_path=os.path.join(output_dir, "tokenizer"))
    
    # Step 2: Create datasets
    logger.info("Creating datasets...")
    # Split texts into train/eval (90/10)
    random.shuffle(texts)
    split_point = int(len(texts) * 0.9)
    train_texts = texts[:split_point]
    eval_texts = texts[split_point:]
    
    # Create datasets
    max_length = kwargs.get("max_length", 512)
    train_dataset = TextDataset(train_texts, tokenizer, max_length=max_length)
    eval_dataset = TextDataset(eval_texts, tokenizer, max_length=max_length)
    
    # Step 3: Initialize model
    logger.info("Initializing model...")
    model_args = {
        "vocab_size": tokenizer.get_vocab_size(),
        "hidden_size": kwargs.get("hidden_size", 768),
        "num_hidden_layers": kwargs.get("num_hidden_layers", 12),
        "num_attention_heads": kwargs.get("num_attention_heads", 12),
        "intermediate_size": kwargs.get("intermediate_size", 3072),
        "max_position_embeddings": kwargs.get("max_position_embeddings", 2048),
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    model = MechanismPointsLLM(**model_args)
    
    # Step 4: Train model
    logger.info("Training model...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=kwargs.get("batch_size", 8),
        learning_rate=kwargs.get("learning_rate", 5e-5),
        num_epochs=kwargs.get("num_epochs", 3),
        output_dir=os.path.join(output_dir, "checkpoints"),
    )
    
    trainer.train()
    
    # Return model and tokenizer
    return model, tokenizer


def load_trained_model(model_dir):
    """Load a trained model and tokenizer."""
    # Check if model exists
    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory {model_dir} does not exist.")
    
    # Load tokenizer
    tokenizer_path = os.path.join(model_dir, "tokenizer")
    if os.path.exists(os.path.join(tokenizer_path, "mechanism_tokenizer.model")):
        tokenizer = MechanismTokenizer(model_path=os.path.join(tokenizer_path, "mechanism_tokenizer.model"))
    else:
        raise ValueError(f"Tokenizer not found in {tokenizer_path}")
    
    # Load best model if exists, otherwise final model
    best_model_path = os.path.join(model_dir, "checkpoints", "best_model")
    final_model_path = os.path.join(model_dir, "checkpoints", "final_model")
    
    if os.path.exists(best_model_path):
        model_path = best_model_path
    elif os.path.exists(final_model_path):
        model_path = final_model_path
    else:
        raise ValueError(f"No trained model found in {model_dir}")
    
    # Load model
    model = MechanismPointsLLM.from_pretrained(model_path)
    model.to(device)
    
    return model, tokenizer


def generate_text(model, tokenizer, prompt, **kwargs):
    """Generate text using the model."""
    # Tokenize prompt
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens["input_ids"].to(device)
    
    # Generate text
    generation_kwargs = {
        "max_length": kwargs.get("max_length", 100),
        "min_length": kwargs.get("min_length", 10),
        "do_sample": kwargs.get("do_sample", True),
        "temperature": kwargs.get("temperature", 0.7),
        "top_k": kwargs.get("top_k", 50),
        "top_p": kwargs.get("top_p", 0.9),
        "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
        "track_mechanism_points": kwargs.get("track_mechanism_points", True),
    }
    
    output = model.generate(input_ids=input_ids, **generation_kwargs)
    
    # Process output
    if isinstance(output, dict):
        generated_ids = output["generated_ids"]
        mechanism_points = output["mechanism_points"]
        element_flows = output["element_flows"]
        
        # Decode generated text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Analyze top mechanism points
        if mechanism_points:
            # Average mechanism strength across tokens
            avg_mech_strength = torch.cat(mechanism_points).cpu().detach().numpy()
            
            # Get top mechanism indices
            top_indices = np.argsort(avg_mech_strength)[-5:][::-1]
            
            # Get associated tokens
            gen_tokens = tokenizer.convert_ids_to_tokens(generated_ids[0].cpu().numpy())
            top_mechanisms = [(gen_tokens[idx], avg_mech_strength[idx], idx) 
                             for idx in top_indices if idx < len(gen_tokens)]
            
            # Get dominant element
            if element_flows:
                avg_elements = {}
                for element in ["wood", "fire", "earth", "metal", "water"]:
                    avg_elements[element] = np.mean([flow[element].item() for flow in element_flows])
                
                dominant_element = max(avg_elements.items(), key=lambda x: x[1])[0]
            else:
                dominant_element = None
            
            return {
                "text": generated_text,
                "top_mechanism_points": top_mechanisms,
                "dominant_element": dominant_element,
                "element_composition": avg_elements if element_flows else None,
            }
        
        return {"text": generated_text}
    else:
        # For older model versions that don't return dict
        return {"text": tokenizer.decode(output[0], skip_special_tokens=True)}


def interactive_generation(model, tokenizer, initial_prompt=None):
    """Interactive text generation with analysis of mechanism points."""
    print("=" * 50)
    print("Interactive Generation with Mechanism Points Analysis")
    print("Type 'exit' to quit")
    print("=" * 50)
    
    context = ""
    if initial_prompt:
        context = initial_prompt
        print(f"Initial prompt: {initial_prompt}")
    
    while True:
        if not context:
            user_input = input("\nEnter prompt: ")
            if user_input.lower() == "exit":
                break
            context = user_input
        else:
            user_input = input("\nContinue generation? [Y/n/clear/exit]: ")
            if user_input.lower() == "exit":
                break
            elif user_input.lower() == "n":
                user_input = input("Enter new prompt: ")
                if user_input.lower() == "exit":
                    break
                context = user_input
            elif user_input.lower() == "clear":
                context = ""
                continue
        
        # Generate text
        result = generate_text(
            model, 
            tokenizer, 
            context, 
            max_length=min(len(context) + 100, 1024),
            temperature=0.7,
            track_mechanism_points=True
        )
        
        # Print generated text
        print("\nGenerated text:")
        print(result["text"])
        
        # Print mechanism point analysis
        if "top_mechanism_points" in result:
            print("\nTop mechanism points:")
            for token, strength, idx in result["top_mechanism_points"]:
                print(f"  {token} (position {idx}): {strength:.4f}")
        
        if "dominant_element" in result and result["dominant_element"]:
            print(f"\nDominant element: {result['dominant_element']}")
            
            if result["element_composition"]:
                print("Element composition:")
                for element, value in result["element_composition"].items():
                    print(f"  {element}: {value:.4f}")
        
        # Update context for continuation
        context = result["text"]


def batch_analyze_texts(model, tokenizer, texts):
    """Analyze mechanism points and transformation patterns in multiple texts."""
    results = []
    
    for i, text in enumerate(texts):
        logger.info(f"Analyzing text {i+1}/{len(texts)}")
        analysis = analyze_mechanism_points(model, text, tokenizer)
        results.append(analysis)
    
    # Aggregate results
    dominant_elements = [result["dominant_element"] for result in results]
    element_counts = {element: dominant_elements.count(element) for element in set(dominant_elements)}
    
    # Average element composition across all texts
    avg_composition = {}
    for element in ["wood", "fire", "earth", "metal", "water"]:
        avg_composition[element] = sum(result["element_composition"][element] for result in results) / len(results)
    
    # Find overall dominant element
    overall_dominant = max(avg_composition.items(), key=lambda x: x[1])[0]
    
    logger.info(f"Analysis complete. Overall dominant element: {overall_dominant}")
    logger.info(f"Element composition: {avg_composition}")
    logger.info(f"Element counts: {element_counts}")
    
    return {
        "text_analyses": results,
        "overall_dominant_element": overall_dominant,
        "avg_element_composition": avg_composition,
        "element_counts": element_counts,
    }


#######################
# Example Usage
#######################

def example_usage():
    """Example of how to use the MechanismPointsLLM framework."""
    
    # Sample texts for training
    sample_texts = [
        "The mechanism points concept focuses on identifying critical leverage points in complex systems.",
        "According to the Five Elements theory, wood represents expansion, fire represents transformation, earth represents stabilization, metal represents refinement, and water represents adaptation.",
        "In Conway's Game of Life, small changes to initial conditions can create dramatically different outcomes, demonstrating sensitivity to mechanism points.",
        "Position-timing unity suggests that 'where' and 'when' to intervene in a system are not separate considerations but aspects of a unified manifold.",
        "The I Ching describes 64 hexagrams representing different transformation patterns between states.",
        # Add more training texts here
    ]
    from datasets import load_dataset

    # 1️⃣  Download + cache (only the first call hits the network)
    tiny = load_dataset("karpathy/tiny_shakespeare", trust_remote_code=True)  # returns a DatasetDict

    # 2️⃣  Peek at what you got
    print(tiny)  # ⇢ DatasetDict({'train': Dataset(num_rows=1, columns=['text'])})
    print(tiny["train"][0]["text"][:500])  # first 500 chars

    # 3️⃣  Split the single long string into usable chunks
    full_text = tiny["train"][0]["text"]  # one giant string (~1 MB)

    # Example: break by newline for simple sentence-ish units
    texts = [line.strip() for line in full_text.split("\n") if line.strip()]

    # 4️⃣  Feed into your pipeline
    model, tokenizer = train_model_from_scratch(
        texts,
        output_dir="model_tiny_shakespeare",
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=6,
        num_epochs=3,
    )
    # Either train a new model or load an existing one
    #try:
    #    model, tokenizer = load_trained_model("model")
    #    logger.info("Loaded existing model")
    #except:
    #    logger.info("No existing model found. Training new model.")
    #    model, tokenizer = train_model_from_scratch(
    #        sample_texts,
    #        output_dir="model",
    #        hidden_size=256,  # Small model for demonstration
    #        num_hidden_layers=4,
    #        num_attention_heads=4,
    #        batch_size=2,
    #        num_epochs=1,
    #    )
    
    # Generate text
    prompt = "The key insight about mechanism points is"
    result = generate_text(model, tokenizer, prompt)
    print(f"Generated text: {result['text']}")
    
    # Analyze mechanism points
    analysis = analyze_mechanism_points(model, result["text"], tokenizer)
    print(f"Dominant element: {analysis['dominant_element']}")
    print("Top mechanism points:")
    for token, strength, idx in analysis["top_mechanism_points"]:
        print(f"  {token} (position {idx}): {strength:.4f}")
    
    # Interactive generation
    interactive_generation(model, tokenizer)


if __name__ == "__main__":
    example_usage()