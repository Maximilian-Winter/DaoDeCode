import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import time
import numpy as np
import re
from typing import Optional, Tuple, List, Dict, Union, Any
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import random
import json
from tqdm import tqdm
from transformers import AutoTokenizer


# --------------------------
# 1. MECHANISM TOKENIZER
# --------------------------

class MechanismTokenizer:
    """Tokenizer that identifies and utilizes mechanism points in text segmentation"""

    def __init__(self, vocab_file=None, vocab_size=10000, unk_token="<unk>",
                 pad_token="<pad>", bos_token="<s>", eos_token="</s>"):
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        # Load vocabulary if provided, otherwise initialize empty
        if vocab_file and os.path.exists(vocab_file):
            with open(vocab_file, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)
            self.vocab_size = len(self.vocab)
            # Create inverse mapping (id -> token)
            self.id_to_token = {v: k for k, v in self.vocab.items()}
        else:
            # Initialize with special tokens
            self.vocab = {
                pad_token: 0,
                unk_token: 1,
                bos_token: 2,
                eos_token: 3
            }
            self.id_to_token = {0: pad_token, 1: unk_token, 2: bos_token, 3: eos_token}
            self.vocab_size = len(self.vocab)

        # Mechanism rating for each token boundary pattern
        self.boundary_mechanisms = {}

    def train_from_texts(self, texts, min_freq=5, mechanism_threshold=0.01):
        """Build vocabulary from texts, identifying mechanism boundaries"""
        # Count token frequencies
        word_freq = {}
        bigram_freq = {}

        # First pass: count words and bigrams
        for text in tqdm(texts, desc="Counting tokens"):
            # Simple word-level tokenization for initial counting
            words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())

            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1

            for i in range(len(words) - 1):
                bigram = words[i] + " " + words[i + 1]
                bigram_freq[bigram] = bigram_freq.get(bigram, 0) + 1

        # Filter by frequency and calculate token boundary mechanisms
        qualified_words = [w for w, c in word_freq.items() if c >= min_freq]

        # Calculate mechanism strength for word boundaries
        # (where breaking/not breaking significantly affects meaning)
        for bigram, count in tqdm(bigram_freq.items(), desc="Calculating mechanisms"):
            if count < min_freq:
                continue

            word1, word2 = bigram.split(" ")
            if word1 in qualified_words and word2 in qualified_words:
                # Calculate how surprising this bigram is compared to random co-occurrence
                expected = word_freq[word1] * word_freq[word2] / sum(word_freq.values())
                mechanism_strength = abs(count - expected) / expected

                # Store mechanism strength for this boundary
                if mechanism_strength > mechanism_threshold:
                    self.boundary_mechanisms[bigram] = mechanism_strength

        # Build vocabulary with the most frequent words
        sorted_words = sorted(qualified_words, key=lambda w: word_freq[w], reverse=True)
        vocab_words = sorted_words[:self.vocab_size - len(self.vocab)]

        # Add to vocabulary
        for i, word in enumerate(vocab_words):
            self.vocab[word] = i + len(self.vocab)
            self.id_to_token[i + len(self.vocab)] = word

        self.vocab_size = len(self.vocab)
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Identified {len(self.boundary_mechanisms)} mechanism boundaries")

    def save_vocabulary(self, vocab_file):
        """Save vocabulary and mechanism boundaries to files"""
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        mechanism_file = vocab_file.replace('.json', '_mechanisms.json')
        with open(mechanism_file, 'w', encoding='utf-8') as f:
            json.dump(self.boundary_mechanisms, f, ensure_ascii=False, indent=2)

        print(f"Saved vocabulary to {vocab_file}")
        print(f"Saved mechanism boundaries to {mechanism_file}")

    def _find_mechanism_boundaries(self, text):
        """Identify points in text that represent strong mechanism boundaries"""
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        boundaries = []

        for i in range(len(words) - 1):
            bigram = words[i] + " " + words[i + 1]
            if bigram in self.boundary_mechanisms:
                # Found a mechanism boundary, note the position
                boundaries.append((i, self.boundary_mechanisms[bigram]))

        return boundaries

    def encode(self, text, add_special_tokens=True):
        """Encode text, with heightened awareness of mechanism boundaries"""
        # First identify mechanism boundaries to respect
        mechanism_boundaries = self._find_mechanism_boundaries(text)

        # Tokenize with awareness of mechanism boundaries
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        token_ids = []

        if add_special_tokens:
            token_ids.append(self.vocab[self.bos_token])

        for i, word in enumerate(words):
            # Check if this word should be kept whole as a mechanism boundary
            is_mechanism = any(b[0] == i for b in mechanism_boundaries)

            if word in self.vocab:
                token_ids.append(self.vocab[word])
            else:
                # For unknown words, use the UNK token
                token_ids.append(self.vocab[self.unk_token])

        if add_special_tokens:
            token_ids.append(self.vocab[self.eos_token])

        return token_ids

    def decode(self, token_ids, skip_special_tokens=True):
        """Decode a list of token ids back to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                # Skip special tokens if requested
                if skip_special_tokens and token in [self.pad_token, self.bos_token, self.eos_token]:
                    continue
                tokens.append(token)
            else:
                tokens.append(self.unk_token)

        # Simple space-joining - in a real implementation, would need more sophisticated detokenization
        return " ".join(tokens)


# --------------------------
# 2. MODEL ARCHITECTURE
# --------------------------

class PositionalEncoding(nn.Module):
    """Standard positional encoding"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class MechanismPointAttention(nn.Module):
    """Attention that identifies and leverages mechanism points"""

    def __init__(self, hidden_size, num_heads, dropout_prob=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Projections for Q, K, V
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        # Five Elements transformation projections
        self.wood_proj = nn.Linear(hidden_size, hidden_size)  # Expansion
        self.fire_proj = nn.Linear(hidden_size, hidden_size)  # Acceleration
        self.earth_proj = nn.Linear(hidden_size, hidden_size)  # Stabilization
        self.metal_proj = nn.Linear(hidden_size, hidden_size)  # Refinement
        self.water_proj = nn.Linear(hidden_size, hidden_size)  # Adaptation

        # Element mixing weights (learnable)
        self.element_weights = nn.Parameter(torch.ones(5) / 5)

        # Mechanism strength detector
        self.mechanism_detector = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim // 2),
            nn.GELU(),
            nn.Linear(self.head_dim // 2, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout_prob)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with characteristics of the Five Elements"""
        # Wood: expansion (eigenvalues > 1)
        nn.init.eye_(self.wood_proj.weight)
        self.wood_proj.weight.data *= 1.2

        # Fire: acceleration (amplifies differences)
        nn.init.eye_(self.fire_proj.weight)
        self.fire_proj.weight.data += torch.randn_like(self.fire_proj.weight.data) * 0.1

        # Earth: stabilization (eigenvalues < 1)
        nn.init.eye_(self.earth_proj.weight)
        self.earth_proj.weight.data *= 0.8

        # Metal: refinement (sparse precision)
        nn.init.eye_(self.metal_proj.weight)
        mask = torch.rand_like(self.metal_proj.weight.data) > 0.8
        self.metal_proj.weight.data *= mask.float()

        # Water: adaptation (smoothing)
        nn.init.eye_(self.water_proj.weight)
        for i in range(self.water_proj.weight.size(0)):
            for j in range(max(0, i - 1), min(self.water_proj.weight.size(1), i + 2)):
                self.water_proj.weight.data[i, j] = 1.0 / (abs(i - j) + 1)

    def apply_five_elements(self, x):
        """Apply Five Elements transformations with learned mixing weights"""
        # Apply transformations
        wood_x = self.wood_proj(x)  # Expansion
        fire_x = self.fire_proj(x)  # Acceleration
        earth_x = self.earth_proj(x)  # Stabilization
        metal_x = self.metal_proj(x)  # Refinement
        water_x = self.water_proj(x)  # Adaptation

        # Get normalized element weights
        weights = F.softmax(self.element_weights, dim=0)

        # Combine transformations
        return (weights[0] * wood_x +
                weights[1] * fire_x +
                weights[2] * earth_x +
                weights[3] * metal_x +
                weights[4] * water_x)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length = hidden_states.shape[:2]

        # Apply Five Elements transformation
        hidden_states = self.apply_five_elements(hidden_states)

        # Standard QKV projections
        q = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Get attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Detect mechanism points - where small changes in keys create large attention shifts
        mechanism_strengths = self.mechanism_detector(k)

        # Apply mechanism strength weighting
        weighted_v = v * mechanism_strengths

        # Compute attention output
        context = torch.matmul(attention_probs, weighted_v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        output = self.o_proj(context)

        return output, mechanism_strengths


class DaoTransformerBlock(nn.Module):
    """Transformer block incorporating Dao principles"""

    def __init__(self, hidden_size, num_heads, intermediate_size, dropout_prob=0.1):
        super().__init__()
        self.attention = MechanismPointAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_prob=dropout_prob
        )
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        # Feedforward with five elements inspired gating
        self.ff_gate = nn.Linear(hidden_size, 5)
        self.ff_up = nn.Linear(hidden_size, intermediate_size)
        self.ff_down = nn.Linear(intermediate_size, hidden_size)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        # Layer norm and attention
        attention_output, mechanism_strengths = self.attention(
            self.ln1(hidden_states),
            attention_mask=attention_mask
        )

        # Residual connection
        hidden_states = hidden_states + attention_output

        # Layer norm and feedforward
        ff_input = self.ln2(hidden_states)

        # Five Elements gating for feedforward
        gates = F.softmax(self.ff_gate(ff_input), dim=-1)

        # Element-aware feedforward operation
        ff_output = self.ff_up(ff_input)

        # Apply element-specific activation scaling
        ff_output = self.act(ff_output)

        # Use gates to control flow through different aspects of the network
        ff_output = ff_output * (
                gates[:, :, 0:1] * 1.2 +  # Wood: amplify
                gates[:, :, 1:2] * 1.1 +  # Fire: accelerate
                gates[:, :, 2:3] * 0.9 +  # Earth: stabilize
                gates[:, :, 3:4] * 0.95 +  # Metal: refine
                gates[:, :, 4:5]  # Water: adapt
        )

        ff_output = self.dropout(self.ff_down(ff_output))

        # Residual connection
        hidden_states = hidden_states + ff_output

        return hidden_states, mechanism_strengths


class MechanismFlowLLM(nn.Module):
    """Complete language model incorporating Dao principles"""

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 max_position_embeddings=512,
                 dropout_prob=0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_encoder = PositionalEncoding(hidden_size, max_position_embeddings)
        self.emb_dropout = nn.Dropout(dropout_prob)

        # Transformer blocks
        self.layers = nn.ModuleList([
            DaoTransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                dropout_prob=dropout_prob
            ) for _ in range(num_hidden_layers)
        ])

        # Output layer
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Use shared weights between input embeddings and output layer
        self.lm_head.weight = self.token_embeddings.weight

        # Track mechanism strengths across the network
        self.mechanism_strengths = []

    def forward(self, input_ids, attention_mask=None):
        # Create position IDs based on attention mask
        if attention_mask is not None:
            position_ids = torch.cumsum(attention_mask, dim=1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
        else:
            position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)

        # Get embeddings
        embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_encoder(embeddings)
        hidden_states = embeddings + position_embeddings
        hidden_states = self.emb_dropout(hidden_states)

        # Create extended attention mask for transformer blocks
        if attention_mask is not None:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None

        # Reset mechanism strengths
        self.mechanism_strengths = []

        # Apply transformer blocks
        for i, layer in enumerate(self.layers):
            hidden_states, mech_strengths = layer(
                hidden_states,
                attention_mask=extended_attention_mask
            )
            self.mechanism_strengths.append(mech_strengths)

        # Apply final layer norm
        hidden_states = self.ln_f(hidden_states)

        # Get logits
        logits = self.lm_head(hidden_states)

        return logits

    def get_mechanism_points(self):
        """Return current mechanism points detected throughout the network"""
        if not self.mechanism_strengths:
            return None

        # Aggregate mechanism strengths across layers
        return torch.cat(self.mechanism_strengths, dim=2).mean(dim=2)


# --------------------------
# 3. TRAINING CODE
# --------------------------

class TextDataset(Dataset):
    """Dataset for training the language model"""

    def __init__(self, tokenizer, file_path, block_size=128):
        assert os.path.isfile(file_path), f"Input file {file_path} not found"

        self.tokenizer = tokenizer
        self.block_size = block_size

        # Load and tokenize text
        print(f"Loading and tokenizing {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Encode the full text
        self.examples = []
        token_ids = tokenizer.encode(text)

        # Create examples of length block_size
        for i in range(0, len(token_ids) - block_size, block_size):
            self.examples.append(token_ids[i:i + block_size])

        print(f"Created {len(self.examples)} examples from {file_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)


def collate_fn(batch):
    """Collate function for dataloader, pads sequences to max length in batch"""
    input_ids = torch.stack(batch)
    attention_mask = (input_ids != 0).float()
    return {"input_ids": input_ids, "attention_mask": attention_mask}


class MechanismAwareTrainer:
    """Trainer incorporating mechanism awareness for efficient training"""

    def __init__(self,
                 model,
                 tokenizer,
                 train_dataset,
                 val_dataset=None,
                 batch_size=16,
                 learning_rate=5e-5,
                 weight_decay=0.01,
                 num_epochs=3,
                 gradient_accumulation_steps=1,
                 warmup_steps=0,
                 max_grad_norm=1.0,
                 output_dir="./outputs"):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )
        else:
            self.val_loader = None

        # Set up optimizer (custom mechanism-aware optimizer)
        self.optimizer = self.create_mechanism_optimizer()

        # Set up learning rate scheduler
        self.scheduler = self.create_scheduler()

        # Track training progress
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_logs = []

    def create_mechanism_optimizer(self):
        """Create optimizer with mechanism awareness"""
        # Group parameters for different regularization
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
                "mechanism_aware": True
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "mechanism_aware": False
            }
        ]

        # Simple mechanism-aware optimizer (adapt learning rates for mechanism points)
        class MechanismAdamW(torch.optim.AdamW):
            def step(self, closure=None):
                # Get mechanism points from model
                if hasattr(self.model, 'get_mechanism_points'):
                    mechanism_points = self.model.get_mechanism_points()
                else:
                    mechanism_points = None

                # Use mechanism points to adjust learning rates
                if mechanism_points is not None:
                    for group in self.param_groups:
                        if group.get('mechanism_aware', False):
                            for p in group['params']:
                                if hasattr(p, 'grad') and p.grad is not None:
                                    # Scale learning rate by mechanism strength
                                    lr_scale = 1.0 + 0.5 * torch.mean(mechanism_points).item()
                                    p.grad.data *= lr_scale

                # Normal step
                return super().step(closure)

        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )

    def create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        # Calculate total training steps
        total_steps = len(self.train_loader) // self.gradient_accumulation_steps * self.num_epochs

        # Create scheduler with warmup
        from transformers import get_linear_schedule_with_warmup
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )

    def train(self):
        """Run the training loop"""
        # Set up timing
        start_time = time.time()

        # Track training progress
        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0

            # Training loop
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(next(self.model.parameters()).device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )

                # Calculate loss (shift logits and labels for next token prediction)
                logits = outputs
                labels = batch["input_ids"][:, 1:].contiguous()  # Shift right for next token prediction
                logits = logits[:, :-1, :].contiguous()
                loss = F.cross_entropy(logits.view(-1, self.model.vocab_size), labels.view(-1))

                # Normalize loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

                # Backward pass
                loss.backward()
                epoch_loss += loss.item() * self.gradient_accumulation_steps

                # Update weights if needed
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    # Update weights
                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()

                    # Update global step
                    self.global_step += 1

                # Update progress bar
                progress_bar.set_postfix({"loss": loss.item() * self.gradient_accumulation_steps})

            # Calculate epoch metrics
            epoch_loss /= len(self.train_loader)
            epoch_time = time.time() - epoch_start_time

            # Log progress
            log_entry = {
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "epoch_time": epoch_time
            }

            # Run validation if available
            if self.val_loader:
                val_loss = self.evaluate()
                log_entry["val_loss"] = val_loss

                # Save checkpoint if best validation loss
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(f"checkpoint_best")
                    print(f"New best model with validation loss: {val_loss:.4f}")

            # Save epoch checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}")

            # Add to logs
            self.training_logs.append(log_entry)

            # Print progress
            print(f"Epoch {epoch + 1}/{self.num_epochs} - "
                  f"Train loss: {epoch_loss:.4f} - "
                  f"Time: {epoch_time:.2f}s" +
                  (f" - Val loss: {val_loss:.4f}" if self.val_loader else ""))

        # Save training logs
        self.save_training_logs()

        # Print final training stats
        total_time = time.time() - start_time
        print(f"Training complete! Total time: {total_time:.2f}s")

    def evaluate(self):
        """Evaluate the model on the validation set"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Move batch to device
                batch = {k: v.to(next(self.model.parameters()).device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )

                # Calculate loss
                logits = outputs
                labels = batch["input_ids"][:, 1:].contiguous()
                logits = logits[:, :-1, :].contiguous()
                loss = F.cross_entropy(logits.view(-1, self.model.vocab_size), labels.view(-1))

                total_loss += loss.item()

        # Calculate average loss
        avg_loss = total_loss / len(self.val_loader)

        # Switch back to training mode
        self.model.train()

        return avg_loss

    def save_checkpoint(self, checkpoint_name):
        """Save model and optimizer state"""
        checkpoint_dir = os.path.join(self.output_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(checkpoint_dir, "model.pt")
        torch.save(self.model.state_dict(), model_path)

        # Save optimizer
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        torch.save(self.optimizer.state_dict(), optimizer_path)

        # Save scheduler
        scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
        torch.save(self.scheduler.state_dict(), scheduler_path)

        # Save training progress
        progress_path = os.path.join(checkpoint_dir, "progress.json")
        with open(progress_path, 'w') as f:
            json.dump({
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
                "logs": self.training_logs
            }, f)

        print(f"Checkpoint saved to {checkpoint_dir}")

    def save_training_logs(self):
        """Save training logs to file"""
        logs_path = os.path.join(self.output_dir, "training_logs.json")
        with open(logs_path, 'w') as f:
            json.dump(self.training_logs, f, indent=2)

        print(f"Training logs saved to {logs_path}")

    def load_checkpoint(self, checkpoint_dir):
        """Load model and training state from checkpoint"""
        # Load model
        model_path = os.path.join(checkpoint_dir, "model.pt")
        self.model.load_state_dict(torch.load(model_path))

        # Load optimizer
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        self.optimizer.load_state_dict(torch.load(optimizer_path))

        # Load scheduler
        scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
        self.scheduler.load_state_dict(torch.load(scheduler_path))

        # Load training progress
        progress_path = os.path.join(checkpoint_dir, "progress.json")
        with open(progress_path, 'r') as f:
            progress = json.load(f)
            self.global_step = progress["global_step"]
            self.best_val_loss = progress["best_val_loss"]
            self.training_logs = progress["logs"]

        print(f"Checkpoint loaded from {checkpoint_dir}")


# --------------------------
# 4. TEXT GENERATION
# --------------------------

class MechanismAwareGenerator:
    """Text generator with mechanism awareness for more natural generation"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def generate(self,
                 prompt,
                 max_length=100,
                 temperature=1.0,
                 top_k=50,
                 top_p=0.9,
                 repetition_penalty=1.2,
                 mechanism_threshold=0.5):
        """Generate text with mechanism-aware decoding"""
        # Tokenize prompt
        input_ids = torch.tensor(self.tokenizer.encode(prompt), device=self.device).unsqueeze(0)

        # Keep track of end token
        eos_token_id = self.tokenizer.vocab[self.tokenizer.eos_token]

        # Set model to evaluation mode
        self.model.eval()

        # Track generated sequence
        generated_ids = input_ids.clone()

        # Generation loop
        with torch.no_grad():
            for _ in range(max_length):
                # Create attention mask
                attention_mask = torch.ones_like(generated_ids)

                # Forward pass
                outputs = self.model(
                    input_ids=generated_ids,
                    attention_mask=attention_mask
                )

                # Get predictions for the next token
                next_token_logits = outputs[:, -1, :]

                # Apply temperature
                next_token_logits = next_token_logits / temperature

                # Apply repetition penalty
                for i in range(generated_ids.shape[0]):
                    for token_id in set(generated_ids[i].tolist()):
                        next_token_logits[i, token_id] /= repetition_penalty

                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')

                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p

                    # Shift the indices to the right to keep the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    # Apply the mask
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = -float('Inf')

                # Get mechanism points (for influenced sampling)
                mechanism_points = self.model.get_mechanism_points()

                # If we have strong mechanism points, bias generation
                if mechanism_points is not None:
                    last_token_mechanism = mechanism_points[0, -1, 0].item()

                    # If we're at a strong mechanism point, be more deterministic
                    if last_token_mechanism > mechanism_threshold:
                        # Increase sharpness of the distribution at key decision points
                        next_token_logits = next_token_logits * (1.0 + last_token_mechanism)

                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)

                # Check if any EOS token was generated
                if next_token.item() == eos_token_id:
                    break

        # Decode generated sequence
        generated_text = self.tokenizer.decode(generated_ids[0].tolist())

        return generated_text


# --------------------------
# 5. FULL USAGE EXAMPLE
# --------------------------

def main():
    """Main function demonstrating tokenizer, model, training, and generation"""
    # Configuration
    vocab_size = 10000
    hidden_size = 768
    num_hidden_layers = 6
    num_attention_heads = 12
    intermediate_size = 3072
    max_position_embeddings = 512
    batch_size = 16
    learning_rate = 5e-5
    num_epochs = 3

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")


    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MechanismFlowLLM(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings
    ).to(device)

    # Create datasets
    train_dataset = TextDataset(tokenizer, "train.txt", block_size=128)
    val_dataset = TextDataset(tokenizer, "val.txt", block_size=128)

    # Create trainer
    trainer = MechanismAwareTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        output_dir="./outputs"
    )

    # Train model
    trainer.train()

    # Create generator
    generator = MechanismAwareGenerator(model, tokenizer)

    # Generate text
    prompt = "Once upon a time"
    generated_text = generator.generate(
        prompt=prompt,
        max_length=100,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        mechanism_threshold=0.5
    )

    print("Generated text:")
    print(generated_text)


if __name__ == "__main__":
    main()