import torch
import torch.nn as nn
import math


class MechanismAttention(nn.Module):
    """Attention mechanism that identifies and focuses on mechanism points in sequences"""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Query, Key, Value projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Mechanism strength projection
        self.mechanism_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # Project queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Calculate mechanism strengths for each position
        mechanism_strengths = self.mechanism_proj(x).squeeze(-1)  # [batch_size, seq_len]

        # Modify attention with mechanism awareness
        mechanism_weights = mechanism_strengths.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
        scores = scores * (1 + mechanism_weights)  # Amplify attention to mechanism points

        # Apply softmax and get weighted values
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)

        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(context)

        return output, mechanism_strengths


class FiveElementsTransformationLayer(nn.Module):
    """Implements the Five Elements transformations as neural network operations"""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # Five transformation projections
        self.wood_proj = nn.Linear(d_model, d_model)  # Expansion
        self.fire_proj = nn.Linear(d_model, d_model)  # Acceleration
        self.earth_proj = nn.Linear(d_model, d_model)  # Stabilization
        self.metal_proj = nn.Linear(d_model, d_model)  # Refinement
        self.water_proj = nn.Linear(d_model, d_model)  # Adaptation

        # Element mixing weights (learnable)
        self.element_weights = nn.Parameter(torch.ones(5) / 5)

        # Initialize with appropriate characteristics
        self._initialize_transformations()

    def _initialize_transformations(self):
        # Wood: expansion matrix (eigenvalues > 1)
        nn.init.eye_(self.wood_proj.weight)
        self.wood_proj.weight.data *= 1.2

        # Fire: acceleration matrix (amplifies differences)
        nn.init.eye_(self.fire_proj.weight)
        self.fire_proj.weight.data += torch.randn_like(self.fire_proj.weight.data) * 0.1

        # Earth: stabilization matrix (eigenvalues < 1)
        nn.init.eye_(self.earth_proj.weight)
        self.earth_proj.weight.data *= 0.8

        # Metal: refinement matrix (sparse precision)
        nn.init.eye_(self.metal_proj.weight)
        mask = torch.rand_like(self.metal_proj.weight.data) > 0.7
        self.metal_proj.weight.data *= mask.float()

        # Water: adaptation matrix (smoothing)
        nn.init.eye_(self.water_proj.weight)
        for i in range(self.water_proj.weight.size(0)):
            for j in range(max(0, i - 1), min(self.water_proj.weight.size(1), i + 2)):
                self.water_proj.weight.data[i, j] = 1.0 / (abs(i - j) + 1)

    def forward(self, x, element_weights=None):
        # Apply each transformation
        wood_x = self.wood_proj(x)
        fire_x = self.fire_proj(x)
        earth_x = self.earth_proj(x)
        metal_x = self.metal_proj(x)
        water_x = self.water_proj(x)

        # Use provided weights or learned weights
        weights = torch.softmax(element_weights if element_weights is not None else self.element_weights, dim=0)

        # Combine transformations
        output = (
                weights[0] * wood_x +
                weights[1] * fire_x +
                weights[2] * earth_x +
                weights[3] * metal_x +
                weights[4] * water_x
        )

        return output, weights


class PositionTimingLayer(nn.Module):
    """Implements position-timing unity for sequence modeling"""

    def __init__(self, d_model, max_seq_len=1024):
        super().__init__()
        self.d_model = d_model

        # Position embedding
        self.position_embedding = nn.Parameter(torch.zeros(max_seq_len, d_model // 2))

        # Timing embedding (for capturing temporal patterns)
        self.timing_embedding = nn.Parameter(torch.zeros(max_seq_len, d_model // 2))

        # Position-timing integration
        self.integration_proj = nn.Linear(d_model, d_model)

        # Initialize embeddings
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        # Initialize position embeddings with sinusoidal pattern
        position = torch.arange(0, self.position_embedding.size(0)).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.position_embedding.size(1), 2) *
                             -(math.log(10000.0) / self.position_embedding.size(1)))
        self.position_embedding.data[:, 0::2] = torch.sin(position * div_term)
        self.position_embedding.data[:, 1::2] = torch.cos(position * div_term)

        # Initialize timing embeddings with different frequency
        timing = torch.arange(0, self.timing_embedding.size(0)).unsqueeze(1)
        div_term_timing = torch.exp(torch.arange(0, self.timing_embedding.size(1), 2) *
                                    -(math.log(5000.0) / self.timing_embedding.size(1)))
        self.timing_embedding.data[:, 0::2] = torch.sin(timing * div_term_timing)
        self.timing_embedding.data[:, 1::2] = torch.cos(timing * div_term_timing)

    def forward(self, x, positions=None, timings=None):
        batch_size, seq_len, _ = x.size()

        # Get position indices
        if positions is None:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        # Get timing indices (default to same as positions if not provided)
        if timings is None:
            timings = positions

        # Get position and timing embeddings
        pos_emb = self.position_embedding[positions]
        time_emb = self.timing_embedding[timings]

        # Concatenate position and timing embeddings
        pos_time_emb = torch.cat([pos_emb, time_emb], dim=-1)

        # Integrate with input
        output = x + self.integration_proj(pos_time_emb)

        return output


class MechanismTransformerBlock(nn.Module):
    """Transformer block with mechanism awareness and five elements transformation"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        # Mechanism-aware attention
        self.mechanism_attn = MechanismAttention(d_model, n_heads)

        # Five elements transformation
        self.five_elements = FiveElementsTransformationLayer(d_model)

        # Position-timing layer
        self.pos_time = PositionTimingLayer(d_model)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Apply position-timing integration
        x = self.pos_time(x)

        # Apply mechanism attention
        attn_output, mechanism_strengths = self.mechanism_attn(self.norm1(x), mask)
        x = x + self.dropout(attn_output)

        # Apply five elements transformation
        five_elem_output, element_weights = self.five_elements(self.norm2(x))
        x = x + self.dropout(five_elem_output)

        # Apply feed-forward network
        ff_output = self.ff(self.norm3(x))
        x = x + self.dropout(ff_output)

        return x, mechanism_strengths, element_weights


class MechanismTransformer(nn.Module):
    """Complete transformer architecture with mechanism awareness"""

    def __init__(self, d_model=512, n_heads=8, n_layers=6, d_ff=2048, dropout=0.1, max_seq_len=1024):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Linear(d_model, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            MechanismTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Apply embedding
        x = self.embedding(x)

        # Store mechanism strengths and element weights from each layer
        all_mechanism_strengths = []
        all_element_weights = []

        # Apply transformer blocks
        for block in self.blocks:
            x, mechanism_strengths, element_weights = block(x, mask)
            all_mechanism_strengths.append(mechanism_strengths)
            all_element_weights.append(element_weights)

        # Apply final normalization
        x = self.norm(x)

        return x, all_mechanism_strengths, all_element_weights

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random

# Assume MechanismTransformer is already defined and imported

# --- Synthetic Dataset Creation ---

class TriggerDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=1000, seq_len=20, vocab_size=50, trigger_token=49):
        self.samples = []
        for _ in range(num_samples):
            sequence = [random.randint(0, vocab_size - 2) for _ in range(seq_len)]
            if random.random() > 0.5:
                insert_pos = random.randint(0, seq_len - 1)
                sequence[insert_pos] = trigger_token
                label = 1
            else:
                label = 0
            self.samples.append((torch.tensor(sequence), label))
        self.vocab_size = vocab_size
        self.trigger_token = trigger_token

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        return seq, torch.tensor(label, dtype=torch.float)

# --- Embedding Layer for tokens ---

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)

# --- Full Model ---

class TriggerMechanismModel(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2, d_ff=128):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.transformer = MechanismTransformer(d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.token_embedding(x)
        x, mechanism_strengths, _ = self.transformer(x)
        # Pool over sequence (mean pooling)
        x = x.mean(dim=1)
        out = self.fc(x).squeeze(-1)
        return out, mechanism_strengths

# --- Training Function ---

def train_model(model, train_loader, epochs=10, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for sequences, labels in train_loader:
            sequences, labels = sequences.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs, _ = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

# --- Visualization Function ---

def visualize_mechanism_strengths(model, dataset, num_examples=5):
    model.eval()
    fig, axs = plt.subplots(num_examples, 1, figsize=(12, 2 * num_examples))

    for i in range(num_examples):
        sequence, label = dataset[i]
        with torch.no_grad():
            seq_input = sequence.unsqueeze(0).cuda()
            _, mechanism_strengths = model(seq_input)
            mechanism_strengths = mechanism_strengths[-1].squeeze(0).cpu().numpy()
        axs[i].bar(range(len(sequence)), mechanism_strengths)
        axs[i].set_title(f"Example {i+1} - Label: {label.item()} - Sequence: {sequence.tolist()}")
        axs[i].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

# --- Running Everything ---

# Hyperparameters
vocab_size = 50
seq_len = 20
batch_size = 32

# Datasets and Loaders
train_dataset = TriggerDataset(num_samples=1000, seq_len=seq_len, vocab_size=vocab_size)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model
model = TriggerMechanismModel(vocab_size=vocab_size).cuda()

# Train
train_model(model, train_loader, epochs=10)

# Visualize
visualize_mechanism_strengths(model, train_dataset)



