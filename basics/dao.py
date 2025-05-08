# Mechanism-Aware Transformer Phase 2: Forged with Auxiliary Loss

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math

# --- Model Components ---

class MechanismAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.mechanism_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1,2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1,2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1,2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        mechanism_strengths = self.mechanism_proj(x).squeeze(-1)
        mech_weights = mechanism_strengths.unsqueeze(1).unsqueeze(1)
        scores = scores * (1 + mech_weights)

        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)

        context = context.transpose(1,2).contiguous().view(B, T, C)
        output = self.out_proj(context)

        return output, mechanism_strengths

class FiveElementsTransformationLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.wood_proj = nn.Linear(d_model, d_model)
        self.fire_proj = nn.Linear(d_model, d_model)
        self.earth_proj = nn.Linear(d_model, d_model)
        self.metal_proj = nn.Linear(d_model, d_model)
        self.water_proj = nn.Linear(d_model, d_model)

        self.element_weights = nn.Parameter(torch.ones(5)/5)

    def forward(self, x):
        wood = self.wood_proj(x)
        fire = self.fire_proj(x)
        earth = self.earth_proj(x)
        metal = self.metal_proj(x)
        water = self.water_proj(x)

        weights = torch.softmax(self.element_weights, dim=0)
        output = (weights[0]*wood + weights[1]*fire + weights[2]*earth + weights[3]*metal + weights[4]*water)
        return output, weights

class PositionTimingLayer(nn.Module):
    def __init__(self, d_model, max_seq_len=1024):
        super().__init__()
        self.position_embedding = nn.Parameter(torch.zeros(max_seq_len, d_model//2))
        self.timing_embedding = nn.Parameter(torch.zeros(max_seq_len, d_model//2))
        self.integration_proj = nn.Linear(d_model, d_model)
        self._init_embeddings()

    def _init_embeddings(self):
        pos = torch.arange(0, self.position_embedding.size(0)).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.position_embedding.size(1), 2) * -(math.log(10000.0) / self.position_embedding.size(1)))
        self.position_embedding.data[:, 0::2] = torch.sin(pos * div_term)
        self.position_embedding.data[:, 1::2] = torch.cos(pos * div_term)

        tim = torch.arange(0, self.timing_embedding.size(0)).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.timing_embedding.size(1), 2) * -(math.log(5000.0) / self.timing_embedding.size(1)))
        self.timing_embedding.data[:, 0::2] = torch.sin(tim * div_term)
        self.timing_embedding.data[:, 1::2] = torch.cos(tim * div_term)

    def forward(self, x):
        B, T, _ = x.size()
        pos_emb = self.position_embedding[:T].unsqueeze(0).expand(B, -1, -1)
        time_emb = self.timing_embedding[:T].unsqueeze(0).expand(B, -1, -1)
        fused_emb = torch.cat([pos_emb, time_emb], dim=-1)
        out = x + self.integration_proj(fused_emb)
        return out

class MechanismTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mech_attn = MechanismAttention(d_model, n_heads)
        self.five_elements = FiveElementsTransformationLayer(d_model)
        self.pos_time = PositionTimingLayer(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.pos_time(x)
        attn_out, mechanism_strengths = self.mech_attn(self.norm1(x), mask)
        x = x + self.dropout(attn_out)

        elem_out, elem_weights = self.five_elements(self.norm2(x))
        x = x + self.dropout(elem_out)

        ff_out = self.ffn(self.norm3(x))
        x = x + self.dropout(ff_out)

        return x, mechanism_strengths, elem_weights

class MechanismTransformer(nn.Module):
    def __init__(self, d_model=128, n_heads=8, n_layers=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([
            MechanismTransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        x = self.embed(x)
        mech_strengths_list = []
        elem_weights_list = []
        for block in self.blocks:
            x, mech_strengths, elem_weights = block(x, mask)
            mech_strengths_list.append(mech_strengths)
            elem_weights_list.append(elem_weights)
        x = self.final_norm(x)
        out = self.classifier(x.mean(dim=1))
        return out, mech_strengths_list, elem_weights_list

# --- Synthetic Dataset ---

def generate_dataset(batch_size, seq_len):
    X = torch.rand(batch_size, seq_len, 1)
    triggers = torch.zeros(batch_size, seq_len)
    labels = torch.zeros(batch_size, 1)
    for i in range(batch_size):
        trigger_pos = torch.randint(0, seq_len, (1,))
        X[i, trigger_pos, 0] += 1.0  # Insert a strong trigger
        triggers[i, trigger_pos] = 1.0
        labels[i] = 1.0 if X[i, trigger_pos, 0] > 1.5 else 0.0
    return X, triggers, labels

# --- Training ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MechanismTransformer().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()
auxiliary_loss_fn = nn.BCELoss()
lambda_mech = 0.1

train_losses = []

for epoch in range(15):
    model.train()
    X_batch, mech_targets, labels = generate_dataset(batch_size=32, seq_len=64)
    X_batch, mech_targets, labels = X_batch.to(device), mech_targets.to(device), labels.to(device)

    optimizer.zero_grad()
    outputs, mech_strengths_list, elem_weights_list = model(X_batch)
    main_loss = criterion(outputs, labels)
    mech_pred = mech_strengths_list[-1]
    mech_loss = auxiliary_loss_fn(mech_pred, mech_targets)
    loss = main_loss + lambda_mech * mech_loss

    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())
    print(f"Epoch {epoch+1}/15, Loss: {loss.item():.4f}")

# --- Visualization ---

plt.plot(train_losses, marker='o')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

X_test, mech_targets_test, labels_test = generate_dataset(batch_size=1, seq_len=64)
X_test = X_test.to(device)
model.eval()
with torch.no_grad():
    outputs, mech_strengths_list, elem_weights_list = model(X_test)

mech_strengths = mech_strengths_list[-1][0].cpu().numpy()
plt.plot(mech_strengths, label='Mechanism Strength')
plt.plot(mech_targets_test[0].cpu().numpy(), 'r--', label='Ground Truth Trigger')
plt.legend()
plt.title('Mechanism Strength vs Ground Truth')
plt.show()

elements = torch.stack(elem_weights_list, dim=0).cpu().numpy()
elements = elements[-1]
for i, name in enumerate(['Wood', 'Fire', 'Earth', 'Metal', 'Water']):
    plt.plot(elements[:, i], label=name)
plt.title('Five Element Strengths')
plt.legend()
plt.show()
