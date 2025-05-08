import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from mechanism_transformer import MechanismTransformer


# Assume the MechanismTransformer and submodules are already imported

# 1. Create a synthetic sequence task
def generate_synthetic_data(batch_size, seq_len, input_dim, device):
    """Simple random input and 'trigger' detection"""
    x = torch.randn(batch_size, seq_len, input_dim, device=device)

    # Create synthetic 'mechanism points' — rare trigger events
    trigger_indices = torch.randint(0, seq_len, (batch_size,), device=device)
    y = torch.zeros(batch_size, seq_len, device=device)
    for i in range(batch_size):
        y[i, trigger_indices[i]] = 1.0  # Mark trigger at random position

    return x, y


# 2. Define the test configuration
config = {
    'd_input': 16,
    'd_model': 128,
    'n_heads': 4,
    'n_layers': 3,
    'd_ff': 512,
    'dropout': 0.1,
    'max_seq_len': 128,
    'sharpening_factor': 3.0,
    'seq_len': 64,
    'batch_size': 64,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'epochs': 15,
    'learning_rate': 3e-4,
}

# 3. Instantiate the model
model = MechanismTransformer(
    d_input=config['d_input'],
    d_model=config['d_model'],
    n_heads=config['n_heads'],
    n_layers=config['n_layers'],
    d_ff=config['d_ff'],
    dropout=config['dropout'],
    max_seq_len=config['max_seq_len'],
    sharpening_factor=config['sharpening_factor']
).to(config['device'])

# 4. Define loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

# 5. Training Loop
loss_history = []

for epoch in range(1, config['epochs'] + 1):
    model.train()
    x_batch, y_batch = generate_synthetic_data(config['batch_size'], config['seq_len'], config['d_input'],
                                               config['device'])

    optimizer.zero_grad()
    outputs, all_mech_strengths, all_elem_weights = model(x_batch)

    # Use just the first block's outputs for simplicity
    predictions = outputs.squeeze(-1).mean(dim=-1)  # [batch_size, seq_len]

    loss = criterion(predictions, y_batch)
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
    print(f"Epoch {epoch}/{config['epochs']}, Loss: {loss.item():.4f}")

# 6. Visualization Stage

# Plot loss
plt.figure(figsize=(8, 5))
plt.plot(loss_history, marker='o')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# Visualize mechanism strengths on a new batch
model.eval()
with torch.no_grad():
    x_test, y_test = generate_synthetic_data(1, config['seq_len'], config['d_input'], config['device'])
    outputs, mech_strengths_list, elem_weights_list = model(x_test)

    # Take mechanism strengths from the last block
    mech_strengths = mech_strengths_list[-1][0].cpu().numpy()  # [seq_len]
    elem_weights = elem_weights_list[-1][0].cpu().numpy()  # [seq_len, 5]

# Plot mechanism strengths
plt.figure(figsize=(10, 4))
plt.plot(mech_strengths, marker='o')
plt.title('Mechanism Strengths Across Sequence')
plt.xlabel('Token Position')
plt.ylabel('Mechanism Strength')
plt.grid(True)
plt.show()

# Plot element dominance per token
plt.figure(figsize=(10, 6))
elements = ['Wood', 'Fire', 'Earth', 'Metal', 'Water']
for i in range(5):
    plt.plot(elem_weights[:, i], label=elements[i])
plt.title('Five Element Strengths Across Sequence')
plt.xlabel('Token Position')
plt.ylabel('Element Weight')
plt.legend()
plt.grid(True)
plt.show()

# Overlay mechanism strength and ground truth
plt.figure(figsize=(10, 5))
plt.plot(mech_strengths, label='Mechanism Strength', color='blue')
plt.plot(y_test.squeeze(0).cpu().numpy(), label='Ground Truth Trigger', color='red', linestyle='--')
plt.title('Mechanism Strength vs Ground Truth Trigger')
plt.xlabel('Token Position')
plt.ylabel('Strength / Trigger')
plt.legend()
plt.grid(True)
plt.show()
