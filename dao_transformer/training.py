import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Tuple

# --- Example Dataset (replace with your own) ---
class TextDataset(Dataset):
    def __init__(self, tokenized_texts, seq_length: int):
        self.inputs = []
        self.targets = []
        for ids in tokenized_texts:
            for i in range(len(ids) - seq_length):
                self.inputs.append(torch.tensor(ids[i: i + seq_length]))
                self.targets.append(torch.tensor(ids[i + 1: i + 1 + seq_length]))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# --- Training Loop ---
def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int = 10,
    clip_norm: Optional[float] = 1.0
) -> None:
    model.to(device)
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for step, (x_batch, y_batch) in enumerate(train_loader, 1):
            x = x_batch.to(device)
            y = y_batch.to(device)

            # Optionally calculate mechanism strengths based on current batch
            if hasattr(optimizer, 'calculate_mechanism_strengths'):
                optimizer.calculate_mechanism_strengths(
                    lambda inp, tgt: criterion(model(inp)[0].view(-1, model.hidden_size), tgt.view(-1)),
                    x,
                    y
                )

            optimizer.zero_grad()
            logits, *_ = model(x)  # assumes model returns (logits, ...)
            # flatten: (batch * seq_len, vocab_size)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = y.view(-1)

            loss = criterion(logits_flat, targets_flat)
            loss.backward()

            # gradient clipping
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}")

# --- Text Generation Loop ---
@torch.no_grad()
def generate(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_length: int = 50,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.9,
    device: torch.device = torch.device('cpu')
) -> str:
    model.to(device)
    model.eval()

    # Tokenize input prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(device)
    past: Optional[Tuple[torch.Tensor]] = None

    for _ in range(max_length):
        outputs = model(
            input_ids,
            past_key_values=past,
            use_cache=True
        )
        logits, past, _ = outputs if len(outputs) == 3 else (outputs[0], outputs[1], None)
        next_token_logits = logits[:, -1, :] / max(temperature, 1e-8)

        # Top-k truncation
        if top_k > 0:
            values, _ = torch.topk(next_token_logits, top_k)
            min_values = values[:, -1].unsqueeze(1)
            next_token_logits = torch.where(
                next_token_logits < min_values,
                torch.full_like(next_token_logits, -float('Inf')),
                next_token_logits
            )

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumulative_probs > top_p
            # shift mask right to keep first token above threshold
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_logits[mask] = -float('Inf')
            next_token_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # append and continue
        input_ids = torch.cat([input_ids, next_token], dim=-1)

    output_ids = input_ids[0].tolist()
    return tokenizer.decode(output_ids)

# --- Example Usage ---
if __name__ == "__main__":
    from transformers import GPT2Tokenizer

    # Prepare tokenizer and data
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # Example raw texts
    raw_texts = ["Hello world! This is a sample.", "Another example sentence for training."]
    tokenized = [tokenizer.encode(t) for t in raw_texts]
    dataset = TextDataset(tokenized, seq_length=32)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Instantiate your model (pseudo-code)
    model = YourDaoTransformerModel(vocab_size=tokenizer.vocab_size)
    optimizer = MechanismOptimizer(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    # Train
    train(model, dataloader, optimizer, criterion, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), num_epochs=5)

    # Generate text
    prompt = "Once upon a time"
    generated = generate(model, tokenizer, prompt, max_length=50, temperature=0.8, top_k=50, top_p=0.9)
    print("Generated:", generated)
