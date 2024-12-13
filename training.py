import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define hyperparameters
hyperparams = {
    "batch_size": 64,
    "seq_len": 256,
    "max_iters": 5000,
    "eval_interval": 500,
    "lr": 3e-4,
    "embedding_dim": 384,
    "num_heads": 6,
    "num_layers": 6,
    "dropout": 0.2,
    "eval_iters": 200
}

torch.manual_seed(0)

# Load dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Character encoding
unique_chars = sorted(list(set(text)))
vocab_size = len(unique_chars)
char_to_idx = {ch: i for i, ch in enumerate(unique_chars)}
idx_to_char = {i: ch for i, ch in enumerate(unique_chars)}
encode = lambda s: [char_to_idx[c] for c in s]  # Encode string to list of ints
decode = lambda l: ''.join([idx_to_char[i] for i in l])  # Decode list of ints to string

# Split data
data = torch.tensor(encode(text), dtype=torch.long)
split_idx = int(0.9 * len(data))
train_data, val_data = data[:split_idx], data[split_idx:]

# Batch generator
def generate_batch(split, params):
    dataset = train_data if split == 'train' else val_data
    indices = torch.randint(len(dataset) - params["seq_len"], (params["batch_size"],))
    x = torch.stack([dataset[i:i + params["seq_len"]] for i in indices])
    y = torch.stack([dataset[i + 1:i + params["seq_len"] + 1] for i in indices])
    return x.to(device), y.to(device)

@torch.no_grad()
def compute_loss():
    results = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(hyperparams["eval_iters"])
        for i in range(hyperparams["eval_iters"]):
            x, y = generate_batch(split, hyperparams)
            _, loss = model(x, y)
            losses[i] = loss.item()
        results[split] = losses.mean().item()
    model.train()
    return results

class SelfAttentionHead(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.key_layer = nn.Linear(hyperparams["embedding_dim"], head_dim, bias=False)
        self.query_layer = nn.Linear(hyperparams["embedding_dim"], head_dim, bias=False)
        self.value_layer = nn.Linear(hyperparams["embedding_dim"], head_dim, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(hyperparams["seq_len"], hyperparams["seq_len"])))
        self.dropout = nn.Dropout(hyperparams["dropout"])

    def forward(self, x):
        k = self.key_layer(x)
        q = self.query_layer(x)
        v = self.value_layer(x)
        attention_weights = q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5)
        attention_weights = attention_weights.masked_fill(self.mask[:x.size(1), :x.size(1)] == 0, float('-inf'))
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)
        return attention_weights @ v

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        head_dim = hyperparams["embedding_dim"] // num_heads
        self.attention_heads = nn.ModuleList([SelfAttentionHead(head_dim) for _ in range(num_heads)])
        self.output_projection = nn.Linear(hyperparams["embedding_dim"], hyperparams["embedding_dim"])
        self.dropout = nn.Dropout(hyperparams["dropout"])

    def forward(self, x):
        concatenated = torch.cat([h(x) for h in self.attention_heads], dim=-1)
        return self.dropout(self.output_projection(concatenated))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hyperparams["embedding_dim"], 4 * hyperparams["embedding_dim"]),
            nn.ReLU(),
            nn.Linear(4 * hyperparams["embedding_dim"], hyperparams["embedding_dim"]),
            nn.Dropout(hyperparams["dropout"])
        )

    def forward(self, x):
        return self.network(x)

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_layer = MultiHeadSelfAttention(hyperparams["num_heads"])
        self.feed_forward_layer = FeedForward()
        self.layer_norm1 = nn.LayerNorm(hyperparams["embedding_dim"])
        self.layer_norm2 = nn.LayerNorm(hyperparams["embedding_dim"])

    def forward(self, x):
        x = x + self.attention_layer(self.layer_norm1(x))
        x = x + self.feed_forward_layer(self.layer_norm2(x))
        return x

class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hyperparams["embedding_dim"])
        self.position_embeddings = nn.Embedding(hyperparams["seq_len"], hyperparams["embedding_dim"])
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(hyperparams["num_layers"])])
        self.final_layer_norm = nn.LayerNorm(hyperparams["embedding_dim"])
        self.output_layer = nn.Linear(hyperparams["embedding_dim"], vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.shape
        tok_emb = self.token_embeddings(idx)
        pos_emb = self.position_embeddings(torch.arange(seq_len, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.final_layer_norm(x)
        logits = self.output_layer(x)

        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -hyperparams["seq_len"]:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
# Utility functions for saving, loading, and training enhancements
def save_model(model, path="model.pt"):
    torch.save(model.state_dict(), path)

def load_model(model, path="model.pt"):
    model.load_state_dict(torch.load(path))

def learning_rate_decay(optimizer, max_steps, step, warmup_steps=500):
    """
    Adjust the learning rate using cosine annealing with a warm-up phase.

    Parameters:
    - optimizer: PyTorch optimizer instance.
    - max_steps: Total training steps.
    - step: Current step.
    - warmup_steps: Number of initial steps with linear warm-up.
    """
    for param_group in optimizer.param_groups:
        initial_lr = param_group.get('initial_lr', param_group['lr'])
        if step < warmup_steps:
            new_lr = initial_lr #* (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            new_lr = initial_lr * 0.5 * (1 + math.cos(math.pi * progress))
        param_group['lr'] = new_lr

def gradient_clipping(model, max_norm=1.0):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)

def evaluate_perplexity(loss):
    if not isinstance(loss, torch.Tensor):
        loss = torch.tensor(loss)
    return torch.exp(loss).item()

# Initialize model and optimizer
model = LanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams["lr"])

# Training loop
for step in range(hyperparams["max_iters"]):
    if step % hyperparams["eval_interval"] == 0:
        losses = compute_loss()
        train_loss = losses['train']
        val_loss = losses['val']
        train_perplexity = evaluate_perplexity(train_loss)
        val_perplexity = evaluate_perplexity(val_loss)
        print(f"Step {step}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Step {step}: Train Perplexity: {train_perplexity:.4f}, Val Perplexity: {val_perplexity:.4f}")
        if step < 1500:
            print(f"Step {step}: Learning Rate Warming Up (Step {step}/1000).")

    x_batch, y_batch = generate_batch('train', hyperparams)
    logits, loss = model(x_batch, y_batch)
    optimizer.zero_grad()
    loss.backward()
    gradient_clipping(model, max_norm=1.0)
    optimizer.step()
    learning_rate_decay(optimizer, hyperparams["max_iters"], step, warmup_steps=1000)


# Save final model
save_model(model, "model.pt")

# Load model
load_model(model, "model.pt")
model.to(device)

# Generate text from the trained model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))









