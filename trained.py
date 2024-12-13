import torch
import torch.nn as nn
import torch.nn.functional as F

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define hyperparameters
hyperparams = {
    "seq_len": 256,
    "embedding_dim": 384,
    "num_heads": 6,
    "num_layers": 6,
    "dropout": 0.2
}

# Load dataset (for character encoding/decoding)
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Character encoding
unique_chars = sorted(list(set(text)))
vocab_size = len(unique_chars)
char_to_idx = {ch: i for i, ch in enumerate(unique_chars)}
idx_to_char = {i: ch for i, ch in enumerate(unique_chars)}
encode = lambda s: [char_to_idx[c] for c in s]
decode = lambda l: ''.join([idx_to_char[i] for i in l])

# Define model components
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

# Load pre-trained model
model = LanguageModel().to(device)
model.load_state_dict(torch.load("model.pt"))
model.eval()

# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Initial token
generated_text = decode(model.generate(context, max_new_tokens=2000)[0].tolist())
print(generated_text)
