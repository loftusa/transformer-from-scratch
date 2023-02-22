#%%
import datasets
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import softmax

# hyperparameters
batch_size = 4  # B
block_size = 25  # T
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 32
n_head = 6
n_layer = 6
dropout = 0.2

# -----------------

torch.manual_seed(42)

# pull in shakespeare data
d = datasets.load_dataset(path="tiny_shakespeare", name="shakespeare")
train = d["train"]["text"][0]
test = d["test"]["text"][0]
val = d["validation"]["text"][0]
text = train + test + val

# create tokenizer
chars = sorted(list(set(text)))
vocab_size = len(chars)

# token encoder and decoder dictionaries
toke = {t: i for i, t in enumerate(chars)}
tokd = {i: t for i, t in enumerate(chars)}
encode = lambda sentence: [toke[t] for t in sentence]
decode = lambda integers: "".join([tokd[i] for i in integers])


# train and test splits
data = torch.tensor(encode(text), dtype=torch.long)


def get_batch(data, batch_size=4, block_size=8):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.long)

    max_idx = len(data) - block_size
    ix = torch.randint(max_idx, size=(batch_size,))
    X = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    X, y = X.to(device), y.to(device)
    return X, y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in [train, val]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class AttentionHead(nn.Module):
    """
    B --> batch size
    T --> number of tokens in each block
    D --> embedding dimension (equal to the vocab size, C?)

    karpathy:
    - block_size = n_embd = T
    - n_embd = head_size = C
    """

    def __init__(self, head_size, mask=True):
        super().__init__()

        # matrices and attributes
        self.K = nn.Linear(n_embd, head_size, bias=False)
        self.Q = nn.Linear(n_embd, head_size, bias=False)
        self.V = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        # hyperparameters
        self.mask = mask

    def _mask(self, xx):
        xx = torch.where(self.tril == 0, -1 * torch.inf, xx)
        return xx

    def forward(self, x):
        B, T, D = x.shape
        k = self.K(x)  # (B, T, D)
        q = self.Q(x)  # (B, T, D)

        # compute attention
        attn_matrix = (
            q @ k.transpose(-2, -1)
        ) * D**-0.5  # (B, T, D) @ (B, D, T) --> (B, T, T)
        attn_matrix = attn_matrix.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # (B, T, T)
        attn_matrix = softmax(attn_matrix, dim=-1)
        v = self.V(x)  # (B, T, T) @ (B, T, D) --> (B, T, D)
        out = attn_matrix @ v  # (B, T, D)
        return out


class BigramLanguageModel(nn.Module):
    """
    T --> n_embed, block_size
    D/C --> vocab length
    B --> batch size
    """

    def __init__(self, vocabulary):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = AttentionHead(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, X, y=None):
        """
        takes a batch, returns the loss and target for that batch
        """
        # x \in (batch_size, block_size)
        B, T = X.shape
        tok_emb = self.token_embedding_table(X)  # (B, T, D)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T, D)
        x = tok_emb + pos_emb
        x = self.sa_head(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if y is None:
            loss = None
        else:
            B, T, D = logits.shape
            logits = logits.view(B * T, D)
            y = y.view(B * T)

            # where D is n_classes, N is batch_size
            # cross_entropy requires shape (D) or shape (N, D),
            # but our data is (N, block_size, D)
            loss = F.cross_entropy(logits, y)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        for everything up to max_new_tokens,
        do a forward pass,
        softmax the logits,
        sample multinomial from the resulting distribution,
        append sample to running sequence
        """
        for _ in range(max_new_tokens):
            # idx is (B, T)
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            # focus only on the last timestep
            logits = logits[:, -1, :]  # becomes (B, D)
            probs = F.softmax(logits, dim=-1)  # (B, D)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


X, y = get_batch(data, batch_size=batch_size, block_size=block_size)
blm = BigramLanguageModel(chars)
logits, loss = blm(X, y)

# get a generation with a zero initiation and 100 new tokens
generation = blm.generate(
    idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100
)[0].tolist()

# train
# Create an optimization object (which eats the parameters of a model)
# then, with a batch size of 32,
# loop through 100 training steps.
#    at each one, sample a batch of data,
#    get the loss, do a backward step,
#    and then step forward on the optimizer

optimizer = torch.optim.AdamW(blm.parameters(), lr=1e-3)
for step in range(10000):
    xb, yb = get_batch(data, batch_size=batch_size, block_size=block_size)
    logits, loss = blm(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

with torch.no_grad():
    idx = blm.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[
        0
    ].tolist()
    print(decode(idx))

#%% [markdown]
# ## The mathematical trick in self-attention
A = torch.tril(torch.ones(3, 3, dtype=torch.float64))
A /= A.sum(dim=1, keepdim=True)  # each row is divided by its sum (count of 1s)
B = torch.arange(3 * 3, dtype=torch.float64).reshape(3, 3) + 1

torch.manual_seed(42)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)


# create x, random batch of shape (B, T, C)
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, : t + 1]
        xbow[b, t] = torch.mean(xprev, 0)


# make x[b, t] = mean_{i<=r} x[b, i]
