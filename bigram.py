# no cheating

#%%
from datasets import load_dataset
import torch
from torch import nn
from torch.nn.functional import cross_entropy


def get_batch(data, batch_size=4, block_size=8):
    """
    Take a set of encoded data.
    Return X (n_batch, tokens_per_batch)
    """
    indices = torch.randint(len(data) - block_size, size=(batch_size,))
    X = torch.tensor([data[t : t + block_size] for t in indices])
    y = torch.tensor([data[t + 1 : t + block_size + 1] for t in indices])
    return X, y


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_len):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_len, vocab_len)

    def forward(self, X, y=None):
        # X is (batch_size:=B, tokens_per_batch:=T)
        # C is the number of letters in the vocabulary
        logits = self.embedding_table(X)  # (B, T, C)

        if y is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # (N, C)
            y = y.view(B * T)  # (N)

            loss = cross_entropy(logits, y)  # torch.float32

        return logits, loss

    def generate(self, idx, max_len=100):
        for _ in range(max_len):
            logits, _ = self(idx)  # (B, T, C)
            logits = logits[:, -1, :]  # (B, 1, C): predict from last logit
            probs = nn.functional.softmax(logits, dim=1)  # (B, 1, C)
            sample = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, sample), dim=1)
        return idx


device = "cuda" if torch.cuda.is_available() else "cpu"
data = load_dataset("tiny_shakespeare")
train, validation, test = [data[k]["text"][0] for k in data.keys()]
text = train + validation + test
vocab = sorted(list(set(text)))
vocab_len = len(vocab)
encoder = lambda sentences: [vocab.index(tok) for tok in sentences]
decoder = lambda integers: "".join([vocab[i] for i in integers])
n_epochs = 1000

# turn data into a bunch of integers
data = encoder(text)
X, y = get_batch(data)
idx = torch.zeros(1, 1, dtype=torch.long)
m = BigramLanguageModel(vocab_len)
logits, loss = m(X, y)
print(loss.item())
# idx = m.generate()

# training loop
optimizer = torch.optim.AdamW(m.parameters())
for epoch in range(n_epochs):
    bx, by = get_batch(data)
    logits, loss = m(bx, by)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(loss.item())

# %%
