#%%
import torch
from torch import nn
from torch.nn.functional import cross_entropy, softmax
from datasets import load_dataset

from bigram import get_batch

# hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
data = load_dataset("tiny_shakespeare")
train, validation, test = [data[k]["text"][0] for k in data.keys()]
text = train + validation + test
vocab = sorted(list(set(text)))
vocab_len = len(vocab)
encoder = lambda sentences: [vocab.index(tok) for tok in sentences]
decoder = lambda integers: "".join([vocab[i] for i in integers])
n_epochs = 1000
batch_size = 4
block_size = 8

# turn data into a bunch of integers
data = encoder(text)
X, y = get_batch(data, batch_size=batch_size, block_size=block_size)
#%%


class AttentionBlock(nn.Module):
    def __init__(self, block_size, embedding_dim, head_size):
        super().__init__(self)

        # weight matrices
        self.K = nn.Linear(embedding_dim, head_size, bias=False)
        self.Q = nn.Linear(embedding_dim, head_size, bias=False)
        self.V = nn.Linear(embedding_dim, head_size, bias=False)

        # hyperparameters
        self.block_size = self.T = block_size
        self.embedding_dim = self.C = embedding_dim
        self.head_size = self.D = head_size

    def _mask(self, xx):
        tril = torch.tril(torch.ones((self.T, self.T)))
        xx = torch.where(tril == 0, -1 * torch.inf, xx)
        return xx

    def forward(self, x, mask=True):
        B, T, C = x.shape
        k = self.K(x)  # (B, T, D)

        # compute attention
        q = self.Q(x)  # (B, T, D)
        attn_matrix = (
            q @ k.transpose(-2, -1)
        ) * head_size**-0.5  # (B, T, C) @ (B, C, T) --> (B, T, T)
        if mask:
            attn_matrix = self._mask(attn_matrix)
        attn_matrix = softmax(attn_matrix, axis=-1)
        v = self.V(x)  # (B, T, D)
        out = attn_matrix @ v  # (B, T, D)
        return out




# create the inner workings

# with a head size of 16,
# make a key and query matrix,
# multiply them with self-attention,
# make the dot product between them,
# then fill in the upper-triangle of that with -inf,
# then softmax
B, T, C = 4, 8, 32
head_size = 16
x = torch.randn(B, T, C)  # (B, T, C)
#%%
K = nn.Linear(C, head_size, bias=False)
Q = nn.Linear(C, head_size, bias=False)
V = nn.Linear(C, head_size, bias=False)
#%%
k = K(x)  # (B, T, 16)
q = Q(x)  # (B, T, 16)

wei = (q @ k.transpose(-2, -1)) * head_size**-0.5
# (B, T, 16) @ (B, 16, T) -> (B, T, T)

tril = torch.tril(torch.ones((T, T)))

wei = torch.where(tril == 0, -1 * torch.inf, wei)
#%%
wei = softmax(wei, dim=-1)  # (T, T)
v = V(x)
out = wei @ v
wei[0]
# X = torch.zeros((T, T))

# %%
