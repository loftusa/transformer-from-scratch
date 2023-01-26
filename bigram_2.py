# no cheating

#%%
from datasets import load_dataset
import torch
from torch import nn


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
        self.embedding_table = nn.Embedding(vocab_len, vocab_len)

    def forward(self, X, y=None):
        """
        
        """

        


data = load_dataset("tiny_shakespeare")
train, validation, test = [data[k]["text"][0] for k in data.keys()]
text = train + validation + test
vocab = sorted(list(set(text)))
encoder = lambda sentences: [vocab.index(tok) for tok in sentences]
decoder = lambda integers: "".join([vocab[i] for i in integers])

# turn data into a bunch of integers
data = encoder(text)
X, y = get_batch(data)

# %%
