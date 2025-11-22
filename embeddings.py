import torch
from tiny_transformer.tokenizer import VOCAB

# Embeddings convert integer token ids into dense vectors the model can use.
# Think of `W` as a lookup table: each row corresponds to a token in the
# vocabulary and contains D_MODEL numbers (the token's vector).

# Model hidden size: how many numbers represent each token
D_MODEL = 8

# Use the tokenizer's vocabulary length so shapes stay consistent
VOCAB_SIZE = len(VOCAB)

# Embedding matrix: one row per vocabulary token, one column per hidden dim.
# Shape: [VOCAB_SIZE, D_MODEL]. Each row is the embedding vector for a token id.
# `requires_grad=True` means PyTorch will track gradients for this tensor so it
# can be updated by the optimizer during training (the model "learns" better
# embeddings over time).
W = torch.randn(VOCAB_SIZE, D_MODEL, requires_grad=True)


def embed(ids):
        """
        Look up embeddings for a list of token ids.

        Args:
            ids: a Python list (or 1-D tensor) of integer token ids, e.g. [0, 1, 2].

        Returns:
            A PyTorch tensor with shape [seq_len, D_MODEL] where seq_len == len(ids).

        Notes for beginners:
            - If you pass `ids=[0,1]` and `D_MODEL=8`, the returned tensor has shape
                [2, 8]. Each row is the vector for a token.
            - The returned type is a tensor; you can convert to a NumPy array with
                `.detach().numpy()` when you only need to inspect values.
        """
        # Index the embedding matrix along the first (vocabulary) dimension.
        return W[ids]