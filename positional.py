import torch

# Positional encodings: these are vectors added to token embeddings so the
# model has a notion of token order. Without positional info, the model
# would only see an unordered bag of tokens and could not distinguish
# "Who created Python ?" from "Python created Who ?".
#
# In production code positional encodings are often deterministic sin/cos
# functions. For this tiny example we use a learned small table to keep the
# implementation explicit and easy to understand.

# Model hidden size (must match embeddings.D_MODEL)
D_MODEL = 8

# Maximum sequence length supported by our positional table. If your input
# has more tokens than this, you'll need to increase MAX_LEN and restart.
MAX_LEN = 32

# POS is a small table of learned positional vectors. Shape: [MAX_LEN, D_MODEL].
# `requires_grad=True` means the position vectors can be updated during
# training so the model can learn position-specific signals.
POS = torch.randn(MAX_LEN, D_MODEL, requires_grad=True)


def add_positional_encoding(x):
    """
    Add positional vectors to a sequence of token embeddings.

    Plain-language steps:
      1) `x` is a tensor with shape [seq_len, D_MODEL] produced by `embed(ids)`.
      2) We select the first `seq_len` rows from the POS table and add them to
         `x` elementwise. This gives each token a position-specific offset.

    Example:
      - If `x` has shape [4, 8], the returned tensor will also have shape
        [4, 8] where each row has been shifted by a learned positional vector.

    Returns:
      A tensor with the same shape as `x` with position information added.
    """
    seq_len = x.shape[0]
    return x + POS[:seq_len]