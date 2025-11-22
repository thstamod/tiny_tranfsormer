from tiny_transformer.tokenizer import tokenizer, VOCAB, id2tok
from tiny_transformer.embeddings import embed
from tiny_transformer.positional import add_positional_encoding
from tiny_transformer.transformer_block import transformer_block
import torch

# NOTE: This module provides a convenience `predict_next` function used for
# quick interactive experiments. It defines its own `Wout` and `bout` for the
# output projection. The training script uses `output_head.py`'s `Wout`/`bout`,
# so `predict.py`'s weights are independent and only for quick local checks.

D_MODEL = 8
VOCAB_SIZE = len(VOCAB)

# Random projection for local prediction experiments (not the training head)
Wout = torch.randn(D_MODEL, VOCAB_SIZE, requires_grad=True)  # note shape
bout = torch.randn(VOCAB_SIZE, requires_grad=True)


def predict_next(text: str):
    """
    Run a forward pass for a single text prompt and return token probabilities.

    Step-by-step (very explicit):
        1) Tokenize the input string -> list of ids (Python list of integers).
        2) Lookup embeddings -> tensor `x` with shape [seq_len, D_MODEL].
        3) Add positional encodings to `x` -> still [seq_len, D_MODEL].
        4) Run the transformer block -> `h` with shape [seq_len, D_MODEL].
        5) Take the last row `h_last` (the final token's vector) -> shape [D_MODEL].
        6) Project to vocabulary logits (`h_last @ Wout + bout`) -> shape [VOCAB_SIZE].
        7) Convert logits to probabilities with softmax.

    Returns:
        A 1-D tensor of length `VOCAB_SIZE` containing probabilities summing to 1.

    Notes for beginners:
        - `logits` are raw scores; `softmax` turns them into probabilities.
        - For training use raw logits with `cross_entropy` (it applies log-softmax
          internally). Here softmax is used only for interactive inspection.
    """

    # 1) tokenize -> list[int]
    ids = tokenizer(text)

    # 2) embeddings + positional encodings -> tensor [seq_len, D_MODEL]
    x = embed(ids)
    x = add_positional_encoding(x)

    # 3) run transformer block -> [seq_len, D_MODEL]
    h = transformer_block(x)

    # 4) take last token representation -> [D_MODEL]
    h_last = h[-1]

    # 5) project to vocab logits -> [VOCAB_SIZE]
    logits = h_last @ Wout + bout

    # 6) softmax -> probabilities
    probs = torch.softmax(logits, dim=-1)  # [VOCAB_SIZE]

    return probs