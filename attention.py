import torch
import math

# Self-attention implementation (single-head, not batched). This is the core
# operation used in transformer models: each token computes a weighted average
# of every token value in the sequence where weights are determined by the
# similarity between query and key vectors.

# Model hidden size (must match embeddings and positional config)
D_MODEL = 8

# Small learned projection matrices that convert token vectors into queries,
# keys and values. In a more idiomatic PyTorch implementation these would be
# `nn.Parameter` inside an `nn.Module`.
Wq = torch.randn(D_MODEL, D_MODEL, requires_grad=True)
Wk = torch.randn(D_MODEL, D_MODEL, requires_grad=True)
Wv = torch.randn(D_MODEL, D_MODEL, requires_grad=True)


def self_attention(x):
    """
    Single-head self-attention (no batching).

    Plain-language summary for beginners:
      - Each token in the sequence asks "how much should I listen to every other
        token?". The answer is a set of normalized weights (one row per token)
        that sum to 1.0. We then take a weighted average of the "value" vectors
        using those weights to produce the output for each token.

    Args:
      x: tensor shape [seq_len, D_MODEL] — input token vectors for one sequence.

    Returns:
      out: tensor shape [seq_len, D_MODEL] — new token vectors after attention.

    See the module-level comments for an analogy and a small numeric intuition.
    """

    # 1. Compute Q, K, V
    # Each has shape [seq_len, D_MODEL]
    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv

    # 2. Compute attention scores: [seq_len, seq_len]
    scores = Q @ K.T

    # 3. Scale scores to stabilize softmax
    scores = scores / math.sqrt(D_MODEL)

    # 4. Convert scores to probabilities (attention weights)
    attn_weights = torch.softmax(scores, dim=-1)  # shape [seq_len, seq_len]

    # 5. Produce the final outputs as weighted sums of values
    out = attn_weights @ V
    return out