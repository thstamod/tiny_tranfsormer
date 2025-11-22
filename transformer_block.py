import torch
from tiny_transformer.attention import self_attention
from tiny_transformer.feedforward import feedforward


def transformer_block(x):
    """
    A single transformer block (simplified).

    Steps applied to an input sequence `x` (shape [seq_len, D_MODEL]):
      1) Self-attention: each token can attend to other tokens in the same
         sequence. The attention output is added back to the input as a
         residual connection.
      2) Feed-forward: a position-wise MLP applied independently to each
         token, followed by another residual connection.

    Note: Common transformer implementations also apply layer normalization
    before/after these sub-layers and include dropout. Here we omit those
    elements to keep the example minimal and easy to follow.
    """

    # x: [seq_len, D_MODEL]

    # 1) Self-attention + residual connection
    attn_out = self_attention(x)    # [seq_len, D_MODEL]
    h = x + attn_out                # residual connection — add elementwise

    # 2) Feed-forward + residual
    ff_out = feedforward(h)         # [seq_len, D_MODEL]
    y = h + ff_out                  # residual again

    return y