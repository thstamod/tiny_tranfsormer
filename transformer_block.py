import torch
from tiny_transformer.attention import self_attention
from tiny_transformer.feedforward import feedforward


def transformer_block(x):
    # x: [seq_len, D_MODEL]

    # 1) Self-attention + residual
    attn_out = self_attention(x)    # [seq_len, D_MODEL]
    h = x + attn_out                # residual connection

    # 2) Feed-forward + residual
    ff_out = feedforward(h)        # [seq_len, D_MODEL]
    y = h + ff_out                  # residual again

    return y   