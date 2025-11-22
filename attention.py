import torch
import math
from tiny_transformer.embeddings import embed
from tiny_transformer.tokenizer import tokenizer
from tiny_transformer.positional import add_positional_encoding

D_MODEL = 8

Wq = torch.randn(D_MODEL, D_MODEL, requires_grad=True)
Wk = torch.randn(D_MODEL, D_MODEL, requires_grad=True)
Wv = torch.randn(D_MODEL, D_MODEL, requires_grad=True)


def self_attention(x):
    """
    x: tensor of shape [seq_len, D_MODEL]
    returns: tensor of shape [seq_len, D_MODEL]
    """
    # 1. Compute Q, K, V
    # x: [seq_len, D]   Wq: [D, D]   => Q: [seq_len, D]
    # If A has shape (m, n) and B has shape (n, p), then A @ B has shape (m, p). Each output element C[i, j] = sum_k A[i, k] * B[k, j].
    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv

    # 2. Compute attention scores = Q @ K^T
    # Q: [seq_len, D],  K.T: [D, seq_len]  => scores: [seq_len, seq_len]
    scores = Q @ K.T

        # 3. Scale by sqrt(D_MODEL) (stabilizes softmax)
    scores = scores / math.sqrt(D_MODEL)

    # 4. Softmax over last dimension -> attention weights
    # Each row i: how much token i attends to each token j
    attn_weights = torch.softmax(scores, dim=-1)  # [seq_len, seq_len]
    
    # 5. Weighted sum of V
    # attn_weights: [seq_len, seq_len]
    # V:            [seq_len, D]
    # result:       [seq_len, D]
    out = attn_weights @ V
    return out


if __name__ == "__main__":
    text = "Who created JavaScript ?"
    ids = tokenizer(text)
    x = embed(ids)  # shape (seq_len, D_MODEL)
    x_pos = add_positional_encoding(x)
    out = self_attention(x_pos)
    print(out)