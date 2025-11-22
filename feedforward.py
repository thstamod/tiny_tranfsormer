import torch
import math
from tiny_transformer.embeddings import embed
from tiny_transformer.tokenizer import tokenizer
from tiny_transformer.positional import add_positional_encoding
from tiny_transformer.attention import self_attention

D_FF = 16
D_MODEL = 8

W1 = torch.randn(D_MODEL, D_FF, requires_grad=True)
b1 = torch.randn(D_FF, requires_grad=True)
W2 = torch.randn(D_FF, D_MODEL, requires_grad=True)
b2 = torch.randn(D_MODEL, requires_grad=True)

def feedforward(x):
     # 1) First linear: x @ W1 + b1
    h = x @ W1          # shape: [seq_len, D_FF]
    h = h + b1          # b1 will broadcast over seq_len

    # 2) ReLU
    h = torch.relu(h)

    # 3) Second linear: h @ W2 + b2
    y = h @ W2          # shape: [seq_len, D_MODEL]
    y = y + b2          # b2 broadcasts

    return y





if __name__ == "__main__":
    text = "Who created JavaScript ?"
    ids = tokenizer(text)
    x = embed(ids)  # shape (seq_len, D_MODEL)
    x_pos = add_positional_encoding(x)
    out = self_attention(x_pos)
    e = feedforward(out)
    print(e)