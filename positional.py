import torch
from tiny_transformer.embeddings import embed
from tiny_transformer.tokenizer import tokenizer

D_MODEL = 8 
MAX_LEN = 32
POS = torch.randn(MAX_LEN, D_MODEL, requires_grad=True)

def add_positional_encoding(x):
    seq_len = x.shape[0]  
    return x + POS[:seq_len]



if __name__ == "__main__":
    text = "Who created JavaScript ?"
    ids = tokenizer(text)
    x = embed(ids)  # shape (seq_len, D_MODEL)
    x_pos = add_positional_encoding(x)
    print(x_pos)