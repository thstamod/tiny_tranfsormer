from tiny_transformer.tokenizer import tokenizer, VOCAB, id2tok
from tiny_transformer.embeddings import embed
from tiny_transformer.positional import add_positional_encoding
from tiny_transformer.transformer_block import transformer_block
import torch

D_MODEL = 8
VOCAB_SIZE = len(VOCAB)

Wout = torch.randn(D_MODEL, VOCAB_SIZE, requires_grad=True)  # note shape
bout = torch.randn(VOCAB_SIZE, requires_grad=True)


def predict_next(text: str):
    # 1) tokenize
    ids = tokenizer(text)

    # 2) embeddings + pos
    x = embed(ids)
    x = add_positional_encoding(x)

    # 3) transformer block
    h = transformer_block(x)              # [seq_len, D_MODEL]

    # 4) take last token representation
    h_last = h[-1]                        # [D_MODEL]

    # 5) project to vocab logits
    logits = h_last @ Wout + bout         # [VOCAB_SIZE]

    # 6) softmax
    probs = torch.softmax(logits, dim=-1) # [VOCAB_SIZE]

    return probs

if __name__ == "__main__":
    #predict_next("Who created JavaScript ?")
    probs = predict_next("Who created JavaScript ?")
    probs = probs.detach()  # in case grad is tracked

    values, indices = torch.topk(probs, k=4)

    for v, idx in zip(values, indices):
        print(id2tok[idx.item()], float(v))