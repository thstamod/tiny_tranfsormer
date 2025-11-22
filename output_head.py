import torch
from tiny_transformer.tokenizer import VOCAB

D_MODEL = 8
VOCAB_SIZE = len(VOCAB)

# Output weights (these must be shared between train & predict)
Wout = torch.randn(D_MODEL, VOCAB_SIZE, requires_grad=True)
bout = torch.randn(VOCAB_SIZE, requires_grad=True)

def logits_for_ids(ids, embed, add_positional_encoding, transformer_block):
    """
    ids: list[int]
    returns: logits over vocab, shape [VOCAB_SIZE]
    """
    x = embed(ids)                           # [seq_len, D_MODEL]
    x = add_positional_encoding(x)           # [seq_len, D_MODEL]
    h = transformer_block(x)                 # [seq_len, D_MODEL]
    h_last = h[-1]                           # [D_MODEL]
    logits = h_last @ Wout + bout            # [VOCAB_SIZE]
    return logits
