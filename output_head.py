import torch
from tiny_transformer.tokenizer import VOCAB

# Output projection: maps the final token representation to a score
# (logit) for every token in the vocabulary.
D_MODEL = 8
VOCAB_SIZE = len(VOCAB)

# `Wout` and `bout` form a linear layer that maps a vector of size D_MODEL
# to a vector of size VOCAB_SIZE (one score per vocabulary token). These
# are initialized randomly here and are expected to be included in the list
# of trainable parameters used by the optimizer.
Wout = torch.randn(D_MODEL, VOCAB_SIZE, requires_grad=True)
bout = torch.randn(VOCAB_SIZE, requires_grad=True)


def logits_for_ids(ids, embed, add_positional_encoding, transformer_block):
        """
        Run a minimal forward pass for a single sequence of token ids and return
        the raw logits (unnormalized scores) over the vocabulary.

        Steps (simple language):
            1) Look up embeddings for the given ids and add positional information.
            2) Run the transformer block to produce token representations.
            3) Use the last token's vector to predict the next token by projecting
                 it to a vector of length `VOCAB_SIZE` (one score per token).

        Important note for beginners:
            - The returned `logits` are raw scores. To get probabilities use
                `torch.softmax(logits, dim=-1)`. When training, pass logits directly
                to `torch.nn.functional.cross_entropy` — it applies log-softmax
                internally for numerical stability.

        Args:
            ids: list[int] token ids for one prompt/sequence.
            embed: embedding lookup function.
            add_positional_encoding: function that adds position vectors.
            transformer_block: function that runs the single transformer block.

        Returns:
            logits: tensor of shape [VOCAB_SIZE] containing raw scores for each token.
        """

        x = embed(ids)                           # [seq_len, D_MODEL]
        x = add_positional_encoding(x)           # [seq_len, D_MODEL]
        h = transformer_block(x)                 # [seq_len, D_MODEL]

        # Use the last token's representation as the context vector for next-token
        # prediction. This is a common choice for causal/next-token models.
        h_last = h[-1]                           # [D_MODEL]

        # Project to vocabulary logits. `logits` contains one raw score per token.
        logits = h_last @ Wout + bout            # [VOCAB_SIZE]
        return logits
