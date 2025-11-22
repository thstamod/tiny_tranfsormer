import torch
import torch.nn.functional as F

# Fix the random seed so training is reproducible and converges reliably.
#torch.manual_seed(0)

from tiny_transformer.tokenizer import tokenizer, tok2id, VOCAB
from tiny_transformer.embeddings import embed
from tiny_transformer.positional import add_positional_encoding
from tiny_transformer.transformer_block import transformer_block
from tiny_transformer.output_head import logits_for_ids  # we’ll define this in a second


# Small example dataset: a few prompts -> target next token
dataset = [
    ("Who created JavaScript ?", "Brendan"),
    ("Who created Python ?", "Guido"),
    ("Who created Linux ?", "Linus"),
]

# We import the global weight tensors from the modules so the optimizer can
# update them in-place. In a real project you'd likely wrap the model in an
# `nn.Module` and register parameters cleanly, but for this tiny example we
# keep the tensors at module scope.
from tiny_transformer import embeddings, positional, attention, feedforward, output_head

params = [
    embeddings.W,
    positional.POS,
    attention.Wq, attention.Wk, attention.Wv,
    feedforward.W1, feedforward.b1, feedforward.W2, feedforward.b2,
    output_head.Wout, output_head.bout,
]

optimizer = torch.optim.Adam(params, lr=0.01)


def train(num_epochs=500):
    """
    Tiny training loop that optimizes the global tensors to predict the
    target token for each prompt.

    This loop is intentionally short and explicit so beginners can see each
    step of training. High-level steps:
        1) Forward pass: compute logits from the model for a prompt.
        2) Compute loss: compare logits with the target token using cross-entropy.
        3) Backward pass: compute gradients with `loss.backward()`.
        4) Optimizer step: update parameters with `optimizer.step()`.

    Important low-level notes for non-experts:
        - `logits_for_ids` returns a 1-D tensor with shape [VOCAB_SIZE]. We
            call `.unsqueeze(0)` to turn it into a batch of size 1 with shape
            [1, VOCAB_SIZE] because `F.cross_entropy` expects a batch dimension.
        - `target` must be an integer dtype tensor (`torch.long`) with shape
            [batch_size] (here [1]).
        - `optimizer.zero_grad()` clears previously accumulated gradients. If
            you forget to call it, gradients will accumulate across steps.
        - `loss.backward()` computes gradients for every tensor that has
            `requires_grad=True` and contributed to `loss`.
        - `optimizer.step()` updates the tensors in `params` using the
            computed gradients.
    """

    for epoch in range(num_epochs):
        total_loss = 0.0

        for prompt, target_token in dataset:
            # 1) Forward: compute logits for this prompt
            ids = tokenizer(prompt)
            logits = logits_for_ids(
                ids,
                embed,
                add_positional_encoding,
                transformer_block,
            )  # shape [VOCAB_SIZE]

            # 2) Build target index (ensure long/int64 dtype for loss)
            target_id = tok2id[target_token]
            target = torch.tensor([target_id], dtype=torch.long)  # shape [1]

            # 3) Compute loss (cross-entropy). Ensure logits are float and have
            # shape [batch_size, num_classes]. cross_entropy expects raw logits
            # (it applies log-softmax internally).
            logits_batch = logits.unsqueeze(0).float()  # shape [1, VOCAB_SIZE]
            loss = F.cross_entropy(logits_batch, target)

            # 4) Backprop and parameter update
            # - Zero gradients (important: PyTorch accumulates grads by default)
            optimizer.zero_grad()
            # - Compute gradients by backpropagating the loss
            loss.backward()
            # - Apply the optimizer update to all tensors in `params`
            optimizer.step()

            total_loss += loss.item()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, loss = {total_loss:.4f}")


def test(prompt):
    """Run a single forward pass and print the top predictions."""
    ids = tokenizer(prompt)
    logits = logits_for_ids(
        ids,
        embed,
        add_positional_encoding,
        transformer_block,
    )
    probs = torch.softmax(logits, dim=-1)
    topv, topi = torch.topk(probs, k=5)

    for v, idx in zip(topv, topi):
        print(VOCAB[idx.item()], float(v.detach()))


if __name__ == "__main__":
    train()
    print("\nAfter training:\n")
    test("Who created JavaScript ?")
