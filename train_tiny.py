import torch
import torch.nn.functional as F

from tiny_transformer.tokenizer import tokenizer, tok2id, VOCAB
from tiny_transformer.embeddings import embed
from tiny_transformer.positional import add_positional_encoding
from tiny_transformer.transformer_block import transformer_block
from tiny_transformer.output_head import logits_for_ids  # we’ll define this in a second


dataset = [
    ("Who created JavaScript ?", "Brendan"),
    ("Who created Python ?", "Guido"),
    ("Who created Linux ?", "Linus"),
]
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

            # 2) Build target index
            target_id = tok2id[target_token]
            target = torch.tensor([target_id])  # shape [1]

            # 3) Compute loss (cross-entropy)
            logits_batch = logits.unsqueeze(0)  # shape [1, VOCAB_SIZE]
            loss = F.cross_entropy(logits_batch, target)

            # 4) Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, loss = {total_loss:.4f}")

if __name__ == "__main__":
    train()

def test(prompt):
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
