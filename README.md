# tiny_transformer

A minimal educational transformer implementation in pure PyTorch (tiny toy model).
This repo contains a tiny tokenizer, embeddings, positional encoding, a single transformer block, and a tiny training script to demonstrate training on a very small dataset.

**Goal**: provide an easy-to-follow example of the forward and backward pass for a transformer-like model and show how logits → softmax → probabilities work.

---

## Requirements
- Python 3.8+ (you are using Python 3.13 — that should work)
- PyTorch (CPU-only is fine for this toy project)

Install via pip if needed:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

(If you have CUDA and want GPU builds, install the appropriate torch package for your system.)

---

## Quick start
From the repository root run:

```bash
# run the training script (it will print progress)
python -m tiny_transformer.train_tiny
# or
python tiny_transformer/train_tiny.py
```

The script trains the model on the small `dataset` hard-coded in `train_tiny.py` and prints top predictions after training.

---

## Files overview
- `tokenizer.py` — small vocab and tokenizer helpers (`VOCAB`, `tok2id`, `id2tok`, `tokenizer`).
- `embeddings.py` — embedding lookup / embedding matrix and `embed` helper.
- `positional.py` — positional encodings and an `add_positional_encoding` helper.
- `attention.py` — simple attention implementation; contains projection matrices (e.g. `Wq`, `Wk`, `Wv`).
- `feedforward.py` — MLP weights for transformer feed-forward.
- `transformer_block.py` — composes attention + feedforward into one block.
- `output_head.py` — projection from final representation to vocabulary logits (contains `Wout`, `bout` and `logits_for_ids`).
- `predict.py` — convenience `predict_next` that runs the forward pass and returns probabilities.
- `train_tiny.py` — small training loop that shows how to compute loss and run optimization.

---

## How training works (short)
1. Tokenize the prompt using `tokenizer(prompt)` to get token ids.
2. Build embeddings with `embed(ids)` and add positional encodings.
3. Run the `transformer_block` to get token representations `h` of shape `[seq_len, D_MODEL]`.
4. Take the last token `h_last = h[-1]` and compute `logits = h_last @ Wout + bout` — shape `[VOCAB_SIZE]`.
5. Use `torch.nn.functional.cross_entropy` with a target index tensor of dtype `torch.long`.

Important: pass raw logits (not probabilities) into `cross_entropy` — PyTorch applies log-softmax internally for numerical stability.

---

## Common errors & troubleshooting
Here are issues you might encounter (and quick fixes):

- AttributeError: module 'tiny_transformer.embeddings' has no attribute 'E'
  - Cause: `train_tiny.py` or other code expects a symbol named `E` but `embeddings.py` defines the embedding matrix under a different name (for example `W`).
  - Fix: either edit `embeddings.py` to add an alias `E = W` or change `train_tiny.py` to use `embeddings.W`.

- cross_entropy / type mismatch errors
  - Cause: `F.cross_entropy` expects `input` (logits) to be a floating tensor and `target` to be a `torch.LongTensor` (integer dtype). Also `input` should have shape `[batch_size, num_classes]` and `target` shape `[batch_size]`.
  - Fix: ensure you call
    ```py
    logits_batch = logits.unsqueeze(0).float()  # shape [1, VOCAB_SIZE]
    target = torch.tensor([target_id], dtype=torch.long)
    loss = F.cross_entropy(logits_batch, target)
    ```

- Small probabilities like `4.555184779591768e-22`
  - That's a perfectly valid probability from softmax — extremely small but not an error. Use `torch.log_softmax` / log-probabilities when inspecting tiny values. If you want less peaky distributions for debugging, use a temperature:
    ```py
    temp = 2.0
    probs = torch.softmax(logits / temp, dim=-1)
    ```

- Vocabulary / token missing
  - If you add new tokens (e.g. new names) to your dataset, add them to `VOCAB`, `tok2id`, and `id2tok` in `tokenizer.py`. After doing so, restart the Python process so weight matrices that depend on `VOCAB_SIZE` are re-created with the correct shape.
  - Optionally add an `UNK` token for unseen words.

---

## What are "logits"?
- Logits are the raw model scores for each token (one real number per vocabulary entry) before softmax. They can be any real number; softmax turns them into a probability distribution.
- In this code: `logits = h_last @ Wout + bout` produces shape `[VOCAB_SIZE]`.
- The index with highest logit is the model's top prediction (`argmax(logits)`), same as `argmax(softmax(logits))`.

---

## Suggested improvements / next steps
- Wrap the model into an `nn.Module` and use `nn.Parameter` for weights — cleaner and easier to save/load.
- Add mini-batching and a `DataLoader` for more data.
- Implement teacher forcing to train the model to predict each next token in a sequence instead of only the last token.
- Add checkpoint saving and model evaluation scripts.
- Add unit tests for small components (tokenizer, embed shapes, loss computation).

---

## Example debugging commands
```bash
# run training (prints epoch progress)
python -m tiny_transformer.train_tiny

# quickly inspect a forward pass in interactive mode
python -c "import tiny_transformer; from tiny_transformer.tokenizer import tokenizer; from tiny_transformer.predict import predict_next; print(predict_next('Who created JavaScript ?'))"
```
