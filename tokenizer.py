"""
tokenizer.py

This is a deliberately tiny, explicit tokenizer meant for teaching. It shows
the two essential concepts you need for the rest of the example:

- A vocabulary (`VOCAB`) — a small ordered list of token strings.
- Mappings between tokens and integer ids (`tok2id` and `id2tok`).

Why this is simple on purpose
- Real NLP systems use large vocabularies and subword tokenizers; those
  details would hide the learning goal here. We instead use a fixed small
  vocabulary so it's easy to inspect every step: token -> id -> embedding.

Design notes for beginners (plain language):
- A token is just a word or punctuation mark from `VOCAB`.
- The tokenizer replaces each token with a small integer id — computers
  and neural networks work with numbers, not words.
- If a word isn't in `VOCAB` it is silently ignored by this tiny tokenizer.
  In a more complete system you'd usually add an `UNK` (unknown) token so
  the model can represent unseen words.

Examples:
  >>> VOCAB
  ['Who', 'created', 'JavaScript', '?', 'Brendan', 'Guido', 'Linus']

  >>> tokenizer('Who created JavaScript ?')
  [0, 1, 2, 3]

If you add new tokens to `VOCAB` (for example new names), update the list
and restart the Python process so any tensors sized by vocabulary (like the
embedding matrix) are recreated to match the new size.
"""

VOCAB = ["Who", "created", "JavaScript", "?", "Brendan", "Guido", "Linus"]

# token -> id: used to convert words to integer indices before embedding
tok2id = {tok: i for i, tok in enumerate(VOCAB)}

# id -> token: convenient when printing model output
id2tok = {i: tok for i, tok in enumerate(VOCAB)}


def tokenizer(text):
    """
    Convert a short text string to a list of integer token ids.

    - `text` is split on whitespace (very simple).
    - any word not in `tok2id` is ignored (silently dropped).

    Returns a Python list of integers, e.g. `[0, 1, 2, 3]`.
    """
    return [tok2id[n] for n in text.split() if n in tok2id]