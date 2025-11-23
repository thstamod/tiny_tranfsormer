"""
================================================================================
__init__.py - Package Initialization File
================================================================================

WHAT IS THIS FILE?
------------------
This is a special Python file that makes the `tiny_transformer` directory
into a Python "package" (a collection of modules that can be imported).

WHY IS IT (MOSTLY) EMPTY?
--------------------------
An __init__.py file can be empty! Just its presence tells Python:
"Hey, this directory contains Python modules that can be imported!"

HOW IT WORKS:
-------------
When you write:
    from tiny_transformer.tokenizer import tokenizer

Python looks for:
1. A directory named `tiny_transformer`
2. An __init__.py file inside it (this file!)
3. A file named `tokenizer.py` inside it

If __init__.py exists (even if empty), Python knows it can import from
the modules inside this directory!

WHAT COULD GO IN HERE?
-----------------------
You COULD add code like:
    from .tokenizer import tokenizer, VOCAB
    from .embeddings import embed
    # etc.

Then users could do:
    from tiny_transformer import tokenizer, embed

Instead of:
    from tiny_transformer.tokenizer import tokenizer
    from tiny_transformer.embeddings import embed

But for educational purposes, we keep it empty/minimal so the structure
is more explicit!

PYTHON PACKAGE STRUCTURE:
--------------------------
tiny_transformer/              ← Directory (package)
    __init__.py                ← This file (makes it a package!)
    tokenizer.py               ← Module
    embeddings.py              ← Module
    attention.py               ← Module
    feedforward.py             ← Module
    positional.py              ← Module
    transformer_block.py       ← Module
    output_head.py             ← Module
    train_tiny.py              ← Module (training script)
    predict.py                 ← Module (prediction script)

Because __init__.py exists, all these .py files can be imported!

FOR COMPLETE BEGINNERS:
-----------------------
If you're new to Python, just know:
- This file makes imports work
- It's like a "marker" that says "this is a package"
- It's totally fine that it's mostly empty!
- You don't need to modify it for this project

That's it! Simple but important. ✓
"""

# This file is intentionally minimal/empty
# Just its existence makes tiny_transformer a package!
