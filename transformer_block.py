"""
================================================================================
transformer_block.py - Putting It All Together: The Core Transformer Layer
================================================================================

WHAT IS A TRANSFORMER BLOCK?
-----------------------------
A transformer block is ONE complete layer of the transformer architecture.
It combines attention (words talking to each other) and feedforward (words
thinking independently) with a special trick called "residual connections".

Think of it as one "reasoning step" in the model:
1. Let words share information (attention)
2. Let words process that information (feedforward)
3. Keep the original information around (residual connections)

ANALOGY: Brainstorming Session
-------------------------------
Imagine a team solving a problem:

STEP 1 - DISCUSSION (Attention):
Everyone shares their ideas with the team. You listen to everyone,
especially those with relevant insights, and form new thoughts.

STEP 2 - PERSONAL REFLECTION (Feedforward):
You privately think about what you heard and refine your understanding.
You don't talk to others in this step - just process internally.

STEP 3 - REMEMBER STARTING POINT (Residual Connection):
You don't forget your original thoughts! You ADD the new insights to
what you already knew, creating a richer understanding.

THE POWER OF MULTIPLE BLOCKS:
------------------------------
Real transformers stack many blocks (GPT-3 has 96 blocks!):
- Block 1: Basic word relationships
- Block 2: Simple patterns  
- Block 3: More complex patterns
- ...
- Block 96: Very abstract, high-level concepts

We use just 1 block to keep things simple and understandable!

WHAT ARE RESIDUAL CONNECTIONS?
-------------------------------
A residual connection (or "skip connection") means:
    output = input + transformation(input)

Instead of just:
    output = transformation(input)

ANALOGY: Editing a Document
----------------------------
BAD APPROACH (no residual):
- You write a draft
- You edit it completely from scratch
- You might lose good original ideas!

GOOD APPROACH (with residual):
- You write a draft
- You ADD improvements to the existing draft
- You keep what was good, add what's new!

WHY ARE RESIDUAL CONNECTIONS IMPORTANT?
----------------------------------------
1. PREVENTS INFORMATION LOSS:
   Original information is preserved, new information is added
   
2. HELPS TRAINING (Gradient Flow):
   In deep networks, gradients can "vanish" (become too small)
   Residual connections provide a direct path for gradients
   This helps the model learn better!

3. IDENTITY SHORTCUT:
   If a transformation isn't helpful, the model can learn to ignore it
   (by setting transformation ≈ 0, so output ≈ input)

MATHEMATICAL NOTATION:
----------------------
If we have input x:

Without residual:
  y = feedforward(attention(x))

With residual:
  temp = x + attention(x)           ← first residual
  y = temp + feedforward(temp)      ← second residual

The model learns transformations that ADD to the input, not replace it!

THE TRANSFORMER BLOCK FLOW:
----------------------------

Input x (word vectors with position info)
    ↓
[ATTENTION] - words share information
    ↓
Add back to input (x + attention_output) → First Residual Connection
    ↓ 
[FEEDFORWARD] - words process information
    ↓
Add back to previous (h + feedforward_output) → Second Residual Connection
    ↓
Output y (enriched word vectors)

WHAT'S MISSING HERE (But Exists in Real Transformers)?
-------------------------------------------------------
To keep this educational, we omit:

1. LAYER NORMALIZATION:
   - Normalizes values to have mean=0, variance=1
   - Helps training stability
   - Usually applied before or after each sub-layer

2. DROPOUT:
   - Randomly "drops out" (sets to zero) some values during training
   - Prevents overfitting (model memorizing training data)
   - Makes the model more robust

3. MULTI-HEAD ATTENTION:
   - Run multiple attention operations in parallel
   - Each "head" can learn different patterns
   - Combine results at the end

These are important for real models, but we skip them to focus on core concepts!
"""

# ============================================================================
# IMPORTS
# ============================================================================
import torch  # PyTorch for tensor operations
from tiny_transformer.attention import self_attention  # The attention mechanism
from tiny_transformer.feedforward import feedforward  # The feedforward network


# ============================================================================
# THE TRANSFORMER BLOCK FUNCTION
# ============================================================================

def transformer_block(x):
    """
    Apply one complete transformer layer with attention, feedforward, and residuals.
    
    This is the CORE of the transformer architecture! Everything else is just
    preparation (embeddings, positions) or post-processing (output projection).
    
    INPUT:
    ------
    x : PyTorch tensor with shape [seq_len, D_MODEL]
        - seq_len = number of words (e.g., 4 for "Who created Python ?")
        - D_MODEL = 8
        
        This should be the output from add_positional_encoding(), containing
        word embeddings with positional information.
    
    OUTPUT:
    -------
    y : PyTorch tensor with shape [seq_len, D_MODEL]
        - Same shape as input
        - Each word now has richer, more contextual representations
        - Ready to be processed further or sent to output layer
    
    THE TWO MAIN STEPS:
    -------------------
    
    STEP 1: Self-Attention + Residual Connection
    ---------------------------------------------
    Let words communicate and share information.
    
    attn_out = self_attention(x)
    - x: [seq_len, 8]  (input word vectors)
    - attn_out: [seq_len, 8]  (context-aware word vectors)
    
    Each word in attn_out has "looked at" other words and incorporated
    their information based on attention weights.
    
    h = x + attn_out  ← RESIDUAL CONNECTION
    
    Instead of replacing x with attn_out, we ADD them together!
    - x: original word information
    - attn_out: new contextual information from other words
    - h: combination of both (original + contextual)
    
    This is element-wise addition (add corresponding numbers):
    h[i][j] = x[i][j] + attn_out[i][j]
    
    Example for one word with 3 dimensions:
    x = [0.5, 0.3, 0.8]
    attn_out = [0.1, -0.2, 0.4]
    h = [0.6, 0.1, 1.2]  ← element-wise sum
    
    STEP 2: Feedforward + Residual Connection
    ------------------------------------------
    Let words independently process the information they've gathered.
    
    ff_out = feedforward(h)
    - h: [seq_len, 8]  (input from previous step)
    - ff_out: [seq_len, 8]  (transformed through feedforward network)
    
    Each word goes through expand→ReLU→compress transformation.
    
    y = h + ff_out  ← RESIDUAL CONNECTION
    
    Again, we ADD instead of replacing!
    - h: word information after attention
    - ff_out: additional transformations from feedforward
    - y: combination of both
    
    VISUAL FLOW DIAGRAM:
    --------------------
    
    x (input)
    |
    ├──────────────┐
    |              ↓
    |         self_attention
    |              ↓
    └──────→ [+] (add) → h
                         |
                         ├──────────────┐
                         |              ↓
                         |         feedforward
                         |              ↓
                         └──────→ [+] (add) → y (output)
    
    The arrows going around (└──────→) are the residual connections!
    They create "shortcut paths" that bypass the transformations.
    
    WHY TWO RESIDUAL CONNECTIONS?
    ------------------------------
    Each major transformation (attention and feedforward) gets its own residual.
    This gives the model maximum flexibility:
    - Can learn to use attention heavily (if useful)
    - Can learn to use feedforward heavily (if useful)  
    - Can learn to use both
    - Can learn to mostly skip them (if input is already good)
    
    EXAMPLE WITH REAL NUMBERS:
    --------------------------
    Let's trace one word through the block:
    
    Input x (after embeddings + positional encoding):
    [0.5, 0.3, 0.8, 0.2, -0.1, 0.6, 0.4, -0.2]
    
    After attention (incorporates context from other words):
    attn_out = [0.1, -0.2, 0.3, 0.1, 0.2, -0.1, 0.0, 0.3]
    
    After first residual (x + attn_out):
    h = [0.6, 0.1, 1.1, 0.3, 0.1, 0.5, 0.4, 0.1]
    
    After feedforward (non-linear transformation):
    ff_out = [0.2, 0.3, -0.1, 0.4, -0.2, 0.1, 0.3, 0.2]
    
    After second residual (h + ff_out):
    y = [0.8, 0.4, 1.0, 0.7, -0.1, 0.6, 0.7, 0.3]
    
    Final output y is now richer than input x!
    
    WHAT HAPPENS NEXT?
    ------------------
    In a deeper model, y would be passed to another transformer_block.
    In our simple model, y goes to the output_head for next-token prediction.
    
    THE BEAUTY OF THIS DESIGN:
    --------------------------
    - Simple components (attention, feedforward, addition)
    - Powerful when combined
    - Residual connections enable deep networks
    - Proven to work incredibly well in practice!
    
    This architecture (with many blocks) powers GPT, BERT, and other
    state-of-the-art language models!
    """

    # ========================================================================
    # Original input - we'll keep referring back to this via residuals
    # ========================================================================
    # x: [seq_len, D_MODEL]

    # ========================================================================
    # STEP 1: Self-Attention + Residual Connection
    # ========================================================================
    # Let each word gather information from other words
    attn_out = self_attention(x)    # [seq_len, D_MODEL]
    
    # Add attention output back to the original input (RESIDUAL)
    # This preserves the original information while adding contextual info
    h = x + attn_out                # element-wise addition, shape: [seq_len, D_MODEL]

    # ========================================================================
    # STEP 2: Feed-Forward + Residual Connection
    # ========================================================================
    # Let each word independently process the information it has gathered
    ff_out = feedforward(h)         # [seq_len, D_MODEL]
    
    # Add feedforward output back to the previous state (RESIDUAL)
    # This preserves previous information while adding new transformations
    y = h + ff_out                  # element-wise addition, shape: [seq_len, D_MODEL]

    # Return the enriched word representations!
    return y