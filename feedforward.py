"""
================================================================================
feedforward.py - The "Thinking" Layer: Processing Information Further
================================================================================

WHAT IS A FEEDFORWARD NETWORK?
-------------------------------
After attention lets words share information with each other, we need another
layer to PROCESS and TRANSFORM that information. This is what the feedforward
network does!

ANALOGY: A Factory Assembly Line
---------------------------------
Imagine building a car:
1. ATTENTION: Workers communicate and share parts
   - "Hey, I need a wheel!" "Here, take mine!"
   - Information flows between stations

2. FEEDFORWARD: Each worker processes their parts
   - Takes the collected parts
   - Transforms them (cuts, welds, paints)
   - Produces a refined component

The feedforward network is like each worker's personal toolkit!

WHY DO WE NEED THIS?
---------------------
Attention is LINEAR - it just mixes vectors together:
  output = 0.3×word1 + 0.5×word2 + 0.2×word3

But language is COMPLEX and NON-LINEAR:
- "not good" ≠ "good" (negation)
- "very good" is stronger than "good" (intensification)

We need NON-LINEAR transformations to capture these complex patterns!

WHAT IS NON-LINEAR?
-------------------
Linear: output = a×input + b (straight line relationship)
Non-linear: curves, bends, thresholds

Example of non-linearity (ReLU function):
- If input is 5 → output is 5
- If input is -3 → output is 0 (clamp negative to zero!)
- This "threshold" behavior is non-linear

THE FEEDFORWARD ARCHITECTURE:
------------------------------
It's a simple two-layer neural network:

1. EXPAND: Increase vector size (8 → 16 numbers)
   - Gives the model more "room" to compute
   - Like giving a worker more tools

2. ACTIVATE: Apply ReLU (non-linear function)
   - Introduces non-linearity
   - Helps model learn complex patterns

3. COMPRESS: Reduce back to original size (16 → 8 numbers)
   - Return to the standard size for the rest of the model
   - Distill the computation back down

VISUAL FLOW:
------------
[8 numbers] → EXPAND → [16 numbers] → ReLU → [16 numbers] → COMPRESS → [8 numbers]
   input       (W1+b1)                                        (W2+b2)      output

WHY EXPAND THEN COMPRESS?
--------------------------
This is called a "bottleneck" architecture (well, reverse bottleneck):

- Expanding gives the network more "computational space"
- It can compute intermediate features in the larger space
- Then compress back to maintain consistent dimensions

ANALOGY: Thinking Space
You might have complex thoughts (16 dimensions) but express them
simply (8 dimensions). The expanded space is your "thinking room"!

POSITION-WISE MEANS INDEPENDENT:
---------------------------------
"Position-wise" means we apply the SAME transformation to each word independently.

If we have 4 words:
- Process word 1 through the network
- Process word 2 through the network (same network, same weights!)
- Process word 3 through the network
- Process word 4 through the network

All words share the same W1, b1, W2, b2 parameters!

This is DIFFERENT from attention where words interact with each other.
Here, each word is processed independently (but with the same function).
"""

# ============================================================================
# IMPORTS
# ============================================================================
import torch  # PyTorch for tensor operations
import math   # Not used here but imported historically

# Import other components (these aren't used in this file but were imported)
# They're here for potential experimentation
from tiny_transformer.embeddings import embed
from tiny_transformer.tokenizer import tokenizer
from tiny_transformer.positional import add_positional_encoding
from tiny_transformer.attention import self_attention


# ============================================================================
# CONFIGURATION
# ============================================================================

# D_FF: The intermediate "expanded" dimension (feedforward dimension)
# - We expand from 8 to 16 (double the size)
# - Gives the network more room to compute complex transformations
# - Real transformers often use D_FF = 4 × D_MODEL
# - We use 2 × D_MODEL (16 = 2 × 8) to keep it simple
D_FF = 16

# D_MODEL: The standard dimension (must match other modules)
# - Input size: 8 numbers
# - Output size: 8 numbers (same as input)
D_MODEL = 8


# ============================================================================
# WEIGHTS AND BIASES: The Learnable Parameters
# ============================================================================

# LAYER 1: Expand from D_MODEL to D_FF
# --------------------------------------

# W1: Weight matrix for first layer [D_MODEL, D_FF] = [8, 16]
# This transforms an 8-number vector into a 16-number vector
# Think of it like: "How do we expand our 8 dimensions into 16?"
W1 = torch.randn(D_MODEL, D_FF, requires_grad=True)

# b1: Bias vector for first layer [D_FF] = [16]
# After multiplying by W1, we add these 16 numbers
# Biases let each neuron have an "offset" or "threshold"
# Think of bias as: "starting point" before considering input
b1 = torch.randn(D_FF, requires_grad=True)

# LAYER 2: Compress from D_FF back to D_MODEL
# ---------------------------------------------

# W2: Weight matrix for second layer [D_FF, D_MODEL] = [16, 8]
# This transforms a 16-number vector back into an 8-number vector
# Think of it like: "How do we compress our 16 dimensions back to 8?"
W2 = torch.randn(D_FF, D_MODEL, requires_grad=True)

# b2: Bias vector for second layer [D_MODEL] = [8]
# After multiplying by W2, we add these 8 numbers
# Final bias adjustment before returning the result
b2 = torch.randn(D_MODEL, requires_grad=True)

# All four parameters start RANDOM and are LEARNED during training!
# requires_grad=True means PyTorch will update them via backpropagation


# ============================================================================
# THE FEEDFORWARD FUNCTION
# ============================================================================

def feedforward(x):
          """
          Apply a two-layer feedforward neural network to each word independently.
          
          WHAT THIS DOES:
          ---------------
          Takes word vectors and transforms them through a small neural network.
          Each word is processed independently using the same network.
          
          INPUT:
          ------
          x : PyTorch tensor with shape [seq_len, D_MODEL]
              - seq_len = number of words (e.g., 4)
              - D_MODEL = 8
              
              This is typically the output from attention, containing
              context-aware word representations.
          
          OUTPUT:
          -------
          y : PyTorch tensor with shape [seq_len, D_MODEL]
              - Same shape as input
              - Each word has been transformed by the feedforward network
          
          THE THREE OPERATIONS:
          ---------------------
          
          OPERATION 1: Expand + Bias (Linear Layer 1)
          --------------------------------------------
          h = x @ W1 + b1
          
          Step-by-step:
          - x @ W1: Matrix multiply x by W1
            - x: [seq_len, 8]
            - W1: [8, 16]
            - Result: [seq_len, 16]
            - Each 8-number vector becomes a 16-number vector
          
          - h + b1: Add bias
            - h: [seq_len, 16]
            - b1: [16]
            - b1 "broadcasts" - it's added to every row of h
            - Result: [seq_len, 16]
          
          What is broadcasting?
          If h is shape [4, 16] and b1 is shape [16]:
          We add b1 to EACH of the 4 rows:
          [[row 0] + b1,
           [row 1] + b1,
           [row 2] + b1,
           [row 3] + b1]
          
          OPERATION 2: ReLU (Non-linear Activation)
          ------------------------------------------
          h = torch.relu(h)
          
          What is ReLU?
          ReLU = "Rectified Linear Unit"
          It's a simple function: ReLU(x) = max(0, x)
          - If x is positive: keep it (output = x)
          - If x is negative: set to zero (output = 0)
          
          Example:
          Input:  [1.5, -0.3, 2.1, -1.0, 0.5]
          Output: [1.5,  0.0, 2.1,  0.0, 0.5]
                       ↑ negative→0  ↑ negative→0
          
          Why ReLU?
          - Introduces NON-LINEARITY (not a straight line)
          - Simple and fast to compute
          - Works well in practice (despite being simple!)
          - Helps model learn complex patterns
          
          Without ReLU (or another non-linear function), stacking
          multiple layers would be pointless - they'd collapse into
          a single linear transformation!
          
          OPERATION 3: Compress + Bias (Linear Layer 2)
          ----------------------------------------------
          y = h @ W2 + b2
          
          Step-by-step:
          - h @ W2: Matrix multiply h by W2
            - h: [seq_len, 16]
            - W2: [16, 8]
            - Result: [seq_len, 8]
            - Each 16-number vector becomes an 8-number vector (compressed!)
          
          - y + b2: Add bias
            - y: [seq_len, 8]
            - b2: [8]
            - b2 broadcasts to every row
            - Result: [seq_len, 8] (final output!)
          
          FULL EXAMPLE WITH NUMBERS:
          --------------------------
          Imagine x has 1 word (seq_len=1) with 8 numbers:
          x = [[0.5, -0.2, 0.8, 0.1, -0.3, 0.6, 0.4, -0.1]]
          
          After x @ W1 + b1 (expand to 16 dimensions):
          h = [[0.3, -0.5, 1.2, 0.7, -0.8, 2.1, ..., 0.4]]  (16 numbers)
          
          After ReLU (clamp negatives to 0):
          h = [[0.3, 0.0, 1.2, 0.7, 0.0, 2.1, ..., 0.4]]  (negatives → 0)
                   ↑ was -0.5        ↑ was -0.8
          
          After h @ W2 + b2 (compress to 8 dimensions):
          y = [[0.7, 0.2, -0.4, 1.1, 0.3, -0.6, 0.8, 0.5]]  (8 numbers)
          
          We've transformed the word through a complex non-linear function!
          
          WHY IS THIS USEFUL?
          -------------------
          The feedforward network can learn to:
          - Recognize complex patterns
          - Combine features in non-obvious ways
          - Transform representations into more useful forms
          
          Combined with attention, it gives the model powerful expressiveness!
          
          Returns:
               - y: [seq_len, D_MODEL] - transformed word vectors
          """

          # ====================================================================
          # STEP 1: First Linear Layer (Expand)
          # ====================================================================
          h = x @ W1          # Matrix multiply: [seq_len, 8] @ [8, 16] = [seq_len, 16]
          h = h + b1          # Add bias: b1 broadcasts across seq_len dimension

          # ====================================================================
          # STEP 2: Non-linearity (ReLU Activation)
          # ====================================================================
          # Apply ReLU: keep positive values, set negative values to zero
          h = torch.relu(h)

          # ====================================================================
          # STEP 3: Second Linear Layer (Compress)
          # ====================================================================
          y = h @ W2          # Matrix multiply: [seq_len, 16] @ [16, 8] = [seq_len, 8]
          y = y + b2          # Add bias: b2 broadcasts across seq_len dimension

          # Return the transformed vectors!
          return y