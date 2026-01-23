# Commonly Asked Questions
## Table Of Contents
- [Input and EMbedding Layer](#1-input--embedding-layer-token--positional-encoding)
    - [Basic Questions](#basic-questions)
        - [What is Token Embedding](#what-is-token-embedding)
        - [Why do we Need Positional Encoding in Transformers](#why-do-we-need-positional-encoding-in-transformers)
        - [What Happens is We Remove Positional Encoding](#what-happens-if-we-remove-positional-encoding)
    - [Intermediate Questions](#intermediate-questions)
        - [Why Can't Self-attention Alone Capture Word Order](#why-cant-self-attention-alone-capture-word-order)
- [Self-Attention](#2-self-attention)
    - [Basic Questions](#basic-questions-1)
        - [What is Self-Attention](#what-is-self-attention)
        - [What are Query, Key, and Value](#what-are-query-key-and-value)
- [Multi-Head Attention](#3-multi-head-attention)
    - [Basic Questions](#basic-questions-2)
    - [Intermediate Questions](#intermediate-questions-2)
- [Attention Masking(Encoder vs Decoder)](#4-attention-masking-encoder-vs-decoder)
    - [Basic Questions](#basic-questions-3)
    - [Intermediate Questions](#intermediate-questions)
- [Feed-Forward Network (FFN)](#5-feed-forward-network-ffn)
    - [Basic Questions](#basic-questions-4)
- [Residual Connections and Layer Normalization](#6-residual-connections--layer-normalization)
- [Encoder vs Decoder(Architecture Level)](#7-encoder-vs-decoder-architecture-level)
## 1. Input & Embedding Layer (Token + Positional Encoding)
### Basic questions
#### What is token embedding?

#### Why do we need positional encoding in Transformers?
##### Short Answer
- We need **positional encoding** because the Transformer’s self-attention mechanism is order-agnostic. 
- Without positional information, the model would treat a sentence as a bag of tokens and would not know **which word comes first, last, or next**.
##### Detailed Answer
1. Self-attention has no notion of order
2. Positional encoding injects sequence order
3. Why positional encoding is added, not concatenated
#### What happens if we remove positional encoding?

### Intermediate questions
#### Why can’t self-attention alone capture word order?
#### What’s the difference between sinusoidal and learned positional embeddings?
#### Why are positional embeddings added, not concatenated?

## 2. Self-Attention
### Basic questions
#### What is self-attention?
#### What are Query, Key, and Value?
##### Short Answer:
    - Query, Key, and Value are three learned vector representations of each token that allow the Transformer to compute attention weights—that is, how much each token should focus on other tokens when forming its contextual representation.
##### **Formal explanation (how it actually works)**
- For each token embedding 𝑥:

$$
    Q = x W_Q,\quad K = x W_K,\quad V = x W_V
$$

- WQ,WK,WV are learned projection matrices
- Q, K, V live in lower-dimensional spaces for efficiency
#### Why is it called self-attention?
### Intermediate questions
#### Why do we divide by √dₖ in scaled dot-product attention?
#### What does the softmax do in attention?
#### How is attention different from convolution or RNNs?
### Advanced questions
#### What happens if we remove the scaling factor √dₖ?

## 3. Multi-Head Attention
### Basic questions
#### Why do we use multiple attention heads?
#### What does each head learn differently?
### Intermediate questions

## 4. Attention Masking (Encoder vs Decoder)
### Basic questions
#### What is an attention mask?
- **Definition**:
    - An attention mask is a matrix that tells the Transformer which tokens are allowed to attend to which other tokens during self-attention.
#### Why does GPT use causal masking?

### Intermediate questions
### Advanced questions
## 5. Feed-Forward Network (FFN)
### Basic questions
### Intermediate questions
### Advanced questions

## 6. Residual Connections + Layer Normalization
### Basic questions
#### Why are residual connections used?
#### What problem does LayerNorm solve?

### Intermediate questions
#### Why LayerNorm instead of BatchNorm?
##### Short answer:
- Transformers use Layer Normalization instead of Batch Normalization because LayerNorm is **independent of batch size** and token position, making it stable for variable-length sequences, small or dynamic batches, and autoregressive generation.
##### Core reason (mechanism-level)
##### BatchNorm normalizes across the batch
- Computes mean/variance **over batch dimension**
- This breaks in NLP because:
    - Batch sizes are often **small or variable**
    - Sequences have **different lengths**
    - Autoregressive decoding often uses **batch size = 1**
##### LayerNorm normalizes within each token
- LayerNorm computes statistics across the feature dimension for each token independently:
$$
\mu = \frac{1}{d} \sum_{i=1}^{d} x_i,
\quad
\sigma = \sqrt{\frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2}
$$

- Why this works for Transformers:
    - No dependence on batch size
##### What breaks if we use BatchNorm in Transformers?
1. Training–inference mismatch
- BatchNorm uses batch statistics during training
- Uses running averages during inference
- Autoregressive generation has different statistics → instability
2. Batch size sensitivity
3. Sequence length variability
4. Hard to parallelize across time
- BN assumes spatial consistency (good for CNNs, bad for sequences)
##### Quick comparison table
| **Feature**             |**LayerNorm**| **BatchNorm**|
| ----------------------- | ----------  | ------------ |
| Depends on batch size   | ❌ No      | ✅ Yes     |
| Works with batch=1      | ✅ Yes     | ❌ No      |
| Handles variable length | ✅ Yes     | ❌ Poorly  |
| Used in Transformers    | ✅ Yes     | ❌ No      |
| Used in CNNs            | ❌ Rare    | ✅ Yes     |

### Advanced questions
#### Why do modern LLMs prefer Pre-LN?
#### How do residuals help prevent gradient vanishing?

## 7. Encoder vs Decoder (Architecture-Level)

### Basic Questions
#### What’s the difference between encoder and decoder?
#### Which models use encoder-only vs decoder-only?
- **Encoder-only models**
    - **What they are**
        - Use only the encoder stack
        - Bidirectional attention (each token sees left + right context)
        - Optimized for **understanding**, not generation
    - **Common encoder-only models**
        - BERT
        - RoBERTa
        - ALBERT
        - DistilBERT
        - ELECTRA
    - Typical tasks
        - Text classification
        - Sentiment analysis
        - Named Entity Recognition (NER)
        - Semantic search / embeddings
        - Extractive QA
- **Decoder-only models**
    - **What they are**
        - Use *only the decoder stack
        - Causal (masked) self-attention

### Intermediate Questions
#### Why is GPT decoder-only?
#### Why does translation use encoder–decoder?