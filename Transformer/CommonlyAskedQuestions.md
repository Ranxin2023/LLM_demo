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
    - [Intermediate Question](#intermediate-questions-1)
        - [Why do we divide by √dₖ in scaled dot-product attention?](#why-do-we-divide-by-dₖ-in-scaled-dot-product-attention)
- [Multi-Head Attention](#3-multi-head-attention)
    - [Basic Questions](#basic-questions-2)
    - [Intermediate Questions](#intermediate-questions-2)
- [Attention Masking(Encoder vs Decoder)](#4-attention-masking-encoder-vs-decoder)
    - [Basic Questions](#basic-questions-3)
    - [Intermediate Questions](#intermediate-questions)
- [Feed-Forward Network (FFN)](#5-feed-forward-network-ffn)
    - [Basic Questions](#basic-questions-4)
        - [What is the Role of the Feed-Forward Network](#what-is-the-role-of-the-feed-forward-network)
- [Residual Connections and Layer Normalization](#6-residual-connections--layer-normalization)
    - [Basic Questions](#basic-questions-5)
        - [What Problem does LayerNorm Solve](#what-problem-does-layernorm-solve)
    - [Interdemdiate Question](#intermediate-questions-5)
- [Encoder vs Decoder(Architecture Level)](#7-encoder-vs-decoder-architecture-level)
    - [Basic Questions](#basic-questions-6)
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
##### Short answer (interview-ready):
- We divide by **√dₖ** to keep dot-product magnitudes **numerically stable**. Without this scaling, attention scores grow with dimension, pushing softmax into saturation, which causes **vanishing gradients** and unstable training.
##### What’s going wrong without scaling?
- In dot-product attention, scores are:

$$
\text{score} = Q K^{\top}
$$

- If the components of 𝑄 and 𝐾 have zero mean and unit variance, then:
    - 

    $$
    \mathbb{E}[QK^{\top}] \propto d_k
    $$
    - Larger 𝑑𝑘 ⇒ larger variance of scores

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
#### What is the role of the feed-forward network?
##### Short Answer
- The **feed-forward network (FFN)** provides **non-linear transformation and feature mixing at each token**, allowing the Transformer to increase representational capacity beyond attention by independently transforming each token’s features.
##### What the FFN is in a Transformer
- Inside every Transformer block, after self-attention, there is a **position-wise feed-forward network**:

$$
\mathrm{FFN}(x) = W_2 \, \sigma\!\left(W_1 x + b_1\right) + b_2
$$
- Same FFN is applied to **every token**
- Operates **independently per position**
- Usually much **wider** than the model dimension
##### Why attention alone is not enough
##### What the FFN actually does (intuition)
- Think of a Transformer block as:
    - **Attention** → “Which tokens should I look at?”
    - **FFN** → “How should I process what I’ve gathered?”
- Attention decides **where** to get information.
- FFN decides **how** to transform it.
##### Why the FFN is “position-wise”
- Same parameters for all tokens
- No token-to-token interaction inside FFN
### Intermediate questions
### Advanced questions

## 6. Residual Connections + Layer Normalization
### Basic questions
#### Why are residual connections used?
#### What problem does LayerNorm solve?
##### Short answer
- Layer Normalization solves the problem of **unstable training** caused by changing activation distributions, especially in deep and sequential models, by normalizing activations within each sample/token, independent of batch size.
##### 1. Internal covariate shift (practical version)
##### 2. Batch-size dependence (what BatchNorm fails at)
- BatchNorm:
    - Depends on batch statistics
    - Breaks with:
        - Small batches
        - Batch size = 1
        - Variable-length sequences
        - Autoregressive generation
- LayerNorm:
    - Works per token
    - Same behavior during training and inference
    - No dependence on other samples
##### 3. Deep network instability
- Transformers stack many attention + FFN layers.
- Without normalization:
    - Gradients explode or vanish
    - Training diverges
    - Deeper models fail to converge
- LayerNorm:
    - Stabilizes gradient flow
    - Enables training of very deep models (100+ layers)
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
    - **Typical tasks**
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