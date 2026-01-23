# Commonly Asked Questions
## Table Of Contents
- [Input and EMbedding Layer](#1-input--embedding-layer-token--positional-encoding)
    - [Basic Questions](#basic-questions)
        - [What is Token Embedding](#what-is-token-embedding)
        - [Why do we Need Positional Encoding in Transformers](#why-do-we-need-positional-encoding-in-transformers)
        - [What Happens is We Remove Positional Encoding](#what-happens-if-we-remove-positional-encoding)
- [Self-Attention](#2-self-attention)
- [Multi-Head Attention](#3-multi-head-attention)
- [Attention Masking(Encoder vs Decoder)](#4-attention-masking-encoder-vs-decoder)
    - [Basic Questions](#basic-questions)
    - [Intermediate Questions](#intermediate-questions)
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
### What is self-attention?
#### What are Query, Key, and Value?
- Short Answer:
    - Query, Key, and Value are three learned vector representations of each token that allow the Transformer to compute attention weights—that is, how much each token should focus on other tokens when forming its contextual representation.
- **Formal explanation (how it actually works)**
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
    
### Intermediate questions
### Advanced questions
## 5. Feed-Forward Network (FFN)
