# Commonly Asked Questions
## Table Of Contents
- [Input and Embedding Layer](#1-input--embedding-layer-token--positional-encoding)
    - [Basic Questions](#basic-questions)
        - [What is Token Embedding](#what-is-token-embedding)
        - [Why do we Need Positional Encoding in Transformers](#why-do-we-need-positional-encoding-in-transformers)
        - [What Happens is We Remove Positional Encoding](#what-happens-if-we-remove-positional-encoding)
    - [Intermediate Questions](#intermediate-questions)
        - [Why Can't Self-attention Alone Capture Word Order](#why-cant-self-attention-alone-capture-word-order)
        - [What's the Difference Between Sinusoidal and Learned Positional Embeddings](#whats-the-difference-between-sinusoidal-and-learned-positional-embeddings)
    - [Advanced Questions](#advanced-questions)
        - [How does RoPE (Rotary Positional Embedding) differ from absolute positional encoding?](#how-does-rope-rotary-positional-embedding-differ-from-absolute-positional-encoding)
        - [Why Do modern LLMs prefer relative position encodings?](#why-do-modern-llms-prefer-relative-position-encodings)
        - [How Does Positional Encoding Affect Extrapolation to Longer Sequences](#how-does-positional-encoding-affect-extrapolation-to-longer-sequences)
- [Self-Attention](#2-self-attention)
    - [Basic Questions](#basic-questions-1)
        - [What is Self-Attention](#what-is-self-attention)
        - [What are Query, Key, and Value](#what-are-query-key-and-value)
    - [Intermediate Question](#intermediate-questions-1)
        - [Why Do We divide by √dₖ in scaled dot-product attention?](#why-do-we-divide-by-dₖ-in-scaled-dot-product-attention)
- [Multi-Head Attention](#3-multi-head-attention)
    - [Basic Questions](#basic-questions-2)
        - [Why Do WE Use Multiple Attention Heads](#why-do-we-use-multiple-attention-heads)
    - [Intermediate Questions](#intermediate-questions-2)
    - [Advanced Question](#advanced-questions-2)
- [Attention Masking(Encoder vs Decoder)](#4-attention-masking-encoder-vs-decoder)
    - [Basic Questions](#basic-questions-3)
    - [Intermediate Questions](#intermediate-questions)
- [Feed-Forward Network (FFN)](#5-feed-forward-network-ffn)
    - [Basic Questions](#basic-questions-4)
        - [What is the Role of the Feed-Forward Network](#what-is-the-role-of-the-feed-forward-network)
- [Residual Connections and Layer Normalization](#6-residual-connections--layer-normalization)
    - [Basic Questions](#basic-questions-5)
        - [What Problem does LayerNorm Solve](#what-problem-does-layernorm-solve)
    - [Intermediate Question](#intermediate-questions-5)
- [Encoder vs Decoder(Architecture Level)](#7-encoder-vs-decoder-architecture-level)
    - [Basic Questions](#basic-questions-6)
        - [What's the Difference between Encoder and Decoder?](#whats-the-difference-between-encoder-and-decoder)
        - [Which Models use Encoder only vs Decoder Only?](#which-models-use-encoder-only-vs-decoder-only)
    - [Intermediate Questions](#intermediate-questions-6)
        - [Why is GPT Decoder Only](#why-is-gpt-decoder-only)
        - [Why does Translation Use Encoder-Decoder](#why-does-translation-use-encoderdecoder)
- [Output Layer & Training Objective](#8-output-layer--training-objective)
    - [Basic Questions](#basic-questions-7)
    - [Intermediate Question](#intermediate-questions-7)
    - [Advanced Question](#advanced-questions-7)
## 1. Input & Embedding Layer (Token + Positional Encoding)
### Basic questions
#### What is token embedding?
- **Short Answer**
    - **Token embedding** is the process of mapping discrete tokens (words, subwords, or symbols) into **dense**, **continuous vectors** that capture semantic meaning and can be processed by neural networks.
- **Why token embedding is needed**
    - Neural networks cannot work directly with text. Tokens like:
    - must be converted into numbers that:
        - Preserve semantic similarity
        - Are differentiable
        - Can be learned end-to-end
    - Token embeddings provide this representation.
#### Why do we need positional encoding in Transformers?
- **Short Answer**
    - We need **positional encoding** because the Transformer’s self-attention mechanism is order-agnostic. 
    - Without positional information, the model would treat a sentence as a bag of tokens and would not know **which word comes first, last, or next**.
- **Detailed Answer**
1. Self-attention has no notion of order
2. Positional encoding injects sequence order
3. Why positional encoding is added, not concatenated
#### What happens if we remove positional encoding?

### Intermediate questions
#### Why can’t self-attention alone capture word order?
#### What’s the difference between sinusoidal and learned positional embeddings?
#### Why are positional embeddings added, not concatenated?

### Advanced Questions
#### How does RoPE (Rotary Positional Embedding) differ from absolute positional encoding?
#### Why do modern LLMs prefer relative position encodings?
#### How does positional encoding affect extrapolation to longer sequences?

## 2. Self-Attention
### Basic questions
#### What is self-attention?
- **Short answer**
    - **Self-attention** is a mechanism that lets each token in a sequence **dynamically focus on other tokens in the same sequence** to build a context-aware representation.
- **What self-attention does (intuition)**
    - When processing a sentence, each token asks:
        - “Which other tokens are relevant to me, and how much should I care about them?”
    - Self-attention computes those relevance weights and uses them to combine information from the whole sequence.
- **How self-attention works (step-by-step)**
    - 1. Project tokens into Q, K, V
    - 2. Compute similarity scores
    - 3. Scale + softmax
    - 4. Weighted sum of values
- **Why self-attention is powerful**
    - 1. Captures long-range dependencies
        - Any token can attend to any other token in one step
        - No recurrence needed
    - 2. Fully parallelizable
        - All tokens processed at once
        - Much faster than RNNs during training
    - 3. Dynamic & data-dependent
        - Attention weights change depending on context
        - Same word behaves differently in different sentences
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
- **Short answer (interview-ready)**:
    - We divide by **√dₖ** to keep dot-product magnitudes **numerically stable**. Without this scaling, attention scores grow with dimension, pushing softmax into saturation, which causes **vanishing gradients** and unstable training.

- **What’s going wrong without scaling?**
    - In dot-product attention, scores are:

        $$\text{score} = Q K^{\top}$$

    - If the components of 𝑄 and 𝐾 have zero mean and unit variance, then:
        
        $$\mathbb{E}[QK^{\top}] \propto d_k$$

        - Larger 𝑑𝑘 ⇒ larger variance of scores

- Why large scores are bad (softmax saturation)

#### What does the softmax do in attention?
#### How is attention different from convolution or RNNs?
### Advanced questions
#### What happens if we remove the scaling factor √dₖ?
#### Why does attention have O(n²) complexity?
#### How does attention help with long-range dependencies?

## 3. Multi-Head Attention
### Basic questions
#### Why do we use multiple attention heads?
- **Short Answer**
    - We use multiple attention heads so the model can attend to different types of relationships in parallel, increasing expressive power and allowing the Transformer to capture diverse patterns at different subspaces of the representation.
- **Intuition first**
    - Think of each attention head as a different lens:
        - One head focuses on **syntax** (e.g., subject–verb)
        - Another on **coreference** (pronouns)
        - Another on **long-range dependencies**
        - Another on **local context**
    - Using one head would force all of this into a single view.
- **How multi-head attention works**
    - Instead of one big attention:
        - $$\mathrm{Attention}(Q, K, V)$$
    - We use ℎ heads:
        - $$\text{head}_i=\mathrm{Attention}\!\left(Q W_i^{Q},\; K W_i^{K},\; V W_i^{V}\right)$$
    - Then:
        - $$\mathrm{MultiHead}(Q, K, V)=\mathrm{Concat}\!\left(\text{head}_1, \ldots, \text{head}_h\right) W^{O}$$
- **Why not just one big head?**
    - **1. Expressiveness**
        - Multiple heads = multiple learned similarity spaces
        - One head collapses all relationships into a single pattern
    - **2. Parallel relationship modeling**
        - Heads specialize independently
        - Model can reason about multiple aspects **at the same time**
    - **3. Better inductive bias**
        - Forces diversity in attention
        - Encourages disentangled representations

#### What does each head learn differently?
### Intermediate questions
### Advanced Questions
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
- **Short Answer**
    - The **feed-forward network (FFN)** provides **non-linear transformation** and **feature mixing at each token**, allowing the Transformer to increase representational capacity beyond attention by independently transforming each token’s features.
-  What the FFN is in a Transformer
    - Inside every Transformer block, after self-attention, there is a **position-wise feed-forward network**:

    $$\mathrm{FFN}(x) = W_2 \, \sigma\!\left(W_1 x + b_1\right) + b_2$$

    - Same FFN is applied to **every token**
    - Operates **independently per position**
    - Usually much **wider** than the model dimension
- Why attention alone is not enough
- What the FFN actually does (intuition)
    - Think of a Transformer block as:
        - **Attention** → “Which tokens should I look at?”
        - **FFN** → “How should I process what I’ve gathered?”
    - Attention decides **where** to get information.
    - FFN decides **how** to transform it.
- Why the FFN is “position-wise”
    - Same parameters for all tokens
    - No token-to-token interaction inside FFN
### Intermediate questions
### Advanced questions

## 6. Residual Connections + Layer Normalization
### Basic questions
#### Why are residual connections used?
#### What problem does LayerNorm solve?
- **Short answer**:
    - Layer Normalization solves the problem of **unstable training** caused by changing activation distributions, especially in deep and sequential models, by normalizing activations within each sample/token, independent of batch size.
1. **Internal covariate shift (practical version)**
- As a network trains:
    - Earlier layers keep changing
    - Later layers constantly receive inputs with **shifting distributions**
    - This makes optimization slow and unstable
- LayerNorm **re-centers** and **re-scales** activations at every layer, keeping them in a stable range.
2. **Batch-size dependence (what BatchNorm fails at)**
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
3. **Deep network instability**
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
- **Short answer**:
    - Transformers use Layer Normalization instead of Batch Normalization because LayerNorm is **independent of batch size** and token position, making it stable for variable-length sequences, small or dynamic batches, and autoregressive generation.

- **Core reason (mechanism-level)**
    - BatchNorm normalizes across the batch
        - Computes mean/variance **over batch dimension**
        - This breaks in NLP because:
            - Batch sizes are often **small or variable**
            - Sequences have **different lengths**
            - Autoregressive decoding often uses **batch size = 1**
    - LayerNorm normalizes within each token
        - LayerNorm computes statistics across the feature dimension for each token independently:
        
            $$\mu = \frac{1}{d} \sum_{i=1}^{d} x_i,\quad\sigma = \sqrt{\frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2}$$

        - Why this works for Transformers:
            - No dependence on batch size
- **What breaks if we use BatchNorm in Transformers?**
    - 1. Training–inference mismatch
        - BatchNorm uses batch statistics during training
        - Uses running averages during inference
        - Autoregressive generation has different statistics → instability
    - 2. Batch size sensitivity
    - 3. Sequence length variability
    - 4. Hard to parallelize across time
        - BN assumes spatial consistency (good for CNNs, bad for sequences)
- **Transformer-specific reasons (important)**
    - Self-attention outputs vary **per token**
    - Tokens in the same batch are **not aligned semantically**
    - LayerNorm treats each token independently → perfect match
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
- **Short Answer**
    - The encoder reads and understands an entire input sequence using bidirectional attention, while the decoder generates an output sequence token-by-token using causal (masked) attention.
- **Encoder**
    - **What the encoder does**
        - Takes the **full input sequence at once**
        - Builds **contextual representations** of every token
        - Uses **bidirectional self-attention**
    - **Encoder block components**
        - 1. Self-attention (no causal mask)
        - 2. Feed-forward network
        - 3. Residual connections + LayerNorm
    - **Key properties**:
        - Full context available
        - 
    - **Example models**:
        - BERT
        - RoBERTa
        - DistilBERT
- **Decoder**
    - **What the decoder does**
        - Generates text **one token at a time**
        - Uses **causal (look-ahead) masking**
        - Can only attend to **past tokens**
    - Decoder block components
        - 1. Masked self-attention
        - 2. (Optional) cross-attention to encoder outputs
        - 3. Feed-forward network
        - 4. Residual connections + LayerNorm
    
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
        - Trained autoregressively (next-token prediction)

    - **Common decoder-only models**
        - GPT (GPT-2 / GPT-3 / GPT-4)
        - LLaMA
        - Mistral
        - Falcon
        - DeepSeek
    - **Typical tasks**
        - Text generation
        - Chatbots
        - Code generation
        - Reasoning
        - Autocomplete
    - **Why decoder-only dominates LLMs**
        - Simple architecture
        - Scales extremely well
        - Unified framework: prompt → generate
        - Perfect for instruction tuning & chat
- **Encoder–decoder models (for completeness)**
    - **What they are**
        - Encoder reads input
        - Decoder generates output
        - Cross-attention connects them
### Intermediate Questions
#### Why is GPT decoder-only?
#### Why does translation use encoder–decoder?
- **Short answer**:
    - Translation uses an **encoder–decoder** architecture because the model must first **fully understand the source sentence** (encoder) and then generate a target sentence of different length and structure (decoder), conditioning every output token on the encoded source.
- **The core reason (intuition)**
    - Translation is not a one-to-one mapping:
        - Word order changes
        - Sentence lengths differ
        - Grammar and morphology differ
    - So the model needs:
        - A **global understanding** of the entire source sentence
        - A **step-by-step generation process** for the target language
- **What each part does (mechanically)**
    - Encoder (understanding)
        - Reads the entire source sentence
        - Uses **bidirectional attention**
        - Produces contextual representations for all source tokens
    - Decoder (generation)
        - Generates target tokens **autoregressively**
        - Uses **causal self-attention**
        - Uses **cross-attention** to focus on relevant source tokens
        - At each step, the decoder asks:
            - “Which parts of the source sentence are relevant for generating the next word?”
        - That’s exactly what cross-attention enables.
- **Why decoder-only is not ideal for translation**
    - A decoder-only model:
        - Treats translation as one long sequence
        - Must interleave understanding + generation
        - Scales poorly for long or complex inputs
        - Struggles with alignment between languages
    - Encoder–decoder:
        - Clean alignment via cross-attention
        - Better handling of reordering
        - More stable and interpretable for seq-to-seq tasks

### Advanced Questions
## 8. Output Layer & Training Objective
### Basic Questions
### Intermediate Questions
### Advanced Questions