# Transformers
## Table of Content
- [What is Transformer in NLP](#what-is-transformer-in-nlp)
- [Key Features of Transformer](#-key-features-of-transformer)
- [How the Transformer Works(Simplified Overview)](#-how-the-transformer-works-simplified-overview)
- [Real World Transofmer-Based Models](#-real-world-transformer-based-models)
- [Commonly Asked Questions](#commonly-asked-questions)
- [Interview Questions](#interview-questions)
    
## What is transformer in NLP
The Transformer is a deep learning architecture introduced in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762). It has become the foundation of modern NLP models like BERT, GPT, and RoBERTa.

It revolutionized NLP by replacing traditional RNNs and CNNs with **self-attention mechanisms**. Unlike RNNs or LSTMs, which process sequences token by token, Transformers use self-attention to analyze an entire sequence in parallel. This allows them to capture global context, making them faster and more effective for long sequences.

## ✅ Key Features of Transformer:
- **Parallelization**: Processes sequences in parallel using attention, not step-by-step like RNNs
- **Long-Range Dependency Handling**: Self-attention lets the model learn relationships between distant words
- **State-of-the-Art Performance**: Used in models like BERT, GPT, T5, achieving top results in NLP tasks

## 🧠 How the Transformer Works (Simplified Overview):
### 1. Token Embedding
Each word is converted into a dense vector using an `nn.Embedding` layer.
### 2. Positional Encoding
Since Transformers don't have sequence order by default, positional encodings (sinusoidal functions) are added to the embeddings to represent the position of each token in the sentence.
$PE(pos, i) = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)$

### 3. Self-Attention
The self-attention mechanism is the core of the Transformer architecture. It allows the model to weigh the importance of different words in a sequence when encoding each word — enabling the model to understand context from surrounding tokens.

Self-attention computes how much each word in a sentence should **attend to (focus on)** every other word — helping models understand context, dependencies, and meaning.
For a word `𝑞`, self-attention is computed as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^\top}{\sqrt{d_k}} \right)V
$$

Where:
- `q`, `k`, `v`: query, key, and value vectors
- `dₖ`: dimension of keys (for scaling)
- The result is a weighted sum of value vectors, based on how relevant each word is to the query

Intuition behind q, k v:
- Q(Query): What you’re **looking for** (e.g., “Who should I pay attention to?”)
- K(Key): What each other word offers as an identity (e.g., “What am I about?”)
- V(Value): The actual content/information each word holds (e.g., “Here’s my info”)

### 4. Multi-Head Attention
Instead of computing a single attention score, multiple attention heads capture different relationships in parallel, and their results are concatenated and projected back to the embedding space.
### 5. **Feedforward Network (FFN)**
After self-attention, each token's vector goes through a fully connected feedforward layer with non-linearity.
### 6. **Residual Connections + Layer Normalization**
Each block includes skip connections (residuals) and layer normalization to stabilize training.

#### 📚 Real-World Transformer-Based Models
| Model          | Architecture    | Use Case                   |
| -------------- | --------------- | -------------------------- |
| **BERT**       | Encoder         | Text classification, QA    |
| **GPT**        | Decoder         | Text generation, chat      |
| **T5**         | Encoder-Decoder | Translation, summarization |
| **DistilBERT** | Compressed BERT | Faster inference           |

## [Commonly Asked Questions](./CommonlyAskedQuestions.md)

## Interview Questions:

### 1. What is the purpose of the **multi-head attention mechanism** in Transformers?

Multi-head attention means using **multiple self-attention layers (heads)** in parallel. Each head learns to focus on different aspects of the input — syntax, semantics, entity relationships, etc.
Code: 
```python
    Q = W_q(x)  # [B, T, D]
    K = W_k(x)
    V = W_v(x)
```
```python
    Q = W_q(x).reshape(B, T, H, D_h).transpose(1, 2)  # [B, H, T, D_h]
    K = W_k(x).reshape(B, T, H, D_h).transpose(1, 2)
    V = W_v(x).reshape(B, T, H, D_h).transpose(1, 2)
```

### 2. Why do Q and K use different weight matrices? Why not just use the same input for dot product?

Q (query) and K (key) play different semantic roles. Using the same weights removes asymmetry and reduces representational capacity. Dot product of the same vector (e.g., Q·Qᵀ) only captures self-similarity and lacks contextual interactions.

### 3. Why does Transformer use multiplication (dot product) for attention instead of addition?

Dot product attention is computationally more efficient and parallelizable using matrix operations. Additive attention (used in RNNs) is harder to scale and slower. Dot product attention performs better with multi-head design on large datasets.

### 4. Why do we scale the dot product by √dk before softmax?

To prevent large dot product values from pushing the softmax into regions with small gradients, leading to vanishing updates. Scaling by √dk keeps the variance of the dot product stable and ensures effective learning.

### 6. Why do we need to reduce the dimensionality for each head in multi-head attention?

If we don’t reduce the dimensionality, the computation for each head will be too large.
For example, if the input embedding size is 512 and we use 8 heads, assigning each head 512 dimensions would result in an output with 4096 dimensions — which is too large.

So we usually set each head’s dimensionality to:

$$
\text{head\_dim} = \frac{d_{\text{model}}}{\text{num\_heads}}
$$

For instance, 512/8=64.
After concatenation, we get back to the original 512 dimensions.

### 13. What is BatchNorm? Advantages and disadvantages?

- **BatchNorm** normalizes each feature (each column) across a batch of inputs:
$$
\hat{x} = \frac{x - \mu_{\text{batch}}}{\sqrt{\sigma^2_{\text{batch}} + \epsilon}}
$$
- ✅ Pros:
    - Reduces internal covariate shift
    - Can accelerate the training process
- ❌ Cons:
    - Sensitive to batch size, especially unstable in NLP tasks
    - Poor at handling long sequences
    - Requires fixed statistics during inference, which affects models that need to adapt dynamically

### 14. Briefly describe the feed-forward network in Transformer. What activation function is used? What are its pros and cons?

- Each Transformer layer contains a feed-forward sublayer (FFN):
$$
\text{FFN}(x) = \text{Linear}(x) \rightarrow \text{ReLU} \rightarrow \text{Linear}(x)
$$
- **Function**: Applies a non-linear transformation independently at each position, giving the model a stronger feature transformation capability.
- **Activation function**: Usually ReLU is used, though some models use GELU.

✅ Pros and Cons:
- **Pros**: Non-linearity enhances the expressive power of the model.
- **Cons**: ReLU outputs zero in the negative range, which may cause gradient vanishing in some cases.

### 15. How do the Encoder and Decoder interact with each other?
- The Decoder contains a **cross-attention layer** that allows it to "see" the Encoder's output.
- **Inputs**:
    - **Query** comes from the Decoder itself
    - **Key** and **Value** come from the Encoder’s output
- This enables the decoder to build connections between the input and output sequences.

```python
attention_output = self.cross_attention(x, enc_output, enc_output, src_mask)
```
- `enc_output` is the output from the Encoder, serving as K and V, and is integrated into the Decoder.
- This completes the `Encoder → Decoder` information bridging interaction.

### 16. What’s the difference between encoder and decoder attention in Transformer?

| Module        | Source of Q / K / V                                                         | Type                  |
| ------------- | --------------------------------------------------------------------------- | --------------------- |
| **Encoder**   | Q, K, V all come from the encoder input (self-attention)                    | Self-Attention        |
| **Decoder-1** | Q, K, V all come from the already generated part of the decoder (with mask) | Masked Self-Attention |
| **Decoder-2** | Q comes from the decoder, K and V come from the encoder output              | Cross-Attention       |

🔹 Notes:
- The first attention sublayer in the decoder is Masked Multi-Head Self-Attention, which prevents information leakage from future tokens (auto-regressive modeling).

- The second attention sublayer in the decoder is Cross-Attention, which integrates the information encoded by the encoder.

### 19. What is the role of Dropout in Transformer? Where is it applied?
- **Dropout** is a regularization technique used to prevent overfitting.
- In Transformers, Dropout is applied at multiple locations:
    - 1. After the attention output (before the residual connection)
    - 2. After the feed-forward output
    - 3. After embedding + positional encoding
- Code implementation:
```python
self.dropout = nn.Dropout(dropout)

# Applied after embedding
src_embedded = self.dropout(self.position_encoding(self.encoder_embedding(src)))

# Applied after attention and feed-forward
x = self.norm1(x + self.dropout(attention_output))
x = self.norm2(x + self.dropout(ff_output))
```