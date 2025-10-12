# LLM DEMO
## Table Of Contents
- [Concepts](#concepts)
    - [Basic Concepts](#1-basic-concepts)
        - [Token](#11-token)
        - [Parameters](#12-parameters)
        - [Embeddings](#13-embeddings)
        - [Transformer Architecture](#14-transformer-architecture)
        - [Fine Tuning](#15-fine-tuning)
    - [Common pre-training objectives for LLM](#3-what-are-some-common-pre-training-objectives-for-llms-and-how-do-they-work)
    - [Fine-Tuning](#4-fine-tuning)
        - [What is Fine-Tuning](#41-what-is-fine-tuning)
        - [Why Fine-Tuning Works](#-42-why-fine-tuning-works)
        - [Types of Fine-Tuning](#️-43-types-of-fine-tuning)
    - [How can bias in prompt-based learning be mitigated?](#9-how-can-bias-in-prompt-based-learning-be-mitigated)
    - [Catastrophic Forgetting](#10-catastrophic-forgetting)
    - [LoRA](#13-lora)
    - [PEFT](#14-peft)
- [Setup](#setup)
## Concepts
### 1. Basic Concepts
#### 1.1 Token
- **Definition**: A token is the smallest unit of text the model processes — usually a word, subword, or symbol.
- **For Example**:
    - “I love cats” → `[I] [love] [cats]` (word-level tokenization)
    - “unbelievable” → `[un] [believ] [able]` (subword tokenization)
- **Why it matters**:
    - The model’s input and output lengths are measured in tokens, not characters or words.
    - LLM pricing, context length, and speed all depend on token count.
#### 1.2 Parameters
- **Definition**: The parameters are the weights inside the neural network that the model learns during training.
They define how the model transforms input tokens into contextual representations.
- **Example**:
    - GPT-3 → 175 billion parameters
    - BERT → 340 million parameters
    - More parameters → greater capacity to model complex relationships.
#### 1.3 Embeddings
- **Definition**: 
    - **Embeddings** are high-dimensional vector representations of words, sentences, or documents that capture **semantic meaning**.
    - Words with similar meanings (e.g., “happy” and “joyful”) are close together in embedding space.
- **Used for**:
    - Semantic search
    - Text similarity
    - Retrieval-Augmented Generation (RAG)

#### 1.4 Transformer Architecture
- **Definition**:
    - The **Transformer** is the backbone of modern LLMs. It uses self-attention to model relationships between all tokens in a sequence simultaneously.
- **Key Components**:
    - **Encoder**: Reads and understands context (used in BERT, T5).
    - **Decoder**: Generates text autoregressively (used in GPT).
    - **Encoder-Decoder**: Both read and generate (used in T5, BART).
#### 1.5 Fine-Tuning
- **Definition**:
    - Fine-tuning is the process of **adapting a pre-trained model** (e.g., GPT or BERT) to a specific domain or task by continuing its training on a smaller, focused dataset.
- **Purpose**:
    - Improves model performance for specific goals like sentiment analysis, summarization, or domain adaptation (e.g., legal or medical texts).
#### 1.6 Perplexity
- **Definition**:
    - Perplexity measures **how well a language model predicts text**.
    - It’s the exponential of the average negative log-likelihood of the predicted tokens.
- **Formula**:
    - Perplexity=e^Loss
- **Interpretation**:
    - Low perplexity → confident and accurate predictions.
    - High perplexity → model is “surprised” by the actual text.
#### 1.7 Accuracy
- **Definition**:
    - The proportion of correct predictions out of all predictions.
    - Often used in classification tasks (e.g., sentiment analysis).
- **Formula**:

#### 1.8 F1 Score
- **Definition**:
    - Combines **precision** and **recall** into a single metric for evaluating classification performance.
    - Useful when data is imbalanced.
- **Formula**:
    - F1=2×(Precision*Recall)/(Precision+Recall)
    - ​
#### 1.9 Recall
- **Definition**:
    - Recall measures how well the model identifies all relevant instances from the data.
    - It’s the proportion of actual positives that the model correctly predicts as positive.
- **Formula**:
    - Recall=(True Positives + False Negatives)/True Positives​
- 
#### 1.10 BLEU (Bilingual Evaluation Understudy)
- **Definition**:
    - BLEU is a **text generation quality metric**, originally for machine translation.
    - It measures **n-gram overlap** between model-generated text and reference text.
- **Interpretation**:
    - BLEU = 1 (or 100) → perfect match with reference.
    - BLEU ≈ 0 → little to no overlap.
- **Used for**:
    - Translation, summarization, dialogue systems.

#### 1.11 ROUGE
- **Definition**:
    - ROUGE (Recall-Oriented Understudy for Gisting Evaluation) evaluates **how much of the reference text is captured** in the generated text.
- **Types**:
    - ROUGE-1 → unigram overlap
    - ROUGE-2 → bigram overlap
    - ROUGE-L → longest common subsequence
- **Used for**:
    - Summarization and paraphrasing evaluation.
#### 1.12 Prompt
- **Definition**:
    - A **prompt** is the input text or instruction given to the LLM to guide its output.
    - The quality and structure of the prompt significantly affect model performance.
- ****:
#### 1.13 Hyperparameters
- **Definition**:
    - 
### 3. What are some common pre-training objectives for LLMs, and how do they work?
#### 3.1  Masked Language Modeling (MLM)
- **Used in models like**: BERT, RoBERTa
- **How it works**:
    - Random tokens in a sentence are masked (replaced with `[MASK]`).
    - The model is trained to predict the masked word using both left and right context (i.e., it's **bidirectional**).
#### 3.2 Autoregressive Language Modeling (AR)
- **Used in models like**: GPT, GPT-2, GPT-3, GPT-4

### 4. 📌 Fine-Tuning
#### 4.1 What Is Fine-Tuning?
- **Fine-tuning** is the process of taking a **pre-trained** language model (like GPT, BERT, or T5) and training it further on a **smaller**, **domain-specific** dataset to make it perform better on a **specific task or language style**.
- A pre-trained model has already learned:
    - grammar, syntax, and general world knowledge
    - context relationships between words and phrases
    - reasoning patterns and text structure
- However, it doesn’t yet “know” how to handle **specialized tasks**, like:
    - classifying sentiment (e.g., positive/negative reviews),
    - generating medical summaries,
    - extracting entities from legal documents,
    - or answering customer queries in a specific tone.
- Fine-tuning adapts this general knowledge to **task-specific objectives**.

#### 🧠 4.2 Why Fine-Tuning Works
- When a model like DistilBERT is pre-trained:
    - It learns general knowledge of language patterns.
    - But it doesn’t know how to perform **task-specific** jobs like classifying IMDb reviews as positive or negative.

#### ⚙️ 4.3 Types of Fine-Tuning
- **Full Fine-Tuning**
    - The **entire model’s parameters** are updated on the new dataset.
    - Pros:
        - Maximum flexibility and task adaptation.
    - Cons:
        - Requires large compute resources (GPUs/TPUs).
        - Risk of **catastrophic forgetting** (losing general knowledge).

- **Parameter-Efficient Fine-Tuning (PEFT)**
#### 🔍 4.4 Fine-Tuning Workflow
1. **Start from a pre-trained base model (e.g., `bert-base-uncased`, `gpt-3.5-turbo`).**
2. **Prepare your dataset:**
- Input–output pairs, labeled text, or conversation data.
- Split into train/validation sets.
3. Choose the fine-tuning method:
- Full fine-tuning, PEFT, or instruction tuning.
4. **Train the model:**
- Define hyperparameters (learning rate, epochs, batch size).
- Use frameworks like Hugging Face Transformers or OpenAI Fine-tuning API.
5. Evaluate:
- Metrics: accuracy, F1 score, BLEU, or perplexity (depending on the task).

### 5. **How do you measure the performance of an LLM?**
#### 5.1 🔢 Perplexity
- **definition**:
Perplexity is a measurement of how well a language model predicts a sequence. It is the exponential of the average negative log-likelihood of the predicted tokens.
- **formula**:
Perplexity = *e*^Loss
- **interpretation**:

**Low perplexity** → Model is confident and accurate in predicting the next tokens.
**High perplexity** → Model is "surprised" by the actual tokens.
#### 5.2 🧮 Accuracy
- **definition**:
Accuracy is the ratio of **correct predictions to total predictions**. It is commonly used in classification tasks (e.g., sentiment analysis, text classification).
- **formula**:
Accuracy = Number of Correct Predictions / Total Predictions
#### 5.3 F1 Score
- **definition**:
F1 Score is the **harmonic mean** of Precision and Recall. It is especially useful for imbalanced datasets.

- **Precision** = How many of the predicted positives are correct?

- **Recall** = How many of the actual positives did the model catch?

- **formula**:
F1=(2*Precision*Recall)/(Precision+Recall)

#### 5.4

### 6. **Techniques for Controlling the Output of an LLM**
These methods let developers influence how a model responds, balancing between randomness, relevance, creativity, and determinism.
#### 6.1 🔥 Temperature
##### **What it does:** 
Controls the level of randomness in token selection.
##### **How it works:** 
During generation, the model uses probabilities to decide the next token. Temperature scales these probabilities:
- A **lower value** (e.g., 0.2) sharpens the distribution — the model is more confident and **chooses the most likely next word**, producing **deterministic and repetitive** outputs.
- A **higher value** (e.g., 1.0 or 1.5) flattens the distribution, allowing for more **diverse, creative, and unpredictable** text.

##### 🧊 Low Temperature (temperature=0.2)
- Explanation:
    - The output is **coherent**, **rhythmic**, and **safe**.
    - GPT-4 chooses tokens with the highest probability, so it sticks to standard poetic themes.
    - Less creative surprises, but more syntactically correct and “professional” sounding.

- ✅ Ideal for:
    - Factual tasks
    - Formal documentation
    - Summarization
    
#####  High Temperature (temperature=1.0)
- Explanation:
    - The output is **more imaginative and colorful**.
    - Words like "Emerald galaxies", "ink-black canvas" indicate a **creative leap**.

#### 6.2 🎯 Top-K Sampling
- **What it does**: Restricts the token selection pool to the **top K most probable tokens** at each generation step.
- **How it works**: If `top_k=50`, the model only chooses from the top 50 most likely next tokens rather than considering all options.


#### 6.3 Top-p Sampling
##### 🔍 What Is Top-P Sampling?
Top-P sampling chooses from the smallest set of tokens whose cumulative probability exceeds the threshold p. Lower values restrict choice to high-confidence tokens; higher values allow more diverse token selection.

##### Explanation in Example
- 0.3:
    - **Summary**: Output is short and nearly identical to 0.6; it stops mid-sentence.
    - **Behavior**: Most focused — selects tokens only from the top ~30% cumulative probability mass. Tends to be **highly relevant but less diverse**.
- 0.6:
    - **Summary**: Nearly identical to 0.3.
    - **Behavior**: Balanced — more flexible than 0.3 but still somewhat focused, but still constrained to safe outputs.
- 0.8 
    - **Summary**: Output starts to diversify — adds some background explanation.
### 7. 

### 8. How can you incorporate external knowledge into an LLM?
####  Knowledge Graph Integration (Simplified)
Use structured facts (triples or graphs `like France → Capital → Paris`) directly in the prompt. This adds factual grounding to help the model reason accurately.

#### RAG
##### **Retrieval-Augmented Generation (RAG)** is a hybrid approach that:

1. Retrieves relevant documents from a large external knowledge base or corpus (like Wikipedia, PDFs, or internal files),

2. Augments the prompt by inserting the retrieved text,

3. Generates the answer using a generative language model (like GPT, BART, or T5).

##### 🧠 How it works (step-by-step):
1. User inputs a query
    - → e.g., “What are the benefits of vitamin D?”
2. 



### 9. How can bias in prompt-based learning be mitigated?
#### 1. Prompt Calibration
- This involves carefully designing and testing prompts so that the LLM produces balanced, unbiased responses.
- For example, if a model tends to associate certain professions with specific genders, you can test multiple prompt formulations and adjust phrasing to reduce bias.
- **Example**:
    - Uncalibrated: “The nurse said he…” → likely produces bias.
    - Calibrated: “A person working as a nurse said…” → reduces gender association.

#### 2. Fine-Tuning
- Fine-tuning retrains a pre-trained model on **diverse and balanced datasets**.
- This process teaches the model to correct its biased patterns learned during pretraining.

#### 3. Data Augmentation
- This expands your dataset with **synthetic or mirrored examples** that counteract bias.
- For example:
    - If 70% of your data says “doctor → he,” generate more examples with “doctor → she.”
    - Use paraphrasing or back-translation to diversify data linguistically.

### 10. catastrophic forgetting
#### Definition:
- Catastrophic forgetting (or catastrophic interference) is the phenomenon where a neural network **forgets previously learned tasks** after being fine-tuned on new data.
- In the context of LLMs, it means:
    - When you fine-tune a model (like GPT, BERT, or T5) on a new dataset or task, its performance on older tasks suddenly drops dramatically.

#### ⚙️ Why It Happens (Mechanism):
1. **Shared Parameters**
- In deep neural networks, the same weights are used across many tasks.
- When fine-tuning, backpropagation updates these shared parameters to fit the new task.
2. **No Replay Memory**:
- Unlike humans, models don’t “remember” earlier tasks unless we retrain them together.
- They only see the new task’s dataset — and gradients push them entirely toward that new distribution.
3. **High Capacity Models Still Forget**:
- Even very large LLMs (billions of parameters) are not immune.
- Their large capacity helps, but without constraints or regularization, they still optimize for the current objective and drift away from older ones.
#### 🧩 Mitigation Techniques
| **Technique**                                  | **How It Works**                                                                        | **Why It Helps**                                                                |
| ------------------------------------------ | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **PEFT (Parameter-Efficient Fine-Tuning)** | Freezes most weights and trains small adapter modules (like LoRA or prefix tuning). | Preserves old knowledge in frozen weights.                                  |
| **EWC (Elastic Weight Consolidation)**     | Penalizes changes to parameters that are important for old tasks.                   | Uses Fisher Information Matrix to identify which parameters are “critical.” |
| **Replay / Rehearsal**                     | Mixes data from old and new tasks during fine-tuning.                               | Helps maintain representation balance.                                      |
| **Regularization Methods**                 | Adds penalty terms that discourage large weight shifts.                             | Keeps parameters near their old values.                                     |

#### 🧮 Intuitive Analogy
- Think of the model’s parameters as a **shared whiteboard**:
    - During pretraining, it writes general knowledge.
    - During fine-tuning, it writes notes for new tasks.
- If you erase and overwrite everything for the new topic (without saving the old ones), you lose the old knowledge — that’s catastrophic forgetting.
- Techniques like PEFT or EWC act like:
    - **PEFT**: “Write on sticky notes” (small, new parameters) — don’t touch the main whiteboard.
    - **EWC**: “Highlight what’s important and don’t erase it” — preserve critical parts of the old notes.

### 11. LoRA
#### What is Low-Rank Adaptation (LoRA)?

**Low-Rank Adaptation (LoRA)** is a **parameter-efficient fine-tuning (PEFT)** technique designed to adapt large pre-trained models for specific tasks **without significantly increasing computational or memory costs**.

As large language models (LLMs) grow in size and complexity, fine-tuning them on new tasks often requires **substantial computational power and GPU memory**.  
LoRA solves this problem by reducing the number of trainable parameters — making the fine-tuning process **faster, lighter, and more efficient**.

---

#### 🧠 Key Idea

LoRA modifies the standard fine-tuning process by **inserting small trainable low-rank matrices** into specific layers (typically the attention projections) of a frozen pre-trained model.  
Instead of updating the full parameter matrix \( W \), LoRA decomposes it into two smaller matrices \( A \) and \( B \):

\[
W' = W + A \cdot B
\]

- \( W \): Original frozen weight matrix  
- \( A \): Low-rank matrix of size \( d \times r \)  
- \( B \): Low-rank matrix of size \( r \times d \)  
- \( r \): Rank (typically much smaller than \( d \))

Only \( A \) and \( B \) are trained, while \( W \) remains frozen — significantly reducing computational overhead.

---

#### ⚙️ Architecture of LoRA

- LoRA is typically integrated into **Transformer-based models** (like GPT, BERT, or T5).  
- Here’s how it works step by step:

1. **Pre-Trained Backbone**  
   - Begin with a large transformer model that has already been trained on massive general-purpose data.

2. **Low-Rank Adaptation Layers**  
   - Insert small, trainable low-rank matrices \( A \) and \( B \) into specific attention projection layers (e.g., query or value matrices).  
   - These are the *only* parameters that get updated during fine-tuning.

3. **Frozen Original Parameters**  
   - The original model weights remain **frozen**.  
   - This ensures that general language knowledge is preserved and prevents **catastrophic forgetting**.

4. **Task-Specific Fine-Tuning**  
   - Fine-tune only the low-rank matrices for a specific task (like sentiment analysis or translation).  
   - The model learns the new task efficiently while maintaining previous capabilities.

### 12. PEFT
#### What is PEFT?
- **Parameter-Efficient Fine-Tuning (PEFT)** adapts a frozen pretrained model by training only a small set of extra parameters (or a tiny subset of existing ones). The backbone weights stay fixed, so you keep the general knowledge while learning a new task/domain cheaply.
#### Major PEFT families (how they plug in)
##### **LoRA (Low-Rank Adapters)**
Learn two small matrices \( A \in \mathbb{R}^{d \times r} \), \( B \in \mathbb{R}^{r \times d} \) and add their product to a frozen weight \( W \):

\[
W' = W + \alpha \cdot A B
\]

Usually applied to attention projections (**q/v**).  
Only \( A, B \) train (rank \( r \ll d \)).

---

##### **Adapters (Bottleneck Blocks)**
Insert a tiny MLP after (or inside) Transformer sublayers:

\[
h \mapsto h + W_\text{up} \, \sigma(W_\text{down} \, \text{LN}(h))
\]

Initialize near identity so the model starts as the base model; only adapter weights train.

---

##### **Prefix / Prompt / P-Tuning**
Learn a small set of **virtual tokens** (or key/value *prefixes*) prepended per layer or sequence — only these embeddings are trainable.

---

##### **IA³ / Gating / BitFit**
Learn per-channel scaling vectors (**IA³**) or just biases (**BitFit**).  
Extremely small parameter count.
#### Why PEFT prevents catastrophic forgetting
- **Catastrophic forgetting** happens when you update the shared backbone and overwrite features needed for older tasks. PEFT avoids that by design:
    - **Parameter isolation**:
        - The backbone is frozen. New knowledge lives in the tiny trainable pieces (LoRA `𝐴`,`𝐵`, adapter layers, prefixes). Old capabilities aren’t overwritten because their weights never change.
    - **Identity initialization**:
        - Adapters/LoRA start as (near) identity/zero-update, so training nudges behavior locally instead of globally rewriting representations.
    - **Low-rank / low-capacity updates**
        - Constraining updates (e.g., low rank 𝑟) regularizes changes; you can’t drastically deform the function even if you try.
    - **Task modularity**:
        - You can **keep one adapter per task**. Switching tasks is swapping small modules—no retraining, no interference. (If you fine-tune Task B, Task A’s adapter is untouched.)
    - **Reversibility**:
        - With LoRA you can “merge” or simply **detach** the adapters; the original backbone remains intact on disk.
#### When PEFT might not be enough
- Huge domain shift or very complex tasks → increase LoRA rank / adapter width, or fall back to partial/full fine-tuning.
- If you keep updating the **same** adapter sequentially across tasks, you can still forget—use separate adapters or multi-task training.

### 13. Vector Store Use Case
#### 🧠 Detailed Explanation
- A **vector store** (or **vector database**) stores embeddings — numerical representations of text that capture semantic meaning rather than literal words.
- This allows the model to **search by meaning** (semantic similarity) instead of by exact keyword matches.
#### When You Need a Vector Store
- Vector stores are essential when your LLM must **retrieve external knowledge** to ground its responses.
- Examples include:
    - **Document Retrieval / Question Answering**
    - **Chat with Documents / PDFs / Knowledge Base**
    - **Retrieval-Augmented Generation (RAG) systems**
- **Reason**:
    - LLMs have limited context windows and can’t remember all your documents.
    - A vector store allows dynamic retrieval of relevant text based on embeddings created by models like text-embedding-3-small.

#### When You Don’t Need a Vector Store
- Tasks like:
    - **Text summarization**
    - **Translation**
    - **Paraphrasing**
    - **Sentiment classification**
    - **Simple conversation flows**

#### ⚖️ Summary Table
| Task Type               | Requires Vector Store? | Why                            |
| ----------------------- | ---------------------- | ------------------------------ |
| Document Q&A / RAG      | ✅ Yes                 | Needs semantic retrieval       |
| Knowledge-grounded chat | ✅ Yes                 | Pulls facts from stored data   |
| Summarization           | ❌ No                  | Uses text directly             |
| Translation             | ❌ No                  | Pure sequence-to-sequence task |
| Sentiment analysis      | ❌ No                  | Only depends on input text     |
    
### 14. MoE
#### 🧠 What Is Mixture of Experts (MoE)?
- **Mixture of Experts (MoE)** is an advanced neural network architecture designed to make large models more efficient and scalable by activating only a subset of the model’s parameters for each input, instead of using the entire model every time.
- In traditional dense models (like GPT-3 or BERT), **all parameters are active** for every input token.
In contrast, MoE distributes the workload across multiple smaller subnetworks — called experts — and selectively activates only the most relevant ones.

#### 🔄How It Works (Step-by-Step)
1. **Input arrives** (e.g., a token embedding or hidden state).
2. The **gating network** analyzes the input and assigns weights to each expert (e.g., “Expert 3: 0.8, Expert 7: 0.6, others: near 0”).
3. Only the **top-k experts** (usually 1–2) are activated to process this input.
4. Their outputs are combined (weighted sum) and passed to the next layer.

#### ⚡ Why MoE Improves Efficiency
1. **Sparse Activation**:
- Only a fraction (e.g., 10–20%) of parameters are used per token → less computation.
2. **Scalability**:
- You can scale up total parameters (e.g., to 1 trillion) while keeping runtime cost close to a smaller dense model.
3. **Specialization**:
- Experts learn to handle specific kinds of data — e.g., “mathematical reasoning,” “dialogue tone,” or “code generation.”
4. **Parallelization**:
- Different experts can run on different hardware shards or GPUs.
### 15. Adapter Tuning
#### 15.1 Background
- As pre-trained models grow larger and larger, fine-tuning all parameters for each downstream task becomes both expensive and time-consuming.
- To address this, the authors proposed **Adapter Tuning** — a technique that inserts adapter layers into pre-trained models. These adapters contain a small number of task-specific parameters (about 3.6% of the full model size).
- During fine-tuning, the **original model parameters remain frozen**, and only the adapter layers are trained for the downstream task. This significantly reduces computational cost.
#### 15.2 Technical Principle
- **Adapter Tuning** (from the paper Parameter-Efficient Transfer Learning for NLP) introduces an **adapter structure** into each Transformer layer.
- Specifically, two adapter modules are added to each Transformer layer —
    - one **after the multi-head attention block**,
    - and another **after the feed-forward network**.
- During fine-tuning, the pre-trained model’s original parameters remain **frozen**.
- Only the parameters in the **new adapter modules** and the **Layer Normalization layers** are updated.
## Setup
1. Clone the Repository
```sh
git clone https://github.com/Ranxin2023/LLM_demo.git
```
2. Install dependencies
```sh
pip install -r requirements.txt
```
3. Set Up Environment Variables
Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_api_key_here
```