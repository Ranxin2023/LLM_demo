# Fine Tuning
## Table Of Contents

- [What is Fine Tuning](#1-what-is-fine-tuning)
    - [Why Fine Tuning Works](#why-fine-tuning-works)
    - [Types of Fine Tuning](#types-of-fine-tuning)
    - [Four Methods for Fine Tuning](#four-methods-for-fine-tuning)
- [Catastrophic Forgetting](#13-catastrophic-forgetting)
    - [Prompt Calibration](#1-prompt-calibration)
    - [Fine Tuning](#2-fine-tuning)
    - [Data Agumentation](#3-data-augmentation)
- [LoRA](#14preknowledge-lora)
- [PEFT](#14-peft)
- [Adapter Tuning](#16-adapter-tuning)
- [Fine Tune Methods](#fine-tune-methods)
    - [Freeze Method](#freeze-method)
    - [P-Tuning](#p-tuning-method)
    - [Full Fine Tuning](#full-fine-tuning)
- [LoRA](#lora)

## 1. What Is Fine-Tuning?
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

## 2. Why Fine-Tuning Works
- When a model like DistilBERT is pre-trained:
    - It learns general knowledge of language patterns.
    - But it doesn’t know how to perform **task-specific** jobs like classifying IMDb reviews as positive or negative.


## 3. Fine-Tuning Workflow
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

### Four Methods for Fine-Tuning
#### Prompt Tuning (a.k.a Soft Prompting / P-Tuning / Prefix Tuning)
- **Definition**:
    - Prompt tuning does not modify the model weights at all.
    - Instead, it learns a small set of trainable prompt embeddings that are prepended to the input.
#### Full Fine-Tuning (Standard Fine-Tuning)
- **Defintion**:
    - You update all the parameters of the model.
    - This is the original and most powerful form of fine-tuning.
- **How It Works**:
    - Unfreeze all layers.
    - Train on your custom dataset.
    - Every weight is updated via backpropagation.
- **Characteristics**

| **Property**       | **Value**                            |
| ------------------ | ------------------------------------ |
| Parameters trained | 100%                                 |
| Cost               | extremely expensive                  |
| Memory             | very high (needs multiple GPUs)      |
| Quality            | best performance possible            |
| Best for           | large datasets, domain-specific LLMs |

## Fine-Tune Methods
### Freeze Method
- The Freeze method literally means freezing parameters.
- In this approach, most of the parameters of the original large model are frozen, and only a small subset of parameters is trained.
- By doing so, memory usage can be significantly reduced, making it possible to fine-tune large models more efficiently.


### P-tuning Method
#### What is P-tuning
- P-tuning is a parameter-efficient fine-tuning (PEFT) method that teaches a language model how to prompt itself.
- Instead of:
    - manually writing text prompts like
        - “Please classify the sentiment of the following sentence…”
    - P-tuning:
        - **learns continuous prompt embeddings automatically**
        - keeps the **entire pretrained model frozen**
        - optimizes only a **small number of prompt parameters**
#### prefix tuning
![Prefix Tuning Diagram](../images/p_tuning)
##### 1. Top Half: Fine-tuning (Full / Partial Model Updates)
- What the diagram shows:
    - You see multiple Transformer stacks, each labeled with a task:
        - **Translation**
        - **Summarization**
        - **Table-to-text**
#### 2. 

### Full Fine-tuning
#### Definition of Full Fine-tuning
- Full fine-tuning is a training method where all parameters of a pre-trained foundation model are updated using a **smaller**, **task-specific dataset**.
- Instead of freezing layers or adding adapter modules (like in LoRA or QLoRA), full fine-tuning **modifies every weight** of the base model.

#### How Full Fine-Tuning Works
##### 1. Prepare Training Data
- You provide a dataset of **input** → **target** output pairs, such as:
    ```vbnet
        Input: "Summarize this text..."
        Output: "This article explains..."
    ```
- These examples define how you want the model to behave.
#### 2. Feed Each Batch Into the Model
- The training data is split into batches (e.g., 32 or 64 samples per batch).
- 

##### Full Fine-Tuning Workflow Explained (Step-by-Step)
![Full Fine-tuning Workflow](images/full_fine_tuning.png)
#### 1. **Training Data (Step 1)**
- **Training data is divided into batches**
    - Training datasets are large.
    - Instead of feeding the entire dataset at once, we split it into **batches** (Batch 1, Batch 2, …).
- **Why batches?**
    - Reduce memory usage
    - More stable optimization
    - Support gradient accumulation
#### 2. **Start: Input a batch into the model (Step 2)**
- Each batch is passed into the model:
```nginx
Batch → Model → Output
```
- This produces predictions (logits or probabilities).
- The model still uses **old parameters** at this stage.
#### 3. Compare model output vs. expected output → Compute Loss (Step 3)
- Compare:
    - **Model output** (predicted result)
    - **Training data output** (true labels)
- Loss function
#### 4. Update Model Weights (Step 4)
- **Optimizer updates all model weights**
    - Since this is full fine-tuning, every parameter in every layer is updated.
    - Common optimizers:
        - AdamW
        - SGD
        - RMSProp
- **Controlled by learning rate**
    - The diagram shows:
        - Adjust by learning rate
    - Learning rate controls the size of each update step.
    - Too high → unstable training
    - Too low → slow training
- **Gradient accumulation**
- **Model ready for next iteration (Step 5)**
    - 

## LoRA
### What is Low-Rank Adaptation (LoRA)?

**Low-Rank Adaptation (LoRA)** is a **parameter-efficient fine-tuning (PEFT)** technique designed to adapt large pre-trained models for specific tasks **without significantly increasing computational or memory costs**.

As large language models (LLMs) grow in size and complexity, fine-tuning them on new tasks often requires **substantial computational power and GPU memory**.  
LoRA solves this problem by reducing the number of trainable parameters — making the fine-tuning process **faster, lighter, and more efficient**.

---

### 🧠 Key Idea

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

### Explanation of How LORA Works
![Lora Workflow](images/lora_workflow.svg)
#### 1. Training Data & Batching (Right side, purple box)
- **Training data**
    - The dataset is split into batches (Batch 1, Batch 2, …).
    - **Batch size** determines how many samples are processed per step.
    - Multiple batches = one epoch.
    - Multiple epochs = full training.
#### 2. Input from Training Data (Top center)
- Each batch provides:
    - **Input text** (e.g. prompt, question)
    - **Expected output** (ground truth)
- This input flows into the model through **the LoRA adapters**, not directly into trainable base weights.
#### 3. Base Foundation Model (Green circle, right)
##### **Frozen model**
- This is the original pre-trained model (GPT, LLaMA, Mistral, etc.).
- All original weights are frozen:
    - Attention layers
    - Feed-forward layers
    - Embeddings
- Unlike full fine-tuning, **no gradients update these weights**.
#### 4. Low-Rank Adapters (Blue dashed box, center-right)
- This is the core of LoRA
- LoRA inserts **small trainable matrices** into specific layers:
    - Usually **Q, K, V** projections in attention
- 