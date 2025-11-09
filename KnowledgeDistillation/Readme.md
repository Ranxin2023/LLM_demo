# Knowledge Distillation
## Table of Contents
- [Definition of Knowledge Distillation](#definition-of-knowledge-distillation-kd)
- [Purpose](#purpose)
- [Step by Step Explanation of KD](#step-by-step-explanation-of-knowledge-distillation)
- [Step by Step Explanation of the Diagram](#step-by-step-explanation-of-the-diagram)
- [Benefits of Distillation](#benefits-of-distillation)
- [Application of Distilled LLMs](#applications-of-distilled-llms)
    - [Efficient NLP Tasks](#1-efficient-nlp-tasks)
    - [Chatbots](#2-chatbots)
    - [Text Summarization](#3-text-summarization)
- [Three Types of Knowedge used in Knowledge Distillation](#three-main-types-of-knowledge-used-in-knowledge-distillation-kd)
    - [Responsed Based Knowledge aka Logit Distillation](#1-response-based-knowledge-aka-logit-distillation)
    - [Feature Based Knowledge Intermediate Layer Distillation](#2-feature-based-knowledge-intermediate-layer-distillation)
    - [Relation Based Knowledge Correlation Distillation](#3-relation-based-knowledge-correlation-distillation)
- [Knowledge Distillation Schemes](#knowledge-distillation-schemes)
    - [Offline Distillation](#1-offline-distillation)
    - [Online Distillation](#2-online-distillation)
    - [Self Distillation](#3-self-distillation)
## Definition of Knowledge Distillation (KD)
- **Knowledge Distillation (KD)** is a **model compression technique** in which a **smaller and faster “student” model** learns to reproduce the behavior and decision patterns of a **larger, more accurate “teacher” model**.
- This process allows the student model to achieve comparable performance with fewer parameters and lower computational cost, making it highly suitable for deployment in **resource-constrained environments** such as mobile devices or real-time applications.
- Instead of training only on the original dataset and hard labels (e.g., “dog” vs “cat”), the student is trained to mimic the soft output probabilities of the teacher model — these contain richer information about how the teacher interprets patterns and relationships between classes.
- **In short**:
    - Knowledge Distillation = transferring the “knowledge” of a large model into a smaller one, so that it performs nearly as well but runs much faster and uses fewer resources.
## Purpose
- LLM distillation focuses on:
    - **Reducing computational demands** (less memory, faster inference)
    - **Maintaining performance** (comparable accuracy to the original model)
    - **Enabling deployment** on limited hardware (like mobile phones, browsers, or edge devices)
- This makes distillation crucial in the production phase of AI systems — when large-scale models need to serve millions of users quickly.
## Question: What is the role of Knowledge Distillation in improving LLM deployment?
- **Detailed Explanation**
    - Large Language Models (LLMs) like GPT, BERT, or T5 often have **billions of parameters**, making them extremely powerful but also **computationally expensive**. Running or deploying them on smaller devices (like phones, IoT devices, or low-latency cloud environments) is often impractical.
    - Knowledge Distillation solves this problem by **creating smaller models** that **inherit the intelligence of larger ones**, leading to more efficient and scalable deployment.
## Step-by-Step Explanation of Knowledge Distillation
1. **Teacher Model Training**
- A large, pre-trained model (the teacher) is trained on a big dataset until it achieves high accuracy and captures complex patterns in the data.
2. **Soft Label Generation**
- The teacher’s output probabilities (called **soft targets**) show how confident it is about each class.
- Example:
```text
Teacher Output:
Cat → 0.85, Dog → 0.10, Rabbit → 0.05

```
- This provides richer knowledge than just saying “Cat”.
3. **Student Model Training**
- A smaller model (the student) is trained to mimic these teacher probabilities instead of just the one-hot labels (0 or 1).
- It learns not just what is correct, but why — capturing the teacher’s generalization behavior.
4. **Combined Loss Function**
- The training loss combines:
    - **Cross-entropy loss** (normal supervised learning)
    - **Kullback–Leibler (KL) divergence** (difference between teacher and student output distributions)
## Step-by-Step Explanation of the Diagram
![Distillation Workflow](../images/DistillationWorkflow.png)
1. **Teacher Model (Left Section)**
- The **Teacher Model** is a **large**, **pre-trained** network with many layers and parameters.
- It has already learned deep and complex relationships from a large dataset.
- In the diagram:
    - The **red**, **orange**, and **blue nodes** represent neurons across layers.
    - The teacher is **rich in knowledge** — it knows not only the right answers but also how confident it is about each class (its probability distribution).
- **Example**：
    - For a text classification task:
    ```yaml
    Input: "The movie was amazing!"
    Teacher output probabilities:
    Positive: 0.90, Neutral: 0.08, Negative: 0.02
    ```
2. **Data Feeding (Bottom Arrow)**
- The same **training data** (or unlabeled data) is used by both models.
- This ensures that both teacher and student see identical examples and contexts during training.
3. **Knowledge Transfer Process (Middle Section)**
- This is the **core of distillation**, where knowledge flows from teacher → student.
- **Step 1: Distill**
    - The teacher’s output (logits or probabilities) is passed through a **softmax function** with a temperature 𝑇 > 1. 
    - This softens the output probabilities so that smaller differences between classes are preserved.\n

    | Class  | Hard Label | Teacher (Soft, T=2) |
    | ------ | ---------- | ------------------- |
    | Cat    | 1          | 0.60                |
    | Dog    | 0          | 0.30                |
    | Rabbit | 0          | 0.10                |
    - This “softer” probability distribution provides more information about inter-class relationships than a hard one-hot label.
- **Step 2: Transfer**
    - The **student model** is trained to **mimic** these soft probabilities.
    - A **Knowledge Distillation loss function** (typically KL Divergence) measures how close the student’s outputs are to the teacher’s outputs.
    - Mathematically:\n

$$
    L_{KD} = \alpha \cdot T^2 \cdot KL(p_{teacher}(T) \parallel p_{student}(T)) + (1 - \alpha) \cdot CE(y_{true}, p_{student})
$$
- Explanation of Terms:\n

| **Symbol**                    | **Meaning**                                                               |
|-------------------------------|---------------------------------------------------------------------------|
| ( L_{KD} )                    | Total Knowledge Distillation loss                                         |
| ( \alpha )                    | Balance factor between teacher imitation and true label learning          |
| ( T )                         | Temperature — controls the softness of the teacher’s output probabilities |
| ( KL(\cdot \parallel \cdot) ) | Kullback–Leibler Divergence between teacher and student distributions     |
| ( CE(\cdot, \cdot) )          | Cross-Entropy loss using ground-truth labels                              |
| ( p_{teacher}(T) )            | Teacher’s softened probability distribution                               |
| ( p_{student}(T) )            | Student’s softened probability distribution                               |
| ( y_{true} )                  | True label of the data sample                                             |

4. **Student Model (Right Section)**
- The **Student Model** is a smaller, more compact neural network — fewer layers and parameters.
- It learns to imitate the teacher’s “behavior” rather than memorizing labels.
- Despite being lightweight, it retains most of the teacher’s decision-making intelligence.
- After training:
    - The student model’s predictions become nearly identical to the teacher’s.
    - It requires less memory, less power, and runs much faster — perfect for **edge devices** or **real-time applications**.
5. **Outcome (Top-Level Summary)**

| **Step** | **Component**         | **Role**                                          |
|----------|-----------------------|-------------------------------------------------- |
| 1️        | Teacher               | Provides deep, soft knowledge                     |
| 2️        | Data                  | Feeds input examples to both models               |
| 3️        | Distill & Transfer    | Passes teacher’s soft knowledge                   |
| 4️        | Student               | Learns a compressed representation of the teacher |
| 5        | Result                | Smaller, faster, and still accurate model         |

## Benefits of Distillation
### 1. Reduced Model Size
- **What It Means**
    - One of the most immediate outcomes of distillation is a dramatic reduction in model parameters.
    - For example:
        - **BERT-base** → 110 million parameters
        - **DistilBERT** → 66 million parameters
- **Why It Matters**
    - **Faster Inference**: Smaller models have fewer layers and operations to compute, so they process data faster.
    - **Reduced Storage**: They consume less disk space and RAM, allowing deployment on devices with limited storage (e.g., smartphones, IoT devices, or microcontrollers).
### 2. Improved Inference Speed
- **What It Means**
    - Inference speed refers to **how fast a model can generate predictions** once trained.
    - Since a distilled model has fewer parameters and layers, it computes outputs significantly faster.
- **Technical Reason**
    - Every neural layer in an LLM adds latency during prediction.
    - Removing redundant layers (via distillation) reduces both **forward-pass time** and **memory transfer time**.
### 3. Lower Computational Costs
- **What It Means**
    - Large LLMs require massive computational resources — expensive GPUs, high energy, and large-scale data centers.
    - Distilled models, being smaller, **consume less power and cost far less to operate**.
- **Cost Advantages**
    - **Cloud Environments**:
        - Smaller models mean fewer GPUs or TPUs are needed, lowering both hardware and energy costs.
        - Example: Running BERT-large on AWS can cost 5× more than DistilBERT for the same workload.
    - **On-Premise Deployments:**
        - Organizations that host models locally (banks, hospitals, etc.) can reduce infrastructure and maintenance costs significantly.
### 4. Broader Accessibility and Deployment
- **What It Means**
    - Smaller, faster LLMs can be **deployed anywhere**, not just on high-end servers.
    - This makes advanced AI features more **accessible to everyone**, including in regions or organizations with limited computational capacity.
- **Benefits Breakdown**
    - **Mobile Devices:**
        - Distilled LLMs can run natively on phones or tablets.
        - Enables **voice assistants**, **text summarization**, or **offline translation** without needing constant internet connectivity.
    - **Edge Devices:**
        - Running models locally (on devices like IoT hubs or drones) means data doesn’t have to travel to the cloud.
        - This enhances **privacy** and **reduces network latency**.

## Applications of Distilled LLMs
### 1. Efficient NLP Tasks
- **What It Means**
    - Distilled models are excellent for standard NLP tasks like:
        - **Text classification**
        - **Sentiment analysis**
    - Because they are smaller, they process text **more quickly** while maintaining comparable accuracy to the original large models.
### 2. Chatbots
- **What It Means**
    - Distilled LLMs are ideal for conversational AI systems — chatbots, virtual assistants, and helpdesk agents.
    - They are smaller but still capable of:
        - Understanding user input (natural language understanding)
        - Generating coherent, context-aware responses (natural language generation)
        - Handling real-time interactions smoothly
- **Why It's Important**
    - Chatbots must:
        - Respond within milliseconds
        - Run 24/7
        - Scale to thousands of users simultaneously
    - Large models like GPT-3 may be too heavy or costly for continuous real-time interaction, whereas distilled versions can:
        - Run on **local servers or mobile devices**
        - Use **less GPU memory**
        - Maintain **low latency**
- **Example**
    - A customer service chatbot at a bank:
        - Uses a distilled GPT model to answer FAQs about account balance, loans, and transactions.
        - Runs on a lightweight server (no GPU required).
        - Responds instantly — improving user experience while saving cloud costs.
### 3. Text Summarization
- **What It Means**
    - Distilled LLMs can power **summarization systems** that condense large texts — articles, reports, or social media threads — into **short**, **readable summaries**.
    - This is crucial for applications that require **information compression** and **time efficiency**.
- **Why It Matters**
    - Summarization requires understanding not just words, but the semantic meaning and context.
    - Distilled models preserve the comprehension ability of large models while being faster, making them perfect for:
        - News aggregation platforms
        - Document management systems
        - Social media analysis tools
- **Example**
    - A news summarization app:
        - Takes a 1,000-word article and generates a **100-word summary** in under a second.
        - The distilled model uses **less memory and compute**, so users get instant summaries even on mobile devices.

## Three main types of knowledge used in Knowledge Distillation (KD).
### 1. Response-Based Knowledge (a.k.a. Logit Distillation)
- **Definition**:
    - This is the **most common and classic form** of knowledge distillation (the one implemented in your Python file).
        - The student learns from the **teacher’s final predictions** (logits or softmax probabilities).
        - The teacher’s output acts as a “soft label” for each example.
- **How It Works**
    - The teacher produces **logits** (pre-softmax outputs) for each input.
    - The logits are **softened** by a temperature (T) parameter:\n

$$
    p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$
- where 𝑇 > 1 produces smoother probability distributions.
- The student is trained to **match these soft probabilities** using KL divergence.
- **Why Temperature Matters**
    - If the teacher is **too confident** (probability ≈ 1.0 for one class), there’s no rich information for the student.
    - Increasing the **temperature** makes the distribution softer — revealing secondary probabilities (e.g., 0.55 cat, 0.35 fox, 0.10 dog).
    - 
### 2. Feature-Based Knowledge (Intermediate-Layer Distillation)
- **Definition**
    - Here, the focus shifts from the final output to the hidden layers of the network — the teacher’s intermediate feature activations.
    - The idea: the student should not only match the teacher’s answers but also think in a similar way.
- **How It Works**
1. Extract feature maps (activations) from one or more hidden layers of the teacher.
2. Train the student to **replicate these feature patterns** at corresponding layers.
3. A feature loss (like L2 or cosine similarity) minimizes the difference between teacher and student activations.\n

$$
    L_{feature} = \left\| F_{teacher} - F_{student} \right\|_2^2
$$
- where:
    - \( F_{teacher} \) — the **feature map** (activation output) extracted from one or more layers of the teacher model.  
    - \( F_{student} \) — the **corresponding feature map** from the student model (same or aligned layer).  
    - \( \| \cdot \|_2^2 \) — the **squared L2 norm**, measuring the Euclidean distance between the two feature representations.
- **Example (Vision)**
    - In CNNs for **image classification**:
        - Early layers detect edges and shapes.
        - Mid layers detect patterns (like fur, wings).
        - Late layers identify classes (cat, dog, bird).
    - Feature-based KD ensures the student captures **all hierarchical representations**, not just final predictions.

- **Example (Language Models)**
    - In transformers, feature-based KD transfers:
        - Hidden states (token embeddings)
        - Attention scores
        - Layer-normalized representations
### 3. Relation-Based Knowledge (Correlation Distillation)
- **Definition:**
    - This is a **higher-order** distillation technique that focuses on **relationships between features**, rather than the features themselves.
    - The student doesn’t just mimic the teacher’s outputs or activations — it learns how the teacher’s internal features relate to each other.
- **How It Works:**
    - Compute **relations or correlations** between different features, layers, or locations in the teacher.
        - Example: correlation matrix, cosine similarity, distance matrix.
    - Train the student to reproduce those relationships:\n

$$
    L_{relation} = \left\| F_{teacher} - F_{student} \right\|_2^2
$$
    - This helps the student capture the **structure of the teacher’s feature space** — how features interact and co-occur.

## Knowledge Distillation Schemes
- While traditional knowledge distillation (KD) focuses mainly on what knowledge is transferred (e.g., logits, features, or relations),distillation schemes define how and when this transfer occurs during training.
- There are three major schemes:
    - **Offline Distillation** — teacher is pre-trained and fixed.
    - **Online Distillation** — teacher and student are trained simultaneously.
    - **Self-Distillation** — a single model acts as both teacher and student.
### 1. **Offline Distillation**
- **Definition:**
    - Offline distillation (also known as **classic distillation**) is the original KD method proposed by Hinton et al. (2015).
    - Here, the **teacher model is pre-trained** on a dataset, and its weights are frozen before the distillation process begins.
- **How It Works**
    - Train a large, high-capacity **teacher model** first.
    - Freeze the teacher’s parameters (no further updates).
    - Train a smaller **student model**, using:
        - Teacher’s **soft logits** (response-based knowledge).
        - Ground-truth labels from the dataset.
    - The student minimizes the KD loss:\n
    
$$
    L_{KD} = \alpha T^2 KL(p_t(T) \parallel p_s(T)) + (1 - \alpha) CE(y, p_s)
$$

- where:  
    - \( \alpha \) — balancing factor between distillation loss and true label loss  
    - \( T \) — temperature parameter (controls the softness of probability distribution)  
    - \( KL(p_t(T) \parallel p_s(T)) \) — Kullback-Leibler divergence between teacher and student outputs  
    - \( CE(y, p_s) \) — cross-entropy loss between true labels and student predictions  

- **Key Idea**
    - The student learns to reproduce the teacher’s output behavior without changing the teacher.
- **Advantages**
    - Stable and easy to implement.
    - Teacher’s knowledge is consistent and already optimized.
    - Most suitable when using large pre-trained models (e.g., BERT, GPT, ResNet).
- **Limitations**
    - Teacher cannot adapt to student feedback.
    - Requires access to a well-trained, high-performing teacher model.
### 2. **Online Distillation**
- **Definition:**
    - In online distillation, both the **teacher and student are trained simultaneously**.
    - The teacher is not frozen — it evolves during training alongside the student.
- **How It Works**
    - Initialize both teacher and student models (teacher can be slightly larger or the same architecture).
    - Train both models together on the same data.
    - At each iteration:
        - The teacher produces soft targets for the student.
        - The student learns from both the ground-truth and the teacher’s outputs.
        - The teacher may also continue to learn or adapt based on new data.

$$
    L_{online} = \alpha KL(p_t \parallel p_s) + (1 - \alpha) CE(y, p_s)
$$

- where:  
    - \( p_t \) — teacher’s output distribution (updated dynamically)  
    - \( p_s \) — student’s output distribution  
    - \( \alpha \) — trade-off parameter between imitation and true label learning  
    - \( CE(y, p_s) \) — cross-entropy with true labels  

### 3. Self-Distillation
- In self-distillation, there is no separate teacher model.
- Instead, a single network acts as both teacher and student — transferring knowledge from its deeper layers to its shallower layers.
#### Summary
| **Scheme**               | **Description**                           | **Key Benefit**                                            |
| ------------------------ | ----------------------------------------- | ---------------------------------------------------------- |
| **Offline Distillation** | Teacher pre-trained, student learns after | Simple and stable for model compression                    |
| **Online Distillation**  | Teacher and student co-trained            | Dynamic adaptation, real-time performance                  |
| **Self-Distillation**    | Teacher and student are same model        | Efficient self-improvement and internal knowledge transfer |

