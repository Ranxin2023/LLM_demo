# Explain for the result of `AB_testing_demo.py`
## 1. Overview of the Experiment
- You ran a **Complex A/B/C Prompt Evaluation** on the topic of **Knowledge Distillation**, comparing three different prompt designs:
    - **Prompt A**: A one-paragraph summary request
    - **Prompt B**: A 3-bullet concise technical summary
    - **Prompt C**: A use-case-focused description
- Each prompt version was tested on:
    - **Relevance** (semantic similarity to an ideal reference summary)
    - **Conciseness** (how efficient and focused the answer is)
    - **Readability** (ease of understanding and sentence structure quality)
- The **Final Score** combines these metrics with specific weights:
    - 50% Relevance + 30% Conciseness + 20% Readability
- The prompt with the highest Final Score is the best-performing one.
## 2. Numerical Breakdown and Meaning
| **Prompt** | **Relevance** | **Conciseness** | Readability | Final Score | Interpretation                                |
| ---------- | ------------- | --------------- | ----------- | ----------- | --------------------------------------------- |
| **A**      | 0.849         | 1.000           | 0.400       | **0.805**   | Accurate but dense and harder to read         |
| **B**      | 0.858         | 1.020           | 0.827       | **0.900**   | Best balance: accurate, concise, and readable |
| **C**      | 0.840         | 0.710           | 0.413       | **0.716**   | Informative but verbose and less readable     |
- **Prompt A — “One Paragraph Summary”**
    - Output Summary:
        - Knowledge Distillation is a model compression technique that enables a smaller, faster student model to replicate the behavior of a larger teacher model...
    - Interpretation:
        - **Relevance (0.849)**: The summary is accurate — it captures the essence of knowledge distillation well.
        - **Conciseness (1.000)**: Ideal length — one compact paragraph with no filler.
        - **Readability (0.400)**: Low readability — likely due to complex, academic phrasing (dense sentences, no structural breaks).
- **Prompt B — “Three Concise Bullet Points”**
    - **Output Summary**:
        - "Knowledge Distillation involves training a smaller student model..."
        - The technique minimizes divergence..."
        - "This approach enables deployment..."
    - **Interpretation**:
        - **Relevance (0.858)**: Slightly higher than A — covers all main ideas accurately.
        - **Conciseness (1.020)**: Excellent — very short, focused, and efficient in communication.
        - **Readability (0.827)**: High — bullet points improve clarity and flow; each statement is direct.
- **Prompt C — “Practical Benefits and Use Cases”**
    - **Output Summary**:
        - “Knowledge Distillation is a technique for compressing machine learning models... Practical benefits include reduced computational resource requirements, faster inference times, and the ability to deploy models on devices with limited processing power...”
    - **Interpretation**
        - **Relevance (0.840)**: Still accurate, but slightly less aligned with the reference summary because it focuses on applications more than on the core mechanism.
## 3. Why Prompt B Won
- Prompt B achieved the **highest final score (0.900)** because:
    - It maintained **excellent relevance** (0.858 → it captured all major ideas).
    - It was **slightly more concise** than needed (1.020 → slightly above optimal, but ideal for summarization).
    - Its **readability (0.827)** was far higher than the others due to its clear, list-based formatting.
- In short:
    - **Prompt B** = technically complete + easy to read + well structured.
- This aligns with prompt engineering best practices:
    - “Clear structure, focus constraints, and task-oriented phrasing yield the most stable results.”
## 4. How Each Metric Interacted
| Metric          | What It Captured                  | Observation                                          |
| --------------- | --------------------------------- | ---------------------------------------------------- |
| **Relevance**   | Semantic similarity to reference  | All prompts scored close → all understood the topic. |
| **Conciseness** | Efficiency and directness         | B and A excelled, C lagged due to verbosity.         |
| **Readability** | Sentence simplicity and structure | B stood out — formatting helped readability most.    |
