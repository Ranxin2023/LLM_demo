# LLM DEMO
## Concepts
1. How do you measure the performance of an LLM?
### 1.1 🔢 Perplexity
- **definition**:
Perplexity is a measurement of how well a language model predicts a sequence. It is the exponential of the average negative log-likelihood of the predicted tokens.
- **formula**:
Perplexity = *e*^Loss
- **interpretation**:

**Low perplexity** → Model is confident and accurate in predicting the next tokens.
**High perplexity** → Model is "surprised" by the actual tokens.
### 1.2 🧮 Accuracy
- **definition**:
Accuracy is the ratio of **correct predictions to total predictions**. It is commonly used in classification tasks (e.g., sentiment analysis, text classification).
- **formula**:
Accuracy = Number of Correct Predictions / Total Predictions

