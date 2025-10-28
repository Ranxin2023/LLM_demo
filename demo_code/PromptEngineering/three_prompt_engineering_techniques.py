# -----------------------------
# Example: Prompt Engineering Demo
# -----------------------------
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1️⃣ Initialize the model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# 2️⃣ Define multiple prompting strategies

## Zero-shot prompting (no examples)
zero_shot_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI expert who explains concepts clearly."),
    ("human", "Explain the concept of Knowledge Distillation in deep learning.")
])

## Few-shot prompting (give the model examples)
few_shot_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a technical AI tutor."),
    ("human", "Q: What is overfitting?\nA: When a model learns noise instead of the general pattern."),
    ("human", "Q: What is dropout?\nA: A regularization technique that randomly ignores neurons during training."),
    ("human", "Q: What is Knowledge Distillation?")
])

## Chain-of-Thought (CoT) prompting (step-by-step reasoning)
cot_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a deep learning researcher. Think step-by-step before answering."),
    ("human", "How does Knowledge Distillation transfer information from a teacher model to a student model?")
])

# 3️⃣ Generate responses
print("🟢 ZERO-SHOT RESULT:\n", llm.invoke(zero_shot_prompt.format_messages()).content)
print("\n🟢 FEW-SHOT RESULT:\n", llm.invoke(few_shot_prompt.format_messages()).content)
print("\n🟢 CHAIN-OF-THOUGHT RESULT:\n", llm.invoke(cot_prompt.format_messages()).content)
