from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from datasets import Dataset, load_dataset
import torch
import os
from contextlib import redirect_stdout

dataset = load_dataset("wiki_dpr", "psgs_w100.nq.exact", trust_remote_code=True)
# Step 1: Use RAG to generate QA pairs from questions
def generate_qa_pairs_from_rag(questions):
    os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "true"
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    retriever = RagRetriever.from_pretrained(
        "facebook/rag-token-nq",
        index_name="exact",
        use_dummy_dataset=True  # This avoids triggering wiki_dpr download if you don’t need it
    )
    model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

    qa_pairs = []
    for q in questions:
        inputs = tokenizer(q, return_tensors="pt")
        generated = model.generate(input_ids=inputs["input_ids"])
        answer = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        qa_pairs.append({"text": q + " " + answer, "label": 1})  # positive
    return qa_pairs

# Step 2: Prepare the fine-tuning dataset
def prepare_dataset():
    # Example prompts to generate training data
    questions = [
        "Who developed the theory of relativity?",
        "What does the Earth revolve around?",
        "What is Python?",
        "What is the capital of France?"
    ]

    # RAG-generated correct answers
    positive_samples = generate_qa_pairs_from_rag(questions)

    # Add some random (fake) negative samples for binary classification
    negative_samples = [
        {"text": "The Moon is made of cheese", "label": 0},
        {"text": "Bananas invented gravity", "label": 0},
        {"text": "Paris is located in Australia", "label": 0},
    ]

    return Dataset.from_list(positive_samples + negative_samples)

# Step 3: Fine-tune DistilBERT on binary classification
def fine_tune():
    print("Preparing dataset for fine-tuning...")
    dataset = prepare_dataset()
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def preprocess(example):
        return tokenizer(example["text"], padding="max_length", truncation=True)

    encoded_dataset = dataset.map(preprocess, batched=True)
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Load classification model
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir="./Agent_demo/fine_tune_result",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=20,
        save_total_limit=1,
        evaluation_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    print("✅ Fine-tuning completed. Model saved to ./Agent_demo/fine_tune_result")

def simple_version():
    with open("./output_results/simple_agent.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            fine_tune()
