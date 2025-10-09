# 📘 Demo: Bias Mitigation in Prompt-based Learning
from contextlib import redirect_stdout
from datasets import Dataset
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
def mitigate_bias():
    # 1️⃣ Step 1: Baseline model
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Test uncalibrated prompt
    prompt = "The doctor said that"
    print("Uncalibrated output:\n", generator(prompt, max_length=20)[0]["generated_text"])

    # 2️⃣ Step 2: Prompt Calibration
    # We rephrase prompts to avoid bias in role association.
    calibrated_prompt = "A person working as a doctor said that"
    print("\nCalibrated output:\n", generator(calibrated_prompt, max_length=20)[0]["generated_text"])

    # 3️⃣ Step 3: Data Augmentation Example
    # Create a small balanced dataset for fine-tuning
    examples = [
        {"text": "The doctor said that she would arrive soon."},
        {"text": "The doctor said that he would arrive soon."},
        {"text": "The nurse said that she was tired."},
        {"text": "The nurse said that he was tired."},
    ]

    dataset = Dataset.from_list(examples)

    # Tokenize the dataset
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

    tokenized_dataset = dataset.map(tokenize_fn, batched=True)

    # 4️⃣ Step 4: Fine-tuning on balanced examples
    training_args = TrainingArguments(
        output_dir="./bias_finetune",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        save_strategy="no",
        logging_steps=10
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
    trainer.train()

    # 5️⃣ Step 5: Evaluate after fine-tuning
    finetuned_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print("\nPost-finetuning output:\n", finetuned_generator(prompt, max_length=20)[0]["generated_text"])

def mitigate_bias_output():
    with open("./output_results/mitigate_bias.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            mitigate_bias()