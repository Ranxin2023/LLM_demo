from contextlib import redirect_stdout
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import numpy as np
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return {"accuracy": acc, "f1": f1}

def fine_tune_demo():
    with open("./output_results/fine_tune_demo.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            print("fine tune the distilbert...")
            # Load example dataset (binary sentiment classification)
            dataset = load_dataset("imdb", split="train[:2000]")  # use a small subset for demo
            dataset = dataset.train_test_split(test_size=0.2)

            # Tokenization
            tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
            def preprocess(examples):
                return tokenizer(examples["text"], truncation=True, padding=True)
            
            encoded_dataset=dataset.map(preprocess, batched=True)
            # Load pre-trained model
            model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
             # ⚠️ Evaluation BEFORE fine-tuning
            trainer_pre = Trainer(model=model, tokenizer=tokenizer)
            outputs_pre = trainer_pre.predict(encoded_dataset["test"])
            pred_labels_pre = np.argmax(outputs_pre.predictions, axis=1)
            true_labels = outputs_pre.label_ids
            acc_pre = accuracy_score(true_labels, pred_labels_pre)
            f1_pre = f1_score(true_labels, pred_labels_pre)
            print(f"📉 Before Fine-Tuning — Accuracy: {acc_pre:.4f}, F1: {f1_pre:.4f}")
            # Training configuration
            training_args = TrainingArguments(
                output_dir="./output_results/fine_tune_result",
                # evaluation_strategy="epoch",
                logging_steps=10,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                num_train_epochs=1,
                save_steps=50,
                save_total_limit=2,
            )

            # Train using Hugging Face Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=encoded_dataset["train"],
                eval_dataset=encoded_dataset["test"],
                tokenizer=tokenizer,
            )

            trainer.train()

            # ✅ Evaluation AFTER fine-tuning
            outputs_post = trainer.predict(encoded_dataset["test"])
            pred_labels_post = np.argmax(outputs_post.predictions, axis=1)
            acc_post = accuracy_score(true_labels, pred_labels_post)
            f1_post = f1_score(true_labels, pred_labels_post)
            print(f"📈 After Fine-Tuning — Accuracy: {acc_post:.4f}, F1: {f1_post:.4f}")