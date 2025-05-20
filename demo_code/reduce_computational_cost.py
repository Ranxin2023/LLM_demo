import os
import torch
from contextlib import redirect_stdout
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
# 1. Model Pruning is in LLM Evaluation
# 2. Quantization
def quantization():
    print("quantization...")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    def print_model_size(model, path):
        torch.save(model.state_dict(), path)
        size_mb = os.path.getsize(path) / 1e6
        print(f"Model size: {size_mb:.2f} MB")
        os.remove(path)
    print("🔹 Original model:")
    print_model_size(model, "original_model.pt")

    print("🔹 Quantized model:")
    print_model_size(quantized_model, "quantized_model.pt")
    # Run sample inference
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer("This is a test after quantization.", return_tensors="pt")

    with torch.no_grad():
        outputs = quantized_model(**inputs)
        logits = outputs.logits

    print("\nSample logits from quantized model:")
    print(logits)
    
# 3. distillation
def distillation():
    print("🧪 Loading DistilBERT tokenizer and model...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    model.eval()
    print("✅ Tokenizer and model loaded successfully.")
    print("🧠 Model architecture:")
    print(model)
    # Sample input for testing
    input_text = "This is a test for distilled model evaluation."
    inputs = tokenizer(input_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    print("\n📊 Logits from the distilled model:")
    print(logits)

def five_method():
    with open("./output_results/reduce_computational.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            quantization()
            distillation()