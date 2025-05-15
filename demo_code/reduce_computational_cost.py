import os
import torch
from contextlib import redirect_stdout
from transformers import BertForSequenceClassification, BertTokenizer
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

def five_method():
    with open("./output_results/reduce_computational.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            quantization()