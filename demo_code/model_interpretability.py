import torch
from transformers import BertTokenizer, BertForSequenceClassification
from captum.attr import Saliency
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
def interpret_model():
    print("model interpretability demo......")
    model_name = "bert-base-uncased"
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model.eval()

def redirect_output():
    with open("./output_results/interpret_model.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            interpret_model()