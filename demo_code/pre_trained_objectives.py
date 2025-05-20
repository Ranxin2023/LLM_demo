from contextlib import redirect_stdout
from transformers import BertTokenizer, BertForMaskedLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
def mask_language_model():
    print("MLM(mask language model) demo......")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")

    # Input sentence with a masked token
    text = "The capital of France is [MASK]."
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted token at masked position
    predicted_index = torch.argmax(outputs.logits[0, inputs["input_ids"][0] == tokenizer.mask_token_id])
    predicted_token = tokenizer.decode(predicted_index)
    print("Predicted word:", predicted_token)

def autoregressive_language_modeling():
    print("ALM(autoregressive language modeling) demo......")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    input_text = "Once upon a time"
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate text
    outputs = model.generate(**inputs, max_new_tokens=20)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Generated text:", generated)

def pre_trained_demo():
    with open("./output_results/pre_trained_objectives.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            mask_language_model()
            autoregressive_language_modeling()