from contextlib import redirect_stdout
from demo_code.init_openai import query_open_ai
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def count_tokens(tokenizer, text):
    return len(tokenizer.encode(text))

def generate_with_gpt2(prompt, max_new_tokens=50):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # GPT-2 has a max context window of 1024 tokens
    if input_ids.size(1) > 1024 - max_new_tokens:
        print(f"⚠️ Input too long ({input_ids.size(1)} tokens). Truncating to fit context window...")
        input_ids = input_ids[:, - (1024 - max_new_tokens):]

    outputs = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def context_window_demo():
    with open("./output_results/context_window.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            short_context = "Once upon a time in a distant kingdom,"
            long_context = "The king had many sons. " * 400 + short_context  # ~1024–2048+ tokens

            print("🧪 Running GPT-2 (local, max ~1024 tokens):")
            print("🔹 Short context output:")
            print(generate_with_gpt2(short_context))
            print("🔹 Long context output:")
            print(generate_with_gpt2(long_context))

            print("\n🧪 Running GPT-3.5-turbo (max ~4,096 tokens):")
            print("🔹 Short context output:")
            print(query_open_ai(short_context, model_name="gpt-3.5-turbo"))
            print("🔹 Long context output:")
            print(query_open_ai(long_context, model_name="gpt-3.5-turbo"))

            print("\n🧪 Running GPT-4-turbo (max ~128,000 tokens):")
            print("🔹 Short context output:")
            print(query_open_ai(short_context, model_name="gpt-4-turbo"))
            print("🔹 Long context output:")
            print(query_open_ai(long_context, model_name="gpt-4-turbo"))