from contextlib import redirect_stdout
from transformers import pipeline, set_seed
from demo_code.init_openai import query_open_ai
def temperature_demo():
    print("temperature......")
    model_name="gpt-4"
    prompt="Write a poem about space"
    max_tokens=50
    temperature=0.2
    response = query_open_ai(model_name=model_name, prompt=prompt, temperature=temperature, max_tokens=max_tokens)
    print("Low Temperature Output:", response)
    temperature=1.0
    response = query_open_ai(model_name=model_name, prompt=prompt, temperature=temperature, max_tokens=max_tokens)
    print("High Temperature Output:", response)

def top_k_sampling():
    print("top k sampling......")
    
    generator = pipeline("text-generation", model="gpt2", framework="pt")
    set_seed(42)

    output = generator("Once upon a time", max_length=50, top_k=40)
    print("Top-K Sampling Output:", output[0]["generated_text"])

def top_p_sampling():
    print("top p sampling...")
    model_name="gpt-4"
    prompt="Explain quantum computing in simple terms"
    top_p_list= [0.3, 0.6, 0.8, 1.0]
    temperature=1.0
    max_tokens=60
    for top_p in top_p_list:
        print(f"\n--- Top-P = {top_p} ---")
        response = query_open_ai(
            model_name=model_name,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        print(response)
def output_control_demo():
    with open("./output_results/output_control.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            temperature_demo()
            top_k_sampling()
            top_p_sampling()