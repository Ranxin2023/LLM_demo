from contextlib import redirect_stdout
from transformers import pipeline, set_seed
from demo_code.init_openai import query_gpt4
def temperature():
    print("temperature......")
    response = query_gpt4( temperature=0.2)
    print("Low Temperature Output:", response)
def top_k_sampling():
    print("top k sampling......")
    
    generator = pipeline("text-generation", model="gpt2")
    set_seed(42)

    output = generator("Once upon a time", max_length=50, top_k=40)
    print("Top-K Sampling Output:", output[0]["generated_text"])
def top_p_sampling():
    pass
def output_control_demo():
    with open("./output_results/output_control.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            temperature()
            top_k_sampling()