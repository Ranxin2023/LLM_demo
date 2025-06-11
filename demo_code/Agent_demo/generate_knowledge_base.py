from transformers import pipeline
from demo_code.init_openai import query_open_ai
def generate_knowledge_base():
    with open("./knowledge_base.txt", "r",  encoding="utf-8") as f:
        knowledge_base=f.read()
    query = "How did Alan Turing contribute to the development of artificial intelligence?"

    prompt = f"""{knowledge_base}

    Think step-by-step and answer the following question:
    Q: {query}
    A:"""
     # Step 3: Query GPT-4 using your wrapped function
    response = query_open_ai(
        model_name="gpt-4",
        prompt=prompt,
        temperature=0.4,
        max_tokens=300
    )

    # Step 4: Print the result
    print("🧠 GPT-4 Reasoned Answer:\n", response)
    