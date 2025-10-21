from contextlib import redirect_stdout
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()
def CoT_demo():
    print("CoT Function......")
    # Initialize model
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Define a Chain-of-Thought style prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful reasoning assistant. Always explain your reasoning step-by-step."),
        ("human", "If a car travels 60 miles in 1.5 hours, what is its average speed in miles per hour?")
    ])

    # Invoke the model
    response = (prompt | llm).invoke({})

    print(f"Response in coTFunction{response.content}")

def CoT_redirect_output():
    with open("./output_results/Cot_demo.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            CoT_demo()