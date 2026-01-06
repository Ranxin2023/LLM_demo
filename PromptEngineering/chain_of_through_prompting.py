from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from contextlib import redirect_stdout

load_dotenv()
# Initialize model
llm = ChatOpenAI(model="gpt-4o-mini")

# Define a Chain-of-Thought style prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful reasoning assistant. Always explain your reasoning step-by-step."),
    ("human", "If a car travels 60 miles in 1.5 hours, what is its average speed in miles per hour?")
])

# Invoke the model
def cot_demo():
    with open("../output_results/PromptEngineering/chain_of_thought_prompting.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            response = (prompt | llm).invoke({})
            print(response.content)
