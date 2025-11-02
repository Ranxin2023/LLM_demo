from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from contextlib import redirect_stdout

load_dotenv()
prompt_v1 = "Summarize this article."
prompt_v2 = "Summarize this article in three concise bullet points focusing on key insights."

text = "AI models like GPT are transforming how people create, learn, and interact with information..."
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
def iterate_refine_prompt():
    for version, prompt_text in enumerate([prompt_v1, prompt_v2], start=1):
        result = llm.invoke([("human", prompt_text + "\n\n" + text)])
        print(f"\nVersion {version} Output:\n{result.content}")
        
def redirect_refinement_output():
    with open("./output_results/refinement_and_optimize.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            iterate_refine_prompt()