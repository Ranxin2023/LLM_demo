from dotenv import load_dotenv
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from contextlib import redirect_stdout
import os

load_dotenv()

def lang_chain_demo():
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    print("🔧 Loading tools...")
    raw_tools = load_tools(["serpapi", "llm-math"], llm=llm)

    tools = []
    for t in raw_tools:
        if not t or not hasattr(t, "name"):
            continue
        tools.append(t)

    if not tools:
        raise ValueError("No valid tools found.")

    print(f"✅ Loaded tools: {[t.name for t in tools]}")

    # Required prompt format includes "tools"
    template = """Answer the following questions as best you can. You have access to the following tools:

{tool_names}

TOOLS DESCRIPTION:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["input", "agent_scratchpad", "tool_names", "tools"]
    )

    print("🧠 Creating agent...")
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    query = "What is the square root of the population of Germany, and who is its current Chancellor?"
    response = agent_executor.invoke({"input": query})

    print("\n🤖 Final Answer:\n", response['output'])

def agent_demo_redirect():
    os.makedirs("./output_results", exist_ok=True)
    with open("./output_results/langchain.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            lang_chain_demo()
