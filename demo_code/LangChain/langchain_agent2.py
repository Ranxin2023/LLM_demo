import os
from contextlib import redirect_stdout
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_openai import ChatOpenAI
import streamlit as st
load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
def entry_node(msg: str):
    print(f"LangChain received: {msg}")
    return msg

def node1(msg: str):
    print(f"Hello I am node 1 and I got the message '{msg}'")
    return msg

def node2(msg: str):
    print(f"I am node 2, I got the message from node 1 '{msg}'")
    print("I am making an llm call...")
    response = llm.invoke(msg)
    print(f"LLM Response: {response.content}")
    return response.content

def node3(msg: str):
    print("I am node3 and I got the message from node 2")
    return msg

def node4(msg: str):
    print("Node4 making API call...")
    # Simulated API call
    api_response = "API response...node4api"
    return api_response

def node5(msg: str):
    print(f"I am node5 and I got the api response from node 4: '{msg}'")
    return msg

def node6(msg: str):
    print(f"Node6 received: {msg}")
    print("Goodbye!")
    return msg
def langchain_demo2():
    # --- wrap functions as Runnables ---
    entry = RunnableLambda(entry_node)
    n1 = RunnableLambda(node1)
    n2 = RunnableLambda(node2)
    n3 = RunnableLambda(node3)
    n4 = RunnableLambda(node4)
    n5 = RunnableLambda(node5)
    n6 = RunnableLambda(node6)

    # Option A: RunnableSequence
    workflow = RunnableSequence(
        first=entry,
        middle=[n1, n2, n3, n4, n5],
        last=n6
    )
    # ---- Streamlit UI ---- #
    st.title("LangChain Workflow Demo 🚀")

    user_input = st.text_input("Enter your message:", "hello world from raj")

    if st.button("Send Message"):
        workflow.invoke(user_input)

def langchain2_redirect():
    os.makedirs("./output_results", exist_ok=True)
    with open("./output_results/langchain2.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            langchain_demo2()