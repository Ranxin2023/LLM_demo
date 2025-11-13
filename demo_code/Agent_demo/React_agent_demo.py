"""
Complex ReAct Agent with LangGraph

- The agent reasons in ReAct style (Thought / Action / Observation).
- It can call tools (web_search, calculator) multiple times.
- LangGraph manages the loop: agent -> tools -> agent ... -> END.
"""

from typing import TypedDict, List, Optional

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


# -----------------------------
# 1. Define Tools (Actions)
# -----------------------------

@tool
def web_search(query: str) -> str:
    """Fake web search. In real use, call an API like SerpAPI/Bing."""
    # Here we just simulate a result for demo purposes.
    return f"[web_search] Top result snippet for: {query}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple math expression like '2 + 3 * 4'."""
    try:
        # ⚠️ eval is fine for a toy demo; use a safe math parser in production.
        result = eval(expression, {"__builtins__": {}})
        return f"Result of {expression} = {result}"
    except Exception as e:
        return f"Calculator error: {e}"


TOOLS = [web_search, calculator]


# -----------------------------
# 2. Define Agent State
# -----------------------------

class AgentState(TypedDict):
    """Shared state flowing through the LangGraph graph."""
    messages: List[BaseMessage]
    # optional field for debug / routing decisions
    done: Optional[bool]


# -----------------------------
# 3. Build the ReAct LLM Node
# -----------------------------

# Bind tools so the LLM can *decide* when to call them
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(TOOLS)


def react_agent_node(state: AgentState) -> AgentState:
    """
    ReAct agent step:
    - Reads conversation + tool results from `state["messages"]`
    - Produces an AIMessage that may contain:
        - pure reasoning (Thought + Final Answer), or
        - a tool call (Action) to be executed by ToolNode.
    """
    response: AIMessage = llm.invoke(state["messages"])
    return {
        "messages": state["messages"] + [response],
        "done": state.get("done"),
    }


# -----------------------------
# 4. Tool Execution Node
# -----------------------------

tool_node = ToolNode(TOOLS)
# ToolNode:
# - scans messages for pending tool calls from the LLM
# - executes them
# - appends ToolMessage observations back into the state


# -----------------------------
# 5. Routing Logic (When to Loop?)
# -----------------------------

def route_after_agent(state: AgentState) -> str:
    """
    Decide what to do after the agent node runs.

    Cases:
    1. If the last AIMessage contains tool calls -> go to 'tools'.
    2. Else if the model says it's finished -> END.
    3. Otherwise, let the agent think again (e.g., self-reflection loops).
    """
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        # Agent decided to "Act" → execute tools
        return "tools"

    # Check for an explicit 'final' marker in the content
    content = last.content if isinstance(last, AIMessage) else ""
    if isinstance(content, str) and ("FINAL ANSWER:" in content or "Final answer:" in content):
        return END

    # Optionally use a max-step safeguard:
    max_turns = 8
    ai_turns = sum(isinstance(m, AIMessage) for m in state["messages"])
    if ai_turns >= max_turns:
        return END

    # No tools + no final answer → let the agent think again
    return "agent"


def route_after_tools(state: AgentState) -> str:
    """
    After tools run, always go back to the agent so it can
    observe the ToolMessage and continue reasoning.
    """
    return "agent"


# -----------------------------
# 6. Build the LangGraph Graph
# -----------------------------

builder = StateGraph(AgentState)

builder.add_node("agent", react_agent_node)
builder.add_node("tools", tool_node)

builder.set_entry_point("agent")

# Conditional edges from 'agent'
builder.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "tools": "tools",
        "agent": "agent",
        END: END,
    },
)

# After tools execute, always go back to agent
builder.add_conditional_edges(
    "tools",
    route_after_tools,
    {
        "agent": "agent",
    },
)

graph = builder.compile()


# -----------------------------
# 7. Helper: Run One Query
# -----------------------------

def run_react_agent(query: str):
    """
    Execute the ReAct agent graph for a single user query
    and stream intermediate steps.
    """
    print(f"\n===============================")
    print(f"USER QUERY: {query}")
    print(f"===============================\n")

    initial_state: AgentState = {
        "messages": [
            SystemMessage(
                content=(
                    "You are a ReAct-style assistant. "
                    "Use the following pattern:\n"
                    "Thought: reason about what to do next.\n"
                    "Action: call a tool when needed.\n"
                    "Observation: reflect on the tool result.\n"
                    "Repeat as needed, then end with:\n"
                    "\"FINAL ANSWER: <your answer>\""
                )
            ),
            HumanMessage(content=query),
        ],
        "done": False,
    }

    # graph.stream yields intermediate states as nodes execute
    for event in graph.stream(initial_state, stream_mode="values"):
        # Each `event` is a partial AgentState after some node finished
        messages = event["messages"]
        last = messages[-1]

        if isinstance(last, HumanMessage):
            print(f"[Human] {last.content}\n")
        elif isinstance(last, AIMessage):
            # Might be pure text, or a tool call
            print("[AI]")
            print(last.content, "\n")
            if last.tool_calls:
                print(f"--> Tool calls: {last.tool_calls}\n")
        elif isinstance(last, ToolMessage):
            print(f"[Tool {last.name}] {last.content}\n")

    # Final state (after END)
    final_state = graph.invoke(initial_state)
    final_msgs = final_state["messages"]
    final_ai = [m for m in final_msgs if isinstance(m, AIMessage)][-1]
    print("===== FINAL ANSWER =====")
    print(final_ai.content)
    print("========================\n")


# -----------------------------
# 8. Demo
# -----------------------------

def react_agent_demo():
    # Example 1: requires search + calculation
    run_react_agent(
        "You are my travel assistant. "
        "Find a realistic 2-day itinerary for Tokyo, "
        "estimate the total cost in USD, and justify your estimate."
    )

    # Example 2: more purely analytical, but tools are still available
    run_react_agent(
        "Compare the long-term value of learning Python vs learning Rust "
        "for backend development. Use web_search if you need fresh info."
    )
