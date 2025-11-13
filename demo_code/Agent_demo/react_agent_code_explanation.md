# Explanation of `react_agent_demo.py`
## Overview
- This example implements a **ReAct Agent** (Reason + Act) using **LangGraph**, where the agent:
1. Receives a **query**,
2. Generates a **thought**,
3. Executes an **action** using a tool,
4. **Observes** the tool output,
5. Loops until it decides to produce a **final answer**.

## 1. Importing Required Dependencies
```python
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

```
- What each component does:
- **TypedDict**: Defines structured state for the agent.
- **HumanMessage**, **AIMessage**, **ToolMessage**:
    - The communication protocol inside ReAct:
        - `HumanMessage`: user query
        - `AIMessage`: agent reasoning (Thought, Action, Final Answer)
        - `ToolMessage`: output from tools (Observation)
- **tool decorator**: Used to define Python functions as callable tools for the agent.
- **ChatOpenAI**: LLM that generates reasoning, tool calls, and answers.
- **StateGraph**: LangGraph component that lets us build the ReAct decision graph.
- **ToolNode**: Prebuilt execution engine that automatically runs tools when agents call them.
## 2. Defining Tools (Action Phase)
```python
@tool
def web_search(query: str) -> str:
    return f"[web_search] Top result snippet for: {query}"

@tool
def calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"Result of {expression} = {result}"
    except Exception as e:
        return f"Calculator error: {e}"

```

- **web_search**:
    - Simulates performing a web search.
    - In real-world ReAct, this could call Google SERP API, Bing API, DuckDuckGo, etc.
- **calculator**:
    - Allows the agent to solve math expressions.
    - 
## 3. Agent State Definition
- 