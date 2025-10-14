# Code Explanation of RAG Planner
## 1. Fake tools (retrievers)
```python
def vector_kb_search(query: str) -> List[str]:
    corpus = {...}
    if "overview" in query: ...
    if "semantic" in query: ...
    if "hybrid" in query: ...

```
- Simulates a **vector store**: given a semantic intent (“overview”, “semantic”, “hybrid”), you get a relevant snippet.
- In production you’d replace this with a vector DB retriever (e.g., Chroma, FAISS, PGVector) that:
1. embeds the query,
2. performs a similarity search,
3. returns top-k chunks with metadata (source, chunk id, score).
```python
def sql_ticket_analytics(query: str) -> List[str]:
    if "6 months" in query:
        return ["Tickets: 42% billing, 35% setup, 23% account; 28% contain paraphrased intents"]

```
- Simulates **analytics/BI** (SQL). In production you’d hit a warehouse (BigQuery/Snowflake/Postgres) and return aggregates that inform retrieval choices (e.g., paraphrase rate → favor vector/hybrid).
```python
def web_research(query: str) -> List[str]:
    if "BM25 vs dense" in query:
        return ["Study A ...", "Case study ..."]
```
- Simulates **web/academic retrieval**. Real systems would use a search API (Tavily, Serp, Bing) + page fetcher + chunker + re-ranker.
## 2. Planner
```python
def planner(user_query: str) -> List[Dict[str, Any]]:
    return [
        {"id": "Q1", "tool": "vector", "query": "KB overview and domains"},
        {"id": "Q2", "tool": "sql",    "query": "Ticket analytics for last 6 months (mix, paraphrase rate)"},
        {"id": "Q3", "tool": "web",    "query": "BM25 vs dense retrieval for FAQs; evidence on hybrid"}
    ]

```
- **Decomposes** the user’s broad task into **three atomic sub-tasks**:
    - Q1 → Context from your KB (vector).
    - Q2 → Empirical usage patterns (SQL).
    - Q3 → External evidence for method choice (web).
- **Chooses tools** per sub-task. This is the essence of an **agentic planner**.
- In a real agent, the planner would itself be an LLM node (possibly with a schema/validator) that:
    - extracts constraints (time windows, domains)
    - sets top-k/budgets
    - defines dependencies and stop criteria
    - and may replan if evidence is weak
## 3. Executor
```python
def execute_plan(plan):
    evidence = {}
    for step in plan:
        if step["tool"] == "vector":
            evidence[step["id"]] = vector_kb_search(step["query"])
        ...
    return evidence

```
- Runs each step and **maps evidence by sub-id** (`Q1`, `Q2`, `Q3`).
- In production:
    - Run **independent** steps in **parallel** (async) for latency.
    - Attach **provenance** (source URL, doc id, chunk id, score).
    - Capture **cost/latency** metrics per step for observability.
## 4. Synthesizer
```python
def synthesize_answer(user_query, evidence):
    kb = " ".join(evidence.get("Q1", []))
    tickets = " ".join(evidence.get("Q2", []))
    studies = " ".join(evidence.get("Q3", []))
    ...
    return (f"User query: {user_query}\n\n"
            f"Evidence — KB: {kb}\n..."

```
- Turns evidence into a **defensible recommendation**:
    - Chooses **HYBRID** (BM25 + vector) and justifies why.
    - Outputs **metrics** to track (top1-hit, MRR@k, etc.) and **risks** (index drift, cost).
- In production, this would be an LLM “generator” node that:
    - 
## 5. Run wrapper + file capture
```python
from contextlib import redirect_stdout

def run_agentic_RAG():
    with open("./output_results/agenticRAGDemo.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            user_q = ("Compare vector search vs keyword search ...")
            plan = planner(user_q)
            evidence = execute_plan(plan)
            print("PLAN:", plan, "\n")
            print(synthesize_answer(user_q, evidence))

```
- What the output file contains (typical)
```text
PLAN: [{'id': 'Q1', 'tool': 'vector', 'query': 'KB overview and domains'},
       {'id': 'Q2', 'tool': 'sql', 'query': 'Ticket analytics for last 6 months (mix, paraphrase rate)'},
       {'id': 'Q3', 'tool': 'web', 'query': 'BM25 vs dense retrieval for FAQs; evidence on hybrid'}]

User query: Compare vector search vs keyword search for our support KB ...

Evidence — KB: Our KB covers billing, account, and product setup. Many duplicate Q&A.
Evidence — Tickets: Tickets: 42% billing, 35% setup, 23% account; 28% contain paraphrased intents
Evidence — Studies: Study A: hybrid improves top1-hit by 8–15% on FAQ corpora Case study: vector-only strong on paraphrases; BM25 strong on short keyword queries

Recommendation: adopt a HYBRID retriever (BM25 + vector)...
Track: top1-hit, MRR@k, ...
Risks: index drift ...

```