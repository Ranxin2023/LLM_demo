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
## 2. Planner
```python
def planner(user_query: str) -> List[Dict[str, Any]]:
    return [
        {"id": "Q1", "tool": "vector", "query": "KB overview and domains"},
        {"id": "Q2", "tool": "sql",    "query": "Ticket analytics for last 6 months (mix, paraphrase rate)"},
        {"id": "Q3", "tool": "web",    "query": "BM25 vs dense retrieval for FAQs; evidence on hybrid"}
    ]

```