# ---------- toy “Agentic-RAG” demo (no external deps) ----------
from typing import Dict, List, Any
from contextlib import redirect_stdout

# --- Fake tools (replace with real retrievers later) ---
def vector_kb_search(query: str) -> List[str]:
    corpus = {
        "kb_overview": "Our KB covers billing, account, and product setup. Many duplicate Q&A.",
        "semantic_benefit": "Dense retrieval handles paraphrases; helps long, fuzzy queries.",
        "hybrid_tip": "Hybrid (BM25 + vector) improves first-hit for head and tail queries."
    }
    if "overview" in query: return [corpus["kb_overview"]]
    if "semantic" in query: return [corpus["semantic_benefit"]]
    if "hybrid" in query: return [corpus["hybrid_tip"]]
    return ["No vector hit"]

def sql_ticket_analytics(query: str) -> List[str]:
    if "6 months" in query:
        return ["Tickets: 42% billing, 35% setup, 23% account; 28% contain paraphrased intents"]
    return ["No analytics"]

def web_research(query: str) -> List[str]:
    if "BM25 vs dense" in query:
        return ["Study A: hybrid improves top1-hit by 8–15% on FAQ corpora",
                "Case study: vector-only strong on paraphrases; BM25 strong on short keyword queries"]
    return ["No web sources"]

# --- Planner: break into sub-queries and route to tools ---
def planner(user_query: str) -> List[Dict[str, Any]]:
    return [
        {"id": "Q1", "tool": "vector", "query": "KB overview and domains"},
        {"id": "Q2", "tool": "sql",    "query": "Ticket analytics for last 6 months (mix, paraphrase rate)"},
        {"id": "Q3", "tool": "web",    "query": "BM25 vs dense retrieval for FAQs; evidence on hybrid"}
    ]

# --- Executor: run subtasks with the chosen tools ---
def execute_plan(plan: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    evidence = {}
    for step in plan:
        if step["tool"] == "vector":
            evidence[step["id"]] = vector_kb_search(step["query"])
        elif step["tool"] == "sql":
            evidence[step["id"]] = sql_ticket_analytics(step["query"])
        elif step["tool"] == "web":
            evidence[step["id"]] = web_research(step["query"])
    return evidence

# --- Synthesizer: merge evidence into a recommendation ---
def synthesize_answer(user_query: str, evidence: Dict[str, List[str]]) -> str:
    kb = " ".join(evidence.get("Q1", []))
    tickets = " ".join(evidence.get("Q2", []))
    studies = " ".join(evidence.get("Q3", []))

    rec = (
        "Recommendation: adopt a HYBRID retriever (BM25 + vector). "
        "Rationale: BM25 excels for short keyword queries and known FAQs; "
        "vector handles paraphrases and semantically vague tickets. "
        "Given ~28% paraphrased intents, hybrid balances precision and recall."
    )
    metrics = (
        "Track: top1-hit, MRR@k, deflection rate, mean handle time, and failure analysis by intent type. "
        "Run a 2-week A/B with 50/50 traffic; success = +10% top1-hit and -8% handle time."
    )
    risks = (
        "Risks: index drift (embed model/versioning), cost spikes from dense search, "
        "and degraded performance on very short queries if vector-only routing is used."
    )
    return (
        f"User query: {user_query}\n\n"
        f"Evidence — KB: {kb}\nEvidence — Tickets: {tickets}\nEvidence — Studies: {studies}\n\n"
        f"{rec}\n{metrics}\n{risks}"
    )

# --- Full run ---
def run_agentic_RAG():
    with open("./output_results/agenticRAGDemo.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            user_q = ("Compare vector search vs keyword search for our support KB and recommend a strategy, "
                    "metrics to track, and a rollout plan with risks.")
            plan = planner(user_q)
            evidence = execute_plan(plan)
            print("PLAN:", plan, "\n")
            print(synthesize_answer(user_q, evidence))
