# ----------------------------
# COMPLEX A/B TESTING DEMO
# ----------------------------
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
from contextlib import redirect_stdout

load_dotenv()
# Initialize LLM and embedding model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# ----------------------------
# INPUT TEXT (Example Passage)
# ----------------------------
text = """
Knowledge Distillation is a model compression technique where a smaller and faster student model learns
to mimic the behavior of a larger teacher model. The process transfers knowledge from the teacher to
the student by minimizing the divergence between their predicted probability distributions. This allows
deploying efficient models in real-world applications without losing much accuracy.
"""

# ----------------------------
# THREE PROMPTS (A/B/C)
# ----------------------------
prompt_A = f"Summarize the following text in one paragraph:\n{text}"
prompt_B = f"Summarize the following text in 3 concise bullet points focusing on key technical ideas:\n{text}"
prompt_C = f"Summarize the text focusing on practical benefits and use-cases of Knowledge Distillation:\n{text}"

prompts = {"A": prompt_A, "B": prompt_B, "C": prompt_C}

# ----------------------------
# IDEAL REFERENCE SUMMARY
# ----------------------------
reference = """
Knowledge Distillation enables smaller models to achieve near-teacher accuracy by learning from the
teacher’s soft predictions. It’s used to deploy efficient models in production with lower latency and
resource costs.
"""




def AB_testing():
    # ----------------------------
    # STEP 1: GENERATE MODEL RESPONSES
    # ----------------------------
    outputs = {}
    for label, p in prompts.items():
        print(f"\n🧠 Running Prompt {label}...")
        outputs[label] = llm.invoke([("human", p)]).content

    # ----------------------------
    # STEP 2: COMPUTE SEMANTIC SIMILARITY (Relevance)
    # ----------------------------
    ref_emb = embedder.embed_query(reference)
    scores = {}

    for label, output in outputs.items():
        out_emb = embedder.embed_query(output)
        sim = cosine_similarity([ref_emb], [out_emb])[0][0]
        scores[label] = {"relevance": sim}

    # ----------------------------
    # STEP 3: MEASURE CONCISENESS
    # ----------------------------
    for label, output in outputs.items():
        word_count = len(output.split())
        conciseness = max(0, 1 - (word_count - 60) / 100)  # penalize long answers
        scores[label]["conciseness"] = round(conciseness, 3)


    # ----------------------------
    # STEP 4: ESTIMATE READABILITY
    # (Simplified Flesch-like score)
    # ----------------------------
    def readability_score(text):
        words = text.split()
        sentences = max(1, text.count('.') + text.count('!') + text.count('?'))
        avg_words_per_sentence = len(words) / sentences
        return max(0, min(1, 1 - (avg_words_per_sentence - 15) / 25))  # normalize between 0–1

    for label, output in outputs.items():
        scores[label]["readability"] = round(readability_score(output), 3)
    # ----------------------------
    # STEP 5: COMBINE INTO FINAL SCORE
    # ----------------------------
    for label in scores:
        s = scores[label]
        s["final_score"] = round(0.5*s["relevance"] + 0.3*s["conciseness"] + 0.2*s["readability"], 3)

    # ----------------------------
    # STEP 6: DISPLAY RESULTS
    # ----------------------------
    print("\n📊 COMPLEX A/B TEST RESULTS")
    for label, s in scores.items():
        print(f"\nPrompt {label}:")
        print(f"Relevance:   {s['relevance']:.3f}")
        print(f"Conciseness: {s['conciseness']:.3f}")
        print(f"Readability: {s['readability']:.3f}")
        print(f"➡️ Final Score: {s['final_score']:.3f}")
        print(f"\nOutput:\n{outputs[label]}")

    best = max(scores.items(), key=lambda x: x[1]["final_score"])
    print(f"\n🏆 Best performing prompt: {best[0]} (score = {best[1]['final_score']:.3f})")

def redirect_AB_testing():
    with open("./output_results/PromptEngineering/ABtesting.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            AB_testing()