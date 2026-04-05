# seed.py
import time
from agents.retrieval import fetch_arxiv
from db.vector import init_db, upsert_papers

init_db()

queries = [
    "evolutionary algorithms neural architecture search",
    "genetic programming agent optimization",
    "neuroevolution deep learning",
    "evolutionary reinforcement learning",
    "LLM agent workflow orchestration",
]

for i, q in enumerate(queries):
    print(f"fetching: {q}")
    papers = fetch_arxiv(q, max_results=10)
    upsert_papers(papers)
    print(f"  upserted {len(papers)} papers")
    if i < len(queries) - 1:
        time.sleep(10)

print("done seeding")