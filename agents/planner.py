import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from core.state import AgentState
from core.events import emit, EventType

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SYSTEM_PROMPT = """You are a research planning agent. Your job is to decompose a 
research question into 4-6 distinct sub-queries that together provide full coverage 
of the topic.

Each sub-query should target a different facet:
- Core methodology / technical approach
- Specific applications or use cases
- Comparisons with alternative approaches
- Known limitations or failure modes
- Open problems or future directions
- Empirical results / benchmarks

Rules:
- Sub-queries must be SHORT keyword strings suitable for arXiv search (5-8 words max)
- No full sentences or question format
- No two sub-queries should overlap significantly
- Return ONLY a JSON array of strings, no preamble, no markdown fences
"""

def planner_node(state: AgentState) -> AgentState:
    emit(EventType.STAGE_START, stage="planner", message="Decomposing research question...")
    emit(EventType.PLANNING, question=state["question"])

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=512,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Decompose this research question into sub-queries: {state['question']}"}
        ]
    )

    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    sub_queries = json.loads(raw.strip())

    print(f"[planner] decomposed into {len(sub_queries)} sub-queries:")
    for i, q in enumerate(sub_queries, 1):
        print(f"  {i}. {q}")

    emit(EventType.SUB_QUERIES, queries=sub_queries)
    emit(EventType.STAGE_END, stage="planner", count=len(sub_queries))

    return {**state, "sub_queries": sub_queries}