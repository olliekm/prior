import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from core.events import emit, EventType

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SYSTEM_PROMPT = """You are a research synthesis agent. Given a list of analyzed papers
as structured ClaimObjects, produce a structured literature review.

Return ONLY a JSON object with these exact keys:
{
  "executive_summary": "3-5 sentence overview of what the field knows",
  "key_claims": [
    {"claim": "...", "supporting_papers": ["paper title 1", "paper title 2"]}
  ],
  "contested_claims": [
    {"claim": "...", "side_a": ["paper titles"], "side_b": ["paper titles"], "reason": "why they disagree"}
  ],
  "methodology_breakdown": {
    "empirical": 0,
    "theoretical": 0,
    "survey": 0,
    "benchmark": 0,
    "system": 0
  },
  "open_problems": ["list of open problems the literature identifies"],
  "suggested_queries": ["2-3 follow-up search queries worth exploring"]
}

Be specific. Ground every claim in the papers provided. No preamble, no markdown fences.
"""

def synthesis_node(state: dict) -> dict:
    claims = state["claims"]

    emit(EventType.STAGE_START, stage="synthesis", message="Synthesizing findings...")
    emit(EventType.SYNTHESIZING, claims_count=len(claims))

    # format claims as readable context for the model
    claims_text = json.dumps(claims, indent=2)

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=2000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Research question: {state['question']}\n\nAnalyzed papers:\n{claims_text}"}
        ]
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    report = json.loads(raw.strip())
    emit(EventType.STAGE_END, stage="synthesis")

    # pretty print the report
    print("\n" + "="*60)
    print("PRIOR — LITERATURE REVIEW")
    print("="*60)
    print(f"\nQ: {state['question']}\n")
    print("EXECUTIVE SUMMARY")
    print(report["executive_summary"])

    print("\nKEY CLAIMS")
    for kc in report["key_claims"]:
        print(f"  • {kc['claim']}")
        for p in kc["supporting_papers"]:
            print(f"      ↳ {p}")

    print("\nCONTESTED CLAIMS")
    if report["contested_claims"]:
        for cc in report["contested_claims"]:
            print(f"  • {cc['claim']}")
            print(f"    reason: {cc['reason']}")
    else:
        print("  none identified")

    print("\nMETHODOLOGY BREAKDOWN")
    for k, v in report["methodology_breakdown"].items():
        print(f"  {k}: {v}")

    print("\nOPEN PROBLEMS")
    for op in report["open_problems"]:
        print(f"  • {op}")

    print("\nSUGGESTED FOLLOW-UP QUERIES")
    for sq in report["suggested_queries"]:
        print(f"  → {sq}")

    print("="*60)

    return {**state, "report": json.dumps(report)}