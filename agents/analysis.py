import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv
from core.state import AgentState, ClaimObject
from core.events import emit, EventType

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SYSTEM_PROMPT = """You are a research analysis agent. Given a paper title and abstract,
extract structured information.

Return ONLY a JSON object with these exact keys:
{
  "core_claims": ["list of 2-3 main claims the paper makes"],
  "methodology": "one of: empirical/theoretical/survey/benchmark/system",
  "key_findings": ["list of 2-3 concrete findings with numbers where available"],
  "limitations": ["list of 1-2 limitations the paper states or implies"]
}

Be specific and grounded in the text. No preamble, no markdown fences.
"""

def analyze_paper(paper: dict) -> ClaimObject | None:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=400,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"}
            ]
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        data = json.loads(raw.strip())

        return ClaimObject(
            paper_id=paper["id"],
            title=paper["title"],
            core_claims=data["core_claims"],
            methodology=data["methodology"],
            key_findings=data["key_findings"],
            limitations=data["limitations"]
        )

    except Exception as e:
        print(f"[analysis] failed on '{paper['title']}': {e}")
        return None


def analysis_node(state: AgentState) -> AgentState:
    """Analyze papers in parallel using ThreadPoolExecutor."""
    papers = state["papers"]
    print(f"[analysis] received {len(papers)} papers")

    emit(EventType.STAGE_START, stage="analysis", message="Analyzing papers...", total=len(papers))

    if not papers:
        emit(EventType.STAGE_END, stage="analysis", count=0)
        return {**state, "claims": []}

    claims = []
    completed = 0

    # Use 10 workers - balances speed vs API rate limits
    # OpenAI gpt-4o-mini allows ~3,500 RPM, so 10 concurrent is safe
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_paper = {
            executor.submit(analyze_paper, paper): paper
            for paper in papers
        }

        for future in as_completed(future_to_paper):
            completed += 1
            paper = future_to_paper[future]
            try:
                result = future.result()
                if result:
                    claims.append(result)
                print(f"[analysis] {completed}/{len(papers)}: {paper['title'][:60]}...")
                emit(
                    EventType.PAPER_COMPLETE,
                    title=paper["title"],
                    progress=completed,
                    total=len(papers),
                )
            except Exception as e:
                print(f"[analysis] {completed}/{len(papers)} failed: {e}")
                emit(
                    EventType.PAPER_FAILED,
                    title=paper["title"],
                    error=str(e),
                    progress=completed,
                    total=len(papers),
                )

    print(f"[analysis] extracted claims from {len(claims)}/{len(papers)} papers")
    emit(EventType.STAGE_END, stage="analysis", count=len(claims))
    return {**state, "claims": claims}