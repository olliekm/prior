"""
Reflector agent - analyzes current findings and decides if more research is needed.

This implements the "self-summarization for long contexts" pattern:
1. Summarize what we've found so far
2. Identify gaps in our understanding
3. Decide whether to loop back for more papers or proceed to synthesis
"""
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from core.state import AgentState, ContextSummary
from core.events import emit, EventType

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SYSTEM_PROMPT = """You are a research reflection agent. Your job is to analyze the current
state of a literature review and decide if more information is needed.

Given:
- The original research question
- Papers analyzed so far (with their claims)
- Previous context summary (if any)
- Queries already searched

Produce a JSON response with:
{
  "summary": {
    "findings": ["list of 3-5 key things we've learned so far"],
    "gaps": ["list of 1-3 important gaps in our understanding"],
    "ruled_out": ["hypotheses or approaches we can eliminate based on evidence"],
    "follow_up_queries": ["2-3 specific arXiv search queries to fill the gaps"]
  },
  "needs_more_info": true/false,
  "reasoning": "brief explanation of your decision"
}

Set needs_more_info to TRUE if:
- Major gaps exist that could be filled with targeted searches
- Key claims are unsupported or contradictory
- The original question has important facets we haven't explored
- Follow-up queries are likely to yield valuable new papers

Set needs_more_info to FALSE if:
- We have good coverage of the main facets
- Additional searches would likely return duplicates
- We have enough evidence to synthesize a useful report
- We've already done multiple iterations

Be conservative - don't loop forever. If in doubt after 2+ iterations, proceed to synthesis.
"""


def reflector_node(state: AgentState) -> AgentState:
    """Analyze findings and decide whether to search for more papers."""
    iteration = state.get("iteration", 1)
    max_iterations = state.get("max_iterations", 3)
    claims = state.get("claims", [])
    searched = state.get("searched_queries", [])
    prev_summary = state.get("context_summary")

    # Hard stop: if we've hit max iterations, proceed to synthesis immediately
    if iteration >= max_iterations:
        print(f"[reflector] max iterations ({max_iterations}) reached, proceeding to synthesis")
        emit(EventType.STAGE_END, stage="reflector", iteration=iteration, needs_more_info=False, reason="max_iterations")
        return {
            **state,
            "needs_more_info": False,
            "iteration": iteration,
        }

    emit(EventType.STAGE_START, stage="reflector", message="Reflecting on findings...")

    # Build context for the LLM
    claims_summary = []
    for c in claims[:20]:  # Limit to avoid token overflow
        claims_summary.append({
            "title": c["title"],
            "claims": c["core_claims"][:2],
            "methodology": c["methodology"],
        })

    context = {
        "question": state["question"],
        "iteration": iteration,
        "max_iterations": max_iterations,
        "papers_analyzed": len(claims),
        "claims_sample": claims_summary,
        "previous_summary": prev_summary,
        "searched_queries": searched,
    }

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Faster model for reflection
        max_tokens=800,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(context, indent=2)}
        ]
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        result = json.loads(raw.strip())
    except json.JSONDecodeError:
        # If parsing fails, default to proceeding to synthesis
        print(f"[reflector] failed to parse response, proceeding to synthesis")
        return {
            **state,
            "needs_more_info": False,
            "iteration": iteration,
        }

    summary = ContextSummary(
        findings=result["summary"]["findings"],
        gaps=result["summary"]["gaps"],
        ruled_out=result["summary"]["ruled_out"],
        follow_up_queries=result["summary"]["follow_up_queries"],
    )

    # Be conservative: only loop if LLM says yes AND we have room for more iterations
    # (iteration is 1-indexed, so iteration 2 means we've done 2 cycles)
    needs_more = result.get("needs_more_info", False) and (iteration + 1) < max_iterations

    print(f"[reflector] iteration {iteration}/{max_iterations}")
    print(f"[reflector] findings: {len(summary['findings'])}, gaps: {len(summary['gaps'])}")
    print(f"[reflector] needs_more_info: {needs_more}")
    if needs_more:
        print(f"[reflector] follow-up queries: {summary['follow_up_queries']}")

    emit(
        EventType.STAGE_END,
        stage="reflector",
        iteration=iteration,
        needs_more_info=needs_more,
        gaps=summary["gaps"],
    )

    # If we need more info, add follow-up queries to sub_queries
    new_queries = state.get("sub_queries", [])
    if needs_more:
        # Add new queries that we haven't searched yet
        for q in summary["follow_up_queries"]:
            if q not in searched:
                new_queries.append(q)

    return {
        **state,
        "context_summary": summary,
        "needs_more_info": needs_more,
        "iteration": iteration + 1,
        "sub_queries": new_queries,
    }


def should_continue(state: AgentState) -> str:
    """Conditional edge: decide whether to loop back or proceed to synthesis."""
    if state.get("needs_more_info", False):
        return "retrieval"
    return "synthesis"
