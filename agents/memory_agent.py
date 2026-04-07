"""
Memory-aware agent node that manages context and archival.

This agent runs at key points in the pipeline to:
1. Inject relevant memories into context
2. Archive important findings
3. Manage context window size
"""
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from core.state import AgentState
from core.memory import get_memory, MemoryManager
from core.events import emit, EventType

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ─── Memory Injection ─────────────────────────────────────────────────────────

def inject_memories(state: AgentState) -> AgentState:
    """
    Inject relevant memories at the start of a session.
    Runs before the planner to provide context from past sessions.
    """
    memory = get_memory()
    question = state["question"]

    # Recall relevant past knowledge
    relevant = memory.recall(question, top_k=5)

    if relevant:
        print(f"[memory] recalled {len(relevant)} relevant memories")
        emit(EventType.STAGE_START, stage="memory", message="Recalling past knowledge...")

        # Add memories to context (could be injected into planner prompt)
        memory_context = []
        for entry in relevant:
            memory_context.append({
                "type": entry.memory_type,
                "content": entry.content,
                "source": entry.source,
            })

        emit(EventType.STAGE_END, stage="memory", memories_recalled=len(relevant))

        return {
            **state,
            "memory_context": memory_context,
        }

    return state


# ─── Memory Extraction ────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are a memory extraction agent. Given research findings,
extract the most important insights worth remembering for future sessions.

For each insight, provide:
- content: The insight itself (1-2 sentences, self-contained)
- importance: 0.0-1.0 (1.0 = critical, always remember; 0.5 = useful; 0.0 = trivial)
- type: "insight" | "fact" | "methodology" | "limitation" | "open_question"

Return JSON array: [{"content": "...", "importance": 0.8, "type": "insight"}, ...]

Extract 3-7 key memories. Focus on:
- Novel findings that challenge assumptions
- Methodological insights
- Cross-paper patterns
- Open questions worth exploring
- Limitations to be aware of

No preamble, no markdown fences.
"""


def extract_memories(state: AgentState) -> AgentState:
    """
    Extract important memories from analysis results.
    Runs after synthesis to archive key findings.
    """
    memory = get_memory()
    claims = state.get("claims", [])
    report = state.get("report")
    question = state["question"]

    if not claims and not report:
        return state

    emit(EventType.STAGE_START, stage="memory_extraction", message="Extracting memories...")

    # Build context for extraction
    context = {
        "question": question,
        "num_papers": len(state.get("papers", [])),
        "claims_sample": [
            {"title": c["title"], "claims": c["core_claims"][:2]}
            for c in claims[:10]
        ],
    }

    # Add report summary if available
    if report:
        try:
            report_data = json.loads(report)
            context["executive_summary"] = report_data.get("executive_summary", "")
            context["key_claims"] = report_data.get("key_claims", [])[:3]
            context["open_problems"] = report_data.get("open_problems", [])[:3]
        except json.JSONDecodeError:
            pass

    # Extract memories using LLM
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=600,
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": json.dumps(context, indent=2)}
            ]
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        memories = json.loads(raw.strip())

        # Store each extracted memory
        stored_count = 0
        for mem in memories:
            if mem.get("importance", 0) >= 0.4:
                memory.remember(
                    content=mem["content"],
                    memory_type=mem.get("type", "insight"),
                    source=f"analysis:{question[:30]}",
                    importance=mem["importance"],
                )
                stored_count += 1

        print(f"[memory] extracted and stored {stored_count} memories")
        emit(EventType.STAGE_END, stage="memory_extraction", memories_stored=stored_count)

    except Exception as e:
        print(f"[memory] extraction failed: {e}")

    return state


# ─── Context Summarization ────────────────────────────────────────────────────

SUMMARIZATION_PROMPT = """Summarize the current research context concisely.
This summary will replace the detailed context to save space.

Include:
1. The research question
2. Key findings so far (3-5 bullet points)
3. What's still unknown
4. Most promising directions

Keep it under 500 words. Be specific and factual.
"""


def summarize_context(state: AgentState) -> AgentState:
    """
    Compress context when it gets too large.
    Triggered when token count exceeds threshold.
    """
    claims = state.get("claims", [])

    # Check if we need summarization (rough token estimate)
    estimated_tokens = len(json.dumps(claims)) // 4

    if estimated_tokens < 6000:
        return state  # No summarization needed

    print(f"[memory] context too large ({estimated_tokens} tokens), summarizing...")
    emit(EventType.STAGE_START, stage="summarization", message="Compressing context...")

    # Build context for summarization
    context = {
        "question": state["question"],
        "papers_analyzed": len(state.get("papers", [])),
        "claims": [
            {
                "title": c["title"],
                "core_claims": c["core_claims"],
                "methodology": c["methodology"],
            }
            for c in claims
        ],
        "context_summary": state.get("context_summary"),
    }

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=800,
            messages=[
                {"role": "system", "content": SUMMARIZATION_PROMPT},
                {"role": "user", "content": json.dumps(context, indent=2)}
            ]
        )

        summary = response.choices[0].message.content.strip()

        # Archive the full claims before replacing
        memory = get_memory()
        memory.remember(
            content=f"Detailed analysis of {len(claims)} papers on: {state['question'][:50]}",
            memory_type="archived_analysis",
            source="summarization",
            importance=0.6,
            metadata={"claims_count": len(claims)},
        )

        # Replace detailed claims with summary
        # Keep only the most important claims
        top_claims = sorted(claims, key=lambda c: len(c.get("core_claims", [])), reverse=True)[:10]

        print(f"[memory] compressed {len(claims)} claims to {len(top_claims)} + summary")
        emit(EventType.STAGE_END, stage="summarization", original=len(claims), compressed=len(top_claims))

        return {
            **state,
            "claims": top_claims,
            "compressed_summary": summary,
        }

    except Exception as e:
        print(f"[memory] summarization failed: {e}")
        return state


# ─── Memory Stats ─────────────────────────────────────────────────────────────

def get_memory_stats() -> dict:
    """Get current memory statistics."""
    return get_memory().get_stats()
