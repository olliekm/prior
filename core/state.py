from typing import Optional, Any
from typing_extensions import TypedDict


class ClaimObject(TypedDict):
    paper_id: str
    title: str
    core_claims: list[str]
    methodology: str
    key_findings: list[str]
    limitations: list[str]


class ContextSummary(TypedDict):
    """Rolling summary of what we've found so far."""
    findings: list[str]          # Key things we've learned
    gaps: list[str]              # What we still need to find
    ruled_out: list[str]         # Hypotheses we've eliminated
    follow_up_queries: list[str] # Suggested next searches


class MemoryContext(TypedDict):
    """Memory recalled from past sessions."""
    type: str
    content: str
    source: str


class AgentState(TypedDict):
    question: str
    sub_queries: list[str]
    papers: list[dict[str, Any]]
    claims: list[ClaimObject]
    report: Optional[str]

    # Adaptive loop fields
    iteration: int                          # Current loop count
    max_iterations: int                     # Loop limit (default 3)
    context_summary: Optional[ContextSummary]  # Rolling summary
    needs_more_info: bool                   # Flag from reflector
    searched_queries: list[str]             # Track what we've already searched

    # Memory fields
    memory_context: list[MemoryContext]     # Recalled memories from past sessions
    compressed_summary: Optional[str]        # Compressed context when too large   