from typing import Optional, Any
from typing_extensions import TypedDict

class ClaimObject(TypedDict):
    paper_id: str
    title: str
    core_claims: list[str]
    methodology: str
    key_findings: list[str]
    limitations: list[str]

class AgentState(TypedDict):
    question: str
    sub_queries: list[str]
    papers: list[dict[str, Any]]
    claims: list[ClaimObject]
    report: Optional[str]   