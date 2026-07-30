from typing import List, Literal
from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class SourceOut(BaseModel):
    source_type: Literal["web", "doc"]
    title: str
    url: str
    score: float
    found_by: str = ""


class ResearchResponse(BaseModel):
    question: str
    report: str
    sub_questions: List[str]
    sources: List[SourceOut]
    critic_verdict: str
    critic_feedback: str
    correction_count: int
    elapsed_seconds: float


class HealthResponse(BaseModel):
    status: Literal["ok"]
    rag_reachable: bool
    rag_url: str
    max_corrections: int


class GraphResponse(BaseModel):
    mermaid: str
    nodes: List[str]