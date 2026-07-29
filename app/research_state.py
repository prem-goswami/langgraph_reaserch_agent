from typing import TypedDict, Annotated
import operator


class ResearchState(TypedDict):
    question: str
    sub_questions: list[str]
    raw_results: Annotated[list[dict], operator.add]
    ranked_results: list[dict]
    critic_verdict: str
    critic_feedback: str
    correction_count: int
    report: str