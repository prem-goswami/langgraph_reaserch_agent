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


def initial_state(question: str) -> ResearchState:
    """Construct a fully-seeded initial state. The single place keys are defaulted."""
    return {
        "question": question,
        "sub_questions": [],
        "raw_results": [],
        "ranked_results": [],
        "critic_verdict": "",
        "critic_feedback": "",
        "correction_count": 0,
        "report": "",
    }