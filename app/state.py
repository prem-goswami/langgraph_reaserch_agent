from typing import TypedDict

class WarmupState(TypedDict):
    question: str
    route: str
    search_results: list[dict]
    answer: str