from app.state import WarmupState
from langchain_core.messages import SystemMessage, HumanMessage
from app.config import ROUTER_MODEL, VALID_ROUTES, DEFAULT_ROUTE
from app.llm import get_llm


ROUTER_SYSTEM = """You are a routing classifier. Decide whether answering the \
user's question requires a live web search.

Reply with exactly one word: search or generate

Choose "search" when the answer depends on information that changes over time:
- current events, news, prices, weather, sports results
- anything about 2024 or later
- questions containing "latest", "current", "now", "today", "recent"
- the status of a named company, person, or product
- specific software versions or releases

Choose "generate" when the answer is stable knowledge that does not change:
- definitions, concepts, and explanations
- mathematics, logic, and established science
- historical facts from before 2024
- programming language syntax and semantics

If you are unsure, reply search. An unnecessary search costs a second. \
A confidently outdated answer costs the user's trust.

Reply with the single word only. No punctuation. No explanation."""

def _normalise_route(raw: str) -> str:
    """Coerce arbitrary model output into a valid route name."""
    cleaned = raw.strip().strip('.`"\'').lower()

    if cleaned in VALID_ROUTES:
        return cleaned

    for token in cleaned.split():
        token = token.strip('.`"\',')
        if token in VALID_ROUTES:
            return token

    print(f"[router]     WARN unparseable route {raw!r} -> {DEFAULT_ROUTE!r}")
    return DEFAULT_ROUTE

def router(state: WarmupState) -> dict:
    """Decide whether this question needs live web search."""
    question = state["question"]
    llm = get_llm(ROUTER_MODEL, temperature=0)

    response = llm.invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=question)
    ])

    print(f"llm-response: {response.content}")

    raw = response.content

    route = _normalise_route(raw)
    print(f"[router]     raw={raw!r} -> route={route!r}")
    return {"route": route}

def search(state: WarmupState) -> dict:
    """Fetch web results. STUB — returns fixed data."""
    print(f"[search]     searching for: {state['question'][:50]!r}")
    return {
        "search_results": [
            {"title": "Stub result A", "url": "https://example.com/a",
             "content": "Placeholder content A."},
            {"title": "Stub result B", "url": "https://example.com/b",
             "content": "Placeholder content B."},
        ]
    }

def generation(state: WarmupState) -> dict:
    """Write the final answer. STUB — echoes what it received."""
    n = len(state["search_results"])
    print(f"[generation] composing answer from {n} result(s)")
    return {"answer": f"Answer to {state['question']!r} using {n} source(s)."}