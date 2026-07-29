from app.state import WarmupState


def router(state: WarmupState) -> dict:
    """Decide whether this question needs live web search."""
    print(f"[router]     question={state['question'][:50]!r}")
    return {"route": "search"}

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