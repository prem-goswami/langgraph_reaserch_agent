from langgraph.graph import StateGraph, START, END

from warmup.state import WarmupState
from warmup.nodes import router, search, generation

def route_decision(state: WarmupState) -> str:
    """Conditional edge. Reads state, returns the name of the next branch."""
    decision = state["route"]
    print(f"[edge]       route_decision -> {decision!r}")
    return decision

def build_graph():
    builder = StateGraph(WarmupState)

    builder.add_node("router", router)
    builder.add_node("search", search)
    builder.add_node("generation", generation)

    builder.add_edge(START, "router")

    builder.add_conditional_edges(
        "router",
        route_decision,
        {
            "search": "search",
            "generate": "generation",
        },
    )

    builder.add_edge("search", "generation")
    builder.add_edge("generation", END)

    return builder.compile()




