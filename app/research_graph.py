from typing import List, Union
from langgraph.graph import StateGraph, START, END
from app.config import MAX_CORRECTIONS
from app.research_state import ResearchState
from app.research_nodes import (
    query_analyzer, tavily_search, rag_retrieval, merge_and_rank, critic, 
)


RETRY_TARGETS = ["tavily_search", "rag_retrieval"]


def route_after_critic(state: ResearchState) -> Union[str, List[str]]:
    """Read-only. Returns the next target(s): a node name, END, or a list to fan out."""
    verdict = state["critic_verdict"]
    count = state["correction_count"]

    if verdict == "PASS" or count > MAX_CORRECTIONS:
        print(f"[edge]       verdict={verdict} count={count} -> generate")
        return END                    # Phase 5: change to "generation"

    print(f"[edge]       verdict={verdict} count={count} -> retry")
    return RETRY_TARGETS

def build_research_graph():
    builder = StateGraph(ResearchState)

    builder.add_node("query_analyzer", query_analyzer)
    builder.add_node("tavily_search", tavily_search)
    builder.add_node("rag_retrieval", rag_retrieval)
    builder.add_node("merge_and_rank", merge_and_rank)
    builder.add_node("critic", critic)

    builder.add_edge(START, "query_analyzer")

    # FAN-OUT: two edges from one node -> both run in the same superstep
    builder.add_edge("query_analyzer", "tavily_search")
    builder.add_edge("query_analyzer", "rag_retrieval")

    # FAN-IN: both edges into one node -> runs once, after both complete
    builder.add_edge("tavily_search", "merge_and_rank")
    builder.add_edge("rag_retrieval", "merge_and_rank")

    builder.add_edge("merge_and_rank", "critic")

    # THE CYCLE: "retry" sends control back to BOTH retrieval nodes
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
    )


    return builder.compile()