from langgraph.graph import StateGraph, START, END

from app.research_state import ResearchState
from app.research_nodes import (
    query_analyzer, tavily_search, rag_retrieval, merge_and_rank,
)


def build_research_graph():
    builder = StateGraph(ResearchState)

    builder.add_node("query_analyzer", query_analyzer)
    builder.add_node("tavily_search", tavily_search)
    builder.add_node("rag_retrieval", rag_retrieval)
    builder.add_node("merge_and_rank", merge_and_rank)

    builder.add_edge(START, "query_analyzer")

    # FAN-OUT: two edges from one node -> both run in the same superstep
    builder.add_edge("query_analyzer", "tavily_search")
    builder.add_edge("query_analyzer", "rag_retrieval")

    # FAN-IN: both edges into one node -> runs once, after both complete
    builder.add_edge("tavily_search", "merge_and_rank")
    builder.add_edge("rag_retrieval", "merge_and_rank")

    builder.add_edge("merge_and_rank", END)

    return builder.compile()