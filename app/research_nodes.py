import asyncio
import math
import httpx
from tavily import AsyncTavilyClient
from app.contract import node
from app.config import (
    TAVILY_API_KEY, TAVILY_MAX_RESULTS, TAVILY_SEARCH_DEPTH,
    RAG_BASE_URL, RAG_MIN_SCORE, RAG_TIMEOUT,
)

from app.research_state import ResearchState


def _sigmoid(logit: float) -> float:
    """Cross-encoder logit -> calibrated relevance probability in [0, 1]."""
    return 1.0 / (1.0 + math.exp(-logit))


def _as_web_result(hit: dict) -> dict:
    return {
        "source_type": "web",
        "title": hit.get("title") or "(untitled)",
        "url": hit.get("url") or "",
        "content": (hit.get("content") or "").strip(),
        "score": float(hit.get("score") or 0.0),
    }


def _as_doc_result(chunk: dict) -> dict:
    return {
        "source_type": "doc",
        "title": chunk.get("source") or "(document)",
        "url": chunk.get("chunk_id") or "",
        "content": (chunk.get("content_preview") or "").strip(),
        "score": _sigmoid(float(chunk.get("rerank_score") or -20.0)),
    }

@node
def query_analyzer(state: ResearchState) -> dict:
    """Split the question into sub-questions. STUB."""
    q = state["question"]
    subs = [q, f"{q} recent developments"]
    print(f"[analyzer]   -> {len(subs)} sub-question(s)")
    return {"sub_questions": subs}


@node
async def tavily_search(state: ResearchState) -> dict:
    """Web retrieval across all sub-questions concurrently (Simplified Version)."""
    sub_questions_list = state["sub_questions"]
    print(f"[tavily]     START  ({len(sub_questions_list)} sub-question(s))")

    try:
        client = AsyncTavilyClient(api_key=TAVILY_API_KEY)

        # Helper function: handles web search for a SINGLE query string
        async def search_single_query(query_text: str) -> list[dict]:
            response = await client.search(
                query=query_text,
                max_results=TAVILY_MAX_RESULTS,
                search_depth=TAVILY_SEARCH_DEPTH,
            )
            raw_hits = response.get("results", [])
            formatted_hits = [_as_web_result(hit) for hit in raw_hits]
            return formatted_hits

        # Run search_single_query on all sub-questions at the exact same time
        async_tasks = [search_single_query(query) for query in sub_questions_list]
        batches_of_results = await asyncio.gather(*async_tasks)

        # Flatten list of lists and filter out corrupted results using simple loops
        valid_results = []
        for batch in batches_of_results:
            for result in batch:
                has_url = bool(result.get("url"))
                has_content = bool(result.get("content"))
                
                if has_url and has_content:
                    valid_results.append(result)

    except Exception as e:
        print(f"[tavily]     ERROR {type(e).__name__}: {e} -> 0 results")
        return {"raw_results": []}

    print(f"[tavily]     DONE   -> {len(valid_results)} result(s)")
    return {"raw_results": valid_results}

@node
async def rag_retrieval(state: ResearchState) -> dict:
    """Document retrieval from the hybrid RAG service (Simplified Version)."""
    sub_questions_list = state["sub_questions"]
    print(f"[rag]        START  ({len(sub_questions_list)} sub-question(s))")

    try:
        async with httpx.AsyncClient(timeout=RAG_TIMEOUT) as client:

            # Helper function: queries RAG service for a SINGLE query string
            async def fetch_docs_for_single_query(query_text: str) -> list[dict]:
                response = await client.post(
                    f"{RAG_BASE_URL}/query",
                    json={"question": query_text},
                )
                response.raise_for_status()
                
                sources = response.json().get("sources", [])
                formatted_docs = [_as_doc_result(chunk) for chunk in sources]
                return formatted_docs

            # Execute RAG requests concurrently across all sub-questions
            async_tasks = [
                fetch_docs_for_single_query(query) 
                for query in sub_questions_list
            ]
            batches_of_chunks = await asyncio.gather(*async_tasks)

        # Flatten list of lists
        all_retrieved_chunks = []
        for batch in batches_of_chunks:
            for chunk in batch:
                all_retrieved_chunks.append(chunk)

        # Filter out low-relevance chunks using a standard loop
        valid_results = []
        for chunk in all_retrieved_chunks:
            meets_score_threshold = chunk.get("score", 0) >= RAG_MIN_SCORE
            has_valid_url = bool(chunk.get("url"))
            has_valid_content = bool(chunk.get("content"))

            if meets_score_threshold and has_valid_url and has_valid_content:
                valid_results.append(chunk)

        dropped_count = len(all_retrieved_chunks) - len(valid_results)

    except Exception as e:
        print(f"[rag]        ERROR {type(e).__name__}: {e} -> degrading to 0 results")
        return {"raw_results": []}

    print(f"[rag]        DONE   -> {len(valid_results)} result(s) ({dropped_count} below threshold)")
    return {"raw_results": valid_results}


@node
def merge_and_rank(state: ResearchState) -> dict:
    """Dedupe and rank. STUB — passthrough, but reports what it received."""
    raw = state["raw_results"]
    by_type = {}
    for r in raw:
        by_type[r["source_type"]] = by_type.get(r["source_type"], 0) + 1
    print(f"[merge]      received {len(raw)} result(s): {by_type}")
    return {"ranked_results": raw[:5]}