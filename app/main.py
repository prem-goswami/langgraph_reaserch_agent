import asyncio
import json
import os
import platform
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from app.models import (
    ResearchRequest, ResearchResponse, SourceOut,
    HealthResponse, GraphResponse,
)
from app.config import RAG_BASE_URL, MAX_CORRECTIONS
from app.research_graph import build_research_graph
from app.research_state import initial_state

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


GRAPH = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Compile the graph once at startup, not per request."""
    GRAPH["app"] = build_research_graph(parallel=True)
    print("[startup]    graph compiled")
    yield
    GRAPH.clear()


api = FastAPI(
    title="Research Agent",
    description=(
        "Parallel multi-source research agent with self-correction. "
        "LangGraph: query decomposition -> parallel web + document retrieval "
        "-> rank fusion -> critic loop -> structured report."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

@api.get("/health", response_model=HealthResponse)
async def health():
    """Liveness plus dependency reachability."""
    rag_ok = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.post(f"{RAG_BASE_URL}/query", json={"question": "ping"})
            rag_ok = resp.status_code < 500
    except Exception:
        rag_ok = False

    return HealthResponse(
        status="ok",
        rag_reachable=rag_ok,
        rag_url=RAG_BASE_URL,
        max_corrections=MAX_CORRECTIONS,
    )

@api.get("/graph", response_model=GraphResponse)
async def graph():
    """The compiled graph as a Mermaid diagram."""
    g = GRAPH["app"].get_graph()
    return GraphResponse(
        mermaid=g.draw_mermaid(),
        nodes=[n for n in g.nodes if n not in ("__start__", "__end__")],
    )

@api.post("/research", response_model=ResearchResponse)
async def research(req: ResearchRequest):
    """Run the full research graph and return the report."""
    t0 = time.perf_counter()

    try:
        final = await GRAPH["app"].ainvoke(initial_state(req.question))
    except Exception as e:
        print(f"[api]        ERROR {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Research failed: {type(e).__name__}",
        )

    return ResearchResponse(
        question=final["question"],
        report=final["report"],
        sub_questions=final["sub_questions"],
        sources=[SourceOut(**r) for r in final["ranked_results"]],
        critic_verdict=final["critic_verdict"],
        critic_feedback=final["critic_feedback"],
        correction_count=final["correction_count"],
        elapsed_seconds=round(time.perf_counter() - t0, 2),
    )

@api.post("/research/stream")
async def research_stream(req: ResearchRequest):
    """Stream node updates as server-sent events."""

    async def events():
        t0 = time.perf_counter()
        try:
            async for chunk in GRAPH["app"].astream(initial_state(req.question)):
                for name, update in chunk.items():
                    payload = {
                        "node": name,
                        "keys": list(update.keys()),
                        "elapsed": round(time.perf_counter() - t0, 2),
                    }
                    if name == "merge_and_rank":
                        payload["n_ranked"] = len(update.get("ranked_results", []))
                    if name == "critic":
                        payload["verdict"] = update.get("critic_verdict")
                        payload["correction_count"] = update.get("correction_count")
                    if name == "generation":
                        payload["report"] = update.get("report")

                    yield f"data: {json.dumps(payload)}\n\n"

        except Exception as e:
            print(f"[api]        STREAM ERROR {type(e).__name__}: {e}")
            yield f"data: {json.dumps({'error': type(e).__name__})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:api",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=False,
    )