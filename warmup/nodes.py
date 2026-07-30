from warmup.state import WarmupState
from langchain_core.messages import SystemMessage, HumanMessage
from app.config import TAVILY_MAX_RESULTS,TAVILY_SEARCH_DEPTH, GENERATION_MODEL
from warmup.config import ROUTER_MODEL, VALID_ROUTES, DEFAULT_ROUTE
from app.llm import get_llm
from warmup.search import get_tavily
from app.contract import node

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

GENERATION_SYSTEM = """You answer questions using the sources provided.

Rules:
- Ground every factual claim in the sources. Cite with bracketed numbers: [1], [2].
- If the sources do not contain the answer, say so plainly. Do not fill gaps \
from your own knowledge.
- If no sources are provided, answer from your own knowledge and open with:
  "No sources retrieved; answering from general knowledge."
- Match the answer's length to the question. A factual lookup gets one or two \
sentences. A comparison or a "how does X work" question gets a short structured \
answer with the relevant detail.
- Do not describe your process. No "based on the sources provided"."""

def _as_result(raw: dict) -> dict:
    """Translate one Tavily hit into the canonical result shape."""
    return {
        "source_type": "web",
        "title": raw.get("title") or "(untitled)",
        "url": raw.get("url") or "",
        "content": (raw.get("content") or "").strip(),
        "score": float(raw.get("score") or 0.0),
    }

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

def _format_sources(results: list[dict]) -> str:
    """Render results as a numbered block for the prompt."""
    if not results:
        return "(no sources retrieved)"

    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(f"[{i}] {r['title']}\n    {r['url']}\n    {r['content']}")
    return "\n\n".join(lines)

@node
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

@node
def search(state: WarmupState) -> dict:
    """Fetch web results. STUB — returns fixed data."""
    question = state['question']

    try:
        client = get_tavily()
        response = client.search(
            query=question,
            max_results=TAVILY_MAX_RESULTS,
            search_depth=TAVILY_SEARCH_DEPTH,
        )
        hits = response.get("results", [])
        results = [_as_result(h) for h in hits]
        results = [r for r in results if r["url"] and r["content"]]
    except Exception as e:

        print(f"[search]     ERROR {type(e).__name__}: {e} -> degrading to 0 results")
        return {"search_results": []}

    print(f"[search]     {len(results)} result(s) for {question[:40]!r}")
    return {"search_results": results}

@node
def generation(state: WarmupState) -> dict:
    """Write the final answer. STUB — echoes what it received."""
    question = state['question']
    results = state['search_results']

    llm = get_llm(GENERATION_MODEL, temperature=0.0)

    user_content = (
        f"Question: {question}\n\n"
        f"Sources:\n{_format_sources(results)}"
    )

    response = llm.invoke([
        SystemMessage(content=GENERATION_SYSTEM),
        HumanMessage(content=user_content),
    ])

    answer = response.content.strip()
    print(f"[generation] {len(results)} source(s) -> {len(answer)} chars")
    return {"answer": answer}
