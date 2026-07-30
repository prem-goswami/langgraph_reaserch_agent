import asyncio
import math
import httpx
import re 
from tavily import AsyncTavilyClient
from app.contract import node
from app.config import (
    TAVILY_API_KEY, TAVILY_MAX_RESULTS, TAVILY_SEARCH_DEPTH,
    RAG_BASE_URL, RAG_MIN_SCORE, RAG_TIMEOUT, TOP_K, CRITIC_MODEL,GENERATION_MODEL, ANALYZER_MODEL, MAX_SUB_QUESTIONS,RRF_K 
)

from app.research_state import ResearchState
from langchain_core.messages import SystemMessage, HumanMessage   
from app.llm import get_llm



def _sigmoid(logit: float) -> float:
    """Cross-encoder logit -> calibrated relevance probability in [0, 1]."""
    return 1.0 / (1.0 + math.exp(-logit))


def _as_web_result(hit: dict, found_by: str = "") -> dict:
    return {
        "source_type": "web",
        "title": hit.get("title") or "(untitled)",
        "url": hit.get("url") or "",
        "content": (hit.get("content") or "").strip(),
        "score": float(hit.get("score") or 0.0),
        "found_by": found_by,                          
    }


def _as_doc_result(chunk: dict, found_by: str = "") -> dict:
    return {
        "source_type": "doc",
        "title": chunk.get("source") or "(document)",
        "url": chunk.get("chunk_id") or "",
        "content": (chunk.get("content_preview") or "").strip(),
        "score": _sigmoid(float(chunk.get("rerank_score") or -20.0)),
        "found_by": found_by,                          
    }

def _effective_queries(state: ResearchState) -> list[str]:
    """Retrieval queries, extended with critic feedback on a retry pass."""
    subs = state["sub_questions"]
    feedback = state.get("critic_feedback", "")
    if not feedback:
        return subs
    return subs + [feedback]

ANALYZER_SYSTEM = """You split a research question into the minimum set of \
search queries needed to answer it.

Output one query per line, numbered. Nothing else — no preamble, no explanation.

Split only when the question genuinely has separable parts:
- It compares two things -> one query per thing.
- It asks about distinct entities, time periods, or aspects -> one each.
- It asks "why" or "how" about a specific event -> one query for the event, \
one for the causes or mechanism.

Do NOT split a question that asks for one thing. Return it as a single query, \
lightly reworded into search terms. Over-splitting wastes retrieval and \
dilutes the results.

Write search queries, not questions. Drop question words and filler; keep \
entity names, qualifiers, and dates.

Maximum queries: {max_queries}

Examples:

Question: What is Walmart's 2026 e-commerce revenue?
1. Walmart 2026 e-commerce revenue

Question: How does Walmart's omnichannel strategy compare to Amazon's current \
market position?
1. Walmart omnichannel strategy
2. Amazon current e-commerce market position

Question: Why did Walmart's stock fall after the Q4 2026 earnings call?
1. Walmart stock decline Q4 2026 earnings
2. Walmart Q4 2026 earnings call analyst reaction"""


def _parse_sub_questions(raw: str, original: str, max_n: int) -> list:
    """Extract numbered queries from the analyzer's output."""
    queries = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # strip a leading "1." / "1)" / "-" / "*"
        cleaned = re.sub(r"^\s*(?:\d+[\.\)]|[-*])\s*", "", line).strip()
        cleaned = cleaned.strip('"\'')
        if cleaned and len(cleaned) > 3:
            queries.append(cleaned)

    if not queries:
        print(f"[analyzer]   WARN unparseable output -> falling back to original")
        return [original]

    return queries[:max_n]


@node
def query_analyzer(state: ResearchState) -> dict:
    """Decompose the question into the minimum set of search queries."""
    question = state["question"]

    llm = get_llm(ANALYZER_MODEL, temperature=0.0)

    try:
        response = llm.invoke([
            SystemMessage(
                content=ANALYZER_SYSTEM.format(max_queries=MAX_SUB_QUESTIONS)
            ),
            HumanMessage(content=f"Question: {question}"),
        ])
        subs = _parse_sub_questions(
            response.content, question, MAX_SUB_QUESTIONS
        )
    except Exception as e:
        print(f"[analyzer]   ERROR {type(e).__name__}: {e} -> using original question")
        subs = [question]

    print(f"[analyzer]   -> {len(subs)} query/queries")
    for i, s in enumerate(subs, start=1):
        print(f"[analyzer]     {i}. {s}")

    return {"sub_questions": subs}

@node
async def tavily_search(state: ResearchState) -> dict:
    """Web retrieval across all sub-questions concurrently (Simplified Version)."""
    sub_questions_list = _effective_queries(state)
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
            formatted_hits = [_as_web_result(hit, query_text) for hit in raw_hits]
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
    sub_questions_list = _effective_queries(state)
    print(f"[rag]        START  ({len(sub_questions_list)} sub-question(s))")

    try:
        async with httpx.AsyncClient(timeout=RAG_TIMEOUT) as client:

            # Helper function: queries RAG service for a SINGLE query string
            async def search_single_query(query_text: str) -> list[dict]:
                response = await client.post(
                    f"{RAG_BASE_URL}/query",
                    json={"question": query_text},
                )
                response.raise_for_status()
                
                sources = response.json().get("sources", [])
                formatted_docs = [_as_doc_result(chunk, query_text) for chunk in sources]
                return formatted_docs

            # Execute RAG requests concurrently across all sub-questions
            async_tasks = [
                search_single_query(query) 
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

def _rrf_fuse(deduped: list) -> list:
    """Reciprocal rank fusion over (source_type, sub_question) groups.

    Ranking within each group means every source AND every sub-question
    contributes its own best result. Prevents one sub-question's high-scoring
    results from occupying all slots for a source.
    """
    groups = {}
    for r in deduped:
        key = (r["source_type"], r.get("found_by", ""))
        groups.setdefault(key, []).append(r)

    scored = []
    for key, items in groups.items():
        items.sort(key=lambda r: r["score"], reverse=True)
        for rank, r in enumerate(items, start=1):
            rrf = 1.0 / (RRF_K + rank)
            scored.append((rrf, r["score"], rank, r))

    # primary: fused score (rank tier). secondary: raw score, ordering only
    # within a tier — it cannot promote a rank-3 item above a rank-1 item.
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)

    return [(rank, r) for _, _, rank, r in scored]


@node
def merge_and_rank(state: ResearchState) -> dict:
    """Dedupe across sources, fuse by reciprocal rank, keep the top K."""
    raw = state["raw_results"]

    # --- dedupe, keeping the highest-scored copy of each source ---
    best_by_key = {}
    for result in raw:
        key = (result.get("url") or "").rstrip("/")
        if not key:
            continue
        existing = best_by_key.get(key)
        if existing is None or result["score"] > existing["score"]:
            best_by_key[key] = result

    deduped = list(best_by_key.values())
    duplicates_removed = len(raw) - len(deduped)

    # --- fuse across sources by rank, not by score value ---
    fused = _rrf_fuse(deduped)
    top = [r for _, r in fused[:TOP_K]]

    # --- diagnostics ---
    print(f"[merge]      in={len(raw)} deduped={len(deduped)} "
          f"({duplicates_removed} duplicate(s) removed)")
    for source_type in ("web", "doc"):
        scores = [r["score"] for r in deduped if r["source_type"] == source_type]
        if scores:
            print(f"[merge]      {source_type}: n={len(scores)} "
                  f"min={min(scores):.3f} max={max(scores):.3f}")
    for position, (source_rank, r) in enumerate(fused[:TOP_K], start=1):
        print(f"[merge]      #{position} {r['source_type']}"
              f"(rank {source_rank}, score {r['score']:.3f}) "
              f"<- {r.get('found_by', '')[:40]}")
    composition = {}
    for r in top:
        composition[r["source_type"]] = composition.get(r["source_type"], 0) + 1
    print(f"[merge]      top-{len(top)} composition: {composition}")

    return {"ranked_results": top}


CRITIC_SYSTEM = """You judge whether a set of retrieved sources is sufficient \
to answer a research question. You do not answer the question yourself.

Reply in exactly this format, with no other text:

VERDICT: PASS
FEEDBACK: <one sentence>

or

VERDICT: FAIL
FEEDBACK: <what is missing, and what to search for instead>

Reply PASS when the sources collectively cover every part of the question. \
Partial coverage of every part is acceptable — you are judging sufficiency, \
not completeness.

Reply FAIL only for a specific, nameable gap. Examples of real gaps:
- The question compares two things and sources cover only one of them.
- The question asks about a time period no source addresses.
- The question asks "why" or "how" and sources give only "what".
- Every source restates the same single fact, so nothing corroborates it.

Do NOT reply FAIL because sources could hypothetically be more numerous, \
more recent, or more authoritative. That is always true and is not a gap.

When you reply FAIL, your feedback must name a concrete search that would \
close the gap. Feedback like "more comprehensive sources needed" is useless \
and counts as a PASS instead."""


def _parse_critic(raw: str) -> tuple[str, str]:
    """Extract (verdict, feedback) from the critic's response."""
    verdict = "PASS"          # safe default: don't loop on a parse failure
    feedback = ""

    for line in raw.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            value = line.split(":", 1)[1].strip().upper()
            verdict = "FAIL" if "FAIL" in value else "PASS"
        elif line.upper().startswith("FEEDBACK:"):
            feedback = line.split(":", 1)[1].strip()

    if verdict == "FAIL" and not feedback:
        print("[critic]     WARN FAIL with no feedback -> treating as PASS")
        return "PASS", ""

    return verdict, feedback


@node
def critic(state: ResearchState) -> dict:
    """Judge source sufficiency. Sole writer of correction_count."""
    question = state["question"]
    results = state["ranked_results"]
    count = state["correction_count"]

    if not results:
        print(f"[critic]     no sources -> PASS (nothing to retry with)")
        return {
            "critic_verdict": "PASS",
            "critic_feedback": "",
            "correction_count": count ,
        }

    llm = get_llm(CRITIC_MODEL, temperature=0.0)

    sources_block = "\n\n".join(
        f"[{i}] ({r['source_type']}, score {r['score']:.2f}) {r['title']}\n"
        f"    {r['content'][:500]}"
        for i, r in enumerate(results, start=1)
    )

    response = llm.invoke([
        SystemMessage(content=CRITIC_SYSTEM),
        HumanMessage(content=f"Question: {question}\n\nSources:\n{sources_block}"),
    ])

    verdict, feedback = _parse_critic(response.content)

    new_count = count + 1 if verdict == "FAIL" else count

    print(f"[critic]     {verdict}  (corrections requested: {new_count})")
    if verdict == "FAIL":
        print(f"[critic]     feedback: {feedback[:120]}")

    return {
        "critic_verdict": verdict,
        "critic_feedback": feedback,
        "correction_count": new_count,
    }


GENERATION_SYSTEM = """You write research answers grounded strictly in the \
sources provided.

Produce exactly two sections, in this order, with these exact headings:

## Summary
A direct answer to the question. Match length to the question: a factual \
lookup gets two or three sentences; a comparison or a "how does X work" \
question gets a fuller paragraph.

## Key Findings
Three to six bullet points, each a specific claim with a citation.

Rules:
- Cite every factual claim with bracketed numbers matching the numbered \
sources: [1], [3]. Cite multiple sources as [1][4], not [1, 4].
- Ground every claim in the sources. Do NOT fill gaps from your own \
knowledge. If the sources do not answer part of the question, say so \
explicitly in the Summary.
- Never cite a number that does not appear in the source list.
- Do not describe your process. No "based on the sources provided".
- Do not add sections beyond the two specified. Source listings and \
confidence are added separately."""


def _format_sources_for_prompt(results: list) -> str:
    """Numbered source block. Numbering is unified across both source types."""
    if not results:
        return "(no sources retrieved)"

    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(
            f"[{i}] ({r['source_type']}) {r['title']}\n"
            f"    {r['url']}\n"
            f"    {r['content'][:800]}"
        )
    return "\n\n".join(lines)


def _validate_citations(text: str, n_sources: int):
    """Strip citation markers pointing at sources that do not exist."""
    invalid = 0

    def replace(match) -> str:
        nonlocal invalid
        index = int(match.group(1))
        if 1 <= index <= n_sources:
            return match.group(0)
        invalid += 1
        return ""

    cleaned = re.sub(r"\[(\d+)\]", replace, text)
    return cleaned, invalid


def _render_source_list(results: list, source_type: str) -> str:
    """Render one source type as a numbered list, preserving citation numbers."""
    rows = [
        (i, r) for i, r in enumerate(results, start=1)
        if r["source_type"] == source_type
    ]
    if not rows:
        return "_None retrieved._"

    lines = []
    for number, r in rows:
        if source_type == "web":
            lines.append(f"{number}. [{r['title']}]({r['url']})")
        else:
            lines.append(f"{number}. {r['title']} — `{r['url'][:8]}…`")
    return "\n".join(lines)


def _build_confidence(state: ResearchState, results: list,
                      invalid_citations: int) -> str:
    """Confidence assessed from state, not generated by the model."""
    n_web = sum(1 for r in results if r["source_type"] == "web")
    n_doc = sum(1 for r in results if r["source_type"] == "doc")
    corrections = state["correction_count"]
    verdict = state["critic_verdict"]

    notes = []

    if not results:
        level = "None"
        notes.append("No sources were retrieved; the answer is ungrounded.")
    elif verdict == "FAIL":
        level = "Low"
        notes.append(
            f"The critic judged the sources insufficient after {corrections} "
            f"correction attempt(s); the answer is based on the best available "
            f"evidence and may be incomplete."
        )
    elif corrections > 0:
        level = "Moderate"
        notes.append(
            f"Sources were judged sufficient after {corrections} "
            f"correction attempt(s)."
        )
    else:
        level = "High"
        notes.append("Sources were judged sufficient on the first retrieval pass.")

    if n_doc == 0:
        notes.append("No document sources contributed — web-only.")
    if n_web == 0 and n_doc > 0:
        notes.append("No web sources contributed — documents only.")
    if invalid_citations:
        notes.append(f"{invalid_citations} invalid citation marker(s) were removed.")

    return (
        f"**{level}** — {n_web} web source(s), {n_doc} document source(s), "
        f"{corrections} correction(s).\n\n" + " ".join(notes)
    )


@node
def generation(state: ResearchState) -> dict:
    """Assemble the final report. The LLM writes prose; code asserts facts."""
    question = state["question"]
    results = state["ranked_results"]

    llm = get_llm(GENERATION_MODEL, temperature=0.0)
    user_content = (
        f"Question: {question}\n\n"
        f"Sources:\n{_format_sources_for_prompt(results)}"
    )

    response = llm.invoke([
        SystemMessage(content=GENERATION_SYSTEM),
        HumanMessage(content=user_content),
    ])

    prose, invalid = _validate_citations(response.content.strip(), len(results))
    if invalid:
        print(f"[generation] WARN removed {invalid} invalid citation marker(s)")

    report = "\n\n".join([
        f"# {question}",
        prose,
        "## Web Sources",
        _render_source_list(results, "web"),
        "## Doc Sources",
        _render_source_list(results, "doc"),
        "## Confidence",
        _build_confidence(state, results, invalid),
    ])

    print(f"[generation] {len(results)} source(s) -> {len(report)} chars")
    return {"report": report}