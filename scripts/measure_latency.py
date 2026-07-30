import os

# Must be set BEFORE app.config is imported — config reads env at module level.
# load_dotenv() does not override already-set variables, so this wins over .env.
os.environ["MAX_CORRECTIONS"] = "0"

import asyncio
import statistics
import time

from app.research_graph import build_research_graph
from app.research_state import initial_state



# One distinct question per trial, so the search provider cannot serve a
# cached response for a repeat of the same query. Within a trial, both
# configurations use the SAME question — that is the controlled comparison.
QUESTIONS = [
    "How does Walmart's omnichannel strategy compare to "
    "Amazon's current market position?",

    "How does Walmart's supply chain automation compare to "
    "Amazon's fulfilment network?",

    "How does Walmart's digital advertising business compare to "
    "Amazon's advertising revenue?",
]

COOLDOWN = 6.0     # seconds between runs, to stay under the RAG rate limit

async def timed_run(app, question: str):
    """Run once. Returns (total_seconds, {node_name: seconds_since_previous}).

    NOTE: astream yields once per NODE UPDATE, not once per superstep. Two
    nodes in one superstep emit two chunks. Timings are therefore intervals
    between emissions, which is still the right basis for comparing the
    retrieval stage across topologies.
    """
    marks = {}
    degraded = False

    t_start = time.perf_counter()
    t_prev = t_start

    async for chunk in app.astream(initial_state(question)):
        t_now = time.perf_counter()
        for name, update in chunk.items():
            marks[name] = t_now - t_prev
            if name in ("tavily_search", "rag_retrieval") and not update.get("raw_results"):
                degraded = True
        t_prev = t_now

    return time.perf_counter() - t_start, marks, degraded


async def main():
    par = build_research_graph(parallel=True)
    seq = build_research_graph(parallel=False)

    print("=" * 72)
    print(f"TRIALS: {len(QUESTIONS)} · distinct question per trial · "
          f"MAX_CORRECTIONS=0 · {COOLDOWN:.0f}s cooldown")
    print("=" * 72)

    print("\nwarm-up (discarded — absorbs DNS, TLS, connection pool setup)")
    await timed_run(par, "What is Walmart's 2026 revenue?")
    await asyncio.sleep(COOLDOWN)

    rows = []

    for i, question in enumerate(QUESTIONS, start=1):
        print(f"\ntrial {i}: {question[:60]}...")

        s_total, s_marks, s_deg = await timed_run(seq, question)
        print(f"  sequential  {s_total:6.2f}s" + ("   [DEGRADED]" if s_deg else ""))
        await asyncio.sleep(COOLDOWN)

        p_total, p_marks, p_deg = await timed_run(par, question)
        print(f"  parallel    {p_total:6.2f}s" + ("   [DEGRADED]" if p_deg else ""))
        await asyncio.sleep(COOLDOWN)

        s_retr = s_marks.get("tavily_search", 0) + s_marks.get("rag_retrieval", 0)
        p_retr = p_marks.get("tavily_search", 0) + p_marks.get("rag_retrieval", 0)

        rows.append({
            "n": i,
            "seq": s_total, "par": p_total,
            "seq_retr": s_retr, "par_retr": p_retr,
            "critic": p_marks.get("critic", 0),
            "gen": p_marks.get("generation", 0),
            "degraded": s_deg or p_deg,
        })

    clean = [r for r in rows if not r["degraded"]]
    used = clean or rows

    if not clean:
        print("\n  WARNING: every trial degraded — results are not comparable.")
    elif len(clean) < len(rows):
        print(f"\n  NOTE: {len(rows) - len(clean)} trial(s) excluded "
              f"(a retrieval branch returned nothing).")

    seq_med = statistics.median(r["seq"] for r in used)
    par_med = statistics.median(r["par"] for r in used)
    seq_retr_med = statistics.median(r["seq_retr"] for r in used)
    par_retr_med = statistics.median(r["par_retr"] for r in used)
    critic_med = statistics.median(r["critic"] for r in used)
    gen_med = statistics.median(r["gen"] for r in used)

    print("\n" + "=" * 72)
    print(f"RESULTS  (median of {len(used)} clean trial(s))")
    print("=" * 72)
    print(f"  retrieval stage    sequential {seq_retr_med:6.2f}s   "
          f"parallel {par_retr_med:6.2f}s   "
          f"saved {seq_retr_med - par_retr_med:5.2f}s "
          f"({(1 - par_retr_med / seq_retr_med) * 100:4.0f}%)")
    print(f"  end to end         sequential {seq_med:6.2f}s   "
          f"parallel {par_med:6.2f}s   "
          f"saved {seq_med - par_med:5.2f}s "
          f"({(1 - par_med / seq_med) * 100:4.0f}%)")
    print()
    print(f"  not parallelisable: critic {critic_med:.2f}s + "
          f"generation {gen_med:.2f}s = {critic_med + gen_med:.2f}s "
          f"({(critic_med + gen_med) / par_med * 100:.0f}% of the parallel run)")

    print("\n  per trial")
    for r in rows:
        flag = "  [excluded: degraded]" if r["degraded"] else ""
        print(f"    {r['n']}.  seq {r['seq']:6.2f}s   par {r['par']:6.2f}s"
              f"   retrieval {r['seq_retr']:5.2f} -> {r['par_retr']:5.2f}{flag}")


if __name__ == "__main__":
    asyncio.run(main())