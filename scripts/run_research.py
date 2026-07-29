import asyncio
import sys
import time

from app.research_graph import build_research_graph


async def main():
    app = build_research_graph()

    question = " ".join(sys.argv[1:]) or (
        "How does Walmart's omnichannel strategy compare to "
        "Amazon's current market position?"
    )

    initial_state = {
        "question": question,
        "sub_questions": [],
        "raw_results": [],
        "ranked_results": [],
        "critic_verdict": "",
        "critic_feedback": "",
        "correction_count": 0,
        "report": "",
    }

    print("=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    t0 = time.perf_counter()
    final_state = None
    async for step in app.astream(initial_state):
        for name, update in step.items():
            print(f"  SUPERSTEP -> {{{name}: {list(update.keys())}}}")
            if name == "generation":
                final_state = update
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 70)
    print(final_state["report"] if final_state else "(no report produced)")
    print("=" * 70)
    print(f"wall clock: {elapsed:.2f}s")

if __name__ == "__main__":
    asyncio.run(asyncio.wait_for(main(), timeout=120))