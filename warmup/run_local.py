import sys

from warmup.graph import build_graph

app = build_graph()

question = " ".join(sys.argv[1:]) or "What were the major AI model releases in 2025?"

initial_state = {
    "question": question,
    "route": "",
    "search_results": [],
    "answer": "",
}

print("=" * 70)
print(f"QUESTION: {question}")
print("=" * 70)

final = None
for step in app.stream(initial_state):
    for node_name, update in step.items():
        if update is None:
            print(f"  [{node_name}] -> RETURNED NONE (expected a dict)")
        elif not isinstance(update, dict):
            print(f"  [{node_name}] -> RETURNED {type(update).__name__}: {update!r}")
        else:
            print(f"  [{node_name}] -> {list(update.keys())}")
    final = step
answer = (final or {}).get("generation", {}).get("answer")
print(answer or "(no answer produced)")

print()
print("=" * 70)
print("ANSWER")
print("=" * 70)
print(final["generation"]["answer"])