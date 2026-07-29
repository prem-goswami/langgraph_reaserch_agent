import sys

from app.graph import build_graph

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
        print(f"  [{node_name}] -> {list(update.keys())}")
    final = step

print()
print("=" * 70)
print("ANSWER")
print("=" * 70)
print(final["generation"]["answer"])