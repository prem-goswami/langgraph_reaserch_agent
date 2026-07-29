from app.graph import build_graph

app = build_graph()

initial_state = {
    "question": "What were the major AI model releases in 2025?",
    "route": "",
    "search_results": [],
    "answer": "",
}

print("=" * 60)
print("STREAMING (one dict per superstep)")
print("=" * 60)
for step in app.stream(initial_state):
    print("  superstep output:", step)

print()
print("=" * 60)
print("FINAL STATE")
print("=" * 60)
final = app.invoke(initial_state)
for k, v in final.items():
    print(f"  {k}: {v}")