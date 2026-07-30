from app.research_nodes import merge_and_rank


def _state(results):
    return {"raw_results": results}


def test_dedupe_keeps_highest_score():
    out = merge_and_rank(_state([
        {"source_type": "doc", "title": "d", "url": "chunk-1",
         "content": "x", "score": 0.6},
        {"source_type": "doc", "title": "d", "url": "chunk-1",
         "content": "x", "score": 0.9},
    ]))
    assert len(out["ranked_results"]) == 1
    assert out["ranked_results"][0]["score"] == 0.9


def test_trailing_slash_is_same_url():
    out = merge_and_rank(_state([
        {"source_type": "web", "title": "a", "url": "https://x.com/p",
         "content": "x", "score": 0.5},
        {"source_type": "web", "title": "a", "url": "https://x.com/p/",
         "content": "x", "score": 0.7},
    ]))
    assert len(out["ranked_results"]) == 1


def test_ranks_across_sources_and_truncates():
    results = [
        {"source_type": "web", "title": f"w{i}", "url": f"https://x.com/{i}",
         "content": "x", "score": 0.1 * i}
        for i in range(1, 8)
    ]
    out = merge_and_rank(_state(results))
    assert len(out["ranked_results"]) == 5
    scores = [r["score"] for r in out["ranked_results"]]
    assert scores == sorted(scores, reverse=True)