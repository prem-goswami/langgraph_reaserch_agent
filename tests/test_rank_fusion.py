from app.research_nodes import _rrf_fuse


def _r(source_type, found_by, score, url):
    return {"source_type": source_type, "found_by": found_by,
            "score": score, "url": url, "title": "t", "content": "c"}


def test_no_source_dominates_when_groups_exceed_topk():
    """4 doc groups + 4 web groups = 8 groups > K=5.

    All eight rank-1 items tie on fused score. A raw-score tiebreak would
    give docs all four top slots, because saturated cross-encoder scores
    (0.92-0.99) all exceed web relevance scores (0.74-0.89).
    """
    results = []
    for i, s in enumerate([0.995, 0.973, 0.971, 0.919]):
        results.append(_r("doc", f"q{i}", s, f"chunk-{i}"))
    for i, s in enumerate([0.888, 0.790, 0.743, 0.700]):
        results.append(_r("web", f"q{i}", s, f"https://x.com/{i}"))

    top5 = [r for _, r in _rrf_fuse(results)[:5]]
    types = [r["source_type"] for r in top5]

    assert types.count("doc") <= 3, f"docs dominated: {types}"
    assert types.count("web") >= 2, f"web excluded: {types}"


def test_within_source_order_is_preserved():
    """Raw score ordering within one source is legitimate — same model, same scale."""
    results = [
        _r("doc", "q1", 0.90, "c1"),
        _r("doc", "q2", 0.95, "c2"),
        _r("web", "q1", 0.80, "u1"),
    ]
    ordered = [r for _, r in _rrf_fuse(results)]
    doc_scores = [r["score"] for r in ordered if r["source_type"] == "doc"]
    assert doc_scores == sorted(doc_scores, reverse=True)