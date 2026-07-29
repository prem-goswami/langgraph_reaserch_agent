from app.graph import route_decision


def test_search_route():
    assert route_decision({"route": "search"}) == "search"


def test_generate_route():
    assert route_decision({"route": "generate"}) == "generate"