import pytest
from langgraph.graph import END

from app.research_graph import route_after_critic, RETRY_TARGETS


def _state(verdict, count):
    return {"critic_verdict": verdict, "correction_count": count}


@pytest.mark.parametrize("verdict,count,expected", [
    ("PASS", 0, "generation"),
    ("PASS", 2, "generation"),
    ("FAIL", 1, RETRY_TARGETS),
    ("FAIL", 2, RETRY_TARGETS),
    ("FAIL", 3, "generation"),
])
def test_routing(verdict, count, expected):
    assert route_after_critic(_state(verdict, count)) == expected