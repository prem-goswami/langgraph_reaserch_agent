import pytest

from warmup.nodes import _normalise_route


@pytest.mark.parametrize("raw,expected", [
    ("search", "search"),
    ("generate", "generate"),
    ("Search", "search"),
    ("  search\n", "search"),
    ("search.", "search"),
    ("`search`", "search"),
    ('"generate"', "generate"),
    ("I would choose search.", "search"),
    ("Based on the question, generate", "generate"),
    ("", "search"),              # falls back to DEFAULT_ROUTE
    ("banana", "search"),        # falls back to DEFAULT_ROUTE
])
def test_normalisation(raw, expected):
    assert _normalise_route(raw) == expected