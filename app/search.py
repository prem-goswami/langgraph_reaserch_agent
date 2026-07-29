from functools import lru_cache
from tavily import TavilyClient
from app.config import TAVILY_API_KEY


@lru_cache(maxsize=None)
def get_tavily() -> TavilyClient:
    """Cached Tavily client. One instance for the process."""
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is not set")
    return TavilyClient(api_key=TAVILY_API_KEY)