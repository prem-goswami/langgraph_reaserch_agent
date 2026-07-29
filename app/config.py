import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gpt-4o-mini")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o-mini")

VALID_ROUTES = {"search", "generate"}
DEFAULT_ROUTE = "search"

TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))
TAVILY_SEARCH_DEPTH = os.getenv("TAVILY_SEARCH_DEPTH", "basic")