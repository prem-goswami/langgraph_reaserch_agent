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

RAG_BASE_URL = os.getenv("RAG_BASE_URL", "http://localhost:8000")
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.5"))
RAG_TIMEOUT = float(os.getenv("RAG_TIMEOUT", "15"))

TOP_K = int(os.getenv("TOP_K", "5"))

CRITIC_MODEL = os.getenv("CRITIC_MODEL", "gpt-4o-mini")
MAX_CORRECTIONS = int(os.getenv("MAX_CORRECTIONS", "2"))

ANALYZER_MODEL = os.getenv("ANALYZER_MODEL", "gpt-4o-mini")
MAX_SUB_QUESTIONS = int(os.getenv("MAX_SUB_QUESTIONS", "3"))

RRF_K = int(os.getenv("RRF_K", "60"))
