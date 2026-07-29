from functools import lru_cache
from langchain_openai import ChatOpenAI
from app.config import OPENAI_API_KEY


@lru_cache(maxsize=None)
def get_llm(model: str, temperature: float = 0.0) -> ChatOpenAI:
    """Cached LLM client factory. One client per (model, temperature) pair."""
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=OPENAI_API_KEY,
        timeout=30,
        max_retries=2,
    )