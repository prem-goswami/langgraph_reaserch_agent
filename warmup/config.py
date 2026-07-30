import os
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gpt-4o-mini")
VALID_ROUTES = {"search", "generate"}
DEFAULT_ROUTE = "search"
