import functools
import inspect


def _check(result, fn):
    if not isinstance(result, dict):
        raise TypeError(
            f"node {fn.__name__!r} returned {type(result).__name__}, expected dict"
        )


def node(fn):
    """Enforce the node contract: a node must return a dict. Sync or async."""
    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def async_wrapper(state):
            result = await fn(state)
            _check(result, fn)
            return result
        return async_wrapper

    @functools.wraps(fn)
    def sync_wrapper(state):
        result = fn(state)
        _check(result, fn)
        return result
    return sync_wrapper