"""Execution boundaries for endpoints backed entirely by synchronous services."""

from functools import wraps
from typing import Awaitable, Callable, ParamSpec, TypeVar

from starlette.concurrency import run_in_threadpool


P = ParamSpec("P")
T = TypeVar("T")


def threadpool_endpoint(operation: Callable[P, T]) -> Callable[P, Awaitable[T]]:
    """Keep the async route contract while moving the whole read off the API loop.

    Apply below the router decorator to synchronous handlers only. Keeping the
    complete operation together also keeps DB sessions and file handles on the
    same worker, including response enrichment and ownership checks.
    """
    @wraps(operation)
    async def endpoint(*args: P.args, **kwargs: P.kwargs) -> T:
        return await run_in_threadpool(operation, *args, **kwargs)

    return endpoint
