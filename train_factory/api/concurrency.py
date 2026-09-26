"""Execution boundaries for synchronous services and protected handoffs."""

import asyncio
import logging
from functools import wraps
from typing import Any, Awaitable, Callable, ParamSpec, TypeVar

from starlette.concurrency import run_in_threadpool

P = ParamSpec("P")
T = TypeVar("T")
logger = logging.getLogger(__name__)


async def await_cancellation_safe(
    operation: Awaitable[T],
    *,
    cancelled_result_cleanup: Callable[[T], None] | None = None,
) -> T:
    """Finish one protected handoff before propagating caller cancellation.

    The inner task is never cancelled by its caller. If cancellation arrives,
    repeated cancellation requests are absorbed until the inner task finishes.
    A result the caller can no longer receive is synchronously cleaned up before
    the original cancellation is re-raised.
    """
    task = asyncio.ensure_future(operation)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError as cancelled:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break

        try:
            result = task.result()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(
                "Protected handoff failed after caller cancellation (%s)",
                type(exc).__name__,
            )
        else:
            if cancelled_result_cleanup is not None:
                try:
                    cancelled_result_cleanup(result)
                except Exception as exc:
                    logger.error(
                        "Protected handoff result cleanup failed after cancellation (%s)",
                        type(exc).__name__,
                    )
        raise cancelled


async def run_in_threadpool_cancellation_safe(
    operation: Callable[..., T],
    *args: Any,
    cancelled_result_cleanup: Callable[[T], None] | None = None,
    **kwargs: Any,
) -> T:
    """Run a protected synchronous handoff without abandoning its worker."""
    return await await_cancellation_safe(
        run_in_threadpool(operation, *args, **kwargs),
        cancelled_result_cleanup=cancelled_result_cleanup,
    )


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
