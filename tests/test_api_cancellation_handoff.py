"""Behavioral tests for cancellation-safe API handoffs."""

import asyncio
import logging
import threading

import pytest

from train_factory.api.concurrency import (
    await_cancellation_safe,
    run_in_threadpool_cancellation_safe,
)


def test_await_cancellation_safe_returns_inner_result():
    async def exercise():
        async def operation():
            return {"status": "ready"}

        return await await_cancellation_safe(operation())

    assert asyncio.run(exercise()) == {"status": "ready"}


def test_await_cancellation_safe_propagates_inner_exception():
    failure = RuntimeError("handoff failed")

    async def exercise():
        async def operation():
            raise failure

        with pytest.raises(RuntimeError) as raised:
            await await_cancellation_safe(operation())
        assert raised.value is failure

    asyncio.run(exercise())


def test_repeated_cancellation_waits_for_inner_completion_and_preserves_first_cancel():
    async def exercise():
        started = asyncio.Event()
        release = asyncio.Event()
        completed = asyncio.Event()

        async def operation():
            started.set()
            await release.wait()
            completed.set()
            return "unreachable result"

        caller = asyncio.create_task(await_cancellation_safe(operation()))
        await started.wait()

        caller.cancel("first cancellation")
        await asyncio.sleep(0)
        assert not caller.done()

        caller.cancel("repeated cancellation")
        await asyncio.sleep(0)
        assert not caller.done()

        release.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await caller

        assert raised.value.args == ("first cancellation",)
        assert completed.is_set()
        assert caller.cancelled()

    asyncio.run(exercise())


def test_inner_failure_after_cancellation_is_consumed_without_replacing_cancel(caplog):
    failure = RuntimeError("PRIVATE-CANARY https://user:secret@example.com/private")

    async def exercise():
        started = asyncio.Event()
        release = asyncio.Event()

        async def operation():
            started.set()
            await release.wait()
            raise failure

        caller = asyncio.create_task(await_cancellation_safe(operation()))
        await started.wait()
        caller.cancel("caller cancellation")
        await asyncio.sleep(0)
        assert not caller.done()

        release.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await caller

        assert raised.value.args == ("caller cancellation",)

    with caplog.at_level(logging.WARNING, logger="train_factory.api.concurrency"):
        asyncio.run(exercise())

    records = [
        record
        for record in caplog.records
        if record.name == "train_factory.api.concurrency"
    ]
    assert len(records) == 1
    assert "RuntimeError" in records[0].getMessage()
    assert "PRIVATE-CANARY" not in records[0].getMessage()
    assert records[0].exc_info is None
    assert "PRIVATE-CANARY" not in caplog.text


def test_cleanup_failure_after_cancellation_is_logged_without_private_details(caplog):
    async def exercise():
        started = asyncio.Event()
        release = asyncio.Event()

        async def operation():
            started.set()
            await release.wait()
            return object()

        def cleanup(_resource):
            raise RuntimeError(
                "PRIVATE-CANARY https://user:secret@example.com/private"
            )

        caller = asyncio.create_task(
            await_cancellation_safe(
                operation(),
                cancelled_result_cleanup=cleanup,
            )
        )
        await started.wait()
        caller.cancel("caller cancellation")
        await asyncio.sleep(0)
        assert not caller.done()

        release.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await caller

        assert raised.value.args == ("caller cancellation",)

    with caplog.at_level(logging.WARNING, logger="train_factory.api.concurrency"):
        asyncio.run(exercise())

    records = [
        record
        for record in caplog.records
        if record.name == "train_factory.api.concurrency"
    ]
    assert len(records) == 1
    assert "RuntimeError" in records[0].getMessage()
    assert "PRIVATE-CANARY" not in records[0].getMessage()
    assert records[0].exc_info is None
    assert "PRIVATE-CANARY" not in caplog.text


def test_cancelled_result_is_synchronously_cleaned_up_once():
    resource = object()
    cleaned = []

    async def exercise():
        started = asyncio.Event()
        release = asyncio.Event()

        async def operation():
            started.set()
            await release.wait()
            return resource

        caller = asyncio.create_task(
            await_cancellation_safe(
                operation(),
                cancelled_result_cleanup=cleaned.append,
            )
        )
        await started.wait()
        caller.cancel()
        await asyncio.sleep(0)
        assert not caller.done()

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await caller

        assert cleaned == [resource]

    asyncio.run(exercise())


def test_threadpool_handoff_finishes_worker_before_cancelled_await_exits():
    worker_started = threading.Event()
    release_worker = threading.Event()
    worker_finished = threading.Event()

    def operation(value, *, suffix):
        worker_started.set()
        release_worker.wait()
        worker_finished.set()
        return f"{value}{suffix}"

    async def exercise():
        caller = asyncio.create_task(
            run_in_threadpool_cancellation_safe(
                operation,
                "result",
                suffix="-from-worker",
            )
        )
        started = await asyncio.to_thread(worker_started.wait, 2)
        assert started, "worker did not start before the test deadline"

        caller.cancel()
        await asyncio.sleep(0)
        assert not caller.done()
        assert not worker_finished.is_set()

        release_worker.set()
        with pytest.raises(asyncio.CancelledError):
            await caller

        assert worker_finished.is_set()

    try:
        asyncio.run(exercise())
    finally:
        release_worker.set()
