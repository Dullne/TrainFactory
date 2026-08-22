"""Safe identity fencing for operating-system processes."""

from __future__ import annotations

import logging
import math
from typing import Optional

import psutil


logger = logging.getLogger(__name__)

_CREATE_TIME_TOLERANCE_SECONDS = 0.01


def capture_process_create_time(pid: int) -> float:
    """Return the immutable process start time used together with a PID."""
    create_time = float(psutil.Process(int(pid)).create_time())
    if not math.isfinite(create_time) or create_time <= 0:
        raise ValueError("process create time must be a positive finite value")
    return create_time


def terminate_process_if_matches(
    pid: int,
    expected_create_time: Optional[float],
    *,
    wait_for_exit: bool = True,
    terminate_timeout: float = 5,
    kill_timeout: float = 5,
) -> bool:
    """Terminate only the process identified by both PID and start time.

    ``True`` means the process was absent, was signalled in non-blocking mode,
    or was confirmed exited. ``False`` preserves durable process/GPU evidence
    because identity or exit could not be confirmed.
    """
    try:
        process = psutil.Process(int(pid))
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return True
    except (psutil.AccessDenied, OSError, TypeError, ValueError):
        return False

    try:
        expected = float(expected_create_time)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(expected) or expected <= 0:
        return False

    try:
        actual = float(process.create_time())
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return True
    except (psutil.AccessDenied, OSError, TypeError, ValueError):
        return False

    if not math.isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=_CREATE_TIME_TOLERANCE_SECONDS,
    ):
        logger.warning(
            "Refusing to terminate reused PID %s: expected create_time=%s, actual=%s",
            pid,
            expected,
            actual,
        )
        return False

    try:
        process.terminate()
        if not wait_for_exit:
            return True
        process.wait(timeout=terminate_timeout)
        return True
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return True
    except (psutil.AccessDenied, OSError):
        return False
    except psutil.TimeoutExpired:
        pass

    try:
        process.kill()
        process.wait(timeout=kill_timeout)
        return True
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return True
    except (psutil.AccessDenied, psutil.TimeoutExpired, OSError):
        return False
