"""Fail-closed GPU compatibility and tensor startup probe."""

from __future__ import annotations

import importlib
import io
import json
import platform
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol

Mode = Literal["required", "off"]
PlatformKind = Literal["linux", "windows", "wsl"]


class _Loader(Protocol):
    def __call__(self) -> object:
        ...


@dataclass(frozen=True)
class GPUPreflightResult:
    mode: Mode
    ok: bool
    compiled_cuda: str | None
    driver_version: str | None
    device_count: int
    probe_executed: bool
    reason_code: str | None


def _numeric_version(value: object, parts: int) -> tuple[int, ...] | None:
    if not isinstance(value, str):
        return None
    components = value.split(".")
    if len(components) != parts or any(
        not component or len(component) > 6 or not component.isascii() or not component.isdecimal()
        for component in components
    ):
        return None
    return tuple(int(component) for component in components)


def compatibility_reason(
    compiled_cuda: object,
    driver_version: object,
    platform_kind: object,
) -> str | None:
    """Return a fixed incompatibility code without reflecting either version."""
    cuda = _numeric_version(compiled_cuda, 2)
    if cuda is None:
        return "version_invalid"
    if platform_kind not in {"linux", "windows", "wsl"}:
        return "platform_unsupported"

    driver_parts = 3 if platform_kind == "linux" else 2
    driver = _numeric_version(driver_version, driver_parts)
    if driver is None:
        return "version_invalid"

    cuda_major = cuda[0]
    if cuda_major == 12:
        minimum = (525, 60, 13) if platform_kind == "linux" else (528, 33)
    elif cuda_major == 13:
        if platform_kind != "linux":
            return "platform_unsupported"
        minimum = (580, 65, 6)
    else:
        return "cuda_unsupported"

    # NVIDIA CUDA 12.4/13.0 release notes and the CUDA Compatibility guide
    # define these minor-version floors. Newer driver branches remain backward
    # compatible, so there is intentionally no upper bound.
    # https://docs.nvidia.com/cuda/archive/12.4.0/cuda-toolkit-release-notes/
    # https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/
    # https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html
    return None if driver >= minimum else "driver_incompatible"


def detect_platform_kind(
    system_name: str,
    osrelease_reader: Callable[[], str],
) -> PlatformKind | None:
    if system_name == "Windows":
        return "windows"
    if system_name != "Linux":
        return None
    try:
        osrelease = osrelease_reader()
    except (OSError, UnicodeError):
        return None
    return "wsl" if "microsoft" in osrelease.lower() else "linux"


def _detect_platform() -> PlatformKind | None:
    return detect_platform_kind(
        platform.system(),
        lambda: Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8"),
    )


def _load_torch() -> object:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return importlib.import_module("torch")


def _load_nvml() -> object:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return importlib.import_module("pynvml")


def _failure(
    reason_code: str,
    *,
    compiled_cuda: str | None = None,
    driver_version: str | None = None,
    device_count: int = 0,
    probe_executed: bool = False,
) -> GPUPreflightResult:
    return GPUPreflightResult(
        mode="required",
        ok=False,
        compiled_cuda=compiled_cuda,
        driver_version=driver_version,
        device_count=device_count,
        probe_executed=probe_executed,
        reason_code=reason_code,
    )


def run_preflight(
    mode: Mode,
    *,
    torch_loader: _Loader | None = None,
    nvml_loader: _Loader | None = None,
    platform_detector: Callable[[], PlatformKind | None] | None = None,
) -> GPUPreflightResult:
    """Run the ordered probe, returning only public fixed-shape evidence."""
    if mode == "off":
        return GPUPreflightResult(
            mode="off",
            ok=True,
            compiled_cuda=None,
            driver_version=None,
            device_count=0,
            probe_executed=False,
            reason_code=None,
        )
    if mode != "required":
        raise ValueError("gpu preflight mode is invalid")

    torch_loader = _load_torch if torch_loader is None else torch_loader
    nvml_loader = _load_nvml if nvml_loader is None else nvml_loader
    platform_detector = _detect_platform if platform_detector is None else platform_detector

    try:
        torch = torch_loader()
    except Exception:
        return _failure("torch_import_failed")
    try:
        compiled_value = getattr(getattr(torch, "version", None), "cuda", None)
    except Exception:
        return _failure("compiled_cuda_invalid")
    if _numeric_version(compiled_value, 2) is None:
        return _failure("compiled_cuda_invalid")
    compiled_cuda = compiled_value

    try:
        nvml = nvml_loader()
    except Exception:
        return _failure("nvml_import_failed", compiled_cuda=compiled_cuda)

    initialized = False
    driver_version: str | None = None
    try:
        try:
            nvml.nvmlInit()
            initialized = True
        except Exception:
            return _failure("nvml_init_failed", compiled_cuda=compiled_cuda)

        try:
            driver_value = nvml.nvmlSystemGetDriverVersion()
            if isinstance(driver_value, bytes):
                driver_value = driver_value.decode("ascii")
        except Exception:
            return _failure("driver_query_failed", compiled_cuda=compiled_cuda)

        try:
            platform_kind = platform_detector()
        except Exception:
            platform_kind = None
        if platform_kind is None:
            return _failure(
                "platform_unsupported",
                compiled_cuda=compiled_cuda,
            )
        driver_parts = 3 if platform_kind == "linux" else 2
        if _numeric_version(driver_value, driver_parts) is None:
            return _failure(
                "driver_version_invalid",
                compiled_cuda=compiled_cuda,
            )
        driver_version = driver_value
        reason = compatibility_reason(
            compiled_cuda,
            driver_version,
            platform_kind,
        )
        if reason is not None:
            return _failure(
                reason,
                compiled_cuda=compiled_cuda,
                driver_version=driver_version,
            )

        try:
            available = torch.cuda.is_available()
        except Exception:
            return _failure(
                "cuda_availability_failed",
                compiled_cuda=compiled_cuda,
                driver_version=driver_version,
            )
        if available is not True:
            return _failure(
                "cuda_unavailable",
                compiled_cuda=compiled_cuda,
                driver_version=driver_version,
            )

        try:
            device_count = torch.cuda.device_count()
        except Exception:
            return _failure(
                "device_count_failed",
                compiled_cuda=compiled_cuda,
                driver_version=driver_version,
            )
        if not isinstance(device_count, int) or isinstance(device_count, bool) or device_count < 0:
            return _failure(
                "device_count_invalid",
                compiled_cuda=compiled_cuda,
                driver_version=driver_version,
            )
        if device_count == 0:
            return _failure(
                "no_devices",
                compiled_cuda=compiled_cuda,
                driver_version=driver_version,
            )

        try:
            tensor = torch.ones(1, device="cuda:0")
        except Exception:
            return _failure(
                "tensor_allocation_failed",
                compiled_cuda=compiled_cuda,
                driver_version=driver_version,
                device_count=device_count,
                probe_executed=True,
            )
        try:
            tensor.add(1)
        except Exception:
            return _failure(
                "tensor_operation_failed",
                compiled_cuda=compiled_cuda,
                driver_version=driver_version,
                device_count=device_count,
                probe_executed=True,
            )
        try:
            torch.cuda.synchronize(0)
        except Exception:
            return _failure(
                "synchronize_failed",
                compiled_cuda=compiled_cuda,
                driver_version=driver_version,
                device_count=device_count,
                probe_executed=True,
            )
        result = GPUPreflightResult(
            mode="required",
            ok=True,
            compiled_cuda=compiled_cuda,
            driver_version=driver_version,
            device_count=device_count,
            probe_executed=True,
            reason_code=None,
        )
        initialized = False
        try:
            nvml.nvmlShutdown()
        except Exception:
            return _failure(
                "nvml_shutdown_failed",
                compiled_cuda=compiled_cuda,
                driver_version=driver_version,
                device_count=device_count,
                probe_executed=True,
            )
        return result
    finally:
        if initialized:
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if (
        len(arguments) != 2
        or arguments[0] != "--mode"
        or arguments[1]
        not in {
            "required",
            "off",
        }
    ):
        print("gpu preflight arguments are invalid", file=sys.stderr)
        return 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = run_preflight(arguments[1])
    print(json.dumps(asdict(result), sort_keys=True, separators=(",", ":")))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
