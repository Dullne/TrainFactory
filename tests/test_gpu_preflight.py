import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).parents[1]


def _module():
    from train_factory.runtime import gpu_preflight

    return gpu_preflight


@pytest.mark.parametrize(
    ("compiled_cuda", "driver", "platform_kind", "expected_reason"),
    (
        ("12.4", "525.60.12", "linux", "driver_incompatible"),
        ("12.4", "525.60.13", "linux", None),
        ("12.4", "528.32", "windows", "driver_incompatible"),
        ("12.4", "528.33", "windows", None),
        ("12.4", "528.32", "wsl", "driver_incompatible"),
        ("12.4", "528.33", "wsl", None),
        ("12.4", "546.92", "wsl", None),
        ("12.4", "580.65.06", "linux", None),
        ("13.0", "580.65.05", "linux", "driver_incompatible"),
        ("13.0", "580.65.06", "linux", None),
        ("13.0", "600.1", "windows", "platform_unsupported"),
        ("13.0", "600.1", "wsl", "platform_unsupported"),
        ("11.8", "999.999.999", "linux", "cuda_unsupported"),
    ),
)
def test_compatibility_thresholds_are_semantic_and_fail_closed(
    compiled_cuda,
    driver,
    platform_kind,
    expected_reason,
):
    module = _module()

    assert module.compatibility_reason(compiled_cuda, driver, platform_kind) == expected_reason


@pytest.mark.parametrize(
    ("compiled_cuda", "driver", "platform_kind"),
    (
        ("12", "525.60.13", "linux"),
        ("12.4rc1", "525.60.13", "linux"),
        ("12.a", "525.60.13", "linux"),
        ("12.4", "525.60", "linux"),
        ("12.4", "525.60.13.1", "linux"),
        ("12.4", "528", "windows"),
        ("12.4", "528.33.1", "windows"),
        ("12.4", "private-canary", "wsl"),
    ),
)
def test_compatibility_rejects_truncated_or_non_numeric_versions(
    compiled_cuda,
    driver,
    platform_kind,
):
    module = _module()

    assert (
        module.compatibility_reason(
            compiled_cuda,
            driver,
            platform_kind,
        )
        == "version_invalid"
    )


@pytest.mark.parametrize(
    ("compiled_cuda", "driver", "platform_kind"),
    (
        ("9" * 5000 + ".4", "550.54.14", "linux"),
        ("12.4", "9" * 5000 + ".54.14", "linux"),
        ("12.4", "9" * 5000 + ".92", "wsl"),
    ),
)
def test_compatibility_rejects_unbounded_numeric_segments_without_exception(
    compiled_cuda,
    driver,
    platform_kind,
):
    module = _module()

    assert (
        module.compatibility_reason(
            compiled_cuda,
            driver,
            platform_kind,
        )
        == "version_invalid"
    )


def test_platform_detection_prefers_wsl_and_fails_closed_on_read_error():
    module = _module()

    assert module.detect_platform_kind("Linux", lambda: "5.15.0-MICROSOFT-standard") == "wsl"
    assert module.detect_platform_kind("Linux", lambda: "6.8.0-generic") == "linux"
    assert module.detect_platform_kind("Windows", lambda: "private-canary") == "windows"

    def fail_reader():
        raise OSError("private-path-canary")

    assert module.detect_platform_kind("Linux", fail_reader) is None
    assert module.detect_platform_kind("Darwin", lambda: "") is None


def test_off_mode_is_immutable_and_never_loads_gpu_dependencies():
    module = _module()

    def forbidden_loader():
        raise AssertionError("off must not load GPU dependencies")

    result = module.run_preflight(
        "off",
        torch_loader=forbidden_loader,
        nvml_loader=forbidden_loader,
        platform_detector=forbidden_loader,
    )

    assert result == module.GPUPreflightResult(
        mode="off",
        ok=True,
        compiled_cuda=None,
        driver_version=None,
        device_count=0,
        probe_executed=False,
        reason_code=None,
    )
    with pytest.raises(FrozenInstanceError):
        result.ok = False


class _Tensor:
    def __init__(self, events, fail_at=None):
        self.events = events
        self.fail_at = fail_at

    def add(self, value):
        self.events.append(("op", value))
        if self.fail_at == "operation":
            raise RuntimeError("private-operation-canary")
        return self


class _Cuda:
    def __init__(self, events, *, available=True, count=1, fail_at=None):
        self.events = events
        self.available = available
        self.count = count
        self.fail_at = fail_at

    def is_available(self):
        self.events.append("is_available")
        if self.fail_at == "is_available":
            raise RuntimeError("private-availability-canary")
        return self.available

    def device_count(self):
        self.events.append("device_count")
        if self.fail_at == "device_count":
            raise RuntimeError("private-count-canary")
        return self.count

    def synchronize(self, device):
        self.events.append(("synchronize", device))
        if self.fail_at == "synchronize":
            raise RuntimeError("private-sync-canary")


class _Torch:
    def __init__(self, events, *, compiled="12.4", available=True, count=1, fail_at=None):
        self.events = events
        self.version = type("Version", (), {"cuda": compiled})()
        self.cuda = _Cuda(
            events,
            available=available,
            count=count,
            fail_at=fail_at,
        )
        self.fail_at = fail_at

    def ones(self, size, *, device):
        self.events.append(("tensor", size, device))
        if self.fail_at == "tensor":
            raise RuntimeError("private-tensor-canary")
        return _Tensor(self.events, self.fail_at)


class _Nvml:
    def __init__(self, events, *, driver=b"546.92", fail_at=None):
        self.events = events
        self.driver = driver
        self.fail_at = fail_at

    def nvmlInit(self):
        self.events.append("nvml_init")
        if self.fail_at == "nvml_init":
            raise RuntimeError("private-init-canary")

    def nvmlSystemGetDriverVersion(self):
        self.events.append("driver")
        if self.fail_at == "driver":
            raise RuntimeError("private-driver-canary")
        return self.driver

    def nvmlShutdown(self):
        self.events.append("nvml_shutdown")
        if self.fail_at == "shutdown":
            raise RuntimeError("private-shutdown-canary")


def test_required_probe_order_and_public_result_are_exact():
    module = _module()
    events = []
    torch = _Torch(events)
    nvml = _Nvml(events)

    result = module.run_preflight(
        "required",
        torch_loader=lambda: events.append("torch") or torch,
        nvml_loader=lambda: events.append("nvml") or nvml,
        platform_detector=lambda: events.append("platform") or "wsl",
    )

    assert result == module.GPUPreflightResult(
        mode="required",
        ok=True,
        compiled_cuda="12.4",
        driver_version="546.92",
        device_count=1,
        probe_executed=True,
        reason_code=None,
    )
    assert events == [
        "torch",
        "nvml",
        "nvml_init",
        "driver",
        "platform",
        "is_available",
        "device_count",
        ("tensor", 1, "cuda:0"),
        ("op", 1),
        ("synchronize", 0),
        "nvml_shutdown",
    ]


@pytest.mark.parametrize(
    ("failure", "expected_reason", "expected_absent"),
    (
        ("torch_load", "torch_import_failed", "nvml"),
        ("compiled", "compiled_cuda_invalid", "nvml"),
        ("nvml_load", "nvml_import_failed", "nvml_init"),
        ("nvml_init", "nvml_init_failed", "driver"),
        ("driver", "driver_query_failed", "platform"),
        ("platform", "platform_unsupported", "is_available"),
        ("incompatible", "driver_incompatible", "is_available"),
        ("is_available", "cuda_availability_failed", "device_count"),
        ("unavailable", "cuda_unavailable", "device_count"),
        ("device_count", "device_count_failed", ("tensor", 1, "cuda:0")),
        ("no_devices", "no_devices", ("tensor", 1, "cuda:0")),
        ("tensor", "tensor_allocation_failed", ("op", 1)),
        ("operation", "tensor_operation_failed", ("synchronize", 0)),
        ("synchronize", "synchronize_failed", None),
    ),
)
def test_required_failures_stop_at_fixed_stage_without_private_details(
    failure,
    expected_reason,
    expected_absent,
):
    module = _module()
    events = []
    torch = _Torch(
        events,
        compiled=None if failure == "compiled" else "12.4",
        available=failure != "unavailable",
        count=0 if failure == "no_devices" else 1,
        fail_at=failure,
    )
    nvml = _Nvml(
        events,
        driver=(b"525.60.12" if failure == "incompatible" else b"550.54.14"),
        fail_at=failure,
    )

    def torch_loader():
        events.append("torch")
        if failure == "torch_load":
            raise RuntimeError("private-torch-canary")
        return torch

    def nvml_loader():
        events.append("nvml")
        if failure == "nvml_load":
            raise RuntimeError("private-nvml-canary")
        return nvml

    result = module.run_preflight(
        "required",
        torch_loader=torch_loader,
        nvml_loader=nvml_loader,
        platform_detector=lambda: events.append("platform") or (None if failure == "platform" else "linux"),
    )

    assert result.ok is False
    assert result.reason_code == expected_reason
    assert "private" not in repr(result)
    if expected_absent is not None:
        assert expected_absent not in events


def test_nvml_shutdown_runs_only_after_init_and_never_masks_primary_failure():
    module = _module()
    init_events = []
    init_failure = module.run_preflight(
        "required",
        torch_loader=lambda: _Torch(init_events),
        nvml_loader=lambda: _Nvml(init_events, fail_at="nvml_init"),
        platform_detector=lambda: "linux",
    )
    assert init_failure.reason_code == "nvml_init_failed"
    assert "nvml_shutdown" not in init_events

    shutdown_events = []
    primary_failure = module.run_preflight(
        "required",
        torch_loader=lambda: _Torch(shutdown_events, available=False),
        nvml_loader=lambda: _Nvml(
            shutdown_events,
            driver=b"550.54.14",
            fail_at="shutdown",
        ),
        platform_detector=lambda: "linux",
    )
    assert primary_failure.reason_code == "cuda_unavailable"
    assert shutdown_events[-1] == "nvml_shutdown"


def test_shutdown_failure_is_reported_only_when_there_is_no_primary_failure():
    module = _module()
    events = []

    result = module.run_preflight(
        "required",
        torch_loader=lambda: _Torch(events),
        nvml_loader=lambda: _Nvml(
            events,
            driver=b"550.54.14",
            fail_at="shutdown",
        ),
        platform_detector=lambda: "linux",
    )

    assert result.ok is False
    assert result.reason_code == "nvml_shutdown_failed"
    assert result.probe_executed is True


def test_compiled_cuda_property_and_driver_decode_failures_are_fixed_and_private():
    module = _module()

    class BadVersion:
        @property
        def cuda(self):
            raise RuntimeError("private-version-canary")

    torch = _Torch([])
    torch.version = BadVersion()
    version_result = module.run_preflight(
        "required",
        torch_loader=lambda: torch,
        nvml_loader=lambda: (_ for _ in ()).throw(AssertionError("NVML must not load")),
        platform_detector=lambda: "linux",
    )
    assert version_result.reason_code == "compiled_cuda_invalid"
    assert "private" not in repr(version_result)

    events = []
    driver_result = module.run_preflight(
        "required",
        torch_loader=lambda: _Torch(events),
        nvml_loader=lambda: _Nvml(events, driver=b"\xffprivate-canary"),
        platform_detector=lambda: "linux",
    )
    assert driver_result.reason_code == "driver_query_failed"
    assert events[-1] == "nvml_shutdown"
    assert "private" not in repr(driver_result)


@pytest.mark.parametrize("device_count", (-1, "1", True, None))
def test_invalid_device_counts_fail_closed_before_tensor(device_count):
    module = _module()
    events = []
    result = module.run_preflight(
        "required",
        torch_loader=lambda: _Torch(events, count=device_count),
        nvml_loader=lambda: _Nvml(events, driver=b"550.54.14"),
        platform_detector=lambda: "linux",
    )

    assert result.reason_code == "device_count_invalid"
    assert result.probe_executed is False
    assert not any(isinstance(event, tuple) and event[0] == "tensor" for event in events)


@pytest.mark.parametrize(
    ("compiled", "driver", "expected_reason"),
    (
        ("11.8", b"550.54.14", "cuda_unsupported"),
        ("12.4", b"private-canary", "driver_version_invalid"),
    ),
)
def test_required_compatibility_failures_stop_before_cuda_calls(
    compiled,
    driver,
    expected_reason,
):
    module = _module()
    events = []
    result = module.run_preflight(
        "required",
        torch_loader=lambda: _Torch(events, compiled=compiled),
        nvml_loader=lambda: _Nvml(events, driver=driver),
        platform_detector=lambda: "linux",
    )

    assert result.reason_code == expected_reason
    assert "is_available" not in events
    assert "private" not in repr(result)


def test_cli_off_output_is_strict_json_and_invalid_mode_is_private(tmp_path):
    marker = tmp_path / "gpu-imported"
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    for name in ("torch", "pynvml"):
        (shadow / f"{name}.py").write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n",
            encoding="utf-8",
        )
    environment = {"PYTHONPATH": str(shadow)}
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "train_factory.runtime.gpu_preflight",
            "--mode",
            "off",
        ],
        cwd=ROOT_DIR,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stderr == ""
    assert set(json.loads(completed.stdout)) == {
        "mode",
        "ok",
        "compiled_cuda",
        "driver_version",
        "device_count",
        "probe_executed",
        "reason_code",
    }
    assert not marker.exists()

    canary = "private-mode-canary"
    invalid = subprocess.run(
        [
            sys.executable,
            "-m",
            "train_factory.runtime.gpu_preflight",
            "--mode",
            canary,
        ],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert invalid.returncode == 2
    assert invalid.stdout == ""
    assert invalid.stderr == "gpu preflight arguments are invalid\n"
    assert canary not in invalid.stderr


def test_cli_failure_does_not_emit_dependency_warning_or_exception(
    monkeypatch,
    capsys,
):
    module = _module()

    def fail_loader():
        import warnings

        print("private-stdout-canary")
        print("private-stderr-canary", file=sys.stderr)
        warnings.warn("private-warning-canary", FutureWarning)
        raise RuntimeError("private-exception-canary")

    monkeypatch.setattr(module, "_load_torch", fail_loader)

    assert module.main(["--mode", "required"]) == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["reason_code"] == "torch_import_failed"
    assert "private" not in captured.out
