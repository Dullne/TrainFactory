import importlib
from types import SimpleNamespace

import pytest

resource_monitor_module = importlib.import_module(
    "train_factory.monitoring.resource_monitor"
)


MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024


class _Process:
    def cpu_percent(self):
        return 12.5

    def open_files(self):
        return [object(), object()]

    def num_threads(self):
        return 4


def _disk(total_gb: int, used_gb: int):
    return SimpleNamespace(
        total=total_gb * GIB,
        used=used_gb * GIB,
        free=(total_gb - used_gb) * GIB,
        percent=(used_gb / total_gb) * 100,
    )


def _patch_healthy_process(monkeypatch):
    monkeypatch.setattr(resource_monitor_module.psutil, "Process", _Process)
    monkeypatch.setattr(
        resource_monitor_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=25.0, used=4 * GIB, total=16 * GIB),
    )
    monkeypatch.setattr(
        resource_monitor_module,
        "get_settings",
        lambda: SimpleNamespace(
            datasets_dir="/configured/datasets",
            models_dir="/configured/models",
            output_dir="/configured/output",
        ),
        raising=False,
    )


def test_cgroup_v2_max_continues_to_probe_v1(monkeypatch):
    reads = []

    def read_text(path):
        value = str(path).replace("\\", "/")
        reads.append(value)
        if value.endswith("/memory.max"):
            return "max"
        if value.endswith("/memory.limit_in_bytes"):
            return str(8 * GIB)
        raise OSError(value)

    monkeypatch.setattr(resource_monitor_module.Path, "read_text", read_text)

    assert resource_monitor_module._cgroup_memory_limit() == 8 * GIB
    assert any(path.endswith("/memory.limit_in_bytes") for path in reads)


def test_cgroup_usage_matches_the_hierarchy_that_supplied_the_limit(monkeypatch):
    def read_text(path):
        value = str(path).replace("\\", "/")
        if value.endswith("/memory.max"):
            return "max"
        if value.endswith("/memory.current"):
            return str(7 * GIB)
        if value.endswith("/memory.limit_in_bytes"):
            return str(8 * GIB)
        if value.endswith("/memory.usage_in_bytes"):
            return str(3 * GIB)
        raise OSError(value)

    monkeypatch.setattr(resource_monitor_module.Path, "read_text", read_text)

    assert resource_monitor_module._cgroup_memory_limit() == 8 * GIB
    assert resource_monitor_module._cgroup_memory_usage() == 3 * GIB


def test_memory_sample_cannot_mix_hierarchies_when_cgroup_files_change(monkeypatch):
    _patch_healthy_process(monkeypatch)
    memory_max_reads = 0

    def read_text(path):
        nonlocal memory_max_reads
        value = str(path).replace("\\", "/")
        if value.endswith("/memory.max"):
            memory_max_reads += 1
            if memory_max_reads == 1:
                raise OSError("v2 temporarily unavailable")
            return str(2 * GIB)
        if value.endswith("/memory.current"):
            return str(1 * GIB)
        if value.endswith("/memory.limit_in_bytes"):
            return str(8 * GIB)
        if value.endswith("/memory.usage_in_bytes"):
            return str(3 * GIB)
        raise OSError(value)

    monkeypatch.setattr(resource_monitor_module.Path, "read_text", read_text)
    monkeypatch.setattr(
        resource_monitor_module.psutil,
        "disk_usage",
        lambda _path: _disk(10, 4),
    )

    usage = resource_monitor_module.ResourceMonitor().get_current_usage()

    assert usage.memory_limit_set is True
    assert usage.memory_total_mb == 8 * 1024
    assert usage.memory_used_mb == 3 * 1024
    assert usage.memory_percent == pytest.approx(37.5)
    assert memory_max_reads == 1


def test_limited_memory_without_matching_usage_never_mixes_host_usage(monkeypatch):
    _patch_healthy_process(monkeypatch)
    monkeypatch.setattr(
        resource_monitor_module,
        "_cgroup_memory_snapshot",
        lambda: (2 * GIB, None),
    )
    monkeypatch.setattr(
        resource_monitor_module.psutil, "disk_usage", lambda _path: _disk(10, 4)
    )

    usage = resource_monitor_module.ResourceMonitor().get_current_usage()

    assert usage.memory_limit_set is True
    assert usage.memory_total_mb == 2048
    assert usage.memory_used_mb is None
    assert usage.memory_percent is None
    assert usage.partial is True
    assert {error["scope"] for error in usage.errors} >= {"memory"}


def test_disk_usage_follows_configured_storage_directories_and_selects_worst_volume(
    monkeypatch,
):
    _patch_healthy_process(monkeypatch)
    monkeypatch.setattr(
        resource_monitor_module,
        "_cgroup_memory_snapshot",
        lambda: (None, None),
    )
    calls = []

    def disk_usage(path):
        normalized = str(path).replace("\\", "/")
        calls.append(normalized)
        if normalized == "/configured/datasets":
            return _disk(100, 90)
        if normalized == "/configured/models":
            return _disk(200, 40)
        if normalized == "/configured/output":
            return _disk(50, 20)
        raise OSError(normalized)

    monkeypatch.setattr(resource_monitor_module.psutil, "disk_usage", disk_usage)
    monkeypatch.setattr(
        resource_monitor_module,
        "_find_mountpoint",
        lambda path: str(path).replace("\\", "/"),
        raising=False,
    )

    usage = resource_monitor_module.ResourceMonitor().get_current_usage()

    assert calls == [
        "/configured/datasets",
        "/configured/models",
        "/configured/output",
    ]
    assert usage.disk_mountpoint == "/configured/datasets"
    assert usage.disk_usage_percent == pytest.approx(90.0)
    assert [volume["scope"] for volume in usage.disk_volumes] == [
        "datasets",
        "models",
        "output",
    ]


@pytest.mark.parametrize("cgroup_snapshot", [(None, None), (4 * GIB, GIB)])
def test_collection_failure_respects_any_remaining_cgroup_memory(monkeypatch, cgroup_snapshot):
    # Do not let the host/container's real memory limit change this unit test.
    monkeypatch.setattr(
        resource_monitor_module, "_cgroup_memory_snapshot", lambda: cgroup_snapshot
    )
    monkeypatch.setattr(
        resource_monitor_module.psutil,
        "Process",
        lambda: (_ for _ in ()).throw(OSError("process unavailable")),
    )
    monkeypatch.setattr(
        resource_monitor_module.psutil,
        "virtual_memory",
        lambda: (_ for _ in ()).throw(OSError("memory unavailable")),
    )
    monkeypatch.setattr(
        resource_monitor_module.psutil,
        "disk_usage",
        lambda _path: (_ for _ in ()).throw(OSError("disk unavailable")),
    )
    monkeypatch.setattr(
        resource_monitor_module,
        "get_settings",
        lambda: SimpleNamespace(
            datasets_dir="/configured/datasets",
            models_dir="/configured/models",
            output_dir="/configured/output",
        ),
        raising=False,
    )

    usage = resource_monitor_module.ResourceMonitor().get_current_usage()

    cgroup_available = cgroup_snapshot[0] is not None
    assert usage.available is cgroup_available
    assert usage.partial is cgroup_available
    assert usage.error_code == (
        "partial_resource_data" if cgroup_available else "resource_monitor_unavailable"
    )
    assert usage.cpu_percent is None
    assert usage.memory_percent == (25.0 if cgroup_available else None)
    assert usage.disk_usage_percent is None
    assert usage.open_files is None
    assert usage.thread_count is None


def test_unexpected_disk_collector_failure_is_explicitly_partial(monkeypatch):
    _patch_healthy_process(monkeypatch)
    monkeypatch.setattr(
        resource_monitor_module,
        "_cgroup_memory_snapshot",
        lambda: (None, None),
    )
    monkeypatch.setattr(
        resource_monitor_module.psutil,
        "disk_usage",
        lambda _path: (_ for _ in ()).throw(RuntimeError("platform plugin failed")),
    )

    usage = resource_monitor_module.ResourceMonitor().get_current_usage()

    assert usage.available is True
    assert usage.partial is True
    assert usage.disk_usage_percent is None
    assert {error["scope"] for error in usage.errors} >= {
        "disk.datasets",
        "disk.models",
        "disk.output",
    }


def test_process_submetric_runtime_failures_are_partial_not_fatal(monkeypatch):
    class PartiallyUnavailableProcess(_Process):
        def open_files(self):
            raise RuntimeError("platform does not expose open files")

        def num_threads(self):
            raise RuntimeError("platform does not expose threads")

    _patch_healthy_process(monkeypatch)
    monkeypatch.setattr(
        resource_monitor_module.psutil,
        "Process",
        PartiallyUnavailableProcess,
    )
    monkeypatch.setattr(
        resource_monitor_module,
        "_cgroup_memory_snapshot",
        lambda: (None, None),
    )
    monkeypatch.setattr(
        resource_monitor_module.psutil, "disk_usage", lambda _path: _disk(10, 4)
    )

    usage = resource_monitor_module.ResourceMonitor().get_current_usage()

    assert usage.available is True
    assert usage.partial is True
    assert usage.cpu_percent == 12.5
    assert usage.open_files is None
    assert usage.thread_count is None
    assert {error["scope"] for error in usage.errors} >= {"open_files", "threads"}
