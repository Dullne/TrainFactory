"""
资源监控和清理工具
监控系统资源使用情况并进行自动清理
"""
import os

import psutil
import threading
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from ..config.settings import get_settings

logger = logging.getLogger(__name__)

@dataclass
class ResourceUsage:
    """资源使用情况"""
    cpu_percent: Optional[float]
    memory_percent: Optional[float]
    memory_used_mb: Optional[float]
    memory_total_mb: Optional[float]
    disk_usage_percent: Optional[float]
    disk_used_gb: Optional[float]
    disk_total_gb: Optional[float]
    open_files: Optional[int]
    thread_count: Optional[int]
    memory_limit_set: bool = False
    disk_mountpoint: Optional[str] = None
    disk_volumes: List[Dict[str, Any]] = field(default_factory=list)
    available: bool = True
    partial: bool = False
    error_code: Optional[str] = None
    errors: List[Dict[str, str]] = field(default_factory=list)


def _cgroup_memory_snapshot() -> tuple[Optional[int], Optional[int]]:
    """Read limit and usage once from the same effective cgroup hierarchy."""
    for limit_path, usage_path in (
        (
            Path("/sys/fs/cgroup/memory.max"),
            Path("/sys/fs/cgroup/memory.current"),
        ),
        (
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ),
    ):
        try:
            raw_limit = limit_path.read_text().strip()
            if raw_limit == "max":
                continue
            limit = int(raw_limit)
        except (OSError, ValueError):
            continue
        if not 0 < limit < (1 << 60):
            continue
        try:
            usage = int(usage_path.read_text().strip())
            return limit, usage if usage >= 0 else None
        except (OSError, ValueError):
            # An effective limit must never be paired with usage from another
            # cgroup hierarchy (or with host memory usage).
            return limit, None
    return None, None


def _cgroup_memory_limit() -> Optional[int]:
    """Return the effective cgroup memory limit for compatibility callers."""
    return _cgroup_memory_snapshot()[0]


def _cgroup_memory_usage() -> Optional[int]:
    """Return usage paired with the effective cgroup hierarchy."""
    return _cgroup_memory_snapshot()[1]


def _find_mountpoint(path: Path) -> str:
    """Return the longest mounted filesystem containing ``path``."""
    resolved = os.path.normcase(os.path.abspath(str(path)))
    matches: List[str] = []
    try:
        partitions = psutil.disk_partitions(all=True)
    except (OSError, RuntimeError):
        partitions = []

    for partition in partitions:
        mountpoint = os.path.normcase(os.path.abspath(partition.mountpoint))
        try:
            if os.path.commonpath((resolved, mountpoint)) == mountpoint:
                matches.append(partition.mountpoint)
        except ValueError:
            continue
    if matches:
        return max(matches, key=lambda value: len(os.path.abspath(value)))
    return Path(resolved).anchor or str(path)


def _configured_disk_paths() -> List[tuple[str, Path]]:
    settings = get_settings()
    return [
        ("datasets", Path(settings.datasets_dir)),
        ("models", Path(settings.models_dir)),
        ("output", Path(settings.output_dir)),
    ]


def _data_disk_usage() -> tuple[
    Optional[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, str]],
]:
    """Collect configured storage volumes and choose the highest-usage summary."""
    volumes_by_mount: Dict[str, Dict[str, Any]] = {}
    errors: List[Dict[str, str]] = []
    try:
        configured_paths = _configured_disk_paths()
    except Exception:
        logger.exception("Failed to resolve configured storage directories")
        return None, [], [{"scope": "disk", "error_code": "storage_config_unavailable"}]

    for scope, path in configured_paths:
        try:
            usage = psutil.disk_usage(str(path))
            mountpoint = _find_mountpoint(path)
            key = os.path.normcase(os.path.abspath(mountpoint))
            existing = volumes_by_mount.get(key)
            if existing is not None:
                existing["scopes"].append(scope)
                existing["paths"].append(str(path))
                continue
            total_gb = usage.total / (1024 ** 3)
            used_gb = usage.used / (1024 ** 3)
            free_gb = usage.free / (1024 ** 3)
            percent = float(
                getattr(
                    usage,
                    "percent",
                    (usage.used / usage.total) * 100 if usage.total else 0.0,
                )
            )
            volumes_by_mount[key] = {
                "scope": scope,
                "scopes": [scope],
                "paths": [str(path)],
                "mountpoint": mountpoint,
                "usage_percent": percent,
                "used_gb": used_gb,
                "total_gb": total_gb,
                "free_gb": free_gb,
            }
        except Exception:
            logger.exception("Failed to collect disk usage for %s", scope)
            errors.append(
                {"scope": f"disk.{scope}", "error_code": "disk_usage_unavailable"}
            )

    volumes = list(volumes_by_mount.values())
    if not volumes:
        return None, [], errors or [
            {"scope": "disk", "error_code": "disk_usage_unavailable"}
        ]
    worst = max(
        volumes,
        key=lambda volume: (volume["usage_percent"], -volume["free_gb"]),
    )
    return worst, volumes, errors


class ResourceMonitor:
    """资源监控器"""

    def __init__(self,
                 check_interval: int = 60,  # 检查间隔秒数
                 cpu_threshold: float = 90.0,  # CPU使用率阈值
                 memory_threshold: float = 85.0,  # 内存使用率阈值
                 monitor_threads: bool = False):  # 是否监控线程数，默认禁用
        self.check_interval = check_interval
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.monitor_threads = monitor_threads
        self.is_running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        # 添加警告去重机制
        self._last_warnings = {
            'cpu': 0,
            'memory': 0,
            'threads': 0,
            'files': 0
        }
        self._warning_cooldown = 300  # 5分钟内不重复相同警告

    def start_monitoring(self):
        """开始监控"""
        with self._lock:
            if self.is_running:
                logger.warning("资源监控已在运行")
                return

            self.is_running = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                name="resource-monitor",
                daemon=True
            )
            self.monitor_thread.start()
            logger.info("资源监控已启动")

    def stop_monitoring(self):
        """停止监控"""
        with self._lock:
            if not self.is_running:
                return

            self.is_running = False
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=10)
            logger.info("资源监控已停止")

    def enable_thread_monitoring(self, enable: bool = True):
        """启用或禁用线程监控"""
        with self._lock:
            self.monitor_threads = enable
            status = "启用" if enable else "禁用"
            logger.info(f"线程监控已{status}")

    def disable_thread_monitoring(self):
        """禁用线程监控"""
        self.enable_thread_monitoring(False)

    def get_current_usage(self) -> ResourceUsage:
        """Collect independent resource scopes without fabricating zero values."""
        errors: List[Dict[str, str]] = []

        try:
            process = psutil.Process()
        except Exception:
            logger.exception("Failed to access the current process for monitoring")
            process = None
            errors.append(
                {"scope": "process", "error_code": "process_monitor_unavailable"}
            )

        cpu_percent: Optional[float] = None
        open_files: Optional[int] = None
        thread_count: Optional[int] = None
        if process is not None:
            try:
                cpu_percent = float(process.cpu_percent())
            except Exception:
                logger.exception("Failed to collect CPU usage")
                errors.append(
                    {"scope": "cpu", "error_code": "cpu_usage_unavailable"}
                )
            try:
                open_files = len(process.open_files())
            except Exception:
                logger.exception("Failed to collect open file count")
                errors.append(
                    {"scope": "open_files", "error_code": "process_stats_unavailable"}
                )
            try:
                thread_count = process.num_threads()
            except Exception:
                logger.exception("Failed to collect thread count")
                errors.append(
                    {"scope": "threads", "error_code": "process_stats_unavailable"}
                )

        memory_percent: Optional[float] = None
        memory_used_mb: Optional[float] = None
        memory_total_mb: Optional[float] = None
        memory_limit, cgroup_used = _cgroup_memory_snapshot()
        memory_limit_set = memory_limit is not None
        if memory_limit is not None:
            memory_total_mb = memory_limit / (1024 * 1024)
            if cgroup_used is None:
                errors.append(
                    {"scope": "memory", "error_code": "cgroup_usage_unavailable"}
                )
            else:
                memory_used_mb = cgroup_used / (1024 * 1024)
                memory_percent = (
                    (memory_used_mb / memory_total_mb) * 100
                    if memory_total_mb > 0
                    else None
                )
        else:
            try:
                system_memory = psutil.virtual_memory()
                memory_percent = float(system_memory.percent)
                memory_used_mb = system_memory.used / (1024 * 1024)
                memory_total_mb = system_memory.total / (1024 * 1024)
            except Exception:
                logger.exception("Failed to collect system memory usage")
                errors.append(
                    {"scope": "memory", "error_code": "memory_usage_unavailable"}
                )

        disk_summary, disk_volumes, disk_errors = _data_disk_usage()
        errors.extend(disk_errors)
        if disk_summary is None:
            disk_usage_percent = None
            disk_used_gb = None
            disk_total_gb = None
            disk_mountpoint = None
        else:
            disk_usage_percent = disk_summary["usage_percent"]
            disk_used_gb = disk_summary["used_gb"]
            disk_total_gb = disk_summary["total_gb"]
            disk_mountpoint = disk_summary["mountpoint"]

        available = any(
            value is not None
            for value in (
                cpu_percent,
                memory_percent,
                disk_usage_percent,
                open_files,
                thread_count,
            )
        )
        partial = available and bool(errors)
        error_code = (
            "partial_resource_data"
            if partial
            else "resource_monitor_unavailable"
            if not available
            else None
        )
        return ResourceUsage(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            memory_total_mb=memory_total_mb,
            disk_usage_percent=disk_usage_percent,
            disk_used_gb=disk_used_gb,
            disk_total_gb=disk_total_gb,
            open_files=open_files,
            thread_count=thread_count,
            memory_limit_set=memory_limit_set,
            disk_mountpoint=disk_mountpoint,
            disk_volumes=disk_volumes,
            available=available,
            partial=partial,
            error_code=error_code,
            errors=errors,
        )

    def _monitor_loop(self):
        """监控循环"""
        logger.info("资源监控循环已启动")

        while self.is_running:
            try:
                usage = self.get_current_usage()
                self._log_usage(usage)

                # 检查是否需要警告（带去重机制）
                current_time = time.time()

                if (
                    usage.cpu_percent is not None
                    and usage.cpu_percent > self.cpu_threshold
                ):
                    if current_time - self._last_warnings['cpu'] > self._warning_cooldown:
                        logger.warning(f"CPU使用率过高: {usage.cpu_percent:.1f}%")
                        self._last_warnings['cpu'] = current_time

                if (
                    usage.memory_percent is not None
                    and usage.memory_percent > self.memory_threshold
                ):
                    if current_time - self._last_warnings['memory'] > self._warning_cooldown:
                        logger.warning(f"内存使用率过高: {usage.memory_percent:.1f}% ({usage.memory_used_mb:.1f}MB)")
                        self._last_warnings['memory'] = current_time

                if (
                    self.monitor_threads
                    and usage.thread_count is not None
                    and usage.thread_count > 50
                ):  # 线程数过多（仅在启用时检查）
                    if current_time - self._last_warnings['threads'] > self._warning_cooldown:
                        logger.warning(f"线程数过多: {usage.thread_count} (将在5分钟后再次提醒)")
                        self._last_warnings['threads'] = current_time

                if usage.open_files is not None and usage.open_files > 1000:
                    if current_time - self._last_warnings['files'] > self._warning_cooldown:
                        logger.warning(f"打开文件数过多: {usage.open_files} (将在5分钟后再次提醒)")
                        self._last_warnings['files'] = current_time

                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"资源监控循环出错: {str(e)}")
                time.sleep(self.check_interval)

        logger.info("资源监控循环已结束")

    def _log_usage(self, usage: ResourceUsage):
        """记录资源使用情况"""
        def display(value: Optional[float]) -> str:
            return f"{value:.1f}" if value is not None else "unavailable"

        logger.debug(
            f"资源使用情况 - CPU: {display(usage.cpu_percent)}%, "
            f"内存: {display(usage.memory_percent)}% "
            f"({display(usage.memory_used_mb)}MB), "
            f"磁盘: {display(usage.disk_usage_percent)}%, "
            f"文件: {usage.open_files}, "
            f"线程: {usage.thread_count}"
        )

    def cleanup_resources(self) -> Dict[str, Any]:
        """清理资源"""
        logger.info("开始资源清理...")
        cleanup_results = {
            "success": True,
            "actions": [],
            "errors": []
        }

        try:
            # 强制垃圾回收
            import gc
            collected = gc.collect()
            cleanup_results["actions"].append(f"垃圾回收: 回收了 {collected} 个对象")

            # 获取清理后的资源使用情况
            usage = self.get_current_usage()
            cleanup_results["final_usage"] = {
                "cpu_percent": usage.cpu_percent,
                "memory_percent": usage.memory_percent,
                "memory_used_mb": usage.memory_used_mb,
                "thread_count": usage.thread_count,
                "open_files": usage.open_files
            }

            logger.info("资源清理完成")

        except Exception as e:
            error_msg = f"资源清理失败: {str(e)}"
            logger.error(error_msg)
            cleanup_results["success"] = False
            cleanup_results["errors"].append(error_msg)

        return cleanup_results


# 全局资源监控器实例
resource_monitor = ResourceMonitor()


def get_resource_monitor() -> ResourceMonitor:
    """获取全局资源监控器"""
    return resource_monitor
