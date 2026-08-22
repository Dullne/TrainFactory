"""
GPU资源动态分配管理器

基于pynvml实现GPU资源监控和动态分配，支持多任务GPU管理。
"""

import os
import threading
import logging
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timedelta
from uuid import UUID

logger = logging.getLogger(__name__)


def _task_log_label(task_id: Any) -> Any:
    """Keep attempt tokens in allocation keys but out of logs and status DTOs."""
    if not isinstance(task_id, str) or not task_id.startswith("training:"):
        return task_id
    parts = task_id.split(":")
    if len(parts) != 3:
        return "training:<invalid>"
    try:
        task_uuid = UUID(parts[1])
        run_uuid = UUID(parts[2])
    except (AttributeError, TypeError, ValueError):
        return "training:<invalid>"
    if str(task_uuid) != parts[1] or str(run_uuid) != parts[2]:
        return "training:<invalid>"
    return f"training:{parts[1]}"


class GPUResourceManager:
    """GPU资源动态分配管理器"""

    def __init__(self):
        # 核心分配数据
        self.gpu_allocations: Dict[int, str] = {}  # {gpu_id: task_id}
        self.task_gpus: Dict[str, Set[int]] = {}   # {task_id: {gpu_ids}}
        self.allocation_times: Dict[str, datetime] = {}  # {task_id: allocation_time}

        self._lock = threading.RLock()
        self.max_gpus = self._detect_max_gpus()

        logger.info(f"GPU资源管理器初始化完成，检测到 {self.max_gpus} 个GPU")

    def sync_from_database(self) -> int:
        """
        从数据库同步GPU分配状态。

        DEPRECATED：当前无调用者，且租约键格式与运行时不一致——运行时键为
        ``training:{task_id}:{run_token}``（attempt 级），此处以裸 task_id
        恢复，直接接入会让 release_gpus_for_task 永远 miss 造成永久泄漏。
        未来接入前必须统一键格式（并按 DB 状态过滤非 running 任务）。
        启动恢复目前由 cleanup_orphan_tasks 处理（标记失败 + 清扫孤儿进程）。

        Returns:
            恢复的任务数量
        """
        try:
            from sqlmodel import select
            from ..storage.database import get_session
            from ..storage.entities.training_task_entity import TrainingTaskDB

            with self._lock:
                with get_session() as session:
                    # 查找所有正在运行的训练任务
                    statement = select(TrainingTaskDB).where(
                        TrainingTaskDB.status == "running"
                    )
                    running_tasks = session.exec(statement).all()

                    restored_count = 0
                    for task in running_tasks:
                        task_id = task.task_id
                        # 尝试从训练参数中恢复GPU信息
                        training_params = task.training_params or {}
                        device = training_params.get("device", "")

                        if device and device.startswith("cuda:"):
                            # 解析设备字符串如 "cuda:0" 或 "cuda:0,cuda:1"
                            gpu_ids = []
                            for part in device.split(","):
                                part = part.strip()
                                if part.startswith("cuda:"):
                                    try:
                                        gpu_id = int(part.split(":")[1])
                                        gpu_ids.append(gpu_id)
                                    except (IndexError, ValueError):
                                        pass

                            if gpu_ids:
                                # 标记这些GPU为已分配
                                for gpu_id in gpu_ids:
                                    if gpu_id < self.max_gpus:
                                        self.gpu_allocations[gpu_id] = task_id

                                self.task_gpus[task_id] = set(gpu_ids)
                                self.allocation_times[task_id] = task.started_at or datetime.now()
                                restored_count += 1
                                logger.info(f"从数据库恢复任务 {task_id} 的GPU分配: {gpu_ids}")

                    if restored_count > 0:
                        logger.info(f"从数据库恢复了 {restored_count} 个任务的GPU分配")

                    return restored_count

        except Exception as e:
            logger.warning(f"从数据库同步GPU分配失败: {e}")
            return 0

    def _detect_max_gpus(self) -> int:
        """检测系统GPU数量"""
        try:
            # 方法1: 使用nvidia-ml-py
            import pynvml
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            return gpu_count
        except Exception:
            pass

        try:
            # 方法2: 使用torch
            import torch
            if torch.cuda.is_available():
                return torch.cuda.device_count()
        except Exception:
            pass

        # 方法3: 默认假设0个GPU（CPU-only模式）
        logger.warning("无法检测GPU数量，使用默认值0（CPU模式）")
        return int(os.environ.get('MAX_GPUS', '0'))

    def allocate_gpus_for_task(self, task_id: str, device_request: str = "auto") -> Optional[str]:
        """
        为任务分配GPU资源

        Args:
            task_id: 任务ID
            device_request: 设备请求，支持:
                - "auto": 自动分配1个GPU
                - "cpu": 使用CPU
                - "cuda:0": 指定GPU 0
                - "cuda:0,cuda:1": 指定多个GPU
                - "auto:2": 自动分配2个GPU

        Returns:
            设备字符串，如 "cuda:0,cuda:1" 或 "cpu" 或 None（分配失败）
        """
        with self._lock:
            try:
                # 处理None、空值、"None"字符串
                if device_request is None or device_request == "None" or (isinstance(device_request, str) and device_request.strip() == ""):
                    logger.warning(f"任务 {_task_log_label(task_id)} 设备请求为空/None，默认使用CPU模式")
                    device_request = "cpu"

                # CPU模式
                if device_request == "cpu":
                    logger.info(f"任务 {_task_log_label(task_id)} 使用CPU模式")
                    return "cpu"

                # 解析设备请求
                requested_gpus, num_gpus = self._parse_device_request(device_request)

                # Empty list means user specified GPUs but all were invalid
                if requested_gpus is not None and len(requested_gpus) == 0:
                    logger.error("指定的GPU全部无效，分配失败")
                    return None

                if requested_gpus is not None:
                    # 指定GPU模式
                    if self._can_allocate_specific_gpus(requested_gpus):
                        return self._do_allocate_gpus(task_id, requested_gpus)
                    else:
                        logger.error(f"指定的GPU {requested_gpus} 不可用")
                        return None
                else:
                    # 自动分配模式
                    available_gpus = self._get_available_gpus(num_gpus)
                    if available_gpus:
                        return self._do_allocate_gpus(task_id, available_gpus)
                    else:
                        logger.error(f"无法自动分配 {num_gpus} 个GPU，当前可用GPU不足")
                        return None

            except Exception as e:
                logger.error(f"GPU分配失败: {e}")
                return None

    def _parse_device_request(self, device_request: str) -> tuple:
        """
        解析设备请求

        Returns:
            (指定的GPU列表或None, 请求的GPU数量)
        """
        if device_request == "auto":
            return None, 1

        if device_request.startswith("auto:"):
            try:
                num_gpus = int(device_request.split(":")[1])
                return None, num_gpus
            except (IndexError, ValueError):
                logger.warning(f"无法解析auto请求: {device_request}，使用默认1个GPU")
                return None, 1

        if device_request.startswith("cuda:"):
            try:
                gpu_ids = []
                invalid_ids = []
                for part in device_request.split(","):
                    part = part.strip()
                    if part.startswith("cuda:"):
                        gpu_id = int(part.split(":")[1])
                        if 0 <= gpu_id < self.max_gpus:
                            gpu_ids.append(gpu_id)
                        else:
                            invalid_ids.append(gpu_id)
                            logger.warning(f"GPU ID {gpu_id} 超出范围 [0, {self.max_gpus-1}]")

                # 任一 GPU ID 无效即整体拒绝：混合有效/无效时静默降级会让
                # 用户期望的并行度/显存被悄然削减（与 _validate_training_resource_limits
                # 的 gpu_ids 校验语义一致）。
                if invalid_ids:
                    logger.error(
                        f"GPU ID {invalid_ids} 无效，系统只有 {self.max_gpus} 个 GPU "
                        f"(0-{self.max_gpus - 1})"
                    )
                    return [], 0

                return gpu_ids if gpu_ids else None, len(gpu_ids)

            except (IndexError, ValueError) as e:
                logger.warning(f"无法解析CUDA请求: {device_request}, 错误: {e}")
                return None, 1

        logger.warning(f"未识别的设备请求: {device_request}，使用auto模式")
        return None, 1

    def _can_allocate_specific_gpus(self, gpu_ids: List[int]) -> bool:
        """检查指定的GPU是否可以分配"""
        for gpu_id in gpu_ids:
            if gpu_id >= self.max_gpus:
                logger.warning(f"GPU {gpu_id} 超出系统GPU数量 ({self.max_gpus})")
                return False
            if gpu_id in self.gpu_allocations:
                allocated_task = self.gpu_allocations[gpu_id]
                logger.warning(f"GPU {gpu_id} 已被任务 {_task_log_label(allocated_task)} 占用")
                return False
        return True

    def _get_available_gpus(self, num_needed: int) -> Optional[List[int]]:
        """获取可用的GPU"""
        available = []
        for gpu_id in range(self.max_gpus):
            if gpu_id not in self.gpu_allocations:
                available.append(gpu_id)
                if len(available) >= num_needed:
                    break

        logger.info(f"请求 {num_needed} 个GPU，找到 {len(available)} 个可用GPU: {available}")
        return available[:num_needed] if len(available) >= num_needed else None

    def _do_allocate_gpus(self, task_id: str, gpu_ids: List[int]) -> str:
        """执行GPU分配"""
        # 分配GPU
        for gpu_id in gpu_ids:
            self.gpu_allocations[gpu_id] = task_id

        # 记录任务GPU映射
        self.task_gpus[task_id] = set(gpu_ids)
        self.allocation_times[task_id] = datetime.now()

        # 生成CUDA设备字符串
        if len(gpu_ids) == 1:
            device_str = f"cuda:{gpu_ids[0]}"
        else:
            device_str = ",".join([f"cuda:{gpu_id}" for gpu_id in gpu_ids])

        logger.info(f"为任务 {_task_log_label(task_id)} 分配GPU: {device_str} (物理GPU: {gpu_ids})")
        return device_str

    def release_gpus_for_task(self, task_id: str) -> bool:
        """释放任务的GPU资源"""
        with self._lock:
            try:
                if task_id not in self.task_gpus:
                    logger.debug(f"任务 {_task_log_label(task_id)} 没有分配GPU资源")
                    return True

                # 获取任务的GPU
                gpu_ids = self.task_gpus[task_id]

                # 释放GPU
                for gpu_id in gpu_ids:
                    if gpu_id in self.gpu_allocations:
                        del self.gpu_allocations[gpu_id]

                # 清理记录
                del self.task_gpus[task_id]
                if task_id in self.allocation_times:
                    del self.allocation_times[task_id]

                logger.info(f"释放任务 {_task_log_label(task_id)} 的GPU资源: {list(gpu_ids)}")
                return True

            except Exception as e:
                logger.error(f"释放GPU资源失败: {e}")
                self._force_cleanup_task_records(task_id)
                return False

    def _force_cleanup_task_records(self, task_id: str):
        """强制清理任务记录，避免永久资源泄漏"""
        try:
            logger.warning(f"强制清理任务 {_task_log_label(task_id)} 的GPU记录")

            if task_id in self.task_gpus:
                gpu_ids = self.task_gpus[task_id]
                for gpu_id in gpu_ids:
                    if gpu_id in self.gpu_allocations:
                        del self.gpu_allocations[gpu_id]
                del self.task_gpus[task_id]

            if task_id in self.allocation_times:
                del self.allocation_times[task_id]

            logger.info(f"强制清理任务 {_task_log_label(task_id)} GPU记录完成")

        except Exception as force_error:
            logger.critical(f"强制清理也失败！任务 {_task_log_label(task_id)} 的GPU资源可能永久泄漏: {force_error}")

    def force_release_gpu_for_task(self, task_id: str) -> bool:
        """强制释放指定任务的GPU资源"""
        with self._lock:
            logger.warning(f"强制释放任务 {_task_log_label(task_id)} 的GPU资源")

            try:
                gpu_ids = []
                if task_id in self.task_gpus:
                    gpu_ids = list(self.task_gpus[task_id])

                for gpu_id in list(self.gpu_allocations.keys()):
                    if self.gpu_allocations[gpu_id] == task_id:
                        del self.gpu_allocations[gpu_id]

                if task_id in self.task_gpus:
                    del self.task_gpus[task_id]
                if task_id in self.allocation_times:
                    del self.allocation_times[task_id]

                logger.info(f"强制释放任务 {_task_log_label(task_id)} GPU资源完成: {gpu_ids}")
                return True

            except Exception as e:
                logger.critical(f"强制释放也失败！: {e}")
                return False

    def _get_gpu_memory_info(self, gpu_id: int) -> Dict:
        """获取GPU显存信息（使用pynvml）"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

            # 获取显存信息
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory = mem_info.total
            used_memory = mem_info.used
            free_memory = mem_info.free

            # 获取GPU利用率
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu
                memory_util = utilization.memory
            except Exception:
                gpu_util = None
                memory_util = None

            # 获取GPU名称
            try:
                gpu_name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(gpu_name, bytes):
                    gpu_name = gpu_name.decode('utf-8')
            except Exception:
                gpu_name = f"GPU {gpu_id}"

            # 获取温度
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temperature = None

            # 获取功耗
            try:
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            except Exception:
                power_usage = None
                power_limit = None

            return {
                "gpu_name": gpu_name,
                "memory": {
                    "total": total_memory,
                    "used": used_memory,
                    "free": free_memory,
                    "total_gb": round(total_memory / 1024**3, 2),
                    "used_gb": round(used_memory / 1024**3, 2),
                    "free_gb": round(free_memory / 1024**3, 2),
                    "usage_percent": round((used_memory / total_memory) * 100, 1) if total_memory > 0 else 0
                },
                "utilization": {
                    "gpu_percent": gpu_util,
                    "memory_percent": memory_util
                },
                "temperature": temperature,
                "power": {
                    "usage_w": power_usage,
                    "limit_w": power_limit
                }
            }
        except Exception as e:
            logger.debug(f"获取GPU {gpu_id} 详细信息失败: {e}")
            return {
                "gpu_name": f"GPU {gpu_id}",
                "memory": {
                    "total": None, "used": None, "free": None,
                    "total_gb": None, "used_gb": None, "free_gb": None,
                    "usage_percent": None
                },
                "utilization": {"gpu_percent": None, "memory_percent": None},
                "temperature": None,
                "power": {"usage_w": None, "limit_w": None},
                "error": str(e)
            }

    def get_resource_status(self) -> Dict:
        """获取资源状态报告（包含显存信息）"""
        with self._lock:
            total_gpus = self.max_gpus

            # 构建详细状态
            gpu_details = {}
            total_memory_gb = 0
            used_memory_gb = 0

            for gpu_id in range(total_gpus):
                # 获取GPU硬件信息
                gpu_hw_info = self._get_gpu_memory_info(gpu_id)

                # 分配状态
                if gpu_id in self.gpu_allocations:
                    task_id = self.gpu_allocations[gpu_id]
                    alloc_time = self.allocation_times.get(task_id)

                    gpu_status = {
                        "status": "allocated",
                        "task_id": _task_log_label(task_id),
                        "allocated_at": alloc_time.isoformat() if alloc_time else None,
                        "duration": str(datetime.now() - alloc_time) if alloc_time else None
                    }
                else:
                    gpu_status = {
                        "status": "free",
                        "task_id": None,
                        "allocated_at": None,
                        "duration": None
                    }

                # 合并硬件信息
                gpu_details[gpu_id] = {**gpu_status, **gpu_hw_info}

                # 累计内存统计
                if gpu_hw_info["memory"]["total_gb"]:
                    total_memory_gb += gpu_hw_info["memory"]["total_gb"]
                if gpu_hw_info["memory"]["used_gb"]:
                    used_memory_gb += gpu_hw_info["memory"]["used_gb"]

            allocated_gpus = len(self.gpu_allocations)
            free_gpus = total_gpus - allocated_gpus

            gpu_mem_summary = {
                "total_gb": round(total_memory_gb, 2),
                "used_gb": round(used_memory_gb, 2),
                "free_gb": round(total_memory_gb - used_memory_gb, 2),
                "usage_percent": round((used_memory_gb / total_memory_gb) * 100, 1) if total_memory_gb > 0 else 0
            }

            result = {
                "total_gpus": total_gpus,
                "allocated_gpus": allocated_gpus,
                "free_gpus": free_gpus,
                "utilization_rate": allocated_gpus / total_gpus if total_gpus > 0 else 0,
                "gpu_memory_summary": gpu_mem_summary,
                "gpu_details": gpu_details,
                "active_tasks": len(self.task_gpus)
            }

            # 追加 CPU / 系统内存 概览
            try:
                cpu_info = self._get_cpu_info()
                result.update(cpu_info)
            except Exception as e:
                logger.debug(f"获取CPU状态失败: {e}")

            return result

    def _get_cpu_info(self) -> Dict:
        """获取CPU与系统内存信息"""
        try:
            import psutil
            logical = psutil.cpu_count(logical=True) or 0
            physical = psutil.cpu_count(logical=False) or 0
            cpu_percent = psutil.cpu_percent(interval=0.0)

            vm = psutil.virtual_memory()
            total_gb = round(vm.total / 1024**3, 2)
            used_gb = round(vm.used / 1024**3, 2)
            free_gb = round(vm.available / 1024**3, 2)
            usage_percent = round(vm.percent, 1)

            return {
                "cpu_summary": {
                    "logical_cores": logical,
                    "physical_cores": physical,
                    "cpu_usage_percent": cpu_percent,
                },
                "system_memory": {
                    "total_gb": total_gb,
                    "used_gb": used_gb,
                    "free_gb": free_gb,
                    "usage_percent": usage_percent,
                },
            }
        except Exception as e:
            logger.debug(f"获取CPU信息失败: {e}")
            return {
                "cpu_summary": {
                    "logical_cores": None,
                    "physical_cores": None,
                    "cpu_usage_percent": None,
                },
                "system_memory": {
                    "total_gb": None,
                    "used_gb": None,
                    "free_gb": None,
                    "usage_percent": None,
                },
            }

    def _run_nvidia_smi(self, args: List[str]) -> List[str]:
        """运行 nvidia-smi 查询命令并返回非空行"""
        try:
            import subprocess
            output = subprocess.check_output(
                ["nvidia-smi", *args],
                text=True,
                timeout=2
            )
        except FileNotFoundError:
            return []
        except Exception as e:
            logger.debug(f"nvidia-smi 查询失败: {e}")
            return []

        lines = []
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("no running processes"):
                continue
            lines.append(line)
        return lines

    def _get_container_id_for_pid(self, pid: int) -> Optional[str]:
        """从 /proc/<pid>/cgroup 解析容器 ID"""
        try:
            with open(f"/proc/{pid}/cgroup", "r", encoding="utf-8") as handle:
                data = handle.read()
        except Exception:
            return None

        import re
        match = re.search(r"/docker/([0-9a-f]{12,64})", data)
        if match:
            return match.group(1)
        match = re.search(r"/kubepods[^/]+/pod[^/]+/([0-9a-f]{12,64})", data)
        if match:
            return match.group(1)
        return None

    def _resolve_container_names(self, container_ids: List[str]) -> Dict[str, str]:
        """将容器 ID 映射为容器名称（可选，依赖 docker CLI）"""
        import shutil
        if not container_ids or not shutil.which("docker"):
            return {}

        resolved: Dict[str, str] = {}
        import subprocess
        for container_id in container_ids:
            if container_id in resolved:
                continue
            try:
                name = subprocess.check_output(
                    ["docker", "inspect", "-f", "{{.Name}}", container_id],
                    text=True,
                    timeout=2
                ).strip()
                resolved[container_id] = name.lstrip("/") if name else container_id[:12]
            except Exception:
                resolved[container_id] = container_id[:12]
        return resolved

    def get_gpu_processes(self) -> List[Dict[str, Any]]:
        """
        获取 GPU 进程占用信息（按 GPU 分组）

        返回列表，每项包含 GPU 信息与进程列表。
        """
        gpu_lines = self._run_nvidia_smi([
            "--query-gpu=index,uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        if not gpu_lines:
            return []

        gpus_by_uuid: Dict[str, Dict[str, Any]] = {}
        for line in gpu_lines:
            parts = [item.strip() for item in line.split(",", 3)]
            if len(parts) < 4:
                continue
            try:
                gpu_index = int(parts[0])
            except ValueError:
                continue
            uuid, name, total_mb_raw = parts[1], parts[2], parts[3]
            try:
                total_memory_mb = int(float(total_mb_raw))
            except ValueError:
                total_memory_mb = None

            gpus_by_uuid[uuid] = {
                "gpu_index": gpu_index,
                "name": name,
                "uuid": uuid,
                "total_memory_mb": total_memory_mb,
                "processes": []
            }

        process_lines = self._run_nvidia_smi([
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ])

        process_container_ids: List[str] = []
        process_details: Dict[int, Dict[str, Any]] = {}
        try:
            import psutil
        except Exception:
            psutil = None  # type: ignore

        for line in process_lines:
            parts = [item.strip() for item in line.split(",", 3)]
            if len(parts) < 4:
                continue
            gpu_uuid, pid_raw, process_name, used_mem_raw = parts
            try:
                pid = int(pid_raw)
            except ValueError:
                continue
            try:
                used_memory_mb = int(float(used_mem_raw))
            except ValueError:
                used_memory_mb = None

            detail = {
                "pid": pid,
                "name": process_name or None,
                "cmdline": None,
                "user": None,
                "gpu_memory_mb": used_memory_mb,
                "container_id": None,
                "container_name": None,
                "started_at": None,
            }

            if psutil is not None:
                try:
                    proc = psutil.Process(pid)
                    detail["name"] = proc.name() or detail["name"]
                    cmdline = " ".join(proc.cmdline()).strip()
                    detail["cmdline"] = cmdline if cmdline else None
                    detail["user"] = proc.username()
                    detail["started_at"] = datetime.fromtimestamp(proc.create_time()).isoformat()
                except Exception:
                    pass

            container_id = self._get_container_id_for_pid(pid)
            if container_id:
                detail["container_id"] = container_id
                process_container_ids.append(container_id)

            process_details[pid] = detail

            gpu_entry = gpus_by_uuid.get(gpu_uuid)
            if not gpu_entry:
                gpu_entry = {
                    "gpu_index": None,
                    "name": None,
                    "uuid": gpu_uuid,
                    "total_memory_mb": None,
                    "processes": []
                }
                gpus_by_uuid[gpu_uuid] = gpu_entry
            gpu_entry["processes"].append(detail)

        container_names = self._resolve_container_names(list(set(process_container_ids)))
        if container_names:
            for detail in process_details.values():
                container_id = detail.get("container_id")
                if container_id and container_id in container_names:
                    detail["container_name"] = container_names[container_id]

        # 排序输出
        result = sorted(
            gpus_by_uuid.values(),
            key=lambda item: (item["gpu_index"] is None, item["gpu_index"] if item["gpu_index"] is not None else 0)
        )
        for item in result:
            item["processes"] = sorted(
                item["processes"],
                key=lambda proc: proc.get("gpu_memory_mb") or 0,
                reverse=True
            )
        return result

    def cleanup_stale_allocations(self, max_age_hours: int = 24):
        """清理长时间未释放的分配（防止资源泄漏）。

        仅在任务缺失、attempt token 已被替换，或任务无活跃状态/进程证据时
        释放。租约解析或数据库查询不确定时 fail closed，避免同卡双训练 OOM。
        """
        with self._lock:
            current_time = datetime.now()
            stale_tasks = []

            for task_id, alloc_time in self.allocation_times.items():
                if current_time - alloc_time > timedelta(hours=max_age_hours):
                    if self._task_is_active(task_id):
                        logger.info(
                            "保留仍活跃或无法安全判定的 GPU 分配: %s（已超 %sh）",
                            _task_log_label(task_id),
                            max_age_hours,
                        )
                        continue
                    stale_tasks.append(task_id)

            for task_id in stale_tasks:
                logger.warning(f"清理过期的GPU分配: 任务 {_task_log_label(task_id)}")
                self.release_gpus_for_task(task_id)

            return len(stale_tasks)

    @staticmethod
    def _parse_training_lease_id(lease_id: str) -> Optional[tuple[str, str]]:
        """Parse the canonical ``training:{task_uuid}:{run_uuid}`` lease key."""
        if not lease_id.startswith("training:"):
            return None

        parts = lease_id.split(":")
        if len(parts) != 3:
            raise ValueError("training lease must contain exactly three segments")

        _, task_id, run_token = parts
        for field_name, value in (("task_id", task_id), ("run_token", run_token)):
            try:
                parsed = UUID(value)
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid {field_name} UUID") from exc
            if str(parsed) != value:
                raise ValueError(f"non-canonical {field_name} UUID")

        return task_id, run_token

    @staticmethod
    def _task_is_active(task_id: str) -> bool:
        """Fail closed unless a stale lease is confirmed safe to release."""
        base_task_id = task_id
        lease_run_token = None
        if task_id.startswith("training:"):
            try:
                parsed_lease = GPUResourceManager._parse_training_lease_id(task_id)
            except ValueError as exc:
                logger.warning(
                    "保留格式异常的训练 GPU 租约 %s: %s",
                    _task_log_label(task_id),
                    exc,
                )
                return True
            if parsed_lease is None:
                return True
            base_task_id, lease_run_token = parsed_lease

        try:
            from ..storage.services.training_task_service import training_task_service

            task = training_task_service.get_task(base_task_id)
        except Exception as exc:
            logger.warning("查询任务 %s 失败，保留 GPU 租约: %s", base_task_id, exc)
            return True

        if not task:
            return False

        if lease_run_token is not None and task.get("run_token") != lease_run_token:
            return False

        if task.get("status") in {"preparing", "running", "evaluating"}:
            return True

        return task.get("process_pid") is not None or task.get("process_status") in {
            "running",
            "stopping",
        }

    def get_gpu_info_simple(self) -> List[Dict]:
        """获取简化的GPU信息列表"""
        gpu_list = []
        for gpu_id in range(self.max_gpus):
            info = self._get_gpu_memory_info(gpu_id)
            gpu_list.append({
                "id": gpu_id,
                "name": info.get("gpu_name", f"GPU {gpu_id}"),
                "memory_total_gb": info["memory"]["total_gb"],
                "memory_used_gb": info["memory"]["used_gb"],
                "memory_free_gb": info["memory"]["free_gb"],
                "memory_usage_percent": info["memory"]["usage_percent"],
                "gpu_utilization": info["utilization"]["gpu_percent"],
                "temperature": info.get("temperature"),
                "power_usage_w": info["power"]["usage_w"] if "power" in info else None,
                "power_limit_w": info["power"]["limit_w"] if "power" in info else None,
                "is_allocated": gpu_id in self.gpu_allocations,
                "allocated_task": _task_log_label(
                    self.gpu_allocations.get(gpu_id)
                )
            })
        return gpu_list


# 全局GPU资源管理器实例
gpu_resource_manager = GPUResourceManager()
