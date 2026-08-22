"""
Training Metrics Logger.

Responsible for saving training loss and metrics data to local files.
Provides structured logging of training progress, epoch summaries, and final results.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from threading import Lock

from ..utils.strict_json import sanitize_json_value

logger = logging.getLogger(__name__)


class TrainingMetricsLogger:
    """训练Loss本地文件管理器"""

    def __init__(self, output_dir: str, task_id: str):
        """
        初始化Loss管理器

        Args:
            output_dir: 训练输出目录
            task_id: 训练任务ID
        """
        self.output_dir = output_dir
        self.task_id = task_id
        self.lock = Lock()

        # 设置日志目录结构: {output_dir}/logs/training/{task_id}/
        self.logs_dir = Path(output_dir) / "logs" / "training" / task_id
        self.loss_history_file = self.logs_dir / "loss_history.jsonl"
        self.training_metrics_file = self.logs_dir / "training_metrics.json"
        self.metadata_file = self.logs_dir / "metadata.json"

        # 创建目录
        self._ensure_directories()

        # 初始化训练指标
        self.training_metrics = {
            "task_id": task_id,
            "start_time": datetime.now().isoformat(),
            "total_steps": 0,
            "epochs_completed": 0,
            "best_train_loss": None,
            "best_eval_loss": None,
            "loss_records_count": 0,
            "last_updated": None,
            "epoch_summaries": [],
            "final_results": {}
        }

        logger.info(f"Loss管理器初始化完成: {self.logs_dir}")

    def _ensure_directories(self):
        """确保目录结构存在"""
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"创建日志目录: {self.logs_dir}")
        except Exception as e:
            logger.error(f"创建日志目录失败: {e}")
            raise

    def save_loss_record(self, step: int, metrics: Dict[str, Any], epoch: Optional[float] = None):
        """
        保存单次loss记录到JSONL文件

        Args:
            step: 训练步数
            metrics: 指标字典（包含train_loss, eval_loss等）
            epoch: 当前epoch（可选）
        """
        with self.lock:
            try:
                # 构建记录
                record = sanitize_json_value({
                    "step": step,
                    "timestamp": datetime.now().isoformat(),
                    **metrics
                })

                if epoch is not None:
                    record["epoch"] = sanitize_json_value(epoch)

                # 写入JSONL文件（每行一个JSON）
                with open(self.loss_history_file, 'a', encoding='utf-8') as f:
                    json.dump(
                        record,
                        f,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    f.write('\n')

                # 更新训练指标
                self._update_training_metrics(
                    step,
                    sanitize_json_value(metrics),
                    sanitize_json_value(epoch),
                )

                logger.debug(f"保存loss记录: step={step}, metrics={list(metrics.keys())}")

            except Exception as e:
                logger.error(f"保存loss记录失败: {e}")

    def _update_training_metrics(self, step: int, metrics: Dict[str, Any], epoch: Optional[float]):
        """更新训练指标汇总"""
        try:
            # 更新基础信息
            self.training_metrics["total_steps"] = max(self.training_metrics["total_steps"], step)
            self.training_metrics["loss_records_count"] += 1
            self.training_metrics["last_updated"] = datetime.now().isoformat()

            if epoch is not None:
                self.training_metrics["epochs_completed"] = max(
                    self.training_metrics["epochs_completed"], epoch
                )

            # 更新最佳loss
            if "train_loss" in metrics:
                train_loss = metrics["train_loss"]
                if (
                    isinstance(train_loss, (int, float))
                    and not isinstance(train_loss, bool)
                    and (
                        self.training_metrics["best_train_loss"] is None
                        or train_loss
                        < self.training_metrics["best_train_loss"]
                    )
                ):
                    self.training_metrics["best_train_loss"] = train_loss

            if "eval_loss" in metrics:
                eval_loss = metrics["eval_loss"]
                if (
                    isinstance(eval_loss, (int, float))
                    and not isinstance(eval_loss, bool)
                    and (
                        self.training_metrics["best_eval_loss"] is None
                        or eval_loss
                        < self.training_metrics["best_eval_loss"]
                    )
                ):
                    self.training_metrics["best_eval_loss"] = eval_loss

            # 保存到文件
            with open(self.training_metrics_file, 'w', encoding='utf-8') as f:
                json.dump(
                    sanitize_json_value(self.training_metrics),
                    f,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )

        except Exception as e:
            logger.error(f"更新训练指标失败: {e}")

    def finalize_epoch(self, epoch: int, epoch_metrics: Dict[str, Any]):
        """
        完成一个epoch，保存epoch汇总信息

        Args:
            epoch: epoch编号
            epoch_metrics: epoch汇总指标
        """
        with self.lock:
            try:
                epoch_summary = sanitize_json_value({
                    "epoch": epoch,
                    "timestamp": datetime.now().isoformat(),
                    **epoch_metrics
                })

                # 添加到epoch汇总列表
                self.training_metrics["epoch_summaries"].append(epoch_summary)

                # 保存到文件
                with open(self.training_metrics_file, 'w', encoding='utf-8') as f:
                    json.dump(
                        sanitize_json_value(self.training_metrics),
                        f,
                        ensure_ascii=False,
                        indent=2,
                        allow_nan=False,
                    )

                logger.info(f"Epoch {epoch} 汇总已保存: {list(epoch_metrics.keys())}")

            except Exception as e:
                logger.error(f"保存epoch汇总失败: {e}")

    def get_loss_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取loss历史记录

        Args:
            limit: 限制返回的记录数量

        Returns:
            loss记录列表
        """
        try:
            if not self.loss_history_file.exists():
                return []

            records = []
            with open(self.loss_history_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        records.append(
                            sanitize_json_value(json.loads(line.strip()))
                        )

            # 应用限制
            if limit and len(records) > limit:
                records = records[-limit:]

            return records

        except Exception as e:
            logger.error(f"获取loss历史失败: {e}")
            return []

    def get_training_metrics(self) -> Dict[str, Any]:
        """获取训练指标汇总"""
        try:
            if self.training_metrics_file.exists():
                with open(self.training_metrics_file, 'r', encoding='utf-8') as f:
                    return sanitize_json_value(json.load(f))
            else:
                return sanitize_json_value(self.training_metrics)
        except Exception as e:
            logger.error(f"获取训练指标失败: {e}")
            return sanitize_json_value(self.training_metrics)

    def get_summary(self) -> Dict[str, Any]:
        """
        获取训练汇总信息

        Returns:
            汇总信息字典
        """
        try:
            summary = {
                "task_id": self.task_id,
                "loss_file_path": str(self.loss_history_file),
                "metrics_file_path": str(self.training_metrics_file),
                "total_records": self.training_metrics.get("loss_records_count", 0),
                "training_duration_seconds": self.training_metrics.get("training_duration_seconds"),
                "best_train_loss": self.training_metrics.get("best_train_loss"),
                "best_eval_loss": self.training_metrics.get("best_eval_loss"),
                "total_steps": self.training_metrics.get("total_steps", 0),
                "epochs_completed": self.training_metrics.get("epochs_completed", 0),
                "start_time": self.training_metrics.get("start_time"),
                "end_time": self.training_metrics.get("end_time"),
                "status": self.training_metrics.get("status", "running"),
                "epoch_summaries": self.training_metrics.get("epoch_summaries", []),
                "final_metrics": self.training_metrics.get("final_metrics", {})
            }
            return sanitize_json_value(summary)
        except Exception as e:
            logger.error(f"生成汇总信息失败: {e}")
            return {}

    def finalize_training(self, final_metrics: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        完成训练，保存最终指标

        Args:
            final_metrics: 最终训练指标

        Returns:
            汇总信息
        """
        with self.lock:
            try:
                # 更新完成时间
                self.training_metrics["end_time"] = datetime.now().isoformat()
                self.training_metrics["status"] = "completed"

                # 添加最终指标
                if final_metrics:
                    self.training_metrics["final_metrics"] = (
                        sanitize_json_value(final_metrics)
                    )

                # 计算训练时长
                if "start_time" in self.training_metrics:
                    start_time = datetime.fromisoformat(self.training_metrics["start_time"])
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    self.training_metrics["training_duration_seconds"] = duration
                    self.training_metrics["duration_formatted"] = self._format_duration(duration)

                # 保存最终指标
                with open(self.training_metrics_file, 'w', encoding='utf-8') as f:
                    json.dump(
                        sanitize_json_value(self.training_metrics),
                        f,
                        ensure_ascii=False,
                        indent=2,
                        allow_nan=False,
                    )

                logger.info(f"训练指标已完成保存: {self.training_metrics_file}")

                return self.get_summary()

            except Exception as e:
                logger.error(f"完成训练指标保存失败: {e}")
                return None

    def save_metadata(self, metadata: Dict[str, Any]):
        """
        保存训练元数据到文件

        Args:
            metadata: 元数据字典
        """
        with self.lock:
            try:
                metadata = sanitize_json_value(
                    {
                        **metadata,
                        "created_at": datetime.now().isoformat(),
                        "task_id": self.task_id,
                    }
                )

                with open(self.metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(
                        metadata,
                        f,
                        ensure_ascii=False,
                        indent=2,
                        allow_nan=False,
                    )

                logger.info(f"元数据已保存: {self.metadata_file}")

            except Exception as e:
                logger.error(f"保存元数据失败: {e}")

    def get_metadata(self) -> Dict[str, Any]:
        """
        读取训练元数据

        Returns:
            元数据字典
        """
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return sanitize_json_value(json.load(f))
            else:
                return {}
        except Exception as e:
            logger.error(f"读取元数据失败: {e}")
            return {}

    def _format_duration(self, seconds: Optional[float]) -> Optional[str]:
        """将秒数转换为人类可读格式"""
        if seconds is None:
            return None

        total_seconds = int(seconds)
        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60

        if days > 0:
            return f"{days}天{hours}时{minutes}分{secs}秒"
        elif hours > 0:
            return f"{hours}时{minutes}分{secs}秒"
        elif minutes > 0:
            return f"{minutes}分{secs}秒"
        else:
            return f"{secs}秒"


# 全局loss管理器实例字典 {task_id: TrainingMetricsLogger}
_metrics_loggers: Dict[str, TrainingMetricsLogger] = {}
_managers_lock = Lock()


def get_training_metrics_logger(output_dir: str, task_id: str) -> TrainingMetricsLogger:
    """获取或创建loss管理器实例"""
    with _managers_lock:
        if task_id not in _metrics_loggers:
            _metrics_loggers[task_id] = TrainingMetricsLogger(output_dir, task_id)
        return _metrics_loggers[task_id]


def cleanup_training_metrics_logger(task_id: str, final_metrics: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    清理loss管理器实例

    Args:
        task_id: 任务ID
        final_metrics: 最终训练指标

    Returns:
        汇总信息
    """
    with _managers_lock:
        if task_id in _metrics_loggers:
            try:
                summary = _metrics_loggers[task_id].finalize_training(final_metrics)
                del _metrics_loggers[task_id]
                logger.info(f"清理loss管理器: {task_id}")
                return summary
            except Exception as e:
                logger.error(f"清理loss管理器失败: {e}")
                return None
        return None
