"""
TrainFactory 命令行接口

支持子命令:
- train: 启动训练任务
- serve: 启动 API 服务器
- status: 查看任务状态
"""

import argparse
import sys
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def cmd_train(args):
    """执行训练命令"""
    from train_factory.train import train_with_config
    from train_factory.utils import setup_logging

    setup_logging(args.log_level)

    config = {
        "model_type": args.model_type,
        "base_model_path": args.model,
        "train_dataset_path": args.dataset,
        "dataset_configs": [{'path': args.dataset, 'max_samples': None, 'split': 'train'}],
        "output_dir": args.output,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "learning_rate": args.lr,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "use_lora": args.lora,
    }

    # 过滤 None 值
    config = {k: v for k, v in config.items() if v is not None}

    print("开始训练任务...")
    print(f"  模型类型: {args.model_type}")
    print(f"  模型: {args.model}")
    print(f"  数据集: {args.dataset}")

    try:
        result = train_with_config(config)
        print("\n训练完成!")
        print(f"  模型保存路径: {result.save_dir}")
        if result.final_metrics:
            print(f"  最终指标: {result.final_metrics}")
        return 0
    except Exception as e:
        logger.error(f"训练失败: {e}")
        print(f"\n训练失败: {e}")
        return 1


def cmd_serve(args):
    """启动 API 服务器"""
    from train_factory.api.server import run_server
    from train_factory.config import settings

    host = args.host or settings.api_host
    port = args.port or settings.api_port
    workers = settings.api_workers if args.workers is None else args.workers

    print("启动 API 服务器...")
    print(f"  地址: http://{host}:{port}")
    print(f"  工作进程: {workers}")

    try:
        run_server(host=host, port=port, workers=workers)
        return 0
    except Exception as e:
        logger.error(f"服务器启动失败: {e}")
        print(f"\n服务器启动失败: {e}")
        return 1


def cmd_status(args):
    """查看任务状态"""
    from train_factory.storage.services.training_task_service import training_task_service
    from train_factory.storage import init_db

    # 初始化数据库
    init_db()

    if args.task_id:
        # 查看特定任务
        task = training_task_service.get_task(args.task_id)
        if task:
            print("\n任务详情:")
            print(f"  ID: {task['task_id']}")
            print(f"  名称: {task.get('task_name')}")
            print(f"  模型类型: {task.get('model_type')}")
            print(f"  状态: {task.get('status')}")
            print(f"  进度: {task.get('progress', 0):.1f}%")
            print(f"  创建时间: {task.get('created_at')}")
            if task.get('error_message'):
                print(f"  错误: {task.get('error_message')}")
        else:
            print(f"任务不存在: {args.task_id}")
            return 1
    else:
        # 列出所有任务
        tasks, total = training_task_service.get_all_tasks(
            status=args.filter_status,
            model_type=args.filter_model_type,
            limit=args.limit
        )

        if total == 0:
            print("没有找到任务")
            return 0

        print(f"\n找到 {total} 个任务:\n")
        print(f"{'ID':<36} {'名称':<20} {'模型类型':<12} {'状态':<10} {'进度':<8}")
        print("-" * 90)

        for task in tasks:
            name = (task.get('task_name') or "")[:18]
            print(f"{task.get('task_id', ''):<36} {name:<20} {task.get('model_type', ''):<12} {task.get('status', ''):<10} {task.get('progress', 0):>5.1f}%")

    return 0


def cmd_version(args):
    """显示版本信息"""
    from train_factory import __version__
    print(f"TrainFactory v{__version__}")
    return 0


def create_parser() -> argparse.ArgumentParser:
    """创建命令行解析器"""
    parser = argparse.ArgumentParser(
        prog="train-factory",
        description="TrainFactory - Embedding/Reranker 模型训练框架"
    )
    parser.add_argument(
        "--version", "-v",
        action="store_true",
        help="显示版本信息"
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # train 子命令
    train_parser = subparsers.add_parser("train", help="启动训练任务")
    train_parser.add_argument(
        "--model-type", "-t",
        choices=["embedding", "reranker", "decoder_reranker", "llm"],
        default="embedding",
        help="模型类型 (默认: embedding)"
    )
    train_parser.add_argument(
        "--model", "-m",
        required=True,
        help="基础模型名称或路径"
    )
    train_parser.add_argument(
        "--dataset", "-d",
        required=True,
        help="训练数据集名称或路径"
    )
    train_parser.add_argument(
        "--output", "-o",
        help="输出目录"
    )
    train_parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=3,
        help="训练轮数 (默认: 3)"
    )
    train_parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=16,
        help="批量大小 (默认: 16)"
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="学习率 (默认: 2e-5)"
    )
    train_parser.add_argument(
        "--bf16",
        action="store_true",
        help="使用 BF16 混合精度"
    )
    train_parser.add_argument(
        "--fp16",
        action="store_true",
        help="使用 FP16 混合精度"
    )
    train_parser.add_argument(
        "--lora",
        action="store_true",
        help="使用 LoRA 微调"
    )
    train_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别 (默认: INFO)"
    )
    train_parser.set_defaults(func=cmd_train)

    # serve 子命令
    serve_parser = subparsers.add_parser("serve", help="启动 API 服务器")
    serve_parser.add_argument(
        "--host",
        help="服务器地址 (默认: 从配置读取)"
    )
    serve_parser.add_argument(
        "--port", "-p",
        type=int,
        help="服务器端口 (默认: 从配置读取)"
    )
    serve_parser.add_argument(
        "--workers", "-w",
        type=int,
        help="工作进程数 (默认: 从配置读取)"
    )
    serve_parser.set_defaults(func=cmd_serve)

    # status 子命令
    status_parser = subparsers.add_parser("status", help="查看任务状态")
    status_parser.add_argument(
        "task_id",
        nargs="?",
        help="任务 ID (不指定则列出所有任务)"
    )
    status_parser.add_argument(
        "--filter-status",
        choices=["pending", "running", "succeeded", "failed", "stopped"],
        help="按状态过滤"
    )
    status_parser.add_argument(
        "--filter-model-type",
        choices=["embedding", "reranker", "decoder_reranker", "llm"],
        help="按模型类型过滤"
    )
    status_parser.add_argument(
        "--limit", "-n",
        type=int,
        default=20,
        help="最大显示数量 (默认: 20)"
    )
    status_parser.set_defaults(func=cmd_status)

    # version 子命令
    version_parser = subparsers.add_parser("version", help="显示版本信息")
    version_parser.set_defaults(func=cmd_version)

    return parser


def main(argv: Optional[list] = None) -> int:
    """CLI 主入口"""
    parser = create_parser()
    args = parser.parse_args(argv)

    # 处理 --version 标志
    if args.version:
        return cmd_version(args)

    # 没有指定命令时显示帮助
    if not args.command:
        parser.print_help()
        return 0

    # 执行对应的命令
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
