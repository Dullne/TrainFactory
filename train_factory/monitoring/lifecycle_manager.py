"""
应用生命周期管理器
处理服务启动和关闭时的清理工作
"""
import atexit
import logging
import signal
import sys
from typing import List, Callable

logger = logging.getLogger(__name__)


class LifecycleManager:
    """应用生命周期管理器"""

    def __init__(self):
        self.shutdown_handlers: List[Callable] = []
        self._shutdown_in_progress = False  # 防止重入标志
        self._register_signal_handlers()
        self._register_exit_handler()

    def add_shutdown_handler(self, handler: Callable):
        """添加关闭处理器"""
        self.shutdown_handlers.append(handler)
        logger.info(f"已添加关闭处理器: {handler.__name__}")

    def _register_signal_handlers(self):
        """注册信号处理器"""
        try:
            # 注册SIGINT (Ctrl+C) 和 SIGTERM信号处理器
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            logger.info("已注册信号处理器")
        except Exception as e:
            logger.warning(f"注册信号处理器失败: {str(e)}")

    def _register_exit_handler(self):
        """注册程序退出处理器"""
        atexit.register(self._exit_handler)
        logger.info("已注册退出处理器")

    def _signal_handler(self, signum, frame):
        """信号处理器"""
        # 防止重入：如果已在关闭过程中，直接退出
        if self._shutdown_in_progress:
            logger.warning(f"收到信号 {signum}，但关闭已在进行中，直接退出...")
            sys.exit(0)

        logger.info(f"收到信号 {signum}，开始优雅关闭...")
        self._shutdown()
        sys.exit(0)

    def _exit_handler(self):
        """退出处理器"""
        # 防止重入：如果已在关闭过程中，跳过
        if self._shutdown_in_progress:
            logger.info("程序即将退出，但关闭已在进行中，跳过清理...")
            return

        logger.info("程序即将退出，执行清理工作...")
        self._shutdown()

    def _shutdown(self):
        """执行关闭逻辑"""
        # 设置关闭标志，防止重入
        if self._shutdown_in_progress:
            logger.warning("关闭已在进行中，跳过重复执行...")
            return

        self._shutdown_in_progress = True
        logger.info("开始执行关闭处理器...")

        try:
            for handler in self.shutdown_handlers:
                try:
                    logger.info(f"执行关闭处理器: {handler.__name__}")
                    result = handler()
                    if result:
                        logger.info(f"关闭处理器 {handler.__name__} 执行成功")
                    else:
                        logger.warning(f"关闭处理器 {handler.__name__} 执行失败")
                except Exception as e:
                    logger.error(f"执行关闭处理器 {handler.__name__} 时出错: {str(e)}", exc_info=True)

            logger.info("关闭处理器执行完成")
        finally:
            # 确保标志被重置（虽然程序即将退出）
            self._shutdown_in_progress = False


# 全局生命周期管理器实例
lifecycle_manager = LifecycleManager()


def get_lifecycle_manager() -> LifecycleManager:
    """获取全局生命周期管理器"""
    return lifecycle_manager
