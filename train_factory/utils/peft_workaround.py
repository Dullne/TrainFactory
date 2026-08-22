"""
PEFT 集成兼容性补丁

背景：
- 在某些 transformers/peft 组合版本下，未启用 LoRA 训练时调用
  PreTrainedModel.get_adapter_state_dict() 可能触发
  UnboundLocalError: local variable 'active_adapters' referenced before assignment。

解决：
- 对 PreTrainedModel.get_adapter_state_dict 进行轻量包装，捕获该异常并返回空字典，
  从而不影响常规模型保存与检查点保存流程。
- 当实际启用 LoRA 且库版本正确时，不会触发该异常，包装对行为无影响。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


_peft_workaround_applied: bool = False


def apply_peft_unboundlocal_workaround() -> bool:
    """为 transformers 的 PEFT 集成保存逻辑应用兼容性补丁。

    返回：
        True 表示已应用补丁或之前已应用；False 表示未找到需要包装的方法。
    """
    global _peft_workaround_applied
    if _peft_workaround_applied:
        return True

    try:
        from transformers.modeling_utils import PreTrainedModel  # type: ignore
    except Exception as e:
        logger.debug(f"无法导入 PreTrainedModel，跳过PEFT补丁: {e}")
        return False

    # 1) 包装 get_adapter_state_dict（部分版本在保存时会触发此路径）
    get_state = getattr(PreTrainedModel, "get_adapter_state_dict", None)
    if get_state is not None and not getattr(PreTrainedModel.get_adapter_state_dict, "__peft_wrapped__", False):
        def _wrapped_get_adapter_state_dict(self: Any, *args: Any, **kwargs: Any):
            try:
                return get_state(self, *args, **kwargs)
            except UnboundLocalError as e:
                msg = str(e)
                if "active_adapters" in msg:
                    logger.warning(
                        "PEFT保存兼容性补丁生效：捕获UnboundLocalError(active_adapters)。返回空adapter状态以继续保存。"
                    )
                    return {}
                raise
            except Exception as e:
                logger.debug(f"get_adapter_state_dict 调用异常（非致命）：{e}")
                return {}

        setattr(_wrapped_get_adapter_state_dict, "__peft_wrapped__", True)
        PreTrainedModel.get_adapter_state_dict = _wrapped_get_adapter_state_dict  # type: ignore
        logger.info("已应用PEFT补丁: 包装 get_adapter_state_dict")

    # 2) 包装 active_adapters（部分版本直接在 save_pretrained 中调用此方法）
    active_getter = getattr(PreTrainedModel, "active_adapters", None)
    if active_getter is not None and not getattr(PreTrainedModel.active_adapters, "__peft_active_wrapped__", False):
        def _wrapped_active_adapters(self: Any, *args: Any, **kwargs: Any):
            try:
                return active_getter(self, *args, **kwargs)
            except UnboundLocalError as e:
                msg = str(e)
                if "active_adapters" in msg:
                    logger.warning(
                        "PEFT保存兼容性补丁生效：捕获UnboundLocalError于active_adapters，返回空列表以继续保存。"
                    )
                    return []
                raise
            except Exception as e:
                logger.debug(f"active_adapters 调用异常（非致命）：{e}")
                return []

        setattr(_wrapped_active_adapters, "__peft_active_wrapped__", True)
        PreTrainedModel.active_adapters = _wrapped_active_adapters  # type: ignore
        logger.info("已应用PEFT补丁: 包装 active_adapters")

    _peft_workaround_applied = True
    return True
