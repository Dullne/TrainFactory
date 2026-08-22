"""
LLM Judge 客户端

为深度评估提供 LLM 调用能力。
"""

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from ..storage.services.outbound_endpoint_policy import create_pinned_async_client


@dataclass
class LLMConfig:
    """LLM 配置"""
    endpoint: str  # API 端点，如 http://localhost:8000/v1
    model: str  # 模型名称
    api_key: Optional[str] = None
    temperature: float = 0.0  # 评估时使用低温度以获得更一致的结果
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: int = 2048
    timeout: int = 60
    max_retries: int = 3
    user_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "model": self.model,
            "api_key": "***" if self.api_key else None,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }


class LLMJudge:
    """
    LLM Judge 客户端

    用于深度评估的 LLM 调用封装。
    支持 OpenAI 兼容的 API 格式。
    """

    def __init__(self, config: LLMConfig):
        """
        初始化 LLM Judge

        Args:
            config: LLM 配置
        """
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        """异步上下文管理器入口"""
        self._client = self._new_http_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        """获取 HTTP 客户端"""
        if self._client is None:
            self._client = self._new_http_client()
        return self._client

    def _new_http_client(self) -> httpx.AsyncClient:
        return create_pinned_async_client(
            self.config.endpoint,
            self.config.user_id,
            timeout=self.config.timeout,
        )

    async def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        发送聊天请求

        Args:
            prompt: 用户提示
            system_prompt: 系统提示（可选）
            temperature: 温度参数（可选，覆盖配置）
            max_tokens: 最大 token 数（可选，覆盖配置）

        Returns:
            str: LLM 响应文本
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return await self._call_api(
            messages=messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
        )

    async def _call_api(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """调用 LLM API"""
        client = self._get_client()

        # 构建请求 URL
        url = f"{self.config.endpoint.rstrip('/')}/chat/completions"

        # 构建请求头
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        # 构建请求体
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.config.top_p is not None:
            payload["top_p"] = self.config.top_p
        if self.config.top_k is not None:
            payload["top_k"] = self.config.top_k

        # 重试机制
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()

                data = response.json()
                content = data["choices"][0]["message"]["content"]

                # 清理响应
                content = self._clean_response(content)
                return content

            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code >= 500 or e.response.status_code in (408, 429):
                    # 服务器错误或限流（408/429）：指数退避但封顶 30s，
                    # 避免高并发 worker 各自退避到数百秒令任务看似挂死
                    await asyncio.sleep(min(2 ** attempt, 30))
                else:
                    # 其他客户端错误，不重试
                    raise
            except httpx.TimeoutException as e:
                last_error = e
                await asyncio.sleep(min(2 ** attempt, 30))
            except Exception as e:
                last_error = e
                await asyncio.sleep(min(2 ** attempt, 30))

        raise RuntimeError(f"LLM API call failed after {self.config.max_retries} retries: {last_error}")

    def _clean_response(self, content: str) -> str:
        """清理 LLM 响应"""
        # 移除 thinking 标签
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)

        # 移除可能的前后空白
        content = content.strip()

        return content

    async def close(self):
        """关闭客户端"""
        if self._client:
            await self._client.aclose()
            self._client = None


class LLMJudgePool:
    """
    LLM Judge 连接池

    支持多个 LLM 端点的负载均衡。
    """

    def __init__(self, configs: List[LLMConfig]):
        """
        初始化连接池

        Args:
            configs: LLM 配置列表
        """
        if not configs:
            raise ValueError("At least one LLM config is required")

        self.configs = configs
        self.judges = [LLMJudge(config) for config in configs]
        self._current_index = 0
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        """异步上下文管理器入口"""
        for judge in self.judges:
            await judge.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        for judge in self.judges:
            await judge.__aexit__(exc_type, exc_val, exc_tb)

    async def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        发送聊天请求（轮询负载均衡）

        Args:
            prompt: 用户提示
            system_prompt: 系统提示（可选）
            temperature: 温度参数（可选）
            max_tokens: 最大 token 数（可选）

        Returns:
            str: LLM 响应文本
        """
        # 轮询选择 judge
        async with self._lock:
            judge = self.judges[self._current_index]
            self._current_index = (self._current_index + 1) % len(self.judges)

        return await judge.chat(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def close(self):
        """关闭所有客户端"""
        for judge in self.judges:
            await judge.close()


def create_llm_judge(
    endpoint: str,
    model: str,
    api_key: Optional[str] = None,
    **kwargs,
) -> LLMJudge:
    """创建 LLM Judge 的便捷函数"""
    config = LLMConfig(
        endpoint=endpoint,
        model=model,
        api_key=api_key,
        user_id=kwargs.pop("user_id", None),
        **kwargs,
    )
    return LLMJudge(config)


def create_llm_judge_from_dict(config_dict: Dict[str, Any]) -> LLMJudge:
    """从字典创建 LLM Judge"""
    config = LLMConfig(
        endpoint=config_dict["endpoint"],
        model=config_dict["model"],
        api_key=config_dict.get("api_key"),
        temperature=config_dict.get("temperature", 0.0),
        top_p=config_dict.get("top_p"),
        top_k=config_dict.get("top_k"),
        max_tokens=config_dict.get("max_tokens", 2048),
        timeout=config_dict.get("timeout", 60),
        max_retries=config_dict.get("max_retries", 3),
        user_id=config_dict.get("user_id"),
    )
    return LLMJudge(config)
