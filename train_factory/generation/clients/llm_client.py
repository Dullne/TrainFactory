"""
LLM 客户端

提供异步批量 LLM 调用能力，支持多端点负载均衡。
"""

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable

import httpx

from ...storage.services.outbound_endpoint_policy import create_pinned_async_client
from ...storage.services.inference_authorization_service import authorize_inference_model


@dataclass
class LLMConfig:
    """LLM 配置"""
    endpoint: str
    model: str
    api_key: Optional[str] = None
    temperature: float = 0.7
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
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }


class LLMClient:
    """
    LLM 客户端

    支持 OpenAI 兼容的 API 格式，提供异步调用能力。
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = self._new_http_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
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
        """发送聊天请求"""
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
        await asyncio.to_thread(
            authorize_inference_model, self.config.endpoint,
            self.config.model, self.config.user_id,
        )
        client = self._get_client()
        url = f"{self.config.endpoint.rstrip('/')}/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()

                data = response.json()
                # content can legitimately be null: reasoning / tool-call models
                # return HTTP 200 with "content": null. Coerce to "" so
                # _clean_response's re.sub does not raise TypeError (which would
                # then be misclassified as a retryable error and waste all retries).
                message = data["choices"][0]["message"]
                content = message.get("content") or ""
                return self._clean_response(content)

            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code >= 500:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise
            except httpx.TransportError as e:
                # 仅重试传输层错误（超时/连接失败）。解析错误（JSON schema 变化、
                # KeyError）是确定性的，立即抛出而不是耗尽全部重试。
                last_error = e
                await asyncio.sleep(2 ** attempt)

        raise RuntimeError(f"LLM API call failed after {self.config.max_retries} retries: {last_error}")

    def _clean_response(self, content: str) -> str:
        """清理 LLM 响应"""
        # 移除 thinking 标签
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)
        return content.strip()

    async def close(self):
        """关闭客户端"""
        if self._client:
            await self._client.aclose()
            self._client = None


class LLMClientPool:
    """
    LLM 客户端池

    支持多端点负载均衡和并发控制。
    """

    def __init__(
        self,
        configs: List[LLMConfig],
        max_concurrency: int = 10,
    ):
        if not configs:
            raise ValueError("At least one LLM config is required")

        self.configs = configs
        self.clients = [LLMClient(config) for config in configs]
        self.max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._current_index = 0
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        for client in self.clients:
            await client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        for client in self.clients:
            await client.__aexit__(exc_type, exc_val, exc_tb)

    async def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """发送聊天请求（带并发控制和负载均衡）"""
        async with self._semaphore:
            # 轮询选择客户端
            async with self._lock:
                client = self.clients[self._current_index]
                self._current_index = (self._current_index + 1) % len(self.clients)

            return await client.chat(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    async def batch_chat(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[str]:
        """
        批量聊天请求

        Args:
            prompts: 提示列表
            system_prompt: 系统提示
            temperature: 温度
            max_tokens: 最大 token 数
            progress_callback: 进度回调 (completed, total)

        Returns:
            响应列表（与输入顺序对应）
        """
        completed = 0
        total = len(prompts)

        async def process_one(index: int, prompt: str) -> tuple:
            nonlocal completed
            result = await self.chat(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            completed += 1
            if progress_callback:
                progress_callback(completed, total)
            return index, result

        tasks = [
            process_one(i, prompt)
            for i, prompt in enumerate(prompts)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 按原始顺序排列结果
        ordered_results = [None] * len(prompts)
        for result in results:
            if isinstance(result, Exception):
                continue
            index, content = result
            ordered_results[index] = content

        return ordered_results

    async def close(self):
        """关闭所有客户端"""
        for client in self.clients:
            await client.close()


def fix_json_errors(json_str: str) -> str:
    """
    尝试修复常见的 JSON 格式错误

    Args:
        json_str: 可能格式不正确的 JSON 字符串

    Returns:
        修复后的 JSON 字符串
    """
    # 移除 markdown 代码块标记
    json_str = json_str.strip()
    if json_str.startswith("```"):
        json_str = re.sub(r'^```\w*\n?', '', json_str)
        json_str = re.sub(r'\n?```$', '', json_str)

    # 移除可能的前缀说明文字
    json_match = re.search(r'[\[{]', json_str)
    if json_match:
        json_str = json_str[json_match.start():]

    # 修复尾部逗号
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

    # 修复单引号
    # 这是一个简单的修复，复杂情况可能需要更复杂的解析
    # json_str = json_str.replace("'", '"')

    return json_str


def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    """
    从 LLM 响应中提取 JSON

    Args:
        response: LLM 响应文本

    Returns:
        解析后的 JSON 对象，解析失败返回 None
    """
    try:
        fixed = fix_json_errors(response)
        return json.loads(fixed)
    except json.JSONDecodeError:
        # 尝试找到 JSON 对象
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, response, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # 尝试找到 JSON 数组
        array_pattern = r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'
        matches = re.findall(array_pattern, response, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        return None
