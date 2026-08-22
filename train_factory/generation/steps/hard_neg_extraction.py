"""
正负例语句提取

- 从 negative chunks 中提取看似相关但实际误导的语句作为 hard negatives
- 从 positive chunks 中提取直接支持答案的语句作为 positive statements
用于 neg_detection_mode=statement 模式。
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from ..prompts import (
    ANSWER_REWRITE_SYSTEM_PROMPT,
    ANSWER_REWRITE_USER_TEMPLATE,
    EVIDENCE_PRUNING_SYSTEM_PROMPT,
    EVIDENCE_PRUNING_USER_TEMPLATE,
    EVIDENCE_REMOVAL_SYSTEM_PROMPT,
    EVIDENCE_REMOVAL_USER_TEMPLATE,
    NEG_CONFIRM_SYSTEM_PROMPT,
    NEG_CONFIRM_USER_TEMPLATE,
    NEG_EXTRACTION_SYSTEM_PROMPT,
    NEG_EXTRACTION_USER_TEMPLATE,
    POS_CONFIRM_SYSTEM_PROMPT,
    POS_CONFIRM_USER_TEMPLATE,
    POS_EXTRACTION_SYSTEM_PROMPT,
    POS_EXTRACTION_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)


def _parse_json_response(content: str) -> dict:
    """通用 JSON 响应解析"""
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    if "{" in content and "}" in content:
        start = content.find("{")
        end = content.rfind("}") + 1
        content = content[start:end]

    return json.loads(content)


class HardNegativeExtractionStep:
    """从 chunks 中提取细粒度的语句（负例：迷惑性语句；正例：支持性语句）"""

    def __init__(self, max_per_chunk: int = 3):
        self.max_per_chunk = max_per_chunk

    async def extract(
        self,
        query: str,
        answer: str,
        negative_chunks: List[str],
        llm_client: Any,
        max_per_chunk: Optional[int] = None,
        rerank_client: Any = None,
        embedding_client: Any = None,
        confirm: bool = False,
    ) -> List[str]:
        """
        从多个负例 chunks 中提取难负例语句

        Args:
            query: 用户查询
            answer: 期望答案
            negative_chunks: 负例 chunk 文本列表
            llm_client: LLM 客户端（LLMClientPool）
            max_per_chunk: 每个 chunk 最多提取的语句数
            rerank_client: Rerank 客户端（用于 delta 计算，优先）
            embedding_client: Embedding 客户端（delta 计算 fallback）
            confirm: 是否基于 delta 进行 LLM 二次确认

        Returns:
            提取的难负例语句列表（按分数降序）
        """
        per_chunk = max_per_chunk or self.max_per_chunk
        all_statements: List[Dict[str, Any]] = []

        for chunk in negative_chunks:
            try:
                statements = await self._extract_from_chunk(
                    query, answer, chunk, llm_client, per_chunk,
                )
                if statements and confirm and (rerank_client or embedding_client):
                    statements = await self._confirm_neg_from_chunk(
                        query, answer, chunk,
                        statements, llm_client,
                        rerank_client=rerank_client, embedding_client=embedding_client,
                    )
                all_statements.extend(statements)
            except Exception as e:
                logger.debug(f"[hard_neg] Failed to extract from chunk: {e}")
                continue

        # 按分数降序排列
        all_statements.sort(key=lambda x: x.get("score", 0), reverse=True)

        return [s["text"] for s in all_statements]

    async def _extract_from_chunk(
        self,
        query: str,
        answer: str,
        chunk_text: str,
        llm_client: Any,
        max_negatives: int,
    ) -> List[Dict[str, Any]]:
        """从单个 chunk 中提取难负例语句"""
        user_prompt = NEG_EXTRACTION_USER_TEMPLATE.format(
            query=query,
            answer=answer,
            chunk_text=chunk_text,
            max_negatives=max_negatives,
        )

        response = await llm_client.chat(
            prompt=user_prompt,
            system_prompt=NEG_EXTRACTION_SYSTEM_PROMPT,
        )

        if not response:
            return []

        try:
            result = _parse_json_response(response)
        except (json.JSONDecodeError, ValueError):
            logger.debug(f"[hard_neg] Failed to parse JSON response: {response[:200]}")
            return []

        negatives = result.get("negatives") or result.get("neg") or []
        scores = result.get("scores") or result.get("neg_scores") or []

        if isinstance(negatives, str):
            negatives = [negatives]

        statements = []
        for i, neg in enumerate(negatives[:max_negatives]):
            # Tolerate malformed elements (objects instead of strings, null or
            # non-numeric scores): skip only the bad element instead of letting
            # the exception propagate and discard the whole chunk's statements.
            if isinstance(neg, dict):
                text = str(neg.get("text") or neg.get("statement") or "").strip()
            elif isinstance(neg, str):
                text = neg.strip()
            else:
                continue
            if len(text) < 10:
                continue
            try:
                raw = scores[i] if i < len(scores) else 5.0
                score = float(raw) if raw is not None else 5.0
            except (TypeError, ValueError):
                score = 5.0
            statements.append({"text": text, "score": score})

        return statements

    async def extract_positives(
        self,
        query: str,
        answer: str,
        positive_chunks: List[str],
        llm_client: Any,
        max_per_chunk: Optional[int] = None,
        confirm: bool = False,
    ) -> List[str]:
        """
        从多个正例 chunks 中提取支持性语句

        Args:
            query: 用户查询
            answer: 期望答案
            positive_chunks: 正例 chunk 文本列表
            llm_client: LLM 客户端（LLMClientPool）
            max_per_chunk: 每个 chunk 最多提取的语句数
            confirm: 是否对提取结果进行二次确认（过滤幻觉和无关语句）

        Returns:
            提取的支持性语句列表（按分数降序）
        """
        per_chunk = max_per_chunk or self.max_per_chunk
        all_statements: List[Dict[str, Any]] = []

        for chunk in positive_chunks:
            try:
                statements = await self._extract_pos_from_chunk(
                    query, answer, chunk, llm_client, per_chunk,
                )
                if statements and confirm:
                    statements = await self._confirm_pos_statements(
                        query, answer, chunk, statements, llm_client,
                    )
                all_statements.extend(statements)
            except Exception as e:
                logger.debug(f"[pos_stmt] Failed to extract from chunk: {e}")
                continue

        all_statements.sort(key=lambda x: x.get("score", 0), reverse=True)

        return [s["text"] for s in all_statements]

    async def _extract_pos_from_chunk(
        self,
        query: str,
        answer: str,
        chunk_text: str,
        llm_client: Any,
        max_positives: int,
    ) -> List[Dict[str, Any]]:
        """从单个 chunk 中提取支持性语句"""
        user_prompt = POS_EXTRACTION_USER_TEMPLATE.format(
            query=query,
            answer=answer,
            chunk_text=chunk_text,
            max_positives=max_positives,
        )

        response = await llm_client.chat(
            prompt=user_prompt,
            system_prompt=POS_EXTRACTION_SYSTEM_PROMPT,
        )

        if not response:
            return []

        try:
            result = _parse_json_response(response)
        except (json.JSONDecodeError, ValueError):
            logger.debug(f"[pos_stmt] Failed to parse JSON response: {response[:200]}")
            return []

        # chunk 不相关则跳过
        if result.get("chunk_relevant") is False:
            logger.debug("[pos_stmt] Chunk marked as not relevant, skipping")
            return []

        positives = result.get("positives") or result.get("pos") or []
        scores = result.get("scores") or result.get("pos_scores") or []

        if isinstance(positives, str):
            positives = [positives]

        statements = []
        for i, pos in enumerate(positives[:max_positives]):
            # Tolerate malformed elements (objects instead of strings, null or
            # non-numeric scores): skip only the bad element instead of discarding
            # the whole chunk's statements.
            if isinstance(pos, dict):
                text = str(pos.get("text") or pos.get("statement") or "").strip()
            elif isinstance(pos, str):
                text = pos.strip()
            else:
                continue
            if len(text) < 10:
                continue
            try:
                raw = scores[i] if i < len(scores) else 5.0
                score = float(raw) if raw is not None else 5.0
            except (TypeError, ValueError):
                score = 5.0
            statements.append({"text": text, "score": score})

        return statements

    async def _confirm_pos_statements(
        self,
        query: str,
        answer: str,
        chunk_text: str,
        statements: List[Dict[str, Any]],
        llm_client: Any,
    ) -> List[Dict[str, Any]]:
        """对提取的正例语句进行二次确认，过滤幻觉和无关语句"""
        if not statements:
            return []

        statements_text = "\n".join(
            f"{i + 1}. {s['text']}" for i, s in enumerate(statements)
        )

        user_prompt = POS_CONFIRM_USER_TEMPLATE.format(
            query=query,
            answer=answer,
            chunk_text=chunk_text,
            statements_text=statements_text,
        )

        response = await llm_client.chat(
            prompt=user_prompt,
            system_prompt=POS_CONFIRM_SYSTEM_PROMPT,
        )

        if not response:
            return statements  # 确认失败则保留原结果

        try:
            result = _parse_json_response(response)
        except (json.JSONDecodeError, ValueError):
            logger.debug(f"[pos_confirm] Failed to parse JSON response: {response[:200]}")
            return statements

        if result.get("chunk_relevant") is False:
            logger.debug("[pos_confirm] Chunk marked as not relevant during confirmation")
            return []

        confirmations = result.get("confirmations") or []
        confirmed = []
        for i, stmt in enumerate(statements):
            if i < len(confirmations) and confirmations[i]:
                confirmed.append(stmt)
            else:
                logger.debug(f"[pos_confirm] Statement rejected: {stmt['text'][:60]}...")

        return confirmed

    # ============================================================
    # 负例 Delta + LLM 确认（支持 rerank / embedding）
    # ============================================================

    async def compute_rerank_delta(
        self,
        query: str,
        chunk_text: str,
        statements: List[str],
        rerank_client: Any = None,
        embedding_client: Any = None,
        precomputed_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        计算每个 statement 对 chunk 相似度的贡献 (delta)

        算法：移除 statement 后重新打分，delta = score_full - score_reduced。
        delta 越大表示该语句对 chunk 与 query 的相似度贡献越大（可能越具迷惑性）。
        支持 rerank（优先）或 embedding 作为打分后端。

        Args:
            query: 用户查询
            chunk_text: 原始 chunk 文本
            statements: 从 chunk 中提取的语句列表
            rerank_client: Rerank 客户端（优先）
            embedding_client: Embedding 客户端（fallback）
            precomputed_score: 预计算的 chunk 完整分数（避免重复打分）

        Returns:
            每个 statement 的 delta 信息列表，包含 keys:
            statement, score_full, score_reduced, delta, present
        """
        if not statements:
            return []

        # 获取 score_full
        if precomputed_score is not None:
            score_full = precomputed_score
        else:
            scores = await self._score_texts(query, [chunk_text], rerank_client, embedding_client)
            score_full = scores[0]

        # 构建 reduced_texts（移除每个 statement）
        reduced_texts = []
        present_flags = []
        for stmt in statements:
            present = stmt in chunk_text
            present_flags.append(present)
            if present:
                reduced_texts.append(chunk_text.replace(stmt, " ", 1))
            else:
                reduced_texts.append(chunk_text)

        # 批量打分
        reduced_scores = await self._score_texts(query, reduced_texts, rerank_client, embedding_client)

        # 计算 delta
        results = []
        for i, stmt in enumerate(statements):
            results.append({
                "statement": stmt,
                "score_full": score_full,
                "score_reduced": reduced_scores[i],
                "delta": score_full - reduced_scores[i],
                "present": present_flags[i],
            })
        return results

    async def confirm_negatives(
        self,
        query: str,
        answer: str,
        chunk_text: str,
        delta_info: List[Dict[str, Any]],
        llm_client: Any,
        threshold: float = 0.05,
        max_negatives: int = 5,
    ) -> List[str]:
        """
        基于 rerank delta 信息，LLM 确认难负例

        Args:
            query: 用户查询
            answer: 期望答案
            chunk_text: 原始 chunk 文本
            delta_info: compute_rerank_delta() 的返回值
            llm_client: LLM 客户端
            threshold: delta 阈值参考
            max_negatives: 最大选择数量

        Returns:
            确认后的难负例语句列表
        """
        if not delta_info:
            return []

        # 格式化 candidates 信息
        candidates_info = json.dumps(
            [
                {
                    "text": d["statement"],
                    "delta": round(d["delta"], 4),
                    "score_full": round(d["score_full"], 4),
                    "score_reduced": round(d["score_reduced"], 4),
                    "present": d["present"],
                }
                for d in delta_info
            ],
            ensure_ascii=False,
            indent=2,
        )

        user_prompt = NEG_CONFIRM_USER_TEMPLATE.format(
            query=query,
            answer=answer,
            chunk_text=chunk_text,
            candidates_info=candidates_info,
            threshold=threshold,
            max_negatives=max_negatives,
        )

        response = await llm_client.chat(
            prompt=user_prompt,
            system_prompt=NEG_CONFIRM_SYSTEM_PROMPT,
        )

        if not response:
            # fallback: 返回 delta >= threshold 的语句
            return [d["statement"] for d in delta_info if d["delta"] >= threshold][:max_negatives]

        try:
            result = _parse_json_response(response)
        except (json.JSONDecodeError, ValueError):
            logger.debug(f"[neg_confirm] Failed to parse JSON response: {response[:200]}")
            return [d["statement"] for d in delta_info if d["delta"] >= threshold][:max_negatives]

        selected = result.get("selected") or result.get("negatives") or result.get("neg") or []
        if isinstance(selected, str):
            selected = [selected]

        return [s for s in selected if isinstance(s, str) and len(s.strip()) >= 10][:max_negatives]

    async def _confirm_neg_from_chunk(
        self,
        query: str,
        answer: str,
        chunk_text: str,
        statements: List[Dict[str, Any]],
        llm_client: Any,
        rerank_client: Any = None,
        embedding_client: Any = None,
    ) -> List[Dict[str, Any]]:
        """对单个 chunk 提取的负例语句进行 delta 计算 + LLM 确认"""
        stmt_texts = [s["text"] for s in statements]

        # 计算 delta
        delta_info = await self.compute_rerank_delta(
            query, chunk_text, stmt_texts,
            rerank_client=rerank_client, embedding_client=embedding_client,
        )

        if not delta_info:
            return statements

        # LLM 确认
        confirmed_texts = await self.confirm_negatives(
            query, answer, chunk_text, delta_info, llm_client,
        )

        if not confirmed_texts:
            logger.debug("[neg_confirm] All statements rejected by confirmation")
            return []

        # 保留确认通过的语句（维持原始 score）
        confirmed_set = {t.strip() for t in confirmed_texts}
        return [s for s in statements if s["text"].strip() in confirmed_set]

    # ============================================================
    # Answer 重述（特性 4）
    # ============================================================

    async def rewrite_answer(
        self,
        query: str,
        answer: str,
        llm_client: Any,
    ) -> Optional[str]:
        """将 query + answer 合并为完整自包含陈述句

        Returns:
            重述后的文本，失败返回 None
        """
        user_prompt = ANSWER_REWRITE_USER_TEMPLATE.format(
            query=query,
            answer=answer,
        )

        try:
            response = await llm_client.chat(
                prompt=user_prompt,
                system_prompt=ANSWER_REWRITE_SYSTEM_PROMPT,
            )
        except Exception as e:
            logger.debug(f"[answer_rewrite] LLM call failed: {e}")
            return None

        if not response:
            return None

        # 返回纯文本（去掉可能的引号/代码块包裹）
        text = response.strip().strip('"').strip("'").strip("`")
        if len(text) < 5:
            logger.debug(f"[answer_rewrite] Response too short: {text}")
            return None

        return text

    # ============================================================
    # 通用打分（支持 rerank / embedding）
    # ============================================================

    async def _score_texts(
        self,
        query: str,
        texts: List[str],
        rerank_client: Any = None,
        embedding_client: Any = None,
    ) -> List[float]:
        """用 rerank 或 embedding 给文本打分

        优先使用 rerank_client.score()，fallback 到 embedding_client.similarity()。
        """
        if rerank_client:
            return await rerank_client.score(query, texts)
        if embedding_client:
            return await embedding_client.similarity(query, texts)
        raise ValueError("No scoring client available")

    # ============================================================
    # 负例 Chunk 按分数分层（特性 6）
    # ============================================================

    async def score_negative_chunks(
        self,
        query: str,
        chunks: List[str],
        worst_pos: float,
        rerank_client: Any = None,
        embedding_client: Any = None,
    ) -> Tuple[List[str], List[str]]:
        """给负例 chunk 打分并分层（支持 rerank / embedding）

        Args:
            query: 用户查询
            chunks: 负例 chunk 列表
            worst_pos: 最差正例 chunk 的分数阈值
            rerank_client: Rerank 客户端（优先）
            embedding_client: Embedding 客户端（fallback）

        Returns:
            (auto_accept, need_llm):
            - auto_accept: 分数 >= worst_pos 的 chunk（确定是难负例）
            - need_llm: 分数 < worst_pos 的 chunk（低分丢弃）
        """
        if not chunks:
            return [], []

        try:
            scores = await self._score_texts(query, chunks, rerank_client, embedding_client)
        except Exception as e:
            logger.debug(f"[neg_chunk_scoring] Scoring failed: {e}")
            return chunks, []  # fallback: 全部当作 auto_accept

        auto_accept = []
        need_llm = []
        for chunk, score in zip(chunks, scores):
            try:
                score_f = float(score)
            except (TypeError, ValueError):
                score_f = 0.0
            if score_f >= worst_pos:
                auto_accept.append(chunk)
            else:
                need_llm.append(chunk)

        logger.debug(
            f"[neg_chunk_scoring] {len(auto_accept)} auto_accept (>= {worst_pos:.4f}), "
            f"{len(need_llm)} need_llm"
        )
        return auto_accept, need_llm

    # ============================================================
    # Rerank 评分筛选（特性 2）
    # ============================================================

    async def score_and_classify_statements(
        self,
        query: str,
        statements: List[str],
        best_pos: float,
        worst_pos: float,
        mode: str = "negative",
        rerank_client: Any = None,
        embedding_client: Any = None,
    ) -> List[Dict[str, Any]]:
        """给 statement 打分并按正例分数区间分类（支持 rerank / embedding）

        Args:
            query: 用户查询
            statements: 语句文本列表
            best_pos: 最佳正例 chunk 的分数
            worst_pos: 最差正例 chunk 的分数
            mode: "negative" 或 "positive"，决定分类标签
            rerank_client: Rerank 客户端（优先）
            embedding_client: Embedding 客户端（fallback）

        Returns:
            [{"text": str, "score": float, "stmt_type": str}, ...]
        """
        if not statements:
            return []

        try:
            scores = await self._score_texts(query, statements, rerank_client, embedding_client)
        except Exception as e:
            logger.debug(f"[score_classify] Scoring failed: {e}")
            # fallback: 不分类，全部返回
            default_type = "hard" if mode == "negative" else "ge_worst"
            return [{"text": s, "score": 0.0, "stmt_type": default_type} for s in statements]

        results = []
        for stmt, score in zip(statements, scores):
            try:
                score_f = float(score)
            except (TypeError, ValueError):
                score_f = 0.0

            if mode == "negative":
                if score_f >= best_pos:
                    stmt_type = "very_hard"
                elif score_f >= worst_pos:
                    stmt_type = "hard"
                else:
                    stmt_type = "medium"
            else:
                if score_f >= best_pos:
                    stmt_type = "gt_best"
                elif score_f >= worst_pos:
                    stmt_type = "ge_worst"
                else:
                    stmt_type = "lt_worst"

            results.append({"text": stmt, "score": score_f, "stmt_type": stmt_type})

        return results

    # ============================================================
    # 证据去除（特性 5）
    # ============================================================

    async def remove_evidence(
        self,
        query: str,
        answer: str,
        chunk_text: str,
        llm_client: Any,
    ) -> Optional[str]:
        """从正例 chunk 中去除证据，生成硬负例变体

        Returns:
            去除证据后的 chunk 文本，失败返回 None
        """
        user_prompt = EVIDENCE_REMOVAL_USER_TEMPLATE.format(
            query=query,
            answer=answer,
            chunk_text=chunk_text,
        )

        try:
            response = await llm_client.chat(
                prompt=user_prompt,
                system_prompt=EVIDENCE_REMOVAL_SYSTEM_PROMPT,
            )
        except Exception as e:
            logger.debug(f"[evidence_removal] LLM call failed: {e}")
            return None

        if not response:
            return None

        try:
            result = _parse_json_response(response)
        except (json.JSONDecodeError, ValueError):
            logger.debug(f"[evidence_removal] Failed to parse JSON: {response[:200]}")
            return None

        if not result.get("is_positive", False):
            logger.debug("[evidence_removal] Chunk marked as not positive, skipping")
            return None

        removed = result.get("removed_chunk", "")
        if not removed or len(removed.strip()) < 10:
            return None

        return removed.strip()

    # ============================================================
    # 证据裁剪（特性 5）
    # ============================================================

    async def prune_evidence(
        self,
        query: str,
        answer: str,
        chunk_text: str,
        llm_client: Any,
    ) -> Optional[str]:
        """裁剪正例 chunk 中的无关内容，生成更干净的正例

        Returns:
            裁剪后的 chunk 文本，失败返回 None
        """
        user_prompt = EVIDENCE_PRUNING_USER_TEMPLATE.format(
            query=query,
            answer=answer,
            chunk_text=chunk_text,
        )

        try:
            response = await llm_client.chat(
                prompt=user_prompt,
                system_prompt=EVIDENCE_PRUNING_SYSTEM_PROMPT,
            )
        except Exception as e:
            logger.debug(f"[evidence_pruning] LLM call failed: {e}")
            return None

        if not response:
            return None

        try:
            result = _parse_json_response(response)
        except (json.JSONDecodeError, ValueError):
            logger.debug(f"[evidence_pruning] Failed to parse JSON: {response[:200]}")
            return None

        if not result.get("is_positive", False):
            logger.debug("[evidence_pruning] Chunk marked as not positive, skipping")
            return None

        trimmed = result.get("trimmed_chunk", "")
        if not trimmed or len(trimmed.strip()) < 10:
            return None

        return trimmed.strip()
