"""
DynamicExplicitNegativesRankingLoss - 支持动态负例数量的对比学习损失函数

该损失函数支持每个样本拥有不同数量的负例，适用于训练数据生成系统产出的数据。

定位说明:
- 这是 TrainFactory 本地保留的 embedding 训练专用损失。
- 它依赖当前 universal-format -> padded negative columns 的数据展平策略。
- sentence-transformers 自带的 MultipleNegativesRankingLoss / CachedMultipleNegativesRankingLoss
  不直接覆盖“每个样本显式负例数不固定、并且需要忽略空 padding 负例”的场景。
- 它不属于 qwen3-rerank-trainer 的职责范围，后者只维护 qwen3 decoder reranker
  的 ranking / RL 能力，不维护 SentenceTransformer embedding 训练侧的自定义 loss。

参考来源: 内部 embedding 训练实验实现。
"""

import math
import os
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DynamicExplicitNegativesRankingLoss(nn.Module):
    """
    动态负例版本的 Row-wise InfoNCE Loss

    支持每个样本拥有不同数量的负例，无需填充到统一长度。

    与标准 MultipleNegativesRankingLoss 的区别：
    - 标准版本：要求所有样本的负例数量一致（使用 torch.stack）
    - 动态版本：每个样本可以有不同数量的负例（逐样本计算）

    预期输入格式: sentence_features: [anchor_batch, positive_batch, negative1_batch, ...].
    对于每个样本 i，计算其与自己的候选（positive + 可用的负例）的相似度，
    然后使用 CrossEntropy 计算损失（target=0，positive 在第0个位置）。

    Args:
        model: SentenceTransformer 模型
        scale: 温度系数，相似度乘以scale后再计算softmax (默认: 20.0)
        normalize_by_logc: 是否用 CE - log(C) 归一化（C为每个样本的候选数量）
                          默认从环境变量 NORMALIZE_LOSS_BY_LOGC 读取
        ignore_empty: 是否忽略空字符串的负例（通过embedding norm判断）(默认: True)
        empty_threshold: 判断空embedding的阈值 (默认: 1e-6)

    Example:
        >>> from sentence_transformers import SentenceTransformer
        >>> model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
        >>> loss = DynamicExplicitNegativesRankingLoss(model, scale=20.0)
        >>>
        >>> # 训练数据格式: anchor, positive, negative_0, negative_1, ...
        >>> # 每个样本的负例数量可以不同
    """

    def __init__(
        self,
        model,
        scale: float = 20.0,
        normalize_by_logc: Optional[bool] = None,
        ignore_empty: bool = True,
        empty_threshold: float = 1e-6,
    ):
        """
        初始化动态负例对比学习损失函数。

        Args:
            model: SentenceTransformer 模型
            scale: 温度系数，相似度乘以scale后再计算softmax
            normalize_by_logc: 是否用 CE - log(C) 归一化（C为每个样本的候选数量）
            ignore_empty: 是否忽略空字符串的负例（通过embedding norm判断）
            empty_threshold: 判断空embedding的阈值
        """
        super().__init__()
        self.model = model
        self.scale = scale

        # 如果未指定，从环境变量读取
        if normalize_by_logc is None:
            normalize_by_logc = os.getenv("NORMALIZE_LOSS_BY_LOGC", "False").lower() == "true"
        self.normalize_by_logc = normalize_by_logc

        self.ignore_empty = ignore_empty
        self.empty_threshold = empty_threshold

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor = None):
        """
        前向传播，支持动态负例数量

        Args:
            sentence_features: List of feature dicts, [anchor, positive, neg1, neg2, ...]
                             每个 dict 包含 token 级特征 (input_ids, attention_mask, ...)
                             或已计算的 'sentence_embedding'
            labels: 不使用（兼容接口）

        Returns:
            loss: Tensor，标量损失值
        """
        # 0. 将 sentence_features 转为 list（可能是 generator）
        if not isinstance(sentence_features, (list, tuple)):
            sentence_features = list(sentence_features)

        # 1. 检测空文本 padding（在 forward 之前，通过 attention_mask 判断）
        # 空字符串 tokenize 后产生的 token 数量因模型而异：
        #   - Decoder 模型 (Qwen3, LLaMA): 1 个 BOS token
        #   - Encoder 模型 (BERT, BGE): 2 个 special tokens ([CLS] + [SEP])
        # 使用 <= 2 以兼容两种架构
        batch_size = None
        empty_mask = []  # empty_mask[col_idx] = BoolTensor(B,), True = 空文本
        for feat in sentence_features:
            attn = feat.get("attention_mask")
            if attn is not None:
                if batch_size is None:
                    batch_size = attn.size(0)
                # 有效 token <= 2 视为空文本（兼容 [CLS]+[SEP] 和单 BOS）
                empty_mask.append(attn.sum(dim=1) <= 2)
            else:
                empty_mask.append(None)

        # 2. 获取/计算所有文本的句向量
        first = sentence_features[0]
        if isinstance(first, dict) and "sentence_embedding" in first:
            embs = [feat["sentence_embedding"] for feat in sentence_features]
        else:
            embs = [self.model(feat)["sentence_embedding"] for feat in sentence_features]

        # embs[0]: anchor_batch (B, D)
        # embs[1:]: candidate_batches [positive, neg1, neg2, ...]

        batch_size = embs[0].size(0)
        device = embs[0].device

        # 3. 逐样本计算 loss
        losses = []

        for i in range(batch_size):
            anchor = embs[0][i]  # (D,)

            candidates = []
            for j, emb_batch in enumerate(embs[1:]):
                col_idx = j + 1  # 0=anchor, 1=positive, 2+=negatives

                # 跳过空文本 padding（j=0 是 positive，不跳过）
                if self.ignore_empty and j > 0:
                    if col_idx < len(empty_mask) and empty_mask[col_idx] is not None:
                        if empty_mask[col_idx][i]:
                            continue
                    else:
                        # Fallback: embedding norm 判断
                        if torch.norm(emb_batch[i]) < self.empty_threshold:
                            continue

                candidates.append(emb_batch[i])

            if len(candidates) < 2:
                losses.append(torch.tensor(0.1, device=device))
                continue

            candidates_tensor = torch.stack(candidates, dim=0)  # (C, D)

            anchor_n = F.normalize(anchor.unsqueeze(0), p=2, dim=1)
            candidates_n = F.normalize(candidates_tensor, p=2, dim=1)

            scores = torch.mm(anchor_n, candidates_n.t()).squeeze(0) * self.scale

            target = torch.zeros(1, dtype=torch.long, device=device)
            loss_i = F.cross_entropy(scores.unsqueeze(0), target, reduction="none").squeeze()

            if self.normalize_by_logc:
                C = len(candidates)
                loss_i = loss_i - math.log(C)

            losses.append(loss_i)

        if len(losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return torch.stack(losses).mean()

    def get_config_dict(self) -> dict[str, any]:
        """
        返回损失函数的配置信息，用于序列化

        Returns:
            配置字典
        """
        return {
            "scale": self.scale,
            "normalize_by_logc": self.normalize_by_logc,
            "ignore_empty": self.ignore_empty,
            "empty_threshold": self.empty_threshold,
        }
