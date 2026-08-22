"""Embedding数据格式与损失函数的统一适配器。

该模块负责：
1. 根据数据列自动识别数据格式（或使用用户配置）。
2. 清理 `_benchmark_*` 等非训练字段，确保列顺序符合 SentenceTransformers Loss 要求。
3. 将标签列标准化为 `label` / `score`，并在需要时进行数值转换。
4. 生成损失函数元数据，供训练阶段创建具体的 SentenceTransformers Loss 实例。

数据格式说明参见 `training_docs/data_formats` 目录。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from sentence_transformers.sentence_transformer.losses import (
        AnglELoss,
        CoSENTLoss,
        CosineSimilarityLoss,
        GISTEmbedLoss,
        MarginMSELoss,
        MultipleNegativesRankingLoss,
    )
except ImportError:  # pragma: no cover - compatibility with sentence-transformers 3.x
    from sentence_transformers.losses import (
        AnglELoss,
        CoSENTLoss,
        CosineSimilarityLoss,
        GISTEmbedLoss,
        MarginMSELoss,
        MultipleNegativesRankingLoss,
    )
try:  # DistillKLDivLoss 仅在较新的 sentence-transformers 版本存在
    from sentence_transformers.sentence_transformer.losses import (  # type: ignore
        DistillKLDivLoss,
    )
except ImportError:  # pragma: no cover - 可选依赖
    try:
        from sentence_transformers.losses import DistillKLDivLoss  # type: ignore
    except ImportError:
        DistillKLDivLoss = None  # type: ignore
from sentence_transformers import SentenceTransformer

# Import custom loss function
from train_factory.losses.contrastive.dynamic_negatives_loss import DynamicExplicitNegativesRankingLoss

try:  # 避免在类型检查以外强制依赖datasets（某些环境可能尚未安装）
    from datasets import Dataset
except Exception:  # pragma: no cover - 仅在runtime缺失datasets时触发
    Dataset = Any  # type: ignore


logger = logging.getLogger(__name__)

DISTILL_KL_LOSS_NAME = "DistillKLDivLoss"


LOSS_ALIAS_MAP = {
    "auto": "auto",
    "cosine": CosineSimilarityLoss.__name__,
    "cosine_similarity": CosineSimilarityLoss.__name__,
    "cosinesimilarityloss": CosineSimilarityLoss.__name__,
    CosineSimilarityLoss.__name__.lower(): CosineSimilarityLoss.__name__,
    "cosent": CoSENTLoss.__name__,
    CoSENTLoss.__name__.lower(): CoSENTLoss.__name__,
    "angle": AnglELoss.__name__,
    AnglELoss.__name__.lower(): AnglELoss.__name__,
    "mnrloss": MultipleNegativesRankingLoss.__name__,
    "multiple_negatives": MultipleNegativesRankingLoss.__name__,
    MultipleNegativesRankingLoss.__name__.lower(): MultipleNegativesRankingLoss.__name__,
    "gist": GISTEmbedLoss.__name__,
    GISTEmbedLoss.__name__.lower(): GISTEmbedLoss.__name__,
    "margin": MarginMSELoss.__name__,
    "marginmse": MarginMSELoss.__name__,
    MarginMSELoss.__name__.lower(): MarginMSELoss.__name__,
    "distill": DISTILL_KL_LOSS_NAME,
    "distill_kl": DISTILL_KL_LOSS_NAME,
    DISTILL_KL_LOSS_NAME.lower(): DISTILL_KL_LOSS_NAME,
    # Custom loss functions
    "dynamic_neg": DynamicExplicitNegativesRankingLoss.__name__,
    "dynamic_negatives": DynamicExplicitNegativesRankingLoss.__name__,
    DynamicExplicitNegativesRankingLoss.__name__.lower(): DynamicExplicitNegativesRankingLoss.__name__,
}


FORMAT_DEFAULT_LOSS = {
    "cosine_pairs": CosineSimilarityLoss.__name__,
    "mnr_pairs": MultipleNegativesRankingLoss.__name__,
    "mnr_triplets": MultipleNegativesRankingLoss.__name__,
    "mnr_multi_negatives": MultipleNegativesRankingLoss.__name__,
    "margin_triplets": MarginMSELoss.__name__,
    "score_triplets": MarginMSELoss.__name__,
    "margin_multi_margins": MarginMSELoss.__name__,
    "margin_multi_scores": MarginMSELoss.__name__,
    # Custom formats
    "dynamic_negatives": DynamicExplicitNegativesRankingLoss.__name__,
}


SUPPORTED_LOSSES = {
    "cosine_pairs": {
        CosineSimilarityLoss.__name__,
        CoSENTLoss.__name__,
        AnglELoss.__name__,
    },
    "mnr_pairs": {
        MultipleNegativesRankingLoss.__name__,
        GISTEmbedLoss.__name__,
    },
    "mnr_triplets": {
        MultipleNegativesRankingLoss.__name__,
        GISTEmbedLoss.__name__,
    },
    "mnr_multi_negatives": {MultipleNegativesRankingLoss.__name__},
    "margin_triplets": {MarginMSELoss.__name__},
    "score_triplets": {MarginMSELoss.__name__, DISTILL_KL_LOSS_NAME},
    "margin_multi_margins": {MarginMSELoss.__name__},
    "margin_multi_scores": {MarginMSELoss.__name__, DISTILL_KL_LOSS_NAME},
    # Custom formats
    "dynamic_negatives": {
        DynamicExplicitNegativesRankingLoss.__name__,
        MultipleNegativesRankingLoss.__name__,  # 支持标准 MNR
    },
}


@dataclass
class EmbeddingRecipeSpec:
    """封装单个数据集/分割的训练配置信息。"""

    format_key: str
    loss_name: str
    target_column: Optional[str]
    dataset_name: str
    split_name: str
    loss_kwargs: Dict[str, Any]
    label_normalization: Optional[str] = None


class EmbeddingRecipeError(RuntimeError):
    """Embedding数据格式适配异常。"""


def normalize_loss_name(loss_name: Optional[str]) -> str:
    if not loss_name:
        return "auto"
    key = loss_name.strip().lower()
    return LOSS_ALIAS_MAP.get(key, loss_name)


def _drop_auxiliary_columns(dataset: Dataset) -> Dataset:
    drop_cols = [c for c in dataset.column_names if c.startswith("_benchmark")]
    if drop_cols:
        dataset = dataset.remove_columns(drop_cols)
    return dataset


def _sort_negative_columns(column_names: List[str]) -> List[str]:
    negatives = [c for c in column_names if c.startswith("negative")]
    def _sort_key(name: str) -> Tuple[int, str]:
        suffix = name.replace("negative", "")
        try:
            return (int(suffix), name)
        except ValueError:
            return (9999, name)
    return sorted(negatives, key=_sort_key)


COSINE_PAIR_COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "sentence1": ("sentence1", "premise", "text1", "text_a"),
    "sentence2": ("sentence2", "hypothesis", "text2", "text_b"),
    "label": ("label", "score", "similarity"),
}

MNR_TRIPLETS_COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "query": ("query",),
    "positive": ("positive", "pos"),
    "negative": ("negative", "neg"),
}


def _contains_alias_columns(columns: Iterable[str], alias_map: Dict[str, Tuple[str, ...]]) -> bool:
    column_set = set(columns)
    for aliases in alias_map.values():
        if not any(alias in column_set for alias in aliases):
            return False
    return True


def _has_any(columns: set, *candidates: str) -> bool:
    return any(c in columns for c in candidates)


def _get_first(sample: Dict[str, Any], *candidates: str) -> Any:
    for c in candidates:
        if c in sample:
            return sample.get(c)
    return None


def detect_data_format(dataset: Dataset) -> str:
    if dataset is None or len(dataset) == 0:
        raise EmbeddingRecipeError("数据集为空，无法自动识别数据格式。")

    base_cols = [c for c in dataset.column_names if not c.startswith("_benchmark")]
    col_set = set(base_cols)
    sample = dataset[0]

    # 结构优先的粗分类（按“逻辑列数”）
    base_len = len(col_set)

    # 2 列：一律按 mnr_pairs 处理（批内负例）
    if base_len == 2:
        logger.info("结构检测: 2列，判定为 mnr_pairs（MultipleNegativesRankingLoss）")
        return "mnr_pairs"

    # 3 列：优先判断是否为带标签的句对相似（cosine_pairs）
    if base_len == 3:
        # 认为 label-like 列指向 Cosine（句对+标签）
        if any(k in col_set for k in ("label", "score", "similarity")):
            logger.info("结构检测: 3列且存在标签列，判定为 cosine_pairs")
            return "cosine_pairs"

        # 未带标签：判断是否是数组多负（pos/neg 为 list）
        pos_val_3 = _get_first(sample, "pos", "positive", "sentence2", "text2")
        neg_val_3 = _get_first(sample, "neg", "negative")
        if isinstance(pos_val_3, list) or isinstance(neg_val_3, list):
            # 根据负例数量是否一致选择 mnr_multi_negatives 或 dynamic_negatives
            def _neg_list3(row):
                v = row.get("neg") if "neg" in row else row.get("negative", [])
                return v if isinstance(v, list) else []
            neg_counts3 = [len(_neg_list3(s)) for s in dataset]
            if len(set(neg_counts3)) == 1:
                logger.info("结构检测: 3列且存在数组负例，数量一致 → mnr_multi_negatives")
                return "mnr_multi_negatives"
            else:
                logger.info("结构检测: 3列且存在数组负例，数量不一致 → dynamic_negatives")
                return "dynamic_negatives"

        # 否则默认为三元组 mnr_triplets
        logger.info("结构检测: 3列无标签，判定为 mnr_triplets")
        return "mnr_triplets"

    # 4 列：统一按 margin 家族处理
    if base_len == 4:
        # 四列细分：
        # - 第三列（negative）不是数组：看最后一列是否为数组 → score_triplets / margin_triplets
        # - 第三列是数组：最后一列若为数组，按 len 比较区分 margin_multi_scores / margin_multi_margins
        neg_val4 = _get_first(sample, "neg", "negative")
        label_like4 = _get_first(sample, "scores", "margins", "label", "margin")

        if not isinstance(neg_val4, list):
            # 非数组负例：
            if isinstance(label_like4, list):
                if len(label_like4) == 2:
                    logger.info("结构检测: 4列 且 neg 为标量 且 label为长度2数组 → score_triplets")
                    return "score_triplets"
                # 其他长度数组：不常见，按 margin_triplets 处理（由训练侧做转换或报错）
                logger.info("结构检测: 4列 且 neg 为标量 且 label为数组(非长度2) → 视作 margin_triplets")
                return "margin_triplets"
            logger.info("结构检测: 4列 且 neg 为标量 且 label为标量 → margin_triplets")
            return "margin_triplets"

        # neg 为数组：多负例
        if isinstance(label_like4, list):
            neg_len = len(neg_val4)
            lab_len = len(label_like4)
            if lab_len == neg_len + 1:
                logger.info("结构检测: 4列 且 neg为数组 且 label长度=neg长度+1 → margin_multi_scores")
                return "margin_multi_scores"
            if lab_len == neg_len:
                logger.info("结构检测: 4列 且 neg为数组 且 label长度=neg长度 → margin_multi_margins")
                return "margin_multi_margins"
            # 长度不匹配，按字段名猜测
            if "scores" in sample:
                logger.info("结构检测: 4列 且 neg为数组 但长度不匹配，存在scores键 → margin_multi_scores")
                return "margin_multi_scores"
            if "margins" in sample:
                logger.info("结构检测: 4列 且 neg为数组 但长度不匹配，存在margins键 → margin_multi_margins")
                return "margin_multi_margins"
            logger.info("结构检测: 4列 且 neg为数组 但长度不匹配，回退 mnr_multi_negatives")
            return "mnr_multi_negatives"

        # neg 为数组，但 label 不是数组：回退 MNR 多负格式
        logger.info("结构检测: 4列 且 neg为数组 且 label非数组 → mnr_multi_negatives")
        return "mnr_multi_negatives"

    # 兜底：保留旧的特征型识别（universal 与细分格式）
    # 优先检测 Universal 格式（固定 5 列：query, pos, neg, pos_scores, neg_scores）
    if {"query", "pos", "neg", "pos_scores", "neg_scores"}.issubset(col_set):
        # 验证一致性：pos/neg 与 pos_scores/neg_scores 数量必须逐条一致；否则直接报错
        def _validate_universal_and_resolve() -> str:
            neg_counts = []
            for idx, row in enumerate(dataset):
                pos = row.get("pos")
                neg = row.get("neg")
                ps = row.get("pos_scores")
                ns = row.get("neg_scores")
                if not (isinstance(pos, list) and isinstance(neg, list) and isinstance(ps, list) and isinstance(ns, list)):
                    raise EmbeddingRecipeError(
                        "Universal: pos/neg/pos_scores/neg_scores 必须均为数组类型"
                    )
                if len(pos) != len(ps) or len(neg) != len(ns):
                    raise EmbeddingRecipeError(
                        f"Universal: 第{idx+1}条样本标签数量不一致，"
                        f"pos={len(pos)} pos_scores={len(ps)}; neg={len(neg)} neg_scores={len(ns)}"
                    )
                neg_counts.append(len(neg))

            # 若都一致，优先走带分数的多负：margin_multi_scores；
            # （如未来需要不带分数的分支，可在此改为 mnr_multi_negatives/dynamic_negatives）
            if len(set(neg_counts)) == 1:
                logger.info(
                    "Universal: 校验通过，负例数一致(%d)，将解析为 margin_multi_scores",
                    neg_counts[0],
                )
                return "margin_multi_scores"
            else:
                logger.info(
                    "Universal: 校验通过，负例数不一致(%d-%d)，将解析为 margin_multi_scores 并在展开时按最小K对齐",
                    min(neg_counts or [0]), max(neg_counts or [0]),
                )
                return "margin_multi_scores"

        return _validate_universal_and_resolve()

    if {"sentence1", "sentence2", "label"}.issubset(col_set) or _contains_alias_columns(col_set, COSINE_PAIR_COLUMN_ALIASES):
        return "cosine_pairs"
    # 支持 query 的别名 anchor
    if (_has_any(col_set, "query", "anchor") and _has_any(col_set, "positive") and _has_any(col_set, "negative")):
        # 检测是否为 4 列格式（有 label 列）
        if "label" in col_set or "margin" in col_set:
            label_col = "label" if "label" in col_set else "margin"
            label_val = sample.get(label_col)
            # score_triplets: label 是长度为 2 的数组 [pos_score, neg_score]
            if isinstance(label_val, list) and len(label_val) == 2:
                return "score_triplets"
            # margin_triplets: label/margin 是单个数值
            elif isinstance(label_val, (int, float)):
                return "margin_triplets"
        # 否则是 mnr_triplets（3 列，无标签）
        return "mnr_triplets"

    # 检测 query/pos/neg 格式（支持pos/neg别名）
    if (_has_any(col_set, "query", "anchor") and _has_any(col_set, "pos", "positive") and _has_any(col_set, "neg", "negative")):
        # margin_multi_scores / margin_multi_margins（数组标签）优先识别
        # - scores: [pos_score, neg1, neg2, ...]（长度应为 1 + #neg）
        # - margins: [m1, m2, ...]（长度应为 #neg）
        scores_val = sample.get("scores")
        margins_val = sample.get("margins")
        pos_val = _get_first(sample, "pos", "positive")
        neg_val = _get_first(sample, "neg", "negative")
        if isinstance(pos_val, list) and isinstance(neg_val, list):
            if isinstance(scores_val, list) and len(scores_val) >= 1:
                return "margin_multi_scores"
            if isinstance(margins_val, list) and len(margins_val) >= 1:
                return "margin_multi_margins"

        # 检测是否为 4 列格式（有 label 列）
        if "label" in col_set or "margin" in col_set:
            label_col = "label" if "label" in col_set else "margin"
            label_val = sample.get(label_col)
            # score_triplets: label 是长度为 2 的数组 [pos_score, neg_score]
            if isinstance(label_val, list) and len(label_val) == 2:
                return "score_triplets"
            # margin_triplets: label/margin 是单个数值
            elif isinstance(label_val, (int, float)):
                return "margin_triplets"

        # 没有 label 列，检测 pos/neg 的类型
        pos_val = _get_first(sample, "pos", "positive")
        neg_val = _get_first(sample, "neg", "negative")
        # 检查是否为数组格式（Universal 格式）
        if isinstance(pos_val, list) and isinstance(neg_val, list):
            # 智能检测：根据负例数量是否一致来决定格式
            # 1. 检查所有样本的负例数量
            def _neg_list(row):
                v = row.get("neg") if "neg" in row else row.get("negative", [])
                return v if isinstance(v, list) else []
            neg_counts = [len(_neg_list(s)) for s in dataset]
            all_same = len(set(neg_counts)) == 1  # 所有样本的负例数量是否一致

            if all_same:
                # 负例数量一致 → mnr_multi_negatives（需要转换为多列格式）
                logger.info(f"检测到 Universal 格式，负例数量一致 ({neg_counts[0]} 个)，将作为 mnr_multi_negatives 格式处理")
                return "mnr_multi_negatives"
            else:
                # 负例数量不一致 → dynamic_negatives
                logger.info(f"检测到 Universal 格式，负例数量不一致（{min(neg_counts)}-{max(neg_counts)} 个），将作为 dynamic_negatives 格式处理")
                return "dynamic_negatives"
        # 如果是字符串格式 -> mnr_triplets
        elif isinstance(pos_val, str) and isinstance(neg_val, str):
            return "mnr_triplets"

    if (_has_any(col_set, "query", "anchor") and _has_any(col_set, "positive")):
        negative_cols = [c for c in base_cols if c.startswith("negative")]
        if negative_cols:
            if "label" in col_set:
                label = sample.get("label")
                if isinstance(label, list):
                    if len(label) == len(negative_cols):
                        return "margin_multi_margins"
                    if len(label) == len(negative_cols) + 1:
                        return "margin_multi_scores"
            # 兼容 scores/margins 与 anchor 别名
            if isinstance(sample.get("scores"), list):
                return "margin_multi_scores"
            if isinstance(sample.get("margins"), list):
                return "margin_multi_margins"
            return "mnr_multi_negatives"
        return "mnr_pairs"
    if "label" in col_set and any(c.startswith("negative") for c in col_set):
        label = sample.get("label")
        if isinstance(label, list):
            negative_cols = [c for c in base_cols if c.startswith("negative")]
            if len(label) == len(negative_cols):
                return "margin_multi_margins"
            if len(label) == len(negative_cols) + 1:
                return "margin_multi_scores"

    raise EmbeddingRecipeError(
        "无法自动识别数据格式，请在配置中显式设置 embedding_data_format。"
    )


def _ensure_columns_order(dataset: Dataset, ordered_columns: List[str]) -> Dataset:
    existing = [c for c in ordered_columns if c in dataset.column_names]
    missing = [c for c in ordered_columns if c not in dataset.column_names]
    if missing:
        raise EmbeddingRecipeError(f"数据列缺失: {missing}")
    return dataset.select_columns(existing)


def _rename_columns(dataset: Dataset, mapping: Dict[str, str]) -> Dataset:
    for src, dst in mapping.items():
        if src in dataset.column_names and src != dst:
            dataset = dataset.rename_column(src, dst)
    return dataset


def _standardize_cosine_pair_columns(dataset: Dataset) -> Dataset:
    mapping: Dict[str, str] = {}
    for canonical, aliases in COSINE_PAIR_COLUMN_ALIASES.items():
        if canonical in dataset.column_names:
            continue
        for alias in aliases:
            if alias in dataset.column_names and alias != canonical:
                mapping[alias] = canonical
                break
    if mapping:
        dataset = _rename_columns(dataset, mapping)
    return dataset


def _standardize_mnr_columns(dataset: Dataset) -> Dataset:
    """标准化 MNR 格式的列名：pos→positive, neg→negative"""
    mapping: Dict[str, str] = {}

    # 处理 positive 别名
    if "positive" not in dataset.column_names and "pos" in dataset.column_names:
        mapping["pos"] = "positive"

    # 处理 negative 别名（单列）
    if "negative" not in dataset.column_names and "neg" in dataset.column_names:
        mapping["neg"] = "negative"

    # 处理 negative_N 别名 (neg_0 → negative_0, neg_1 → negative_1, ...)
    for col in dataset.column_names:
        if col.startswith("neg_"):
            new_col = col.replace("neg_", "negative_", 1)
            if new_col not in dataset.column_names:  # 避免覆盖已存在的列
                mapping[col] = new_col

    if mapping:
        dataset = _rename_columns(dataset, mapping)

    return dataset


def _convert_score_pairs_to_margin(dataset: Dataset) -> Dataset:
    def _convert(row: Dict[str, Any]) -> Dict[str, Any]:
        scores = row["label"]
        if not isinstance(scores, list) or len(scores) != 2:
            raise EmbeddingRecipeError("score_triplets 数据期望label为长度2的列表： [positive_score, negative_score]")
        return {"label": scores[0] - scores[1]}

    return dataset.map(_convert)


def _convert_multi_scores_for_margin(dataset: Dataset) -> Dataset:
    def _convert(row: Dict[str, Any]) -> Dict[str, Any]:
        scores = row["label"]
        if not isinstance(scores, list) or len(scores) < 2:
            raise EmbeddingRecipeError("multi_scores 数据期望label包含正例+若干负例分数")
        pos, negs = scores[0], scores[1:]
        return {"label": [pos - neg for neg in negs]}

    return dataset.map(_convert)


def _normalize_similarity_labels(dataset: Dataset, strategy: str) -> Dataset:
    strategy = (strategy or "auto").lower()
    if strategy in ("none", "off"):
        return dataset

    if strategy in ("auto", "minus_one_to_one"):
        def _to_minus_one_to_one(row: Dict[str, Any]) -> Dict[str, Any]:
            value = row["label"]
            return {"label": value * 2 - 1}

        return dataset.map(_to_minus_one_to_one)

    if strategy == "zero_to_one":
        def _to_zero_one(row: Dict[str, Any]) -> Dict[str, Any]:
            value = row["label"]
            return {"label": (value + 1) / 2}

        return dataset.map(_to_zero_one)

    raise EmbeddingRecipeError(f"未知的标签归一化策略: {strategy}")


def prepare_dataset_for_recipe(
    dataset: Dataset,
    *,
    split_name: str,
    dataset_name: str,
    requested_format: Optional[str] = None,
    requested_loss: Optional[str] = None,
    label_norm_strategy: str = "auto",
    loss_temperature: Optional[float] = None,
    guide_model_name: Optional[str] = None,
    max_negatives: Optional[int] = None,
) -> Tuple[Dataset, EmbeddingRecipeSpec]:
    dataset = _drop_auxiliary_columns(dataset)

    format_key = requested_format or "auto"
    format_key = format_key.lower()
    if format_key == "auto":
        format_key = detect_data_format(dataset)

    default_loss = FORMAT_DEFAULT_LOSS.get(format_key)
    if not default_loss:
        raise EmbeddingRecipeError(f"数据格式 {format_key} 暂不支持")

    normalized_loss = normalize_loss_name(requested_loss)
    loss_name = default_loss if normalized_loss in (None, "auto") else normalized_loss

    if loss_name not in SUPPORTED_LOSSES.get(format_key, {default_loss}):
        raise EmbeddingRecipeError(
            f"数据格式 {format_key} 不支持损失函数 {loss_name}。可选: {sorted(SUPPORTED_LOSSES.get(format_key, []))}"
        )

    target_column: Optional[str] = None
    applied_label_norm: Optional[str] = None

    if format_key == "cosine_pairs":
        target_column = "label"
        dataset = _standardize_cosine_pair_columns(dataset)
        if loss_name in {CoSENTLoss.__name__, AnglELoss.__name__}:
            dataset = _normalize_similarity_labels(dataset, label_norm_strategy)
            applied_label_norm = label_norm_strategy
        dataset = _ensure_columns_order(dataset, ["sentence1", "sentence2", "label"])

    elif format_key == "mnr_pairs":
        dataset = _standardize_mnr_columns(dataset)  # 标准化 pos→positive, neg→negative
        dataset = _ensure_columns_order(dataset, ["query", "positive"])
        # MultipleNegativesRankingLoss、CachedMultipleNegativesRankingLoss 和 GISTEmbedLoss 都需要 anchor 列名
        if loss_name in {
            MultipleNegativesRankingLoss.__name__,
            "CachedMultipleNegativesRankingLoss",
            GISTEmbedLoss.__name__
        }:
            dataset = _rename_columns(dataset, {"query": "anchor"})

    elif format_key == "mnr_triplets":
        dataset = _standardize_mnr_columns(dataset)  # 标准化 pos→positive, neg→negative
        dataset = _ensure_columns_order(dataset, ["query", "positive", "negative"])
        # MultipleNegativesRankingLoss、CachedMultipleNegativesRankingLoss 和 GISTEmbedLoss 都需要 anchor 列名
        if loss_name in {
            MultipleNegativesRankingLoss.__name__,
            "CachedMultipleNegativesRankingLoss",
            GISTEmbedLoss.__name__
        }:
            dataset = _rename_columns(dataset, {"query": "anchor"})

    elif format_key == "mnr_multi_negatives":
        # MNR 多负样本格式处理
        # 统一架构：converter 输出数组格式 {query, pos:[p], neg:[n1,n2]}
        #          training 时转为多列格式 {query, positive, negative_0, negative_1}
        #          与 MARGIN_MULTI_MARGINS 处理方式一致

        if "pos" in dataset.column_names or "positive" in dataset.column_names:
            first_sample = dataset[0]
            pos_val = first_sample.get("pos") or first_sample.get("positive")
            neg_val = first_sample.get("neg") or first_sample.get("negative")

            # 如果是数组格式（来自 converter 或 Universal），转换为多列格式
            if isinstance(pos_val, list) and isinstance(neg_val, list):
                logger.info("检测到数组格式，转换为 mnr_multi_negatives 多列格式（用于 MNR Loss）")

                def convert_array_to_multi_columns(row):
                    """将数组格式转为多列格式"""
                    query = row["query"]
                    pos_list = row.get("pos") or row.get("positive")
                    neg_list = row.get("neg") or row.get("negative")

                    # converter 输出的格式：pos 只有1个元素
                    # 但为了兼容性，仍然支持多个 pos（展开）
                    records = []
                    for pos in pos_list:
                        if pos:  # 跳过空 positive
                            record = {
                                "query": query,
                                "positive": pos
                            }
                            # 添加多个 negative 列
                            for idx, neg in enumerate(neg_list):
                                if neg:  # 跳过空 negative
                                    record[f"negative_{idx}"] = neg

                            # 只有当有负例时才添加记录
                            if neg_list:
                                records.append(record)

                    return records

                # 转换数据集
                all_data = [row for row in dataset]
                expanded_data = []
                for row in all_data:
                    records = convert_array_to_multi_columns(row)
                    expanded_data.extend(records)

                logger.info(f"数组格式转换: {len(all_data)} 条存储记录 → {len(expanded_data)} 条训练记录")

                from datasets import Dataset as HFDataset
                dataset = HFDataset.from_list(expanded_data)

        dataset = _standardize_mnr_columns(dataset)  # 标准化 pos→positive, neg_N→negative_N
        ordered = ["query", "positive"] + _sort_negative_columns(dataset.column_names)
        dataset = _ensure_columns_order(dataset, ordered)
        # MultipleNegativesRankingLoss、CachedMultipleNegativesRankingLoss 和 GISTEmbedLoss 都需要 anchor 列名
        if loss_name in {
            MultipleNegativesRankingLoss.__name__,
            "CachedMultipleNegativesRankingLoss",
            GISTEmbedLoss.__name__
        }:
            dataset = _rename_columns(dataset, {"query": "anchor"})

    elif format_key == "margin_triplets":
        # 标准化列名: anchor→query, pos→positive, neg→negative, margin→label
        dataset = _rename_columns(dataset, {
            "anchor": "query",
            "pos": "positive",
            "neg": "negative",
            "margin": "label"
        })
        dataset = _ensure_columns_order(dataset, ["query", "positive", "negative", "label"])
        target_column = "label"

    elif format_key == "score_triplets":
        # score_triplets: 4列格式 {query, pos, neg, label:[pos_score, neg_score]}
        # 标准化列名: pos→positive, neg→negative
        dataset = _rename_columns(dataset, {
            "pos": "positive",
            "neg": "negative"
        })

        dataset = _ensure_columns_order(dataset, ["query", "positive", "negative", "label"])
        target_column = "label"

        # 如果使用 MarginMSELoss，将 [pos_score, neg_score] 转换为 margin
        if loss_name == MarginMSELoss.__name__:
            dataset = _convert_score_pairs_to_margin(dataset)

    elif format_key == "margin_multi_margins":
        # 标准化列名: anchor→query
        dataset = _rename_columns(dataset, {"anchor": "query"})

        # 将数组格式展开为多个三元组 (MarginMSELoss 只支持三元组)
        def expand_margins(row):
            query = row["query"]
            neg_list = row.get("neg", [])
            margins = row.get("margins", [])

            records = []
            pos = pos_list[0] if pos_list else ""
            eff = min(len(neg_list), len(margins))
            if len(neg_list) != len(margins):
                logger.warning(
                    "margin_multi_margins: 样本负例数量(%d)与margins数量(%d)不一致，按最小值 %d 对齐",
                    len(neg_list), len(margins), eff,
                )
            # 统一对齐：若配置了 max_negatives，或数据集已计算 allowed_k，则受其约束
            local_k = eff
            if allowed_k_for_margins is not None:
                local_k = min(local_k, allowed_k_for_margins)
            if local_k <= 0:
                return records
            for idx in range(local_k):
                neg = neg_list[idx]
                margin = margins[idx]
                records.append({
                    "query": query,
                    "positive": pos,
                    "negative": neg,
                    "label": margin
                })
            return records

        # 展开数据集
        # 先计算数据集级的 allowed_k（默认取每条样本可对齐数的最小值），支持 max_negatives 覆盖上限
        allowed_k_for_margins: Optional[int] = None
        # 统计每条的可对齐数 eff_i
        eff_list = []
        for row in dataset:
            pos_v = row.get("pos", [])
            pos_list = pos_v if isinstance(pos_v, list) else ([pos_v] if pos_v else [])
            neg_list = row.get("neg", [])
            margins = row.get("margins", [])
            eff_i = min(len(neg_list), len(margins))
            if eff_i > 0:
                eff_list.append(eff_i)
        if eff_list:
            allowed_k_for_margins = min(eff_list)
            if max_negatives is not None and max_negatives > 0:
                allowed_k_for_margins = min(allowed_k_for_margins, max_negatives)
            logger.info(f"margin_multi_margins: 统一对齐负例数 = {allowed_k_for_margins}")
        else:
            raise EmbeddingRecipeError("margin_multi_margins: 数据集中无可用的(neg, margins)成对项，无法展开")

        expanded_data = []
        for row in dataset:
            expanded_data.extend(expand_margins(row))

        from datasets import Dataset as HFDataset
        dataset = HFDataset.from_list(expanded_data)
        dataset = _ensure_columns_order(dataset, ["query", "positive", "negative", "label"])
        target_column = "label"

    elif format_key == "margin_multi_scores":
        # 标准化列名: anchor→query
        dataset = _rename_columns(dataset, {"anchor": "query"})

        # 将数组格式展开为多个三元组 (MarginMSELoss 只支持三元组)
        def expand_scores(row):
            query = row["query"]
            pos_v = row.get("pos", [])
            pos_list = pos_v if isinstance(pos_v, list) else ([pos_v] if pos_v else [])
            neg_list = row.get("neg", [])
            scores = row.get("scores", [])
            if (not isinstance(scores, list) or not scores) and isinstance(row.get("pos_scores"), list) and isinstance(row.get("neg_scores"), list):
                # 从 universal 的 pos_scores/neg_scores 组装为统一的 scores
                ps = row.get("pos_scores") or []
                ns = row.get("neg_scores") or []
                head = ps[0] if len(ps) > 0 else 0.9
                scores = [head] + list(ns)

            records = []
            pos = pos_list[0] if pos_list else ""
            if not isinstance(scores, list) or not scores:
                logger.warning("margin_multi_scores: 样本缺少scores或为空，跳过该样本")
                return records
            pos_score = scores[0]

            # 对齐数量：neg_count 与 (scores-1) 的最小值
            eff = min(len(neg_list), max(0, len(scores) - 1))
            if eff == 0:
                logger.warning(
                    "margin_multi_scores: 样本负例数量(%d)与scores数量(%d)不匹配，无法展开，跳过", len(neg_list), len(scores)
                )
                return records
            if len(neg_list) != len(scores) - 1:
                logger.warning(
                    "margin_multi_scores: 样本负例数量(%d)与scores数量(%d)不一致，按最小值 %d 对齐",
                    len(neg_list), len(scores) - 1, eff,
                )

            # 数据集级统一对齐：若配置了 max_negatives，或数据集已计算 allowed_k，则受其约束
            local_k = eff
            if allowed_k_for_scores is not None:
                local_k = min(local_k, allowed_k_for_scores)
            if local_k <= 0:
                return records

            # scores = [pos_score, neg1_score, neg2_score, ...]，仅展开前 local_k 个
            for idx in range(local_k):
                neg = neg_list[idx]
                neg_score = scores[idx + 1]
                margin = float(pos_score) - float(neg_score)
                records.append({
                    "query": query,
                    "positive": pos,
                    "negative": neg,
                    "label": margin
                })
            return records

        # 展开数据集
        # 先计算数据集级的 allowed_k（默认取每条样本可对齐数的最小值），支持 max_negatives 覆盖上限
        allowed_k_for_scores: Optional[int] = None
        eff_list = []
        for row in dataset:
            neg_list = row.get("neg", [])
            scores = row.get("scores", [])
            if isinstance(scores, list) and len(scores) >= 2:
                eff_i = min(len(neg_list), len(scores) - 1)
                if eff_i > 0:
                    eff_list.append(eff_i)
        if eff_list:
            allowed_k_for_scores = min(eff_list)
            if max_negatives is not None and max_negatives > 0:
                allowed_k_for_scores = min(allowed_k_for_scores, max_negatives)
            logger.info(f"margin_multi_scores: 统一对齐负例数 = {allowed_k_for_scores}")
        else:
            raise EmbeddingRecipeError("margin_multi_scores: 数据集中无可用的(neg, scores)成对项，无法展开")

        expanded_data = []
        for row in dataset:
            expanded_data.extend(expand_scores(row))

        from datasets import Dataset as HFDataset
        dataset = HFDataset.from_list(expanded_data)
        dataset = _ensure_columns_order(dataset, ["query", "positive", "negative", "label"])
        target_column = "label"

    elif format_key == "dynamic_negatives":
        # 转换为 anchor, positive, negative_0, negative_1, ... 格式
        # 支持两种模式：
        # 1. 标准 MNR：要求所有样本负例数量一致（必须指定 max_negatives）
        # 2. 动态负例：支持不同数量的负例（使用 DynamicExplicitNegativesRankingLoss）

        # 判断是否使用标准 MNR
        use_standard_mnr = (loss_name == MultipleNegativesRankingLoss.__name__)

        if use_standard_mnr:
            # === 标准 MNR 模式 ===
            # 要求：所有样本的负例数量必须一致

            if max_negatives is None:
                # 未指定 max_negatives：尝试自动对齐
                # 策略：取第一个样本的负例数量，检查是否所有样本都对齐
                first_sample = dataset[0]
                first_neg_count = len(first_sample.get("neg", []))

                # 检查所有样本是否对齐
                all_aligned = all(len(s.get("neg", [])) == first_neg_count for s in dataset)

                if all_aligned:
                    # 所有样本负例数量一致，使用全部负例
                    max_negatives = first_neg_count
                    logger.info(f"标准 MNR 模式: 检测到所有样本负例数量一致 ({first_neg_count} 个)，使用全部负例")
                else:
                    # 负例数量不一致，使用最小值（保留所有样本，最大化负例利用）
                    min_neg_count = min(len(s.get("neg", [])) for s in dataset)
                    max_negatives = min_neg_count
                    logger.info(
                        f"标准 MNR 模式: 检测到样本负例数量不一致，"
                        f"自动使用最小负例数量 {min_neg_count} 个（保留所有样本）。"
                        f"如需使用更多负例，请明确指定 max_negatives 参数。"
                    )

            # 过滤掉负例数量不足的样本
            before_count = len(dataset)
            dataset = dataset.filter(lambda s: len(s.get("neg", [])) >= max_negatives)
            after_count = len(dataset)

            if after_count == 0:
                raise EmbeddingRecipeError(
                    f"过滤后没有样本满足 max_negatives={max_negatives} 的要求。"
                    f"请检查数据或降低 max_negatives 值。"
                )

            logger.info(f"标准 MNR 模式: 固定负例数量 = {max_negatives}")
            if before_count > after_count:
                logger.warning(
                    f"过滤样本: {before_count} -> {after_count} "
                    f"(丢弃 {before_count - after_count} 个负例不足的样本，占比 {(before_count - after_count) / before_count * 100:.1f}%)"
                )

            # 转换为固定负例数量
            def convert_for_standard_mnr(sample):
                pos_list = sample.get("pos", [])
                neg_list = sample.get("neg", [])

                result = {
                    "anchor": sample.get("query", ""),
                    "positive": pos_list[0] if pos_list else "",
                }

                # 添加固定数量的负例（已经过滤，保证 len(neg_list) >= max_negatives）
                for idx in range(max_negatives):
                    result[f"negative_{idx}"] = neg_list[idx] if neg_list[idx] else ""

                return result

            dataset = dataset.map(convert_for_standard_mnr, remove_columns=dataset.column_names)

        else:
            # === 动态负例模式 ===
            # 支持每个样本拥有不同数量的负例

            # 1. 统计最大负例数量
            max_negs = 0
            for sample in dataset:
                neg_list = sample.get("neg", [])
                max_negs = max(max_negs, len(neg_list))

            # 如果指定了 max_negatives，则限制最大负例数量
            if max_negatives is not None and max_negatives > 0:
                logger.info(f"限制最大负例数量: {max_negs} -> {max_negatives}")
                max_negs = min(max_negs, max_negatives)

            logger.info(f"动态负例格式: 最大负例数量 = {max_negs}")

            # 2. 转换函数：所有样本填充到相同的负例数量
            def convert_dynamic_negatives(sample):
                pos_list = sample.get("pos", [])
                neg_list = sample.get("neg", [])

                result = {
                    "anchor": sample.get("query", ""),
                    "positive": pos_list[0] if pos_list else "",
                }

                # 添加负例（最多 max_negs 个），不足的用空字符串填充
                for idx in range(max_negs):
                    if idx < len(neg_list):
                        result[f"negative_{idx}"] = neg_list[idx] if neg_list[idx] else ""
                    else:
                        result[f"negative_{idx}"] = ""  # 填充空字符串，损失函数会自动忽略

                return result

            dataset = dataset.map(convert_dynamic_negatives, remove_columns=dataset.column_names)

    else:  # pragma: no cover - 上述逻辑已覆盖所有已知格式
        raise EmbeddingRecipeError(f"尚未实现的数据格式: {format_key}")

    loss_kwargs: Dict[str, Any] = {}
    if loss_name == DISTILL_KL_LOSS_NAME:
        if DistillKLDivLoss is None:
            raise EmbeddingRecipeError(
                "当前环境的 sentence-transformers 未提供 DistillKLDivLoss，请升级依赖或选择其他损失函数。"
            )
        loss_kwargs["temperature"] = loss_temperature or 1.0
    if loss_name == GISTEmbedLoss.__name__:
        if not guide_model_name:
            raise EmbeddingRecipeError("使用GISTEmbedLoss需要提供 guide_model_path 配置")
        loss_kwargs["guide_model_path"] = guide_model_name

    spec = EmbeddingRecipeSpec(
        format_key=format_key,
        loss_name=loss_name,
        target_column=target_column,
        dataset_name=dataset_name,
        split_name=split_name,
        loss_kwargs=loss_kwargs,
        label_normalization=applied_label_norm,
    )

    return dataset, spec


def build_embedding_loss(model: SentenceTransformer, spec: EmbeddingRecipeSpec) -> Any:
    """根据规格实例化具体的 SentenceTransformers loss。"""

    if spec.loss_name == MultipleNegativesRankingLoss.__name__:
        return MultipleNegativesRankingLoss(model)
    if spec.loss_name == MarginMSELoss.__name__:
        return MarginMSELoss(model)
    if spec.loss_name == DISTILL_KL_LOSS_NAME:
        if DistillKLDivLoss is None:
            raise EmbeddingRecipeError(
                "DistillKLDivLoss 在当前 sentence-transformers 版本中不可用，请升级依赖或切换其他损失函数。"
            )
        temperature = spec.loss_kwargs.get("temperature", 1.0)
        return DistillKLDivLoss(model, temperature=temperature)
    if spec.loss_name == CosineSimilarityLoss.__name__:
        return CosineSimilarityLoss(model)
    if spec.loss_name == CoSENTLoss.__name__:
        return CoSENTLoss(model)
    if spec.loss_name == AnglELoss.__name__:
        return AnglELoss(model)
    if spec.loss_name == GISTEmbedLoss.__name__:
        guide_name = spec.loss_kwargs.get("guide_model_path")
        guide_model = SentenceTransformer(guide_name)
        return GISTEmbedLoss(model, guide_model)
    if spec.loss_name == DynamicExplicitNegativesRankingLoss.__name__:
        # 从环境变量或配置获取参数
        import os

        scale = float(os.getenv("DYNAMIC_NEG_SCALE", "20.0"))
        normalize_by_logc = os.getenv("NORMALIZE_LOSS_BY_LOGC", "False").lower() == "true"
        ignore_empty = os.getenv("IGNORE_EMPTY_NEGATIVES", "True").lower() == "true"
        empty_threshold = float(os.getenv("EMPTY_THRESHOLD", "1e-6"))

        return DynamicExplicitNegativesRankingLoss(
            model=model,
            scale=scale,
            normalize_by_logc=normalize_by_logc,
            ignore_empty=ignore_empty,
            empty_threshold=empty_threshold,
        )

    raise EmbeddingRecipeError(f"暂不支持的损失函数: {spec.loss_name}")
