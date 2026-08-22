"""统一评估模块

支持embedding和reranker两种模型类型的评估，
根据模型类型自动选择合适的评估器。
"""
import logging
import os
from typing import Optional, Dict, Any
from datasets import Dataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, BinaryClassificationEvaluator, SimilarityFunction, SequentialEvaluator
try:
    from sentence_transformers.cross_encoder.evaluation import (
        CrossEncoderClassificationEvaluator,
        CrossEncoderCorrelationEvaluator,
    )
except ImportError:
    from sentence_transformers.cross_encoder.evaluation import (
        CEBinaryClassificationEvaluator as CrossEncoderClassificationEvaluator,
        CECorrelationEvaluator as CrossEncoderCorrelationEvaluator,
    )

logger = logging.getLogger(__name__)

class UnifiedEvaluator:
    """
    统一的评估器工厂类
    
    根据模型类型（embedding或reranker）自动创建相应的评估器，
    简化了不同模型类型的评估流程。
    
    Attributes:
        model_type (str): 模型类型，'embedding' 或 'reranker'
    """
    
    def __init__(self, model_type: Optional[str] = None):
        """
        初始化评估器
        
        Args:
            model_type: 'embedding' 或 'reranker'，如果为None则从MODEL_TYPE环境变量获取
        """
        if model_type is None:
            model_type = os.getenv("MODEL_TYPE", "embedding").lower()
            
        if model_type not in ['embedding', 'reranker']:
            raise ValueError(f"不支持的模型类型: {model_type}. 只支持 'embedding' 或 'reranker'")
        self.model_type = model_type
        logger.info(f"初始化统一评估器，模型类型: {self.model_type}")
    
    def create_evaluator(self, dataset: Dataset, target_column: str, name: str):
        """
        根据模型类型创建相应的评估器
        
        Args:
            dataset: 评估数据集
            target_column: 目标列名 ('score' 或 'label')
            name: 评估器名称
            
        Returns:
            评估器实例
        """
        # 应用与训练数据集相同的列过滤逻辑
        filtered_dataset = self._filter_dataset_columns(dataset, target_column, name)
        
        if self.model_type == "embedding":
            return self._create_embedding_evaluator(filtered_dataset, target_column, name)
        else:  # reranker
            return self._create_reranker_evaluator(filtered_dataset, target_column, name)
    
    def _create_embedding_evaluator(self, dataset: Dataset, target_column: str, name: str):
        """
        创建embedding模型评估器
        
        根据标签类型自动选择评估器：
        - score字段（浮点数）→ EmbeddingSimilarityEvaluator（回归任务）
        - label字段（整数）→ BinaryClassificationEvaluator（分类任务）
        
        Args:
            dataset: 包含句子对和目标值的数据集
            target_column: 目标列名 ('score' 或 'label')
            name: 评估器名称
            
        Returns:
            对应的评估器实例
        """
        # 获取句子对数据（使用与reranker相同的灵活处理）
        if hasattr(dataset, 'column_names'):
            column_names = dataset.column_names
        elif isinstance(dataset, dict):
            # 处理字典类型的数据集
            column_names = list(dataset.keys()) if dataset else []
        else:
            logger.warning(f"无法获取数据集列名，类型: {type(dataset)}")
            column_names = []

        logger.info(f"数据集列名: {column_names}")
        
        # 提取句子对
        sentences1, sentences2 = self._get_sentence_columns(dataset, column_names, target_column)
        
        # 检查数据类型和标签值来决定评估器类型
        labels = list(dataset[target_column])
        unique_labels = set(labels)
        
        # 只有当所有标签都是 0 或 1 时才使用 BinaryClassificationEvaluator
        is_binary_classification = (
            len(unique_labels) <= 2 and 
            all(label in [0, 1] for label in unique_labels)
        )
        
        logger.info(f"数据集标签分析 - 列名: {target_column}, 唯一值: {sorted(unique_labels)}, 样本: {labels[:5]}")
        
        is_classification = is_binary_classification
        
        if is_classification:
            logger.info(f"使用BinaryClassificationEvaluator（检测到分类任务，列名: {target_column}）")
            return BinaryClassificationEvaluator(
                sentences1=sentences1,
                sentences2=sentences2,
                labels=list(dataset[target_column]),
                name=name,
            )
        else:
            logger.info(f"使用EmbeddingSimilarityEvaluator（检测到回归任务，列名: {target_column}）")
            return EmbeddingSimilarityEvaluator(
                sentences1=sentences1,
                sentences2=sentences2,
                scores=list(dataset[target_column]),
                main_similarity=SimilarityFunction.COSINE,
                name=name,
            )
    
    def _create_reranker_evaluator(self, dataset: Dataset, target_column: str, name: str):
        """
        创建reranker模型评估器
        
        根据标签类型自动选择评估器：
        - label字段（整数）→ CrossEncoderClassificationEvaluator（分类任务）
        - score字段（浮点数）→ CrossEncoderCorrelationEvaluator（回归任务）
        
        Args:
            dataset: 包含句子对和目标值的数据集
            target_column: 目标列名 ('score' 或 'label')
            name: 评估器名称
            
        Returns:
            对应的CrossEncoder评估器实例
        """
        # 检查数据集是否为None
        if dataset is None:
            logger.error("数据集为None，无法创建评估器")
            return None
        
        # 自动识别句子对列名
        if hasattr(dataset, 'column_names'):
            column_names = dataset.column_names
        elif isinstance(dataset, dict):
            # 处理字典类型的数据集
            column_names = list(dataset.keys()) if dataset else []
        else:
            logger.warning(f"无法获取数据集列名，类型: {type(dataset)}")
            column_names = []

        logger.info(f"数据集列名: {column_names}")
        
        # 获取句子对数据
        sentence_pairs = self._get_sentence_pairs(dataset, column_names, target_column)
        
        # 根据标签类型选择评估器
        labels = list(dataset[target_column])
        unique_labels = set(labels)
        
        # 只有当所有标签都是 0 或 1 时才使用分类评估器
        is_classification = (
            len(unique_labels) <= 2 and 
            all(label in [0, 1] for label in unique_labels)
        )
        
        logger.info(f"数据集标签分析 - 列名: {target_column}, 唯一值: {sorted(unique_labels)}, 样本: {labels[:5]}")
        
        if is_classification:
            logger.info(f"使用CrossEncoderClassificationEvaluator（检测到分类任务，列名: {target_column}）")
            return CrossEncoderClassificationEvaluator(
                sentence_pairs=sentence_pairs,
                labels=dataset[target_column],
                name=name,
            )
        else:
            logger.info(f"使用CrossEncoderCorrelationEvaluator（检测到回归任务，列名: {target_column}）")
            return CrossEncoderCorrelationEvaluator(
                sentence_pairs=sentence_pairs,
                scores=dataset[target_column],
                name=name,
            )
    
    def _get_sentence_pairs(self, dataset: Dataset, column_names: list, target_column: str):
        """提取句子对数据"""
        # 尝试不同的句子对列名组合
        if "sentence1" in column_names and "sentence2" in column_names:
            sentence_pairs = list(zip(dataset["sentence1"], dataset["sentence2"]))
            logger.info("使用 sentence1/sentence2 列")
        elif "query" in column_names and "passage" in column_names:
            sentence_pairs = list(zip(dataset["query"], dataset["passage"]))
            logger.info("使用 query/passage 列")
        elif "anchor" in column_names and "positive" in column_names:
            sentence_pairs = list(zip(dataset["anchor"], dataset["positive"]))
            logger.info("使用 anchor/positive 列")
        elif "text1" in column_names and "text2" in column_names:
            sentence_pairs = list(zip(dataset["text1"], dataset["text2"]))
            logger.info("使用 text1/text2 列")
        else:
            # 如果找不到标准的句子对列，尝试使用前两个文本列
            text_columns = [col for col in column_names if col != target_column and isinstance(dataset[col][0], str)]
            if len(text_columns) >= 2:
                sentence_pairs = list(zip(dataset[text_columns[0]], dataset[text_columns[1]]))
                logger.info(f"使用 {text_columns[0]}/{text_columns[1]} 列")
            else:
                raise ValueError(f"无法在数据集中找到句子对列。可用列: {column_names}")
        
        return sentence_pairs
    
    def _get_sentence_columns(self, dataset: Dataset, column_names: list, target_column: str):
        """提取句子列数据（用于embedding评估器）"""
        # 尝试不同的句子对列名组合
        if "sentence1" in column_names and "sentence2" in column_names:
            sentences1 = list(dataset["sentence1"])
            sentences2 = list(dataset["sentence2"])
            logger.info("使用 sentence1/sentence2 列")
        elif "query" in column_names and "passage" in column_names:
            sentences1 = list(dataset["query"])
            sentences2 = list(dataset["passage"])
            logger.info("使用 query/passage 列")
        elif "anchor" in column_names and "positive" in column_names:
            sentences1 = list(dataset["anchor"])
            sentences2 = list(dataset["positive"])
            logger.info("使用 anchor/positive 列")
        elif "text1" in column_names and "text2" in column_names:
            sentences1 = list(dataset["text1"])
            sentences2 = list(dataset["text2"])
            logger.info("使用 text1/text2 列")
        else:
            # 如果找不到标准的句子对列，尝试使用前两个文本列
            text_columns = [col for col in column_names if col != target_column and isinstance(dataset[col][0], str)]
            if len(text_columns) >= 2:
                sentences1 = list(dataset[text_columns[0]])
                sentences2 = list(dataset[text_columns[1]])
                logger.info(f"使用 {text_columns[0]}/{text_columns[1]} 列")
            else:
                raise ValueError(f"无法在数据集中找到句子对列。可用列: {column_names}")
        
        return sentences1, sentences2
    
    def _filter_dataset_columns(self, dataset: Dataset, target_column: str, name: str) -> Dataset:
        """
        过滤数据集列，确保只有2个输入列 + 1个目标列

        这与训练数据集的列过滤逻辑保持一致，确保CrossEncoder评估器
        能够正确处理数据集格式。

        DEPRECATED：UnifiedEvaluator 当前无运行时调用方（各 trainer 自建
        evaluator）；≥4 列且标准句对列不在前两位时此处会静默选错句子对，
        采用前需先识别标准句对列名。

        Args:
            dataset: 原始数据集
            target_column: 目标列名
            name: 数据集名称（用于日志）
            
        Returns:
            过滤后的数据集
        """
        if hasattr(dataset, 'column_names'):
            column_names = dataset.column_names
        elif isinstance(dataset, dict):
            column_names = list(dataset.keys()) if dataset else []
        else:
            logger.warning(f"无法获取数据集列名，类型: {type(dataset)}，返回原数据集")
            return dataset

        if len(column_names) >= 3:
            # 确保只有3列：2个输入列 + 1个目标列
            input_columns = [col for col in column_names if col != target_column][:2]
            columns_to_keep = input_columns + [target_column]
            filtered_dataset = dataset.select_columns(columns_to_keep)
            logger.info(f"评估数据集 '{name}' 列过滤: {column_names} → {columns_to_keep}")
            return filtered_dataset
        else:
            logger.info(f"评估数据集 '{name}' 无需列过滤（已是 {len(column_names)} 列）")
            return dataset
    
    def evaluate_model(self, model, evaluator) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            model: 待评估的模型
            evaluator: 评估器
            
        Returns:
            评估结果字典
        """
        try:
            results = evaluator(model)
            logger.info(f"评估完成，结果: {results}")
            return results
        except Exception as e:
            logger.error(f"评估过程中出现错误: {e}")
            raise
    
    def create_evaluators_from_datasets(self, 
                                      eval_dataset: Optional[Dataset],
                                      test_dataset: Optional[Dataset],
                                      target_column: str,
                                      run_name: str) -> Dict[str, Any]:
        """
        从数据集创建所有必要的评估器
        
        Args:
            eval_dataset: 验证数据集
            test_dataset: 测试数据集  
            target_column: 目标列名
            run_name: 运行名称
            
        Returns:
            包含所有评估器的字典
        """
        evaluators = {}
        
        # 创建验证评估器
        if eval_dataset is not None and not (isinstance(eval_dataset, dict) and len(eval_dataset) == 0):
            if isinstance(eval_dataset, dict):
                # 多数据集：使用 create_multi_evaluator
                eval_evaluator = self.create_multi_evaluator(
                    eval_dataset, target_column, f"{run_name}-eval"
                )
                logger.info(f"创建多数据集验证评估器成功: {run_name}-eval")
            else:
                # 单数据集：使用 create_evaluator
                eval_evaluator = self.create_evaluator(
                    eval_dataset, target_column, f"{run_name}-eval"
                )
                logger.info(f"创建单数据集验证评估器成功: {run_name}-eval")
            
            # 验证用途的评估器统一使用 'dev' 键名
            evaluators['dev'] = eval_evaluator
        else:
            logger.info("跳过验证评估器创建（没有验证数据集）")
        
        # 创建测试评估器
        if test_dataset is not None and not (isinstance(test_dataset, dict) and len(test_dataset) == 0):
            if isinstance(test_dataset, dict):
                # 多数据集：使用 create_multi_evaluator
                evaluators['test'] = self.create_multi_evaluator(
                    test_dataset, target_column, f"{run_name}-test"
                )
                logger.info(f"创建多数据集测试评估器成功: {run_name}-test")
            else:
                # 单数据集：使用 create_evaluator
                evaluators['test'] = self.create_evaluator(
                    test_dataset, target_column, f"{run_name}-test"
                )
                logger.info(f"创建单数据集测试评估器成功: {run_name}-test")
        else:
            logger.info("跳过测试评估器创建（没有测试数据集）")
        
        return evaluators
    
    def create_multi_evaluator(self,
                              eval_datasets: Dict[str, Dataset],
                              target_column: str,
                              run_name: str,
                              data_source_mapping: Optional[Dict[str, str]] = None) -> Optional[Any]:
        """
        为多个数据集创建 SequentialEvaluator

        Args:
            eval_datasets: 多个验证数据集的字典
            target_column: 目标列名
            run_name: 运行名称
            data_source_mapping: dataset_name到source_id的映射（可选）

        Returns:
            SequentialEvaluator 实例，如果没有有效数据集则返回 None
        """
        if not eval_datasets:
            logger.info("没有验证数据集，跳过多评估器创建")
            return None
        
        evaluators = []
        for dataset_name, dataset in eval_datasets.items():
            if dataset is None:
                logger.warning(f"数据集 {dataset_name} 为空，跳过")
                continue
            
            # 为每个数据集单独确定目标列名
            dataset_target_column = self._get_dataset_target_column(dataset)
            logger.info(f"数据集 {dataset_name} 使用目标列: {dataset_target_column}")

            # 使用source_id作为evaluator name（如果提供了映射）
            if data_source_mapping and dataset_name in data_source_mapping:
                evaluator_name = data_source_mapping[dataset_name]
                logger.info(f"使用source_id作为evaluator name: {dataset_name} -> {evaluator_name}")
            else:
                evaluator_name = f"{run_name}-{dataset_name}"
                logger.info(f"使用默认名称作为evaluator name: {evaluator_name}")

            evaluator = self.create_evaluator(
                dataset, dataset_target_column, evaluator_name
            )
            
            if evaluator is not None:
                evaluators.append(evaluator)
                logger.info(f"为数据集 {dataset_name} 创建评估器成功")
        
        if not evaluators:
            logger.info("没有有效的评估器，返回None")
            return None
        
        if len(evaluators) == 1:
            logger.info("只有一个评估器，直接返回单个评估器")
            return evaluators[0]
        
        # 创建 SequentialEvaluator
        sequential_evaluator = SequentialEvaluator(evaluators)
        # 启用前缀功能，这样每个metric会加上evaluator name（即source_id）作为前缀
        sequential_evaluator.prefix_name_to_metrics = True
        logger.info(f"创建 SequentialEvaluator 成功，包含 {len(evaluators)} 个评估器，已启用metric前缀")
        return sequential_evaluator
    
    def _get_dataset_target_column(self, dataset: Dataset) -> str:
        """
        为单个数据集确定目标列名
        
        Args:
            dataset: 数据集
            
        Returns:
            目标列名
        """
        column_names = dataset.column_names
        
        # 三列格式：直接使用第三列作为目标列
        if len(column_names) == 3:
            target_col = column_names[2]
            logger.info(f"🎯 三列格式，使用第三列作为目标列: '{target_col}'")
            return target_col
        
        # 优先使用标准列名
        if "score" in column_names:
            return "score"
        elif "label" in column_names:
            return "label"
        elif "similarity_score" in column_names:
            return "similarity_score"
        else:
            # 如果没有标准列名，使用最后一列
            target_col = column_names[-1]
            logger.warning(f"未找到标准目标列名，使用最后一列: '{target_col}'")
            return target_col

def create_evaluator(model_type: str, dataset: Dataset, target_column: str, name: str):
    """
    创建评估器的便捷函数
    
    这是一个独立的便捷函数，用于快速创建单个评估器。
    对于批量创建多个评估器，建议使用UnifiedEvaluator类。
    
    Args:
        model_type (str): 模型类型，'embedding' 或 'reranker'
        dataset (Dataset): 评估数据集
        target_column (str): 目标列名
        name (str): 评估器名称
        
    Returns:
        评估器实例（EmbeddingSimilarityEvaluator 或 CrossEncoderCorrelationEvaluator）
        
    Example:
        evaluator = create_evaluator('embedding', eval_dataset, 'score', 'dev-eval')
    """
    evaluator_factory = UnifiedEvaluator(model_type)
    return evaluator_factory.create_evaluator(dataset, target_column, name)
