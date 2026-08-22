# 评估模块

## 概述

评估模块支持两种评估框架：

| 框架 | 说明 | 适用场景 |
|------|------|----------|
| MTEB | 有标签数据评估，多模型并行 | Embedding/Reranker 基准测试 |
| DeepEval | LLM-as-Judge 评估，多模型组 | 检索/生成质量评估 |

## 核心组件

```
train_factory/evaluation/           # MTEB 评估
├── evaluation_runner.py            # 评估任务执行器（多模型并行、断点续评）
├── evaluator.py                    # UnifiedEvaluator 统一评估器
├── metric_registry.py              # MetricRegistry 指标注册表
└── result.py                       # EvaluationResult 结果处理

train_factory/deep_evaluation/      # 深度评估（DeepEval）
├── deep_evaluation_runner.py       # 深度评估任务执行器
├── evaluator.py                    # LLM-as-Judge 评估器
└── metrics/                        # DeepEval 指标
```

## 相关文档

- 数据库表: [evaluation_tasks](../database/evaluation_tasks.md)
- API 文档: [评估 API](../api/evaluations.md)
