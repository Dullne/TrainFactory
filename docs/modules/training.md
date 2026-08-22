# 训练模块

## 训练类型矩阵

训练类型是一个**二维矩阵**，由模型类型和训练方法组成：

```
                         Training Method
                 ┌─────────┬─────────┬─────────┬───────────┐
                 │   SFT   │   DPO   │  GRPO   │ REINFORCE │
    ┌────────────┼─────────┼─────────┼─────────┼───────────┤
    │ Embedding  │    ✅   │    -    │    -    │     -     │
    ├────────────┼─────────┼─────────┼─────────┼───────────┤
M   │ Reranker   │    ✅   │    -    │    -    │     -     │
o   │(CrossEnc)  │         │         │         │           │
d   ├────────────┼─────────┼─────────┼─────────┼───────────┤
e   │ Decoder    │    ✅   │    ✅   │    ✅   │     -     │
l   │ Reranker   │         │         │         │           │
    ├────────────┼─────────┼─────────┼─────────┼───────────┤
T   │ LLM        │    ✅   │    ✅   │    -    │     -     │
y   └────────────┴─────────┴─────────┴─────────┴───────────┘
p
e   ✅ = 已实现    📋 = 部分能力待补齐    - = 不支持
```

补充说明：
- `decoder_reranker` 已支持 `sft / dpo / grpo / dapo / dr_grpo`
- `llm` 已支持 `sft / dpo / orpo`

### 模型类型说明

| 模型类型 | 架构 | 说明 | 示例模型 |
|----------|------|------|----------|
| `embedding` | Encoder | 句子向量模型 | bge-m3, e5-large |
| `reranker` | Encoder | CrossEncoder 重排序 | bge-reranker-v2-m3 |
| `decoder_reranker` | Decoder | Decoder-only 重排序 | Qwen3-Reranker |
| `llm` | Decoder | 大语言模型 | Qwen2.5, Llama3 |

### 训练方法说明

| 训练方法 | 说明 | 适用场景 |
|----------|------|----------|
| `sft` | 监督微调 (Supervised Fine-Tuning) | 通用微调 |
| `cpt` | 增量预训练 (Continual Pre-Training) | 领域适应 |
| `dpo` | 直接偏好优化 (Direct Preference Optimization) | 对齐训练 |
| `grpo` | 群体相对策略优化 (Group Relative Policy Optimization) | RL 训练 |
| `kto` | Kahneman-Tversky 优化 | 对齐训练 |
| `orpo` | Odds Ratio 偏好优化 | 对齐训练 |
| `simpo` | 简单偏好优化 | 对齐训练 |
| `ppo` | 近端策略优化 (Proximal Policy Optimization) | RL 训练 |
| `reinforce` | REINFORCE 算法 | RL 训练 |
| `two_stage` | 两阶段训练 | SFT → RL |

### 微调方法说明

| 微调方法 | 说明 | 内存占用 | 适用场景 |
|----------|------|----------|----------|
| `lora` | Low-Rank Adaptation | 低 | 大多数微调场景 |
| `qlora` | 量化 LoRA (4-bit/8-bit) | 极低 | 资源受限环境 |
| `full` | 全参数微调 | 高 | 小模型或充足资源 |
| `freeze` | 冻结部分层 | 中 | 迁移学习 |
| `adapter` | Adapter 层 (待实现) | 低 | 多任务学习 |
| `prefix` | Prefix Tuning (待实现) | 低 | 生成任务 |

## 数据集统计

`BaseTrainer._record_dataset_stats()` 在训练开始时将各 split 的实际样本数量写回 `training_params.dataset_configs[].num_rows`，前端详情页据此展示每个数据集的行数。

## DPO / ORPO 参数配置

DPO 和 ORPO 训练通过 `rl_config` 传递偏好优化相关参数。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `beta` | float | 0.1 | DPO/ORPO 的 beta 参数，控制 KL 散度惩罚强度。取值范围：0.01-10 |
| `rankings_direction` | string | `auto` | 排序方向，用于 `responses + rankings` 格式。可选值：`auto`、`higher_is_better`、`lower_is_better` |

### `beta`

`beta` 控制偏好优化中策略更新相对于参考分布的约束强度。值越大，KL 散度惩罚越强，训练会更保守；值越小，模型更容易根据偏好样本产生更大的更新。

调优建议：
- 默认从 `0.1` 开始
- 模型更新过于激进、生成风格漂移明显时，适当增大 `beta`
- 偏好信号学习不足、chosen / rejected 区分不明显时，适当减小 `beta`
- 建议在 `0.01-10` 范围内逐步调参，优先小步调整

### `rankings_direction`

`rankings_direction` 用于解释 `responses + rankings` 格式中的排序值语义：
- `higher_is_better`: 将 rankings 视为分数，数值越大表示偏好越强
- `lower_is_better`: 将 rankings 视为名次，数值越小表示排序越靠前
- `auto`: 自动推断排序方向

自动推断逻辑如下：
- 若 rankings 是不重复的连续整数序列，如 `[1, 2, 3]` 或 `[0, 1, 2]`，则视为 ordinal ranks，按 `lower_is_better` 处理
- 其他情况视为 scores，按 `higher_is_better` 处理

当数据来自人工排序结果、标注名次或比赛排名时，建议显式使用 `lower_is_better`；当数据来自奖励模型分数、质量分或置信度分值时，建议使用 `higher_is_better`。当同一数据源的排序约定稳定时，优先显式配置，避免依赖自动推断。

## 训练详情页

前端训练详情页 (`TrainingDetail.tsx`) 展示：

- **基础模型**: 自动解析 `base_model_path` 为注册模型名称，显示为可点击链接跳转到模型详情弹窗 (`/models?detail=model_id`)
- **训练产出模型**: 通过 `trained_model_registry_id` 或 `final_model_path` 匹配注册模型，显示为可点击链接
- **数据集卡片**: 独立展示训练数据集配置，按 split 分组（训练集/验证集/测试集），显示路径、实际样本数和采样上限

路径匹配使用后缀比对，兼容容器内路径 (`/app/models/`) 与宿主机路径的差异。

## 类层次结构

```
BaseTrainer (抽象基类)
│
├── EncoderBaseTrainer (Encoder 模型基类) - 仅支持 SFT
│   ├── EmbeddingTrainer        ✅ 已实现
│   └── RerankerTrainer         ✅ 已实现 (CrossEncoder)
│
└── Decoder Trainers
    ├── LLMTrainer              ✅ 已实现 (SFT / DPO / ORPO)
    └── DecoderRerankerTrainer  ✅ 已实现 (SFT / DPO / GRPO / DAPO / DR_GRPO)
```

**注意：** RL 训练（DPO、GRPO、KTO 等）仅适用于 Decoder 架构模型，因此统一放在 `decoder/` 目录下。

## 训练器工厂

通过 `TrainerFactory` 根据 `(model_type, training_method)` 组合选择合适的训练器：

```python
from train_factory.trainers import TrainerFactory

# 创建 Embedding SFT 训练器
trainer = TrainerFactory.create("embedding", "sft", **config)

# 创建 Decoder Reranker GRPO 训练器
trainer = TrainerFactory.create("decoder_reranker", "grpo", **config)

# 创建 LLM DPO 训练器
trainer = TrainerFactory.create("llm", "dpo", **config)
```

## Tuner 微调器

通过 `TunerRegistry` 管理和使用微调方法：

```python
from train_factory.tuners import TunerRegistry, LoRATuner

# 方式 1: 通过注册表创建
tuner = TunerRegistry.create("lora", config={
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
})
model = tuner.prepare_model(model)

# 方式 2: 直接使用类
tuner = LoRATuner({"r": 8, "lora_alpha": 16})
model = tuner.prepare_model(model)

# 查看可训练参数
tuner.print_trainable_parameters(model)
# Output: Trainable parameters: 1,234,567 / 123,456,789 (1.00%)
```

### 注册自定义 Tuner

```python
from train_factory.tuners import TunerRegistry, BaseTuner

@TunerRegistry.register("my_tuner")
class MyCustomTuner(BaseTuner):
    def prepare_model(self, model):
        # 自定义逻辑
        return model

    def get_trainable_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @property
    def tuner_type(self):
        return "my_tuner"
```

---

## 待实现功能

### Phase 1: Decoder 训练器补全 (统一在 decoder/ 目录)
- [ ] `trainers/decoder/decoder_base_trainer.py` - Decoder 基类
- [x] `trainers/decoder/llm_trainer.py` - LLM 训练器 (SFT/DPO/ORPO)
- [ ] `trainers/decoder/cpt_trainer.py` - 增量预训练
- [ ] `trainers/decoder/grpo_trainer.py` - 通用 LLM GRPO 训练 (基于 TRL GRPOTrainer)
- [ ] `trainers/decoder/kto_trainer.py` - KTO 训练 (基于 TRL KTOTrainer)
- [ ] `trainers/decoder/ppo_trainer.py` - PPO 训练 (基于 TRL PPOTrainer)
- [x] `trainers/decoder/decoder_reranker_trainer.py` - Decoder Reranker (Qwen3-Reranker, SFT/DPO/GRPO/DAPO/DR_GRPO)

### Phase 2: 损失函数集成
- [ ] `losses/ranking/` - 排序损失（LambdaLoss, ListMLE 等）
- [ ] `losses/rl/` - RL 损失函数
- [ ] `losses/similarity/` - 相似度损失

### Phase 3: 数据预处理器完善
- [ ] `data/preprocessors/dpo_preprocessor.py` - 完整实现
- [ ] `data/preprocessors/grpo_preprocessor.py` - GRPO 数据处理

### Phase 4: 评估模块增强
- [x] `evaluation/evaluator.py` - 通用评估器 (UnifiedEvaluator)
- [x] `evaluation/metric_registry.py` - 指标注册表 (MetricRegistry)
- [x] `evaluation/result.py` - 结果处理 (EvaluationResultProcessor)
- [x] `evaluation/evaluation_runner.py` - MTEB 评估任务执行器 (多模型并行、断点续评)
- [ ] `evaluation/llm_evaluator.py` - LLM 专用评估扩展
- [ ] `evaluation/embedding_evaluator.py` - Embedding 专用评估扩展

## 相关文档

- 数据库表: [training_tasks](../database/training_tasks.md)
- API 文档: [训练 API](../api/training.md)
