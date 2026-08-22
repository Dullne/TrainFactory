"""
正负例语句提取 / 确认的 LLM 提示词模板

供 hard_neg_extraction.py 使用。
"""

# ============================================================
# 负例 Statement 提取
# ============================================================
NEG_EXTRACTION_SYSTEM_PROMPT = """你是一个专业的文本分析助手。请从给定的文本片段中识别出与查询相关但可能误导模型的语句。

任务：
1. 分析chunk中哪些语句看起来与query相关，但实际上不能支持expected_output
2. 特别注意识别以下类型的迷惑性内容：
   - 相似但不同的实体（如相近的名称、拼写、编号等）
   - 同类但不同的成员
   - 相近但不同的时间、数值、条件等
3. 提取这些误导性语句
4. 对每个语句打分（0-10分，分数越高表示越容易误导）

输出JSON格式：
{
    "negatives": ["语句1", "语句2", ...],
    "scores": [8.5, 7.0, ...]
}"""

NEG_EXTRACTION_USER_TEMPLATE = """查询(query): {query}

期望输出(expected_output): {answer}

文本片段(chunk): {chunk_text}

最大提取数量: {max_negatives}

请分析并提取误导性语句。"""


# ============================================================
# 正例 Statement 提取
# ============================================================
POS_EXTRACTION_SYSTEM_PROMPT = """你是一个专业的文本分析助手。请判断chunk与query的相关性，并从相关chunk中提取回答query的完整陈述句。

任务流程：
1. **首先判断chunk是否与query相关**（能提供回答query所需的信息）
2. 如果相关，识别query中的关键实体和条件
3. 从chunk中找到回答query的信息
4. 生成包含query关键点+答案的完整陈述句
5. 对每个语句打分（0-10分）

生成语句要求：
- 语句应覆盖query的核心信息点，而非只有答案片段
- 确保语句完整、自包含，不看query也能理解其含义
- **只能使用chunk中存在的信息**，不要编造不存在的内容
- 可以适当改写，但关键实体和信息点要保留

示例：
- query: "X在Y条件下的Z属性是什么？"
- 关键点: X(实体), Y(条件), Z(属性)
- chunk包含: "属性值A"
- 错误: "属性值A"（只有答案，缺少关键上下文）
- 正确: "X在Y条件下的Z属性是属性值A"（覆盖query关键点+填充答案）

输出JSON格式：
{
    "chunk_relevant": true/false,
    "positives": ["完整陈述句1", "完整陈述句2", ...],
    "scores": [9.5, 8.0, ...]
}

注意：如果chunk_relevant为false，positives应为空数组[]"""

POS_EXTRACTION_USER_TEMPLATE = """查询(query): {query}

期望输出(expected_output): {answer}

文本片段(chunk): {chunk_text}

最大提取数量: {max_positives}

请先判断chunk相关性，再提取支持答案的完整陈述句。"""


# ============================================================
# 正例 Statement 确认
# ============================================================
POS_CONFIRM_SYSTEM_PROMPT = """你是一个专业的文本分析助手。请判断chunk和提取的语句是否有效。

任务：
1. 首先判断chunk是否与query相关（能提供回答query的信息）
2. 对于每个提取的语句，判断：
   - 语句是否能回答query
   - 语句内容是否确实来源于chunk（非幻觉）

输出JSON格式：
{
    "chunk_relevant": true/false,
    "confirmations": [true, false, true, ...]
}

判定规则：
- chunk_relevant=false时，所有语句都应为false
- chunk_relevant=true时，仅当语句回答query且内容来源于chunk时为true
- 语句如果包含chunk中不存在的信息（幻觉），应为false"""

POS_CONFIRM_USER_TEMPLATE = """查询(query): {query}

期望输出(expected_output): {answer}

原始文本片段(chunk): {chunk_text}

待确认的语句：
{statements_text}

请判断chunk相关性及每个语句的有效性。"""


# ============================================================
# 负例 Statement 确认（基于 Rerank Delta）
# ============================================================
NEG_CONFIRM_SYSTEM_PROMPT = """你是一个专业的难负例确认助手。

任务：从候选语句中选择真正的难负例（看起来相关但实际误导的语句）。

参考信息：
- delta = sim_full - sim_reduced: 移除该语句后相似度下降幅度
- delta越大说明该语句对相似度贡献越大（可能越误导）
- 但最终仍需基于语义判断是否真的不相关

输出JSON格式：
{
    "selected": ["确认的难负例1", "确认的难负例2", ...]
}"""

NEG_CONFIRM_USER_TEMPLATE = """查询(query): {query}

期望输出(expected_output): {answer}

原始文本片段(chunk): {chunk_text}

候选语句及其delta信息:
{candidates_info}

Delta阈值参考: {threshold}
最大选择数量: {max_negatives}

请选择真正的难负例。"""


# ============================================================
# Answer 重述（将 query + answer 合并为完整陈述句）
# ============================================================
ANSWER_REWRITE_SYSTEM_PROMPT = """你是一个专业的文本重述助手。请将给定的query和answer合并为一个完整、自包含的陈述句。

要求：
1. 陈述句必须包含query中的关键实体、条件等上下文信息
2. 陈述句必须包含answer中的答案信息
3. 陈述句应该完整、自然，不看原query也能理解其含义
4. 保持专业术语的准确性
5. 不要添加原文中没有的信息
6. **语义必须与原文完全一致，只是换种表达方式**

示例：
- query: "X在Y条件下的Z属性是什么？"
- answer: "属性值A"
- 陈述句: "X在Y条件下的Z属性是属性值A"

输出要求：
- 只输出生成的陈述句，不要输出解释或额外说明"""

ANSWER_REWRITE_USER_TEMPLATE = """查询(query): {query}

答案(answer): {answer}

请将query和answer合并改写成完整的陈述句，只生成1个版本。"""


# ============================================================
# 证据去除（从正例 chunk 中移除证据 → 硬负例变体）
# ============================================================
EVIDENCE_REMOVAL_SYSTEM_PROMPT = """你是一个负责生成"去除证据"变体的文本助手。

任务流程：
1. **首先判断 chunk 是否为正例**：chunk 是否包含能够回答 query 的关键证据？
   - 如果 chunk 不能回答 query（非正例），直接返回 {"is_positive": false, "removed_chunk": "", "note": "chunk不是正例，无法回答query"}
2. **如果是正例**，生成去除证据的变体：
   - 保留原始 chunk 的上下文信息、叙述风格和结构
   - 删除或改写直接回答 query 的关键句/数字/实体，使 chunk 不再提供足够证据来支持 answer
   - 不要引入新的事实，也不要增加与 query 无关的信息
   - 如果 chunk 中没有可以删除的证据（删除后会导致文本语义大幅空缺），返回空串

输出 JSON 格式：{"is_positive": true/false, "removed_chunk": "...", "note": "..."}
- is_positive: chunk 是否能回答 query
- removed_chunk: 去除证据后的文本（非正例时为空串）
- note: 说明，如 "removed evidence" / "no evidence to remove" / "chunk不是正例"等"""

EVIDENCE_REMOVAL_USER_TEMPLATE = """查询(query): {query}

答案(answer): {answer}

文本片段(chunk): {chunk_text}

请按规则生成去除证据后的变体，只输出 JSON。"""


# ============================================================
# 证据裁剪（裁剪正例 chunk 中无关内容 → 更干净正例）
# ============================================================
EVIDENCE_PRUNING_SYSTEM_PROMPT = """你是一个专业的文本裁剪助手。

任务流程：
1. **首先判断 chunk 是否为正例**：chunk 是否包含能够回答 query 的关键证据？
   - 如果 chunk 不能回答 query（非正例），直接返回 {"is_positive": false, "trimmed_chunk": "", "note": "chunk不是正例，无法回答query"}
2. **如果是正例**，进行裁剪：
   - 保留 chunk 中直接支持 answer 的句子、数据、关键实体，不要移除核心证据
   - 删去少量与 query 无关的背景描述、附带解释或重复句子，保持整体语义一致
   - 尽量只删除少量无关内容，不要把 chunk 全部重写/精简掉
   - 输出应保持可读性和原文风格，优先保留原句，仅删减边缘段落
   - 如果 chunk 中没有可以去除的部分，返回空串

输出 JSON 格式：{"is_positive": true/false, "trimmed_chunk": "...", "note": "..."}
- is_positive: chunk 是否能回答 query
- trimmed_chunk: 裁剪后的文本（非正例时为空串）
- note: 说明，如 "trimmed evidence" / "no change" / "chunk不是正例"等"""

EVIDENCE_PRUNING_USER_TEMPLATE = """查询(query): {query}

答案(answer): {answer}

文本片段(chunk): {chunk_text}

请根据规则裁剪无关内容，仅输出 JSON。"""
