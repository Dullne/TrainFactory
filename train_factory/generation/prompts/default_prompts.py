"""
默认 Prompt 模板

基于 rerank/gen_data/generation_prompts.py 对齐，完整来源类型配置 + 长度配置。
"""

import random
from typing import Dict, List, Tuple

# ========== 来源类型配置 ==========

SOURCE_TYPES: Dict[str, Dict[str, str]] = {
    # 学术类
    "academic_paper": {
        "name": "学术论文",
        "description": "学术论文、期刊文章、会议论文",
        "style": "正式学术语言，客观陈述，可能包含引用标记，结构化论述",
        "examples": "Nature, Science, IEEE, ACM 论文",
    },
    "thesis": {
        "name": "学位论文",
        "description": "硕博论文、学位论文章节",
        "style": "详细论述，文献综述风格，理论与实证结合",
        "examples": "硕士论文、博士论文的文献综述或方法论章节",
    },
    "textbook": {
        "name": "教材",
        "description": "教科书、教材章节",
        "style": "教学导向，概念清晰，循序渐进，可能包含例题",
        "examples": "大学教材、专业课本",
    },

    # 技术类
    "documentation": {
        "name": "技术文档",
        "description": "技术文档、API 文档、SDK 指南",
        "style": "精确技术描述，代码示例，参数说明，版本信息",
        "examples": "Python docs, React 文档, AWS 文档",
    },
    "tutorial": {
        "name": "教程",
        "description": "教程、操作指���、How-to 文章",
        "style": "步骤化指导，实践导向，包含代码/操作示例",
        "examples": "Medium 技术教程, 掘金文章, DigitalOcean 指南",
    },
    "stackoverflow": {
        "name": "技术问答",
        "description": "技术问答、Stack Overflow 风格",
        "style": "问题-解答格式，代码片段，实用解决方案",
        "examples": "Stack Overflow, SegmentFault, 知乎技术问答",
    },
    "api_reference": {
        "name": "API 参考",
        "description": "API 参考文档、接口说明",
        "style": "结构化参数列表，请求/响应示例，错误码说明",
        "examples": "Swagger 文档, GraphQL Schema, REST API 文档",
    },
    "code_comment": {
        "name": "代码注释",
        "description": "代码注释、docstring、内联文档",
        "style": "简洁技术描述，参数/返回值说明，使用示例",
        "examples": "Python docstring, JSDoc, Javadoc",
    },

    # 百科/参考
    "encyclopedia": {
        "name": "百科",
        "description": "百科词条、维基百科风格",
        "style": "中立客观，全面概述，结构化信息，可验证事实",
        "examples": "Wikipedia, 百度百科, 专业百科",
    },
    "glossary": {
        "name": "术语表",
        "description": "术语表、词典定义、概念解释",
        "style": "简洁精确，定义式语言，可能包含同义词/反义词",
        "examples": "专业术语表, 技术词典, Glossary",
    },
    "faq": {
        "name": "FAQ",
        "description": "常见问题解答、帮助中心",
        "style": "问答格式，简洁直接，面向终端用户",
        "examples": "产品 FAQ, 帮助中心, 客服知识库",
    },

    # 新闻/媒体
    "news": {
        "name": "新闻",
        "description": "新闻报道、媒体文章",
        "style": "倒金字塔结构，时效性信息，引用消息源",
        "examples": "BBC, CNN, 新华社, 财经新闻",
    },
    "blog": {
        "name": "博客",
        "description": "博客文章、个人观点文章",
        "style": "个人化语气，观点表达，经验分享，相对随意",
        "examples": "个人技术博客, Medium 文章, 公众号文章",
    },
    "press_release": {
        "name": "新闻稿",
        "description": "新闻稿、企业公告、官方声明",
        "style": "正式官方语言，强调要点，包含引用和联系方式",
        "examples": "企业新闻稿, 产品发布公告, 官方声明",
    },

    # 社区/论坛
    "forum": {
        "name": "论坛",
        "description": "论坛帖子、社区讨论",
        "style": "口语化，个人经验，可能包含问答互动",
        "examples": "Reddit, 贴吧, V2EX, 专业论坛",
    },
    "review": {
        "name": "评测",
        "description": "产品评测、用户评价、对比分析",
        "style": "评价性语言，优缺点分析，个人体验",
        "examples": "产品测评, 用户评价, 对比评测文章",
    },
    "qa_community": {
        "name": "问答社区",
        "description": "知乎、Quora 风格的问答",
        "style": "深度回答，个人见解，可能包含故事性",
        "examples": "知乎回答, Quora 回答",
    },

    # 官方/正式
    "legal": {
        "name": "法律文书",
        "description": "法律法规、政策文件、合同条款",
        "style": "严谨法律语言，条款化，定义精确",
        "examples": "法律条文, 政策文件, 用户协议",
    },
    "manual": {
        "name": "说明书",
        "description": "产品说明书、用户手册",
        "style": "操作指导，注意事项，图文结合描述",
        "examples": "产品说明书, 用户手册, 操作指南",
    },
    "specification": {
        "name": "规格文档",
        "description": "��术规格、标准文档、协议规范",
        "style": "精确技术参数，标准化格式，可能包含表格",
        "examples": "RFC 文档, ISO 标准, 技术规格书",
    },
    "guide": {
        "name": "指南",
        "description": "最佳实践指南、入门指南、实施指南",
        "style": "结构化建议，实践导向，分步骤说明",
        "examples": "官方最佳实践, 入门指南, 迁移指南",
    },

    # 商业/研究
    "report": {
        "name": "研究报告",
        "description": "研究报告、行业分析、调研报告",
        "style": "数据驱动，分析性语言，图表引用，结论导向",
        "examples": "行业研究报告, 市场分析, 咨询报告",
    },
    "whitepaper": {
        "name": "白皮书",
        "description": "白皮书、技术白皮书",
        "style": "权威性论述，问题-解决方案结构，技术深度",
        "examples": "技术白皮书, 行业白皮书, 区块链白皮书",
    },
    "case_study": {
        "name": "案例研究",
        "description": "案例分析、成功案例、实践案例",
        "style": "叙事性结���，问题-方案-结果，数据支撑",
        "examples": "企业案例, 技术案例, 教学案例",
    },
}


# ========== 长度类型配置 ==========

LENGTH_TYPES: Dict[str, Tuple[int, int]] = {
    "short": (50, 200),        # 定义、摘要、简短回答
    "medium": (200, 500),      # 段落、片段、一般回答
    "long": (500, 1000),       # 章节、详细段落、深度回答
    "very_long": (1000, 2000), # 完整文章、长文
}

LENGTH_TYPE_NAMES: Dict[str, str] = {
    "short": "简短（50-200字）",
    "medium": "中等（200-500字）",
    "long": "较长（500-1000字）",
    "very_long": "长文（1000-2000字）",
}


# ========== 来源-长度关联 ==========

SOURCE_LENGTH_DISTRIBUTION: Dict[str, List[str]] = {
    # 学术类 - 偏长
    "academic_paper": ["medium", "long", "very_long"],
    "thesis": ["long", "very_long"],
    "textbook": ["medium", "long"],

    # 技术类 - 中等偏长
    "documentation": ["short", "medium", "long"],
    "tutorial": ["medium", "long", "very_long"],
    "stackoverflow": ["short", "medium"],
    "api_reference": ["short", "medium"],
    "code_comment": ["short"],

    # 百科/参考 - 中等
    "encyclopedia": ["medium", "long"],
    "glossary": ["short"],
    "faq": ["short", "medium"],

    # 新闻/媒体 - 中等
    "news": ["medium", "long"],
    "blog": ["medium", "long", "very_long"],
    "press_release": ["medium", "long"],

    # 社区/论坛 - 偏短
    "forum": ["short", "medium"],
    "review": ["short", "medium", "long"],
    "qa_community": ["medium", "long"],

    # 官方/正式 - 偏长
    "legal": ["medium", "long", "very_long"],
    "manual": ["medium", "long"],
    "specification": ["medium", "long"],
    "guide": ["medium", "long"],

    # 商业/研究 - 偏长
    "report": ["long", "very_long"],
    "whitepaper": ["long", "very_long"],
    "case_study": ["medium", "long", "very_long"],
}


# ========== 辅助函数 ==========

def get_source_info(source_type: str) -> Dict[str, str]:
    """获取指定来源类型的信息"""
    return SOURCE_TYPES.get(source_type, SOURCE_TYPES["encyclopedia"])


def sample_source_config(
    source_type: str = None,
    length_type: str = None,
) -> Dict[str, any]:
    """随机采样来源和长度配置

    Args:
        source_type: 指定来源类型，None 则随机选择
        length_type: 指定长度类型，None 则根据来源随机选择

    Returns:
        配置字典，包含 source_type, source_info, length_type, min_words, max_words
    """
    if source_type is None:
        source_type = random.choice(list(SOURCE_TYPES.keys()))

    source_info = SOURCE_TYPES.get(source_type, SOURCE_TYPES["encyclopedia"])

    if length_type is None:
        valid_lengths = SOURCE_LENGTH_DISTRIBUTION.get(
            source_type, list(LENGTH_TYPES.keys())
        )
        length_type = random.choice(valid_lengths)

    min_words, max_words = LENGTH_TYPES.get(length_type, (200, 500))

    return {
        "source_type": source_type,
        "source_name": source_info["name"],
        "source_description": source_info["description"],
        "source_style": source_info["style"],
        "length_type": length_type,
        "length_name": LENGTH_TYPE_NAMES.get(length_type, "中等"),
        "min_words": min_words,
        "max_words": max_words,
    }


# 来源类型选项字符串，用于角色生成 prompt
_SOURCE_TYPE_OPTIONS = "\n".join([
    f"   - {key}: {info['name']}（{info['description']}）"
    for key, info in SOURCE_TYPES.items()
])


# ===== 角色生成 =====

ROLE_GENERATION_PROMPT = """根据以下 Query 和 Answer，生成一个可能会提出该问题的角色配置。

Query：{query}
Answer：{answer}

每个角色配置包含：
1. character: 角色身份（如：医学研究员、产品经理、大学生、投资分析师）
2. question_type: 问题类型（factual, analytical, procedural, conceptual, comparative）
3. difficulty: 难度级别
   - beginner: 入门水平
   - intermediate: 中等水平
   - advanced: 高级水平
   - expert: 专家水平
4. source_type: 该角色可能检索到的文档来源类型，从以下选择：
""" + _SOURCE_TYPE_OPTIONS + """
5. length_type: 文档长度类型
   - short: 简短（50-200字），如术语定义、简短回答
   - medium: 中等（200-500字），如段落、一般回答
   - long: 较长（500-1000字），如章节、详细解释
   - very_long: 长文（1000-2000字），如完整文章

输出格式（严格 JSON）：
{{"character": "角色描述", "question_type": "问题类型", "difficulty": "难度级别", "source_type": "文档类型", "length_type": "长度类型"}}

直接输出 JSON，不要其��内容。"""

ROLE_GENERATION_FROM_DOC_PROMPT = """根据以下文档内容，生成 {num_roles} 个可能会对该文档提问的角色配置。

## 文档内容
{document}

要求：
- 角色之间应有明显差异（不同背景、不同关注点、不同专业水平）
- 角色应与文档主题相关
- 根据文档复杂度决定角色多样性

每个角色配置包含：
1. character: 角色身份（如：医学研究员、产品经理、大学生）
2. question_type: 问题类型（factual, analytical, procedural, conceptual, comparative）
3. difficulty: 难度级别（beginner, intermediate, advanced, expert）
4. source_type: 最匹配的文档来源类型，从以下选择：
""" + _SOURCE_TYPE_OPTIONS + """
5. length_type: 文档长度类型（short, medium, long, very_long）

输出格式（严格 JSON）：
{{
    "roles": [
        {{"character": "角色描述", "question_type": "问题类型", "difficulty": "难度级别", "source_type": "文档类型", "length_type": "长度类型"}}
    ]
}}

直接输出 JSON，不要其他内容。"""


# ===== 关键点提取 =====

KEYPOINT_EXTRACTION_PROMPT = """你是一个关键信息提取专家。请从以下文档中提取关键信息点。

## 文档内容
{document}

## 任务
提取文档中的关键信息点，这些信息点应该：
1. 是文档的核心内容
2. 可以作为生成问题的基础
3. 包含具体的事实、概念或操作步骤

## 输出格式
请以 JSON 格式返回：
{{
    "keypoints": [
        "关键点1",
        "关键点2",
        ...
    ]
}}

最多提取 {max_keypoints} 个关键点。"""


# ===== QA 生成 =====

QA_GENERATION_PROMPT = """你是一个专业的问答对生成专家。请根据以下文档内容生成高质量的问答对。

## 文档内容
{document}

## 关键点（可选参考）
{keypoints}

## 任务
首先判断文档内容是否适合生成有价值的问答对。如果文档存在以下任一问题，直接返回空结果：
- 内容过短、碎片化，缺乏实质信息（如仅包含年份、数字、标题等）
- 内容为乱码、无意义字符或纯格式标记
- 信息过于笼统泛化，无法生成有具体答案的问题
- 内容为纯广告、导航链接或无关噪声

如果内容适合，生成 {num_qa} 个问答对，要求：
1. 问题必须是独立的检索式查询，不得引用文档本身。禁止使用"文档中"、"文中"、"根据文档"、"上文"、"该文"等指代——问题脱离文档后仍需语义完整
2. 问题应该自然、具体，像真实用户在搜索引擎中输入的查询
3. 答案应该准确、完整，直接基于文档内容
4. 问题类型多样化（事实型、分析型、操作型等）
5. 难度适中，覆盖文档的不同方面

## 反面示例（禁止）
- ❌ "文档中的目标函数如何表示？" → ✅ "RLHF 的目标函数如何表示为最大化期望奖励减去KL散度？"
- ❌ "文中提到的分区函数Z(x)是如何定义的？" → ✅ "强化学习中分区函数Z(x)的定义和作用是什么？"

## 输出格式
请以 JSON 格式返回：
{{
    "qa_pairs": [
        {{"query": "问题1", "answer": "答案1"}},
        ...
    ]
}}

不适合生成时，qa_pairs 返回空数组 []。"""


QA_GENERATION_WITH_ROLE_PROMPT = """你是一个专业的问答对生成专家。请根据以下文档内容和用户角色生成问答对。

## 文档内容
{document}

## 用户角色配置
- 角色: {character}
- 问题类型: {question_type}
- 难度级别: {difficulty}

## 任务
首先判断文档内容是否适合生成有价值的问答对。如果内容过短、碎片化、缺乏实质信息或为无意义噪声，直接返回空结果。

如果内容适合，以该用户角色的视角生成一个问答对：
1. 问题必须是独立的检索式查询，禁止使用"文档中"、"文中"、"根据文档"、"上文"等指代词。问题脱离文档后仍需语义完整
2. 问题应该符合该角色的专业背景和知识水平
3. 问题类型应该与配置一致
4. 答案应该准确、完整，难度适配用户水平

## 输出格式
请以 JSON 格式返回：
{{
    "query": "问题",
    "answer": "答案"
}}

不适合生成时，query 和 answer 返回空字符串。"""


# ===== 正负例生成 =====

POSITIVE_CHUNK_PROMPT = """生成一段能够回答问题的文档片段。

**文档来源**：{source_type}（{source_name}）
  - 类型说明：{source_description}
  - 写作风格：{source_style}

**目标长度**：{length_name}（约 {min_words}-{max_words} 字）

**写作要求**：
1. 必须包含能够回答 Query 的关键信息
2. 严格按照 {source_type} 的写作风格和语言特点
3. 长度控制在 {min_words}-{max_words} 字范围内
4. 内容准确、专业、自然流畅

Query：{query}
Answer：{answer}

输出格式（严格JSON）：
{{"output": "文档片段内容"}}"""


POSITIVE_CHUNK_WITH_ROLE_PROMPT = """生成一段能够回答问题的文档片段。

**文档来源**：{source_type}（{source_name}）
  - 类型说明：{source_description}
  - 写作风格��{source_style}

**目标读者**：{character}（{difficulty} 级别）
**目标长度**：{length_name}（约 {min_words}-{max_words} 字）

**写作要求**：
1. 必须包含能够回答 Query 的关键信息
2. 严格按照 {source_type} 的写作风格和语言特点
3. 长度控制在 {min_words}-{max_words} 字范围内
4. 内容准确、专业、自然流畅

Query：{query}
Answer：{answer}

输出格式（严格JSON）：
{{"output": "文档片段内容"}}"""


NEGATIVE_CHUNK_PROMPT = """生成一段与 Query 主题相关但**无法回答问题**的文档片段。

**文档来源**：{source_type}（{source_name}）
  - 类型说明：{source_description}
  - 写作风格：{source_style}

**目标长度**：{length_name}（约 {min_words}-{max_words} 字）

**写作要求**：
1. 主题相关：涉及相同领域或相似概念
2. **不能包含回答 Query 的信息**
3. 可以是：相关背景知识、相邻话题、相似但不同的概念、同领域其他问题的答案
4. 严格按照 {source_type} 的写作风格和语言特点
5. 长度控制在 {min_words}-{max_words} 字范围内

Query：{query}
Answer（仅供参考，不要包含）：{answer}

输出格式（严格JSON）：
{{"output": "文档片段内容"}}"""


NEGATIVE_CHUNK_WITH_ROLE_PROMPT = """生成���段与 Query 主题相关但**无法回答问题**的文档片段。

**文档来源**：{source_type}（{source_name}）
  - 类型说明：{source_description}
  - 写作风格：{source_style}

**目标读者**：{character}（{difficulty} 级别）
**目标长度**：{length_name}（约 {min_words}-{max_words} 字）

**写作要求**：
1. 主题相关：涉及相同领域或相似概念
2. **不能包含回答 Query 的信息**
3. 可以是：相关背景知识、相邻话题、相似但不同的概念、同领域其他问题的答案
4. 严格按照 {source_type} 的写作风格和语言特点
5. 长度控制在 {min_words}-{max_words} 字范围内

Query：{query}
Answer（仅供参考，不要包含）：{answer}

输出格式（严格JSON）：
{{"output": "文档片段内容"}}"""


# ===== 校验 =====

VALIDATION_PROMPT = """你是一个专业的数据质量评估专家。请评估以下正负例的质量。

## 问答对
Query: {query}
Answer: {answer}

## 正例
{positive}

## 负例
{negative}

## 任务
评估正负例的质量：
1. 正例是否包含回答问题所需的关键信息？
2. 负例是否与问题相关但确实无法回答问题？
3. 正负例的区分度是否足够？

## 输出格式
请以 JSON 格式返回：
{{
    "positive_valid": true/false,
    "positive_reason": "正例评估理由",
    "negative_valid": true/false,
    "negative_reason": "负例评估理由",
    "overall_valid": true/false,
    "overall_reason": "整体评估理由"
}}"""


# ===== 文档质量 =====

DOC_QUALITY_PROMPT = """你是一个专业的文档质量评估专家。请评估以下文档的质量，重点关注该文档能否被用于生成独立的检索式问答训练数据。

## 文档内容
{document}

## 任务
评估文档质量，考虑以下维度：
1. 内容完整性：信息是否完整、有实质内容
2. 语言质量：是否通顺、无乱码或无意义内容
3. 信息密度：是否包含有价值的信息，而非纯广告或无关内容
4. 实体丰富度：文档中是否包含具体的概念、术语、方法名、人名等命名实体，使得可以生成不依赖上下文的独立问题。如果文档充斥代词（"它"、"该方法"、"上述"）而缺少明确指称对象，则实体丰富度低
5. 检索可用性：是否适合生成独立的检索式查询。以下情况不适合：
   - 文档是纯目录、索引、参考文献列表
   - 文档大量依赖其他章节上下文，独立阅读无法理解
   - 文档内容过于抽象笼统，缺少具体可查询的知识点

## 输出格式
请以 JSON 格式返回：
{{
    "completeness": 0.0-1.0,
    "language_quality": 0.0-1.0,
    "information_density": 0.0-1.0,
    "entity_richness": 0.0-1.0,
    "retrieval_usability": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "reason": "评估理由",
    "is_usable": true/false
}}"""
