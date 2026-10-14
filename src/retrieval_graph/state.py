"""检索图的状态管理。

本模块定义检索图使用的状态结构，包括 agent 状态、输入状态和路由器分类模式的定义。
"""

from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


# 可选的：InputState 是 State 的受限版本，用于
# 定义比内部维护的更窄的外部接口。
@dataclass(kw_only=True)
class InputState:
    """表示 agent 的输入状态。

    该类定义了输入状态的结构，包括用户与 agent 之间交换的消息。
    它是完整 State 的受限版本，与内部维护的状态相比，向外部提供了更窄的接口。
    """

    messages: Annotated[list[AnyMessage], add_messages]
    """消息跟踪 agent 的主要执行状态。

    通常积累 Human/AI/Human/AI 交替的消息模式；如果
    你要将此模板与调用工具类的 ReAct agent 模式结合，
    它可能是这样的：

    1. HumanMessage - 用户输入
    2. 带 .tool_calls 的 AIMessage - agent 选择要使用的工具来收集
        信息
    3. ToolMessage - 执行工具后返回的响应（或错误）

        （...根据需要重复第 2 和第 3 步...）
    4. 不带 .tool_calls 的 AIMessage - agent 以非结构化格式回复用户。

    5. HumanMessage - 用户回复下一个对话轮次。

        （...根据需要重复第 2-5 步...）

    合并两个消息列表，按 ID 更新已有消息。

    默认情况下，这会确保状态是"只追加"的，除非
    新消息与已有消息具有相同的 ID。

    Returns:
        返回一个新消息列表，其中 `right` 中的消息已合并到 `left`。
        如果 `right` 中的消息与 `left` 中的消息 ID 相同，则
        `right` 中的消息将替换 `left` 中的消息。"""


class Router(TypedDict):
    """对用户查询进行分类（仅分类，不含追问话术）。"""

    type: Literal["more-info", "langchain", "general"]


# 这是 agent 的主要状态，你可以在这里存储任何信息


@dataclass(kw_only=True)
class AgentState(InputState):
    """检索图 / agent 的状态。"""

    router: Router = field(default_factory=lambda: Router(type="general"))
    """路由器对用户查询的分类结果。"""
    route_layer: str = ""
    """本轮路由命中的层：rule / semantic / jev（由 analyze_and_route_query 写入）。"""
    rewritten_question: str = ""
    """意图分析节点产出的有效检索问题：问题表述清晰时为用户原问题，
    表述模糊时为改写后的规范问题。由 analyze_and_rewrite_question 节点写入，
    retrieve_context 以此为输入检索。"""
    need_rewrite: bool = False
    """意图分析是否对原问题做了改写（False=直接用原问题，True=改写后检索）。
    记录该标志便于评估意图分析节点对召回的实际影响。"""
    intent_reason: str = ""
    """意图分析的判断依据（为什么清晰 / 为什么需要这样改写），供日志与评估分析。"""
    english_query: str = ""
    """有效问题的英文检索关键词串（由英文关键词抽取器产出：逐字英文专名 +
    关键概念英译），仅供 BM25 稀疏检索使用；稠密路仍用 rewritten_question
    原文（BGE-M3 多语言可跨语种命中英文语料）。"""
    use_sparse: bool = False
    """有效问题是否抽到英文检索关键词（keywords 非空）：为 True 时稀疏路启用，
    查询文本为 english_query；为 False 时 BM25 通用词无判别力，检索关闭稀疏路
    仅用稠密向量（避免 RRF 挤掉稠密路好文档）。"""
    documents: list[Document] = field(default_factory=list)
    """由检索器填充。这是 agent 可以引用的文档列表（retrieve_context 单 query
    混合检索后截断写入前 context_top_k 条；节点串行写入故直接覆盖而非合并）。
    grade / respond 基于它工作。"""
    grade: Literal["good", "bad"] = "good"
    """grade_documents 节点对当前检索文档的评判结果（good=可直接回答，bad=需要重新检索）。"""
    grade_attempts: int = 0
    """grade 评判为 bad 的累计次数，用于限制重新检索的最大迭代次数。"""
    grade_feedback: str = ""
    """最近一次 bad 评判的理由（reasoning），作为反馈传给下一轮意图改写/检索。"""
    grade_score: float = 0.0
    """最近一次 grade 的数值分（0~1，对"文档足以回答该问题"的充分/相关程度打分）。"""
    best_score: float = -1.0
    """历轮 grade 的最高数值分；-1 表示本轮尚无任何评分（新一轮问答的初始值）。"""
    best_documents: list[Document] = field(default_factory=list)
    """历轮 grade 最高分那一轮的文档快照；全部轮次都判 bad 且超出重试上限时，
    以此替代当前轮文档作为最终 context，保证 respond 尽量基于相对最好的一轮检索。"""
    clarify_count: int = 0
    """本轮问答中因 more-info 触发澄清中断的累计次数，达到上限后不再追问。"""

    # 如有需要，可以随意为你的状态添加更多属性。
    # 常见示例包括检索到的文档、提取的实体、API 连接等。
