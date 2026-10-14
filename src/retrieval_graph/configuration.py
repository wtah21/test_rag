"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from retrieval_graph import prompts
from shared.configuration import BaseConfiguration


@dataclass(kw_only=True)
class AgentConfiguration(BaseConfiguration):
    """agent 的配置。"""

    # models

    query_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="zhipu/glm-5.3-flash",
        metadata={
            "description": "用于处理和优化查询的语言模型。格式应为：provider/model-name。"
        },
    )

    response_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="zhipu/glm-5.3-flash",
        metadata={
            "description": "用于生成回复的语言模型（需支持 function calling 以调用记忆工具）。格式应为：provider/model-name。"
        },
    )

    memory_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="zhipu/glm-5.3-flash",
        metadata={
            "description": "用于从对话中抽取/更新长期记忆的语言模型。格式应为：provider/model-name。"
        },
    )

    # 提示词

    router_system_prompt: str = field(
        default=prompts.JEV_ROUTER_SYSTEM_PROMPT,
        metadata={
            "description": "三层路由中第三层 JEV 决策模型的分类提示词（前两层无法确定时使用）。"
        },
    )

    general_system_prompt: str = field(
        default=prompts.GENERAL_SYSTEM_PROMPT,
        metadata={
            "description": "用于回复普通问题的系统提示词。"
        },
    )

    intent_rewrite_system_prompt: str = field(
        default=prompts.INTENT_REWRITE_SYSTEM_PROMPT,
        metadata={
            "description": "用于分析用户问题意图并决定是否改写的系统提示词。"
        },
    )

    enable_intent_rewrite: bool = field(
        default=True,
        metadata={
            "description": "是否启用意图分析改写节点：开启时对模糊问题改写后再检索；"
                            "关闭时直接用用户原问题检索（对照实验用）。"
        },
    )

    translate_query_system_prompt: str = field(
        default=prompts.TRANSLATE_QUERY_SYSTEM_PROMPT,
        metadata={
            "description": "用于从中文检索问题抽取英文检索关键词（逐字英文专名 + "
                            "关键概念英译，BM25 稀疏检索用）的系统提示词。"
        },
    )

    context_top_k: int = field(
        default=8,
        metadata={
            "description": "单 query 直检索流程进入 grade/respond 的最终上下文条数"
                            "（经 top3/top8 对照实验验证，8 条在 reasoning/cross_doc 题型"
                            "上显著优于 3 条）。"
        },
    )

    generate_queries_system_prompt: str = field(
        default=prompts.GENERATE_QUERIES_SYSTEM_PROMPT,
        metadata={
            "description": "研究员用于根据研究计划中的步骤生成查询的系统提示词。"
        },
    )

    grade_documents_system_prompt: str = field(
        default=prompts.GRADE_DOCUMENTS_SYSTEM_PROMPT,
        metadata={
            "description": "用于评判检索到的文档是否足以回答用户问题的系统提示词。"
        },
    )

    response_system_prompt: str = field(
        default=prompts.RESPONSE_SYSTEM_PROMPT,
        metadata={"description": "用于生成回复的系统提示词。"},
    )

    # 分层路由

    routing_rule_keywords: list[str] = field(
        default_factory=lambda: [
            "langchain", "langgraph", "langsmith", "langgraph-checkpoint",
            "agent", "agents", "memory", "tool", "tools", "retriever", "rag",
            "chain", "embedding", "vector", "prompt", "llm", "checkpoint",
            "interrupt", "human-in-the-loop", "middleware", "subgraph",
            "structured output", "text splitter",
        ],
        metadata={"description": "第一层规则路由的匹配关键词。"},
    )

    routing_semantic_examples: dict[str, list[str]] = field(
        default_factory=lambda: {
            "langchain": [
                "如何在 LangGraph 中配置检查点持久化？",
                "LangChain 的 ToolNode 怎么和条件边配合使用？",
                "怎样给 create_react_agent 添加长期记忆？",
                "LangSmith 上如何追踪一次图执行的每个节点？",
                "如何实现 human-in-the-loop 中断？",
                "MarkdownHeaderTextSplitter 怎么用？",
                "怎么给 RAG 接入检索器？",
                "with_structured_output 支持哪些模式？",
                "子图之间状态如何传递？",
                "自定义 middleware 中间件怎么写？",
            ],
            "general": [
                "今天天气怎么样？",
                "帮我写一首关于秋天的诗。",
                "推荐几部好看的科幻电影。",
                "介绍一下量子计算的原理。",
                "怎么做红烧肉？",
                "Python 里列表和元组的区别是什么？",
                "写一封辞职邮件。",
            ],
            "more-info": [
                "我的代码报错了，怎么办？",
                "这个功能不行了。",
                "跑不通，帮我看看。",
                "它有问题，你懂我意思吗？",
                "帮我优化一下。",
            ],
        },
        metadata={"description": "第二层语义路由各标签的示例问题。"},
    )

    routing_semantic_negatives: list[str] = field(
        default_factory=lambda: [
            "法国的首都是哪里？",
            "帮我订一张明天去上海的机票。",
            "推荐一家附近的餐厅。",
            "今年世界杯谁夺冠了？",
        ],
        metadata={
            "description": "语义路由阈值 fit 用的负样本（不属于任何标签的无关问题）。"
        },
    )

    routing_semantic_top_k: int = field(
        default=5,
        metadata={"description": "第二层语义路由检索的示例问题条数。"},
    )

    routing_semantic_auto_fit: bool = field(
        default=True,
        metadata={"description": "是否用 semantic-router 的 fit 自动调优各标签阈值。"},
    )

    clarify_system_prompt: str = field(
        default=prompts.CLARIFY_SYSTEM_PROMPT,
        metadata={"description": "澄清节点内生成面向用户追问话术的系统提示词。"},
    )

    # 重排

    rerank_enabled: bool = field(
        default=True,
        metadata={"description": "是否在 RRF 融合后启用智谱 cross-encoder 重排。"},
    )

    rerank_candidates: int = field(
        default=20,
        metadata={"description": "RRF 融合后送进重排的候选文档数。"},
    )

    rerank_top_n: int = field(
        default=4,
        metadata={"description": "重排后保留并进入 grade/respond 的最终文档数。"},
    )

    memory_update_system_prompt: str = field(
        default=prompts.MEMORY_UPDATE_SYSTEM_PROMPT,
        metadata={
            "description": "用于从对话中抽取/更新长期记忆的模型系统提示词。"
        },
    )
