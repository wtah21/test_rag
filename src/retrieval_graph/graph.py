"""对话式检索图的主入口。

本模块定义了对话式检索图的核心结构和功能，包括主图定义、状态管理，
以及处理与路由用户查询、分析用户问题意图、检索相关文档并生成回答等
关键函数。

路由与澄清：采用规则（rule）/ 语义（semantic）/ JEV（jev）三层级联路由，将
用户提问分类为 langchain（进入检索链路）、general（普通问题）或 more-info
（信息不足）。more-info 由 ask_for_clarification 节点调用智谱大模型生成追问
话术，并以 langgraph interrupt 中断执行，向用户追问缺失信息并等待其补充
（同一轮最多 MAX_CLARIFY_ROUNDS 次；无法中断或超限时降级为 general 尽力作答）。

检索链路（langchain 分支，单 query 直检索 + 条件改写）：
- analyze_and_rewrite_question：分析用户原始问题的意图——问题表述清晰时
  不需要改写（直接用原问题检索），表述模糊时分析真实意图并改写为适合检索的
  规范问题（可配置 enable_intent_rewrite 关闭，此时恒用原问题）；
- retrieve_context：先抽取有效问题的英文检索关键词（知识库语料为英文、BM25
  是纯词匹配，中文分词打不中英文词表；整句英译有功能词稀释，故抽关键词），
  再对有效问题做单 query 混合检索：
  稠密路用中文有效问题（BGE-M3 多语言可跨语种命中英文语料），稀疏路用英译
  （BM25），RRF 融合后取前 context_top_k（默认 8）条直接作为最终 context
  （不再生成多 query、不再做 step/query 级配额合并）。

检索到文档后由 grade_documents 节点评判其是否足以回答问题，并输出 0~1
数值分（grade_score，记录相对最好的一轮 best_documents/best_score）：
- good：直接进入 respond 生成回答；
- bad：带着评判理由回到 analyze_and_rewrite_question 重新改写/翻译/检索
  （最多重试 MAX_GRADE_ATTEMPTS 次）；
- bad 且重试超限：用历轮分数最高（best_score）那一轮的文档覆盖 documents
  作为最终 context 强制 respond，保证回答尽量基于相对最好的一轮检索。

记忆能力：
- 短期记忆：通过 Postgres checkpointer 按 thread_id 保存会话消息历史。
- 长期记忆：通过 Postgres store 按用户保存事实与偏好；各回复类节点
  （respond / respond_to_general_query）都以工具型 agent
  形式向模型提供 load_user_memories / update_user_memory，由模型自行判断
  是否按需读取或保存长期记忆（见 retrieval_graph.memory）。
"""

import asyncio
import logging
import os
from typing import Any, Literal, TypedDict, cast

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from retrieval_graph import memory
from retrieval_graph.configuration import AgentConfiguration
from retrieval_graph.routing import _get_bge_m3_encoder, resolve_route
from retrieval_graph.state import AgentState, InputState, Router
from shared.hybrid_retrieval import hybrid_search
from shared.utils import format_docs, load_chat_model

logger = logging.getLogger(__name__)

# grade_documents 连续判定 bad 的最大次数，达到后强制进入 respond 生成回答
MAX_GRADE_ATTEMPTS = 3
# respond 节点中记忆工具调用的最大迭代轮数
MAX_RESPONSE_TOOL_ROUNDS = 3
# 同一轮问答中 more-info 澄清中断的最大次数，超过后不再追问、降级尽力作答
MAX_CLARIFY_ROUNDS = 2
# LangGraph 平台（langgraph dev / LangGraph API 的 local_dev）会自动依据
# POSTGRES_URI 提供持久化，并禁止图再携带自定义 checkpointer/store
# （否则平台加载时报 "includes a custom checkpointer and store"）。
# 因此平台运行时编译时不显式接入；脱离平台直接运行（本地脚本/测试）时仍显式接入。
_ON_LANGGRAPH_PLATFORM = (
    os.environ.get("LANGSMITH_LANGGRAPH_API_VARIANT") == "local_dev"
)


def _latest_human_question(state: AgentState) -> str:
    """取对话中最近一条用户提问文本。"""
    for message in reversed(state.messages or []):
        if getattr(message, "type", None) == "human":
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


async def analyze_and_route_query(
        state: AgentState, *, config: RunnableConfig
    ) -> dict[str, Any]:
        """分析用户的查询并通过三层路由确定合适的路由类型。

        采用级联的三层路由（见 retrieval_graph.routing.resolve_route）：
        - 规则层（rule）：命中 routing_rule_keywords 关键词即短路判定；
        - 语义层（semantic）：规则层无法确定时，用 BGE-M3 编码的语义路由匹配
          示例问题，达到阈值即短路判定；
        - JEV 层（jev）：前两层均无法确定时，调用 JEV 决策模型（以
          router_system_prompt 为提示词）做最终分类。
        命中层写入 route_layer，分类结果（langchain / general / more-info）
        写入 router。

        各分类的走向：
        - langchain：进入检索链路（analyze_and_rewrite_question）；
        - general：进入普通问题回复（respond_to_general_query）；
        - more-info：进入澄清中断（ask_for_clarification），追问话术由该节点
          调用大模型生成。若检查点后端不可用（无法中断）或同一轮澄清次数已达
          MAX_CLARIFY_ROUNDS，则降级为 general 基于现有信息作答。

        Args:
            state (AgentState): agent 的当前状态，包括对话历史与澄清计数。
            config (RunnableConfig): 包含路由相关配置（规则关键词、语义示例、
                语义阈值拓扑、JEV 提示词等）的配置。

        Returns:
            dict[str, Any]: 包含 'router'（分类结果）、'route_layer'（命中层）
            与 'clarify_count'（澄清累计次数，非澄清流程中复位为 0）的字典；
            非澄清分支还会重置本轮 grade 记账字段。
        """
        configuration = AgentConfiguration.from_runnable_config(config)
        question = _latest_human_question(state)
        if not question:
            raise ValueError("未找到用户提问，无法进行路由")
        # 最近若干轮对话（截断）作为 JEV 的上下文
        history = "\n".join(
            f"{m.type}: {m.content}"
            for m in (state.messages or [])[-6:]
            if getattr(m, "content", None)
        )
        router, layer = await resolve_route(
            question,
            history,
            examples=configuration.routing_semantic_examples,
            negatives=configuration.routing_semantic_negatives,
            top_k=configuration.routing_semantic_top_k,
            auto_fit=configuration.routing_semantic_auto_fit,
            encoder=_get_bge_m3_encoder(),
            keywords=configuration.routing_rule_keywords,
            jev_prompt=configuration.router_system_prompt,
        )
        if router["type"] not in ("langchain", "more-info", "general"):
            raise ValueError(f"路由输出无效: {router!r}")

        if router["type"] == "more-info":
            can_clarify = _ON_LANGGRAPH_PLATFORM or memory.get_checkpointer() is not None
            if can_clarify and state.clarify_count < MAX_CLARIFY_ROUNDS:
                return {"router": router, "route_layer": layer}
            logger.warning(
                "路由判定 more-info 但无法继续澄清"
                "（checkpointer 可用=%s，已澄清 %d 次），降级为 general",
                can_clarify,
                state.clarify_count,
            )
            return {
                "router": Router(type="general"),
                "route_layer": layer,
                "clarify_count": 0,
            }
        return {
            "router": router,
            "route_layer": layer,
            "clarify_count": 0,
            "grade": "good",
            "grade_attempts": 0,
            "grade_feedback": "",
            "grade_score": 0.0,
            "best_score": -1.0,
            "best_documents": [],
            "documents": [],
        }


def route_query(
        state: AgentState,
    ) -> Literal["analyze_and_rewrite_question", "ask_for_clarification", "respond_to_general_query"]:
    """根据查询分类确定下一步。

    根据查询分类确定下一步要执行的操作。

    Args:
        state (AgentState): agent 的当前状态，包括路由器的分类结果。

    Returns:
        Literal["analyze_and_rewrite_question", "ask_for_clarification",
        "respond_to_general_query"]: 下一步要执行的操作。

    Raises:
        ValueError: 如果遇到未知的路由器类型。
    """
    _type = state.router["type"]
    if _type == "langchain":
        return "analyze_and_rewrite_question"
    elif _type == "more-info":
        return "ask_for_clarification"
    elif _type == "general":
        return "respond_to_general_query"
    else:
        raise ValueError(f"Unknown router type {_type}")


async def ask_for_clarification(
        state: AgentState, *, config: RunnableConfig
    ) -> dict[str, Any]:
    """因 more-info 中断本轮执行，向用户追问缺失信息并等待其补充。

    通过 langgraph 的 interrupt 机制暂停图的执行：追问话术由本节点调用大模型
    （以 clarify_system_prompt 为提示词，结合最近对话上下文与用户最新提问）实时
    生成，作为返回给调用方的载荷。用户补充（Command(resume=...)）后，把补充
    内容作为一条人类消息并入对话历史，并回到 analyze_and_route_query 重新分类。

    Args:
        state (AgentState): agent 的当前状态，提供对话历史以生成追问话术。
        config (RunnableConfig): 运行配置，提供用于生成追问话术的模型配置。

    Returns:
        dict[str, Any]: 包含追加的澄清消息 'messages' 与递增后的 'clarify_count'。
    """
    configuration = AgentConfiguration.from_runnable_config(config)
    latest = _latest_human_question(state)
    history = "\n".join(
        f"{m.type}: {m.content}"
        for m in (state.messages or [])[-6:]
        if getattr(m, "content", None)
    )
    clarify_prompt = configuration.clarify_system_prompt
    question = ""
    try:
        model = load_chat_model(configuration.query_model)
        resp = await model.ainvoke(
            [
                {"role": "system", "content": clarify_prompt},
                {"role": "user", "content": f"对话上下文：\n{history}\n\n用户最新提问：\n{latest}"},
            ]
        )
        question = resp.content if isinstance(resp.content, str) else str(resp.content)
    except Exception:  # noqa: BLE001
        logger.warning("澄清追问生成失败，使用通用追问", exc_info=True)
    if not question.strip():
        question = "请补充一些信息，以便我更好地帮助你："
    reply = interrupt(question)
    content = (
        str(reply).strip()
        if reply
        else "（用户暂未提供补充信息，请基于现有信息尽力回答。）"
    )
    return {
        "messages": [HumanMessage(content=content)],
        "clarify_count": state.clarify_count + 1,
    }


async def respond_to_general_query(
        state: AgentState, *, config: RunnableConfig
    ) -> dict[str, list[BaseMessage]]:
    """生成对与 LangChain 无关的普通问题的回复。

    当路由器将查询归类为普通问题时，会调用此节点。

    Args:
        state (AgentState): agent 的当前状态，包括对话历史。
        config (RunnableConfig): 包含用于回复的模型的配置。

    Returns:
        dict[str, list[str]]: 包含 'messages' 键的字典，值为生成的回复。
    """
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    system_prompt = configuration.general_system_prompt
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await _answer_with_memory_tools(model, messages, config=config)
    return {"messages": [response]}


async def analyze_and_rewrite_question(
        state: AgentState, *, config: RunnableConfig
    ) -> dict[str, Any]:
    """分析用户问题意图，输出用于检索的有效问题。

    输入为最近一条用户提问。若配置关闭意图改写（enable_intent_rewrite=False，
    供对照实验），直接把原问题作为有效问题返回，不调用模型；否则让模型判断
    问题表述是否清晰：
    - 清晰：不需要改写，有效问题=用户原问题（强制回填原问题，防止模型擅自改词
      稀释精确术语）；
    - 模糊：分析用户真实意图，改写为术语完整、便于检索的规范问题。

    若上一轮检索被 grade_documents 判为 bad，会把评判理由（reasoning）追加到
    系统提示词中，引导模型据此改写，避开已发现的检索盲区。

    同时清空上一轮的检索文档，保证新一轮检索只基于新的有效问题。

    Args:
        state (AgentState): agent 的当前状态，包括对话历史与上轮评判反馈。
        config (RunnableConfig): 包含用于意图分析的模型的配置。

    Returns:
        dict[str, Any]: 包含 'rewritten_question'（有效检索问题）、'need_rewrite'
        （是否改写）、'intent_reason'（判断依据）与 'documents'（清空）的字典。
    """
    question = _latest_human_question(state)
    if not question:
        raise ValueError("未找到用户提问，无法进行意图分析")

    configuration = AgentConfiguration.from_runnable_config(config)
    if not configuration.enable_intent_rewrite:
        # 对照实验：跳过意图分析改写，直接用原问题检索
        return {
            "rewritten_question": question,
            "need_rewrite": False,
            "intent_reason": "意图改写已关闭（对照），直接用原问题检索",
            "documents": [],
        }

    class IntentRewrite(TypedDict):
        """意图分析结果。"""

        need_rewrite: bool
        rewritten_question: str
        reasoning: str

    model = load_chat_model(configuration.query_model).with_structured_output(
        IntentRewrite
    )
    system_prompt = configuration.intent_rewrite_system_prompt
    if state.grade_feedback:
        # 携带上一轮检索质量评判，引导改写时避开已发现的检索盲区
        system_prompt += (
            "\n\n注意：上一轮检索未能获得足以回答用户问题的文档，"
            "检索质量评估给出的理由如下：\n"
            f"{state.grade_feedback}\n"
            "请据此判断问题表述是否需要改写，使改写后的问题更贴近知识库中"
            "实际存在的概念、API 名与章节（不要臆造不存在的术语）。"
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "human", "content": question},
    ]
    response = cast(IntentRewrite, await model.ainvoke(messages))
    if not isinstance(response, dict):
        # GLM 偶发漏调工具返回 None，抛错由题级重试兜底
        raise ValueError(f"意图分析输出无效: {response!r}")

    need_rewrite = bool(response.get("need_rewrite"))
    # 表述清晰（或模型未产出有效改写）时一律用原问题，避免改写稀释精确术语
    rewritten = question
    if need_rewrite:
        candidate = str(response.get("rewritten_question") or "").strip()
        if candidate:
            rewritten = candidate
    return {
        "rewritten_question": rewritten,
        "need_rewrite": need_rewrite,
        "intent_reason": str(response.get("reasoning") or ""),
        "documents": [],
    }


async def retrieve_context(
        state: AgentState, *, config: RunnableConfig
    ) -> dict[str, Any]:
    """用有效问题做单 query 检索，产出最终 context。

    知识库语料为英文、提问常为中文：稠密路（BGE-M3，多语言/跨语种）直接用
    中文有效问题即可命中英文语料；稀疏路 BM25 是纯词匹配、依赖词在语料中的
    判别频率，因此先用语言模型把有效问题抽取成对检索有判别力的英文关键词
    （含原问中逐字出现的英文专名与关键概念的英文译词），抽到关键词即启用
    稀疏路并把关键词串送入 BM25；纯中文且无可译判别词时关键词为空，关闭
    稀疏路仅用稠密检索。RRF 融合后取前 context_top_k（默认 8）条直接写入
    documents 作为最终 context，供 grade_documents 评判与 respond 生成回答
    ——不再生成多 query，不再做 query/step 级配额合并。关键词抽取失败时回退
    为纯稠密检索，不让单题因抽取失败而中止。

    Args:
        state (AgentState): agent 的当前状态，含待检索的有效问题（rewritten_question）。
        config (RunnableConfig): 包含抽取提示词、上下文条数与模型的配置。

    Returns:
        dict[str, Any]: 包含 'documents'（最终文档列表，恒 context_top_k 条，
        候选不足时按实际数量）、'english_query'（英文检索关键词串）与
        'use_sparse'（稀疏路是否启用）。
    """
    configuration = AgentConfiguration.from_runnable_config(config)
    question = state.rewritten_question

    class KeywordExtraction(TypedDict):
        """英文检索关键词抽取结果。"""

        keywords: list[str]

    # 1) 抽取英文检索关键词：逐字英文专名 + 关键概念英译，抽到即启用稀疏路
    english_query = ""
    use_sparse = False
    sparse_text = None
    try:
        model = load_chat_model(configuration.query_model).with_structured_output(
            KeywordExtraction
        )
        messages = [
            {
                "role": "system",
                "content": configuration.translate_query_system_prompt,
            },
            {"role": "human", "content": question},
        ]
        # GLM 结构化输出偶发返回 None 或漏抽（keywords 为空但问题含英文/数字）：
        # 最多重试 1 次以降低"随机降级纯 dense"，不改变抽取语义
        has_ascii = any(c.isascii() and c.isalnum() for c in question)
        reply = None
        keywords: list[str] = []
        for _attempt in range(2):
            reply = await model.ainvoke(messages)
            if not isinstance(reply, dict):
                continue  # 返回 None：重试
            keywords = [
                t.strip() for t in (reply.get("keywords") or []) if str(t).strip()
            ]
            if keywords or not has_ascii:
                break  # 抽到词，或纯中文问题模型判无词：均视为有效结果
        if reply is not None:
            english_query = " ".join(keywords)
            # kw-v1 基线语义：抽到英文关键词即启用稀疏路（BM25 纯词匹配，用这些
            # 判别词补充稠密路的精确匹配）
            use_sparse = bool(keywords)
            if use_sparse:
                sparse_text = " ".join(keywords)
    except Exception:  # noqa: BLE001
        logger.warning("英文关键词抽取失败，关闭稀疏路仅用稠密检索", exc_info=True)

    # 2) 单 query 检索：dense=中文有效问题；sparse 仅当抽到英文关键词时启用，
    #    截断 context_top_k 条
    documents = await asyncio.to_thread(
        hybrid_search,
        question,
        final_top_k=configuration.context_top_k,
        sparse_query=sparse_text,
        enable_sparse=use_sparse,
    )
    return {
        "documents": documents,
        "english_query": english_query,
        "use_sparse": use_sparse,
    }


async def grade_documents(
        state: AgentState, *, config: RunnableConfig
    ) -> dict[str, Any]:
    """评判检索到的文档是否足以回答用户问题，并对本轮检索打数值分。

    该函数以用户最近一条提问和当前检索文档为输入，使用语言模型输出
    score（0~1 数值分，表示该组文档回答问题的充分/相关程度）、grade
    （good=足以回答 / bad=信息不足需重新检索）以及评判理由 reasoning。

    每次评判后做历轮记账：若本轮 score 高于已有最高分（best_score），快照本轮
    文档为 best_documents 作为兜底候选。累计 bad 次数达到 MAX_GRADE_ATTEMPTS
    时记录告警日志，并用历轮最高分那一轮的文档覆盖 documents（respond 将基于
    相对最好的一轮尽力作答），随后由 route_after_grade 强制进入 respond。

    Args:
        state (AgentState): agent 的当前状态，包括对话消息与检索到的文档。
        config (RunnableConfig): 包含用于评判的模型的配置。

    Returns:
        dict[str, Any]: 更新 grade（good/bad）、grade_score（0~1 数值分）、
        grade_attempts（累计 bad 次数）、grade_feedback（bad 时的评判理由）、
        best_score/best_documents（历轮最高分记账）；bad 超限时额外覆盖 documents。
    """

    class Verdict(TypedDict):
        """文档质量评判结果。"""

        grade: Literal["good", "bad"]
        score: float
        reasoning: str

    # 取最近一条人类消息作为评判针对的用户问题
    question = _latest_human_question(state)

    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model).with_structured_output(Verdict)
    system_prompt = configuration.grade_documents_system_prompt
    context = format_docs(state.documents)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "human",
            "content": f"用户问题：\n{question}\n\n检索到的文档：\n{context}",
        },
    ]
    verdict = cast(Verdict, await model.ainvoke(messages))
    if not isinstance(verdict, dict):
        # GLM 偶发漏调工具返回 None，抛错由题级重试兜底
        raise ValueError(f"文档评判输出无效: {verdict!r}")
    grade = verdict.get("grade", "bad")
    attempts = state.grade_attempts + 1 if grade == "bad" else state.grade_attempts

    # 数值分解析与规整（0~1）；模型漏输出时按 grade 保守兜底
    try:
        score = round(max(0.0, min(1.0, float(verdict.get("score")))), 4)
    except (TypeError, ValueError):
        score = 0.0 if grade == "bad" else 0.75

    update: dict[str, Any] = {
        "grade": grade,
        "grade_attempts": attempts,
        "grade_score": score,
    }
    # 历轮记账：本轮分数高于历史最高时快照本轮文档（good/bad 均记，供 bad 超限兜底）
    if score > state.best_score:
        update["best_score"] = score
        update["best_documents"] = list(state.documents)

    if grade == "bad":
        update["grade_feedback"] = verdict.get("reasoning", "")
        if attempts >= MAX_GRADE_ATTEMPTS:
            # 覆盖用最高分轮文档：本轮刷新则用 update 中的新快照，否则用历史快照
            best_docs = update.get("best_documents") or state.best_documents
            best_val = float(update.get("best_score", state.best_score))
            logger.warning(
                "grade_documents 连续 %d 次判定 bad，已达最大重试次数"
                "（MAX_GRADE_ATTEMPTS=%d），将采用历轮最高分（%.4f）那一轮的"
                "文档强制进入 respond 生成回答。最近一次评判理由：%s",
                attempts,
                MAX_GRADE_ATTEMPTS,
                best_val,
                update["grade_feedback"],
            )
            # 用历轮最高分轮的文档覆盖当前 context（无快照时保持当前轮）
            if best_docs:
                update["documents"] = list(best_docs)
    return update


def route_after_grade(
        state: AgentState,
    ) -> Literal["analyze_and_rewrite_question", "respond"]:
    """根据文档评判结果决定下一步。

    该函数检查 grade_documents 的评判结果：
        - 判 good：直接进入 respond 生成回答
        - 判 bad 且未超过最大重试次数：回到 analyze_and_rewrite_question
          重新改写/检索
        - 判 bad 但已超过最大重试次数：强制进入 respond 生成回答

    Args:
        state (AgentState): agent 的当前状态，包括文档评判结果与重试次数。

    Returns:
        Literal["analyze_and_rewrite_question", "respond"]: 下一步要执行的操作。
    """
    if state.grade == "bad" and state.grade_attempts < MAX_GRADE_ATTEMPTS:
        return "analyze_and_rewrite_question"
    return "respond"


async def _answer_with_memory_tools(
        model,
        messages: list,
        *,
        config: RunnableConfig,
    ) -> BaseMessage:
    """让回复模型在可调用长期记忆工具的情况下生成一条回复。

    以工具型 agent 的方式运行：绑定 load_user_memories / update_user_memory，
    由模型自行判断是否需要按需提取或保存长期记忆（避免无脑把长期记忆注入
    上下文）。记忆后端未启用时直接单次生成，行为与普通生成一致。

    Args:
        model: 待绑定的聊天模型（需支持 function calling）。
        messages: 已含系统提示词的对话消息列表（会被追加工具调用消息）。
        config: 运行时配置（含 user_id，供记忆工具定位命名空间）。

    Returns:
        模型生成的最终回复消息。
    """
    if not memory.is_memory_enabled():
        return await model.ainvoke(messages)

    tooled_model = model.bind_tools(
        [memory.load_user_memories, memory.update_user_memory]
    )
    memory_tools = {
        "load_user_memories": memory.load_user_memories,
        "update_user_memory": memory.update_user_memory,
    }
    response = await tooled_model.ainvoke(messages)
    for _ in range(MAX_RESPONSE_TOOL_ROUNDS):
        if not getattr(response, "tool_calls", None):
            return response
        messages.append(response)
        for tool_call in response.tool_calls:
            fn = memory_tools.get(tool_call["name"])
            if fn is None:
                output = f"未知记忆工具：{tool_call['name']}"
            else:
                output = await fn.ainvoke(dict(tool_call["args"]), config=config)
            messages.append(
                ToolMessage(content=str(output), tool_call_id=tool_call["id"])
            )
        response = await tooled_model.ainvoke(messages)
    # 连续多轮都在调用记忆工具仍未收尾：强制模型直接给出最终回答
    logger.warning(
        "回复模型经 %d 轮记忆工具调用后仍未结束，已强制模型直接生成回答",
        MAX_RESPONSE_TOOL_ROUNDS,
    )
    return await tooled_model.ainvoke(
        [*messages, SystemMessage(content="请直接给出最终回答，不要再调用工具。")]
    )


async def respond(
        state: AgentState, *, config: RunnableConfig
    ) -> dict[str, list[BaseMessage]]:
    """基于已检索到的文档生成对用户查询的最终回复。

    该函数利用对话历史与检索到的文档来组织一个全面的回答；同时以
    工具型 agent 的形式给回复模型绑定长期记忆工具（按需读取 load_user_memories、
    更新 update_user_memory），由模型自行判断是否需要提取或保存用户长期记忆，
    避免无脑将长期记忆注入上下文。记忆后端未启用时不绑定工具，行为与普通
    生成一致。

    Args:
        state (AgentState): agent 的当前状态，包括检索到的文档和对话历史。
        config (RunnableConfig): 包含用于回复的模型的配置。

    Returns:
        dict[str, list[str]]: 包含 'messages' 键的字典，值为生成的回复。
    """
    configuration = AgentConfiguration.from_runnable_config(config)
    context = format_docs(state.documents)
    prompt = configuration.response_system_prompt.format(context=context)
    messages = [{"role": "system", "content": prompt}, *state.messages]
    model = load_chat_model(configuration.response_model)
    response = await _answer_with_memory_tools(model, messages, config=config)
    return {"messages": [response]}


# 定义图
builder = StateGraph(AgentState, input=InputState, config_schema=AgentConfiguration)
builder.add_node(analyze_and_route_query)
builder.add_node(ask_for_clarification)
builder.add_node(respond_to_general_query)
builder.add_node(analyze_and_rewrite_question)
builder.add_node(retrieve_context)
builder.add_node(grade_documents)
builder.add_node(respond)

builder.add_edge(START, "analyze_and_route_query")
builder.add_conditional_edges("analyze_and_route_query", route_query)
# more-info 澄清完成后（用户已补充）回到路由节点重新分类
builder.add_edge("ask_for_clarification", "analyze_and_route_query")
# langchain 分支：意图分析（改写/直接用原问题）→ 单次检索产出最终 context
builder.add_edge("analyze_and_rewrite_question", "retrieve_context")
builder.add_edge("retrieve_context", "grade_documents")
# 评判结果：good -> respond；bad（未超限）-> 回意图分析重新改写/检索
builder.add_conditional_edges("grade_documents", route_after_grade)
builder.add_edge("respond_to_general_query", END)
builder.add_edge("respond", END)

# 编译成可以调用和部署的图对象。
# 接入 Postgres：checkpointer 提供短期记忆（线程消息历史 / interrupt 澄清中断），
# store 提供长期记忆。若 Postgres 未配置或连接失败，两者均为 None，图退化为无记忆模式。
if _ON_LANGGRAPH_PLATFORM:
    # 平台（langgraph dev / LangGraph API local_dev）运行：持久化由平台依据
    # POSTGRES_URI 自动管理，此时不传自定义 checkpointer/store（传了会被拒绝加载）。
    # 平台注入的 checkpointer 同样支撑中断澄清；长期记忆工具仍经 memory 模块
    # 直连同一 Postgres 读写，记忆功能不受影响。
    graph = builder.compile()
else:
    # 脱离平台直接运行（本地脚本 / 测试）：显式接入 Postgres 以保留记忆能力。
    # 若后续要恢复本地直连运行，只需去掉上面的平台分支即可。
    graph = builder.compile(
        checkpointer=memory.get_checkpointer(), store=memory.get_store()
    )
graph.name = "RetrievalGraph"
