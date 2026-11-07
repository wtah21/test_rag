"""三层分层意图路由：规则匹配 → 语义路由 → JEV 决策。

级联短路：规则层命中即返回 langchain；否则进入语义层，语义层某标签
平均分超过其阈值即采用；前两层均无结论时由 JEV 决策模型兜底裁决——
按 JEV 决策协议发送 payload（state + 带类型的 questions），直接消费其
结构化 answers（choice 原语），不解析模型生成的文本。
只输出分类（Router.type），不产出面向用户的追问话术。
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import requests
from numpy.typing import NDArray

from retrieval_graph.state import Router

if TYPE_CHECKING:
    from semantic_router.encoders import DenseEncoder

logger = logging.getLogger(__name__)


def _rule_route(question: str, keywords: list[str]) -> Router | None:
    """第一层规则路由：问题含关键词则直接归为 langchain。

    以「非字母数字边界」匹配关键词，避免 agent 误命中 management。

    Args:
        question: 用户原始提问。
        keywords: 配置的匹配关键词列表。

    Returns:
        命中时返回 Router(type="langchain")，否则返回 None。
    """
    text = (question or "").lower()
    if not text:
        return None
    for kw in keywords:
        kw_l = kw.lower().strip()
        if not kw_l:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(kw_l)}(?![a-z0-9])", text):
            logger.info("规则路由命中关键词 %r", kw)
            return Router(type="langchain")
    return None


def _cosine_scores(query_vec: list[float], mat: NDArray[Any]) -> NDArray[Any]:
    """计算查询向量与矩阵各行向量的余弦相似度。"""
    q = np.asarray(query_vec, dtype=float)
    q_norm = np.linalg.norm(q) + 1e-12
    m_norm = np.linalg.norm(mat, axis=1) + 1e-12
    return cast(NDArray[Any], (mat @ q) / (m_norm * q_norm))


def _semantic_match(
    question: str,
    labels: tuple[str, ...],
    utt_vecs: NDArray[Any],
    thresholds: dict[str, float],
    *,
    encoder: DenseEncoder,
    top_k: int = 5,
) -> Router | None:
    """第二层语义匹配：检索 top_k 示例问题，按标签求平均分，超阈值则采用。

    示例问题的向量由调用方预先算好（见 _get_example_vectors 的模块级缓存），
    这里只对 query 编码一次。

    Args:
        question: 用户原始提问。
        labels: 与 utt_vecs 逐行对应的示例问题标签。
        utt_vecs: 示例问题的向量矩阵（行数 == len(labels)）。
        thresholds: 各标签的阈值（由 fit 得到）。
        encoder: 可将 List[str] 编码为向量的对象（BGE-M3）。
        top_k: 检索的示例问题条数。

    Returns:
        平均分最高且超过该标签阈值时返回 Router(type=该标签)，否则 None。
    """
    if not labels or not question:
        return None

    q_vec = encoder([question])[0]
    sims = _cosine_scores(q_vec, utt_vecs)

    order = np.argsort(-sims, kind="stable")[:top_k]
    bucket: dict[str, list[float]] = {}
    for i in order:
        label = labels[int(i)]
        bucket.setdefault(label, []).append(float(sims[int(i)]))

    avg = {label: sum(v) / len(v) for label, v in bucket.items()}
    if not avg:
        return None
    best_label = max(avg, key=lambda k: avg[k])
    thr = float(thresholds.get(best_label, 1.0))
    logger.info(
        "语义路由平均分 %s，最高标签=%s(%.4f) 阈值=%.4f",
        {k: round(v, 4) for k, v in avg.items()},
        best_label,
        avg[best_label],
        thr,
    )
    if avg[best_label] >= thr:
        route_type = cast(Literal["more-info", "langchain", "general"], best_label)
        return Router(type=route_type)
    return None


@lru_cache(maxsize=1)
def _get_bge_m3_encoder() -> DenseEncoder:
    """构造（并缓存）BGE-M3 稠密编码器（适配 semantic-router 的 DenseEncoder）。"""
    from semantic_router.encoders import DenseEncoder

    from shared.retrieval import make_text_encoder

    base = make_text_encoder("siliconflow/BAAI/bge-m3")

    class _BgeM3Encoder(DenseEncoder):
        name: str = "siliconflow-bge-m3"

        def __call__(self, docs: list[str]) -> list[list[float]]:
            return base.embed_documents(list(docs))

    return _BgeM3Encoder()


@lru_cache(maxsize=4)
def _get_semantic_router(
    examples_key: tuple[tuple[str, tuple[str, ...]], ...],
) -> Any:
    """构建（并缓存）semantic-router 实例（本地内存索引）。

    仅以可哈希的 examples_key 作为缓存键；编码器经 _get_bge_m3_encoder()
    取模块级单例（pydantic 编码器不可哈希，不能作为缓存键）。

    Args:
        examples_key: ((label, (utterance, ...)), ...) 形式的示例问题。

    Returns:
        已完成示例问题索引的 SemanticRouter。
    """
    from semantic_router import Route
    from semantic_router.routers import SemanticRouter

    routes = [Route(name=label, utterances=list(utts)) for label, utts in examples_key]
    return SemanticRouter(
        encoder=_get_bge_m3_encoder(), routes=routes, auto_sync="local"
    )


def _build_thresholds(
    examples: dict[str, list[str]],
    negatives: list[str],
    *,
    auto_fit: bool,
    encoder: DenseEncoder | None = None,
) -> dict[str, float]:
    """获取各标签的阈值：auto_fit 时用 semantic-router 的 fit 自动调优。

    Args:
        examples: 各标签的示例问题。
        negatives: 不属于任何标签的负样本问题（y 记为 None）。
        auto_fit: 是否用 fit 调优阈值。
        encoder: DenseEncoder 实例（关闭 fit 时取其默认阈值；auto_fit=False 时必须传入，auto_fit=True 时不使用、可省略）。

    Returns:
        {label: threshold}。
    """
    if not auto_fit:
        default = float(getattr(encoder, "score_threshold", None) or 0.5)
        return {label: default for label in examples}

    examples_key = tuple((label, tuple(utts)) for label, utts in examples.items())
    sr = _get_semantic_router(examples_key)
    X = [u for utts in examples.values() for u in utts] + list(negatives)
    y = [label for label, utts in examples.items() for _ in utts] + [None] * len(
        negatives
    )
    sr.fit(X=X, y=y)
    return {str(k): float(v) for k, v in sr.get_thresholds().items()}


def _examples_key(
    examples: dict[str, list[str]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """把示例问题转为可哈希的缓存键。"""
    return tuple((label, tuple(utts)) for label, utts in examples.items())


@lru_cache(maxsize=8)
def _get_thresholds(
    examples_key: tuple[tuple[str, tuple[str, ...]], ...],
    negatives_key: tuple[str, ...],
    auto_fit: bool,
) -> dict[str, float]:
    """构建（并缓存）各标签阈值：模块级单例，仅首次调用执行 fit。"""
    examples = {label: list(utts) for label, utts in examples_key}
    return _build_thresholds(
        examples,
        list(negatives_key),
        auto_fit=auto_fit,
        encoder=None if auto_fit else _get_bge_m3_encoder(),
    )


@lru_cache(maxsize=8)
def _get_example_vectors(
    examples_key: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[tuple[str, ...], NDArray[Any]]:
    """构建（并缓存）示例问题的标签与向量矩阵：模块级单例，仅首次编码。

    Args:
        examples_key: ((label, (utterance, ...)), ...) 形式的示例问题。

    Returns:
        (labels, utt_vecs)：与示例问题逐行对应的标签元组与向量矩阵。
    """
    pairs = [(label, u) for label, utts in examples_key for u in utts]
    if not pairs:
        return (), np.zeros((0, 0))
    vecs = np.asarray(_get_bge_m3_encoder()([u for _, u in pairs]), dtype=float)
    return tuple(label for label, _ in pairs), vecs


# JEV choice 问题的选项语义（同时作为合法分类的取值集合）
_JEV_ROUTE_CRITERIA = {
    "langchain": "需要查阅 LangChain、LangGraph、LangSmith 及其生态（集成、检索、"
    "智能体、记忆、工具等）的技术资料才能准确回答的问题。",
    "more-info": "仅凭当前对话上下文不足以开始解答（如只描述了报错但没有报错文本/"
    "复现步骤，或问题表述有歧义、缺少关键前提）。",
    "general": "与上述生态无关的一般性问题、寒暄或闲聊。",
}
# 承载分类结论的 choice 问题名，响应中按此名索引 answers
_JEV_QUESTION_ID = "intent_category"


def _post_jev(payload: dict[str, Any]) -> dict[str, Any]:
    """向 JEV 决策端点发送 payload 并返回原始 JSON。

    独立成函数以便测试中 mock。端点默认取 TypeSafe 官方 System One 端点
    （https://api.typesafe.ai/v1/systemone），可用 JEV_BASE_URL /
    JEV_DECIDE_PATH 覆盖；模型名用 JEV_MODEL 覆盖。
    """
    api_key = os.environ.get("JEV_API_KEY", "")
    if not api_key:
        raise ValueError("缺少 JEV_API_KEY，无法调用 JEV 决策模型")
    base_url = os.environ.get("JEV_BASE_URL", "https://api.typesafe.ai/v1").rstrip("/")
    path = os.environ.get("JEV_DECIDE_PATH", "/systemone")
    resp = requests.post(
        f"{base_url}{path}",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    return cast(dict[str, Any], resp.json())


async def _jev_route(question: str, history: str, instructions: str) -> Router:
    """第三层 JEV 决策：前两层无结论时由决策模型兜底分类。

    按 JEV 决策协议调用：payload 含 state（对话上下文 + 最新提问）与
    questions（一个 choice 问题，criteria 为三类别的语义说明），
    响应即结构化 answers，直接取 answers[intent_category].choice，
    无需解析生成文本。调用失败或返回非法选项时兜底 general。

    Args:
        question: 用户原始提问。
        history: 最近若干轮对话的纯文本摘要（可为空）。
        instructions: choice 问题的判定指令（对应 router_system_prompt）。

    Returns:
        Router(type=...)；调用失败或选项非法时兜底 general。
    """
    payload: dict[str, Any] = {
        "model": os.environ.get("JEV_MODEL", "jev-latest"),
        "state": f"对话上下文：\n{history}\n\n用户最新提问：\n{question}",
        "questions": {
            _JEV_QUESTION_ID: {
                "type": "choice",
                "instructions": instructions,
                "criteria": _JEV_ROUTE_CRITERIA,
            }
        },
    }
    try:
        data = await asyncio.to_thread(_post_jev, payload)
        answer = (data.get("answers") or {}).get(_JEV_QUESTION_ID) or {}
        choice = str(answer.get("choice", "")).strip()
    except Exception:  # noqa: BLE001
        logger.warning("JEV 决策模型调用失败，兜底 general", exc_info=True)
        return Router(type="general")
    if choice in _JEV_ROUTE_CRITERIA:
        return Router(type=cast(Literal["more-info", "langchain", "general"], choice))
    logger.warning("JEV 返回未知类别 %r，兜底 general", choice)
    return Router(type="general")


def _semantic_route(
    question: str,
    *,
    examples_key: tuple[tuple[str, tuple[str, ...]], ...],
    negatives: list[str],
    top_k: int,
    auto_fit: bool,
) -> Router | None:
    """同步执行第二层语义路由：阈值构建 + 示例编码 + 匹配。

    整条链路都是同步阻塞的（BGE-M3 走网络编码、semantic-router 的 fit 与
    本地索引读写，后者会触碰文件系统），故独立成同步函数，由 resolve_route
    经 asyncio.to_thread 放进线程池执行，避免阻塞 ASGI 事件循环。
    """
    encoder = _get_bge_m3_encoder()
    thresholds = _get_thresholds(examples_key, tuple(negatives), auto_fit)
    labels, utt_vecs = _get_example_vectors(examples_key)
    return _semantic_match(
        question, labels, utt_vecs, thresholds, encoder=encoder, top_k=top_k
    )


async def resolve_route(
    question: str,
    history: str,
    *,
    examples: dict[str, list[str]],
    negatives: list[str],
    top_k: int,
    auto_fit: bool,
    keywords: list[str],
    jev_prompt: str,
) -> tuple[Router, str]:
    """级联执行三层路由，返回 (Router, 命中层)。

    编码器统一取模块级单例 _get_bge_m3_encoder()（示例向量与阈值需按
    examples_key 缓存，编码器不可哈希、不能作为缓存键）。
    规则层为纯正则判断，直接在事件循环内执行；语义层与 JEV 层均为同步阻塞
    调用（网络请求 / 本地索引 IO），统一经 asyncio.to_thread 放进线程池。
    命中层取值：rule / semantic / jev，便于日志与评估。
    """
    hit = _rule_route(question, keywords)
    if hit is not None:
        return hit, "rule"

    if examples:
        sem = await asyncio.to_thread(
            _semantic_route,
            question,
            examples_key=_examples_key(examples),
            negatives=negatives,
            top_k=top_k,
            auto_fit=auto_fit,
        )
        if sem is not None:
            return sem, "semantic"

    return await _jev_route(question, history, jev_prompt), "jev"
