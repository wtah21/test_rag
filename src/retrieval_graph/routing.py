"""三层分层意图路由：规则匹配 → 语义路由 → JEV 决策。

级联短路：规则层命中即返回 langchain；否则进入语义层，语义层某标签
平均分超过其阈值即采用；前两层均无结论时由 JEV 决策模型兜底裁决。
只输出分类（Router.type），不产出面向用户的追问话术。
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

import numpy as np

from retrieval_graph.state import Router

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


def _cosine_scores(query_vec: list[float], mat: np.ndarray) -> np.ndarray:
    """计算查询向量与矩阵各行向量的余弦相似度。"""
    q = np.asarray(query_vec, dtype=float)
    q_norm = np.linalg.norm(q) + 1e-12
    m_norm = np.linalg.norm(mat, axis=1) + 1e-12
    return (mat @ q) / (m_norm * q_norm)


def _semantic_route(
    question: str,
    examples: dict[str, list[str]],
    thresholds: dict[str, float],
    *,
    encoder,
    top_k: int = 5,
) -> Router | None:
    """第二层语义路由：检索 top_k 示例问题，按标签求平均分，超阈值则采用。

    Args:
        question: 用户原始提问。
        examples: 各标签的示例问题。
        thresholds: 各标签的阈值（由 fit 得到）。
        encoder: 可将 List[str] 编码为向量的对象（BGE-M3）。
        top_k: 检索的示例问题条数。

    Returns:
        平均分最高且超过该标签阈值时返回 Router(type=该标签)，否则 None。
    """
    pairs = [(label, u) for label, utts in examples.items() for u in utts]
    if not pairs or not question:
        return None

    utt_vecs = np.asarray(encoder([u for _, u in pairs]), dtype=float)
    q_vec = encoder([question])[0]
    sims = _cosine_scores(q_vec, utt_vecs)

    order = np.argsort(-sims, kind="stable")[:top_k]
    bucket: dict[str, list[float]] = {}
    for i in order:
        label = pairs[int(i)][0]
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
        return Router(type=best_label)
    return None


@lru_cache(maxsize=1)
def _get_bge_m3_encoder():
    """构造（并缓存）BGE-M3 稠密编码器（适配 semantic-router 的 DenseEncoder）。"""
    from semantic_router.encoders import DenseEncoder

    from shared.retrieval import make_text_encoder

    base = make_text_encoder("siliconflow/BAAI/bge-m3")

    class _BgeM3Encoder(DenseEncoder):
        name: str = "siliconflow-bge-m3"

        def __call__(self, docs):  # noqa: ANN001, ANN204
            return base.embed_documents(list(docs))

    return _BgeM3Encoder()


@lru_cache(maxsize=4)
def _get_semantic_router(examples_key: tuple):
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

    routes = [
        Route(name=label, utterances=list(utts)) for label, utts in examples_key
    ]
    return SemanticRouter(
        encoder=_get_bge_m3_encoder(), routes=routes, auto_sync="local"
    )


def _build_thresholds(
    examples: dict[str, list[str]],
    negatives: list[str],
    *,
    auto_fit: bool,
    encoder,
) -> dict[str, float]:
    """获取各标签的阈值：auto_fit 时用 semantic-router 的 fit 自动调优。

    Args:
        examples: 各标签的示例问题。
        negatives: 不属于任何标签的负样本问题（y 记为 None）。
        auto_fit: 是否用 fit 调优阈值。
        encoder: DenseEncoder 实例（关闭 fit 时取其默认阈值）。

    Returns:
        {label: threshold}。
    """
    if not auto_fit:
        default = float(getattr(encoder, "score_threshold", 0.5))
        return {label: default for label in examples}

    examples_key = tuple((label, tuple(utts)) for label, utts in examples.items())
    sr = _get_semantic_router(examples_key)
    X = [u for utts in examples.values() for u in utts] + list(negatives)
    y = [label for label, utts in examples.items() for _ in utts] + [None] * len(negatives)
    sr.fit(X=X, y=y)
    return {str(k): float(v) for k, v in sr.get_thresholds().items()}
