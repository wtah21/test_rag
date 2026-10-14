"""三层分层意图路由：规则匹配 → 语义路由 → JEV 决策。

级联短路：规则层命中即返回 langchain；否则进入语义层，语义层某标签
平均分超过其阈值即采用；前两层均无结论时由 JEV 决策模型兜底裁决。
只输出分类（Router.type），不产出面向用户的追问话术。
"""

from __future__ import annotations

import logging
import re

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
