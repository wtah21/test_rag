"""智谱 cross-encoder 重排客户端。

调用智谱 Rerank API（OpenAI 兼容的 REST 端点）对候选文档按与 query 的
相关性重排，供混合检索 RRF 融合之后精排使用。鉴权复用 GLM_API_KEY。
"""

from __future__ import annotations

import logging
import os

import requests
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

ZHIPU_RERANK_URL = os.environ.get(
    "ZHIPU_RERANK_URL", "https://open.bigmodel.cn/api/paas/v4/rerank"
)
RERANK_MODEL = os.environ.get("ZHIPU_RERANK_MODEL", "rerank")


def _post_rerank(query: str, documents: list[str], top_n: int) -> dict:
    """调用智谱 Rerank API 并返回原始 JSON。

    独立成函数以便测试中 mock。
    """
    api_key = os.environ.get("GLM_API_KEY", "")
    if not api_key:
        raise ValueError("缺少 GLM_API_KEY，无法调用智谱重排接口")
    resp = requests.post(
        ZHIPU_RERANK_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": RERANK_MODEL,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def zhipu_rerank(
    query: str, documents: list[Document], top_n: int = 4
) -> list[Document]:
    """用智谱 cross-encoder 重排候选文档，返回前 top_n 条。

    失败（网络/额度/超时/返回异常）时回退为原顺序取前 top_n 条，
    保证检索链路不因重排失败而中断。

    Args:
        query: 用于计算相关性的查询文本。
        documents: RRF 融合后的候选文档（按 rrf_score 降序）。
        top_n: 重排后保留的文档数。

    Returns:
        重排后的前 top_n 条文档；失败时返回原序前 top_n 条。
    """
    if not documents:
        return []
    if not query:
        return documents[:top_n]

    try:
        data = _post_rerank(query, [d.page_content for d in documents], top_n)
        results = data.get("results") or []
        if not results:
            return documents[:top_n]
        ordered: list[Document] = []
        for item in sorted(
            results, key=lambda r: r.get("relevance_score", 0.0), reverse=True
        ):
            idx = int(item["index"])
            if 0 <= idx < len(documents):
                doc = documents[idx]
                metadata = dict(doc.metadata or {})
                metadata["rerank_score"] = round(
                    float(item.get("relevance_score", 0.0)), 4
                )
                ordered.append(
                    Document(page_content=doc.page_content, metadata=metadata)
                )
        return ordered[:top_n]
    except Exception:  # noqa: BLE001
        logger.warning("智谱重排失败，回退 RRF 原序取前 %d 条", top_n, exc_info=True)
        return documents[:top_n]
