"""本地 cross-encoder 重排的单元测试。"""

from unittest.mock import MagicMock

from langchain_core.documents import Document

from shared.rerank import rerank_documents


def _docs():
    return [
        Document(page_content="doc-a", metadata={"pk": 1}),
        Document(page_content="doc-b", metadata={"pk": 2}),
        Document(page_content="doc-c", metadata={"pk": 3}),
    ]


def test_rerank_reorders_and_truncates(monkeypatch):
    """按相关性分数降序重排并截断到 top_n。"""
    monkeypatch.setattr(
        "shared.rerank._score_documents", lambda query, texts: [0.5, 0.1, 0.9]
    )
    out = rerank_documents("q", _docs(), top_n=2)
    assert [d.page_content for d in out] == ["doc-c", "doc-a"]
    assert out[0].metadata["rerank_score"] == 0.9


def test_rerank_falls_back_on_error(monkeypatch):
    """推理失败时回退原序取前 top_n。"""

    def boom(query: str, texts: list[str]) -> list[float]:
        raise RuntimeError("boom")

    monkeypatch.setattr("shared.rerank._score_documents", boom)
    out = rerank_documents("q", _docs(), top_n=2)
    assert [d.page_content for d in out] == ["doc-a", "doc-b"]


def test_rerank_empty_input():
    """空文档列表直接返回空。"""
    assert rerank_documents("q", [], top_n=4) == []


def test_hybrid_search_uses_rerank(monkeypatch):
    """rerank=True 时最终结果来自 rerank_documents（离线，不触达模型）。"""
    from shared import hybrid_retrieval as hr

    cand = [
        Document(page_content=f"d{i}", metadata={"rrf_score": 1.0 - i * 0.1})
        for i in range(3)
    ]
    limits: list[int] = []
    monkeypatch.setattr(
        hr, "_rrf_candidates", lambda m, d, s, limit: limits.append(limit) or cand
    )
    monkeypatch.setattr(
        hr, "rerank_documents", lambda q, docs, top_n: docs[:top_n][::-1]
    )
    monkeypatch.setattr(hr, "make_client", MagicMock())
    monkeypatch.setattr("shared.retrieval.make_text_encoder", MagicMock())
    out = hr.hybrid_search(
        "q", rerank=True, rerank_top_n=2, rerank_candidates=3, enable_sparse=False
    )
    assert [d.page_content for d in out] == ["d1", "d0"]
    assert limits == [3]
