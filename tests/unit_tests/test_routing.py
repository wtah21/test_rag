"""三层分层路由的单元测试。"""

import asyncio

import numpy as np
import pytest

from retrieval_graph import routing
from retrieval_graph.routing import (
    _build_thresholds,
    _rule_route,
    _semantic_match,
)

KEYWORDS = ["langchain", "langgraph", "agent", "memory", "rag"]


@pytest.fixture(autouse=True)
def _clear_routing_caches():
    """隔离模块级 lru_cache，避免跨用例污染。"""
    routing._get_thresholds.cache_clear()
    routing._get_example_vectors.cache_clear()
    routing._get_semantic_router.cache_clear()
    yield
    routing._get_thresholds.cache_clear()
    routing._get_example_vectors.cache_clear()
    routing._get_semantic_router.cache_clear()


def test_rule_route_hits_keyword():
    assert _rule_route("怎么用 LangGraph 建图？", KEYWORDS) == {"type": "langchain"}


def test_rule_route_is_case_insensitive():
    assert _rule_route("LANGCHAIN 是什么", KEYWORDS) == {"type": "langchain"}


def test_rule_route_respects_word_boundary():
    # management 内部含 agent，但不应命中
    assert _rule_route("我想了解 management 培训", KEYWORDS) is None


def test_rule_route_no_hit():
    assert _rule_route("帮我写一首诗", KEYWORDS) is None


class _FakeEncoder:
    """把文本映射到 2 维向量的假编码器（无需网络）。"""

    def __call__(self, docs):
        table = {
            "langchain-utter": [1.0, 0.0],
            "general-utter": [0.0, 1.0],
            "q-langchain": [0.9, 0.1],
            "q-unknown": [0.6, 0.8],
        }
        return [table.get(d, [0.5, 0.5]) for d in docs]


EXAMPLES = {
    "langchain": ["langchain-utter"],
    "general": ["general-utter"],
}
THRESHOLDS = {"langchain": 0.8, "general": 0.8}


def test_semantic_match_above_threshold():
    """平均分超过阈值的标签被采用。"""
    out = _semantic_match(
        "q-langchain",
        ("langchain", "general"),
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        THRESHOLDS,
        encoder=_FakeEncoder(),
        top_k=5,
    )
    assert out == {"type": "langchain"}


def test_semantic_match_below_threshold():
    """平均分都低于阈值时返回 None。"""
    out = _semantic_match(
        "q-unknown",
        ("langchain", "general"),
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        THRESHOLDS,
        encoder=_FakeEncoder(),
        top_k=5,
    )
    assert out is None


def test_build_thresholds_uses_fit(monkeypatch):
    """fit 开启时返回 semantic-router 的 get_thresholds 结果。"""

    class _FakeSR:
        def fit(self, X, y):
            return None

        def get_thresholds(self):
            return {"langchain": 0.42, "general": 0.33}

    monkeypatch.setattr(
        "retrieval_graph.routing._get_semantic_router", lambda *a, **k: _FakeSR()
    )
    thr = _build_thresholds(
        EXAMPLES, ["无关问题"], auto_fit=True, encoder=_FakeEncoder()
    )
    assert thr == {"langchain": 0.42, "general": 0.33}


def test_build_thresholds_no_fit_uses_encoder_default(monkeypatch):
    """fit 关闭时回退为编码器默认阈值。"""
    thr = _build_thresholds(
        EXAMPLES, ["无关问题"], auto_fit=False, encoder=_FakeEncoder()
    )
    assert set(thr.keys()) == {"langchain", "general"}


class _NoneThresholdEncoder:
    """score_threshold 为 None 的真实编码器形态。"""

    score_threshold = None

    def __call__(self, docs):
        return [[1.0, 0.0] for _ in docs]


def test_build_thresholds_no_fit_handles_none_threshold():
    """编码器 score_threshold 为 None 时回退 0.5，而不是抛 TypeError。"""
    thr = _build_thresholds(
        EXAMPLES, ["无关问题"], auto_fit=False, encoder=_NoneThresholdEncoder()
    )
    assert thr == {"langchain": 0.5, "general": 0.5}


def test_jev_route_sends_payload_and_reads_structured_choice(monkeypatch):
    """按 JEV 决策协议发送 payload，并直接消费结构化 answers.choice。"""
    captured = {}

    def _fake_post(payload):
        captured.update(payload)
        return {
            "model": "jev-1.13.0",
            "answers": {
                "intent_category": {
                    "type": "choice",
                    "choice": "langchain",
                    "confidence": 0.91,
                    "probabilities": {"langchain": 0.91, "general": 0.09},
                }
            },
        }

    monkeypatch.setattr(routing, "_post_jev", _fake_post)
    out = asyncio.run(routing._jev_route("怎么建图", "user: 你好", "判断类别"))
    assert out == {"type": "langchain"}
    question = captured["questions"]["intent_category"]
    assert question["type"] == "choice"
    assert question["instructions"] == "判断类别"
    assert set(question["criteria"]) == {"langchain", "more-info", "general"}
    assert "怎么建图" in captured["state"]
    assert "你好" in captured["state"]
    assert captured["model"]


def test_jev_route_call_failure_falls_back_general(monkeypatch):
    """JEV 调用失败时兜底 general。"""

    def _boom(payload):
        raise RuntimeError("网络错误")

    monkeypatch.setattr(routing, "_post_jev", _boom)
    out = asyncio.run(routing._jev_route("x", "", "p"))
    assert out == {"type": "general"}


def test_jev_route_unknown_choice_falls_back_general(monkeypatch):
    """JEV 返回未知类别时兜底 general。"""
    monkeypatch.setattr(
        routing,
        "_post_jev",
        lambda payload: {"answers": {"intent_category": {"choice": "unknown"}}},
    )
    out = asyncio.run(routing._jev_route("x", "", "p"))
    assert out == {"type": "general"}


def test_post_jev_requires_api_key(monkeypatch):
    """缺少 JEV_API_KEY 时明确报错，不发出请求。"""
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    with pytest.raises(ValueError):
        routing._post_jev({"model": "m", "state": "s", "questions": {}})


def test_post_jev_targets_systemone_endpoint(monkeypatch):
    """默认打到 TypeSafe 官方 System One 端点，并以 Bearer 鉴权。"""
    captured = {}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"answers": {}}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured.update({"url": url, "headers": headers})
        return _Resp()

    monkeypatch.setenv("JEV_API_KEY", "k")
    monkeypatch.delenv("JEV_BASE_URL", raising=False)
    monkeypatch.delenv("JEV_DECIDE_PATH", raising=False)
    monkeypatch.setattr(routing.requests, "post", _fake_post)
    routing._post_jev({"model": "jev-latest", "state": "s", "questions": {}})
    assert captured["url"] == "https://api.typesafe.ai/v1/systemone"
    assert captured["headers"]["Authorization"] == "Bearer k"


def test_resolve_route_rule_short_circuits(monkeypatch):
    """规则命中时不再调用语义层与 JEV。"""
    called = {"semantic": False, "jev": False}

    def _fake_semantic(*a, **k):
        called["semantic"] = True
        return None

    async def _fake_jev(*a, **k):
        called["jev"] = True
        return {"type": "general"}

    monkeypatch.setattr(routing, "_semantic_match", _fake_semantic)
    monkeypatch.setattr(routing, "_jev_route", _fake_jev)
    out = asyncio.run(
        routing.resolve_route(
            "怎么用 LangGraph 建图？",
            "",
            examples={},
            negatives=[],
            top_k=5,
            auto_fit=False,
            keywords=KEYWORDS,
            jev_prompt="p",
        )
    )
    assert out == ({"type": "langchain"}, "rule")
    assert called == {"semantic": False, "jev": False}


def test_resolve_route_falls_back_to_jev(monkeypatch):
    """规则与语义都无结论时调用 JEV。"""
    monkeypatch.setattr(routing, "_semantic_match", lambda *a, **k: None)

    async def _fake_jev(*a, **k):
        return {"type": "more-info"}

    monkeypatch.setattr(routing, "_jev_route", _fake_jev)
    out = asyncio.run(
        routing.resolve_route(
            "这个不行",
            "",
            examples={},
            negatives=[],
            top_k=5,
            auto_fit=False,
            keywords=KEYWORDS,
            jev_prompt="p",
        )
    )
    assert out == ({"type": "more-info"}, "jev")


def test_get_thresholds_cached(monkeypatch):
    """阈值（fit）只构建一次，第二次直接复用缓存。"""
    calls = {"fit": 0}

    class _FakeSR:
        def fit(self, X, y):
            calls["fit"] += 1

        def get_thresholds(self):
            return {"langchain": 0.2, "general": 0.2}

    monkeypatch.setattr(routing, "_get_semantic_router", lambda *a, **k: _FakeSR())
    key = (("langchain", ("langchain-utter",)), ("general", ("general-utter",)))
    assert routing._get_thresholds(key, ("n",), True) == {
        "langchain": 0.2,
        "general": 0.2,
    }
    assert routing._get_thresholds(key, ("n",), True) == {
        "langchain": 0.2,
        "general": 0.2,
    }
    assert calls["fit"] == 1


def test_get_example_vectors_cached(monkeypatch):
    """示例问题只在首次编码，后续复用缓存向量。"""
    calls = {"n": 0}

    class _CountingEncoder:
        def __call__(self, docs):
            calls["n"] += 1
            return [[1.0, 0.0] for _ in docs]

    monkeypatch.setattr(routing, "_get_bge_m3_encoder", lambda: _CountingEncoder())
    key = (("langchain", ("langchain-utter",)), ("general", ("general-utter",)))
    routing._get_example_vectors(key)
    routing._get_example_vectors(key)
    assert calls["n"] == 1


def test_resolve_route_uses_cached_semantic_artifacts(monkeypatch):
    """examples 非空时走语义层；阈值（fit）与示例编码只构建一次。"""
    calls = {"fit": 0, "encode": 0}

    class _FakeSR:
        def fit(self, X, y):
            calls["fit"] += 1

        def get_thresholds(self):
            return {"langchain": 0.5, "general": 0.5}

    class _CountingEncoder:
        def __call__(self, docs):
            calls["encode"] += 1
            return [[1.0, 0.0] if "lang" in d else [0.0, 1.0] for d in docs]

    monkeypatch.setattr(routing, "_get_semantic_router", lambda *a, **k: _FakeSR())
    monkeypatch.setattr(routing, "_get_bge_m3_encoder", lambda: _CountingEncoder())

    async def _fake_jev(*a, **k):
        return {"type": "general"}

    monkeypatch.setattr(routing, "_jev_route", _fake_jev)
    for _ in range(2):
        out = asyncio.run(
            routing.resolve_route(
                "q-langchain",
                "",
                examples=EXAMPLES,
                negatives=["n"],
                top_k=5,
                auto_fit=True,
                keywords=[],
                jev_prompt="p",
            )
        )
        assert out == ({"type": "langchain"}, "semantic")
    assert calls["fit"] == 1
    assert calls["encode"] == 3  # 1 次建示例索引 + 2 次 query 编码
