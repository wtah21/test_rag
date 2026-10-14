"""三层分层路由的单元测试。"""

from retrieval_graph.routing import (
    _build_thresholds,
    _rule_route,
    _semantic_route,
)

KEYWORDS = ["langchain", "langgraph", "agent", "memory", "rag"]


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


def test_semantic_route_above_threshold():
    """平均分超过阈值的标签被采用。"""
    out = _semantic_route(
        "q-langchain", EXAMPLES, THRESHOLDS, encoder=_FakeEncoder(), top_k=5
    )
    assert out == {"type": "langchain"}


def test_semantic_route_below_threshold():
    """平均分都低于阈值时返回 None。"""
    out = _semantic_route(
        "q-unknown", EXAMPLES, THRESHOLDS, encoder=_FakeEncoder(), top_k=5
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
    thr = _build_thresholds(EXAMPLES, ["无关问题"], auto_fit=True, encoder=_FakeEncoder())
    assert thr == {"langchain": 0.42, "general": 0.33}


def test_build_thresholds_no_fit_uses_encoder_default(monkeypatch):
    """fit 关闭时回退为编码器默认阈值。"""
    thr = _build_thresholds(EXAMPLES, ["无关问题"], auto_fit=False, encoder=_FakeEncoder())
    assert set(thr.keys()) == {"langchain", "general"}
