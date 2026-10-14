"""三层分层路由的单元测试。"""

from retrieval_graph.routing import _rule_route

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
