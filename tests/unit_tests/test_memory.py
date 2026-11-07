"""retrieval_graph.memory 的单元测试。

重点覆盖：最近两轮对话的整理（供 extractor 观察）；update_user_memory 不再要求
回复模型提供内容、而是经 InjectedState 取对话交给 extractor 的新语义；三类记忆
（fact / preference / status）的权重计算、Similarity × Weight 重排与访问计数。
测试全程离线：store 与 extractor 均以桩替代。
"""

import asyncio
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from retrieval_graph import memory

NOW = datetime(2026, 10, 15, tzinfo=timezone.utc)


@dataclass
class FakeItem:
    """桩 store 的检索结果条目。"""

    key: str
    value: dict
    score: float | None = None


class FakeStore:
    """内存版 store 桩：search 记录检索参数并返回预设条目，put 记录写入。"""

    def __init__(self, items=None):
        self.items = list(items or [])
        self.puts = []
        self.put_indexes = []
        self.searched_namespace = None
        self.searched_query = None
        self.searched_limit = None

    def search(self, namespace, *, query=None, limit=None):
        self.searched_namespace = namespace
        self.searched_query = query
        self.searched_limit = limit
        return self.items

    def put(self, namespace, key, value, index=None):
        self.puts.append((namespace, key, value))
        self.put_indexes.append(index)


class FakeExtractor:
    """桩 extractor：记录收到的 payload，返回预设记忆条目。"""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    async def ainvoke(self, payload):
        self.calls.append(payload)
        return {"responses": self.responses}


def _tool_call_message() -> AIMessage:
    """构造一条带工具调用、无文本内容的助手消息。"""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "update_user_memory",
                "args": {"question": "用户在做的事"},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )


def test_format_recent_dialogue_keeps_last_two_rounds():
    messages = [
        HumanMessage(content="第一轮提问"),
        AIMessage(content="第一轮回答"),
        HumanMessage(content="第二轮提问"),
        AIMessage(content="第二轮回答"),
        HumanMessage(content="第三轮提问"),
    ]
    assert memory.format_recent_dialogue(messages).splitlines() == [
        "用户：第二轮提问",
        "助手：第二轮回答",
        "用户：第三轮提问",
    ]


def test_format_recent_dialogue_skips_tool_and_empty_messages():
    messages = [
        HumanMessage(content="我叫小明"),
        _tool_call_message(),
        ToolMessage(content="已更新长期记忆", tool_call_id="call_1"),
        AIMessage(content="好的，小明。"),
        HumanMessage(content="  "),
    ]
    assert memory.format_recent_dialogue(messages).splitlines() == [
        "用户：我叫小明",
        "助手：好的，小明。",
    ]


def test_format_recent_dialogue_returns_empty_without_user_turn():
    assert memory.format_recent_dialogue([]) == ""
    assert memory.format_recent_dialogue([SystemMessage(content="系统提示")]) == ""


def test_update_user_memory_exposes_only_question():
    """模型可见入参只有 question：内容由 extractor 提炼，state 由系统注入。"""
    schema = memory.update_user_memory.tool_call_schema.model_json_schema()
    assert list(schema["properties"]) == ["question"]
    assert schema["required"] == ["question"]


def test_build_index_config_uses_bge_m3_1024_dims(monkeypatch):
    from langchain_core.embeddings import Embeddings

    monkeypatch.setenv("SiliconEmbedding_API_KEY", "test-key")
    config = memory.build_index_config()
    assert config["dims"] == 1024
    assert config["fields"] == ["content"]
    assert isinstance(config["embed"], Embeddings)


def test_load_user_memories_uses_question_as_vector_query(monkeypatch):
    store = FakeStore(
        items=[
            FakeItem(
                key="k1", value={"kind": "fact", "content": "用户叫小明"}, score=0.8
            )
        ]
    )
    monkeypatch.setattr(memory, "get_store", lambda: store)

    out = asyncio.run(
        memory.load_user_memories.ainvoke(
            {"question": "用户称呼"}, config={"configurable": {"user_id": "u1"}}
        )
    )

    assert "用户叫小明" in out
    assert store.searched_namespace == ("memories", "u1")
    assert store.searched_query == "用户称呼"
    assert store.searched_limit == memory.MEMORY_CANDIDATE_LIMIT


def test_search_memories_falls_back_without_query(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(memory, "get_store", lambda: store)

    asyncio.run(memory.load_user_memories.ainvoke({"question": "   "}, config={}))

    assert store.searched_query is None
    assert store.searched_limit == memory.MAX_MEMORIES


def test_load_user_memories_reports_search_failure(monkeypatch):
    """检索（含嵌入调用）失败时按提示语降级，不打断回复生成。"""

    class BrokenStore:
        def search(self, namespace, *, query=None, limit=None):
            raise RuntimeError("嵌入服务不可用")

    monkeypatch.setattr(memory, "get_store", lambda: BrokenStore())

    out = asyncio.run(memory.load_user_memories.ainvoke({"question": "用户称呼"}))

    assert "检索失败" in out


def test_update_user_memory_extracts_from_recent_dialogue(monkeypatch):
    store = FakeStore(
        items=[FakeItem(key="k1", value={"kind": "fact", "content": "用户叫小明"})]
    )
    extractor = FakeExtractor(
        [memory.MemoryItem(kind="preference", content="用户偏好简短回答")]
    )
    monkeypatch.setattr(memory, "get_store", lambda: store)
    monkeypatch.setattr(memory, "make_extractor", lambda *a, **k: extractor)

    state = {
        "messages": [
            HumanMessage(content="我叫小明"),
            AIMessage(content="你好，小明"),
            HumanMessage(content="以后回答请简短一点"),
        ]
    }
    out = asyncio.run(
        memory.update_user_memory.ainvoke(
            {"question": "用户回答偏好", "state": state},
            config={"configurable": {"user_id": "u1"}},
        )
    )

    assert "用户偏好简短回答" in out
    assert store.searched_namespace == ("memories", "u1")
    assert store.searched_query == "用户回答偏好"
    assert store.searched_limit == memory.MEMORY_CANDIDATE_LIMIT
    assert len(store.puts) == 1
    namespace, key, value = store.puts[0]
    assert namespace == ("memories", "u1")
    assert key
    assert value["kind"] == "preference"
    assert value["content"] == "用户偏好简短回答"
    # 新增记忆：访问次数从 0 起，权重为 1（exp(0) + 0.1·ln(1)）
    assert value["access_count"] == 0
    assert value["weight"] == 1.0
    assert memory.parse_timestamp(value["last_accessed_at"])
    # 写入新记忆时正文变了，需要重新嵌入（index 为 None）
    assert store.put_indexes == [None]

    payload = extractor.calls[0]
    prompt_text = payload["messages"][1].content
    assert "我叫小明" in prompt_text
    assert "以后回答请简短一点" in prompt_text
    assert "用户叫小明" in prompt_text
    assert payload["existing"] == [
        (
            "k1",
            memory.MemoryItem(kind="fact", content="用户叫小明", memory_key="k1"),
        )
    ]


def test_update_user_memory_without_dialogue_is_noop(monkeypatch):
    store = FakeStore()
    extractor = FakeExtractor([memory.MemoryItem(kind="fact", content="不该出现")])
    monkeypatch.setattr(memory, "get_store", lambda: store)
    monkeypatch.setattr(memory, "make_extractor", lambda *a, **k: extractor)

    out = asyncio.run(
        memory.update_user_memory.ainvoke(
            {"question": "用户信息", "state": {"messages": []}}
        )
    )

    assert "未做修改" in out
    assert store.puts == []
    assert extractor.calls == []


def test_update_user_memory_reports_unavailable_store(monkeypatch):
    monkeypatch.setattr(memory, "get_store", lambda: None)
    out = asyncio.run(
        memory.update_user_memory.ainvoke(
            {
                "question": "用户信息",
                "state": {"messages": [HumanMessage(content="我叫小明")]},
            }
        )
    )
    assert "不可用" in out


def test_answer_with_memory_tools_injects_state_into_tool(monkeypatch):
    """回复模型的 tool_call 参数不含 state，图侧必须显式注入最近两轮对话。"""
    from retrieval_graph.graph import _answer_with_memory_tools
    from retrieval_graph.state import AgentState

    monkeypatch.setenv("POSTGRES_URI", "postgresql://stub/stub")
    store = FakeStore()
    extractor = FakeExtractor(
        [memory.MemoryItem(kind="fact", content="用户在做 LangGraph 项目")]
    )
    monkeypatch.setattr(memory, "get_store", lambda: store)
    monkeypatch.setattr(memory, "make_extractor", lambda *a, **k: extractor)

    class FakeModel:
        """第一轮请求工具调用，第二轮给出最终回答。"""

        def __init__(self):
            self.rounds = 0

        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            self.rounds += 1
            if self.rounds == 1:
                return _tool_call_message()
            return AIMessage(content="好的")

    messages = [HumanMessage(content="我在做一个 LangGraph 项目")]
    state = AgentState(messages=list(messages))
    out = asyncio.run(
        _answer_with_memory_tools(FakeModel(), messages, state=state, config={})
    )

    assert out.content == "好的"
    assert len(extractor.calls) == 1
    prompt_text = extractor.calls[0]["messages"][1].content
    assert "我在做一个 LangGraph 项目" in prompt_text
    assert store.searched_query == "用户在做的事"
    assert store.puts[0][2]["content"] == "用户在做 LangGraph 项目"


def test_memory_item_accepts_status_kind():
    item = memory.MemoryItem(kind="status", content="用户正在学习 LangGraph")
    assert item.kind == "status"


def test_compute_weight_of_permanent_kinds_always_one():
    """fact / preference 是永久性记忆，多久没访问、访问多少次权重都是 1。"""
    long_ago = NOW - timedelta(days=3650)
    for kind in ("fact", "preference"):
        assert memory.compute_weight(kind, 0, long_ago, NOW) == 1.0
        assert memory.compute_weight(kind, 999, long_ago, NOW) == 1.0


def test_compute_weight_of_status_decays_and_rewards_access():
    """status 用 exp(-λ·Δt) + α·ln(1 + access_count)：半衰期 30 天，α = 0.1。"""
    # 刚访问过、尚无访问加成 → 1.0
    assert memory.compute_weight("status", 0, NOW, NOW) == pytest.approx(1.0)
    # 经过一个半衰期 → 0.5
    half_life_ago = NOW - timedelta(days=memory.MEMORY_DECAY_HALF_LIFE_DAYS)
    assert memory.compute_weight("status", 0, half_life_ago, NOW) == pytest.approx(
        0.5, abs=1e-9
    )
    # 衰减项 0.5 + 访问 9 次的对数加成 0.1·ln(10)
    expected = 0.5 + memory.MEMORY_ACCESS_ALPHA * math.log(10)
    assert memory.compute_weight("status", 9, half_life_ago, NOW) == pytest.approx(
        expected, abs=1e-9
    )
    # 未来时间戳（时钟偏移）不应产生 >1 的衰减项
    future = NOW + timedelta(days=30)
    assert memory.compute_weight("status", 0, future, NOW) == pytest.approx(1.0)


def test_rank_memories_sorts_by_similarity_times_weight():
    stale = (NOW - timedelta(days=180)).isoformat()
    items = [
        FakeItem(
            key="stale",
            value={
                "kind": "status",
                "content": "用户之前在学习 LangChain",
                "access_count": 0,
                "last_accessed_at": stale,
            },
            score=0.9,
        ),
        FakeItem(
            key="fresh",
            value={
                "kind": "status",
                "content": "用户正在学习 LangGraph",
                "access_count": 0,
                "last_accessed_at": NOW.isoformat(),
            },
            score=0.8,
        ),
    ]
    # 相似度更高的陈年 status 记忆，输给权重更高的新近记忆
    assert [item.key for item in memory.rank_memories(items, 2, NOW)] == [
        "fresh",
        "stale",
    ]
    # limit 截断
    assert len(memory.rank_memories(items, 1, NOW)) == 1


def test_rank_memories_keeps_similarity_order_for_permanent_kinds():
    items = [
        FakeItem(key="a", value={"kind": "fact", "content": "用户叫小明"}, score=0.3),
        FakeItem(
            key="b", value={"kind": "preference", "content": "用户偏好简短"}, score=0.9
        ),
    ]
    assert [item.key for item in memory.rank_memories(items, 2, NOW)] == ["b", "a"]


def test_load_user_memories_records_access_without_reembedding(monkeypatch):
    store = FakeStore(
        items=[
            FakeItem(
                key="k1",
                value={
                    "kind": "status",
                    "content": "用户正在学习 LangGraph",
                    "access_count": 2,
                    "last_accessed_at": NOW.isoformat(),
                },
                score=0.9,
            )
        ]
    )
    monkeypatch.setattr(memory, "get_store", lambda: store)
    # 固定“当前时间”，避免权重断言随真实日期漂移
    monkeypatch.setattr(memory, "_utcnow", lambda: NOW)

    asyncio.run(
        memory.load_user_memories.ainvoke(
            {"question": "用户在学什么"}, config={"configurable": {"user_id": "u1"}}
        )
    )

    assert len(store.puts) == 1
    namespace, key, value = store.puts[0]
    assert (namespace, key) == (("memories", "u1"), "k1")
    assert value["access_count"] == 3
    assert value["content"] == "用户正在学习 LangGraph"
    # 访问次数 3 → 权重约为 1 + 0.1·ln(4)（时间衰减项接近 1）
    expected = 1.0 + memory.MEMORY_ACCESS_ALPHA * math.log(4)
    assert value["weight"] == pytest.approx(expected, abs=0.05)
    # 正文未变，写回时跳过重新嵌入
    assert store.put_indexes == [False]


def test_update_user_memory_keeps_access_stats_of_existing(monkeypatch):
    """改写既有记忆时沿用其访问次数与上次访问时间，权重按当前时间重算。"""
    store = FakeStore(
        items=[
            FakeItem(
                key="k1",
                value={
                    "kind": "status",
                    "content": "用户正在学习 LangChain",
                    "access_count": 4,
                    "last_accessed_at": "2026-06-01T00:00:00+00:00",
                },
                score=0.9,
            )
        ]
    )
    extractor = FakeExtractor(
        [
            memory.MemoryItem(
                kind="status",
                content="用户已学完 LangChain，正在学习 LangGraph",
                memory_key="k1",
            )
        ]
    )
    monkeypatch.setattr(memory, "get_store", lambda: store)
    monkeypatch.setattr(memory, "make_extractor", lambda *a, **k: extractor)

    out = asyncio.run(
        memory.update_user_memory.ainvoke(
            {
                "question": "用户在学什么",
                "state": {
                    "messages": [HumanMessage(content="我已经学完 LangChain 了")]
                },
            },
            config={"configurable": {"user_id": "u1"}},
        )
    )

    assert "更新" in out
    assert len(store.puts) == 1
    _, key, value = store.puts[0]
    assert key == "k1"
    assert value["access_count"] == 4
    assert value["last_accessed_at"] == "2026-06-01T00:00:00+00:00"
    # 距上次访问已过数月，阶段性记忆权重明显衰减
    assert value["weight"] < 0.5
    # 正文变了 → 需要重新嵌入
    assert store.put_indexes == [None]


def test_format_memories_groups_status_separately():
    text = memory.format_memories(
        [
            FakeItem(key="a", value={"kind": "fact", "content": "用户叫小明"}),
            FakeItem(key="b", value={"kind": "preference", "content": "用户偏好简短"}),
            FakeItem(
                key="c", value={"kind": "status", "content": "用户正在学习 LangGraph"}
            ),
        ]
    )
    assert "【事实性记忆】" in text and "用户叫小明" in text
    assert "【偏好与行为准则】" in text and "用户偏好简短" in text
    assert "【阶段性状态" in text and "用户正在学习 LangGraph" in text
