"""长期/短期记忆：基于 Postgres 的持久化与记忆读写服务。

记忆架构：
- 短期记忆：线程级对话消息历史，由 langgraph 的 Postgres checkpointer 保存
  （graph.compile(checkpointer=...)），按 thread_id 区分会话，跨请求保持上下文。
- 长期记忆：跨会话保存的记忆，写入 Postgres store（graph.compile(store=...)）。
  命名空间为 ("memories", user_id)，每条记忆按 kind 分三类：
  - fact：关于用户长期稳定的客观事实（称呼、职业、所在城市等）；
  - preference：用户偏好与回答要求（语气、风格、长度、语言等）；
  - status：阶段性状态/进展（正在学习或使用的技术、正在准备的面试、
    正在推进的项目与求职目标等），会随时间失效。
  fact 与 preference 是永久性记忆，权重恒为 1；status 是阶段性记忆，权重按
  Weight = exp(-λ·Δt) + α·ln(1 + access_count) 随"距上次访问的时间"衰减、
  随访问次数呈对数加成（半衰期 30 天，α = 0.1，见 compute_weight）。

长期记忆检索：store 初始化时以 BGE-M3（1024 维，与检索链路同一编码器）对记忆
正文 content 建索引；两个记忆工具都用模型给出的 question 做向量检索——先按相似度
召回候选池，再按 Final_Score = Similarity × Weight 取前若干条，而非把该用户全部
记忆塞进上下文：
- load_user_memories：取回最相关的记忆交给回复模型，并把命中的记忆记为"已访问"
  （access_count 累加、刷新权重与访问时间）。
- update_user_memory：取回最相关的既有记忆交给 extractor 做去重/改写参考；
  回复模型只负责决定"是否触发"，不负责给出要保存的内容——该工具通过
  InjectedState 自行读取图状态里最近两轮对话，连同既有记忆一并交给 langmem 的
  create_extractor（配独立提示词 MEMORY_UPDATE_SYSTEM_PROMPT）抽取、规范化并
  写入 store，即记忆提取职责完全归 extractor。

若环境未配置 POSTGRES_URI、未安装 pgvector 扩展或连接失败，记忆功能直接置为
不可用（返回 None / 提示不可用），不做无向量检索的降级，不影响主图其余逻辑。
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import InjectedToolArg, tool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langgraph.checkpoint.base import (
        ChannelVersions,
        Checkpoint,
        CheckpointMetadata,
        CheckpointTuple,
    )
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.store.base import BaseStore, SearchItem
    from langgraph.store.postgres import PostgresStore
    from langgraph.store.postgres.base import PostgresIndexConfig

logger = logging.getLogger(__name__)

POSTGRES_URI_KEY = "POSTGRES_URI"
# langgraph 平台通常注入 langgraph_user_id；本地可直接传 user_id，兼容两者
USER_ID_CONFIG_KEYS = ("langgraph_user_id", "user_id")
DEFAULT_USER_ID = "default-user"
MAX_MEMORIES = 100
# 记忆抽取时回看的对话轮数：取最近这么多个用户提问及其之间的助手回复
MEMORY_EXTRACT_ROUNDS = 2
# 长期记忆的向量索引：BGE-M3 稠密向量维度与参与嵌入的字段
MEMORY_VECTOR_DIMS = 1024
MEMORY_VECTOR_FIELDS = ["content"]
# 向量检索先按相似度召回候选池，再按 Similarity × Weight 取前 MEMORY_VECTOR_LIMIT 条
MEMORY_CANDIDATE_LIMIT = 20
MEMORY_VECTOR_LIMIT = 5

# 记忆类型：fact/preference 长期稳定（永久性，权重恒为 1），status 为阶段性状态
KIND_FACT = "fact"
KIND_PREFERENCE = "preference"
KIND_STATUS = "status"
PERMANENT_KINDS = (KIND_FACT, KIND_PREFERENCE)

# 权重公式 Weight = exp(-λ·Δt) + α·ln(1 + access_count)：
# λ 由衰减半衰期推出，α 为访问频次加成系数（取对数抑制频繁访问的记忆无上限膨胀）
MEMORY_DECAY_HALF_LIFE_DAYS = 30.0
MEMORY_DECAY_LAMBDA = math.log(2) / MEMORY_DECAY_HALF_LIFE_DAYS
MEMORY_ACCESS_ALPHA = 0.1

# 模块级缓存：进程内只连接/建表一次。
# 注意：from_conn_string 返回 context manager（生成器），必须保留其引用，
# 否则临时对象被 GC 时会触发 __exit__ 关闭底层数据库连接。
_store: PostgresStore | None = None
_store_cm: AbstractContextManager[PostgresStore] | None = None
_checkpointer: PostgresSaver | None = None
_checkpointer_cm: AbstractContextManager[PostgresSaver] | None = None
_extractor_cache: dict[tuple[str, str], Runnable[Any, Any]] = {}


class MemoryItem(BaseModel):
    """一条规范化后的长期记忆（extractor 的抽取 schema）。"""

    kind: Literal["fact", "preference", "status"] = Field(
        description=(
            "记忆类型：fact=关于用户长期稳定的客观事实（称呼、职业、所在城市等）；"
            "preference=用户偏好或对回答方式的要求（语气、风格、长度、语言等）；"
            "status=阶段性状态/进展（正在学习或使用的技术、正在准备面试、正在推进的"
            "项目与求职目标等会随时间失效的信息）"
        )
    )
    content: str = Field(description="以一句话清晰完整表述的长期记忆，使用第三人称")
    memory_key: str = Field(
        default="",
        description="既有记忆的唯一标识：仅当改写已存在的记忆时填写，新增记忆必须留空",
    )


def get_postgres_uri() -> str | None:
    """读取 Postgres 连接串（未配置时返回 None）。"""
    return os.environ.get(POSTGRES_URI_KEY) or None


def is_memory_enabled() -> bool:
    """记忆功能是否启用（要求配置了 POSTGRES_URI）。"""
    return get_postgres_uri() is not None


def get_user_id(config: RunnableConfig) -> str:
    """从运行时配置中解析用户标识，未提供时回退默认用户。"""
    configurable = (config or {}).get("configurable", {}) or {}
    for key in USER_ID_CONFIG_KEYS:
        value = configurable.get(key)
        if value:
            return str(value)
    return DEFAULT_USER_ID


def memory_namespace(user_id: str) -> tuple[str, ...]:
    """返回某用户的长期记忆命名空间。"""
    return ("memories", user_id)


def _utcnow() -> datetime:
    """当前 UTC 时间（权重衰减与访问时间统一用带时区的 UTC）。"""
    return datetime.now(timezone.utc)


def parse_timestamp(value: object, fallback: datetime | None = None) -> datetime:
    """把存储的访问时间解析为带时区的 datetime；缺失或非法时用 fallback。"""
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            pass
        else:
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return fallback or _utcnow()


def compute_weight(
    kind: str,
    access_count: int,
    last_accessed_at: datetime,
    now: datetime | None = None,
) -> float:
    r"""计算一条记忆的权重。

    - 永久性记忆（fact / preference）：长期稳定，权重恒为 1，不随时间衰减。
    - 阶段性记忆（status）：按 Weight = exp(-λ·Δt) + α·ln(1 + access_count)
      计算，Δt 为距上次访问的天数，λ = ln2 / 半衰期，α 为访问频次加成系数。

    Args:
        kind: 记忆类型。
        access_count: 该记忆被访问（读取）的次数。
        last_accessed_at: 上次访问时间。
        now: 计算基准时间，默认当前 UTC 时间。

    Returns:
        记忆权重（阶段性记忆可能 > 1，因为访问加成未被截断）。
    """
    if kind in PERMANENT_KINDS:
        return 1.0
    now = now or _utcnow()
    delta_days = max((now - last_accessed_at).total_seconds() / 86400.0, 0.0)
    decay = math.exp(-MEMORY_DECAY_LAMBDA * delta_days)
    bonus = MEMORY_ACCESS_ALPHA * math.log1p(max(int(access_count or 0), 0))
    return decay + bonus


def memory_weight(value: dict[str, Any], now: datetime | None = None) -> float:
    """按记忆的存储结构（store value）计算它当前的权重。"""
    value = value or {}
    now = now or _utcnow()
    return compute_weight(
        value.get("kind", KIND_FACT),
        value.get("access_count", 0),
        parse_timestamp(value.get("last_accessed_at"), now),
        now,
    )


def build_memory_value(
    kind: str, content: str, previous: dict[str, Any] | None = None
) -> dict[str, Any]:
    """构造一条记忆的存储结构（正文 + 访问统计 + 权重）。

    新增记忆：access_count 从 0 起、访问时间与权重初始化为 1.0。
    改写既有记忆（previous 非空）：沿用其 access_count 与上次访问时间，
    权重按当前时间重算，避免改写把访问统计清零。
    """
    previous = previous or {}
    now = _utcnow()
    access_count = int(previous.get("access_count", 0) or 0)
    last_accessed_at = previous.get("last_accessed_at") or now.isoformat()
    return {
        "kind": kind,
        "content": content,
        "access_count": access_count,
        "last_accessed_at": last_accessed_at,
        "weight": compute_weight(
            kind, access_count, parse_timestamp(last_accessed_at, now), now
        ),
    }


def rank_memories(
    items: list[SearchItem], limit: int, now: datetime | None = None
) -> list[SearchItem]:
    """按 Final_Score = Similarity × Weight 对候选记忆重排，取前 limit 条。

    相似度取检索返回的 score（无分数时按 1 处理）；权重见 compute_weight，
    因此"相关但久未提及的阶段性记忆"会让位给"同样相关且新近活跃的记忆"。
    """
    now = now or _utcnow()
    scored = [
        (
            (item.score if item.score is not None else 1.0)
            * memory_weight(item.value, now),
            item,
        )
        for item in items
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:limit]]


def build_index_config() -> PostgresIndexConfig:
    """构造 PostgresStore 的向量索引配置（用 BGE-M3 编码记忆正文）。

    与检索链路共用同一编码器（shared.retrieval.make_text_encoder），保证记忆
    向量与知识库向量的语义空间一致；dims 必须与编码器输出维度一致。

    Returns:
        传给 PostgresStore 的 index 配置字典。
    """
    from shared.hybrid_retrieval import DENSE_EMBEDDING_MODEL
    from shared.retrieval import make_text_encoder

    return {
        "dims": MEMORY_VECTOR_DIMS,
        "embed": make_text_encoder(DENSE_EMBEDDING_MODEL),
        "fields": MEMORY_VECTOR_FIELDS,
    }


def get_store() -> PostgresStore | None:
    """惰性创建带向量索引的 PostgresStore 单例（含幂等建表）；不可用时返回 None。

    默认（未配置连接池）从连接串建立的单连接生命周期由 from_conn_string
    返回的 context manager 管理，这里持有该对象引用以避免连接被提前关闭。

    建库时会带上向量索引配置并执行 setup()（含 pgvector 扩展与向量表迁移）；
    若扩展缺失或连接失败则直接置为不可用（不降级为无向量检索）。
    """
    global _store, _store_cm
    if _store is not None or not is_memory_enabled():
        return _store
    uri = get_postgres_uri()
    if uri is None:
        return None
    try:
        from langgraph.store.postgres import PostgresStore

        _store_cm = PostgresStore.from_conn_string(uri, index=build_index_config())
        _store = _store_cm.__enter__()
        _store.setup()
        logger.info(
            "长期记忆 PostgresStore 已初始化（含 %d 维向量索引）", MEMORY_VECTOR_DIMS
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("长期记忆 PostgresStore 初始化失败，长期记忆不可用: %s", exc)
        _store = None
        _store_cm = None
    return _store


_async_saver_cls: type[PostgresSaver] | None = None


def _get_async_saver_cls() -> type[PostgresSaver]:
    """返回 PostgresSaver 的 async 兼容子类（进程内只定义一次）。

    langgraph 1.2 的 AsyncPregelLoop 要求 checkpointer 提供
    aget_tuple/aput/aput_writes 等 async 方法，而 PostgresSaver 仅有同步实现，
    这里用 asyncio.to_thread 把同步实现桥接到 async 接口上。
    """
    global _async_saver_cls
    if _async_saver_cls is None:
        from langgraph.checkpoint.postgres import PostgresSaver

        class AsyncCompatPostgresSaver(PostgresSaver):
            """把同步 PostgresSaver 的 async 接口桥接到线程池执行。"""

            async def aget_tuple(
                self, config: RunnableConfig
            ) -> CheckpointTuple | None:
                return await asyncio.to_thread(super().get_tuple, config)

            async def aput(
                self,
                config: RunnableConfig,
                checkpoint: Checkpoint,
                metadata: CheckpointMetadata,
                new_versions: ChannelVersions,
            ) -> RunnableConfig:
                return await asyncio.to_thread(
                    super().put, config, checkpoint, metadata, new_versions
                )

            async def aput_writes(
                self,
                config: RunnableConfig,
                writes: Sequence[tuple[str, Any]],
                task_id: str,
                task_path: str = "",
            ) -> None:
                await asyncio.to_thread(
                    super().put_writes, config, writes, task_id, task_path
                )

            async def alist(
                self,
                config: RunnableConfig | None,
                *,
                filter: dict[str, Any] | None = None,
                before: RunnableConfig | None = None,
                limit: int | None = None,
            ) -> AsyncIterator[CheckpointTuple]:
                def _list() -> list[CheckpointTuple]:
                    return list(
                        super(AsyncCompatPostgresSaver, self).list(
                            config, filter=filter, before=before, limit=limit
                        )
                    )

                for item in await asyncio.to_thread(_list):
                    yield item

        _async_saver_cls = AsyncCompatPostgresSaver
    return _async_saver_cls


def get_checkpointer() -> PostgresSaver | None:
    """惰性创建 PostgresSaver 单例（短记忆消息历史）；不可用时返回 None。"""
    global _checkpointer, _checkpointer_cm
    if _checkpointer is not None or not is_memory_enabled():
        return _checkpointer
    uri = get_postgres_uri()
    if uri is None:
        return None
    try:
        _checkpointer_cm = _get_async_saver_cls().from_conn_string(uri)
        _checkpointer = _checkpointer_cm.__enter__()
        _checkpointer.setup()
        logger.info("短期记忆 PostgresSaver 已初始化")
    except Exception as exc:  # noqa: BLE001
        logger.warning("短期记忆 PostgresSaver 初始化失败，将无会话记忆: %s", exc)
        _checkpointer = None
        _checkpointer_cm = None
    return _checkpointer


def format_memories(items: list[SearchItem]) -> str:
    """将 store 检索结果格式化为便于模型阅读的文本（按记忆类型分组）。"""
    facts: list[str] = []
    preferences: list[str] = []
    statuses: list[str] = []
    for item in items:
        value = item.value or {}
        kind = value.get("kind", KIND_FACT)
        content = str(value.get("content", "")).strip()
        if not content:
            continue
        if kind == KIND_PREFERENCE:
            preferences.append(f"- {content}")
        elif kind == KIND_STATUS:
            statuses.append(f"- {content}")
        else:
            facts.append(f"- {content}")
    if not facts and not preferences and not statuses:
        return "（当前没有已保存的长期记忆）"
    lines: list[str] = []
    if facts:
        lines.append("【事实性记忆】\n" + "\n".join(facts))
    if preferences:
        lines.append("【偏好与行为准则】\n" + "\n".join(preferences))
    if statuses:
        lines.append(
            "【阶段性状态（可能已随时间变化，必要时向用户确认）】\n"
            + "\n".join(statuses)
        )
    return "\n".join(lines)


def format_recent_dialogue(
    messages: list[AnyMessage], rounds: int = MEMORY_EXTRACT_ROUNDS
) -> str:
    r"""把最近若干轮对话整理成纯文本，供记忆抽取模型观察用户信息。

    以最近 rounds 条用户提问为界，从其中最早的一条一直取到消息列表末尾，
    只保留用户与助手的自然语言内容（跳过带工具调用的助手消息、工具结果等
    中间消息，以及内容为空的消息）。

    Args:
        messages: 图状态里的对话消息列表。
        rounds: 回看的用户提问轮数。

    Returns:
        形如「用户：...\n助手：...」的对话文本；无有效内容时返回空串。
    """
    if not messages:
        return ""
    human_indexes = [
        index
        for index, message in enumerate(messages)
        if getattr(message, "type", None) == "human"
    ]
    if not human_indexes:
        return ""
    start = human_indexes[-min(rounds, len(human_indexes))]
    lines: list[str] = []
    for message in messages[start:]:
        kind = getattr(message, "type", None)
        if kind not in ("human", "ai"):
            continue
        content = message.content
        text = content if isinstance(content, str) else str(content)
        if not text.strip():
            continue
        lines.append(f"{'用户' if kind == 'human' else '助手'}：{text.strip()}")
    return "\n".join(lines)


def make_extractor(memory_model: str, system_prompt: str) -> Runnable[Any, Any]:
    """构建（并缓存）用于抽取/更新长期记忆的 create_extractor。

    Args:
        memory_model: 记忆抽取所用模型，格式 provider/model-name。
        system_prompt: 记忆抽取模型的专属系统提示词。

    Returns:
        可运行的 create_extractor 实例（异步调用）。
    """
    cache_key = (memory_model, system_prompt)
    if cache_key in _extractor_cache:
        return _extractor_cache[cache_key]
    from langmem.knowledge.extraction import create_extractor

    from shared.utils import load_chat_model

    llm = load_chat_model(memory_model)
    extractor = cast(
        Runnable[Any, Any],
        create_extractor(
            llm,
            tools=[MemoryItem],
            enable_inserts=True,
            enable_updates=True,
            enable_deletes=False,
        ),
    )
    _extractor_cache[cache_key] = extractor
    return extractor


async def _record_access(
    store: BaseStore, namespace: tuple[str, ...], items: list[SearchItem]
) -> None:
    """把命中的记忆标记为"已访问"：access_count 累加并刷新权重。

    写回用 index=False 跳过重新嵌入（记忆正文未变，向量仍有效）；写回失败只
    告警，不影响本次检索结果。
    """
    now = _utcnow()
    for item in items:
        if not item.key:
            continue
        bumped = dict(item.value or {})
        access_count = int(bumped.get("access_count", 0) or 0) + 1
        bumped["access_count"] = access_count
        bumped["weight"] = compute_weight(
            bumped.get("kind", KIND_FACT),
            access_count,
            parse_timestamp(bumped.get("last_accessed_at"), now),
            now,
        )
        bumped["last_accessed_at"] = now.isoformat()
        try:
            await asyncio.to_thread(store.put, namespace, item.key, bumped, index=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("记忆访问统计写回失败: %s", exc)


async def _search_memories(
    store: BaseStore,
    namespace: tuple[str, ...],
    question: str,
    *,
    record_access: bool = False,
) -> list[SearchItem]:
    """按 question 向量检索记忆：先召回候选池，再按 Final_Score 取前若干条。

    Final_Score = Similarity × Weight（Weight 见 compute_weight）。question 为空
    时退回按更新时间取前若干条，不做加权重排。record_access 为真时把最终返回的
    记忆记为"已访问"（access_count 累加）。

    PostgresStore 为同步接口，放入独立线程执行，避免在 ASGI 事件循环里阻塞。
    """
    query = (question or "").strip()
    if not query:
        return await asyncio.to_thread(store.search, namespace, limit=MAX_MEMORIES)
    candidates = await asyncio.to_thread(
        store.search, namespace, query=query, limit=MEMORY_CANDIDATE_LIMIT
    )
    items = rank_memories(candidates, MEMORY_VECTOR_LIMIT)
    if record_access and items:
        await _record_access(store, namespace, items)
    return items


@tool
async def load_user_memories(
    question: str,
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> str:
    """按需读取该用户已保存的长期记忆（事实与偏好），供个性化回答时使用。

    当你认为回答可能需要用到用户的个人信息（称呼、职业、正在准备的事等）
    或用户对语气/风格的要求时，才调用本工具。

    Args:
        question: 你想了解的用户信息（如：用户称呼、用户背景、回答偏好）。
            系统据此做向量检索，只返回与该问题最相关的长期记忆。
    """
    store = await asyncio.to_thread(get_store)
    if store is None:
        return "（长期记忆功能当前不可用，请直接作答）"
    try:
        items = await _search_memories(
            store, memory_namespace(get_user_id(config)), question, record_access=True
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("长期记忆检索失败: %s", exc)
        return f"（长期记忆检索失败：{exc}，请直接作答）"
    return format_memories(items)


@tool
async def update_user_memory(
    question: str,
    state: Annotated[dict[str, Any], InjectedState],
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> str:
    """当最近对话中出现了值得跨会话记住的稳定个人信息或回答偏好时，更新长期记忆。

    仅当你在最近两轮对话里识别到真正值得长期记住的用户信息时才调用，并在给出
    最终回复前完成。你无需提供要保存的内容：本工具会把最近两轮对话交给记忆抽取
    模型，由它自行提炼并决定新增还是改写既有记忆。

    Args:
        question: 用于检索既有记忆的查询（概括最近对话中涉及的用户信息方向，
            如：用户称呼、用户背景、回答偏好）。系统据此做向量检索，只把最相关
            的既有记忆交给抽取模型做去重/改写参考。
    """
    store = await asyncio.to_thread(get_store)
    if store is None:
        return "（长期记忆功能当前不可用，未能保存记忆）"
    user_id = get_user_id(config)
    namespace = memory_namespace(user_id)

    dialogue = format_recent_dialogue(state.get("messages") or [])
    if not dialogue:
        return "最近对话中没有可提取的用户信息，未做修改。"

    from retrieval_graph.configuration import AgentConfiguration

    configuration = AgentConfiguration.from_runnable_config(config)
    try:
        existing_items = await _search_memories(store, namespace, question)
    except Exception as exc:  # noqa: BLE001
        logger.warning("既有记忆检索失败: %s", exc)
        return f"（既有记忆检索失败：{exc}，未能更新记忆）"
    existing = [
        (
            item.key,
            MemoryItem(
                kind=(item.value or {}).get("kind", KIND_FACT),
                content=str((item.value or {}).get("content", "")),
                memory_key=item.key,
            ),
        )
        for item in existing_items
    ]
    previous_values = {item.key: dict(item.value or {}) for item in existing_items}
    existing_text = format_memories(existing_items)
    extractor = make_extractor(
        configuration.memory_model, configuration.memory_update_system_prompt
    )
    messages = [
        SystemMessage(content=configuration.memory_update_system_prompt),
        HumanMessage(
            content=(
                "以下是最近两轮对话，请从中观察值得长期记住的用户信息：\n"
                f"{dialogue}\n\n"
                f"当前已保存的长期记忆（供去重/更新参考，含义相同的信息请改写既有记忆，"
                "不要重复新增）：\n"
                f"{existing_text}"
            )
        ),
    ]
    try:
        result = await extractor.ainvoke({"messages": messages, "existing": existing})
    except Exception as exc:  # noqa: BLE001
        logger.warning("记忆抽取失败: %s", exc)
        return f"（记忆抽取失败：{exc}，未能更新记忆）"

    items = result.get("responses", [])
    if not items:
        return "本次未识别到值得新增或更新的长期记忆，未做修改。"
    saved_lines: list[str] = []
    for item in items:
        # 既有记忆（带 memory_key）直接覆盖并沿用其访问统计；新记忆生成新的 key
        key = item.memory_key or uuid.uuid4().hex
        value = build_memory_value(
            item.kind, item.content, previous_values.get(item.memory_key or key)
        )
        await asyncio.to_thread(store.put, namespace, key, value)
        tag = "更新" if item.memory_key else "新增"
        saved_lines.append(f"{tag}[{item.kind}] {item.content}")
    logger.info("用户 %s 长期记忆已更新 %d 条", user_id, len(items))
    return "已更新长期记忆：\n" + "\n".join(saved_lines)
