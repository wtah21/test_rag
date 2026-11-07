"""定义 agent 的可配置参数。"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Optional, Type, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config


@dataclass(kw_only=True)
class BaseConfiguration:
    """检索流程的公共配置类。

    定义检索链路共用的参数（当前为嵌入模型选择）。
    """

    embedding_model: Annotated[
        str,
        {"__template_metadata__": {"kind": "embeddings"}},
    ] = field(
        default="siliconflow/BAAI/bge-m3",
        metadata={"description": "要使用的嵌入模型名称。必须是有效的嵌入模型名称。"},
    )

    @classmethod
    def from_runnable_config(
        cls: Type[T], config: Optional[RunnableConfig] = None
    ) -> T:
        """根据 RunnableConfig 对象创建配置实例。

        参数:
        cls (Type[T]): 类本身。
        config (Optional[RunnableConfig]): 要使用的配置对象。

        返回:
        T: 具有指定配置的配置实例。
        """
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})


T = TypeVar("T", bound=BaseConfiguration)
