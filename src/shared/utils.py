"""项目中使用到的共享工具函数。

函数：
    format_docs: 将文档转换为 xml 格式的字符串。
    load_chat_model: 根据模型名称加载聊天模型。
"""

import json
import os
from collections.abc import Callable, Sequence
from typing import Any, Optional, cast

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import SecretStr

# 智谱 GLM 的 OpenAI 兼容端点
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"


def _format_doc(doc: Document) -> str:
    """将单个文档格式化为 XML。

    Args:
        doc (Document): 要格式化的文档。

    Returns:
        str: 格式化后的文档 XML 字符串。
    """
    metadata = doc.metadata or {}
    meta = "".join(f" {k}={v!r}" for k, v in metadata.items())
    if meta:
        meta = f" {meta}"

    return f"<document{meta}>\n{doc.page_content}\n</document>"


def coerce_str_list(value: Any) -> list[str]:
    """把 LLM 结构化输出中的字符串列表字段规范化为纯文本列表。

    GLM 经工具调用输出时偶发把 str 元素写成 dict（如 {"description": "..."}），
    而 TypedDict 路径的 with_structured_output 无 schema 校验，会原样透传
    污染下游（如把 dict 当 message content）；这里统一还原为字符串。
    """
    items = value if isinstance(value, list) else [value] if value else []
    result: list[str] = []
    for item in items:
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            # 取第一个非空字符串值（如 {"description": "..."}）；否则序列化兜底
            found = next(
                (v for v in item.values() if isinstance(v, str) and v.strip()),
                None,
            )
            text = found if found is not None else json.dumps(item, ensure_ascii=False)
        else:
            text = str(item)
        if text.strip():
            result.append(text)
    return result


def format_docs(docs: Optional[list[Document]]) -> str:
    """将文档列表格式化为 XML。

    该函数接收 Document 对象列表，并将其格式化为一整段 XML 字符串。

    Args:
        docs (Optional[list[Document]]): 要格式化的 Document 对象列表，或 None。

    Returns:
        str: 包含格式化后文档的 XML 字符串。

    示例:
        >>> docs = [Document(page_content="Hello"), Document(page_content="World")]
        >>> print(format_docs(docs))
        <documents>
        <document>
        Hello
        </document>
        <document>
        World
        </document>
        </documents>

        >>> print(format_docs(None))
        <documents></documents>
    """
    if not docs:
        return "<documents></documents>"
    formatted = "\n".join(_format_doc(doc) for doc in docs)
    return f"""<documents>
{formatted}
</documents>"""


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """从模型文本里解析出 JSON 对象，失败返回 None。

    先整体解析，再退回截取首个 "{" 到末个 "}" 之间的片段（可容忍 ```json
    代码围栏与前后多余文字）。解析结果不是对象（如数组、标量）时同样返回 None。
    """
    if not isinstance(text, str) or not text.strip():
        return None
    candidates = [text.strip()]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict):
            return data
    return None


def _first_tool_name(tools: Any) -> str | None:
    """取工具列表中第一个工具的名字，取不到时返回 None。"""
    if not isinstance(tools, list | tuple) or not tools:
        return None
    try:
        return cast(str, convert_to_openai_tool(tools[0])["function"]["name"])
    except (KeyError, TypeError, ValueError):
        return None


def _tool_call_from_content(tool_name: str) -> RunnableLambda[AIMessage, AIMessage]:
    """构造一个把「JSON 落在 content 里」的响应补成工具调用的 Runnable。

    智谱端点的 tool_choice 仅支持 auto（官方文档该字段的枚举只有 auto），
    langchain-openai 传出的 tool_choice="required" 会被静默忽略，模型偶发
    不发起工具调用、直接把 JSON 文本写进 content。此时
    with_structured_output 的 JsonOutputKeyToolsParser 取不到工具调用会返回
    None，结构化结果被整个丢掉。

    这里在解析前做一次纠正：仅当「没有 tool_calls 且 content 是 JSON 对象」时，
    用 content 合成一个同名 tool_call，使后续解析与正常工具调用完全一致；
    content 不是 JSON 对象时原样放行，保持原有的失败语义。
    """

    def _fix(message: AIMessage) -> AIMessage:
        if not isinstance(message, AIMessage) or message.tool_calls:
            return message
        data = (
            _parse_json_object(message.content)
            if isinstance(message.content, str)
            else None
        )
        if data is None:
            return message
        return message.model_copy(
            update={
                "tool_calls": [
                    {
                        "name": tool_name,
                        "args": data,
                        "id": "call_from_content",
                        "type": "tool_call",
                    }
                ]
            }
        )

    return RunnableLambda(_fix)


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """根据完整的名称加载聊天模型。

    Args:
        fully_specified_name (str): 格式为 'provider/model' 的字符串。
    """
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = ""
        model = fully_specified_name
    if provider in ("zhipu", "glm"):
        # 智谱 GLM：走 OpenAI 兼容端点，密钥从 GLM_API_KEY 读取
        from langchain_openai import ChatOpenAI

        class ZhipuChatOpenAI(ChatOpenAI):
            """适配智谱兼容端点的 with_structured_output。

            langchain-openai 1.x 的 ChatOpenAI 默认 method="json_schema"（response_format），
            智谱端点不支持该形式，模型会忽略并自由输出导致解析失败；显式改为
            function_calling（工具调用）。同时智谱不支持字符串函数名形式的
            tool_choice（with_structured_output 内部默认 tool_choice=<工具名>），
            改用 "any"（强制调用工具）实测可用。kwargs 后展开可覆盖内部默认值。

            但智谱对 tool_choice 的实现只认 auto，强制取值会被静默忽略，模型仍会
            偶发不调用工具、把 JSON 直接写进 content；故在强制调用工具时于
            bind_tools 里挂一个把 content JSON 补成工具调用的 Runnable 兜底
            （见 _tool_call_from_content）。
            """

            def with_structured_output(
                self,
                schema: dict[str, Any] | type | None = None,
                **kwargs: Any,
            ) -> Runnable[LanguageModelInput, Any]:
                kwargs.setdefault("method", "function_calling")
                kwargs.setdefault("tool_choice", "any")
                return super().with_structured_output(schema, **kwargs)

            def bind_tools(
                self,
                tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
                *,
                tool_choice: dict[str, Any] | str | bool | None = None,
                **kwargs: Any,
            ) -> Runnable[LanguageModelInput, AIMessage]:
                """绑定工具；强制调用工具时补一个 content JSON → tool_call 的纠正。

                仅在 tool_choice 为 "any"/"required"（即结构化输出强制调用场景）
                时挂载纠正，普通工具绑定（tool_choice=None，如 respond 节点的
                记忆工具）保持原样，避免把模型正常的文本输出误判成工具调用。
                """
                bound = super().bind_tools(tools, tool_choice=tool_choice, **kwargs)
                if tool_choice not in ("any", "required"):
                    return bound
                name = _first_tool_name(tools)
                if name is None:
                    return bound
                return bound | _tool_call_from_content(name)

        return ZhipuChatOpenAI(
            model=model,
            api_key=SecretStr(os.environ["GLM_API_KEY"]),
            base_url=ZHIPU_BASE_URL,
        )
    return init_chat_model(model, model_provider=provider)
