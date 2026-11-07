"""shared.utils 的单元测试，重点覆盖智谱结构化输出的 content JSON 兜底。"""

from typing import TypedDict

import pytest
from langchain_core.messages import AIMessage
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

from shared.utils import (
    _first_tool_name,
    _parse_json_object,
    _tool_call_from_content,
    load_chat_model,
)


class KeywordExtraction(TypedDict):
    """测试用结构化输出 schema（名字即工具名）。"""

    keywords: list[str]


def test_parse_json_object_plain():
    assert _parse_json_object('{"keywords": ["a"]}') == {"keywords": ["a"]}


def test_parse_json_object_with_fence_and_prefix():
    text = '抽取结果如下：\n```json\n{"keywords": ["a", "b"]}\n```'
    assert _parse_json_object(text) == {"keywords": ["a", "b"]}


def test_parse_json_object_rejects_non_object_and_prose():
    assert _parse_json_object('["a", "b"]') is None
    assert _parse_json_object("这段问题没有可抽取的关键词。") is None
    assert _parse_json_object("") is None


def test_first_tool_name():
    assert _first_tool_name([KeywordExtraction]) == "KeywordExtraction"
    assert _first_tool_name([]) is None


def test_parser_alone_drops_content_json():
    """先固化失败语义：content 里的 JSON 不被 function_calling 解析器采纳。"""
    parser = JsonOutputKeyToolsParser(
        key_name="KeywordExtraction", first_tool_only=True
    )
    assert parser.invoke(AIMessage(content='{"keywords": ["a"]}')) is None


def test_tool_call_from_content_is_parsed_by_real_parser():
    """兜底后，content 里的 JSON 能被真实解析器取出，与工具调用路径一致。"""
    parser = JsonOutputKeyToolsParser(
        key_name="KeywordExtraction", first_tool_only=True
    )
    chain = _tool_call_from_content("KeywordExtraction") | parser
    assert chain.invoke(AIMessage(content='{"keywords": ["a", "b"]}')) == {
        "keywords": ["a", "b"]
    }


def test_tool_call_from_content_keeps_existing_tool_call():
    """已有工具调用时不改写，避免破坏正常路径。"""
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "KeywordExtraction",
                "args": {"keywords": ["x"]},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )
    fixed = _tool_call_from_content("KeywordExtraction").invoke(message)
    assert fixed.tool_calls == message.tool_calls


def test_tool_call_from_content_ignores_prose():
    """content 不是 JSON 对象时原样放行，保持原有失败语义。"""
    message = AIMessage(content="无法从该问题中抽取关键词。")
    fixed = _tool_call_from_content("KeywordExtraction").invoke(message)
    assert fixed.tool_calls == []


@pytest.fixture
def _glm_key(monkeypatch):
    monkeypatch.setenv("GLM_API_KEY", "test-key")


def _stub_generate(content: str):
    """替换 ChatOpenAI._generate，返回指定的纯文本回复（不发起网络请求）。"""

    def _fake_generate(self, *args, **kwargs):
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )

    return _fake_generate


def test_structured_output_recovers_content_json(monkeypatch, _glm_key):
    """模型把 JSON 写进 content 而不调用工具时，结构化输出仍能取到结果。"""
    monkeypatch.setattr(
        ChatOpenAI, "_generate", _stub_generate('{"keywords": ["LangGraph", "Store"]}')
    )
    model = load_chat_model("glm/glm-4.6")
    structured = model.with_structured_output(KeywordExtraction)
    assert structured.invoke([{"role": "user", "content": "x"}]) == {
        "keywords": ["LangGraph", "Store"]
    }


def test_structured_output_still_none_for_prose(monkeypatch, _glm_key):
    """模型输出的是散文而非 JSON 时，仍按失败处理（返回 None）。"""
    monkeypatch.setattr(ChatOpenAI, "_generate", _stub_generate("我不知道该怎么分类。"))
    model = load_chat_model("glm/glm-4.6")
    structured = model.with_structured_output(KeywordExtraction)
    assert structured.invoke([{"role": "user", "content": "x"}]) is None


def test_bind_tools_without_forced_choice_keeps_plain_binding(monkeypatch, _glm_key):
    """普通工具绑定（tool_choice=None）不挂兜底，避免误判正常文本输出。"""
    monkeypatch.setattr(ChatOpenAI, "_generate", _stub_generate('{"keywords": ["a"]}'))
    model = load_chat_model("glm/glm-4.6")
    messages = [{"role": "user", "content": "x"}]
    plain = model.bind_tools([KeywordExtraction]).invoke(messages)
    forced = model.bind_tools([KeywordExtraction], tool_choice="any").invoke(messages)
    assert plain.tool_calls == []
    assert [call["name"] for call in forced.tool_calls] == ["KeywordExtraction"]
