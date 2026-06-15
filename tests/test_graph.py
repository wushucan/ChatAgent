"""Tests for the LangGraph ReAct agent — graph structure, tools, state."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages

from agent.graph import AgentState, create_agent


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def mock_tavily_env():
    """Provide TAVILY_API_KEY so tools get registered."""
    with patch.dict("os.environ", {"TAVILY_API_KEY": "mock-key"}):
        yield


@pytest.fixture
def mock_llm_cls():
    """Mock ChatDeepSeek class.

    Yields the class mock so tests can assert on both class-level
    (ChatDeepSeek()) and instance-level (invoke, bind_tools) calls.
    """
    with patch("agent.graph.ChatDeepSeek") as m:
        instance = m.return_value
        instance.bind_tools.return_value = instance
        instance.invoke.return_value = AIMessage(content="测试回复")
        yield m  # class mock — .return_value gives the instance mock


@pytest.fixture
def mock_rag():
    rag = MagicMock()
    rag.search.return_value = []
    return rag


@pytest.fixture
def mock_memory():
    memory = MagicMock()
    memory.format_for_prompt.return_value = ""
    return memory


# ── AgentState ───────────────────────────────────────────────────────


class TestAgentState:
    def test_has_messages_field(self):
        state = AgentState(messages=[])
        assert "messages" in state

    def test_add_messages_reducer(self):
        state = {"messages": [HumanMessage(content="hi")]}
        added = {"messages": [AIMessage(content="hello")]}
        merged = {k: add_messages(state[k], added[k]) for k in state}
        assert len(merged["messages"]) == 2


# ── create_agent ─────────────────────────────────────────────────────


class TestCreateAgent:
    def test_creates_compiled_graph(self, mock_llm_cls):
        app = create_agent(api_key="sk-test", model="deepseek-v4-flash")
        assert hasattr(app, "invoke")
        assert hasattr(app, "stream")

    def test_with_web_search(self, mock_llm_cls):
        app = create_agent(api_key="sk-test", web_search=True)
        assert app is not None

    def test_without_web_search(self, mock_llm_cls):
        app = create_agent(api_key="sk-test", web_search=False)
        assert app is not None

    def test_with_rag(self, mock_llm_cls, mock_rag):
        app = create_agent(api_key="sk-test", rag_engine=mock_rag)
        assert app is not None

    def test_with_memory(self, mock_llm_cls, mock_memory):
        app = create_agent(api_key="sk-test", memory_store=mock_memory)
        assert app is not None

    def test_model_default_loaded_from_env(self, mock_llm_cls):
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "env-key", "DEEPSEEK_MODEL": "deepseek-v4-flash", "TAVILY_API_KEY": "mock-key"}):
            create_agent()
            mock_llm_cls.assert_called()

    def test_bind_tools_called_when_tools_exist(self, mock_llm_cls):
        create_agent(api_key="sk-test", web_search=True)
        mock_llm_cls.return_value.bind_tools.assert_called_once()

    def test_no_bind_tools_when_no_tools(self, mock_llm_cls):
        with patch.dict("os.environ", {"TAVILY_API_KEY": ""}):
            create_agent(api_key="sk-test", web_search=True)
            mock_llm_cls.return_value.bind_tools.assert_not_called()


# ── call_model — System Prompt & Memory ──────────────────────────────


class TestCallModel:
    def _invoke_and_get_messages(self, mock_llm_cls, **kwargs):
        """Helper: invoke the agent and return the messages sent to the LLM."""
        app = create_agent(api_key="sk-test", **kwargs)
        app.invoke({"messages": [HumanMessage(content="你好")]}, {"configurable": {"thread_id": "1"}})
        call_args = mock_llm_cls.return_value.invoke.call_args
        return call_args[0][0] if call_args else []

    def test_system_prompt_included(self, mock_llm_cls):
        msgs = self._invoke_and_get_messages(mock_llm_cls, system_prompt="你是助手")
        assert any(isinstance(m, SystemMessage) and "你是助手" in m.content for m in msgs)

    def test_memory_injected(self, mock_llm_cls):
        memory = MagicMock()
        memory.format_for_prompt.return_value = "已知信息：\n  [user_prefs] language: Chinese"
        msgs = self._invoke_and_get_messages(mock_llm_cls, memory_store=memory)
        assert any("language: Chinese" in m.content for m in msgs)

    def test_memory_instruction_included(self, mock_llm_cls):
        memory = MagicMock()
        memory.format_for_prompt.return_value = ""
        msgs = self._invoke_and_get_messages(mock_llm_cls, memory_store=memory)
        assert any("save_memory" in m.content for m in msgs)

    def test_no_system_prompt_when_empty(self, mock_llm_cls):
        msgs = self._invoke_and_get_messages(mock_llm_cls, system_prompt="")
        assert not any(isinstance(m, SystemMessage) for m in msgs)


# ── Invocation Integration ──────────────────────────────────────────


class TestInvoke:
    def test_simple_chat(self, mock_llm_cls):
        instance = mock_llm_cls.return_value
        instance.invoke.return_value = AIMessage(content="你好！")
        app = create_agent(api_key="sk-test", web_search=False)
        result = app.invoke(
            {"messages": [HumanMessage(content="你好")]},
            {"configurable": {"thread_id": "test-1"}},
        )
        msgs = result["messages"]
        assert any(getattr(m, "content", "") == "你好！" for m in msgs)

    def test_tool_call_creates_tool_message(self, mock_llm_cls):
        """When LLM returns a tool call, call_tool node should execute it."""
        instance = mock_llm_cls.return_value
        tool_call_response = AIMessage(
            content="",
            tool_calls=[{
                "name": "web_search",
                "args": {"query": "test"},
                "id": "call_1",
                "type": "tool_call",
            }],
        )
        instance.invoke.side_effect = [
            tool_call_response,
            AIMessage(content="搜索结果"),
        ]

        with patch("agent.graph.interrupt") as mock_interrupt:
            mock_interrupt.return_value = "approved"
            app = create_agent(api_key="sk-test", web_search=True)
            result = app.invoke(
                {"messages": [HumanMessage(content="搜索")]},
                {"configurable": {"thread_id": "test-2"}},
            )
            msgs = result["messages"]
            assert any("搜索结果" in getattr(m, "content", "") for m in msgs)

    def test_interrupt_denied(self, mock_llm_cls):
        """When user denies a tool call, should get cancelled message."""
        instance = mock_llm_cls.return_value
        tool_call_response = AIMessage(
            content="",
            tool_calls=[{
                "name": "web_search",
                "args": {"query": "test"},
                "id": "call_1",
                "type": "tool_call",
            }],
        )
        instance.invoke.side_effect = [
            tool_call_response,
            AIMessage(content="好的"),
        ]

        with patch("agent.graph.interrupt") as mock_interrupt:
            mock_interrupt.return_value = "denied"
            app = create_agent(api_key="sk-test", web_search=True)
            result = app.invoke(
                {"messages": [HumanMessage(content="搜索")]},
                {"configurable": {"thread_id": "test-3"}},
            )
            msgs = result["messages"]
            tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
            assert any("已取消" in m.content for m in tool_msgs)


# ── Edge Cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_tool_execution_error_handled(self, mock_llm_cls):
        """Tool execution errors should not crash the agent."""
        mock_search = MagicMock(side_effect=RuntimeError("搜索失败"))
        with patch("agent.graph._tavily_search", mock_search):
            instance = mock_llm_cls.return_value
            tool_call_response = AIMessage(
                content="",
                tool_calls=[{
                    "name": "web_search",
                    "args": {"query": "test"},
                    "id": "call_1",
                    "type": "tool_call",
                }],
            )
            instance.invoke.side_effect = [
                tool_call_response,
                AIMessage(content="完成"),
            ]
            with patch("agent.graph.interrupt", return_value="approved"):
                app = create_agent(api_key="sk-test", web_search=True)
                result = app.invoke(
                    {"messages": [HumanMessage(content="搜索")]},
                    {"configurable": {"thread_id": "test-4"}},
                )
                assert result is not None

    def test_no_tools_no_crash(self, mock_llm_cls):
        """Agent with no tools should work for simple chat."""
        instance = mock_llm_cls.return_value
        instance.invoke.return_value = AIMessage(content="OK")
        app = create_agent(api_key="sk-test", web_search=False)
        result = app.invoke(
            {"messages": [HumanMessage(content="hi")]},
            {"configurable": {"thread_id": "test-5"}},
        )
        assert result is not None
