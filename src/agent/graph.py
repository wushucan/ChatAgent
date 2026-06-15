"""Custom StateGraph ReAct agent with optional web search (Tavily).

手写 LangGraph StateGraph 替换预置 ``create_react_agent``。
显式定义节点、条件边、状态管理，图结构完全可控。
"""

from __future__ import annotations

import os
from typing import Annotated, Sequence

_SENTINEL = object()

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from typing_extensions import TypedDict

from agent.context import BM25Compressor
from agent.rag import RAGEngine
from agent.store import MemoryStore

load_dotenv()


# ── Tavily 搜索（惰性 import） ──────────────────────────────────────


def _tavily_search(query: str) -> str:
    """Run a Tavily web search and return formatted results."""
    from tavily import TavilyClient

    try:
        client = TavilyClient()
        result = client.search(query=query, search_depth="advanced", max_results=5)
        items = result.get("results", [])
        if not items:
            return "未找到相关结果。"
        lines = [
            f"「{r['title']}」: {r['content']}\n来源: {r['url']}"
            for r in items
        ]
        return "\n\n".join(lines)
    except Exception as e:
        return f"[搜索出错] {e}"


# ── Agent 状态 ─────────────────────────────────────────────────────


class AgentState(TypedDict):
    """LangGraph 状态。

    ``messages`` 通过 ``add_messages`` reducer 自动合并，
    确保每次节点返回值与历史消息正确归并，不产生重复。
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ── Agent 工厂 ─────────────────────────────────────────────────────


def create_agent(
    api_key: str = "",
    model: str = "",
    system_prompt: str = "",
    web_search: bool = True,
    rag_engine: RAGEngine | None = None,
    memory_store: MemoryStore | None = None,
    context_compression: bool = True,
    context_window: int = 40,
    context_recent_keep: int = 8,
    context_bm25_top_k: int = 12,
    checkpointer: MemorySaver | None = _SENTINEL,  # _SENTINEL = default (use MemorySaver); None = no checkpointer
):
    """构建自定义 ReAct Agent（StateGraph 手写实现）。

    区别于预置 ``create_react_agent``，本函数显式定义了：

    - **状态** — ``AgentState`` 含 ``messages`` 字段
    - **节点** — ``call_model``（LLM 调用）、``call_tool``（工具执行）
    - **条件边** — ``should_continue`` 根据 LLM 是否调工具路由

    参数
    ----
    api_key : str
        DeepSeek API Key，默认从环境变量读取。
    model : str
        模型名称，默认 ``deepseek-v4-flash``。
    system_prompt : str
        系统提示词，拼入 LLM 调用的上下文头部。
    web_search : bool
        是否开启联网搜索（需要 ``TAVILY_API_KEY`` 环境变量）。
    memory_store : MemoryStore | None
        跨会话记忆存储。开启后 LLM 可调用 ``save_memory`` 工具保存信息，
        并在回答时携带已知记忆。
    """
    api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
    model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    llm = ChatDeepSeek(model=model, api_key=api_key)

    # ── 工具注册 ────────────────────────────────────────────────
    tools: dict[str, tool] = {}
    if web_search and os.getenv("TAVILY_API_KEY"):

        @tool
        def web_search(query: str) -> str:
            """搜索互联网获取最新信息。适用于查询新闻、实时数据、当前事件等需要联网获取的内容。"""
            return _tavily_search(query)

        tools["web_search"] = web_search

    if rag_engine is not None:

        @tool
        def rag_search(query: str) -> str:
            """搜索已上传的文档内容。适用于查询文档知识库中上传的 PDF、TXT、MD、DOCX 文件中的信息。"""
            results = rag_engine.search(query, k=5)
            if not results:
                return "未找到相关文档内容。"
            return "\n\n".join(
                f"[来自 {r.metadata.get('filename', '文档')}] {r.page_content}"
                for i, r in enumerate(results)
            )

        tools["rag_search"] = rag_search

    if memory_store is not None:

        @tool
        def save_memory(namespace: str, key: str, value: str) -> str:
            """保存跨会话记忆。适用于记住用户偏好、个人信息、重要事实等需要在未来对话中保留的信息。"""
            memory_store.put(namespace, key, value)
            return f"已保存记忆 [{namespace}] {key}: {value}"

        tools["save_memory"] = save_memory

    bound_llm = llm.bind_tools(list(tools.values())) if tools else llm

    compressor = (
        BM25Compressor(
            recent_keep=context_recent_keep,
            bm25_top_k=context_bm25_top_k,
        )
        if context_compression and context_window > 0
        else None
    )

    # ── LLM 节点 ────────────────────────────────────────────────

    def call_model(state: AgentState, config) -> dict:
        """调用 LLM，返回生成的回复消息。

        如果启用了上下文压缩且消息数超过 ``context_window``，
        先用 BM25 选出相关历史，再发给 LLM。
        """
        messages = list(state["messages"])

        # BM25 上下文压缩
        if compressor is not None and len(messages) > context_window:
            before = len(messages)
            messages = compressor.compress(messages)
            print(f"[BM25] 上下文压缩: {before} 条 → {len(messages)} 条消息")

        # Build system messages (prompt + memory instructions + stored memories)
        system_msgs = []
        if system_prompt:
            system_msgs.append(SystemMessage(content=system_prompt))
        if memory_store is not None:
            system_msgs.append(SystemMessage(
                content="你可以使用 `save_memory` 工具跨会话持久化记忆。"
                        "当用户告诉你个人信息、偏好或者重要事实时，请主动调用 save_memory 保存。"
            ))
            memory_text = memory_store.format_for_prompt()
            if memory_text:
                system_msgs.append(SystemMessage(content=memory_text))
        if system_msgs:
            messages = [*system_msgs, *messages]

        response = bound_llm.invoke(messages, config=config)
        return {"messages": [response]}

    # ── 工具执行节点（含人工确认中断） ────────────────────────────

    def call_tool(state: AgentState) -> dict:
        """执行所有待处理的工具调用，返回结果消息。

        ``save_memory`` 跳过 interrupt 静默执行（存记忆无需确认），
        其余工具仍触发中断等待用户确认。
        """
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", [])
        if not tool_calls:
            return {"messages": []}

        # save_memory 静默执行，不走 interrupt
        silent_calls = [tc for tc in tool_calls if tc["name"] == "save_memory"]
        confirm_calls = [tc for tc in tool_calls if tc["name"] != "save_memory"]

        results = []

        for tc in silent_calls:
            func = tools.get(tc["name"])
            if func:
                try:
                    output = func.invoke(tc["args"])
                except Exception as e:
                    output = f"[工具执行出错] {e}"
                results.append(
                    ToolMessage(content=str(output), tool_call_id=tc["id"])
                )

        # 需要用户确认的工具走 interrupt
        if confirm_calls:
            user_response = interrupt({
                "type": "tool_call_confirmation",
                "tool_calls": [
                    {"name": tc["name"], "args": tc["args"], "id": tc["id"]}
                    for tc in confirm_calls
                ],
            })

            if user_response != "approved":
                results.extend(
                    ToolMessage(content="用户已取消搜索", tool_call_id=tc["id"])
                    for tc in confirm_calls
                )
            else:
                for tc in confirm_calls:
                    func = tools.get(tc["name"])
                    if func:
                        try:
                            output = func.invoke(tc["args"])
                        except Exception as e:
                            output = f"[工具执行出错] {e}"
                        results.append(
                            ToolMessage(content=str(output), tool_call_id=tc["id"])
                        )

        return {"messages": results}

    # ── 条件边 ──────────────────────────────────────────────────

    def should_continue(state: AgentState) -> str:
        """条件边：LLM 请求工具调用则走 tools，否则结束。"""
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    # ── 组装图 ──────────────────────────────────────────────────

    builder = StateGraph(AgentState)
    builder.add_node("model", call_model)
    builder.add_edge("__start__", "model")

    if tools:
        # 完整 ReAct 循环：model → tools → model → ...
        builder.add_node("tools", call_tool)
        builder.add_conditional_edges(
            "model",
            should_continue,
            {"tools": "tools", END: END},
        )
        builder.add_edge("tools", "model")
    else:
        # 无工具：单步执行 model → end
        builder.add_conditional_edges(
            "model",
            should_continue,
            {END: END},
        )

    if checkpointer is _SENTINEL:
        checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# ── LangGraph Server 入口（使用 .env 配置，带搜索） ─────────────────
# LangGraph Server 自动处理持久化，不要传自定义 checkpointer

graph = create_agent(checkpointer=None)
