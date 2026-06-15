"""Desktop agent wrapper — streaming chat with DeepSeek."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.types import Command

from .graph import create_agent
from .rag import RAGEngine
from .store import MemoryStore

load_dotenv()


def _load_config(config: dict) -> dict:
    """Merge config.json with .env fallbacks."""
    return {
        "api_key": config.get("api_key") or os.getenv("DEEPSEEK_API_KEY", ""),
        "model": config.get("model") or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "system_prompt": config.get("system_prompt", ""),
    }


def _build_messages(history: list[dict], system_prompt: str = ""):
    """Convert session message dicts to LangChain message objects."""
    msgs: list = []
    if system_prompt:
        msgs.append(SystemMessage(content=system_prompt))
    for msg in history:
        role, content = msg["role"], msg.get("content", "")
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    return msgs


def stream_response(
    history: list[dict],
    config: dict,
    web_search: bool = False,
    rag_engine: RAGEngine | None = None,
    memory_store: MemoryStore | None = None,
    context_compression: bool = True,
    thread_id: str | None = None,
):
    """Yields (type, value) tuples from the custom ReAct agent.

    Uses ``stream_mode="messages"`` for real per-token streaming.

    Types:
        "token"    — text chunk to append to output
        "search"   — bool, True while searching, False when done
        "interrupt" — dict, tool call info waiting for user confirmation

    When ``thread_id`` is provided and the agent has a checkpointer,
    the caller can ``send()`` a resume value back to the generator to
    continue execution after an ``"interrupt"``.
    """
    cfg = _load_config(config)
    if not cfg["api_key"]:
        yield "token", "[ERROR] 未配置 API Key，请在 config.json 中设置 api_key"
        return

    agent = create_agent(
        api_key=cfg["api_key"],
        model=cfg["model"],
        system_prompt=cfg["system_prompt"],
        web_search=web_search,
        rag_engine=rag_engine,
        memory_store=memory_store,
        context_compression=context_compression,
    )
    messages = _build_messages(history, "")

    run_config = {"configurable": {"thread_id": thread_id}} if thread_id else None
    input_data: dict | Command = {"messages": messages}
    searching = False
    interrupted = False

    while True:
        try:
            for msg_chunk, metadata in agent.stream(
                input_data,
                stream_mode="messages",
                config=run_config,
            ):
                node = metadata.get("langgraph_node", "")

                if node == "model":
                    # Tool call detected → entering search
                    if msg_chunk.tool_call_chunks and not searching:
                        searching = True
                        # 提取工具名称，区分 RAG 查询与联网搜索
                        tool_name = ""
                        for chunk in msg_chunk.tool_call_chunks:
                            tn = chunk.get("name", "") if isinstance(chunk, dict) else getattr(chunk, "name", "")
                            if tn:
                                tool_name = tn
                                break
                        yield "search", tool_name or "web_search"

                    # Real content token from final response
                    if msg_chunk.content:
                        if searching:
                            searching = False
                            yield "search", False
                        yield "token", msg_chunk.content

        except Exception as e:
            yield "token", f"\n\n[错误] {e}"
            yield "search", False
            return

        # ── 检查是否有中断 ─────────────────────────────────────────
        if run_config is None:
            break  # 无 checkpointer，不会发生中断

        try:
            state = agent.get_state(run_config)
        except Exception:
            break

        # 检查是否有带中断的待处理任务
        pending_interrupts = []
        if state and hasattr(state, "tasks") and state.tasks:
            for task in state.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    pending_interrupts.extend(task.interrupts)

        if pending_interrupts:
            interrupted = True
            val = pending_interrupts[0].value
            # 发出中断信号，等待 caller 通过 send() 传回用户确认结果
            resume_value = yield ("interrupt", val)
            input_data = Command(resume=resume_value)
            continue  # 继续执行图

        break  # 正常执行完毕

    if interrupted:
        yield "search", False


def generate_title(history: list[dict], config: dict) -> str | None:
    """Generate a short session title from the conversation."""
    cfg = _load_config(config)
    if not cfg["api_key"]:
        return None

    conversation = "\n".join(
        f"{'用户' if m['role'] == 'user' else 'AI'}: {m['content'][:100]}"
        for m in history
    )

    llm = ChatDeepSeek(model=cfg["model"], api_key=cfg["api_key"])
    messages = [
        SystemMessage(
            content="你是一个会话标题生成器。根据对话内容生成一个10字以内的简短标题，"
                    "直接返回标题文本，不要标点、引号或任何多余内容。"
        ),
        HumanMessage(content=conversation),
    ]
    try:
        result = llm.invoke(messages)
        title = result.content.strip().strip("\"'")
        return title[:30] if title else None
    except Exception:
        return None
