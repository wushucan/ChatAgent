"""PyWebview JS API bridge — session management + streaming."""

from __future__ import annotations

import json
import os
import threading
from typing import Any

from .desktop import generate_title, stream_response
from .persist import SessionStore
from .rag import RAGEngine
from .store import MemoryStore


class AgentAPI:
    """Methods exposed to JavaScript via ``pywebview.api``."""

    def __init__(self, config: dict, data_dir: str = "sessions"):
        self._config = config
        self._data_dir = data_dir
        self._store = SessionStore(data_dir)
        self._rag = RAGEngine(data_dir=os.path.join(data_dir, ".rag"))
        self._memory = MemoryStore(data_dir)
        # 后台预热 embedding 模型（约 6s），不阻塞 UI
        threading.Thread(target=self._rag.warmup, daemon=True).start()
        self._buffers: dict[str, dict[str, Any]] = {}  # session_id → stream state

    # ── session management ────────────────────────────────────────

    def get_sessions(self) -> list[dict]:
        return self._store.list_sessions()

    def create_session(self) -> dict:
        return self._store.create_session()

    def delete_session(self, session_id: str):
        # clean up any active stream
        self._buffers.pop(session_id, None)
        self._store.delete_session(session_id)

    def get_messages(self, session_id: str) -> list[dict]:
        return self._store.get_messages(session_id)

    def rename_session(self, session_id: str, title: str):
        self._store.rename_session(session_id, title)

    def delete_message(self, session_id: str, index: int) -> dict:
        """删除会话中的单条消息。"""
        ok = self._store.delete_message(session_id, index)
        return {"status": "ok" if ok else "error"}

    # ── streaming ──────────────────────────────────────────────────

    def send_message(self, session_id: str, message: str, web_search: bool = False, context_compression: bool = True) -> dict:
        """Start streaming a reply in a background thread, return immediately."""
        if not session_id:
            return {"status": "error", "error": "无效的会话 ID"}
        # Persist user message
        self._store.add_message(session_id, "user", message)

        # Prepare stream buffer (includes interrupt support)
        buffer: dict[str, Any] = {
            "tokens": [],
            "sent_count": 0,
            "done": False,
            "error": None,
            "searching": False,
            "interrupt": None,                 # interrupt data when waiting for user
            "resume_event": threading.Event(),  # signal to resume after interrupt
            "resume_value": {},                 # user's confirmation response
        }
        self._buffers[session_id] = buffer

        # No history loading — checkpoint handles accumulation
        history = self._store.get_messages(session_id)

        def _run():
            gen = stream_response(
                history,
                self._config,
                web_search=web_search,
                thread_id=session_id,
                context_compression=context_compression,
                rag_engine=self._rag,
                memory_store=self._memory,
            )
            resume_value = None
            try:
                while True:
                    try:
                        if resume_value is not None:
                            typ, val = gen.send(resume_value)
                            resume_value = None
                        else:
                            typ, val = next(gen)
                    except StopIteration:
                        break

                    if typ == "interrupt":
                        # Graph paused — wait for user to confirm via confirm_tool()
                        buffer["interrupt"] = val
                        buffer["resume_event"].wait()
                        buffer["resume_event"].clear()
                        buffer["interrupt"] = None
                        resume_value = buffer.pop("resume_value", {})
                        continue

                    if typ == "token":
                        buffer["tokens"].append(val)
                    elif typ == "search":
                        buffer["searching"] = val if isinstance(val, str) else bool(val)

                # Regenerate session title based on full conversation
                full = "".join(buffer["tokens"])
                title = generate_title(history + [{"role": "assistant", "content": full}], self._config)
                if title:
                    self._store.rename_session(session_id, title)

            except Exception as e:
                buffer["error"] = str(e)
            finally:
                buffer["done"] = True

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return {"status": "started"}

    def get_stream_output(self, session_id: str) -> dict:
        """Return accumulated tokens since last poll (called from JS)."""
        buffer = self._buffers.get(session_id)
        if not buffer:
            return {"tokens": [], "done": True}

        new_tokens = buffer["tokens"][buffer["sent_count"] :]
        buffer["sent_count"] = len(buffer["tokens"])

        searching = buffer.get("searching", False)
        done = buffer["done"]
        result: dict[str, Any] = {
            "tokens": new_tokens,
            "done": done,
            "searching": searching,
        }

        # Propagate interrupt data to frontend
        if buffer.get("interrupt"):
            result["interrupt"] = buffer["interrupt"]

        if buffer.get("error") and done:
            result["error"] = buffer["error"]

        # When stream ends, persist the assistant message and clean up
        if done:
            full_text = "".join(buffer["tokens"])
            self._store.add_message(session_id, "assistant", full_text)
            self._buffers.pop(session_id, None)

        return result

    # ── interrupt confirmation ────────────────────────────────────

    def confirm_tool(self, session_id: str, approved: bool = True) -> dict:
        """Resume the graph after an interrupt with user's decision.

        Called from JS when the user confirms or cancels a tool call.
        """
        buffer = self._buffers.get(session_id)
        if not buffer or buffer.get("interrupt") is None:
            return {"status": "no_interrupt"}

        buffer["resume_value"] = "approved" if approved else "rejected"
        buffer["resume_event"].set()
        return {"status": "resumed"}

    # ── memory management ────────────────────────────────────────────

    def get_memories(self) -> list[dict]:
        """返回所有跨会话记忆。"""
        return self._memory.search(limit=50)

    def delete_memory(self, namespace: str, key: str) -> dict:
        """删除一条记忆。"""
        ok = self._memory.delete(namespace, key)
        return {"status": "ok" if ok else "not_found"}

    # ── document management ─────────────────────────────────────────

    def upload_documents(self) -> list[dict]:
        """打开文件对话框 → 解析索引 → 返回文档列表。"""
        import webview

        window = webview.active_window()
        files = window.create_file_dialog(
            webview.FileDialog.OPEN,
            allow_multiple=True,
            file_types=(
                "文档 (*.pdf;*.txt;*.md;*.docx)",
                "所有文件 (*.*)",
            ),
        )
        if not files:
            return []

        results = []
        for f in files:
            info = self._rag.add_document(f)
            results.append(info)
        return results

    def get_documents(self) -> list[dict]:
        """返回已索引的文档列表。"""
        return self._rag.list_documents()

    def get_rag_status(self) -> dict:
        """返回 RAG 引擎状态（模型是否加载完成）。"""
        status = "ready" if self._rag.model_ready else "loading"
        error = self._rag.model_error
        if error:
            status = "error"
        return {"status": status, "error": error}

    def delete_document(self, doc_id: str) -> dict:
        """删除文档索引。"""
        ok = self._rag.delete_document(doc_id)
        return {"status": "ok" if ok else "not_found"}
