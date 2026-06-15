"""Multi-session persistence — JSON-file backed."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone


class SessionStore:
    """Manage chat sessions stored as JSON files.

    Layout::
        <data_dir>/
            index.json       # {sessions: [{id, title, created_at, updated_at}]}
            <id>.json        # {messages: [{role, content, timestamp}]}
    """

    def __init__(self, data_dir: str = "sessions"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self._index_path = os.path.join(data_dir, "index.json")
        self._ensure_index()

    # ── index helpers ──────────────────────────────────────────────

    def _ensure_index(self):
        if not os.path.isfile(self._index_path):
            self._write_index([])

    def _read_index(self) -> list[dict]:
        with open(self._index_path, "r", encoding="utf-8") as f:
            return json.load(f)["sessions"]

    def _write_index(self, sessions: list[dict]):
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump({"sessions": sessions}, f, ensure_ascii=False, indent=2)

    def _session_path(self, session_id: str) -> str:
        if not session_id or not isinstance(session_id, str):
            raise ValueError(f"Invalid session_id: {session_id!r}")
        return os.path.join(self.data_dir, f"{session_id}.json")

    # ── public API ─────────────────────────────────────────────────

    def list_sessions(self) -> list[dict]:
        sessions = self._read_index()
        sessions.sort(key=lambda s: s["updated_at"], reverse=True)
        return sessions

    def create_session(self) -> dict:
        sessions = self._read_index()
        now = datetime.now(timezone.utc).isoformat()
        sid = uuid.uuid4().hex[:8]
        entry = {
            "id": sid,
            "title": "新会话",
            "created_at": now,
            "updated_at": now,
        }
        # empty message file
        with open(self._session_path(sid), "w", encoding="utf-8") as f:
            json.dump({"messages": []}, f, ensure_ascii=False)
        sessions.insert(0, entry)
        self._write_index(sessions)
        return entry

    def delete_session(self, session_id: str):
        sessions = [s for s in self._read_index() if s["id"] != session_id]
        self._write_index(sessions)
        path = self._session_path(session_id)
        if os.path.isfile(path):
            os.remove(path)

    def get_messages(self, session_id: str) -> list[dict]:
        path = self._session_path(session_id)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)["messages"]
        return []

    def add_message(self, session_id: str, role: str, content: str):
        path = self._session_path(session_id)
        data: dict = {"messages": []}
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

        now = datetime.now(timezone.utc).isoformat()
        msg = {"role": role, "content": content, "timestamp": now}
        data["messages"].append(msg)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Auto-title from first user message
        title_update = None
        if role == "user" and len(data["messages"]) == 1:
            truncated = content[:28]
            title_update = truncated + ("..." if len(content) > 28 else "")

        self._touch_session(session_id, title=title_update)

    def rename_session(self, session_id: str, title: str):
        self._touch_session(session_id, title=title)

    def delete_message(self, session_id: str, index: int) -> bool:
        """删除指定会话中的第 index 条消息（0 -based）。"""
        path = self._session_path(session_id)
        if not os.path.isfile(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        msgs = data.get("messages", [])
        if index < 0 or index >= len(msgs):
            return False
        msgs.pop(index)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self._touch_session(session_id)
        return True

    def _touch_session(self, session_id: str, title: str | None = None):
        sessions = self._read_index()
        now = datetime.now(timezone.utc).isoformat()
        for s in sessions:
            if s["id"] == session_id:
                s["updated_at"] = now
                if title is not None:
                    s["title"] = title
                break
        self._write_index(sessions)
