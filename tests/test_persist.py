"""Tests for SessionStore — JSON-file backed session persistence."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pytest

from agent.persist import SessionStore


# ── Helpers ──────────────────────────────────────────────────────────


def _read_index(data_dir: str) -> list[dict]:
    path = os.path.join(data_dir, "index.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["sessions"]


def _read_session(data_dir: str, session_id: str) -> dict:
    path = os.path.join(data_dir, f"{session_id}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Init ─────────────────────────────────────────────────────────────


class TestInit:
    def test_creates_data_dir(self, tmp_path):
        d = str(tmp_path / "sessions")
        SessionStore(d)
        assert os.path.isdir(d)

    def test_creates_index(self, tmp_path):
        d = str(tmp_path / "sessions")
        SessionStore(d)
        assert os.path.isfile(os.path.join(d, "index.json"))

    def test_empty_index(self, tmp_path):
        d = str(tmp_path / "sessions")
        store = SessionStore(d)
        assert store.list_sessions() == []

    def test_existing_index_reused(self, tmp_path):
        d = str(tmp_path / "sessions")
        os.makedirs(d)
        with open(os.path.join(d, "index.json"), "w", encoding="utf-8") as f:
            json.dump({"sessions": [{"id": "abc", "updated_at": "2025-01-01T00:00:00"}]}, f)
        store = SessionStore(d)
        assert len(store.list_sessions()) == 1


class TestCreateSession:
    def test_returns_entry_with_fields(self, tmp_path):
        store = SessionStore(str(tmp_path))
        entry = store.create_session()
        assert "id" in entry
        assert entry["title"] == "新会话"
        assert "created_at" in entry
        assert "updated_at" in entry

    def test_adds_to_index(self, tmp_path):
        store = SessionStore(str(tmp_path))
        entry = store.create_session()
        sessions = _read_index(str(tmp_path))
        assert len(sessions) == 1
        assert sessions[0]["id"] == entry["id"]

    def test_creates_message_file(self, tmp_path):
        store = SessionStore(str(tmp_path))
        entry = store.create_session()
        data = _read_session(str(tmp_path), entry["id"])
        assert data == {"messages": []}

    def test_multiple_sessions_ordered(self, tmp_path):
        store = SessionStore(str(tmp_path))
        s1 = store.create_session()
        s2 = store.create_session()
        sessions = store.list_sessions()
        assert sessions[0]["id"] == s2["id"]  # newest first
        assert sessions[1]["id"] == s1["id"]


class TestDeleteSession:
    def test_removes_from_index(self, tmp_path):
        store = SessionStore(str(tmp_path))
        e = store.create_session()
        store.delete_session(e["id"])
        assert store.list_sessions() == []

    def test_removes_message_file(self, tmp_path):
        store = SessionStore(str(tmp_path))
        e = store.create_session()
        path = os.path.join(str(tmp_path), f"{e['id']}.json")
        assert os.path.isfile(path)
        store.delete_session(e["id"])
        assert not os.path.isfile(path)

    def test_delete_nonexistent_does_not_error(self, tmp_path):
        store = SessionStore(str(tmp_path))
        store.delete_session("nonexistent")  # should not raise


class TestMessages:
    def test_add_and_get_messages(self, tmp_path):
        store = SessionStore(str(tmp_path))
        e = store.create_session()
        store.add_message(e["id"], "user", "你好")
        store.add_message(e["id"], "assistant", "你好！")
        msgs = store.get_messages(e["id"])
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "你好"
        assert msgs[1]["role"] == "assistant"

    def test_message_has_timestamp(self, tmp_path):
        store = SessionStore(str(tmp_path))
        e = store.create_session()
        store.add_message(e["id"], "user", "hi")
        msgs = store.get_messages(e["id"])
        assert "timestamp" in msgs[0]

    def test_get_messages_empty_session(self, tmp_path):
        store = SessionStore(str(tmp_path))
        e = store.create_session()
        assert store.get_messages(e["id"]) == []

    def test_get_messages_nonexistent(self, tmp_path):
        store = SessionStore(str(tmp_path))
        assert store.get_messages("nope") == []

    def test_delete_message(self, tmp_path):
        store = SessionStore(str(tmp_path))
        e = store.create_session()
        store.add_message(e["id"], "user", "a")
        store.add_message(e["id"], "user", "b")
        store.add_message(e["id"], "user", "c")
        assert store.delete_message(e["id"], 1) is True
        msgs = store.get_messages(e["id"])
        assert [m["content"] for m in msgs] == ["a", "c"]

    def test_delete_message_out_of_range(self, tmp_path):
        store = SessionStore(str(tmp_path))
        e = store.create_session()
        assert store.delete_message(e["id"], 0) is False

    def test_delete_message_nonexistent_session(self, tmp_path):
        store = SessionStore(str(tmp_path))
        assert store.delete_message("nope", 0) is False


class TestAutoTitle:
    def test_first_user_message_sets_title(self, tmp_path):
        store = SessionStore(str(tmp_path))
        e = store.create_session()
        store.add_message(e["id"], "user", "今天天气怎么样")
        sessions = store.list_sessions()
        assert sessions[0]["title"].startswith("今天天气怎么样")

    def test_assistant_message_does_not_set_title(self, tmp_path):
        store = SessionStore(str(tmp_path))
        e = store.create_session()
        store.add_message(e["id"], "assistant", "回复")
        sessions = store.list_sessions()
        assert sessions[0]["title"] == "新会话"

    def test_title_truncated(self, tmp_path):
        store = SessionStore(str(tmp_path))
        e = store.create_session()
        long_msg = "A" * 50
        store.add_message(e["id"], "user", long_msg)
        sessions = store.list_sessions()
        title = sessions[0]["title"]
        assert len(title) <= 31  # 28 + "..."
        assert title.endswith("...")
        assert title.startswith("A" * 28)


class TestRenameSession:
    def test_rename(self, tmp_path):
        store = SessionStore(str(tmp_path))
        e = store.create_session()
        store.rename_session(e["id"], "新标题")
        sessions = store.list_sessions()
        assert sessions[0]["title"] == "新标题"

    def test_rename_updates_timestamp(self, tmp_path):
        store = SessionStore(str(tmp_path))
        e = store.create_session()
        old = store.list_sessions()[0]["updated_at"]
        store.rename_session(e["id"], "标题")
        new = store.list_sessions()[0]["updated_at"]
        assert new >= old


class TestEdgeCases:
    def test_invalid_session_id_raises(self, tmp_path):
        store = SessionStore(str(tmp_path))
        with pytest.raises(ValueError):
            store._session_path("")
        with pytest.raises(ValueError):
            store._session_path(None)  # type: ignore[arg-type]

    def test_add_message_updates_updated_at(self, tmp_path):
        store = SessionStore(str(tmp_path))
        e = store.create_session()
        old = store.list_sessions()[0]["updated_at"]
        store.add_message(e["id"], "user", "hi")
        new = store.list_sessions()[0]["updated_at"]
        assert new >= old

    def test_add_message_invalid_id_does_not_crash(self, tmp_path):
        store = SessionStore(str(tmp_path))
        # session_id "ghost" passes the validity check but isn't in index;
        # add_message should not raise
        store.add_message("ghost", "user", "hi")
        # file was created but not in index (undefined behavior, just check no crash)
        assert store.list_sessions() == []
