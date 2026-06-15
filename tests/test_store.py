"""Tests for MemoryStore — cross-session JSON-file backed memory."""

from __future__ import annotations

from agent.store import MemoryStore


# ── Init ─────────────────────────────────────────────────────────────


class TestInit:
    def test_creates_file(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        assert store._path.exists()

    def test_empty_store(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        assert store.search() == []
        assert store.list_namespaces() == []


# ── Put / Get ────────────────────────────────────────────────────────


class TestPutGet:
    def test_put_new_memory(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        store.put("user_prefs", "language", "Chinese")
        m = store.get("user_prefs", "language")
        assert m is not None
        assert m["value"] == "Chinese"
        assert m["namespace"] == "user_prefs"

    def test_put_updates_existing(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        store.put("user_prefs", "language", "Chinese")
        store.put("user_prefs", "language", "English")
        m = store.get("user_prefs", "language")
        assert m["value"] == "English"

    def test_put_updates_timestamp(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        store.put("ns", "k", "v1")
        m1 = store.get("ns", "k")
        store.put("ns", "k", "v2")
        m2 = store.get("ns", "k")
        assert m2["updated_at"] >= m1["updated_at"]

    def test_put_adds_timestamps(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        store.put("ns", "k", "v")
        m = store.get("ns", "k")
        assert "created_at" in m
        assert "updated_at" in m

    def test_get_nonexistent(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        assert store.get("none", "x") is None

    def test_get_wrong_namespace(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        store.put("ns1", "k", "v")
        assert store.get("ns2", "k") is None


# ── Search ───────────────────────────────────────────────────────────


class TestSearch:
    def test_search_all(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        store.put("ns1", "k1", "v1")
        store.put("ns2", "k2", "v2")
        results = store.search()
        assert len(results) == 2

    def test_search_by_namespace(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        store.put("ns1", "k1", "v1")
        store.put("ns2", "k2", "v2")
        results = store.search(namespace="ns1")
        assert len(results) == 1
        assert results[0]["key"] == "k1"

    def test_search_by_query(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        store.put("ns1", "name", "Alice")
        store.put("ns2", "pet", "cat")
        results = store.search(query="Alice")
        assert len(results) == 1
        assert results[0]["key"] == "name"

    def test_search_by_query_matches_key(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        store.put("ns1", "favorite_color", "blue")
        results = store.search(query="favorite")
        assert len(results) == 1

    def test_search_combined(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        store.put("ns1", "name", "Alice")
        store.put("ns1", "age", "30")
        store.put("ns2", "name", "Bob")
        results = store.search(namespace="ns1", query="Alice")
        assert len(results) == 1

    def test_search_returns_most_recent_first(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        store.put("ns", "a", "1")
        store.put("ns", "b", "2")
        results = store.search()
        assert results[0]["key"] == "b"

    def test_search_limit(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        for i in range(5):
            store.put("ns", str(i), str(i))
        results = store.search(limit=3)
        assert len(results) == 3

    def test_search_no_match(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        store.put("ns", "k", "v")
        assert store.search(query="zzz") == []


# ── Delete ────────────────────────────────────────────────────────────


class TestDelete:
    def test_delete_existing(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        store.put("ns", "k", "v")
        assert store.delete("ns", "k") is True
        assert store.search() == []

    def test_delete_nonexistent(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        assert store.delete("ns", "k") is False

    def test_delete_wrong_namespace(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        store.put("ns1", "k", "v")
        assert store.delete("ns2", "k") is False
        assert store.get("ns1", "k") is not None

    def test_delete_only_matching(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        store.put("ns", "a", "1")
        store.put("ns", "b", "2")
        store.delete("ns", "a")
        assert store.get("ns", "a") is None
        assert store.get("ns", "b") is not None


# ── Namespaces ───────────────────────────────────────────────────────


class TestListNamespaces:
    def test_unique_sorted(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        store.put("zzz", "k", "v")
        store.put("aaa", "k", "v")
        store.put("zzz", "k2", "v2")  # duplicate should not appear twice
        assert store.list_namespaces() == ["aaa", "zzz"]

    def test_empty(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        assert store.list_namespaces() == []


# ── Format for prompt ────────────────────────────────────────────────


class TestFormatForPrompt:
    def test_empty(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        assert store.format_for_prompt() == ""

    def test_formats_memories(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        store.put("user_prefs", "language", "Chinese")
        store.put("facts", "pet", "cat")
        result = store.format_for_prompt()
        assert "已知信息：" in result
        assert "[user_prefs]" in result
        assert "[facts]" in result
        assert "language: Chinese" in result

    def test_filtered_by_namespace(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        store.put("user_prefs", "a", "1")
        store.put("other", "b", "2")
        result = store.format_for_prompt(namespace="user_prefs")
        assert "[user_prefs]" in result
        assert "[other]" not in result

    def test_respects_limit(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        for i in range(25):
            store.put("ns", str(i), str(i))
        result = store.format_for_prompt()
        # 20 items max
        assert result.count("[ns]") <= 20
