"""Cross-session memory store — JSON-file backed."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MemoryStore:
    """跨会话持久化记忆存储。

    与 ``SessionStore``（按会话存消息）不同，``MemoryStore`` 按命名空间存储键值对，
    所有会话共享。适用场景：

    - ``user_prefs`` — 用户偏好（语言、称呼等）
    - ``facts`` — 学到的用户信息（"养了一只猫叫咪咪"）
    - ``summaries`` — 历史会话摘要

    Layout::
        <data_dir>/memories.json
    """

    def __init__(self, data_dir: str):
        self._path = Path(data_dir) / "memories.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._write({"memories": []})

    # ── 读写 ─────────────────────────────────────────────────────────

    def _read(self) -> list[dict]:
        with open(self._path, "r", encoding="utf-8") as f:
            return json.load(f)["memories"]

    def _write(self, data: dict):
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ── 核心 API ────────────────────────────────────────────────────

    def put(self, namespace: str, key: str, value: str):
        """保存或更新一条记忆。"""
        memories = self._read()
        now = datetime.now(timezone.utc).isoformat()
        for m in memories:
            if m["namespace"] == namespace and m["key"] == key:
                m["value"] = value
                m["updated_at"] = now
                self._write({"memories": memories})
                return
        memories.append({
            "namespace": namespace,
            "key": key,
            "value": value,
            "created_at": now,
            "updated_at": now,
        })
        self._write({"memories": memories})

    def get(self, namespace: str, key: str) -> dict | None:
        """获取单条记忆。"""
        for m in self._read():
            if m["namespace"] == namespace and m["key"] == key:
                return m
        return None

    def search(
        self,
        namespace: str | None = None,
        query: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """搜索记忆。

        Args:
            namespace: 过滤命名空间（如 ``"user_prefs"``），None 表示不限。
            query: 关键词匹配 value（子串匹配，大小写不敏感）。
            limit: 返回上限。
        """
        result = self._read()
        if namespace:
            result = [m for m in result if m["namespace"] == namespace]
        if query:
            lower = query.lower()
            result = [
                m for m in result
                if lower in m["key"].lower() or lower in m["value"].lower()
            ]
        result.sort(key=lambda m: m["updated_at"], reverse=True)
        return result[:limit]

    def list_namespaces(self) -> list[str]:
        """列出所有命名空间。"""
        namespaces: set[str] = set()
        for m in self._read():
            namespaces.add(m["namespace"])
        return sorted(namespaces)

    def delete(self, namespace: str, key: str) -> bool:
        """删除一条记忆。"""
        memories = self._read()
        new = [m for m in memories if not (m["namespace"] == namespace and m["key"] == key)]
        if len(new) == len(memories):
            return False
        self._write({"memories": new})
        return True

    # ── 辅助 ────────────────────────────────────────────────────────

    def format_for_prompt(self, namespace: str | None = None) -> str:
        """将记忆格式化为提示词片段，供注入 SystemPrompt 使用。"""
        items = self.search(namespace=namespace, limit=20)
        if not items:
            return ""
        lines = []
        for m in items:
            ns = m["namespace"]
            lines.append(f"  [{ns}] {m['key']}: {m['value']}")
        return "已知信息：\n" + "\n".join(lines)
