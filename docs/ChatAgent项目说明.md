# ChatAgent 项目说明

## 概述

基于 LangGraph + DeepSeek 的桌面 AI 助手。使用 PyWebview 构建原生桌面窗口，支持多会话、文档知识库（RAG）、跨会话记忆、联网搜索等功能。

---

## 技术栈

| 层 | 技术 |
|------|------|
| 桌面窗口 | PyWebview（Chromium 内核） |
| AI 模型 | DeepSeek（deepseek-v4-flash，OpenAI 兼容 API） |
| Agent 框架 | LangGraph（StateGraph + ReAct 模式） |
| 文档检索 | Sentence-Transformers + BM25 混合检索 |
| 联网搜索 | Tavily Search API |
| 前端 | 原生 HTML + CSS + JS（深色主题） |
| 消息持久化 | JSON 文件 |
| 跨会话记忆 | JSON 文件（MemoryStore） |

---

## 项目结构

```
demo-agent/
├── run_chat.py              # 入口：启动 PyWebview 窗口
├── config.json              # 配置（API Key / 模型 / 系统提示词）
├── .env                     # 环境变量（可替代 config.json）
├── pyproject.toml           # 项目元数据 + 依赖
├── .gitignore
├── LICENSE                  # MIT
├── docs/
│   ├── ChatAgent操作手册.md  # 用户使用手册
│   └── ChatAgent项目说明.md  # 本文件（项目技术说明）
├── frontend/
│   └── index.html           # 桌面 UI（所有界面 + JS 逻辑）
└── src/agent/
    ├── __init__.py           # 导出 graph 供 LangGraph Server
    ├── graph.py              # LangGraph StateGraph（ReAct Agent）
    ├── desktop.py            # 流式响应封装（token 流 + interrupt 处理）
    ├── api.py                # PyWebview JS API 桥接
    ├── persist.py            # 多会话 JSON 持久化
    ├── context.py            # BM25 上下文压缩
    ├── store.py              # 跨会话记忆存储
    └── rag.py                # RAG 文档知识库引擎
```

---

## 核心架构

### Agent 图结构（`graph.py`）

```
┌─────────┐    tool_calls?    ┌──────────┐
│  model  │ ───────────────→  │  tools   │
│  (LLM)  │                   │ (执行)   │
└────┬────┘ ←────────────────└──────────┘
     │                       返回结果
     │ 无工具调用
     ▼
    END
```

- **StateGraph** 手写实现，而非 `create_react_agent` 预置函数
- **`call_model` 节点**：调用 DeepSeek，可选 BM25 上下文压缩
- **`call_tool` 节点**：执行工具调用（web_search / rag_search / save_memory）
- **`should_continue` 条件边**：根据 LLM 是否调工具决定路由

### 流式处理（`desktop.py`）

- 使用 LangGraph `stream_mode="messages"` 实现逐 token 流式输出
- 工具调用中断走 `interrupt()` 机制，等待用户确认后恢复
- 通过 Python 生成器 `yield` 向 UI 层推送事件

### JS API 桥接（`api.py`）

PyWebview 自动将 `AgentAPI` 类的同步方法暴露为 `pywebview.api.*`：

| JS 调用 | 功能 |
|---------|------|
| `get_sessions()` | 获取会话列表 |
| `create_session()` | 新建会话 |
| `delete_session(id)` | 删除会话 |
| `get_messages(id)` | 获取消息列表 |
| `send_message(id, msg, web_search, context_compression)` | 发送消息（异步流式） |
| `get_stream_output(id)` | 轮询获取流式输出 |
| `delete_message(id, index)` | 删除单条消息 |
| `rename_session(id, title)` | 重命名会话 |
| `confirm_tool(id, approved)` | 确认/拒绝工具调用 |
| `upload_documents()` | 上传文档 |
| `get_documents()` | 获取文档列表 |
| `delete_document(id)` | 删除文档 |
| `get_rag_status()` | 查询 RAG 引擎状态 |
| `get_memories()` | 获取记忆列表 |
| `delete_memory(namespace, key)` | 删除记忆 |

### 流式数据流

```
用户输入 → JS send_message()
  → API send_message() → 启动后台线程
    → graph.stream() → token / interrupt / search 事件
      → buffer 累积
        → JS 每 60ms 轮询 get_stream_output()
          → 逐 token 渲染到界面
```

---

## 关键依赖

| 包 | 用途 |
|-----|------|
| `langgraph` | Agent 框架、StateGraph、checkpoint |
| `langchain-deepseek` | DeepSeek LLM 调用 |
| `pywebview` | 桌面窗口 |
| `sentence-transformers` | RAG 文档向量化 |
| `rank-bm25` | BM25 上下文压缩 |
| `PyMuPDF` | PDF 解析 |
| `python-docx` | DOCX 解析 |
| `tavily-python` | 联网搜索 |
| `python-dotenv` | .env 配置加载 |

---

## 打包指南

### PyInstaller

```bash
pip install pyinstaller
pyinstaller --onefile --windowed ^
  --add-data "frontend;frontend" ^
  --add-data "config.json;." ^
  --icon icon.ico ^
  run_chat.py
```

注意：
- `frontend/index.html` 必须作为数据文件打包
- `sessions/` 目录在运行时自动创建，不需要打包
- `.env` 和 `config.json` 建议分发给用户自行配置

### 依赖处理

`sentence-transformers` 和 `PyMuPDF` 体积较大，PyInstaller 打包时可能需要额外配置 hidden-import：

```bash
--hidden-import fitz --hidden-import sentence_transformers
```

---

## 开发

```bash
# 安装
conda create -n langgraph_env python=3.12
conda activate langgraph_env
pip install -e .
pip install pywebview tavily-python

# 运行
python run_chat.py

# Ruff 代码检查
python -m ruff check .

# Ruff 格式化
python -m ruff format .
```
