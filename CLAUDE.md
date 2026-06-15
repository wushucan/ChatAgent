# ChatAgent 项目指南

## 项目简介

基于 LangGraph + DeepSeek 的桌面 AI 助手。使用 PyWebview 构建原生桌面窗口，支持多会话、文档知识库（RAG）、跨会话记忆、联网搜索。

## 技术栈

- **桌面窗口**: PyWebview（Chromium 内核）
- **AI 模型**: DeepSeek（deepseek-v4-flash，OpenAI 兼容 API）
- **Agent 框架**: LangGraph（StateGraph + ReAct 模式）
- **向量存储**: Chroma + FastEmbed（BAAI/bge-small-zh-v1.5）
- **联网搜索**: Tavily Search API
- **前端**: 原生 HTML + CSS + JS（深色主题）
- **消息持久化**: JSON 文件

## 关键文件

| 文件 | 作用 |
|------|------|
| `run_chat.py` | 入口，启动 PyWebview 窗口 |
| `src/agent/graph.py` | LangGraph StateGraph（ReAct Agent，工具绑定） |
| `src/agent/api.py` | PyWebview JS API 桥接 |
| `src/agent/desktop.py` | 流式响应封装 |
| `src/agent/persist.py` | 多会话 JSON 持久化 |
| `src/agent/rag.py` | RAG 文档知识库引擎 |
| `src/agent/store.py` | 跨会话记忆存储 |
| `src/agent/context.py` | BM25 上下文压缩 |
| `frontend/index.html` | 桌面 UI（所有界面 + JS 逻辑） |
| `tests/` | 单元测试（94 个，覆盖率 75%-100%） |

## 运行

```bash
conda activate langgraph_env
python run_chat.py
```

## 测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行测试 + 覆盖率报告
pytest tests/ --cov=agent --cov-report=html

# 打开覆盖率报告
# htmlcov/index.html
```

## 打包

```bash
conda activate langgraph_env
rm -rf dist build_pkg
pyinstaller --onefile --windowed --name "ChatAgent" \
  --add-data "frontend;frontend" \
  --add-data "config.json;." \
  --add-data "src/agent;agent" \
  --paths "src" \
  --collect-all "chromadb" \
  --collect-all "langchain_community" \
  --collect-all "langchain_core" \
  --hidden-import "webview" \
  --hidden-import "fastembed" \
  --hidden-import "fitz" \
  --hidden-import "langgraph" \
  --hidden-import "langchain_deepseek" \
  --hidden-import "tavily" \
  --hidden-import "rank_bm25" \
  --exclude "torch" \
  --exclude "sentence_transformers" \
  run_chat.py
```

## 架构要点

1. **手写 StateGraph**: 非 `create_react_agent` 预置函数，显式定义 `call_model` / `call_tool` 节点和 `should_continue` 条件边
2. **流式处理**: 使用 LangGraph `stream_mode="messages"`，通过 Python 生成器 yield 事件，JS 每 60ms 轮询 `get_stream_output()`
3. **工具中断**: `rag_search` / `web_search` 走 `interrupt()` 等待用户确认；`save_memory` 静默执行
4. **RAG**: FastEmbedEmbeddings（ONNX）+ Chroma，支持 PDF/TXT/MD/DOCX
5. **上下文压缩**: 消息数超过阈值时用 BM25 压缩历史

## 工具

- `web_search(query)` — Tavily 联网搜索
- `rag_search(query)` — 文档知识库检索
- `save_memory(namespace, key, value)` — 跨会话记忆

## 配置

`config.json`（优先）或 `.env` 文件：

| 配置项 | 说明 |
|--------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API Key |
| `DEEPSEEK_MODEL` | 模型名（默认 deepseek-v4-flash） |
| `TAVILY_API_KEY` | Tavily 搜索 API Key |
| `SYSTEM_PROMPT` | 系统提示词 |

参考模板文件：`config.example.json` / `.env.example`

## 注意事项

- 国内需设置 `HF_ENDPOINT=https://hf-mirror.com`（已在 rag.py 中预设）
- 首次运行会在后台下载 embedding 模型（~30MB）
- PyInstaller 打包后 exe 约 276MB，首次解压约 5 秒
- 打包时用 `--collect-all` 处理 chromadb 等动态导入的包，避免运行时报缺模块
- `config.json` 和 `.env` 含有 API Key，已加入 `.gitignore`
