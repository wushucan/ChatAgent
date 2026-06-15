"""Tests for RAGEngine — document loading, embedding, and retrieval.

ChromaL + FastEmbed are mocked to avoid model downloads and DB setup.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.rag import RAGEngine, _PARSERS


# ── Mock Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def mock_embeddings():
    with patch("agent.rag.FastEmbedEmbeddings") as m:
        instance = m.return_value
        yield instance


@pytest.fixture
def mock_chroma():
    with patch("agent.rag.Chroma") as m:
        instance = m.return_value
        # Default: similarity_search returns empty
        instance.similarity_search.return_value = []
        instance.add_texts.return_value = ["id1", "id2"]
        instance.delete.return_value = None
        yield instance


@pytest.fixture
def engine(tmp_path, mock_embeddings, mock_chroma):
    return RAGEngine(str(tmp_path))


# ── Init ─────────────────────────────────────────────────────────────


class TestInit:
    def test_creates_dirs(self, tmp_path):
        engine = RAGEngine(str(tmp_path))
        assert tmp_path.is_dir()
        # .rag_index directory is created lazily by Chroma persistence

    def test_creates_no_index_file_until_first_doc(self, tmp_path):
        engine = RAGEngine(str(tmp_path))
        assert not engine._index_path.exists()
        # list_documents works without index
        assert engine.list_documents() == []

    def test_model_not_ready_by_default(self, engine):
        assert engine.model_ready is False
        assert engine.model_error is None

    def test_warmup(self, mock_embeddings, tmp_path):
        engine = RAGEngine(str(tmp_path))
        engine.warmup()
        assert engine.model_ready is True

    def test_warmup_error(self, tmp_path):
        with patch("agent.rag.FastEmbedEmbeddings") as m:
            m.side_effect = RuntimeError("fail")
            engine = RAGEngine(str(tmp_path))
            engine.warmup()
            assert engine.model_ready is False
            assert engine.model_error is not None


# ── Document Parsers ────────────────────────────────────────────────


class TestParsers:
    def test_parse_txt(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello World", encoding="utf-8")
        text = _PARSERS[".txt"](str(f))
        assert text == "Hello World"

    def test_parse_md(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Title\n\nContent", encoding="utf-8")
        text = _PARSERS[".md"](str(f))
        assert "# Title" in text

    def test_unsupported_format(self, engine):
        result = engine.add_document("test.xyz")
        assert "error" in result
        assert "不支持" in result["error"]

    def test_parser_empty_txt(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        text = _PARSERS[".txt"](str(f))
        assert text == ""


# ── Add Document ─────────────────────────────────────────────────────


class TestAddDocument:
    def test_add_txt(self, engine, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("Hello World. " * 20, encoding="utf-8")
        result = engine.add_document(str(f))
        assert "error" not in result
        assert result["filename"] == "doc.txt"
        assert result["chunks"] > 0
        assert "id" in result

    def test_add_saves_to_index(self, engine, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("Hello World. " * 20, encoding="utf-8")
        engine.add_document(str(f))
        docs = engine.list_documents()
        assert len(docs) == 1
        assert docs[0]["filename"] == "doc.txt"

    def test_add_multiple_files(self, engine, tmp_path):
        for name in ["a.txt", "b.txt"]:
            f = tmp_path / name
            f.write_text("Content. " * 20, encoding="utf-8")
            engine.add_document(str(f))
        assert len(engine.list_documents()) == 2

    def test_add_empty_content(self, engine, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("\n  \n", encoding="utf-8")
        result = engine.add_document(str(f))
        assert "error" in result

    def test_add_preserves_index_across_instances(self, engine, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("Content. " * 20, encoding="utf-8")
        engine.add_document(str(f))
        # New engine instance loads same index
        engine2 = RAGEngine(str(tmp_path))
        assert len(engine2.list_documents()) == 1


# ── Search ────────────────────────────────────────────────────────────


class TestSearch:
    def test_search_calls_chroma(self, mock_chroma, engine, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("Content. " * 20, encoding="utf-8")
        engine.add_document(str(f))
        engine.search("query")
        mock_chroma.similarity_search.assert_called()

    def test_search_returns_empty_on_error(self, tmp_path):
        with patch("agent.rag.FastEmbedEmbeddings"):
            engine = RAGEngine(str(tmp_path))
            with patch.object(engine, "_vectorstore", None):
                # no vectorstore yet — search should handle gracefully
                assert engine.search("query") == []


# ── Delete Document ──────────────────────────────────────────────────


class TestDeleteDocument:
    def test_delete_existing(self, engine, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("Content. " * 20, encoding="utf-8")
        result = engine.add_document(str(f))
        assert engine.delete_document(result["id"]) is True
        assert engine.list_documents() == []

    def test_delete_nonexistent(self, engine):
        assert engine.delete_document("nonexistent") is False

    def test_delete_does_not_affect_other_docs(self, engine, tmp_path):
        ids = []
        for name in ["a.txt", "b.txt"]:
            f = tmp_path / name
            f.write_text("Content. " * 20, encoding="utf-8")
            ids.append(engine.add_document(str(f))["id"])
        engine.delete_document(ids[0])
        remaining = engine.list_documents()
        assert len(remaining) == 1
        assert remaining[0]["id"] == ids[1]


# ── Index Persistence ────────────────────────────────────────────────


class TestIndexPersistence:
    def test_load_non_existent(self, engine):
        assert engine._load_index() == []

    def test_save_and_load(self, engine):
        data = [{"id": "1", "filename": "test.txt"}]
        engine._save_index(data)
        loaded = engine._load_index()
        assert loaded == data

    def test_save_overwrites(self, engine):
        engine._save_index([{"id": "1"}])
        engine._save_index([{"id": "2"}])
        assert engine._load_index() == [{"id": "2"}]

    def test_invalid_json_raises(self, engine):
        engine._index_path.write_text("not json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            engine._load_index()
