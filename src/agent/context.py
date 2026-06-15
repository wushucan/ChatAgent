"""BM25-based context window compression for LangGraph agents.

Provides a tokenizer for mixed CJK/English text and a
:class:`BM25Compressor` that selects a compact, relevant subset
of conversation history before the LLM call.
"""

from __future__ import annotations

import re
from typing import Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Tokenize mixed CJK + Latin text for BM25 indexing.

    Latin words/numbers are lower-cased and kept as whole tokens.
    CJK ideographs are split into individual characters (unigram).
    Whitespace and punctuation are discarded.
    """
    text = text.lower()
    tokens: list[str] = []
    for match in re.finditer(
        r"[a-z0-9]+|[一-鿿㐀-䶿豈-﫿]", text
    ):
        tokens.append(match.group())
    return tokens


def _extract_text(message: BaseMessage) -> str:
    """Safely extract plain text from a message, handling list content."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    return str(content)


class BM25Compressor:
    """Stateless compressor; instantiated per-call for thread safety.

    Parameters
    ----------
    recent_keep : int
        Number of most-recent messages to always retain.
    bm25_top_k : int
        Number of top-ranking historical messages to select via BM25.

    Usage::

        compressor = BM25Compressor(recent_keep=8, bm25_top_k=12)
        compact = compressor.compress(all_messages)
    """

    def __init__(
        self,
        recent_keep: int = 8,
        bm25_top_k: int = 12,
    ):
        self.recent_keep = recent_keep
        self.bm25_top_k = bm25_top_k

    def compress(self, messages: Sequence[BaseMessage]) -> list[BaseMessage]:
        """Return a compact subset of *messages* suitable for LLM invocation.

        Always keeps:
        - every ``SystemMessage`` at the front (unchanged order)
        - the last ``recent_keep`` messages ("immediate context")

        From the remaining history, selects the top ``bm25_top_k`` messages
        ranked by BM25 relevance to the latest ``HumanMessage`` query.
        """
        system_msgs: list[BaseMessage] = []
        body: list[BaseMessage] = []
        for m in messages:
            if isinstance(m, SystemMessage):
                system_msgs.append(m)
            else:
                body.append(m)

        total_body = len(body)
        too_few = total_body <= self.recent_keep + self.bm25_top_k
        if too_few:
            return system_msgs + body

        query_text = ""
        for msg in reversed(body):
            if isinstance(msg, HumanMessage):
                query_text = _extract_text(msg)
                break
        if not query_text:
            query_text = _extract_text(body[-1])

        query_tokens = tokenize(query_text)

        recent = body[-self.recent_keep :]
        history = body[: -self.recent_keep]

        if not history:
            return system_msgs + recent

        corpus = [tokenize(_extract_text(m)) for m in history]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query_tokens)

        scored = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        keep_indices = set(scored[: self.bm25_top_k])
        selected = [history[i] for i in sorted(keep_indices)]

        return system_msgs + selected + recent
