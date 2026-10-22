from __future__ import annotations

from typing import Any

from mlops.agents.rag.retriever import ContextualRetriever
from mlops.agents.rag.schemas import KnowledgeChunk, RetrievalResult


def _chunk(
    chunk_id: str,
    doc_id: str,
    section: str,
    text: str,
) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        title=f"Documento {doc_id}",
        section=section,
        text=text,
        citation=f"Documento {doc_id} > {section} (v1)",
    )


class FakeRetriever:
    def __init__(self, chunks: list[KnowledgeChunk], order: list[int]):
        self.chunks = chunks
        self.order = order

    def search(self, query: str, k: int = 5, **kwargs: Any) -> list[RetrievalResult]:
        return [
            RetrievalResult(chunk=self.chunks[index], score=1.0 - rank * 0.1)
            for rank, index in enumerate(self.order[:k])
        ]

    def citations(self, results: Any) -> list[dict[str, str]]:
        return [
            {"doc_id": result.chunk.doc_id, "section": result.chunk.section}
            for result in results
        ]


def test_contextual_retriever_expands_neighbouring_chunks() -> None:
    chunks = [
        _chunk("doc_a:sec-1:0", "doc_a", "Seção 1", "Contexto anterior."),
        _chunk("doc_a:sec-2:0", "doc_a", "Seção 2", "Trecho em foco."),
        _chunk("doc_a:sec-3:0", "doc_a", "Seção 3", "Contexto posterior."),
        _chunk("doc_b:sec-1:0", "doc_b", "Seção 1", "Outro documento."),
    ]
    retriever = ContextualRetriever(
        FakeRetriever(chunks, order=[1, 3]),
        window=1,
        max_context_chars=1200,
        candidate_multiplier=2,
    )

    results = retriever.search("foco", k=2)

    assert [result.chunk.chunk_id for result in results] == ["doc_a:sec-2:0", "doc_b:sec-1:0"]
    assert results[0].context_chunk_ids == ["doc_a:sec-1:0", "doc_a:sec-2:0", "doc_a:sec-3:0"]
    assert "Contexto anterior." in results[0].context
    assert "Contexto posterior." in results[0].context
    assert "Outro documento." not in results[0].context
    assert results[1].context_chunk_ids == ["doc_b:sec-1:0"]


def test_contextual_retriever_formats_context_and_citations() -> None:
    chunks = [
        _chunk("doc_a:sec-1:0", "doc_a", "Seção 1", "Texto principal."),
    ]
    retriever = ContextualRetriever(
        FakeRetriever(chunks, order=[0]),
        window=1,
        max_context_chars=1200,
    )

    results = retriever.search("principal", k=1)
    context = retriever.format_context(results)
    citations = retriever.citations(results)

    assert "[doc_a | Seção 1]" in context
    assert "Documento: Documento doc_a (doc_a, versão n/a)" in context
    assert "Fonte: Documento doc_a > Seção 1 (v1)" in context
    assert citations[0]["doc_id"] == "doc_a"
    assert results[0].to_dict()["context_chunk_ids"] == ["doc_a:sec-1:0"]


def test_contextual_retriever_respects_context_budget() -> None:
    chunks = [
        _chunk("doc_a:sec-1:0", "doc_a", "Seção 1", "A" * 400),
        _chunk("doc_a:sec-2:0", "doc_a", "Seção 2", "B" * 100),
        _chunk("doc_a:sec-3:0", "doc_a", "Seção 3", "C" * 400),
    ]
    retriever = ContextualRetriever(
        FakeRetriever(chunks, order=[1]),
        window=1,
        max_context_chars=650,
    )

    result = retriever.search("foco", k=1)[0]

    assert "B" * 100 in result.context
    assert len(result.context) <= 650
    assert "A" * 400 not in result.context or "C" * 400 not in result.context
