from __future__ import annotations

import json
import os
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from mlops.agents.rag.expand import expand_query
from mlops.agents.rag.ingest import tokenize
from mlops.agents.rag.schemas import KnowledgeChunk, RetrievalResult
from mlops.config import resolve_paths

RRF_K = 60
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
WEIGHTS_LEXICAL = {"tfidf": 0.50, "bm25": 0.50}
WEIGHTS_HYBRID = {"tfidf": 0.40, "bm25": 0.30, "embeddings": 0.30}


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    minimum, maximum = float(values.min()), float(values.max())
    if maximum - minimum < 1e-12:
        return np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)


class HybridRetriever:
    def __init__(
        self,
        index_dir: str | Path,
        use_embeddings: bool = False,
        use_reranker: bool = False,
        reranker_model: str | None = None,
        use_expansion: bool = True,
    ):
        self.index_dir = Path(index_dir)
        meta_path = self.index_dir / "index_meta.json"
        if not meta_path.exists():
            raise RuntimeError(
                f"Índice RAG ausente em {self.index_dir}. Execute `python scripts/build_rag_index.py`."
            )
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        raw_chunks = json.loads((self.index_dir / "chunks.json").read_text(encoding="utf-8"))
        self.chunks = [KnowledgeChunk(**chunk) for chunk in raw_chunks]
        self.vectorizer = joblib.load(self.index_dir / "tfidf_vectorizer.joblib")
        self.matrix = joblib.load(self.index_dir / "tfidf_matrix.joblib")
        self.bm25 = joblib.load(self.index_dir / "bm25.joblib")
        self.embeddings: np.ndarray | None = None
        self.encoder: Any = None
        self.use_embeddings = bool(use_embeddings)
        if self.use_embeddings and (self.index_dir / "embeddings.npy").exists():
            self.embeddings = np.load(self.index_dir / "embeddings.npy")
        else:
            self.use_embeddings = False
        self.use_reranker = bool(use_reranker)
        self.reranker_model = reranker_model or os.getenv("MLOPS_RERANK_MODEL", RERANK_MODEL)
        self.reranker: Any = None
        self.use_expansion = bool(use_expansion)

    def _get_reranker(self) -> Any:
        if self.reranker is None:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(self.reranker_model, max_length=512)
        return self.reranker

    def _rerank_results(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        if not results:
            return results
        encoder = self._get_reranker()
        pairs = [
            (query, f"{result.chunk.title} | {result.chunk.section}\n{result.chunk.text}")
            for result in results
        ]
        scores = np.asarray(encoder.predict(pairs), dtype=np.float64).ravel()
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            RetrievalResult(
                chunk=results[index].chunk,
                score=float(scores[index]),
                scores={**results[index].scores, "rerank": float(scores[index])},
            )
            for index in order
        ]

    def _embedding_scores(self, query: str) -> np.ndarray:
        if self.embeddings is None:
            return np.zeros(len(self.chunks))
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer

            model_name = self.meta.get(
                "embedding_model",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            )
            self.encoder = SentenceTransformer(model_name)
        query_embedding = self.encoder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]
        return np.asarray(self.embeddings @ query_embedding, dtype=np.float64)

    def _scores(self, query: str) -> dict[str, np.ndarray]:
        lexical_query = expand_query(query) if self.use_expansion else query
        tfidf = cosine_similarity(self.vectorizer.transform([lexical_query]), self.matrix).ravel()
        bm25 = _minmax(
            np.clip(np.asarray(self.bm25.get_scores(tokenize(lexical_query))), 0, None)
        )
        scores = {"tfidf": np.asarray(tfidf, dtype=np.float64), "bm25": bm25}
        if self.use_embeddings:
            scores["embeddings"] = _minmax(self._embedding_scores(query))
        weights = WEIGHTS_HYBRID if "embeddings" in scores else WEIGHTS_LEXICAL
        scores["hybrid"] = self._rrf(scores, weights)
        return scores

    def _rrf(self, scores: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
        fused = np.zeros(len(self.chunks), dtype=np.float64)
        for name, weight in weights.items():
            values = np.asarray(scores[name], dtype=np.float64)
            order = np.argsort(-values, kind="stable")
            ranks = np.empty(values.size, dtype=np.float64)
            ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
            fused += weight / (RRF_K + ranks)
        return fused

    def _redundancy(self, index: int, selected: list[int]) -> float:
        if self.embeddings is not None:
            return float(np.max(self.embeddings[selected] @ self.embeddings[index]))
        similarity = cosine_similarity(self.matrix[index], self.matrix[selected]).ravel()
        return float(np.max(similarity))

    def _select(
        self,
        candidate_indices: list[int],
        scores: np.ndarray,
        k: int,
        lambda_: float,
        max_per_doc: int,
    ) -> list[int]:
        ranked = _minmax(scores)
        selected: list[int] = []
        doc_counts: dict[str, int] = {}
        seen_sections: set[tuple[str, str]] = set()
        remaining = list(candidate_indices)
        while remaining and len(selected) < k:
            best: int | None = None
            best_value = -np.inf
            for index in remaining:
                chunk = self.chunks[index]
                section_key = (chunk.doc_id, chunk.section)
                if section_key in seen_sections:
                    continue
                if doc_counts.get(chunk.doc_id, 0) >= max_per_doc:
                    continue
                value = float(ranked[index])
                if selected:
                    value -= lambda_ * self._redundancy(index, selected)
                if value > best_value:
                    best, best_value = index, value
            if best is None:
                break
            selected.append(best)
            seen_sections.add((self.chunks[best].doc_id, self.chunks[best].section))
            doc_counts[self.chunks[best].doc_id] = doc_counts.get(self.chunks[best].doc_id, 0) + 1
            remaining.remove(best)
        return selected

    def search(
        self,
        query: str,
        k: int = 5,
        doc_type: str | None = None,
        mmr_lambda: float = 0.75,
        mode: str = "hybrid",
        max_per_doc: int = 2,
        rerank: bool | None = None,
        rerank_candidates: int = 20,
    ) -> list[RetrievalResult]:
        scores = self._scores(query)
        if mode == "embeddings" and "embeddings" not in scores:
            raise ValueError("Embeddings indisponíveis; reconstrua o índice com --embeddings.")
        if mode not in scores:
            raise ValueError(f"Modo de recuperação inválido: {mode}")
        combined = scores[mode]
        allowed = list(range(len(self.chunks)))
        if doc_type:
            allowed = [index for index in allowed if self.chunks[index].doc_type == doc_type]
        if not allowed:
            return []
        use_rerank = self.use_reranker if rerank is None else bool(rerank)
        candidate_k = max(k, rerank_candidates) if use_rerank else k
        ranked = self._select(
            allowed,
            combined,
            k=min(candidate_k, len(allowed)),
            lambda_=mmr_lambda,
            max_per_doc=max_per_doc,
        )
        results = [
            RetrievalResult(
                chunk=self.chunks[index],
                score=float(combined[index]),
                scores={name: float(values[index]) for name, values in scores.items()},
            )
            for index in ranked
        ]
        if use_rerank:
            rerank_query = expand_query(query) if self.use_expansion else query
            return self._rerank_results(rerank_query, results, top_k=k)
        return results

    def format_context(self, results: Iterable[RetrievalResult]) -> str:
        blocks = []
        for result in results:
            blocks.append(
                f"[{result.chunk.doc_id} | {result.chunk.section}]\n"
                f"{result.chunk.text}\n"
                f"Fonte: {result.chunk.citation}"
            )
        return "\n\n".join(blocks)

    def citations(self, results: Iterable[RetrievalResult]) -> list[dict[str, str]]:
        seen: set[str] = set()
        citations = []
        for result in results:
            if result.chunk.chunk_id in seen:
                continue
            seen.add(result.chunk.chunk_id)
            citations.append(
                {
                    "doc_id": result.chunk.doc_id,
                    "title": result.chunk.title,
                    "section": result.chunk.section,
                    "citation": result.chunk.citation,
                }
            )
        return citations


class ContextualRetriever:
    """Hybrid retrieval plus document-neighbour context expansion.

    The base retriever selects precise chunks. This wrapper then adds adjacent
    chunks from the same document so the language model sees the surrounding
    policy language instead of an isolated fragment.
    """

    def __init__(
        self,
        base_retriever: HybridRetriever,
        window: int = 1,
        max_context_chars: int = 2400,
        candidate_multiplier: int = 3,
    ):
        self.base = base_retriever
        self.chunks = base_retriever.chunks
        self.window = max(int(window), 0)
        self.max_context_chars = max(int(max_context_chars), 320)
        self.candidate_multiplier = max(int(candidate_multiplier), 1)
        self._doc_groups: dict[str, list[int]] = {}
        self._index_by_id: dict[str, int] = {}
        for index, chunk in enumerate(self.chunks):
            self._doc_groups.setdefault(chunk.doc_id, []).append(index)
            self._index_by_id[chunk.chunk_id] = index
        self._positions = {
            index: position
            for indices in self._doc_groups.values()
            for position, index in enumerate(indices)
        }

    @property
    def use_embeddings(self) -> bool:
        return bool(getattr(self.base, "use_embeddings", False))

    @property
    def use_reranker(self) -> bool:
        return bool(getattr(self.base, "use_reranker", False))

    def _block(self, index: int, role: str) -> str:
        chunk = self.chunks[index]
        return f"[{role}] {chunk.chunk_id} | {chunk.section}\n{chunk.text}"

    def _context_for(self, index: int) -> tuple[str, list[str]]:
        chunk = self.chunks[index]
        indices = self._doc_groups[chunk.doc_id]
        focus_position = self._positions[index]
        focus = self._block(index, "FOCO")
        header = (
            f"Documento: {chunk.title} ({chunk.doc_id}, versão {chunk.version or 'n/a'})\n"
            f"Seção em foco: {chunk.section}"
        )
        budget = self.max_context_chars - len(header) - len(focus) - 8
        before: list[str] = []
        after: list[str] = []
        chosen = [index]
        for distance in range(1, self.window + 1):
            for candidate_position, bucket in (
                (focus_position - distance, before),
                (focus_position + distance, after),
            ):
                if not 0 <= candidate_position < len(indices) or budget <= 0:
                    continue
                candidate = indices[candidate_position]
                role = f"CONTEXTO -{distance}" if bucket is before else f"CONTEXTO +{distance}"
                block = self._block(candidate, role)
                if len(block) + 2 > budget:
                    continue
                bucket.append(block)
                chosen.append(candidate)
                budget -= len(block) + 2
        blocks = list(reversed(before)) + [focus] + after
        context = f"{header}\n\n" + "\n\n".join(blocks)
        chosen_ids = [self.chunks[item].chunk_id for item in sorted(chosen)]
        return context, chosen_ids

    def search(
        self,
        query: str,
        k: int = 5,
        doc_type: str | None = None,
        mmr_lambda: float = 0.75,
        mode: str = "hybrid",
        max_per_doc: int = 2,
        rerank: bool | None = None,
        rerank_candidates: int = 20,
    ) -> list[RetrievalResult]:
        pool_size = min(
            len(self.chunks),
            max(k, k * self.candidate_multiplier),
        )
        candidates = self.base.search(
            query,
            k=pool_size,
            doc_type=doc_type,
            mmr_lambda=mmr_lambda,
            mode=mode,
            max_per_doc=max_per_doc,
            rerank=rerank,
            rerank_candidates=rerank_candidates,
        )
        results: list[RetrievalResult] = []
        seen_sections: set[tuple[str, str]] = set()
        for candidate in candidates:
            section_key = (candidate.chunk.doc_id, candidate.chunk.section)
            if section_key in seen_sections:
                continue
            seen_sections.add(section_key)
            context, context_ids = self._context_for(self._index_by_id[candidate.chunk.chunk_id])
            results.append(
                RetrievalResult(
                    chunk=candidate.chunk,
                    score=candidate.score,
                    scores=dict(candidate.scores),
                    context=context,
                    context_chunk_ids=context_ids,
                )
            )
            if len(results) >= k:
                break
        return results

    def format_context(self, results: Iterable[RetrievalResult]) -> str:
        blocks = []
        for result in results:
            body = result.context or result.chunk.text
            blocks.append(
                f"[{result.chunk.doc_id} | {result.chunk.section}]\n"
                f"{body}\n"
                f"Fonte: {result.chunk.citation}"
            )
        return "\n\n".join(blocks)

    def citations(self, results: Iterable[RetrievalResult]) -> list[dict[str, str]]:
        return self.base.citations(results)


@lru_cache(maxsize=4)
def get_retriever(
    use_embeddings: bool | None = None,
    use_reranker: bool | None = None,
    use_expansion: bool | None = None,
) -> HybridRetriever:
    paths = resolve_paths()
    if use_embeddings is None:
        use_embeddings = os.getenv("MLOPS_USE_EMBEDDINGS", "0") == "1"
    if use_reranker is None:
        use_reranker = os.getenv("MLOPS_USE_RERANK", "0") == "1"
    if use_expansion is None:
        use_expansion = os.getenv("MLOPS_USE_EXPANSION", "1") == "1"
    return HybridRetriever(
        paths.rag,
        use_embeddings=use_embeddings,
        use_reranker=use_reranker,
        use_expansion=use_expansion,
    )


@lru_cache(maxsize=4)
def get_contextual_retriever(
    use_embeddings: bool | None = None,
    use_reranker: bool | None = None,
    use_expansion: bool | None = None,
    window: int | None = None,
    max_context_chars: int | None = None,
) -> ContextualRetriever:
    base = get_retriever(
        use_embeddings=use_embeddings,
        use_reranker=use_reranker,
        use_expansion=use_expansion,
    )
    if window is None:
        window = int(os.getenv("MLOPS_CONTEXT_WINDOW", "1"))
    if max_context_chars is None:
        max_context_chars = int(os.getenv("MLOPS_CONTEXT_MAX_CHARS", "2400"))
    return ContextualRetriever(
        base,
        window=window,
        max_context_chars=max_context_chars,
        candidate_multiplier=int(os.getenv("MLOPS_CONTEXT_CANDIDATES", "3")),
    )
