from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mlops.agents.rag.retriever import HybridRetriever

DEFAULT_MODES = ("tfidf", "bm25", "hybrid")


def _ndcg(ranking: Iterable[str], expected: set[str], k: int) -> float:
    unique_docs = list(dict.fromkeys(ranking))[:k]
    dcg = sum(1.0 / math.log2(rank + 1) for rank, doc_id in enumerate(unique_docs, start=1) if doc_id in expected)
    ideal_hits = min(len(expected), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def _metrics(
    reciprocal_ranks: list[float],
    hits_at_1: int,
    hits_at_k: int,
    recalls: list[float],
    ndcgs: list[float],
    term_hits: int,
    total: int,
) -> dict[str, float]:
    return {
        "hit_at_1": round(hits_at_1 / total, 4),
        "hit_at_k": round(hits_at_k / total, 4),
        "mrr": round(sum(reciprocal_ranks) / total, 4),
        "recall_at_k": round(sum(recalls) / total, 4),
        "ndcg_at_k": round(sum(ndcgs) / total, 4),
        "term_coverage": round(term_hits / total, 4),
    }


def evaluate_retriever(
    retriever: HybridRetriever,
    questions_path: str | Path,
    k: int = 5,
    mode: str = "hybrid",
    include_rows: bool = True,
) -> dict[str, Any]:
    questions = [
        json.loads(line)
        for line in Path(questions_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = []
    reciprocal_ranks: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    hits_at_1 = 0
    hits_at_k = 0
    term_hits = 0
    base_mode, rerank = mode.removesuffix("+rerank"), mode.endswith("+rerank")
    for item in questions:
        results = retriever.search(item["question"], k=k, mode=base_mode, rerank=rerank)
        doc_ids = [result.chunk.doc_id for result in results]
        expected = set(item["expected_doc_ids"])
        first_rank = next(
            (rank for rank, doc_id in enumerate(doc_ids, start=1) if doc_id in expected),
            None,
        )
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        hits_at_1 += int(first_rank == 1)
        hits_at_k += int(first_rank is not None)
        recalls.append(len(expected & set(doc_ids)) / len(expected) if expected else 0.0)
        ndcgs.append(_ndcg(doc_ids, expected, k))
        context = " ".join(result.chunk.text for result in results).lower()
        terms = [term.lower() for term in item.get("expected_terms", [])]
        term_hits += int(all(term in context for term in terms)) if terms else 1
        if include_rows:
            rows.append(
                {
                    "question": item["question"],
                    "expected_doc_ids": sorted(expected),
                    "retrieved_doc_ids": doc_ids,
                    "first_relevant_rank": first_rank,
                    "ndcg_at_k": round(ndcgs[-1], 4),
                    "recall_at_k": round(recalls[-1], 4),
                    "top_citation": results[0].chunk.citation if results else None,
                }
            )
    total = max(len(questions), 1)
    report: dict[str, Any] = {
        "questions": len(questions),
        "k": k,
        "mode": mode,
        "rerank": rerank,
        **_metrics(reciprocal_ranks, hits_at_1, hits_at_k, recalls, ndcgs, term_hits, total),
    }
    if include_rows:
        report["rows"] = rows
    return report


def evaluate_modes(
    retriever: HybridRetriever,
    questions_path: str | Path,
    k: int = 5,
    modes: Iterable[str] = DEFAULT_MODES,
    rerank: bool = False,
) -> dict[str, dict[str, Any]]:
    available = list(modes)
    if retriever.use_embeddings and "embeddings" not in available:
        available.append("embeddings")
    evaluated = available + [f"{mode}+rerank" for mode in available] if rerank else available
    return {
        mode: evaluate_retriever(
            retriever,
            questions_path,
            k=k,
            mode=mode,
            include_rows=mode == "hybrid",
        )
        for mode in evaluated
    }
