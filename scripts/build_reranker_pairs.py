from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops.agents.rag.retriever import get_retriever
from mlops.config import ensure_dirs, resolve_paths

DOMAIN_SUFFIXES = (
    " de Crédito e Risco",
    " de Crédito",
    " e Recuperação",
    " do Analista",
    " e Governança",
)


def _domain(title: str) -> str:
    for suffix in DOMAIN_SUFFIXES:
        if title.endswith(suffix):
            return title[: -len(suffix)]
    return title


def _chunk_text(chunk: dict) -> str:
    return f"{chunk['title']} | {chunk['section']}\n{chunk['text']}"


def _queries(section: str, domain: str) -> set[str]:
    cleaned = re.sub(r"^\d+\.\s*", "", section).strip()
    normalized = unicodedata.normalize("NFKD", cleaned).encode("ascii", "ignore").decode().lower()
    if not cleaned or normalized.startswith("visao geral"):
        return set()
    if cleaned.endswith("?"):
        return {cleaned}
    return {
        f"Como funciona {cleaned.lower()} na política de {domain}?",
        f"O que a política de {domain} define sobre {cleaned.lower()}?",
        f"Quais são as regras de {cleaned.lower()}?",
    }


def _negatives_for(
    retriever,
    query: str,
    positive: dict,
    count: int,
    candidates: int,
    rerank: bool,
) -> list[str]:
    results = retriever.search(query, k=candidates, rerank=rerank)
    negatives = [
        _chunk_text(result.chunk.to_dict())
        for result in results
        if result.chunk.chunk_id != positive["chunk_id"]
        and (result.chunk.doc_id, result.chunk.section) != (positive["doc_id"], positive["section"])
    ]
    return negatives[:count]


def _labeled_rows(
    retriever,
    questions_path: Path,
    count: int,
    candidates: int,
    rerank: bool,
) -> list[dict]:
    rows = []
    for line in questions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        expected = set(item["expected_doc_ids"])
        results = retriever.search(item["question"], k=20, rerank=rerank)
        positive = next((result for result in results if result.chunk.doc_id in expected), None)
        if positive is None:
            continue
        chunk = positive.chunk.to_dict()
        rows.append(
            {
                "query": item["question"],
                "positive": _chunk_text(chunk),
                "negatives": _negatives_for(
                    retriever, item["question"], chunk, count, candidates, rerank
                ),
                "positive_chunk_id": chunk["chunk_id"],
                "source": "labeled",
            }
        )
    return rows


def build_pairs(
    chunks_path: Path,
    output_path: Path,
    negatives: int,
    candidates: int,
    seed: int,
    questions_path: Path | None,
    mine_rerank: bool,
) -> dict:
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    retriever = get_retriever(use_reranker=mine_rerank, use_expansion=True)
    rng = random.Random(seed)
    rows: list[dict] = []
    seen_queries: set[str] = set()

    if questions_path is not None:
        for row in _labeled_rows(
            retriever, questions_path, negatives, candidates, mine_rerank
        ):
            if row["query"] not in seen_queries:
                seen_queries.add(row["query"])
                rows.append(row)

    for chunk in chunks:
        section = chunk["section"]
        domain = _domain(chunk["title"])
        queries = _queries(section, domain) - seen_queries
        seen_queries.update(queries)
        for query in sorted(queries):
            hard_negatives = _negatives_for(
                retriever, query, chunk, negatives, candidates, mine_rerank
            )
            if len(hard_negatives) < negatives:
                pool = [
                    _chunk_text(other)
                    for other in chunks
                    if other["chunk_id"] != chunk["chunk_id"]
                    and (other["doc_id"], other["section"]) != (chunk["doc_id"], section)
                ]
                rng.shuffle(pool)
                hard_negatives.extend(pool[: negatives - len(hard_negatives)])
            rows.append(
                {
                    "query": query,
                    "positive": _chunk_text(chunk),
                    "negatives": hard_negatives,
                    "positive_chunk_id": chunk["chunk_id"],
                    "source": "synthetic",
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "chunks": len(chunks),
        "queries": len(rows),
        "labeled_queries": sum(1 for row in rows if row["source"] == "labeled"),
        "pairs": len(rows) * (1 + negatives),
        "negatives": negatives,
        "mine_rerank": mine_rerank,
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera pares de domínio para fine-tuning do reranker.")
    parser.add_argument("--chunks", default="artifacts/rag/chunks.json")
    parser.add_argument("--output", default="data/reranker_pairs.jsonl")
    parser.add_argument("--questions", default=None, help="JSONL com perguntas rotuladas (held-out deve ficar de fora).")
    parser.add_argument("--negatives", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--mine-rerank", action="store_true", help="Minera negativos difíceis com o reranker base.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = ensure_dirs(resolve_paths())
    chunks = Path(args.chunks)
    output = Path(args.output)
    questions = Path(args.questions) if args.questions else None
    if not chunks.is_absolute():
        chunks = paths.project_root / chunks
    if not output.is_absolute():
        output = paths.project_root / output
    if questions is not None and not questions.is_absolute():
        questions = paths.project_root / questions
    summary = build_pairs(
        chunks,
        output,
        args.negatives,
        args.candidates,
        args.seed,
        questions,
        args.mine_rerank,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
