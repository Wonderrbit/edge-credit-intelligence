from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class KnowledgeChunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    citation: str
    doc_type: str = "policy"
    version: str = ""
    source_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    chunk: KnowledgeChunk
    score: float
    scores: dict[str, float] = field(default_factory=dict)
    context: str = ""
    context_chunk_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "score": round(self.score, 6),
            "scores": {key: round(value, 6) for key, value in self.scores.items()},
            "context": self.context,
            "context_chunk_ids": self.context_chunk_ids,
        }
