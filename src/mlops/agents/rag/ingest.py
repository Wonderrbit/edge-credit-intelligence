from __future__ import annotations

import json
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from mlops.agents.rag.loaders import LoadedDocument, load_documents
from mlops.agents.rag.redact import redact_pii
from mlops.agents.rag.schemas import KnowledgeChunk
from mlops.config import PROJECT_ROOT

DOC_ID = re.compile(r"\*\*Documento:\*\*\s*([A-Za-z0-9_\-]+)")
VERSION = re.compile(r"\*\*Versão:\*\*\s*([^\n]+)")
TITLE = re.compile(r"^#\s+(.+)$", re.MULTILINE)

PT_STOPWORDS = {
    "a", "ao", "aos", "as", "com", "como", "da", "das", "de", "do", "dos", "e", "em",
    "entre", "era", "essa", "esse", "esta", "este", "eu", "foi", "ha", "isso", "isto",
    "ja", "mais", "mas", "na", "nas", "no", "nos", "o", "os", "ou", "para", "pela",
    "pelas", "pelo", "pelos", "por", "qual", "que", "se", "sem", "ser", "sua", "suas",
    "tem", "um", "uma", "voce", "voces",
}


def slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-") or "secao"


_PLURAL_RULES = (
    ("ns", "m"),
    ("aes", "ao"),
    ("aos", "ao"),
    ("ais", "al"),
    ("eis", "el"),
    ("ois", "ol"),
    ("uis", "ul"),
    ("s", ""),
)


def stem_pt(token: str) -> str:
    if len(token) <= 4 or token.isdigit():
        return token
    if token.endswith("mente") and len(token) > 9:
        return token[:-5]
    for suffix, replacement in _PLURAL_RULES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)] + replacement
    return token


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", text.lower())
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    tokens = re.findall(r"\b[a-z0-9]{2,}\b", ascii_text)
    return [stem_pt(token) for token in tokens if token not in PT_STOPWORDS]


def analyze_text(text: str) -> list[str]:
    tokens = tokenize(text)
    return tokens + [f"{left} {right}" for left, right in zip(tokens, tokens[1:], strict=False)]


def _metadata(text: str) -> tuple[str, str, str]:
    doc_match = DOC_ID.search(text)
    version_match = VERSION.search(text)
    title_match = TITLE.search(text)
    doc_id = doc_match.group(1) if doc_match else "doc"
    version = version_match.group(1).strip() if version_match else "n/a"
    title = title_match.group(1).strip() if title_match else doc_id
    return doc_id, version, title


def _split_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_name = "Visão geral"
    current_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current_lines and "\n".join(current_lines).strip():
                sections.append((current_name, "\n".join(current_lines).strip()))
            current_name = line[3:].strip()
            current_lines = []
        elif line.startswith("# "):
            continue
        else:
            current_lines.append(line)
    if current_lines and "\n".join(current_lines).strip():
        sections.append((current_name, "\n".join(current_lines).strip()))
    return sections


def _window_text(text: str, max_chars: int = 900, overlap: int = 120) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    windows: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        if len(buffer) + len(paragraph) + 2 <= max_chars:
            buffer = f"{buffer}\n\n{paragraph}".strip()
            continue
        if buffer:
            windows.append(buffer)
        if len(paragraph) <= max_chars:
            buffer = paragraph
        else:
            start = 0
            while start < len(paragraph):
                windows.append(paragraph[start : start + max_chars])
                start += max_chars - overlap
            buffer = ""
    if buffer:
        windows.append(buffer)
    return windows


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def chunks_from_document(document: LoadedDocument) -> list[KnowledgeChunk]:
    doc_id, version, title = _metadata(document.text)
    chunks: list[KnowledgeChunk] = []
    for section, body in _split_sections(document.text):
        body = redact_pii(body).strip()
        if len(body) < 40:
            continue
        for index, window in enumerate(_window_text(body)):
            chunk_id = f"{doc_id}:{slugify(section)}:{index}"
            chunks.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    title=title,
                    section=section,
                    text=window,
                    citation=f"{title} > {section} (v{version})",
                    doc_type=document.doc_type,
                    version=version,
                    source_path=_portable_path(document.source_path),
                )
            )
    return chunks


def build_index(
    knowledge_dir: str | Path,
    output_dir: str | Path,
    use_embeddings: bool = False,
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
) -> dict[str, Any]:
    knowledge_dir = Path(knowledge_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    documents = load_documents(knowledge_dir)
    chunks: list[KnowledgeChunk] = []
    for document in documents:
        chunks.extend(chunks_from_document(document))
    if not chunks:
        raise RuntimeError(f"Nenhum chunk gerado a partir de {knowledge_dir}")

    texts = [f"{chunk.title} {chunk.section}\n{chunk.text}" for chunk in chunks]
    vectorizer = TfidfVectorizer(
        analyzer=analyze_text,
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        max_features=12000,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.joblib")
    joblib.dump(tfidf_matrix, output_dir / "tfidf_matrix.joblib")

    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi([tokenize(text) for text in texts])
    joblib.dump(bm25, output_dir / "bm25.joblib")

    embedding_used = False
    embedding_error = None
    if use_embeddings:
        try:
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(embedding_model)
            embeddings = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            np.save(output_dir / "embeddings.npy", embeddings.astype(np.float32))
            (output_dir / "embedding_model.txt").write_text(embedding_model, encoding="utf-8")
            embedding_used = True
        except Exception as exc:  # pragma: no cover - optional dependency/download
            embedding_error = str(exc)

    meta = {
        "created_at": datetime.now(UTC).isoformat(),
        "knowledge_dir": _portable_path(knowledge_dir),
        "chunk_count": len(chunks),
        "document_count": len(documents),
        "documents": sorted({chunk.doc_id for chunk in chunks}),
        "embedding_model": embedding_model if embedding_used else None,
        "embedding_error": embedding_error,
        "vectorizer": "tfidf_stemmed_1_2",
        "tokenization": "pt_light_stem",
        "bm25": "okapi",
    }
    (output_dir / "chunks.json").write_text(
        json.dumps([chunk.to_dict() for chunk in chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "index_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return meta
