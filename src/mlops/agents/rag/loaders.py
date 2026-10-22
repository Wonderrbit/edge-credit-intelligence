from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LoadedDocument:
    source_path: Path
    text: str
    doc_type: str


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"[ \t]+", " ", text)


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Instale `pypdf` para ingerir PDFs.") from exc
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def infer_doc_type(path: Path) -> str:
    name = path.stem.lower()
    if "politica" in name or "policy" in name:
        return "policy"
    if "faq" in name:
        return "faq"
    if "lgpd" in name or "privacidade" in name or "security" in name:
        return "compliance"
    if "modelo" in name or "model" in name:
        return "model"
    return "reference"


def load_document(path: Path) -> LoadedDocument:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        text = path.read_text(encoding="utf-8")
    elif suffix in {".html", ".htm"}:
        text = _strip_html(path.read_text(encoding="utf-8", errors="ignore"))
    elif suffix == ".pdf":
        text = _read_pdf(path)
    else:
        raise ValueError(f"Formato não suportado: {path}")
    return LoadedDocument(source_path=path, text=text, doc_type=infer_doc_type(path))


def load_documents(root: str | Path) -> list[LoadedDocument]:
    root = Path(root)
    documents: list[LoadedDocument] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt", ".html", ".htm", ".pdf"}:
            documents.append(load_document(path))
    return documents
